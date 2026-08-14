// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <arrow/status.h>
#include <aws/core/http/URI.h>

#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"

namespace milvus_storage {

namespace {

class TestGcpCredentialProvider final : public GcpCredentialProvider {
  public:
  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override { return std::nullopt; }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }
};

}  // namespace

TEST(GcpCredentialRegistryTest, CanonicalizesConfigEndpoints) {
  auto implicit_https = NormalizeGcpEndpoint("Storage.GoogleApis.com", true);
  auto explicit_https = NormalizeGcpEndpoint("https://storage.googleapis.com:443/", false);
  EXPECT_EQ(implicit_https, explicit_https);
  EXPECT_EQ(implicit_https.scheme, Aws::Http::Scheme::HTTPS);
  EXPECT_EQ(implicit_https.port, Aws::Http::HTTPS_DEFAULT_PORT);
  EXPECT_EQ(implicit_https.host, "storage.googleapis.com");

  auto implicit_http = NormalizeGcpEndpoint("example.test", false);
  auto explicit_http = NormalizeGcpEndpoint("http://example.test:80/", true);
  EXPECT_EQ(implicit_http, explicit_http);
  EXPECT_EQ(implicit_http.scheme, Aws::Http::Scheme::HTTP);
  EXPECT_EQ(implicit_http.port, Aws::Http::HTTP_DEFAULT_PORT);

  auto non_default = NormalizeGcpEndpoint("example.test:8443", true);
  EXPECT_EQ(non_default.scheme, Aws::Http::Scheme::HTTPS);
  EXPECT_EQ(non_default.port, 8443);
  EXPECT_EQ(non_default.host, "example.test");

  auto ipv6 = NormalizeGcpEndpoint("[2001:DB8::1]:443", true);
  EXPECT_EQ(ipv6.scheme, Aws::Http::Scheme::HTTPS);
  EXPECT_EQ(ipv6.port, Aws::Http::HTTPS_DEFAULT_PORT);
  EXPECT_EQ(ipv6.host, "[2001:db8::1]");
}

TEST(GcpCredentialRegistryTest, LooksUpDefaultHttpsPort) {
  const std::string bucket = "gcp-registry-default-port-bucket";
  auto provider = std::make_shared<TestGcpCredentialProvider>();
  auto& registry = GcpCredentialRegistry::Instance();
  registry.Register({NormalizeGcpEndpoint("storage.googleapis.com:443", true), bucket}, provider);

  auto implicit_port_uri = Aws::Http::URI(("https://storage.googleapis.com/" + bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(implicit_port_uri), provider);

  auto explicit_port_uri = Aws::Http::URI(("https://storage.googleapis.com:443/" + bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(explicit_port_uri), provider);
}

TEST(GcpCredentialRegistryTest, PreservesNonDefaultPortAndScheme) {
  const std::string host = "gcp-registry-port.example";
  const std::string bucket = "gcp-registry-port-bucket";
  auto provider = std::make_shared<TestGcpCredentialProvider>();
  auto& registry = GcpCredentialRegistry::Instance();
  registry.Register({NormalizeGcpEndpoint(host + ":8443", true), bucket}, provider);

  auto matching_uri = Aws::Http::URI(("https://" + host + ":8443/" + bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(matching_uri), provider);

  auto different_port_uri = Aws::Http::URI(("https://" + host + ":9443/" + bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(different_port_uri), nullptr);

  auto different_scheme_uri = Aws::Http::URI(("http://" + host + ":8443/" + bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(different_scheme_uri), nullptr);
}

TEST(GcpCredentialRegistryTest, LooksUpVirtualHostAndIpv6Endpoints) {
  auto& registry = GcpCredentialRegistry::Instance();

  const std::string virtual_host = "gcp-registry-vhost.example";
  const std::string virtual_bucket = "gcp-registry-vhost-bucket";
  auto virtual_provider = std::make_shared<TestGcpCredentialProvider>();
  registry.Register({NormalizeGcpEndpoint(virtual_host + ":8443", true), virtual_bucket}, virtual_provider);
  auto virtual_uri = Aws::Http::URI(("https://" + virtual_bucket + "." + virtual_host + ":8443/key").c_str());
  EXPECT_EQ(registry.Lookup(virtual_uri), virtual_provider);

  const std::string ipv6_bucket = "gcp-registry-ipv6-bucket";
  auto ipv6_provider = std::make_shared<TestGcpCredentialProvider>();
  registry.Register({NormalizeGcpEndpoint("[2001:db8::7]:80", false), ipv6_bucket}, ipv6_provider);
  auto ipv6_uri = Aws::Http::URI(("http://[2001:db8::7]/" + ipv6_bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(ipv6_uri), ipv6_provider);
}

}  // namespace milvus_storage
