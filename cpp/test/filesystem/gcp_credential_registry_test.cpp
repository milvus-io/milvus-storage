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
  explicit TestGcpCredentialProvider(std::string identity = "test") : identity_(std::move(identity)) {}

  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override { return std::nullopt; }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return identity_; }

  private:
  std::string identity_;
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
  ASSERT_TRUE(registry.Register({NormalizeGcpEndpoint("storage.googleapis.com:443", true), bucket}, provider).ok());

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
  ASSERT_TRUE(registry.Register({NormalizeGcpEndpoint(host + ":8443", true), bucket}, provider).ok());

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
  ASSERT_TRUE(
      registry.Register({NormalizeGcpEndpoint(virtual_host + ":8443", true), virtual_bucket}, virtual_provider).ok());
  auto virtual_uri = Aws::Http::URI(("https://" + virtual_bucket + "." + virtual_host + ":8443/key").c_str());
  EXPECT_EQ(registry.Lookup(virtual_uri), virtual_provider);

  const std::string ipv6_bucket = "gcp-registry-ipv6-bucket";
  auto ipv6_provider = std::make_shared<TestGcpCredentialProvider>();
  ASSERT_TRUE(registry.Register({NormalizeGcpEndpoint("[2001:db8::7]:80", false), ipv6_bucket}, ipv6_provider).ok());
  auto ipv6_uri = Aws::Http::URI(("http://[2001:db8::7]/" + ipv6_bucket + "/key").c_str());
  EXPECT_EQ(registry.Lookup(ipv6_uri), ipv6_provider);
}

TEST(GcpCredentialRegistryTest, BearerTokensGoOnlyToHttpsGoogleApisOrTrustedEndpoints) {
  auto& registry = GcpCredentialRegistry::Instance();
  EXPECT_TRUE(registry.MayCarryBearerToken(Aws::Http::URI("https://storage.googleapis.com/bucket/key")));
  EXPECT_TRUE(registry.MayCarryBearerToken(Aws::Http::URI("https://bucket.storage.googleapis.com/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("http://storage.googleapis.com/bucket/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("https://googleapis.com/bucket/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("https://storage.googleapis.com.example/bucket/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("https://examplegoogleapis.com/bucket/key")));

  // An external filesystem's endpoint stays untrusted even with a provider.
  ASSERT_TRUE(registry
                  .Register({NormalizeGcpEndpoint("external-trust-test.invalid", true), "bucket"},
                            std::make_shared<TestGcpCredentialProvider>())
                  .ok());
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("https://external-trust-test.invalid/bucket/key")));

  // The primary filesystem's endpoint is trusted in path and virtual-host
  // styles, for its own scheme and port only; http is the operator's choice.
  ASSERT_TRUE(registry
                  .Register({NormalizeGcpEndpoint("primary-trust-test.invalid:8443", true), "bucket"},
                            std::make_shared<TestGcpCredentialProvider>(), true)
                  .ok());
  EXPECT_TRUE(registry.MayCarryBearerToken(Aws::Http::URI("https://primary-trust-test.invalid:8443/bucket/key")));
  EXPECT_TRUE(registry.MayCarryBearerToken(Aws::Http::URI("https://bucket.primary-trust-test.invalid:8443/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("http://primary-trust-test.invalid:8443/bucket/key")));
  EXPECT_FALSE(registry.MayCarryBearerToken(Aws::Http::URI("https://primary-trust-test.invalid/bucket/key")));
  ASSERT_TRUE(registry
                  .Register({NormalizeGcpEndpoint("http://emulator-trust-test.invalid:4443", false), "bucket"},
                            std::make_shared<TestGcpCredentialProvider>(), true)
                  .ok());
  EXPECT_TRUE(registry.MayCarryBearerToken(Aws::Http::URI("http://emulator-trust-test.invalid:4443/bucket/key")));
}

TEST(GcpCredentialRegistryTest, OneIdentityPerBucket) {
  auto& registry = GcpCredentialRegistry::Instance();
  const GcpBucketKey external{NormalizeGcpEndpoint("storage.googleapis.com", true), "identity-test-external"};
  auto a = std::make_shared<TestGcpCredentialProvider>("impersonate:a@example.test");
  auto a_again = std::make_shared<TestGcpCredentialProvider>("impersonate:a@example.test");
  auto b = std::make_shared<TestGcpCredentialProvider>("impersonate:b@example.test");
  const Aws::Http::URI uri("https://storage.googleapis.com/identity-test-external/key");

  ASSERT_TRUE(registry.Register(external, a).ok());
  // Same identity again: the first provider stays.
  ASSERT_TRUE(registry.Register(external, a_again).ok());
  EXPECT_EQ(registry.Lookup(uri), a);
  // Another identity on the same bucket is refused and changes nothing.
  auto conflict = registry.Register(external, b);
  EXPECT_TRUE(conflict.IsInvalid()) << conflict.ToString();
  EXPECT_EQ(registry.Lookup(uri), a);

  // Operator configuration replaces a user identity, and is then protected.
  const GcpBucketKey primary{NormalizeGcpEndpoint("storage.googleapis.com", true), "identity-test-primary"};
  auto user = std::make_shared<TestGcpCredentialProvider>("impersonate:user@example.test");
  auto iam = std::make_shared<TestGcpCredentialProvider>("iam");
  const Aws::Http::URI primary_uri("https://storage.googleapis.com/identity-test-primary/key");
  ASSERT_TRUE(registry.Register(primary, user).ok());
  ASSERT_TRUE(registry.Register(primary, iam, true).ok());
  EXPECT_EQ(registry.Lookup(primary_uri), iam);
  auto takeover = registry.Register(primary, user);
  EXPECT_TRUE(takeover.IsInvalid()) << takeover.ToString();
  EXPECT_EQ(registry.Lookup(primary_uri), iam);
}

}  // namespace milvus_storage
