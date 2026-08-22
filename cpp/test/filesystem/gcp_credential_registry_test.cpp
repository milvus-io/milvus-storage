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

#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/http/URI.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/s3/S3ErrorMarshaller.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

#include "filesystem/gcp/gcp_filesystem_producer_internal.h"

namespace milvus_storage {

namespace {

class TestGcpCredentialProvider final : public GcpCredentialProvider {
  public:
  arrow::Result<std::optional<std::pair<std::string, std::string>>> AuthorizationHeader() override {
    return std::optional<std::pair<std::string, std::string>>{};
  }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }
};

// Forces request A's token lookup to finish after request B's. This is the
// interleaving that made provider-global last_token_status_ associate A's
// failure with B's request (or B's success with A's request).
class InterleavedGcpCredentialProvider final : public GcpCredentialProvider {
  public:
  arrow::Result<std::optional<std::pair<std::string, std::string>>> AuthorizationHeader() override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (calls_++ == 0) {
      first_call_started_ = true;
      cv_.notify_all();
      cv_.wait(lock, [this] { return release_first_call_; });
      return arrow::Status::IOError("request A token lookup failed");
    }
    return std::optional<std::pair<std::string, std::string>>(std::in_place, "Authorization", "Bearer request-b-token");
  }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  void WaitForFirstCall() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return first_call_started_; });
  }

  void ReleaseFirstCall() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      release_first_call_ = true;
    }
    cv_.notify_all();
  }

  private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int calls_ = 0;
  bool first_call_started_ = false;
  bool release_first_call_ = false;
};

}  // namespace

class GcpCredentialProviderTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(EnsureS3Initialized().ok());
    static std::once_flag finalize_once;
    std::call_once(finalize_once, [] { std::atexit([] { EnsureS3Finalized().ok(); }); });
  }
};

TEST_F(GcpCredentialProviderTest, AuthorizationResultBelongsToTheRequestThatResolvedIt) {
  auto provider = std::make_shared<InterleavedGcpCredentialProvider>();
  auto request_a = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "gcp-request-local-test", Aws::Http::URI("https://storage.googleapis.com/bucket/a"),
      Aws::Http::HttpMethod::HTTP_GET);
  auto request_b = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "gcp-request-local-test", Aws::Http::URI("https://storage.googleapis.com/bucket/b"),
      Aws::Http::HttpMethod::HTTP_GET);

  arrow::Status status_a;
  arrow::Status status_b;
  std::thread thread_a([&] { status_a = ApplyGcpAuthorizationHeader(provider, request_a); });
  provider->WaitForFirstCall();
  std::thread thread_b([&] { status_b = ApplyGcpAuthorizationHeader(provider, request_b); });
  thread_b.join();
  provider->ReleaseFirstCall();
  thread_a.join();

  EXPECT_FALSE(status_a.ok());
  EXPECT_NE(status_a.message().find("request A token lookup failed"), std::string::npos);
  EXPECT_FALSE(request_a->HasHeader("Authorization"));

  EXPECT_TRUE(status_b.ok()) << status_b.ToString();
  ASSERT_TRUE(request_b->HasHeader("Authorization"));
  EXPECT_EQ(request_b->GetHeaderValue("Authorization"), "Bearer request-b-token");
}

TEST_F(GcpCredentialProviderTest, HmacConditionalWriteStillUsesGoogV4Signing) {
  ArrowFileSystemConfig config;
  config.access_key_id = "GOOGACCESSKEY";
  config.access_key_value = "secret";
  auto provider_result = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider_result.ok()) << provider_result.status().ToString();
  auto provider = std::move(provider_result).ValueOrDie();

  auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "gcp-hmac-test", Aws::Http::URI("https://storage.googleapis.com/bucket/object"), Aws::Http::HttpMethod::HTTP_PUT);
  request->SetHeaderValue("Authorization", "AWS4-HMAC-SHA256 old-signature");
  request->SetHeaderValue("x-amz-date", "20260819T000000Z");
  request->SetHeaderValue("x-amz-content-sha256", "UNSIGNED-PAYLOAD");
  request->SetHeaderValue("x-goog-if-generation-match", "0");

  auto authorization_status = ApplyGcpAuthorizationHeader(provider, request);
  ASSERT_TRUE(authorization_status.ok()) << authorization_status.ToString();
  EXPECT_EQ(request->GetHeaderValue("Authorization"), "AWS4-HMAC-SHA256 old-signature");

  auto signing_status = provider->MaybeSignConditionalWrite(request);
  ASSERT_TRUE(signing_status.ok()) << signing_status.ToString();
  EXPECT_NE(std::string(request->GetHeaderValue("Authorization")).find("GOOG4-HMAC-SHA256"), std::string::npos);
  EXPECT_FALSE(request->HasHeader("x-amz-date"));
  EXPECT_TRUE(request->HasHeader("x-goog-date"));
}

TEST_F(GcpCredentialProviderTest, TokenFailureSubtypeSurvivesS3Marshalling) {
  struct TestCase {
    ExtendStatusCode input;
    ExtendStatusCode expected;
  };
  const TestCase test_cases[] = {
      {ExtendStatusCode::StorageTransientNetwork, ExtendStatusCode::StorageTransientNetwork},
      {ExtendStatusCode::StorageTransientTimeout, ExtendStatusCode::StorageTransientTimeout},
      {ExtendStatusCode::StorageTransientThrottling, ExtendStatusCode::StorageTransientThrottling},
      {ExtendStatusCode::StorageTransientService, ExtendStatusCode::StorageTransientService},
      {ExtendStatusCode::StorageAccessDenied, ExtendStatusCode::StorageAccessDenied},
      {ExtendStatusCode::StorageConfigInvalid, ExtendStatusCode::StorageConfigInvalid},
  };

  auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "gcp-token-marshalling-test", Aws::Http::URI("https://storage.googleapis.com/bucket/object"),
      Aws::Http::HttpMethod::HTTP_GET);
  request->SetResponseStreamFactory(Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  Aws::Client::S3ErrorMarshaller marshaller;
  for (const auto& test_case : test_cases) {
    auto token_status = MakeExtendErrorMsg(test_case.input, "bad <token> & endpoint");
    auto response = gcp_internal::MakeTokenErrorResponse(request, token_status);
    Aws::Client::AWSError<Aws::S3::S3Errors> error(marshaller.BuildAWSError(response));
    EXPECT_EQ(error.ShouldRetry(), RetryableForExtendStatusCode(test_case.expected));
    auto status = fs::internal::ErrorToStatus("gcp token: ", "GetObject", error, fs::internal::S3ErrorProvenance{});
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), test_case.expected) << status.ToString();
  }

  auto response = gcp_internal::MakeTokenErrorResponse(request, arrow::Status::IOError("malformed token response"));
  Aws::Client::AWSError<Aws::S3::S3Errors> error(marshaller.BuildAWSError(response));
  EXPECT_FALSE(error.ShouldRetry());
  auto status = fs::internal::ErrorToStatus("gcp token: ", "GetObject", error, fs::internal::S3ErrorProvenance{});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
}

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
