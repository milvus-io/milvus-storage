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

// Drives the GCP HTTP client factory's Bearer delivery gate directly: which
// requests get an Authorization header from a registered provider.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <arrow/status.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/http/URI.h>
#include <aws/core/utils/stream/ResponseStream.h>

#include "filesystem/gcp/gcp_filesystem_producer_internal.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"

namespace milvus_storage {

namespace {

// Base for the test providers: counts how often the factory asked for a header.
class CountingProvider : public GcpCredentialProvider {
  public:
  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  int header_requests() const { return header_requests_.load(); }

  protected:
  std::atomic<int> header_requests_{0};
};

class BearerTestProvider final : public CountingProvider {
  public:
  explicit BearerTestProvider(std::string token) : token_(std::move(token)) {}

  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override {
    ++header_requests_;
    return std::make_pair(std::string("Authorization"), "Bearer " + token_);
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return "bearer:" + token_; }

  private:
  std::string token_;
};

// Stands in for a token fetch that fails by throwing (the factory runs inside
// the exception-free AWS SDK).
class ThrowingTestProvider final : public CountingProvider {
  public:
  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override {
    ++header_requests_;
    throw std::runtime_error("token fetch failed");
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return "throwing"; }
};

// HMAC-style provider: signs requests, never issues a header.
class SigningTestProvider final : public CountingProvider {
  public:
  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override {
    ++header_requests_;
    return std::nullopt;
  }

  bool IssuesBearerToken() const override { return false; }

  std::string Identity() const override { return "hmac:test"; }
};

arrow::Status RegisterProvider(const std::string& address,
                               bool use_ssl,
                               const std::string& bucket,
                               std::shared_ptr<GcpCredentialProvider> provider,
                               bool trusted_endpoint) {
  return GcpCredentialRegistry::Instance().Register({NormalizeGcpEndpoint(address, use_ssl), bucket},
                                                    std::move(provider), trusted_endpoint);
}

std::shared_ptr<Aws::Http::HttpRequest> CreateRequest(const Aws::Http::HttpClientFactory& factory,
                                                      const std::string& uri) {
  return factory.CreateHttpRequest(Aws::Http::URI(uri.c_str()), Aws::Http::HttpMethod::HTTP_GET,
                                   Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
}

std::string AuthorizationOf(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  if (request == nullptr || !request->HasHeader("authorization")) {
    return "";
  }
  return std::string(request->GetHeaderValue("authorization").c_str());
}

}  // namespace

class GcpHttpFactoryTest : public ::testing::Test {
  protected:
  void SetUp() override {
    factory_ = gcp_internal::MakeGoogleHttpClientFactory("");
    ASSERT_NE(factory_, nullptr);
  }

  std::shared_ptr<Aws::Http::HttpClientFactory> factory_;
};

TEST_F(GcpHttpFactoryTest, BearerGoesToHttpsGoogleApisInBothAddressingStyles) {
  // Unique bucket names: the registry is process-wide and shared with other tests.
  const std::string bucket = "gcp-http-factory-google-bucket";
  auto provider = std::make_shared<BearerTestProvider>("google-token");
  // An external filesystem's registration (untrusted) for both schemes, so the
  // http request below reaches the gate instead of missing the lookup.
  auto status = RegisterProvider("storage.googleapis.com", true, bucket, provider, false);
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = RegisterProvider("storage.googleapis.com", false, bucket, provider, false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto path_style = CreateRequest(*factory_, "https://storage.googleapis.com/" + bucket + "/dir/object");
  EXPECT_EQ(AuthorizationOf(path_style), "Bearer google-token");

  auto virtual_host = CreateRequest(*factory_, "https://" + bucket + ".storage.googleapis.com/dir/object");
  EXPECT_EQ(AuthorizationOf(virtual_host), "Bearer google-token");

  // A Bearer token is a reusable credential: never over plain HTTP, even to Google.
  auto plain_http = CreateRequest(*factory_, "http://storage.googleapis.com/" + bucket + "/dir/object");
  ASSERT_NE(plain_http, nullptr);
  EXPECT_EQ(AuthorizationOf(plain_http), "");
  auto plain_http_vhost = CreateRequest(*factory_, "http://" + bucket + ".storage.googleapis.com/dir/object");
  EXPECT_EQ(AuthorizationOf(plain_http_vhost), "");

  // The token was fetched only for the two requests that may carry it.
  EXPECT_EQ(provider->header_requests(), 2);
}

TEST_F(GcpHttpFactoryTest, UntrustedNonGoogleEndpointNeverReceivesBearer) {
  const std::string bucket = "gcp-http-factory-untrusted-bucket";
  auto provider = std::make_shared<BearerTestProvider>("untrusted-token");
  // External filesystem: the endpoint comes from a user spec, so it is not trusted.
  auto status = RegisterProvider("https://gcs-mirror-factory-test.invalid", true, bucket, provider, false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto path_style = CreateRequest(*factory_, "https://gcs-mirror-factory-test.invalid/" + bucket + "/object");
  ASSERT_NE(path_style, nullptr);
  EXPECT_EQ(AuthorizationOf(path_style), "");
  auto virtual_host = CreateRequest(*factory_, "https://" + bucket + ".gcs-mirror-factory-test.invalid/object");
  EXPECT_EQ(AuthorizationOf(virtual_host), "");
  EXPECT_EQ(provider->header_requests(), 0);
}

TEST_F(GcpHttpFactoryTest, TrustedOperatorEndpointReceivesBearerOverHttp) {
  const std::string bucket = "gcp-http-factory-trusted-bucket";
  auto provider = std::make_shared<BearerTestProvider>("operator-token");
  // Primary filesystem: the operator configured this emulator endpoint.
  auto status = RegisterProvider("http://gcs-emulator-factory-test.invalid:4443", false, bucket, provider, true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto path_style = CreateRequest(*factory_, "http://gcs-emulator-factory-test.invalid:4443/" + bucket + "/object");
  EXPECT_EQ(AuthorizationOf(path_style), "Bearer operator-token");
  auto virtual_host = CreateRequest(*factory_, "http://" + bucket + ".gcs-emulator-factory-test.invalid:4443/object");
  EXPECT_EQ(AuthorizationOf(virtual_host), "Bearer operator-token");
  EXPECT_EQ(provider->header_requests(), 2);
}

TEST_F(GcpHttpFactoryTest, ProviderExceptionDoesNotEscapeAndLeavesRequestUnsigned) {
  const std::string bucket = "gcp-http-factory-throwing-bucket";
  auto provider = std::make_shared<ThrowingTestProvider>();
  auto status = RegisterProvider("storage.googleapis.com", true, bucket, provider, false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::shared_ptr<Aws::Http::HttpRequest> request;
  EXPECT_NO_THROW(request = CreateRequest(*factory_, "https://storage.googleapis.com/" + bucket + "/object"));
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(AuthorizationOf(request), "");
  EXPECT_EQ(provider->header_requests(), 1);
}

TEST_F(GcpHttpFactoryTest, SigningProviderIsNotAskedForAHeader) {
  const std::string bucket = "gcp-http-factory-hmac-bucket";
  auto provider = std::make_shared<SigningTestProvider>();
  auto status = RegisterProvider("storage.googleapis.com", true, bucket, provider, false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto request = CreateRequest(*factory_, "https://storage.googleapis.com/" + bucket + "/object");
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(AuthorizationOf(request), "");
  // The gate consults IssuesBearerToken() before asking for a header, so the
  // AWS signer (or the GOOG4 re-sign in the delegator) owns authentication.
  EXPECT_EQ(provider->header_requests(), 0);
}

TEST_F(GcpHttpFactoryTest, UnregisteredBucketYieldsAnUnsignedRequest) {
  auto request = CreateRequest(*factory_, "https://storage.googleapis.com/gcp-http-factory-unregistered-bucket/object");
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(AuthorizationOf(request), "");
}

}  // namespace milvus_storage
