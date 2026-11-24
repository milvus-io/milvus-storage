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
#include <aws/s3/model/HeadObjectRequest.h>
#include <atomic>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"
#include "milvus-storage/filesystem/gcp/gcp_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"

namespace milvus_storage {
namespace {

class TestBearer final : public GcpCredentialProvider {
  public:
  explicit TestBearer(const ArrowFileSystemConfig& config) : GcpCredentialProvider(config) {}
  std::atomic<int> calls{0};
  arrow::Status error;

  private:
  arrow::Result<bool> AuthorizeRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request) override {
    ++calls;
    ARROW_RETURN_NOT_OK(error);
    request->SetHeaderValue("Authorization", "Bearer test-token");
    return true;
  }
};

class GcpRequestAuthorizationTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    const auto status = S3FileSystemProducer(ArrowFileSystemConfig{}).InitS3();
    ASSERT_TRUE(status.ok()) << status;
  }
  static arrow::Status OpenPrimary(const std::string& address) {
    ArrowFileSystemConfig config;
    config.cloud_provider = "gcp";
    config.storage_type = "remote";
    config.address = address;
    config.use_ssl = address.starts_with("https://");
    config.bucket_name = "primary-bucket";
    config.region = "auto";
    config.access_key_id = "test-key";
    config.access_key_value = "test-secret";
    return GcpFileSystemProducer(config).Make().status();
  }

  static arrow::Result<std::string> Authorize(const ArrowFileSystemConfig& config,
                                              const std::string& uri,
                                              const std::shared_ptr<TestBearer>& provider) {
    S3Options options;
    options.cloud_provider = config.cloud_provider;
    options.credentials_provider = provider;
    options.endpoint_override = config.address;
    options.region = "auto";
    options.scheme = config.use_ssl ? "https" : "http";
    ARROW_ASSIGN_OR_RAISE(auto client, ClientBuilder<S3Client>(options).BuildClient());
    ARROW_ASSIGN_OR_RAISE(auto lock, client->Lock());
    if (!GcpCredentialProvider::ConfigS3HttpClient(options, *lock.get()))
      return std::string{};
    Aws::Endpoint::AWSEndpoint endpoint;
    endpoint.SetURI(Aws::Http::URI(uri.c_str()));
    Aws::S3::Model::HeadObjectRequest request;
    ARROW_ASSIGN_OR_RAISE(bool authorized, GcpCredentialProvider::AuthorizeRead(options.credentials_provider, request,
                                                                                std::move(endpoint), "",
                                                                                Aws::Http::HttpMethod::HTTP_HEAD));
    return authorized ? std::string(request.GetAdditionalCustomHeaders().at("authorization").c_str()) : "";
  }
};

TEST_F(GcpRequestAuthorizationTest, ChecksResolvedBucketEndpointAndTlsBeforeFetchingBearer) {
  ArrowFileSystemConfig config;
  config.cloud_provider = "gcp";
  config.alias = "external";
  config.address = "https://Storage.GoogleApis.com:443/";
  config.bucket_name = "test.bucket";
  auto provider = std::make_shared<TestBearer>(config);
  for (const char* uri :
       {"https://storage.googleapis.com/test.bucket/key", "https://test.bucket.storage.googleapis.com:443/key"}) {
    EXPECT_EQ(Authorize(config, uri, provider).ValueOrDie(), "Bearer test-token");
  }
  for (const char* uri :
       {"https://storage.googleapis.com:8443/test.bucket/key", "http://storage.googleapis.com/test.bucket/key",
        "https://storage.googleapis.com/other/key", "https://storage.googleapis.com.example/test.bucket/key",
        "https://other.storage.googleapis.com/test.bucket/key"}) {
    EXPECT_FALSE(Authorize(config, uri, provider).ok()) << uri;
  }
  EXPECT_EQ(provider->calls, 2);

  for (const char* endpoint : {"http://storage.googleapis.com", "https://googleapis.com",
                               "https://examplegoogleapis.com", "https://storage.googleapis.com.example"}) {
    config.address = endpoint;
    auto external = std::make_shared<TestBearer>(config);
    EXPECT_EQ(Authorize(config, std::string(endpoint) + "/test.bucket/key", external).ValueOrDie(), "");
    EXPECT_EQ(external->calls, 0);
  }
  config.alias.clear();
  config.address = "http://[2001:DB8::7]:80";
  ASSERT_TRUE(OpenPrimary(config.address).ok());
  auto primary = std::make_shared<TestBearer>(config);
  EXPECT_EQ(Authorize(config, "http://[2001:db8::7]/test.bucket/key", primary).ValueOrDie(), "Bearer test-token");
}

TEST_F(GcpRequestAuthorizationTest, CustomEndpointRequiresPrimaryRegistration) {
  // Endpoint trust is process-wide, including across gtest repetitions.
  static unsigned iteration = 0;
  const auto domain = std::to_string(++iteration) + ".invalid";
  const auto host = "bearer-trust-" + domain;
  ArrowFileSystemConfig config;
  config.cloud_provider = "gcp";
  config.alias = "external";
  config.bucket_name = "bucket";
  config.address = "https://" + host + ":8443";
  config.use_ssl = true;
  auto provider = std::make_shared<TestBearer>(config);
  const auto path_style = config.address + "/bucket/key";
  const auto virtual_host = "https://bucket." + host + ":8443/key";
  EXPECT_EQ(Authorize(config, path_style, provider).ValueOrDie(), "");
  EXPECT_EQ(Authorize(config, virtual_host, provider).ValueOrDie(), "");
  EXPECT_EQ(provider->calls, 0);

  ASSERT_TRUE(OpenPrimary("https://BEARER-TRUST-" + domain + ":8443").ok());
  EXPECT_EQ(Authorize(config, path_style, provider).ValueOrDie(), "Bearer test-token");
  EXPECT_EQ(Authorize(config, virtual_host, provider).ValueOrDie(), "Bearer test-token");
  for (const auto& address : {"http://" + host + ":8443", "https://" + host}) {
    config.address = address;
    auto other = std::make_shared<TestBearer>(config);
    EXPECT_EQ(Authorize(config, config.address + "/bucket/key", other).ValueOrDie(), "");
    EXPECT_EQ(other->calls, 0);
  }
}

TEST_F(GcpRequestAuthorizationTest, UnavailableBearerLeavesRequestUnsigned) {
  ArrowFileSystemConfig config;
  config.cloud_provider = "gcp";
  config.address = "storage.googleapis.com";
  config.use_ssl = true;
  config.bucket_name = "bucket";
  auto provider = std::make_shared<TestBearer>(config);
  provider->error = arrow::Status::IOError("token fetch deadline");
  EXPECT_EQ(Authorize(config, "https://storage.googleapis.com/bucket/key", provider).ValueOrDie(), "");
  EXPECT_EQ(provider->calls, 1);
}

TEST_F(GcpRequestAuthorizationTest, NonGcpClientDoesNotUseGcpAuthorization) {
  ArrowFileSystemConfig config;
  config.cloud_provider = "aws";
  config.address = "storage.googleapis.com";
  config.bucket_name = "bucket";
  config.use_ssl = true;
  auto provider = std::make_shared<TestBearer>(config);
  const auto header = Authorize(config, "https://storage.googleapis.com/bucket/key", provider);
  ASSERT_TRUE(header.ok()) << header.status();
  EXPECT_TRUE(header->empty());
  EXPECT_EQ(provider->calls, 0);
}

}  // namespace
}  // namespace milvus_storage
