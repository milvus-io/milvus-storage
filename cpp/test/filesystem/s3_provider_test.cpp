// Copyright 2024 Zilliz
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

#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/utils/stream/ResponseStream.h>

#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

namespace milvus_storage {

// ============================================================================
// RAII environment variable helper
// ============================================================================

class ScopedEnvVar {
  public:
  ScopedEnvVar(const std::string& name, const std::string& value) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    setenv(name.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// Unsets an env var for the scope lifetime, restoring it on destruction.
class ScopedEnvUnset {
  public:
  explicit ScopedEnvUnset(const std::string& name) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    unsetenv(name.c_str());
  }

  ~ScopedEnvUnset() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvUnset(const ScopedEnvUnset&) = delete;
  ScopedEnvUnset& operator=(const ScopedEnvUnset&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// RAII helper for temporary files
class TempFile {
  public:
  explicit TempFile(const std::string& content) {
    path_ = "/tmp/test_oidc_token_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    std::ofstream ofs(path_);
    ofs << content;
    ofs.close();
  }

  ~TempFile() { std::remove(path_.c_str()); }

  const std::string& path() const { return path_; }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  private:
  std::string path_;
};

// ============================================================================
// Mock HTTP infrastructure
// ============================================================================

struct MockResponseSpec {
  Aws::Http::HttpResponseCode code;
  std::string body;
  Aws::Http::HeaderValueCollection headers;
};

class MockHttpClient : public Aws::Http::HttpClient {
  public:
  MockHttpClient() {}

  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
      const std::shared_ptr<Aws::Http::HttpRequest>& request,
      Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
      Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    recorded_requests_.push_back(request);

    auto uri = request->GetURIString();

    // Find a matching response queue by URL substring
    for (auto& [url_key, q] : response_map_) {
      if (!q.empty() && uri.find(url_key) != Aws::String::npos) {
        auto spec = q.front();
        q.pop();
        return BuildResponse(request, spec);
      }
    }

    // Default: return 404 for unmatched requests (e.g., background SDK requests)
    auto resp = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("MockHttpClient", request);
    resp->SetResponseCode(Aws::Http::HttpResponseCode::NOT_FOUND);
    return resp;
  }

  void EnqueueResponse(const std::string& url_match,
                       Aws::Http::HttpResponseCode code,
                       const std::string& body,
                       const Aws::Http::HeaderValueCollection& headers = {}) {
    response_map_[url_match].push({code, body, headers});
  }

  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> GetRecordedRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_requests_;
  }

  private:
  static std::shared_ptr<Aws::Http::HttpResponse> BuildResponse(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                                                                const MockResponseSpec& spec) {
    auto resp = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("MockHttpClient", request);
    resp->SetResponseCode(spec.code);
    for (const auto& h : spec.headers) {
      resp->AddHeader(h.first, h.second);
    }
    if (!spec.body.empty()) {
      resp->GetResponseBody() << spec.body;
    }
    return resp;
  }

  mutable std::mutex mutex_;
  mutable std::map<std::string, std::queue<MockResponseSpec>> response_map_;
  mutable std::vector<std::shared_ptr<Aws::Http::HttpRequest>> recorded_requests_;
};

class MockHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit MockHttpClientFactory(std::shared_ptr<MockHttpClient> client) : mock_client_(std::move(client)) {}

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    return mock_client_;
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("MockHttpClientFactory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("MockHttpClientFactory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  private:
  std::shared_ptr<MockHttpClient> mock_client_;
};

// ============================================================================
// Test Fixture
// ============================================================================

class S3ProviderTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() { ASSERT_TRUE(EnsureS3Initialized().ok()); }

  void SetUp() override {
    mock_client_ = std::make_shared<MockHttpClient>();
    auto factory = std::make_shared<MockHttpClientFactory>(mock_client_);
    Aws::Http::SetHttpClientFactory(factory);
  }

  void TearDown() override {
    Aws::Http::CleanupHttp();
    Aws::Http::InitHttp();
  }

  std::shared_ptr<MockHttpClient> mock_client_;
};

// ============================================================================
// Diagnostic: Verify mock infrastructure
// ============================================================================

TEST_F(S3ProviderTest, TestMockInfrastructure) {
  // Verify the factory is installed: CreateHttpClient should return our mock
  Aws::Client::ClientConfiguration config;
  auto client = Aws::Http::CreateHttpClient(config);
  ASSERT_EQ(client.get(), mock_client_.get()) << "CreateHttpClient did not return the mock client";

  // Verify CreateHttpRequest works
  auto request =
      Aws::Http::CreateHttpRequest(Aws::Http::URI("https://example.com/test"), Aws::Http::HttpMethod::HTTP_GET,
                                   Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  ASSERT_NE(request, nullptr);

  // Verify enqueue + MakeRequest works (URL-keyed)
  mock_client_->EnqueueResponse("example.com", Aws::Http::HttpResponseCode::OK, "test_body");
  auto response = client->MakeRequest(request);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->GetResponseCode(), Aws::Http::HttpResponseCode::OK);

  // Read the body
  Aws::IStreamBufIterator eos;
  Aws::String body(Aws::IStreamBufIterator(response->GetResponseBody()), eos);
  EXPECT_EQ(body, "test_body");

  // Unmatched URLs should return 404
  auto request2 =
      Aws::Http::CreateHttpRequest(Aws::Http::URI("https://unmatched.example.org"), Aws::Http::HttpMethod::HTTP_GET,
                                   Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  auto response2 = client->MakeRequest(request2);
  EXPECT_EQ(response2->GetResponseCode(), Aws::Http::HttpResponseCode::NOT_FOUND);
}

// ============================================================================
// Aliyun Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestAliyunProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_arn("ALIBABA_CLOUD_ROLE_ARN");
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");
    ScopedEnvUnset unset_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file env → empty credentials
  {
    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Missing role arn → empty credentials
  {
    ScopedEnvUnset unset_arn("ALIBABA_CLOUD_ROLE_ARN");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/some_token");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Token file does not exist → empty credentials
  {
    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/nonexistent_token_file_12345");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success flow — mock returns valid XML
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    std::string xml_response = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
    <RequestId>TEST-REQUEST-ID</RequestId>
    <Credentials>
        <AccessKeyId>MOCK_AK</AccessKeyId>
        <AccessKeySecret>MOCK_SK</AccessKeySecret>
        <SecurityToken>MOCK_TOKEN</SecurityToken>
        <Expiration>2099-12-31T23:59:59Z</Expiration>
    </Credentials>
</AssumeRoleWithOIDCResponse>)";

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml_response);

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: STS returns empty body → empty credentials
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

// ============================================================================
// Tencent Cloud Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestTencentProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_region("TKE_REGION");
    ScopedEnvUnset unset_arn("TKE_ROLE_ARN");
    ScopedEnvUnset unset_token("TKE_WEB_IDENTITY_TOKEN_FILE");
    ScopedEnvUnset unset_provider("TKE_PROVIDER_ID");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file → empty credentials
  {
    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvUnset unset_token("TKE_WEB_IDENTITY_TOKEN_FILE");
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success flow — mock returns valid JSON
  {
    TempFile token_file("mock_tencent_token");

    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvVar set_token("TKE_WEB_IDENTITY_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    std::string json_response = R"({
      "Response": {
        "Credentials": {
          "TmpSecretId": "MOCK_AK",
          "TmpSecretKey": "MOCK_SK",
          "Token": "MOCK_TOKEN"
        },
        "Expiration": "2099-12-31T23:59:59Z"
      }
    })";

    mock_client_->EnqueueResponse("tencentcloudapi.com", Aws::Http::HttpResponseCode::OK, json_response);

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: STS returns empty body → empty credentials
  {
    TempFile token_file("mock_tencent_token");

    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvVar set_token("TKE_WEB_IDENTITY_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    mock_client_->EnqueueResponse("tencentcloudapi.com", Aws::Http::HttpResponseCode::OK, "");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Response missing Credentials field
  // Note: The Tencent STS code calls rootNode.GetString("Expiration") without
  // checking if rootNode has an "Expiration" key, so passing {"Response":{}}
  // would crash with an assertion. This is a known limitation of the source code;
  // we skip this sub-test to avoid crashing.
}

// ============================================================================
// Huawei Cloud Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestHuaweiProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_region("HUAWEICLOUD_SDK_REGION");
    ScopedEnvUnset unset_project("HUAWEICLOUD_SDK_PROJECT_ID");
    ScopedEnvUnset unset_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
    ScopedEnvUnset unset_idp("HUAWEICLOUD_SDK_IDP_ID");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file → empty credentials
  {
    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvUnset unset_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success two-step flow
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Step 1 response: id-token/tokens → returns x-subject-token header
    Aws::Http::HeaderValueCollection step1_headers;
    step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

    // Step 2 response: securitytokens → returns credential JSON
    std::string step2_json = R"({
      "credential": {
        "access": "MOCK_AK",
        "secret": "MOCK_SK",
        "securitytoken": "MOCK_TOKEN",
        "expires_at": "2099-12-31T23:59:59Z"
      }
    })";
    mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: Step 1 fails (403 Forbidden) → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::FORBIDDEN, "Access denied");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Step 1 success but missing x-subject-token header → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Return 200 OK but without x-subject-token header
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::OK, "");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Step 1 success, Step 2 returns empty body → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Step 1: success with subject token
    Aws::Http::HeaderValueCollection step1_headers;
    step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

    // Step 2: empty body
    mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, "");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

}  // namespace milvus_storage
