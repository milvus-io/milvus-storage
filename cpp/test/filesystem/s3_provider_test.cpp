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

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
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
#include "milvus-storage/common/arrow_util.h"
#include "test_env.h"

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

  [[nodiscard]] const std::string& path() const { return path_; }

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

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    return mock_client_;
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::String& uri, Aws::Http::HttpMethod method, const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("MockHttpClientFactory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::Http::URI& uri,
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
  static void SetUpTestSuite() {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      return;
    }
    ASSERT_TRUE(EnsureS3Initialized().ok());
    // Register S3 cleanup at process exit, so it runs after all test suites
    // but before AwsInstance's static destructor (which would warn otherwise).
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit([] { EnsureS3Finalized().ok(); }); });
  }

  void SetUp() override {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      GTEST_SKIP() << "S3 provider tests only run for AWS provider";
    }
    mock_client_ = std::make_shared<MockHttpClient>();
    auto factory = std::make_shared<MockHttpClientFactory>(mock_client_);
    Aws::Http::SetHttpClientFactory(factory);
  }

  void TearDown() override {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      return;
    }
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

// ============================================================================
// Test Helper — friend class to access private members
// ============================================================================

class HuaweiCloudCredentialsProviderTestHelper {
  public:
  using Provider = HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider;

  static void setLastReloadFailed(Provider& p,
                                  bool failed,
                                  std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now()) {
    p.m_lastReloadFailed = failed;
    p.m_lastFailedReloadTime = time;
  }

  static void setCredentials(Provider& p, const Aws::Auth::AWSCredentials& creds) { p.m_credentials = creds; }

  static void setTokenFile(Provider& p, const Aws::String& path) { p.m_tokenFile = path; }

  static void setInitialized(Provider& p, bool val) { p.m_initialized = val; }

  static bool isInCooldown(const Provider& p) { return p.IsInCooldown(); }
};

using Helper = HuaweiCloudCredentialsProviderTestHelper;

// ============================================================================
// Huawei Cloud Provider — Cooldown & Resilience Tests
// ============================================================================

// Helper: count requests whose URL contains a given substring
static size_t CountRequestsByUrl(const std::vector<std::shared_ptr<Aws::Http::HttpRequest>>& requests,
                                 const std::string& url_substr) {
  size_t count = 0;
  for (const auto& req : requests) {
    if (req->GetURIString().find(url_substr) != Aws::String::npos) {
      count++;
    }
  }
  return count;
}

TEST_F(S3ProviderTest, TestHuaweiProviderCooldownBlocksRetry) {
  // After STS failure, immediate retry should be blocked by cooldown.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Enqueue failure for step 1
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::FORBIDDEN, "Access denied");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;

  // First call: attempts STS and fails
  auto creds1 = provider.GetAWSCredentials();
  EXPECT_TRUE(creds1.GetAWSAccessKeyId().empty());

  size_t requests_after_first = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_GT(requests_after_first, 0u) << "First call should attempt STS";

  // Second call immediately: cooldown should block new STS attempt
  auto creds2 = provider.GetAWSCredentials();
  EXPECT_TRUE(creds2.GetAWSAccessKeyId().empty());

  size_t requests_after_second = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_after_first, requests_after_second) << "Cooldown should prevent new STS requests";

  // Third call also blocked
  auto creds3 = provider.GetAWSCredentials();
  EXPECT_TRUE(creds3.GetAWSAccessKeyId().empty());

  size_t requests_after_third = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_after_first, requests_after_third) << "Cooldown should still prevent STS requests";
}

TEST_F(S3ProviderTest, TestHuaweiProviderCooldownExpiresAndRetries) {
  // Simulate cooldown expiry via helper (no sleep needed).
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call fails
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::FORBIDDEN, "Access denied");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds1 = provider.GetAWSCredentials();
  EXPECT_TRUE(creds1.GetAWSAccessKeyId().empty());

  size_t requests_after_first = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");

  // Manipulate the failure timestamp to simulate cooldown expiry (6s ago > 5s urgent cooldown)
  Helper::setLastReloadFailed(provider, true, std::chrono::steady_clock::now() - std::chrono::seconds(6));

  // Enqueue success responses for retry
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  std::string step2_json = R"({
    "credential": {
      "access": "RECOVERED_AK",
      "secret": "RECOVERED_SK",
      "securitytoken": "RECOVERED_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  // After cooldown expires, should retry and succeed
  auto creds2 = provider.GetAWSCredentials();
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "RECOVERED_AK");
  EXPECT_EQ(creds2.GetAWSSecretKey(), "RECOVERED_SK");
  EXPECT_EQ(creds2.GetSessionToken(), "RECOVERED_TOKEN");

  size_t requests_after_retry = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_GT(requests_after_retry, requests_after_first) << "Should have made new STS request after cooldown expired";
}

TEST_F(S3ProviderTest, TestHuaweiProviderCachesValidCredentials) {
  // Valid credentials with far-future expiration should be cached without new STS requests.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call: success with far-future expiration
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  std::string step2_json = R"({
    "credential": {
      "access": "VALID_AK",
      "secret": "VALID_SK",
      "securitytoken": "VALID_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds1 = provider.GetAWSCredentials();
  EXPECT_EQ(creds1.GetAWSAccessKeyId(), "VALID_AK");

  // Subsequent calls should use cached credentials without new STS requests
  size_t requests_before = mock_client_->GetRecordedRequests().size();

  auto creds2 = provider.GetAWSCredentials();
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "VALID_AK");

  size_t requests_after = mock_client_->GetRecordedRequests().size();
  EXPECT_EQ(requests_before, requests_after) << "Should use cached credentials without new STS calls";
}

TEST_F(S3ProviderTest, TestHuaweiProviderReturnsEmptyWhenCredsFullyExpired) {
  // GetAWSCredentials should return empty credentials when cached credentials have fully expired,
  // to avoid silent auth failures.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call: success with short expiration (already expired)
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Use an expiration time in the past
  auto now = std::chrono::system_clock::now();
  auto expired = now - std::chrono::seconds(60);
  auto expire_time_t = std::chrono::system_clock::to_time_t(expired);
  char expire_buf[64];
  std::strftime(expire_buf, sizeof(expire_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&expire_time_t));

  std::string step2_json =
      std::string(
          R"({"credential":{"access":"EXPIRED_AK","secret":"EXPIRED_SK","securitytoken":"EXPIRED_TK","expires_at":")") +
      expire_buf + R"("}})";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;

  // First call loads expired credentials, then RefreshIfExpired tries to reload but no more
  // mock responses available. The credentials are stored but expired.
  // Put provider in cooldown so RefreshIfExpired skips reload on subsequent calls.
  provider.GetAWSCredentials();
  Helper::setLastReloadFailed(provider, true, std::chrono::steady_clock::now());

  // Now GetAWSCredentials should return empty, not the expired credentials
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty()) << "Should return empty credentials instead of expired ones";
}

TEST_F(S3ProviderTest, TestHuaweiProviderCooldownRetainsCredsOnRefreshFailure) {
  // When credentials expire soon and refresh fails, cooldown activates and existing creds are retained.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call: success with short-lived expiration (expires in 60s, within 180s grace period)
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  auto now = std::chrono::system_clock::now();
  auto expires = now + std::chrono::seconds(60);
  auto expire_time_t = std::chrono::system_clock::to_time_t(expires);
  char expire_buf[64];
  std::strftime(expire_buf, sizeof(expire_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&expire_time_t));

  std::string step2_json =
      std::string(R"({"credential":{"access":"AK1","secret":"SK1","securitytoken":"TK1","expires_at":")") + expire_buf +
      R"("}})";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds1 = provider.GetAWSCredentials();
  EXPECT_EQ(creds1.GetAWSAccessKeyId(), "AK1");

  // Second call: credentials expire soon (within grace period), refresh fails
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::FORBIDDEN, "fail");

  auto creds2 = provider.GetAWSCredentials();
  // Should retain existing AK1 since it's not fully expired yet
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "AK1");

  // Third call immediately: should be in cooldown, still return AK1
  auto creds3 = provider.GetAWSCredentials();
  EXPECT_EQ(creds3.GetAWSAccessKeyId(), "AK1");

  // Verify cooldown is blocking requests
  size_t requests_before = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  auto creds4 = provider.GetAWSCredentials();
  EXPECT_EQ(creds4.GetAWSAccessKeyId(), "AK1");
  size_t requests_after = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_before, requests_after) << "Cooldown should still block retry";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2HttpFailure) {
  // Step 1 succeeds but Step 2 returns HTTP error → empty credentials.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Step 1 success
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2 fails with 500
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR, "server error");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());

  // Subsequent call should be in cooldown
  size_t requests_before = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  auto creds2 = provider.GetAWSCredentials();
  EXPECT_TRUE(creds2.GetAWSAccessKeyId().empty());
  size_t requests_after = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_before, requests_after) << "Cooldown should block retry after Step 2 failure";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2MissingCredentialField) {
  // Step 2 returns JSON without "credential" field → empty credentials + cooldown.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid JSON but missing "credential" key
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, R"({"error": "something wrong"})");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());

  // Verify cooldown is active
  size_t requests_before = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  provider.GetAWSCredentials();
  size_t requests_after = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_before, requests_after) << "Cooldown should be active after missing credential field";
}

TEST_F(S3ProviderTest, TestHuaweiProviderDurationSeconds7200) {
  // Verify the STS request body contains duration_seconds: 7200.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Step 1 success
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2 success
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

  // Find the securitytokens request and check its body for duration_seconds
  bool found_duration = false;
  for (const auto& req : mock_client_->GetRecordedRequests()) {
    if (req->GetURIString().find("securitytokens") != Aws::String::npos) {
      auto& bodyStream = req->GetContentBody();
      if (bodyStream) {
        bodyStream->seekg(0);
        std::string body((std::istreambuf_iterator<char>(*bodyStream)), std::istreambuf_iterator<char>());
        if (body.find("7200") != std::string::npos) {
          found_duration = true;
        }
      }
      break;
    }
  }
  EXPECT_TRUE(found_duration) << "STS request body should contain duration_seconds: 7200";
}

TEST_F(S3ProviderTest, TestHuaweiProviderConcurrentAccessNoStorm) {
  // Multiple threads calling GetAWSCredentials concurrently should not cause STS request storm.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Enqueue one success response — only one thread should consume it
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  std::string step2_json = R"({
    "credential": {
      "access": "CONCURRENT_AK",
      "secret": "CONCURRENT_SK",
      "securitytoken": "CONCURRENT_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;

  constexpr int NUM_THREADS = 8;
  std::vector<std::thread> threads;
  std::vector<Aws::Auth::AWSCredentials> results(NUM_THREADS);

  threads.reserve(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back([&provider, &results, i]() { results[i] = provider.GetAWSCredentials(); });
  }

  for (auto& t : threads) {
    t.join();
  }

  // At least one thread should have gotten valid credentials
  int valid_count = 0;
  for (const auto& cred : results) {
    if (!cred.GetAWSAccessKeyId().empty()) {
      EXPECT_EQ(cred.GetAWSAccessKeyId(), "CONCURRENT_AK");
      valid_count++;
    }
  }
  EXPECT_GT(valid_count, 0) << "At least one thread should have gotten valid credentials";

  // The key assertion: STS id-token requests should be limited (not 8 separate requests)
  size_t sts_requests = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_LE(sts_requests, 2u) << "Concurrent access should not cause STS request storm (got " << sts_requests << ")";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2EmptySessionToken) {
  // Step 2 returns valid ak/sk but empty securitytoken → should fail and activate cooldown.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk but empty securitytoken
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Empty session token should cause credential rejection";

  // Verify cooldown is active
  size_t requests_before = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  provider.GetAWSCredentials();
  size_t requests_after = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_EQ(requests_before, requests_after) << "Cooldown should be active after empty session token";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2MissingExpiresAt) {
  // Step 2 returns valid credentials but missing expires_at → should fail.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk/token but no expires_at field
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "MOCK_TOKEN"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Missing expires_at should cause credential rejection";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2InvalidExpiresAtFormat) {
  // Step 2 returns credentials with unparseable expires_at → should fail.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk/token but garbage expires_at
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "MOCK_TOKEN",
      "expires_at": "not-a-valid-date"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Invalid expires_at format should cause credential rejection";
}

}  // namespace milvus_storage
