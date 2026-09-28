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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cctype>
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
#include "milvus-storage/filesystem/s3/provider/AliyunOIDCAssumeRoleChainProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunRAMCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunRAMSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineOIDCAssumeRoleChainProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineSTSClient.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/common/arrow_util.h"
#include "test_env.h"

namespace milvus_storage {

// ============================================================================
// RAII environment variable helper
// ============================================================================

Aws::Client::ClientConfiguration MakeNoImdsClientConfiguration() {
  return Aws::Client::ClientConfiguration(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
}

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
  auto config = MakeNoImdsClientConfiguration();
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

  // Sub-test: Parameterized ctor — args populate roleArn/sessionName, machine
  // identity still comes from env. STS request body should carry the arg role,
  // not the env role.
  {
    TempFile token_file("mock_oidc_token_content");

    // Env has a DIFFERENT role to prove args win.
    ScopedEnvVar set_arn_env("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::000:role/env-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::000:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    std::string xml_response = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
    <RequestId>TEST-REQUEST-ID</RequestId>
    <Credentials>
        <AccessKeyId>ARG_AK</AccessKeyId>
        <AccessKeySecret>ARG_SK</AccessKeySecret>
        <SecurityToken>ARG_TOKEN</SecurityToken>
        <Expiration>2099-12-31T23:59:59Z</Expiration>
    </Credentials>
</AssumeRoleWithOIDCResponse>)";

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml_response);

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "ARG_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "ARG_SK");
    EXPECT_EQ(creds.GetSessionToken(), "ARG_TOKEN");

    // Verify the STS request body used the arg role, not the env role.
    auto recorded = mock_client_->GetRecordedRequests();
    ASSERT_FALSE(recorded.empty());
    auto& req = recorded.back();
    auto body_stream = req->GetContentBody();
    ASSERT_NE(body_stream, nullptr);
    std::string body((std::istreambuf_iterator<char>(*body_stream)), std::istreambuf_iterator<char>());
    EXPECT_NE(body.find("tenant-A"), std::string::npos) << "body should contain tenant-A role: " << body;
    EXPECT_EQ(body.find("env-role"), std::string::npos) << "body should not contain env role: " << body;
  }

  // Sub-test: Parameterized ctor — missing OIDC_TOKEN_FILE env → empty creds
  {
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::111:oidc-provider/test");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Parameterized ctor — token file path set but file missing on disk
  // → empty creds (Reload fails to open)
  {
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/nonexistent_token_file_param_ctor");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::111:oidc-provider/test");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

// ============================================================================
// Volcengine Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestVolcengineProvider) {
  // Sub-test: Env-only ctor, uninitialized (missing env vars) → empty creds
  {
    ScopedEnvUnset unset_arn("VOLCENGINE_OIDC_ROLE_TRN");
    ScopedEnvUnset unset_token("VOLCENGINE_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Env-only ctor, missing token file env → empty creds
  {
    ScopedEnvVar set_arn("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::123456:role/test-role");
    ScopedEnvUnset unset_token("VOLCENGINE_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Env-only ctor, missing role trn → empty creds
  {
    ScopedEnvUnset unset_arn("VOLCENGINE_OIDC_ROLE_TRN");
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", "/tmp/some_token");
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Env-only ctor, token file path set but file missing on disk
  // → empty creds (Reload fails to open)
  {
    ScopedEnvVar set_arn("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::123456:role/test-role");
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", "/tmp/nonexistent_volc_token_file_12345");
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Env-only ctor, success flow — mock returns valid JSON
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::123456:role/test-role");
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    std::string json_response = R"({
        "ResponseMetadata": {"RequestId": "TEST-REQUEST-ID"},
        "Result": {
            "Credentials": {
                "AccessKeyId": "MOCK_AK",
                "SecretAccessKey": "MOCK_SK",
                "SessionToken": "MOCK_TOKEN",
                "Expiration": "2099-12-31T23:59:59Z"
            }
        }
    })";

    mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, json_response);

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: STS returns empty body → empty credentials
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::123456:role/test-role");
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, "");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Parameterized (per-tenant) ctor — the arg RoleTrn/session drive
  // the request, while the OIDC token file is still read from env. Volcengine
  // carries RoleTrn in the URL query string (not the body), so assert on the
  // recorded request URI. The env has a DIFFERENT role to prove args win.
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn_env("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::000:role/env-role");
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

    std::string json_response = R"({
        "Result": {
            "Credentials": {
                "AccessKeyId": "ARG_AK",
                "SecretAccessKey": "ARG_SK",
                "SessionToken": "ARG_TOKEN",
                "Expiration": "2099-12-31T23:59:59Z"
            }
        }
    })";

    mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, json_response);

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider("trn:iam::111:role/tenant-A",
                                                                   "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "ARG_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "ARG_SK");
    EXPECT_EQ(creds.GetSessionToken(), "ARG_TOKEN");

    // Verify the STS request used the arg role (in the URL RoleTrn query), not
    // the env role, and that the session name arg reached the request body.
    auto recorded = mock_client_->GetRecordedRequests();
    ASSERT_FALSE(recorded.empty());
    auto& req = recorded.back();
    auto uri = req->GetURIString();
    EXPECT_NE(uri.find("tenant-A"), std::string::npos) << "URI should contain tenant-A role: " << uri;
    EXPECT_EQ(uri.find("env-role"), std::string::npos) << "URI should not contain env role: " << uri;

    auto body_stream = req->GetContentBody();
    ASSERT_NE(body_stream, nullptr);
    std::string body((std::istreambuf_iterator<char>(*body_stream)), std::istreambuf_iterator<char>());
    EXPECT_NE(body.find("tenant-A-session"), std::string::npos) << "body should contain arg session: " << body;
  }

  // Sub-test: Parameterized ctor — missing OIDC_TOKEN_FILE env → empty creds
  {
    ScopedEnvUnset unset_token("VOLCENGINE_OIDC_TOKEN_FILE");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider("trn:iam::111:role/tenant-A",
                                                                   "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Parameterized ctor — token file path set but file missing on disk
  // → empty creds (Reload fails to open)
  {
    ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", "/tmp/nonexistent_volc_token_param_ctor");

    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider("trn:iam::111:role/tenant-A",
                                                                   "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

// ============================================================================
// Volcengine V4 signing — golden vector
// ============================================================================

// Locks the hand-rolled Volcengine V4 signer against a reference vector
// computed independently (Python hmac/hashlib). If any of the 5 Volcengine-vs-
// AWS-SigV4 differences regress (AWS4 prefix, aws4_request tail, X-Amz-* header
// names, algorithm token), this signature diverges and the test fails.
TEST_F(S3ProviderTest, TestVolcengineV4SignGoldenVector) {
  const Aws::String ak = "AKGOLDEN";
  const Aws::String sk = "SKGOLDENSECRET";
  const Aws::String region = "cn-beijing";
  const Aws::String service = "sts";
  const Aws::String host = "sts.volcengineapi.com";
  const Aws::String canonical_query = "Action=AssumeRole&Version=2018-01-01";
  // RoleTrn percent-encoded (colons -> %3A, slash -> %2F) exactly as the
  // production body builder emits it.
  const Aws::String body =
      "RoleTrn=trn%3Aiam%3A%3A2112796134%3Arole%2Fllqtestexternal&RoleSessionName=golden-session";
  const Aws::String content_type = "application/x-www-form-urlencoded";
  const Aws::String x_date = "20260717T120000Z";
  const Aws::String scope_date = "20260717";

  auto sig = VolcengineSTSCredentialsClient::SignRequestV4(ak, sk, region, service, host, canonical_query, body,
                                                           content_type, x_date, scope_date);

  EXPECT_EQ(sig.xContentSha256, "2060595ad22ad6cfd245503adb78bb3701cc0ebbaff38add038cf0be8fd85347");
  EXPECT_EQ(sig.signature, "3ef4e60d3d6990843010c51f1d92e021c5f7a6a087fc9fa03b7a35cbc361a5c2");
  EXPECT_EQ(sig.signedHeaders, "content-type;host;x-content-sha256;x-date");
  EXPECT_EQ(sig.authorization,
            "HMAC-SHA256 Credential=AKGOLDEN/20260717/cn-beijing/sts/request, "
            "SignedHeaders=content-type;host;x-content-sha256;x-date, "
            "Signature=3ef4e60d3d6990843010c51f1d92e021c5f7a6a087fc9fa03b7a35cbc361a5c2");
}

// ============================================================================
// Volcengine OIDC AssumeRole chain provider (two-step)
// ============================================================================

namespace {

// Read the form-urlencoded body of a recorded POST request.
std::string ReadRequestBody(const std::shared_ptr<Aws::Http::HttpRequest>& req) {
  auto& stream = req->GetContentBody();
  if (!stream)
    return {};
  stream->seekg(0);
  return {std::istreambuf_iterator<char>(*stream), std::istreambuf_iterator<char>()};
}

// step-1 AssumeRoleWithOIDC success (big-account machine identity).
constexpr const char* kVolcInnerOidcSuccessJson = R"({
  "ResponseMetadata": {"RequestId": "INNER-RID"},
  "Result": {
    "Credentials": {
      "AccessKeyId": "INNER_AK",
      "SecretAccessKey": "INNER_SK",
      "SessionToken": "INNER_TOKEN",
      "Expiration": "2099-12-31T23:59:59Z"
    }
  }
})";

// step-2 sts:AssumeRole success (customer target role). Volcengine's AssumeRole
// response names the expiry "ExpiredTime" (not "Expiration", which is what the
// OIDC step returns) and carries a +08:00 offset rather than a Z suffix — mirror
// that here so the mock matches the real wire format the provider must parse.
constexpr const char* kVolcOuterAssumeRoleSuccessJson = R"({
  "ResponseMetadata": {"RequestId": "OUTER-RID"},
  "Result": {
    "Credentials": {
      "AccessKeyId": "OUTER_AK",
      "SecretAccessKey": "OUTER_SK",
      "SessionToken": "OUTER_TOKEN",
      "ExpiredTime": "2099-12-31T23:59:59+08:00"
    }
  }
})";

}  // namespace

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderEndToEnd) {
  // Two responses queued under the same STS URL; consumed FIFO. The chain
  // issues AssumeRoleWithOIDC first (inner) then sts:AssumeRole (outer).
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK,
                                kVolcOuterAssumeRoleSuccessJson);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2112796134:role/llqtestexternal", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");
  EXPECT_EQ(creds.GetAWSSecretKey(), "OUTER_SK");
  EXPECT_EQ(creds.GetSessionToken(), "OUTER_TOKEN");
  // AssumeRole reports expiry as "ExpiredTime" with a +08:00 offset; the provider
  // must read that field (not "Expiration") and fold the offset to UTC, so the
  // 23:59:59+08:00 instant lands at 15:59:59Z. A regression on either the field
  // name or the offset math would throw "Invalid RFC3339 time" and fail earlier.
  EXPECT_EQ(creds.GetExpiration().Millis(),
            Aws::Utils::DateTime("2099-12-31T15:59:59Z", Aws::Utils::DateFormat::ISO_8601).Millis())
      << "ExpiredTime +08:00 must normalize to the equivalent UTC Z instant";

  const auto recorded = mock_client_->GetRecordedRequests();
  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> sts_reqs;
  for (const auto& r : recorded) {
    if (r->GetURIString().find("sts.volcengineapi.com") != Aws::String::npos) {
      sts_reqs.push_back(r);
    }
  }
  ASSERT_EQ(sts_reqs.size(), 2u) << "chain must issue exactly two STS calls";

  // Inner request: AssumeRoleWithOIDC against the env-driven machine-identity
  // role. Customer's target role must NOT appear here — that was the bug this
  // provider exists to fix. Volcengine carries RoleTrn in the URL query.
  const auto inner_uri = sts_reqs[0]->GetURIString();
  EXPECT_NE(inner_uri.find("Action=AssumeRoleWithOIDC"), std::string::npos) << inner_uri;
  EXPECT_NE(inner_uri.find("MilvusTos"), std::string::npos) << inner_uri;
  EXPECT_EQ(inner_uri.find("llqtestexternal"), std::string::npos)
      << "inner OIDC step must not carry the customer target role: " << inner_uri;

  // Outer request: sts:AssumeRole signed by the inner step's STS creds. RoleTrn
  // rides in the body (percent-encoded); the caller session token rides in the
  // X-Security-Token header; the Authorization credential names INNER_AK.
  const auto outer_uri = sts_reqs[1]->GetURIString();
  EXPECT_NE(outer_uri.find("Action=AssumeRole"), std::string::npos) << outer_uri;
  const auto outer_body = ReadRequestBody(sts_reqs[1]);
  EXPECT_NE(outer_body.find("RoleTrn=trn%3Aiam%3A%3A2112796134%3Arole%2Fllqtestexternal"), std::string::npos)
      << outer_body;
  EXPECT_NE(outer_body.find("RoleSessionName=tenant-A-session"), std::string::npos) << outer_body;

  const auto& outer_headers = sts_reqs[1]->GetHeaders();
  auto header_value = [&](const std::string& lname) -> std::string {
    for (const auto& h : outer_headers) {
      std::string key(h.first.c_str());
      std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
      if (key == lname)
        return std::string(h.second.c_str());
    }
    return {};
  };
  EXPECT_EQ(header_value("x-security-token"), "INNER_TOKEN") << "step-1 session token must ride in X-Security-Token";
  const auto authz = header_value("authorization");
  EXPECT_NE(authz.find("HMAC-SHA256 Credential=INNER_AK/"), std::string::npos)
      << "outer request must be signed by the inner step's AK: " << authz;
  EXPECT_NE(authz.find("/cn-beijing/sts/request"), std::string::npos) << authz;
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderInnerStepFailsReturnsEmpty) {
  // Inner AssumeRoleWithOIDC returns empty — outer must never fire and the
  // provider surfaces empty creds rather than falling back to anonymous.
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2112796134:role/llqtestexternal", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());

  size_t sts_calls = 0;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.volcengineapi.com") != Aws::String::npos)
      ++sts_calls;
  }
  EXPECT_EQ(sts_calls, 1u) << "outer AssumeRole must be skipped when inner step fails";
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderOuterStepEmptyReturnsEmpty) {
  // Inner succeeds, outer AssumeRole returns empty (e.g. cross-account trust
  // policy not yet configured). Provider must surface empty creds, not the
  // inner step's creds — those would let the caller into the customer bucket
  // using the big account's identity.
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2112796134:role/llqtestexternal", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderShortCircuitsWhenTargetEqualsMachineRole) {
  // Single-account degradation: target role == step-1 machine role. Step 2
  // must be skipped and the step-1 creds returned directly, so historical
  // single-account role_arn deployments keep working without the target role
  // needing to trust itself for sts:AssumeRole. Only ONE STS call fires.
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2100211764:role/MilvusTos", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "INNER_AK") << "single-account path must reuse step-1 credentials";
  EXPECT_EQ(creds.GetSessionToken(), "INNER_TOKEN");

  size_t sts_calls = 0;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.volcengineapi.com") != Aws::String::npos)
      ++sts_calls;
  }
  EXPECT_EQ(sts_calls, 1u) << "short-circuit must skip the step-2 AssumeRole";
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderShortCircuitsWhenTargetEmpty) {
  // Empty target role is also the single-account path (extfs.role_arn unset):
  // return step-1 creds, no second STS call.
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider(/*target_role_trn=*/"", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "INNER_AK");

  size_t sts_calls = 0;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.volcengineapi.com") != Aws::String::npos)
      ++sts_calls;
  }
  EXPECT_EQ(sts_calls, 1u);
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderEmptySessionNameDefaults) {
  // Empty target session name should be replaced by a UUID — the outer
  // AssumeRole body must never carry an empty RoleSessionName.
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK,
                                kVolcOuterAssumeRoleSuccessJson);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2112796134:role/llqtestexternal",
                                                 /*target_session_name=*/"");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");

  std::shared_ptr<Aws::Http::HttpRequest> outer_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("Action=AssumeRole&") != std::string::npos) {
      outer_req = r;
    }
  }
  ASSERT_NE(outer_req, nullptr);
  const auto body = ReadRequestBody(outer_req);
  const auto pos = body.find("RoleSessionName=");
  ASSERT_NE(pos, std::string::npos) << body;
  const auto value_start = pos + std::string("RoleSessionName=").size();
  const auto value_end = body.find('&', value_start);
  const auto value = body.substr(value_start, value_end - value_start);
  EXPECT_FALSE(value.empty());
}

TEST_F(S3ProviderTest, TestVolcengineOIDCChainProviderToleratesMalformedExpiry) {
  // The Volcengine STS docs only give an example expiry value, not a format
  // contract. This ExpiredTime is a bare Unix-epoch integer — a realistic drift
  // (Tencent's ExpiredTime is exactly that) that NormalizeToUtcZ's regex throws
  // on AND the SDK's ISO-8601 parser rejects. The credentials must still come
  // through; only the expiry falls back to a bounded TTL. A regression would
  // throw "Invalid RFC3339 time" and drop the whole credential set.
  static constexpr const char* kOuterMalformedExpiryJson = R"({
    "ResponseMetadata": {"RequestId": "OUTER-RID"},
    "Result": {
      "Credentials": {
        "AccessKeyId": "OUTER_AK",
        "SecretAccessKey": "OUTER_SK",
        "SessionToken": "OUTER_TOKEN",
        "ExpiredTime": "4102415999"
      }
    }
  })";
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kVolcInnerOidcSuccessJson);
  mock_client_->EnqueueResponse("sts.volcengineapi.com", Aws::Http::HttpResponseCode::OK, kOuterMalformedExpiryJson);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_machine_role("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::2100211764:role/MilvusTos");
  ScopedEnvVar set_token("VOLCENGINE_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvUnset unset_session("VOLCENGINE_OIDC_ROLE_SESSION_NAME");

  VolcengineOIDCAssumeRoleChainProvider provider("trn:iam::2112796134:role/llqtestexternal", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();

  // Credentials survive the unparseable expiry.
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK") << "unparseable expiry must not discard the credentials";
  EXPECT_EQ(creds.GetSessionToken(), "OUTER_TOKEN");

  // Expiry falls back to a bounded future TTL rather than throwing or landing in
  // the past (which would make the creds look permanently expired). The impl
  // uses 1h; anything in (now, now+2h] proves a sane fallback was applied rather
  // than an epoch/zero value.
  const auto now = Aws::Utils::DateTime::Now();
  EXPECT_GT(creds.GetExpiration().Millis(), now.Millis())
      << "fallback expiry must be in the future so the creds are usable now";
  EXPECT_LE(creds.GetExpiration().Millis(), (now + std::chrono::hours(2)).Millis())
      << "fallback expiry must be bounded so the provider re-fetches";
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

// ============================================================================
// Aliyun RAM STS Client Tests (POP v1 signing, XML response parsing)
// ============================================================================

namespace {

constexpr const char* kSTSSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse>
  <RequestId>TEST-RID</RequestId>
  <Credentials>
    <AccessKeyId>STS_AK</AccessKeyId>
    <AccessKeySecret>STS_SK</AccessKeySecret>
    <SecurityToken>STS_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleResponse>)";

constexpr const char* kAwsStsSuccessXml = R"(<?xml version="1.0" encoding="UTF-8"?>
<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleResult>
    <Credentials>
      <AccessKeyId>ASSUMED_AK</AccessKeyId>
      <SecretAccessKey>ASSUMED_SK</SecretAccessKey>
      <SessionToken>ASSUMED_TOKEN</SessionToken>
      <Expiration>2099-12-31T23:59:59Z</Expiration>
    </Credentials>
    <AssumedRoleUser>
      <Arn>arn:aws:sts::123456789012:assumed-role/reader-role/reader-session</Arn>
      <AssumedRoleId>AROATEST:reader-session</AssumedRoleId>
    </AssumedRoleUser>
  </AssumeRoleResult>
  <ResponseMetadata><RequestId>TEST-RID</RequestId></ResponseMetadata>
</AssumeRoleResponse>)";

}  // namespace

TEST_F(S3ProviderTest, AwsAssumeRoleForwardsFilesystemConfigurationToSts) {
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "BASE_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "BASE_SK");
  ScopedEnvUnset unset_session_token("AWS_SESSION_TOKEN");
  ScopedEnvVar set_default_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvVar set_region("AWS_REGION", "us-east-1");
  ScopedEnvVar disable_imds("AWS_EC2_METADATA_DISABLED", "true");

  mock_client_->EnqueueResponse("sts.", Aws::Http::HttpResponseCode::OK, kAwsStsSuccessXml);

  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderAWS;
  config.region = "us-west-2";
  config.role_arn = "arn:aws:iam::123456789012:role/reader-role";
  config.session_name = "reader-session";
  config.external_id = "tenant-external-id";
  config.load_frequency = 1800;

  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  auto credentials = options.credentials_provider->GetAWSCredentials();
  ASSERT_EQ(credentials.GetAWSAccessKeyId(), "ASSUMED_AK");

  const auto recorded = mock_client_->GetRecordedRequests();
  auto sts_request = std::find_if(recorded.begin(), recorded.end(), [](const auto& request) {
    return request->GetURIString().find("sts") != Aws::String::npos;
  });
  ASSERT_NE(sts_request, recorded.end());

  ASSERT_TRUE((*sts_request)->HasHeader("Authorization"));
  const std::string authorization = (*sts_request)->GetHeaderValue("Authorization").c_str();
  EXPECT_NE(authorization.find("/us-west-2/sts/aws4_request"), std::string::npos) << authorization;

  const auto body = ReadRequestBody(*sts_request);
  EXPECT_NE(body.find("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Freader-role"), std::string::npos) << body;
  EXPECT_NE(body.find("RoleSessionName=reader-session"), std::string::npos) << body;
  EXPECT_NE(body.find("ExternalId=tenant-external-id"), std::string::npos) << body;
  EXPECT_NE(body.find("DurationSeconds=1800"), std::string::npos) << body;
}

TEST_F(S3ProviderTest, AwsAssumeRoleFallsBackToAwsRegionWhenFilesystemRegionIsEmpty) {
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "BASE_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "BASE_SK");
  ScopedEnvUnset unset_session_token("AWS_SESSION_TOKEN");
  ScopedEnvVar set_default_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvVar set_region("AWS_REGION", "us-east-1");
  ScopedEnvVar disable_imds("AWS_EC2_METADATA_DISABLED", "true");

  mock_client_->EnqueueResponse("sts.", Aws::Http::HttpResponseCode::OK, kAwsStsSuccessXml);

  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderAWS;
  config.region = "";
  config.role_arn = "arn:aws:iam::123456789012:role/reader-role";
  config.session_name = "reader-session";

  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  auto credentials = options.credentials_provider->GetAWSCredentials();
  ASSERT_EQ(credentials.GetAWSAccessKeyId(), "ASSUMED_AK");

  const auto recorded = mock_client_->GetRecordedRequests();
  auto sts_request = std::find_if(recorded.begin(), recorded.end(), [](const auto& request) {
    return request->GetURIString().find("sts") != Aws::String::npos;
  });
  ASSERT_NE(sts_request, recorded.end());

  ASSERT_TRUE((*sts_request)->HasHeader("Authorization"));
  const std::string authorization = (*sts_request)->GetHeaderValue("Authorization").c_str();
  EXPECT_NE(authorization.find("/us-east-1/sts/aws4_request"), std::string::npos) << authorization;
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientSuccess) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "test-session";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_EQ(result.creds.GetAWSAccessKeyId(), "STS_AK");
  EXPECT_EQ(result.creds.GetAWSSecretKey(), "STS_SK");
  EXPECT_EQ(result.creds.GetSessionToken(), "STS_TOKEN");
  EXPECT_FALSE(result.creds.IsEmpty());

  // Verify the POST body carries the expected POP v1 params, with the role
  // ARN percent-encoded (colons and slashes escaped).
  auto recorded = mock_client_->GetRecordedRequests();
  ASSERT_FALSE(recorded.empty());
  const auto body = ReadRequestBody(recorded.back());
  EXPECT_NE(body.find("Action=AssumeRole"), std::string::npos) << body;
  EXPECT_NE(body.find("SignatureMethod=HMAC-SHA1"), std::string::npos) << body;
  EXPECT_NE(body.find("SignatureVersion=1.0"), std::string::npos) << body;
  EXPECT_NE(body.find("Version=2015-04-01"), std::string::npos) << body;
  EXPECT_NE(body.find("RoleSessionName=test-session"), std::string::npos) << body;
  // Role ARN must be percent-encoded: colons -> %3A, slashes -> %2F.
  EXPECT_NE(body.find("acs%3Aram%3A%3A123456%3Arole%2Ftarget-role"), std::string::npos) << body;
  EXPECT_EQ(body.find("acs:ram::123456:role/target-role"), std::string::npos)
      << "raw ARN should not appear un-encoded: " << body;
  // Caller's session token must be included when present.
  EXPECT_NE(body.find("SecurityToken=CALLER_TOKEN"), std::string::npos) << body;
  // Signature is appended last; verify it exists and is URL-encoded (base64
  // output contains '+' '/' '=' which all get percent-encoded).
  EXPECT_NE(body.find("&Signature="), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientOmitsSecurityTokenForLongTermCaller) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "LONGTERM_AK";
  req.callerAccessKeySecret = "LONGTERM_SK";
  // Long-term AK/SK: no session token. Aliyun rejects requests that include
  // SecurityToken= with an empty value, so it must be omitted entirely.
  req.callerSecurityToken = "";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "longterm-session";

  client.GetAssumeRoleCredentials(req);

  auto recorded = mock_client_->GetRecordedRequests();
  ASSERT_FALSE(recorded.empty());
  const auto body = ReadRequestBody(recorded.back());
  EXPECT_EQ(body.find("SecurityToken="), std::string::npos) << "SecurityToken must be omitted: " << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientEmptyResponse) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientMissingCredentialsElement) {
  // Response shape the dispatcher would reject: root is right but no
  // <Credentials> child.
  const char* xml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse><RequestId>rid</RequestId></AssumeRoleResponse>)";
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/t";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientUnexpectedRoot) {
  // Root element isn't AssumeRoleResponse — should bail before credential
  // parsing.
  const char* xml = R"(<?xml version='1.0' encoding='UTF-8'?>
<ErrorResponse><Code>AccessDenied</Code></ErrorResponse>)";
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/t";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

// ============================================================================
// Aliyun RAM Credentials Provider Tests (IMDS → AssumeRole chain)
// ============================================================================

namespace {

// Queue a full successful IMDS → STS round trip:
//   PUT  /latest/api/token                                   -> v2 token
//   GET  /latest/meta-data/ram/security-credentials/         -> role name
//   GET  /latest/meta-data/ram/security-credentials/<role>   -> caller JSON
//   POST sts.aliyuncs.com                                    -> STS XML
// The role-list URL (ending in '/') overlaps with the creds URL as a prefix,
// so the mock's substring match has to be disambiguated by key ordering.
// 'my-imds-role' sorts before 'security-credentials/' (ASCII 'm' < 's'), so
// the creds GET matches its own key first; the list GET only matches the
// broader 'security-credentials/' key.
void EnqueueImdsHappyPath(MockHttpClient& mock, const std::string& sts_xml = kSTSSuccessXml) {
  mock.EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token-opaque");
  mock.EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock.EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock.EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, sts_xml);
}

}  // namespace

TEST_F(S3ProviderTest, TestAliyunRAMProviderEndToEnd) {
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");
  EXPECT_EQ(creds.GetAWSSecretKey(), "STS_SK");
  EXPECT_EQ(creds.GetSessionToken(), "STS_TOKEN");

  // The STS POST body's caller AK/SK/Token must come from IMDS, and it must
  // carry the target role ARN from the provider ctor (not from env).
  auto recorded = mock_client_->GetRecordedRequests();
  size_t sts_idx = recorded.size();
  for (size_t i = 0; i < recorded.size(); ++i) {
    if (recorded[i]->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_idx = i;
      break;
    }
  }
  ASSERT_LT(sts_idx, recorded.size());
  const auto body = ReadRequestBody(recorded[sts_idx]);
  EXPECT_NE(body.find("AccessKeyId=IMDS_AK"), std::string::npos) << body;
  EXPECT_NE(body.find("SecurityToken=IMDS_TOKEN"), std::string::npos) << body;
  EXPECT_NE(body.find("RoleSessionName=tenant-A-session"), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsV1Fallback) {
  // V2 token PUT returns 403 (IMDSv2 disabled on this instance) → empty token
  // body; provider falls back to V1-style bare GETs. The rest of the chain
  // still succeeds.
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  // None of the IMDS GETs should have carried the V2 token header.
  for (const auto& req : mock_client_->GetRecordedRequests()) {
    if (req->GetURIString().find("100.100.100.200") != Aws::String::npos &&
        req->GetMethod() == Aws::Http::HttpMethod::HTTP_GET) {
      EXPECT_FALSE(req->HasHeader("x-aliyun-ecs-metadata-token")) << "V1 fallback must not send the V2 token header";
    }
  }
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsRoleListFails) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::NOT_FOUND, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsRoleListEmpty) {
  // 200 OK but empty body (should not happen in practice, but defensive).
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsFails) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsMalformedJson) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, "{not valid json");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsMissingFields) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  // Valid JSON but missing SecurityToken.
  const char* partial = R"({"AccessKeyId":"AK","AccessKeySecret":"SK"})";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, partial);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderSTSReturnsEmpty) {
  // IMDS chain succeeds, but the STS call returns an empty body (e.g. upstream
  // outage) → empty creds, no silent success.
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderCachesValidCredentials) {
  // Second GetAWSCredentials call within the refresh grace must reuse the
  // cached creds without re-hitting IMDS or STS.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds1 = provider.GetAWSCredentials();
  ASSERT_EQ(creds1.GetAWSAccessKeyId(), "STS_AK");

  const size_t before = mock_client_->GetRecordedRequests().size();
  auto creds2 = provider.GetAWSCredentials();
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "STS_AK");
  const size_t after = mock_client_->GetRecordedRequests().size();
  EXPECT_EQ(before, after) << "Valid cached credentials must not trigger a new refresh";
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderEmptySessionNameDefaults) {
  // Empty session name should be replaced by a UUID in the ctor, so the STS
  // body never carries an empty RoleSessionName.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", /*role_session_name=*/"");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  // Find the STS request and verify RoleSessionName is non-empty.
  const auto recorded = mock_client_->GetRecordedRequests();
  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& req : recorded) {
    if (req->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = req;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  const auto pos = body.find("RoleSessionName=");
  ASSERT_NE(pos, std::string::npos) << body;
  // The value lives between '=' and the next '&'.
  const auto value_start = pos + std::string("RoleSessionName=").size();
  const auto value_end = body.find('&', value_start);
  const auto value = body.substr(value_start, value_end - value_start);
  EXPECT_FALSE(value.empty()) << "ctor must synthesize a non-empty session name when caller passes empty";
}

// ============================================================================
// Aliyun OIDC AssumeRole Chain Provider Tests
// ============================================================================

namespace {

constexpr const char* kInnerOidcSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
  <RequestId>TEST-INNER-RID</RequestId>
  <Credentials>
    <AccessKeyId>INNER_AK</AccessKeyId>
    <AccessKeySecret>INNER_SK</AccessKeySecret>
    <SecurityToken>INNER_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleWithOIDCResponse>)";

constexpr const char* kOuterAssumeRoleSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse>
  <RequestId>TEST-OUTER-RID</RequestId>
  <Credentials>
    <AccessKeyId>OUTER_AK</AccessKeyId>
    <AccessKeySecret>OUTER_SK</AccessKeySecret>
    <SecurityToken>OUTER_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleResponse>)";

}  // namespace

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderEndToEnd) {
  // Two responses queued under the same URL substring; consumed FIFO. The
  // chain provider issues AssumeRoleWithOIDC first (inner step) and then
  // sts:AssumeRole (outer step), so this ordering matches.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");
  EXPECT_EQ(creds.GetAWSSecretKey(), "OUTER_SK");
  EXPECT_EQ(creds.GetSessionToken(), "OUTER_TOKEN");

  const auto recorded = mock_client_->GetRecordedRequests();
  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> sts_reqs;
  for (const auto& r : recorded) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_reqs.push_back(r);
    }
  }
  ASSERT_EQ(sts_reqs.size(), 2u) << "chain must issue exactly two STS calls";

  // Inner request: AssumeRoleWithOIDC against the env-driven machine-identity
  // role, with the env's OIDC provider ARN. Customer's target role must NOT
  // appear here — that was the bug this provider exists to fix.
  const auto inner_body = ReadRequestBody(sts_reqs[0]);
  EXPECT_NE(inner_body.find("Action=AssumeRoleWithOIDC"), std::string::npos) << inner_body;
  EXPECT_NE(inner_body.find("zilliz-machine-role"), std::string::npos) << inner_body;
  EXPECT_NE(inner_body.find("zilliz-rrsa"), std::string::npos) << inner_body;
  EXPECT_EQ(inner_body.find("customer-target"), std::string::npos)
      << "inner OIDC step must not carry the customer target role: " << inner_body;

  // Outer request: AssumeRole signed by the inner step's STS creds, targeting
  // the customer role with the caller-supplied session name.
  const auto outer_body = ReadRequestBody(sts_reqs[1]);
  EXPECT_NE(outer_body.find("Action=AssumeRole"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("customer-target"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("AccessKeyId=INNER_AK"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("SecurityToken=INNER_TOKEN"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("RoleSessionName=tenant-A-session"), std::string::npos) << outer_body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderInnerStepFailsReturnsEmpty) {
  // STS replies to the inner AssumeRoleWithOIDC with an empty body — the
  // outer call must never fire and the provider must surface empty creds
  // rather than silently falling back to anonymous.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());

  // Exactly one STS hit (the inner one); outer must be skipped.
  size_t sts_calls = 0;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos)
      ++sts_calls;
  }
  EXPECT_EQ(sts_calls, 1u);
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderOuterStepEmptyReturnsEmpty) {
  // Inner step succeeds, outer AssumeRole returns an empty body (e.g. cross-
  // account trust policy not yet configured). Provider must surface empty
  // creds, not the inner step's creds — those would let the caller through
  // to the customer's bucket using zilliz's own identity.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderForwardsExternalId) {
  // ExternalId belongs in the step-2 sts:AssumeRole body when the caller
  // supplies one. Aliyun's AssumeRole semantics match AWS: empty == not sent
  // (so the parameter behaves like "absent" from the trust policy's POV),
  // non-empty == sent verbatim.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess",
                                        /*external_id=*/"tenant-A-ext-id");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = r;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  EXPECT_NE(body.find("ExternalId=tenant-A-ext-id"), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderOmitsExternalIdWhenEmpty) {
  // Sending an empty ExternalId would still tip the request into the
  // "ExternalId-supplied" branch on Aliyun's side and fail the trust-policy
  // check whenever the policy doesn't list ExternalId. The provider must
  // omit the parameter entirely when the caller leaves it empty.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  ASSERT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = r;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  EXPECT_EQ(body.find("ExternalId="), std::string::npos)
      << "RAM provider must not emit ExternalId at all when caller passes empty: " << body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderForwardsExternalIdToStep2Only) {
  // ExternalId is a step-2 (sts:AssumeRole) concern. Aliyun's
  // AssumeRoleWithOIDC API has no ExternalId parameter, so step 1 must NOT
  // carry it; step 2 must. Verifying both halves catches the easy bug of
  // accidentally putting ExternalId in the inner provider's body.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess",
                                             /*target_external_id=*/"tenant-A-ext-id");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");

  const auto recorded = mock_client_->GetRecordedRequests();
  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> sts_reqs;
  for (const auto& r : recorded) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_reqs.push_back(r);
    }
  }
  ASSERT_EQ(sts_reqs.size(), 2u);

  const auto inner_body = ReadRequestBody(sts_reqs[0]);
  EXPECT_NE(inner_body.find("Action=AssumeRoleWithOIDC"), std::string::npos) << inner_body;
  EXPECT_EQ(inner_body.find("ExternalId"), std::string::npos)
      << "AssumeRoleWithOIDC has no ExternalId concept; step 1 must not carry it: " << inner_body;

  const auto outer_body = ReadRequestBody(sts_reqs[1]);
  EXPECT_NE(outer_body.find("Action=AssumeRole"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("ExternalId=tenant-A-ext-id"), std::string::npos) << outer_body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderEmptySessionNameDefaults) {
  // Empty target session name should be replaced by a UUID — the outer
  // AssumeRole body must never carry an empty RoleSessionName.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", /*target_session_name=*/"");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");

  std::shared_ptr<Aws::Http::HttpRequest> outer_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      const auto body = ReadRequestBody(r);
      if (body.find("Action=AssumeRole&") != std::string::npos ||
          body.find("&Action=AssumeRole&") != std::string::npos) {
        outer_req = r;
      }
    }
  }
  ASSERT_NE(outer_req, nullptr);
  const auto body = ReadRequestBody(outer_req);
  const auto pos = body.find("RoleSessionName=");
  ASSERT_NE(pos, std::string::npos) << body;
  const auto value_start = pos + std::string("RoleSessionName=").size();
  const auto value_end = body.find('&', value_start);
  const auto value = body.substr(value_start, value_end - value_start);
  EXPECT_FALSE(value.empty());
}

}  // namespace milvus_storage
