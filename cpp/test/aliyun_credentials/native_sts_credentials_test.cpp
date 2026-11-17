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

#include "sts_test_utils.h"

#include <aws/core/utils/json/JsonSerializer.h>

#include "milvus-storage/filesystem/s3/provider/AliyunOIDCAssumeRoleChainProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunRAMCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudOIDCAssumeRoleChainProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineOIDCAssumeRoleChainProvider.h"

namespace milvus_storage {

class HuaweiCloudCredentialsProviderTestHelper {
  public:
  using Provider = HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider;

  static void BackdateLastFailure(Provider& provider, std::chrono::seconds elapsed) {
    provider.m_lastReloadFailed = true;
    provider.m_lastFailedReloadTime = std::chrono::steady_clock::now() - elapsed;
  }

  static bool IsInCooldown(const Provider& provider) { return provider.IsInCooldown(); }
};

namespace {

// Keep the production cooldown intact. Parameterized cases exercise a failed
// public getter followed by an explicit reload; dedicated Huawei tests below
// verify normal/urgent cooldown blocking and natural retry after cooldown.
class ReloadableHuaweiProvider : public HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider {
  public:
  void RetryNow() {
    Aws::Utils::Threading::WriterLockGuard guard(m_reloadLock);
    Reload();
  }
};

class NativeSTSCredentialsTest : public ::testing::TestWithParam<std::string> {
  protected:
  void Env(const char* name, const std::string& value) {
    environment_.push_back(std::make_unique<ScopedEnvironment>(name, value));
  }
  bool Aliyun() const { return GetParam().find("Aliyun") == 0; }
  bool Tencent() const { return GetParam().find("Tencent") == 0; }
  bool Huawei() const { return GetParam() == "Huawei"; }
  bool Volcengine() const { return GetParam().find("Volcengine") == 0; }
  bool Chain() const { return GetParam().find("Chain") != std::string::npos; }
  bool UsesOidcTokenFile() const { return GetParam() != "AliyunRAM"; }

  void SetUp() override {
    client_ = std::make_shared<STSHttpClient>();
    options_.httpOptions.httpClientFactory_create_fn = [this] { return std::make_shared<STSHttpFactory>(client_); };
    Aws::InitAPI(options_);
    token_path_ = std::filesystem::temp_directory_path() / ("sts-test-" + std::string(Aws::Utils::UUID::RandomUUID()));
    std::ofstream(token_path_) << "synthetic-oidc-token";
    if (Aliyun()) {
      Env("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::111:role/machine");
      Env("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_path_.string());
      Env("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "synthetic-provider");
      if (Chain())
        provider_ = std::make_unique<AliyunOIDCAssumeRoleChainProvider>("acs:ram::222:role/target", "test");
      else
        provider_ = std::make_unique<AliyunRAMCredentialsProvider>("acs:ram::222:role/target", "test");
    } else if (Tencent()) {
      Env("TKE_REGION", "ap-guangzhou");
      Env("TKE_ROLE_ARN", "qcs::cam::uin/111:roleName/machine");
      Env("TKE_WEB_IDENTITY_TOKEN_FILE", token_path_.string());
      Env("TKE_PROVIDER_ID", "synthetic-provider");
      if (Chain())
        provider_ =
            std::make_unique<TencentCloudOIDCAssumeRoleChainProvider>("qcs::cam::uin/222:roleName/target", "test");
      else
        provider_ = std::make_unique<TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>();
    } else if (Huawei()) {
      Env("HUAWEICLOUD_SDK_REGION", "cn-north-4");
      Env("HUAWEICLOUD_SDK_PROJECT_ID", "synthetic-project");
      Env("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_path_.string());
      Env("HUAWEICLOUD_SDK_IDP_ID", "synthetic-provider");
      provider_ = std::make_unique<ReloadableHuaweiProvider>();
    } else {
      Env("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::111:role/machine");
      Env("VOLCENGINE_OIDC_TOKEN_FILE", token_path_.string());
      Env("VOLCENGINE_OIDC_STS_ENDPOINT", "sts.volcengineapi.com");
      if (Chain())
        provider_ = std::make_unique<VolcengineOIDCAssumeRoleChainProvider>("trn:iam::222:role/target", "test", 900);
      else
        provider_ = std::make_unique<VolcengineSTSAssumeRoleWebIdentityCredentialsProvider>(900);
    }
  }

  void TearDown() override {
    EXPECT_EQ(client_->Pending(), 0u);
    provider_.reset();
    environment_.clear();
    std::filesystem::remove(token_path_);
    client_.reset();
    Aws::ShutdownAPI(options_);
  }

  std::string Response(const std::string& prefix, int seconds, const std::string& invalid = "", bool inner = false) {
    std::string key = prefix + "_ak", secret = prefix + "_sk", token = prefix + "_token";
    std::string expiry = ExpirationAfter(std::chrono::seconds(seconds));
    if (invalid == "blank_key")
      key = " \t";
    if (invalid == "blank_secret")
      secret = " \t";
    if (invalid == "blank_token")
      token = " \t";
    if (invalid == "blank_expiration")
      expiry = " \t";
    if (invalid == "invalid_expiration")
      expiry = "not-a-date";
    if (invalid == "expired")
      expiry = ExpirationAfter(std::chrono::seconds(-120));
    if (invalid == "empty")
      return "";
    if (invalid == "malformed") {
      return Response(prefix, seconds, "", inner) + (Aliyun() ? "<broken" : "{broken");
    }
    if (invalid == "wrong_root") {
      if (Aliyun()) {
        return "<OtherResponse><Secret>" + secret + "</Secret><Token>" + token + "</Token></OtherResponse>";
      }
      Aws::Utils::Json::JsonValue other;
      other.WithString("secret", secret).WithString("token", token);
      return Aws::Utils::Json::JsonValue().WithObject("UnexpectedRoot", other).View().WriteCompact();
    }
    if (invalid == "no_credentials") {
      if (Aliyun()) {
        const std::string root = inner ? "AssumeRoleWithOIDCResponse" : "AssumeRoleResponse";
        return "<" + root + "><RequestId>" + prefix + "</RequestId></" + root + ">";
      }
      if (Huawei()) {
        return R"({"status":"ok"})";
      }
      Aws::Utils::Json::JsonValue empty_res;
      empty_res.WithString("RequestId", "req-1");
      return Aws::Utils::Json::JsonValue()
          .WithObject(Tencent() ? "Response" : "Result", empty_res)
          .View()
          .WriteCompact();
    }

    if (Aliyun()) {
      const std::string root = inner ? "AssumeRoleWithOIDCResponse" : "AssumeRoleResponse";
      std::string xml = "<" + root + "><Credentials>";
      if (invalid != "missing_key")
        xml += "<AccessKeyId>" + key + "</AccessKeyId>";
      if (invalid != "missing_secret")
        xml += "<AccessKeySecret>" + secret + "</AccessKeySecret>";
      if (invalid != "missing_token")
        xml += "<SecurityToken>" + token + "</SecurityToken>";
      if (invalid != "missing_expiration")
        xml += "<Expiration>" + expiry + "</Expiration>";
      xml += "</Credentials></" + root + ">";
      return xml;
    }

    using Aws::Utils::Json::JsonValue;
    JsonValue fields;
    const char* key_name = Tencent() ? "TmpSecretId" : Huawei() ? "access" : "AccessKeyId";
    const char* secret_name = Tencent() ? "TmpSecretKey" : Huawei() ? "secret" : "SecretAccessKey";
    const char* token_name = Tencent() ? "Token" : Huawei() ? "securitytoken" : "SessionToken";
    if (invalid != "missing_key")
      fields.WithString(key_name, key);
    if (invalid != "missing_secret")
      fields.WithString(secret_name, secret);
    if (invalid != "missing_token")
      fields.WithString(token_name, token);
    if (invalid == "wrong_type")
      fields.WithInteger(token_name, 42);

    const char* expiry_name = Huawei()                            ? "expires_at"
                              : (!Tencent() && Chain() && !inner) ? "ExpiredTime"
                                                                  : "Expiration";
    if (!Tencent()) {
      if (invalid == "wrong_expiration_type") {
        fields.WithInteger(expiry_name, 42);
      } else if (invalid != "missing_expiration") {
        fields.WithString(expiry_name, expiry);
      }
    }

    JsonValue result;
    if (Huawei())
      return result.WithObject("credential", fields).View().WriteCompact();
    result.WithObject("Credentials", fields);
    if (Tencent()) {
      if (invalid == "wrong_expiration_type") {
        result.WithString("ExpiredTime", "not-an-integer").WithString("Expiration", expiry);
      } else if (invalid != "missing_expiration") {
        result.WithString("Expiration", expiry);
      }
    }
    return JsonValue().WithObject(Tencent() ? "Response" : "Result", result).View().WriteCompact();
  }

  std::string ImdsCredentialsJson(const std::string& prefix = "machine", int seconds = 3600) const {
    return R"({"AccessKeyId":")" + prefix + R"(_ak","AccessKeySecret":")" + prefix + R"(_sk","SecurityToken":")" +
           prefix + R"(_token","Expiration":")" + ExpirationAfter(std::chrono::seconds(seconds)) +
           R"(","Code":"Success"})";
  }

  void PrepareSource() {
    if (GetParam() == "AliyunRAM") {
      client_->Enqueue("synthetic-imds-token");
      client_->Enqueue("machine-role");
      client_->Enqueue(ImdsCredentialsJson());
    } else if (Huawei()) {
      client_->Enqueue("{}", Aws::Http::HttpResponseCode::OK, {{"x-subject-token", "synthetic-iam-token"}});
    } else if (Chain() && !source_loaded_) {
      client_->Enqueue(Response("machine", 3600, "", true));
      source_loaded_ = true;
    }
  }

  void Enqueue(const std::string& response, Aws::Http::HttpResponseCode code = Aws::Http::HttpResponseCode::OK) {
    PrepareSource();
    client_->Enqueue(response, code);
  }

  void ExpectNoSecretLeaks(const CapturedLogs& logs) const {
    for (const char* secret :
         {"old_sk", "old_token", "bad_sk", "bad_token", "new_sk", "new_token", "machine_sk", "machine_token",
          "synthetic-oidc-token", "synthetic-iam-token", "synthetic-imds-token"}) {
      EXPECT_FALSE(logs.Contains(secret)) << "Leaked secret in logs: " << secret;
    }
  }

  Aws::Auth::AWSCredentials LoadOld(int seconds = 120) {
    Enqueue(Response("old", seconds));
    auto old = provider_->GetAWSCredentials();
    EXPECT_EQ(old.GetAWSAccessKeyId(), "old_ak");
    EXPECT_EQ(old.GetAWSSecretKey(), "old_sk");
    EXPECT_EQ(old.GetSessionToken(), "old_token");
    EXPECT_FALSE(old.IsExpired());
    return old;
  }

  void ExpectRecovery() {
    Enqueue(Response("new", 3600));
    if (Huawei())
      static_cast<ReloadableHuaweiProvider*>(provider_.get())->RetryNow();
    const auto next = provider_->GetAWSCredentials();
    EXPECT_EQ(next.GetAWSAccessKeyId(), "new_ak");
    EXPECT_EQ(next.GetAWSSecretKey(), "new_sk");
    EXPECT_EQ(next.GetSessionToken(), "new_token");
    EXPECT_FALSE(next.IsExpired());
    const auto requests = client_->Requests();
    EXPECT_EQ(provider_->GetAWSCredentials(), next);
    EXPECT_EQ(client_->Requests(), requests);
  }

  void ExerciseInvalidRefreshAndRecovery(const std::string& invalid) {
    CapturedLogs logs;
    const auto old = LoadOld();
    Enqueue(Response("bad", 120, invalid));
    EXPECT_EQ(provider_->GetAWSCredentials(), old);
    ExpectRecovery();
    ExpectNoSecretLeaks(logs);
  }

  Aws::SDKOptions options_;
  std::shared_ptr<STSHttpClient> client_;
  std::filesystem::path token_path_;
  std::vector<std::unique_ptr<ScopedEnvironment>> environment_;
  std::unique_ptr<Aws::Auth::AWSCredentialsProvider> provider_;
  bool source_loaded_ = false;
};

TEST_P(NativeSTSCredentialsTest, RejectsEmpty) { ExerciseInvalidRefreshAndRecovery("empty"); }

TEST_P(NativeSTSCredentialsTest, RejectsMalformed) { ExerciseInvalidRefreshAndRecovery("malformed"); }

TEST_P(NativeSTSCredentialsTest, RejectsWrongRoot) { ExerciseInvalidRefreshAndRecovery("wrong_root"); }

TEST_P(NativeSTSCredentialsTest, RejectsMissingCredentialsObject) {
  ExerciseInvalidRefreshAndRecovery("no_credentials");
}

TEST_P(NativeSTSCredentialsTest, RejectsMissingKey) { ExerciseInvalidRefreshAndRecovery("missing_key"); }

TEST_P(NativeSTSCredentialsTest, RejectsMissingSecret) { ExerciseInvalidRefreshAndRecovery("missing_secret"); }

TEST_P(NativeSTSCredentialsTest, RejectsMissingToken) { ExerciseInvalidRefreshAndRecovery("missing_token"); }

TEST_P(NativeSTSCredentialsTest, RejectsBlankKey) { ExerciseInvalidRefreshAndRecovery("blank_key"); }

TEST_P(NativeSTSCredentialsTest, RejectsBlankSecret) { ExerciseInvalidRefreshAndRecovery("blank_secret"); }

TEST_P(NativeSTSCredentialsTest, RejectsBlankToken) { ExerciseInvalidRefreshAndRecovery("blank_token"); }

TEST_P(NativeSTSCredentialsTest, RejectsMissingExpiration) { ExerciseInvalidRefreshAndRecovery("missing_expiration"); }

TEST_P(NativeSTSCredentialsTest, RejectsBlankExpiration) { ExerciseInvalidRefreshAndRecovery("blank_expiration"); }

TEST_P(NativeSTSCredentialsTest, RejectsInvalidExpiration) { ExerciseInvalidRefreshAndRecovery("invalid_expiration"); }

TEST_P(NativeSTSCredentialsTest, RejectsExpired) { ExerciseInvalidRefreshAndRecovery("expired"); }

TEST_P(NativeSTSCredentialsTest, RejectsWrongType) {
  if (Aliyun())
    GTEST_SKIP() << "XML fields are text";
  ExerciseInvalidRefreshAndRecovery("wrong_type");
}

TEST_P(NativeSTSCredentialsTest, RejectsWrongExpirationType) {
  if (Aliyun())
    GTEST_SKIP() << "XML fields are text";
  ExerciseInvalidRefreshAndRecovery("wrong_expiration_type");
}

TEST_P(NativeSTSCredentialsTest, HttpFailureRetainsValidCacheAndRecovers) {
  CapturedLogs logs;
  const auto old = LoadOld();
  Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  ExpectRecovery();
  ExpectNoSecretLeaks(logs);
}

TEST_P(NativeSTSCredentialsTest, InitialInvalidLoadCanRecover) {
  CapturedLogs logs;
  Enqueue(Response("bad", 120, "missing_token"));
  const auto empty = provider_->GetAWSCredentials();
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_TRUE(empty.GetSessionToken().empty());
  ExpectRecovery();
  ExpectNoSecretLeaks(logs);
}

TEST_P(NativeSTSCredentialsTest, InitialHttpFailureCanRecover) {
  CapturedLogs logs;
  Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  const auto empty = provider_->GetAWSCredentials();
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_TRUE(empty.GetSessionToken().empty());
  ExpectRecovery();
  ExpectNoSecretLeaks(logs);
}

TEST_P(NativeSTSCredentialsTest, TokenFileFailureRetainsValidCacheAndRejectsOnceExpired) {
  if (!UsesOidcTokenFile() || Chain()) {
    GTEST_SKIP() << "Direct OIDC token-file rotation test";
  }
  const auto old = LoadOld(2);
  const auto before_requests = client_->Requests();
  std::filesystem::remove(token_path_);
  // While still unexpired, missing token file preserves cached tuple without STS round-trip.
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  EXPECT_EQ(client_->Requests(), before_requests);

  std::this_thread::sleep_until(
      std::chrono::system_clock::time_point(std::chrono::milliseconds(old.GetExpiration().Millis() + 10)));
  if (Huawei())
    static_cast<ReloadableHuaweiProvider*>(provider_.get())->RetryNow();
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());

  std::ofstream(token_path_) << "rotated-synthetic-oidc-token";
  ExpectRecovery();
}

TEST_P(NativeSTSCredentialsTest, ExpiredCacheIsNotReturnedAfterHttpFailure) {
  const auto old = LoadOld(2);
  ASSERT_FALSE(old.IsEmpty());
  std::this_thread::sleep_until(
      std::chrono::system_clock::time_point(std::chrono::milliseconds(old.GetExpiration().Millis() + 10)));
  Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());
  ExpectRecovery();
}

TEST_P(NativeSTSCredentialsTest, ConcurrentReadersSeeCompleteTuples) {
  const auto old = LoadOld();
  std::promise<void> entered_refresh, release_refresh;
  auto entered = entered_refresh.get_future();
  const auto release = release_refresh.get_future().share();
  client_->BeforeNextResponse([&] {
    entered_refresh.set_value();
    release.wait();
  });
  Enqueue(Response("new", 3600));

  auto writer = std::async(std::launch::async, [&] { return provider_->GetAWSCredentials(); });
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);

  std::promise<void> start;
  const auto ready = start.get_future().share();
  std::vector<std::future<Aws::Auth::AWSCredentials>> readers;
  for (int i = 0; i < 16; ++i) {
    readers.push_back(std::async(std::launch::async, [&, ready] {
      ready.wait();
      return provider_->GetAWSCredentials();
    }));
  }
  start.set_value();
  for (auto& reader : readers) {
    EXPECT_EQ(reader.wait_for(std::chrono::milliseconds(5)), std::future_status::timeout);
  }
  release_refresh.set_value();

  const auto next = writer.get();
  ASSERT_EQ(next.GetAWSAccessKeyId(), "new_ak");
  ASSERT_EQ(next.GetAWSSecretKey(), "new_sk");
  ASSERT_EQ(next.GetSessionToken(), "new_token");
  for (auto& reader : readers) {
    const auto value = reader.get();
    EXPECT_TRUE(value == old || value == next);
    EXPECT_FALSE(value.GetSessionToken().empty());
    EXPECT_FALSE(value.IsExpired());
  }
}

TEST_P(NativeSTSCredentialsTest, ChainNeverFallsBackToMachineIdentity) {
  if (!Chain()) {
    GTEST_SKIP() << "AssumeRole chain only";
  }
  CapturedLogs logs;

  // 1. Step 1 (machine) succeeds, Step 2 fails with HTTP 403 / invalid / expired -> must return empty, never machine.
  Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());
  Enqueue(Response("bad", 120, "expired"));
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());

  // 2. Target role loads short-lived credentials, then expires while machine role is still valid (3600s).
  //    Failed Step 2 refresh must return empty, never old target and never machine identity.
  const auto old = LoadOld(2);
  std::this_thread::sleep_until(
      std::chrono::system_clock::time_point(std::chrono::milliseconds(old.GetExpiration().Millis() + 10)));
  Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  const auto after_expiry = provider_->GetAWSCredentials();
  EXPECT_TRUE(after_expiry.IsEmpty());
  EXPECT_NE(after_expiry.GetAWSAccessKeyId(), "machine_ak");

  // 3. Subsequent valid Step 2 response recovers cleanly.
  ExpectRecovery();
  ExpectNoSecretLeaks(logs);
}

TEST_P(NativeSTSCredentialsTest, ChainStep1FailurePreservesValidTargetUntilExpiryAndRecovers) {
  if (!Chain()) {
    GTEST_SKIP() << "AssumeRole chain only";
  }
  // 1. Step 1 has 2s TTL; target has 120s TTL (< 180s grace, so next lookup refreshes).
  const auto machine_expiry = Aws::Utils::DateTime::Now() + std::chrono::milliseconds(2000);
  client_->Enqueue(Response("machine", 2, "", true));
  client_->Enqueue(Response("old", 120));
  source_loaded_ = true;
  const auto old = provider_->GetAWSCredentials();
  ASSERT_EQ(old.GetAWSAccessKeyId(), "old_ak");

  // 2. Before Step 1 expires, a failed Step 1 refresh preserves cached machine creds so Step 2 still rotates target.
  client_->Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  client_->Enqueue(Response("mid", 120));
  const auto mid = provider_->GetAWSCredentials();
  ASSERT_EQ(mid.GetAWSAccessKeyId(), "mid_ak");

  // 3. Once Step 1 expires and Step 1 refresh fails, Step 2 is not called and still-valid target (mid) is preserved.
  std::this_thread::sleep_until(
      std::chrono::system_clock::time_point(std::chrono::milliseconds(machine_expiry.Millis() + 1100)));
  client_->Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_EQ(provider_->GetAWSCredentials(), mid);

  // 4. Both hops recover with long-lived credentials.
  client_->Enqueue(Response("machine2", 3600, "", true));
  client_->Enqueue(Response("new", 3600));
  const auto recovered = provider_->GetAWSCredentials();
  EXPECT_EQ(recovered.GetAWSAccessKeyId(), "new_ak");
  EXPECT_EQ(recovered.GetAWSSecretKey(), "new_sk");
  EXPECT_EQ(recovered.GetSessionToken(), "new_token");
}

INSTANTIATE_TEST_SUITE_P(
    Providers,
    NativeSTSCredentialsTest,
    ::testing::Values("AliyunRAM", "AliyunChain", "Tencent", "TencentChain", "Huawei", "Volcengine", "VolcengineChain"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

class NativeSTSStandaloneBehaviorTest : public ::testing::Test {
  protected:
  void Env(const char* name, const std::string& value) {
    environment_.push_back(std::make_unique<ScopedEnvironment>(name, value));
  }

  void SetUp() override {
    client_ = std::make_shared<STSHttpClient>();
    options_.httpOptions.httpClientFactory_create_fn = [this] { return std::make_shared<STSHttpFactory>(client_); };
    Aws::InitAPI(options_);
    token_path_ =
        std::filesystem::temp_directory_path() / ("sts-standalone-" + std::string(Aws::Utils::UUID::RandomUUID()));
    std::ofstream(token_path_) << "synthetic-oidc-token";
  }

  void TearDown() override {
    EXPECT_EQ(client_->Pending(), 0u);
    environment_.clear();
    std::filesystem::remove(token_path_);
    client_.reset();
    Aws::ShutdownAPI(options_);
  }

  Aws::SDKOptions options_;
  std::shared_ptr<STSHttpClient> client_;
  std::filesystem::path token_path_;
  std::vector<std::unique_ptr<ScopedEnvironment>> environment_;
};

TEST_F(NativeSTSStandaloneBehaviorTest, AliyunRAMImdsFailuresRetainValidCacheRejectExpiredAndNeverLeak) {
  CapturedLogs logs;
  AliyunRAMCredentialsProvider provider("acs:ram::222:role/target", "test");

  auto enqueue_imds_prefix = [&] {
    client_->Enqueue("synthetic-imds-token");
    client_->Enqueue("machine-role");
  };
  auto sts_xml = [](const std::string& prefix, int seconds) {
    return "<AssumeRoleResponse><Credentials><AccessKeyId>" + prefix + "_ak</AccessKeyId><AccessKeySecret>" + prefix +
           "_sk</AccessKeySecret><SecurityToken>" + prefix + "_token</SecurityToken><Expiration>" +
           ExpirationAfter(std::chrono::seconds(seconds)) + "</Expiration></Credentials></AssumeRoleResponse>";
  };

  // Load initial short-lived valid cache (2s).
  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeyId":"imds_ak","AccessKeySecret":"imds_secret_leak","SecurityToken":"imds_tok_leak",)"
                   R"("Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(3600)) + R"(","Code":"Success"})");
  client_->Enqueue(sts_xml("old", 120));
  const auto old = provider.GetAWSCredentials();
  ASSERT_EQ(old.GetAWSAccessKeyId(), "old_ak");

  // 1. Malformed IMDS JSON containing a secret -> retains old cache and must NOT leak raw JSON snippet in error logs.
  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeySecret":"imds_secret_leak", broken)");
  EXPECT_EQ(provider.GetAWSCredentials(), old);

  // 2. Non-Success Code in IMDS JSON -> retains old cache.
  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeyId":"a","AccessKeySecret":"imds_secret_leak","SecurityToken":"t","Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(3600)) + R"(","Code":"Failed"})");
  EXPECT_EQ(provider.GetAWSCredentials(), old);

  // 3. Blank field and expired Expiration in IMDS JSON -> retains old cache.
  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeyId":"a","AccessKeySecret":" ","SecurityToken":"t","Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(3600)) + R"("})");
  EXPECT_EQ(provider.GetAWSCredentials(), old);

  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeyId":"a","AccessKeySecret":"imds_secret_leak","SecurityToken":"t","Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(-60)) + R"("})");
  EXPECT_EQ(provider.GetAWSCredentials(), old);

  EXPECT_FALSE(logs.Contains("imds_secret_leak"));
  EXPECT_FALSE(logs.Contains("imds_tok_leak"));

  // 4. Recovery succeeds.
  enqueue_imds_prefix();
  client_->Enqueue(R"({"AccessKeyId":"imds_ak2","AccessKeySecret":"imds_sk2","SecurityToken":"imds_tok2",)"
                   R"("Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(3600)) + R"(","Code":"Success"})");
  client_->Enqueue(sts_xml("new", 3600));
  const auto recovered = provider.GetAWSCredentials();
  EXPECT_EQ(recovered.GetAWSAccessKeyId(), "new_ak");
}

TEST_F(NativeSTSStandaloneBehaviorTest, HuaweiPreservesUrgentAndNormalCooldownAndStage1Validation) {
  Env("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  Env("HUAWEICLOUD_SDK_PROJECT_ID", "synthetic-project");
  Env("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_path_.string());
  Env("HUAWEICLOUD_SDK_IDP_ID", "synthetic-provider");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto huawei_json = [](const std::string& prefix, int seconds) {
    return R"({"credential":{"access":")" + prefix + R"(_ak","secret":")" + prefix + R"(_sk","securitytoken":")" +
           prefix + R"(_token","expires_at":")" + ExpirationAfter(std::chrono::seconds(seconds)) + R"("}})";
  };

  // 1. Initial Stage-1 blank x-subject-token fails without hitting Stage 2, enters urgent (5s) cooldown.
  client_->Enqueue("{}", Aws::Http::HttpResponseCode::OK, {{"x-subject-token", "   "}});
  EXPECT_TRUE(provider.GetAWSCredentials().IsEmpty());
  EXPECT_EQ(client_->Requests(), 1u);
  EXPECT_TRUE(HuaweiCloudCredentialsProviderTestHelper::IsInCooldown(provider));

  // Immediate call is blocked by urgent cooldown.
  EXPECT_TRUE(provider.GetAWSCredentials().IsEmpty());
  EXPECT_EQ(client_->Requests(), 1u);

  // After urgent cooldown (6s > 5s), natural GetAWSCredentials() retries and loads expiring-soon (120s) credentials.
  HuaweiCloudCredentialsProviderTestHelper::BackdateLastFailure(provider, std::chrono::seconds(6));
  client_->Enqueue("{}", Aws::Http::HttpResponseCode::CREATED, {{"x-subject-token", "iam-token-1"}});
  client_->Enqueue(huawei_json("old", 120));
  const auto old = provider.GetAWSCredentials();
  ASSERT_EQ(old.GetAWSAccessKeyId(), "old_ak");
  EXPECT_FALSE(HuaweiCloudCredentialsProviderTestHelper::IsInCooldown(provider));

  // 2. Refresh fails at Stage 1 (403) -> retains old, enters normal (30s) cooldown.
  client_->Enqueue("", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  const auto after_fail_requests = client_->Requests();

  // At 10s (< 30s normal cooldown), still in cooldown: returns old without HTTP request.
  HuaweiCloudCredentialsProviderTestHelper::BackdateLastFailure(provider, std::chrono::seconds(10));
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  EXPECT_EQ(client_->Requests(), after_fail_requests);

  // At 31s (> 30s normal cooldown), natural GetAWSCredentials() retries and rotates to new.
  HuaweiCloudCredentialsProviderTestHelper::BackdateLastFailure(provider, std::chrono::seconds(31));
  client_->Enqueue("{}", Aws::Http::HttpResponseCode::CREATED, {{"x-subject-token", "iam-token-2"}});
  client_->Enqueue(huawei_json("new", 3600));
  const auto next = provider.GetAWSCredentials();
  EXPECT_EQ(next.GetAWSAccessKeyId(), "new_ak");
}

TEST_F(NativeSTSStandaloneBehaviorTest, PreservesRetryStrategyOnRetryableStsError) {
  Env("VOLCENGINE_OIDC_ROLE_TRN", "trn:iam::111:role/machine");
  Env("VOLCENGINE_OIDC_TOKEN_FILE", token_path_.string());
  Env("VOLCENGINE_OIDC_STS_ENDPOINT", "sts.volcengineapi.com");

  VolcengineSTSAssumeRoleWebIdentityCredentialsProvider provider(900);
  client_->Enqueue(
      R"({"ResponseMetadata":{"RequestId":"r1","Error":{"Code":"RequestLimitExceeded","Message":"throttled"}}})",
      Aws::Http::HttpResponseCode::BAD_REQUEST);
  client_->Enqueue(R"({"Result":{"Credentials":{"AccessKeyId":"retried_ak","SecretAccessKey":"retried_sk",)"
                   R"("SessionToken":"retried_token","Expiration":")" +
                   ExpirationAfter(std::chrono::seconds(3600)) + R"("}}})");

  const auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "retried_ak");
  EXPECT_EQ(client_->Requests(), 2u);
}

}  // namespace
}  // namespace milvus_storage
