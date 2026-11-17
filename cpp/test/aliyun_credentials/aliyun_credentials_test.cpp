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
#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"

namespace milvus_storage {
namespace {

class AliyunCredentialsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    client_ = std::make_shared<STSHttpClient>();
    options_.httpOptions.httpClientFactory_create_fn = [this] { return std::make_shared<STSHttpFactory>(client_); };
    Aws::InitAPI(options_);
    token_path_ =
        std::filesystem::temp_directory_path() / ("aliyun-test-" + std::string(Aws::Utils::UUID::RandomUUID()));
    std::ofstream(token_path_) << "synthetic-oidc-token";
    token_env_ = std::make_unique<ScopedEnvironment>("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_path_.string());
    oidc_env_ = std::make_unique<ScopedEnvironment>("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "synthetic-provider");
    provider_ = std::make_unique<AliyunSTSAssumeRoleWebIdentityCredentialsProvider>("synthetic-role", "test-session");
  }
  void TearDown() override {
    EXPECT_EQ(client_->Pending(), 0u);
    provider_.reset();
    oidc_env_.reset();
    token_env_.reset();
    std::filesystem::remove(token_path_);
    client_.reset();
    Aws::ShutdownAPI(options_);
  }
  Aws::Auth::AWSCredentials LoadOld(std::chrono::seconds lifetime = std::chrono::seconds(120)) {
    client_->Enqueue(CredentialsXML(ExpirationAfter(lifetime)));
    auto credentials = provider_->GetAWSCredentials();
    EXPECT_EQ(credentials.GetAWSAccessKeyId(), "old_ak");
    EXPECT_EQ(credentials.GetAWSSecretKey(), "old_sk");
    EXPECT_EQ(credentials.GetSessionToken(), "old_token");
    EXPECT_FALSE(credentials.IsExpired());
    return credentials;
  }
  void ExpectRecovery() {
    client_->Enqueue(CredentialsXML(ExpirationAfter(std::chrono::hours(1)), "new"));
    auto credentials = provider_->GetAWSCredentials();
    EXPECT_EQ(credentials.GetAWSAccessKeyId(), "new_ak");
    EXPECT_EQ(credentials.GetAWSSecretKey(), "new_sk");
    EXPECT_EQ(credentials.GetSessionToken(), "new_token");
    auto requests = client_->Requests();
    EXPECT_EQ(provider_->GetAWSCredentials(), credentials);
    EXPECT_EQ(client_->Requests(), requests);
  }
  Aws::SDKOptions options_;
  std::shared_ptr<STSHttpClient> client_;
  std::filesystem::path token_path_;
  std::unique_ptr<ScopedEnvironment> token_env_, oidc_env_;
  std::unique_ptr<AliyunSTSAssumeRoleWebIdentityCredentialsProvider> provider_;
};

// Exercise the public getter, the SDK HTTP resource client and the real XML parser.
TEST_F(AliyunCredentialsTest, EmptyRefreshPreservesValidCacheAndRecovers) {
  const auto old = LoadOld();
  CapturedLogs logs;
  client_->Enqueue("");
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  EXPECT_EQ(client_->Requests(), 2);
  EXPECT_FALSE(logs.Contains("Successfully retrieved credentials"));
  ExpectRecovery();
  EXPECT_TRUE(logs.Contains("Successfully retrieved credentials"));
  EXPECT_FALSE(logs.Contains("new_sk"));
  EXPECT_FALSE(logs.Contains("new_token"));
  EXPECT_FALSE(logs.Contains("synthetic-oidc-token"));
}

TEST_F(AliyunCredentialsTest, HttpFailurePreservesValidCacheAndRecovers) {
  const auto old = LoadOld();
  client_->Enqueue("<Error><Code>AccessDenied</Code></Error>", Aws::Http::HttpResponseCode::FORBIDDEN);
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  ExpectRecovery();
}

TEST_F(AliyunCredentialsTest, InitialLoadFailureCanRetry) {
  client_->Enqueue("");
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());
  ExpectRecovery();
}

TEST_F(AliyunCredentialsTest, SuccessfulRotation) {
  LoadOld();
  ExpectRecovery();
}

TEST_F(AliyunCredentialsTest, TokenFileFailurePreservesValidCache) {
  const auto old = LoadOld();
  std::filesystem::remove(token_path_);
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  EXPECT_EQ(client_->Requests(), 1);
  std::ofstream(token_path_) << "rotated-synthetic-oidc-token";
  ExpectRecovery();
}

TEST_F(AliyunCredentialsTest, ExpiredCacheIsNotReturnedAfterRefreshFailures) {
  const auto old = LoadOld(std::chrono::seconds(2));
  std::this_thread::sleep_until(
      std::chrono::system_clock::time_point(std::chrono::milliseconds(old.GetExpiration().Millis())) +
      std::chrono::milliseconds(10));
  ASSERT_TRUE(old.IsExpired());
  // Token-file failure leaves the old tuple in the cache: the getter must still reject it.
  std::filesystem::remove(token_path_);
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());
  EXPECT_EQ(client_->Requests(), 1);
  std::ofstream(token_path_) << "synthetic-oidc-token";
  client_->Enqueue("");
  EXPECT_TRUE(provider_->GetAWSCredentials().IsEmpty());
  ExpectRecovery();
}

TEST_F(AliyunCredentialsTest, ConcurrentRefreshReturnsCompleteTuples) {
  const auto old = LoadOld();
  client_->Enqueue("");
  const auto expiration = ExpirationAfter(std::chrono::hours(1));
  client_->Enqueue(CredentialsXML(expiration, "new"));
  const Aws::Auth::AWSCredentials rotated("new_ak", "new_sk", "new_token",
                                          Aws::Utils::DateTime(expiration.c_str(), Aws::Utils::DateFormat::ISO_8601));
  std::promise<void> entered_refresh, release_refresh;
  auto entered = entered_refresh.get_future();
  const auto release = release_refresh.get_future().share();
  client_->BeforeNextResponse([&] {
    entered_refresh.set_value();
    release.wait();
  });
  auto refreshing = std::async(std::launch::async, [&] { return provider_->GetAWSCredentials(); });
  // Hold the STS response while other readers enter GetAWSCredentials().
  EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  std::promise<void> start;
  const auto ready = start.get_future().share();
  std::vector<std::future<Aws::Auth::AWSCredentials>> readers;
  for (int index = 0; index < 16; ++index) {
    readers.push_back(std::async(std::launch::async, [&, ready] {
      ready.wait();
      return provider_->GetAWSCredentials();
    }));
  }
  start.set_value();
  // Readers cannot finish while Reload() is waiting for the HTTP response.
  for (auto& reader : readers) {
    EXPECT_EQ(reader.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  }
  release_refresh.set_value();
  const auto refreshed = refreshing.get();
  EXPECT_TRUE(refreshed == old || refreshed == rotated);
  for (auto& reader : readers) {
    const auto credentials = reader.get();
    EXPECT_TRUE(credentials == old || credentials == rotated);
  }
  EXPECT_EQ(provider_->GetAWSCredentials(), rotated);
  EXPECT_EQ(client_->Requests(), 3);
}

class InvalidAliyunCredentialsTest : public AliyunCredentialsTest, public ::testing::WithParamInterface<std::string> {
  protected:
  std::string InvalidResponse() {
    const auto kind = GetParam();
    if (kind == "empty")
      return "";
    if (kind == "malformed")
      return CredentialsXML(ExpirationAfter(std::chrono::hours(1)), "bad") + "<broken";
    if (kind == "wrong_root")
      return "<OtherResponse><Credentials/></OtherResponse>";
    if (kind == "no_credentials")
      return "<AssumeRoleWithOIDCResponse/>";
    if (kind == "invalid_expiration")
      return CredentialsXML("not-a-date", "bad");
    if (kind == "expired")
      return CredentialsXML(ExpirationAfter(std::chrono::seconds(-120)), "bad");
    auto response = CredentialsXML(ExpirationAfter(std::chrono::hours(1)), "bad");
    const auto separator = kind.find('_');
    const auto field = kind.substr(separator + 1);
    const auto first = response.find("<" + field + ">");
    const auto end = response.find("</" + field + ">", first);
    if (kind.substr(0, separator) == "missing")
      response.erase(first, end + field.size() + 3 - first);
    else
      response.replace(first + field.size() + 2, end - first - field.size() - 2, " ");
    return response;
  }
};

TEST_P(InvalidAliyunCredentialsTest, PreservesValidCache) {
  const auto old = LoadOld();
  CapturedLogs logs;
  client_->Enqueue(InvalidResponse());
  EXPECT_EQ(provider_->GetAWSCredentials(), old);
  EXPECT_EQ(client_->Requests(), 2);
  EXPECT_FALSE(logs.Contains("Successfully retrieved credentials"));
  EXPECT_FALSE(logs.Contains("bad_ak"));
  EXPECT_FALSE(logs.Contains("bad_sk"));
  EXPECT_FALSE(logs.Contains("bad_token"));
  ExpectRecovery();
}

TEST_P(InvalidAliyunCredentialsTest, InitialFailureReturnsEmptyAndCanRetry) {
  client_->Enqueue(InvalidResponse());
  const auto credentials = provider_->GetAWSCredentials();
  EXPECT_TRUE(credentials.GetAWSAccessKeyId().empty());
  EXPECT_TRUE(credentials.GetAWSSecretKey().empty());
  EXPECT_TRUE(credentials.GetSessionToken().empty());
  ExpectRecovery();
}

INSTANTIATE_TEST_SUITE_P(InvalidResponses,
                         InvalidAliyunCredentialsTest,
                         ::testing::Values("empty",
                                           "malformed",
                                           "wrong_root",
                                           "no_credentials",
                                           "invalid_expiration",
                                           "expired",
                                           "missing_AccessKeyId",
                                           "missing_AccessKeySecret",
                                           "missing_SecurityToken",
                                           "missing_Expiration",
                                           "blank_AccessKeyId",
                                           "blank_AccessKeySecret",
                                           "blank_SecurityToken",
                                           "blank_Expiration"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

}  // namespace
}  // namespace milvus_storage
