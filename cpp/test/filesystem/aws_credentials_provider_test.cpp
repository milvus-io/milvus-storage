// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "milvus-storage/filesystem/s3/provider/AwsCredentialsProvider.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <aws/core/Aws.h>
#include <aws/core/auth/CrtCredentialsProvider.h>
#include <aws/core/client/UserAgent.h>
#include <aws/core/utils/UUID.h>
#include <aws/crt/auth/Credentials.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/AssumeRoleResult.h>

namespace milvus_storage {
namespace {
using Credentials = Aws::Auth::AWSCredentials;

class AwsCredentialRefreshTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() { Aws::InitAPI(options_); }
  static void TearDownTestSuite() { Aws::ShutdownAPI(options_); }
  inline static Aws::SDKOptions options_;
};

class FunctionProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  explicit FunctionProvider(std::function<Credentials()> fetch) : fetch_(std::move(fetch)) {}
  Credentials GetAWSCredentials() override { return fetch_(); }

  private:
  std::function<Credentials()> fetch_;
};

Credentials Temporary(const char* key, const Aws::Utils::DateTime& expiration) {
  return {key, "synthetic-secret", "synthetic-token", expiration};
}

TEST_F(AwsCredentialRefreshTest, FailurePreservesOnlyUnexpiredCacheAndRecovers) {
  auto now = Aws::Utils::DateTime::Now();
  auto response = Temporary("old", now + std::chrono::seconds(120));
  size_t calls = 0;
  auto source = std::make_shared<FunctionProvider>([&] {
    ++calls;
    return response;
  });
  ValidatedAwsCredentialsProvider provider([source] { return source; }, false, [&] { return now; });
  const auto old = provider.GetAWSCredentials();
  response = {};
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  now = now + std::chrono::seconds(121);
  EXPECT_TRUE(provider.GetAWSCredentials().IsEmpty());
  response = Temporary("new", now + std::chrono::hours(1));
  EXPECT_EQ(provider.GetAWSCredentials(), response);
  EXPECT_EQ(provider.GetAWSCredentials(), response);
  EXPECT_EQ(calls, 4u);
}

TEST_F(AwsCredentialRefreshTest, InvalidCandidatesDoNotReplaceCacheAndPoisonedDelegateIsRecreated) {
  const auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(120));
  const auto rotated = Temporary("new", now + std::chrono::hours(1));
  const auto complete = Temporary("partial", now + std::chrono::hours(1));
  std::vector<Credentials> invalid;
  auto candidate = complete;
  candidate.SetAWSAccessKeyId("");
  invalid.push_back(candidate);
  candidate = complete;
  candidate.SetAWSSecretKey("");
  invalid.push_back(candidate);
  candidate = complete;
  candidate.SetSessionToken("");
  invalid.push_back(candidate);
  candidate = complete;
  candidate.SetAWSAccessKeyId("   ");
  invalid.push_back(candidate);
  candidate = complete;
  candidate.SetExpiration(now - std::chrono::seconds(1));
  invalid.push_back(candidate);
  candidate = complete;
  candidate.SetExpiration(Aws::Utils::DateTime("bad-expiry", Aws::Utils::DateFormat::ISO_8601));
  invalid.push_back(candidate);
  invalid.emplace_back("unknown-expiry", "secret", "token");
  for (const auto& bad : invalid) {
    size_t factories = 0;
    size_t first_calls = 0;
    ValidatedAwsCredentialsProvider provider(
        [&] {
          ++factories;
          return std::make_shared<FunctionProvider>(
              [&, generation = factories] { return generation > 1 ? rotated : (++first_calls == 1 ? old : bad); });
        },
        false, [&] { return now; });
    EXPECT_EQ(provider.GetAWSCredentials(), old);
    EXPECT_EQ(provider.GetAWSCredentials(), old);
    EXPECT_EQ(provider.GetAWSCredentials(), rotated);
    EXPECT_EQ(factories, 2u);
  }
}

TEST_F(AwsCredentialRefreshTest, StaticCredentialsAreExplicitlyAllowedWithoutSessionTokenOrExpiry) {
  const Credentials fixed("static-key", "static-secret");
  auto source = std::make_shared<FunctionProvider>([&] { return fixed; });
  ValidatedAwsCredentialsProvider allowed([source] { return source; }, true);
  ValidatedAwsCredentialsProvider temporary([source] { return source; });
  EXPECT_EQ(allowed.GetAWSCredentials(), fixed);
  EXPECT_TRUE(temporary.GetAWSCredentials().IsEmpty());
}

TEST_F(AwsCredentialRefreshTest, StaticProcessContractDoesNotAllowTemporaryRefreshToLoseExpiration) {
  const auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(120));
  const auto fresh = Temporary("new", now + std::chrono::hours(1));
  size_t generations = 0;
  size_t first_calls = 0;
  ValidatedAwsCredentialsProvider provider(
      [&] {
        return std::make_shared<FunctionProvider>([&, generation = ++generations] {
          if (generation > 1)
            return fresh;
          return ++first_calls == 1 ? old : Credentials("missing-expiration", "secret");
        });
      },
      true);
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  EXPECT_EQ(provider.GetAWSCredentials(), fresh);
  EXPECT_EQ(generations, 2u);
}

TEST_F(AwsCredentialRefreshTest, DefaultChainPreservesExplicitStaticProfileAndEnvironmentPrecedence) {
  struct Environment {
    std::map<std::string, std::optional<std::string>> saved;
    void Set(const char* name, const char* value) {
      if (!saved.count(name)) {
        const auto old = std::getenv(name);
        saved[name] = old ? std::optional<std::string>(old) : std::nullopt;
      }
      if (value)
        setenv(name, value, 1);
      else
        unsetenv(name);
    }
    ~Environment() {
      for (const auto& [name, value] : saved) {
        if (value)
          setenv(name.c_str(), value->c_str(), 1);
        else
          unsetenv(name.c_str());
      }
    }
  } env;
  const auto path = std::filesystem::temp_directory_path() /
                    ("aws-refresh-" + std::string(Aws::String(Aws::Utils::UUID::RandomUUID()).c_str()));
  struct FileCleanup {
    std::filesystem::path path;
    ~FileCleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};
  std::ofstream(path) << "[refresh-test]\naws_access_key_id = profile-key\naws_secret_access_key = profile-secret\n";
  env.Set("AWS_SHARED_CREDENTIALS_FILE", path.c_str());
  env.Set("AWS_CONFIG_FILE", "/nonexistent/aws-refresh-config");
  env.Set("AWS_PROFILE", "refresh-test");
  env.Set("AWS_DEFAULT_PROFILE", "refresh-test");
  env.Set("AWS_ACCESS_KEY_ID", "env-key");
  env.Set("AWS_SECRET_ACCESS_KEY", "env-secret");
  env.Set("AWS_SESSION_TOKEN", nullptr);
  env.Set("AWS_ROLE_ARN", nullptr);
  env.Set("AWS_WEB_IDENTITY_TOKEN_FILE", nullptr);
  env.Set("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", nullptr);
  env.Set("AWS_CONTAINER_CREDENTIALS_FULL_URI", nullptr);
  env.Set("AWS_EC2_METADATA_DISABLED", "true");
  auto chain = MakeAwsDefaultCredentialsProviderChain();
  EXPECT_EQ(chain->GetAWSCredentials().GetAWSAccessKeyId(), "env-key");
  env.Set("AWS_ACCESS_KEY_ID", nullptr);
  env.Set("AWS_SECRET_ACCESS_KEY", nullptr);
  const auto profile = chain->GetAWSCredentials();
  EXPECT_EQ(profile.GetAWSAccessKeyId(), "profile-key");
  EXPECT_EQ(profile.GetAWSSecretKey(), "profile-secret");
  EXPECT_TRUE(profile.GetSessionToken().empty());
  EXPECT_EQ(chain->GetAWSCredentials(), profile);

  // A role-free credential_process may rotate temporary credentials. Its
  // original profile and file paths must survive an invalid refresh, even
  // when unrelated process environment later selects a different profile.
  const auto response_path = std::filesystem::path(path.string() + ".response");
  const auto other_path = std::filesystem::path(path.string() + ".other");
  FileCleanup response_cleanup{response_path};
  FileCleanup other_cleanup{other_path};
  std::ofstream(path) << "[refresh-test]\ncredential_process = cat " << response_path.string() << "\n";
  const auto now = Aws::Utils::DateTime::Now();
  const auto reply = [&](const char* key, bool token, std::chrono::seconds lifetime) {
    std::ofstream output(response_path);
    output << "{\"Version\":1,\"AccessKeyId\":\"" << key << "\",\"SecretAccessKey\":\"synthetic-secret\",";
    if (token)
      output << "\"SessionToken\":\"synthetic-token\",";
    output << "\"Expiration\":\"" << (now + lifetime).ToGmtString(Aws::Utils::DateFormat::ISO_8601) << "\"}";
  };
  reply("process-old", true, std::chrono::seconds(120));
  auto process_chain = MakeAwsDefaultCredentialsProviderChain();
  ASSERT_EQ(process_chain->GetAWSCredentials().GetAWSAccessKeyId(), "process-old");
  reply("process-incomplete", false, std::chrono::hours(1));
  EXPECT_EQ(process_chain->GetAWSCredentials().GetAWSAccessKeyId(), "process-old");
  std::ofstream(other_path)
      << "[other-profile]\naws_access_key_id = WRONG_IDENTITY\naws_secret_access_key = other-secret\n";
  env.Set("AWS_SHARED_CREDENTIALS_FILE", other_path.c_str());
  env.Set("AWS_PROFILE", "other-profile");
  env.Set("AWS_DEFAULT_PROFILE", "other-profile");
  reply("process-new", true, std::chrono::hours(1));
  EXPECT_EQ(process_chain->GetAWSCredentials().GetAWSAccessKeyId(), "process-new");
}

TEST_F(AwsCredentialRefreshTest, ExpiryIsRecheckedAfterFetchAndExceptionsNeverEscape) {
  auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(120));
  size_t calls = 0;
  auto source = std::make_shared<FunctionProvider>([&]() -> Credentials {
    ++calls;
    if (calls == 1)
      return old;
    if (calls == 2)
      throw std::runtime_error("synthetic failure");
    const auto result = Temporary("already-expired-at-publication", now + std::chrono::seconds(1));
    now = now + std::chrono::seconds(121);
    return result;
  });
  ValidatedAwsCredentialsProvider provider([source] { return source; }, false, [&] { return now; });
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  EXPECT_TRUE(provider.GetAWSCredentials().IsEmpty());
}

using CrtCallback = Aws::Crt::Auth::OnCredentialsResolved;
class FakeCrtSource final : public Aws::Crt::Auth::ICredentialsProvider {
  public:
  std::vector<std::function<bool(const CrtCallback&)>> replies;
  mutable size_t calls = 0;
  bool GetCredentials(const CrtCallback& callback) const override { return replies.at(calls++)(callback); }
  aws_credentials_provider* GetUnderlyingHandle() const noexcept override { return nullptr; }
  bool IsValid() const noexcept override { return true; }
};

std::shared_ptr<Aws::Crt::Auth::Credentials> CrtTuple(const char* key, uint64_t expiry) {
  return std::make_shared<Aws::Crt::Auth::Credentials>(Aws::Crt::ByteCursorFromCString(key),
                                                       Aws::Crt::ByteCursorFromCString("synthetic-secret"),
                                                       Aws::Crt::ByteCursorFromCString("synthetic-token"), expiry);
}

TEST_F(AwsCredentialRefreshTest, RealSdkCrtTimeoutAndLateCallbackCannotDestroyProtectedCache) {
  const auto now = Aws::Utils::DateTime::Now();
  const auto expiry = static_cast<uint64_t>(now.Seconds() + 120);
  CrtCallback late;
  auto source = std::make_shared<FakeCrtSource>();
  source->replies = {[=](const CrtCallback& cb) {
                       cb(CrtTuple("old", expiry), 0);
                       return true;
                     },
                     [&](const CrtCallback& cb) {
                       late = cb;
                       return true;
                     },
                     [=](const CrtCallback& cb) {
                       cb(CrtTuple("new", expiry + 3600), 0);
                       return true;
                     }};
  ValidatedAwsCredentialsProvider provider([source] {
    return std::make_shared<Aws::Auth::CrtCredentialsProvider>(
        [source] { return source; }, std::chrono::milliseconds(1),
        Aws::Client::UserAgentFeature::CREDENTIALS_STS_WEB_IDENTITY_TOKEN, "Test");
  });
  const auto old = provider.GetAWSCredentials();
  ASSERT_EQ(old.GetAWSAccessKeyId(), "old");
  EXPECT_EQ(provider.GetAWSCredentials(), old);
  late(CrtTuple("late", expiry + 3600), 0);
  EXPECT_EQ(provider.GetAWSCredentials().GetAWSAccessKeyId(), "new");
  EXPECT_EQ(source->calls, 3u);
}

TEST_F(AwsCredentialRefreshTest, CrtImdsBridgePreservesActualExpiryAndRejectsUnknownLifetime) {
  auto now = Aws::Utils::DateTime::Now();
  const auto expiry = static_cast<uint64_t>(now.Seconds() + 120);
  auto source = std::make_shared<FakeCrtSource>();
  source->replies = {[=](const CrtCallback& cb) {
                       cb(CrtTuple("old", expiry), 0);
                       return true;
                     },
                     [](const CrtCallback& cb) {
                       cb(nullptr, 1);
                       return true;
                     },
                     [](const CrtCallback& cb) {
                       cb(CrtTuple("unknown", std::numeric_limits<uint64_t>::max()), 0);
                       return true;
                     },
                     [](const CrtCallback& cb) {
                       cb(nullptr, 1);
                       return true;
                     }};
  const auto provider = MakeAwsCrtCredentialsProvider(source, std::chrono::milliseconds(1), [&] { return now; });
  const auto old = provider->GetAWSCredentials();
  EXPECT_EQ(old.GetExpiration().Seconds(), expiry);
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  now = now + std::chrono::seconds(121);
  EXPECT_TRUE(provider->GetAWSCredentials().IsEmpty());
}

TEST_F(AwsCredentialRefreshTest, ConcurrentFailedRefreshNeverPublishesPartialCredentials) {
  const auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(120));
  const auto fresh = Temporary("new", now + std::chrono::hours(1));
  size_t calls = 0;
  auto source =
      std::make_shared<FunctionProvider>([&] { return ++calls == 1 ? old : (calls == 2 ? Credentials{} : fresh); });
  auto provider = std::make_shared<ValidatedAwsCredentialsProvider>([source] { return source; });
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  std::promise<void> start;
  const auto ready = start.get_future().share();
  std::vector<std::future<Credentials>> results;
  for (int i = 0; i < 16; ++i)
    results.push_back(std::async(std::launch::async, [provider, ready] {
      ready.wait();
      return provider->GetAWSCredentials();
    }));
  start.set_value();
  for (auto& result : results) {
    const auto value = result.get();
    EXPECT_TRUE(value == old || value == fresh);
  }
  EXPECT_EQ(calls, 3u);
}

class TestChain final : public Aws::Auth::AWSCredentialsProviderChain {
  public:
  TestChain(std::shared_ptr<Aws::Auth::AWSCredentialsProvider> first,
            std::shared_ptr<Aws::Auth::AWSCredentialsProvider> second) {
    AddProvider(std::move(first));
    AddProvider(std::move(second));
  }
};

TEST_F(AwsCredentialRefreshTest, FailedEarlyRefreshDoesNotSwitchToFallbackIdentity) {
  const auto now = Aws::Utils::DateTime::Now();
  size_t calls = 0;
  auto source = std::make_shared<FunctionProvider>(
      [&] { return ++calls == 1 ? Temporary("irsa", now + std::chrono::seconds(120)) : Credentials{}; });
  auto protected_source = std::make_shared<ValidatedAwsCredentialsProvider>([source] { return source; });
  auto fallback = std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>("node", "node-secret");
  TestChain chain(protected_source, fallback);
  EXPECT_EQ(chain.GetAWSCredentials().GetAWSAccessKeyId(), "irsa");
  EXPECT_EQ(chain.GetAWSCredentials().GetAWSAccessKeyId(), "irsa");
  EXPECT_EQ(chain.GetAWSCredentials().GetAWSAccessKeyId(), "irsa");
}

class FakeStsClient final : public Aws::STS::STSClient {
  public:
  FakeStsClient()
      : Aws::STS::STSClient(
            Credentials("synthetic-source", "synthetic-secret"),
            Aws::Client::ClientConfiguration(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true})) {}
  std::vector<Aws::STS::Model::AssumeRoleOutcome> replies;
  mutable size_t calls = 0;
  Aws::STS::Model::AssumeRoleOutcome AssumeRole(const Aws::STS::Model::AssumeRoleRequest&) const override {
    return replies.at(calls++);
  }
};

Aws::STS::Model::AssumeRoleOutcome RoleReply(const Credentials& tuple) {
  Aws::STS::Model::Credentials credentials;
  credentials.SetAccessKeyId(tuple.GetAWSAccessKeyId());
  credentials.SetSecretAccessKey(tuple.GetAWSSecretKey());
  credentials.SetSessionToken(tuple.GetSessionToken());
  credentials.SetExpiration(tuple.GetExpiration());
  Aws::STS::Model::AssumeRoleResult result;
  result.SetCredentials(credentials);
  return result;
}

TEST_F(AwsCredentialRefreshTest, AssumeRoleRejectsPartialSuccessPreservesRealExpiryAndRecovers) {
  auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(30));
  auto invalid = Temporary("partial", now + std::chrono::hours(1));
  invalid.SetAWSSecretKey("");
  const auto fresh = Temporary("new", now + std::chrono::hours(1));
  auto client = std::make_shared<FakeStsClient>();
  client->replies = {RoleReply(old), RoleReply(invalid), RoleReply(fresh)};
  auto provider =
      MakeAwsAssumeRoleCredentialsProvider("synthetic-role", "synthetic-session", "", 900, client, [&] { return now; });
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  EXPECT_EQ(provider->GetAWSCredentials(), fresh);
  EXPECT_EQ(provider->GetAWSCredentials(), fresh);
  EXPECT_EQ(client->calls, 3u);
}

TEST_F(AwsCredentialRefreshTest, AssumeRoleFailureAfterActualExpirationReturnsEmpty) {
  auto now = Aws::Utils::DateTime::Now();
  const auto old = Temporary("old", now + std::chrono::seconds(30));
  auto client = std::make_shared<FakeStsClient>();
  client->replies = {RoleReply(old),
                     Aws::STS::STSError(Aws::Client::AWSError<Aws::STS::STSErrors>(
                         Aws::STS::STSErrors::NETWORK_CONNECTION, "Synthetic", "Synthetic failure", true))};
  auto provider =
      MakeAwsAssumeRoleCredentialsProvider("synthetic-role", "synthetic-session", "", 900, client, [&] { return now; });
  EXPECT_EQ(provider->GetAWSCredentials(), old);
  now = now + std::chrono::seconds(31);
  EXPECT_TRUE(provider->GetAWSCredentials().IsEmpty());
}

}  // namespace
}  // namespace milvus_storage
