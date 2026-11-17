// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <azure/identity/detail/token_cache.hpp>
#include <chrono>
#include <functional>
#include <map>
#include "milvus-storage/filesystem/azure/refreshing_token_credential.h"

namespace milvus_storage::fs {
namespace {
using namespace std::chrono_literals;
using Time = std::chrono::system_clock::time_point;
using Azure::Core::Credentials::AccessToken;
using Azure::Core::Credentials::AuthenticationException;
using Azure::Core::Credentials::TokenRequestContext;

// Exercise the real SDK TokenCache, including its acceptance of empty future
// tokens. The wrapper must force the same identity to refresh after rejection.
class CachedDelegate : public Azure::Core::Credentials::TokenCredential {
  public:
  CachedDelegate() : TokenCredential("fixed-test-identity") {}
  std::function<AccessToken()> fetch;
  mutable int fetches = 0;
  mutable bool forced = false;
  AccessToken GetToken(const TokenRequestContext& request, const Azure::Core::Context&) const override {
    forced = request.MinimumExpiration == Azure::DateTime::duration::max();
    std::string scopes;
    for (const auto& scope : request.Scopes) {
      scopes += scope + " ";
    }
    return cache_.GetToken(scopes, request.TenantId, request.MinimumExpiration, [&] {
      ++fetches;
      return fetch();
    });
  }

  private:
  Azure::Identity::_detail::TokenCache cache_;
};

struct AzureRefreshTest : ::testing::Test {
  Time now = std::chrono::system_clock::now();
  std::shared_ptr<CachedDelegate> delegate = std::make_shared<CachedDelegate>();
  RefreshingAzureCredential credential{delegate, [&] { return now; }};
  TokenRequestContext context;
  void SetUp() override {
    context.Scopes = {"storage"};
    context.TenantId = "tenant-a";
  }
  AccessToken Token(const std::string& value, std::chrono::seconds lifetime) {
    return {value, Azure::DateTime(now + lifetime)};
  }
  void LoadOld() {
    delegate->fetch = [&] { return Token("OLD", 60s); };
    ASSERT_EQ(credential.GetToken(context, {}).Token, "OLD");
  }
};

TEST_F(AzureRefreshTest, ExceptionRetainsValidOldAndLaterRecovers) {
  LoadOld();
  delegate->fetch = []() -> AccessToken { throw AuthenticationException("synthetic"); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
  delegate->fetch = [&] { return Token("NEW", 3600s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "NEW");
  EXPECT_TRUE(delegate->forced);
}

TEST_F(AzureRefreshTest, EmptySdkCacheIsBypassedOnNextRefreshWithoutChangingIdentity) {
  LoadOld();
  delegate->fetch = [&] { return Token("", 3600s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
  delegate->fetch = [&] { return Token("RECOVERED", 3600s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "RECOVERED");
  EXPECT_TRUE(delegate->forced);
  EXPECT_EQ(delegate->fetches, 3);
  EXPECT_EQ(credential.GetCredentialName(), "fixed-test-identity");
}

TEST_F(AzureRefreshTest, RejectsExpiredWhitespaceAndHeaderInjection) {
  for (const auto* value : {"", " ", "bad\r\nheader"}) {
    LoadOld();
    delegate->fetch = [&] { return Token(value, 3600s); };
    EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
  }
  delegate->fetch = [&] { return Token("EXPIRED", -1s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
}

TEST_F(AzureRefreshTest, FailureDoesNotReturnExpiredOldAfterSlowFetch) {
  LoadOld();
  delegate->fetch = [&]() -> AccessToken {
    now += 61s;
    throw AuthenticationException("synthetic");
  };
  EXPECT_THROW(credential.GetToken(context, {}), AuthenticationException);
  delegate->fetch = [&] { return Token("NEW", 3600s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "NEW");
}

TEST_F(AzureRefreshTest, CandidateMustStillBeValidWhenFetchCompletes) {
  LoadOld();
  delegate->fetch = [&] {
    auto token = Token("SHORT", 1s);
    now += 2s;
    return token;
  };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
}

TEST_F(AzureRefreshTest, NeverFallsBackAcrossTenantOrScopes) {
  LoadOld();
  delegate->fetch = []() -> AccessToken { throw AuthenticationException("synthetic"); };
  auto other = context;
  other.TenantId = "tenant-b";
  EXPECT_THROW(credential.GetToken(other, {}), AuthenticationException);
  other = context;
  other.Scopes = {"other-service"};
  EXPECT_THROW(credential.GetToken(other, {}), AuthenticationException);
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
}

TEST_F(AzureRefreshTest, InitialFailureRecoversAndForcedInvalidationDoesNotReuseRejectedToken) {
  delegate->fetch = [&] { return Token("", 3600s); };
  EXPECT_THROW(credential.GetToken(context, {}), AuthenticationException);
  delegate->fetch = [&] { return Token("OLD", 60s); };
  EXPECT_EQ(credential.GetToken(context, {}).Token, "OLD");
  auto rejected = context;
  rejected.MinimumExpiration = Azure::DateTime::duration::max();
  delegate->fetch = []() -> AccessToken { throw AuthenticationException("synthetic"); };
  EXPECT_THROW(credential.GetToken(rejected, {}), AuthenticationException);
  EXPECT_THROW(credential.GetToken(context, {}), AuthenticationException);
}
}  // namespace
}  // namespace milvus_storage::fs
