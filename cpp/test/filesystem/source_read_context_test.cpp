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

#include "milvus-storage/source_read_context.h"

#include <future>
#include <set>
#include <stdexcept>

#include <gtest/gtest.h>

namespace milvus_storage {
namespace {

constexpr int64_t kNow = 1893456000000;  // 2030-01-01T00:00:00Z
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data/file.parquet";

class FunctionProvider final : public SourceCredentialProvider {
  public:
  explicit FunctionProvider(std::function<SourceCredentialResponse(const SourceCredentialRequest&)> callback)
      : callback_(std::move(callback)) {}
  SourceCredentialResponse Fetch(const SourceCredentialRequest& request) const override { return callback_(request); }

  private:
  std::function<SourceCredentialResponse(const SourceCredentialRequest&)> callback_;
};

SourceCredentialResponse Credential(const SourceCredentialRequest& request) {
  return {LOON_SOURCE_CREDENTIAL_OK,
          LOON_SOURCE_CREDENTIAL_KIND_AZURE_SAS,
          "acct",
          request.scope_root_uri,
          "sv=2023-11-03&sr=d&sdd=2&sp=rl&se=2030-01-01T00%3A02%3A00Z&sig=test-signature",
          kNow + 120000};
}

SourceReadContextPtr MakeContext(std::function<SourceCredentialResponse(const SourceCredentialRequest&)> callback,
                                 const std::shared_ptr<std::atomic<int64_t>>& clock,
                                 int64_t deadline = 0) {
  auto result = SourceReadContext::Create({"acct", {kRoot}, deadline, 3000},
                                          std::make_shared<FunctionProvider>(std::move(callback)),
                                          [clock] { return clock->load(); });
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : nullptr;
}

TEST(SourceReadContext, ConfinesReadsBeforeCallingProvider) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  int calls = 0;
  const auto context = MakeContext(
      [&](const auto& request) {
        ++calls;
        EXPECT_EQ(request.scope_root_uri, kRoot);
        EXPECT_EQ(request.resource_uri, kFile);
        EXPECT_EQ(request.request_deadline_unix_ms, kNow + 3000);
        return Credential(request);
      },
      clock);
  ASSERT_TRUE(context);
  EXPECT_EQ(context->Fetch("https://acct.blob.core.windows.net/tables/team/table-other/data", LOON_SOURCE_READ).status,
            LOON_SOURCE_CREDENTIAL_DENIED);
  EXPECT_EQ(context->Fetch("https://other.blob.core.windows.net/tables/team/table/data", LOON_SOURCE_READ).status,
            LOON_SOURCE_CREDENTIAL_DENIED);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(context->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_OK);
  EXPECT_EQ(calls, 1);
}

TEST(SourceReadContext, NativeHttpReadAndListReachIssuerWithTheActualOperation) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  std::vector<SourceCredentialRequest> calls;
  const auto context = MakeContext(
      [&](const auto& request) {
        calls.push_back(request);
        return Credential(request);
      },
      clock);
  ASSERT_TRUE(context);
  EXPECT_EQ(context->FetchHttpRequest("HEAD", kFile).status, LOON_SOURCE_CREDENTIAL_OK);
  constexpr auto blob_list =
      "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table/&delimiter=/";
  constexpr auto dfs_list =
      "https://acct.dfs.core.windows.net/tables?resource=filesystem&directory=team/table&recursive=true";
  for (const auto* uri : {blob_list, dfs_list}) {
    const auto target = context->ResolveHttpRequest("GET", uri);
    EXPECT_EQ(target.status, LOON_SOURCE_CREDENTIAL_OK);
    EXPECT_EQ(target.root_index, 0);
    EXPECT_EQ(target.resource_uri, kRoot);
    EXPECT_EQ(target.operation, LOON_SOURCE_LIST);
    EXPECT_EQ(context->FetchHttpRequest("GET", uri).status, LOON_SOURCE_CREDENTIAL_OK);
  }
  ASSERT_EQ(calls.size(), 3);
  EXPECT_EQ(calls[0].operation, LOON_SOURCE_READ);
  EXPECT_EQ(calls[0].resource_uri, kFile);
  for (size_t i = 1; i < calls.size(); ++i) {
    EXPECT_EQ(calls[i].operation, LOON_SOURCE_LIST);
    EXPECT_EQ(calls[i].resource_uri, kRoot);
    EXPECT_EQ(calls[i].scope_root_uri, kRoot);
  }
}

TEST(SourceReadContext, NativeHttpRejectionAndTerminalStateNeverCallIssuer) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  int calls = 0;
  const auto context = MakeContext(
      [&](const auto& request) {
        ++calls;
        return Credential(request);
      },
      clock);
  ASSERT_TRUE(context);
  const std::pair<const char*, const char*> rejected[] = {
      {"PUT", kFile},
      {"HEAD", "https://acct.dfs.core.windows.net/tables/?action=getAccessControl"},
      {"GET", "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table"},
      {"GET", "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table-other/"},
      {"GET", "https://acct.blob.core.windows.net/tables/team/table/data?sig=untrusted"},
  };
  for (const auto& [method, uri] : rejected) {
    EXPECT_EQ(context->FetchHttpRequest(method, uri).status, LOON_SOURCE_CREDENTIAL_DENIED);
  }
  EXPECT_EQ(calls, 0);
  context->Cancel();
  EXPECT_EQ(context->FetchHttpRequest("GET", kFile).status, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(context->ResolveHttpRequest("GET", kFile).status, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(calls, 0);

  const auto expired = MakeContext(
      [&](const auto& request) {
        ++calls;
        return Credential(request);
      },
      clock, kNow + 1000);
  ASSERT_TRUE(expired);
  clock->store(kNow + 1000);
  EXPECT_EQ(expired->FetchHttpRequest("GET", kFile).status, LOON_SOURCE_CREDENTIAL_EXPIRED);
  EXPECT_EQ(calls, 0);
}

TEST(SourceReadContext, NativeHttpListCannotUseReadOnlyCredential) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  const auto context = MakeContext(
      [](const auto& request) {
        auto response = Credential(request);
        const auto permission = response.sas_token.find("sp=rl");
        response.sas_token.replace(permission, 5, "sp=r");
        return response;
      },
      clock);
  ASSERT_TRUE(context);
  EXPECT_EQ(context->FetchHttpRequest("GET", kFile).status, LOON_SOURCE_CREDENTIAL_OK);
  EXPECT_EQ(context
                ->FetchHttpRequest(
                    "GET", "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table/")
                .status,
            LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
}

TEST(SourceReadContext, SuppliedRootCannotRedirectSasToAnArbitraryHost) {
  auto provider = std::make_shared<FunctionProvider>([](const auto& request) {
    ADD_FAILURE() << "Invalid context must not request credentials";
    return Credential(request);
  });
  for (const auto* root : {"https://acct.blob.core.windows.net.attacker.example/tables/team/table",
                           "abfss://tables@acct.dfs.attacker.example/team/table"}) {
    const auto context = SourceReadContext::Create({"acct", {root}, 0, 3000}, provider);
    EXPECT_FALSE(context.ok());
  }
  const auto sovereign = SourceReadContext::Create(
      {"acct", {"abfss://tables@acct.dfs.core.usgovcloudapi.net/team/table"}, 0, 3000}, provider);
  EXPECT_TRUE(sovereign.ok()) << sovereign.status();
}

TEST(SourceReadContext, RejectsInvalidIssuerScopePermissionsAndActualSignedExpiry) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  const std::vector<std::function<void(SourceCredentialResponse&)>> mutations{
      [](auto& response) { response.account_name = "other"; },
      [](auto& response) { response.scope_root_uri = "https://acct.blob.core.windows.net/tables/team"; },
      [](auto& response) { response.kind = LOON_SOURCE_CREDENTIAL_KIND_UNSPECIFIED; },
      [](auto& response) { response.expires_at_unix_ms = kNow; },
      [](auto& response) { response.sas_token = "sr=c&sp=rl&se=2030-01-01T00:02:00Z&sig=x"; },
      [](auto& response) { response.sas_token = "sr=d&sdd=1&sp=rl&se=2030-01-01T00:02:00Z&sig=x"; },
      [](auto& response) { response.sas_token = "sr=d&sdd=2&sp=rlw&se=2030-01-01T00:02:00Z&sig=x"; },
      [](auto& response) { response.sas_token = "sr=d&sdd=2&sp=l&se=2030-01-01T00:02:00Z&sig=x"; },
      [](auto& response) { response.sas_token = "sr=d&sdd=2&sp=rl&se=2030-01-01T00:00:00Z&sig=x"; },
      [](auto& response) { response.sas_token += "&se=2099-01-01T00:00:00Z"; },
      [](auto& response) { response.sas_token += "&st=2030-01-01T00:01:00Z"; },
      [](auto& response) { response.sas_token = "sr=d&sdd=2&sp=rl&se=2030-02-31T00:00:00Z&sig=x"; },
  };
  for (size_t i = 0; i < mutations.size(); ++i) {
    SCOPED_TRACE(i);
    const auto context = MakeContext(
        [&](const auto& request) {
          auto response = Credential(request);
          mutations[i](response);
          return response;
        },
        clock);
    ASSERT_TRUE(context);
    const auto response = context->Fetch(kFile, LOON_SOURCE_READ);
    EXPECT_EQ(response.status, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
    EXPECT_TRUE(response.sas_token.empty());
  }
  const auto context = MakeContext(
      [](const auto& request) {
        auto response = Credential(request);
        response.expires_at_unix_ms = kNow + 600000;
        return response;
      },
      clock);
  const auto response = context->Fetch(kFile, LOON_SOURCE_READ);
  EXPECT_EQ(response.status, LOON_SOURCE_CREDENTIAL_OK);
  EXPECT_EQ(response.expires_at_unix_ms, kNow + 120000);
}

TEST(SourceReadContext, DeadlineBoundsSignedAuthorityAndLateResponses) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  const auto overlong = MakeContext(Credential, clock, kNow + 60000);
  ASSERT_TRUE(overlong);
  EXPECT_EQ(overlong->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);

  const auto late = MakeContext(
      [clock](const auto& request) {
        clock->store(request.request_deadline_unix_ms);
        return Credential(request);
      },
      clock);
  ASSERT_TRUE(late);
  EXPECT_EQ(late->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_TRANSIENT);

  clock->store(kNow);
  const auto expired = MakeContext(
      [clock](const auto& request) {
        clock->store(request.operation_deadline_unix_ms);
        return Credential(request);
      },
      clock, kNow + 1000);
  ASSERT_TRUE(expired);
  EXPECT_EQ(expired->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_EXPIRED);
  clock->store(kNow);  // A clock correction must not resurrect an expired context.
  EXPECT_EQ(expired->CheckState(), LOON_SOURCE_CREDENTIAL_EXPIRED);
}

TEST(SourceReadContext, DenialIsTerminalAndNeverReturnsCallbackSecrets) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  int calls = 0;
  const auto context = MakeContext(
      [&](const auto& request) {
        ++calls;
        auto response = Credential(request);
        response.status = LOON_SOURCE_CREDENTIAL_DENIED;
        return response;
      },
      clock);
  ASSERT_TRUE(context);
  EXPECT_EQ(context->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_DENIED);
  EXPECT_EQ(context->CheckState(), LOON_SOURCE_CREDENTIAL_DENIED);
  const auto retry = context->Fetch(kFile, LOON_SOURCE_READ);
  EXPECT_EQ(retry.status, LOON_SOURCE_CREDENTIAL_DENIED);
  EXPECT_TRUE(retry.sas_token.empty());
  EXPECT_EQ(calls, 1);
}

TEST(SourceReadContext, CancellationRejectsAnInFlightSuccess) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  std::promise<void> entered;
  std::promise<void> finish;
  const auto finished = finish.get_future().share();
  const auto context = MakeContext(
      [&](const auto& request) {
        entered.set_value();
        finished.wait();
        return Credential(request);
      },
      clock);
  ASSERT_TRUE(context);
  auto pending = std::async(std::launch::async, [context] { return context->Fetch(kFile, LOON_SOURCE_READ); });
  entered.get_future().wait();
  context->Cancel();
  finish.set_value();
  const auto response = pending.get();
  EXPECT_EQ(response.status, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_TRUE(response.sas_token.empty());
}

TEST(SourceReadContext, ConcurrentContextsNeverReuseIdentity) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  std::vector<std::future<SourceReadContextPtr>> pending;
  for (int i = 0; i < 64; ++i) {
    pending.push_back(std::async(std::launch::async, [clock] { return MakeContext(Credential, clock); }));
  }
  std::set<uint64_t> identities;
  for (auto& future : pending) {
    const auto context = future.get();
    ASSERT_TRUE(context);
    EXPECT_TRUE(identities.insert(context->identity()).second);
    EXPECT_NE(context->CacheIdentity(0), context->CacheIdentity(1));
  }
}

TEST(SourceReadContext, ThrowingProviderProducesNoExceptionTextOrCredentials) {
  const auto clock = std::make_shared<std::atomic<int64_t>>(kNow);
  const auto context = MakeContext(
      [](const auto&) -> SourceCredentialResponse {
        throw std::runtime_error("issuer URL containing sig=private-value");
      },
      clock);
  ASSERT_TRUE(context);
  const auto response = context->Fetch(kFile, LOON_SOURCE_READ);
  EXPECT_EQ(response.status, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  EXPECT_TRUE(response.sas_token.empty());
  EXPECT_TRUE(response.account_name.empty());
  EXPECT_TRUE(response.scope_root_uri.empty());
}

}  // namespace
}  // namespace milvus_storage
