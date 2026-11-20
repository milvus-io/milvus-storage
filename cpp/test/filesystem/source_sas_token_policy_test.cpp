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

#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"
#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/ffi_internal/result.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <future>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <azure/core/datetime.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/internal/client_options.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/io/body_stream.hpp>
#include <azure/storage/blobs.hpp>
#include <azure/storage/files/datalake.hpp>
#include <gtest/gtest.h>

namespace milvus_storage::fs {
namespace {
constexpr int64_t kNow = 1893456000000;
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kOtherRoot = "https://acct.blob.core.windows.net/tables/team/other";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data.parquet";
constexpr auto kList = "https://acct.blob.core.windows.net/tables?restype=container&comp=list&prefix=team/table/";

class Provider final : public SourceCredentialProvider {
  public:
  std::function<SourceCredentialResponse(const SourceCredentialRequest&)> fetch;
  SourceCredentialResponse Fetch(const SourceCredentialRequest& request) const override { return fetch(request); }
};

class Transport final : public Azure::Core::Http::HttpTransport {
  public:
  std::function<Azure::Core::Http::HttpStatusCode(size_t)> respond;
  std::vector<std::map<std::string, std::string>> requests;
  std::vector<std::string> urls;
  bool range_body = false;
  std::mutex mutex;
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       const Azure::Core::Context&) override {
    std::lock_guard lock(mutex);
    requests.push_back(request.GetUrl().GetQueryParameters());
    urls.push_back(request.GetUrl().GetAbsoluteUrl());
    const auto status = respond ? respond(requests.size()) : Azure::Core::Http::HttpStatusCode::Ok;
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(1, 1, status, "injected");
    // TransportPolicy consumes a non-null stream, including for empty bodies.
    static const std::vector<uint8_t> empty_body;
    static const std::vector<uint8_t> data_body{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
    if (range_body) {
      response->SetHeader("Content-Length", "8");
      response->SetHeader("Content-Range", "bytes 0-7/8");
      response->SetHeader("Last-Modified", "Tue, 01 Jan 2030 00:00:00 GMT");
      response->SetHeader("x-ms-creation-time", "Tue, 01 Jan 2030 00:00:00 GMT");
      response->SetHeader("ETag", "\"test-etag\"");
      response->SetHeader("x-ms-blob-type", "BlockBlob");
      response->SetHeader("x-ms-server-encrypted", "true");
    }
    response->SetBodyStream(std::make_unique<Azure::Core::IO::MemoryBodyStream>(range_body ? data_body : empty_body));
    return response;
  }
};

struct PolicyFailure {
  arrow::Status status;
};

class SourceSasTokenPolicyTest : public ::testing::Test {
  protected:
  std::shared_ptr<std::atomic<int64_t>> now = std::make_shared<std::atomic<int64_t>>(kNow);
  std::shared_ptr<Provider> provider = std::make_shared<Provider>();
  std::shared_ptr<Transport> transport = std::make_shared<Transport>();
  std::atomic<int> calls{0};

  SourceCredentialResponse Lease(const SourceCredentialRequest& request, const std::string& signature = "lease") {
    const auto expiry = now->load() + 120000;
    const auto time = Azure::DateTime(std::chrono::system_clock::time_point(std::chrono::milliseconds(expiry)));
    return {LOON_SOURCE_CREDENTIAL_OK,
            LOON_SOURCE_CREDENTIAL_KIND_AZURE_SAS,
            "acct",
            request.scope_root_uri,
            "sv=2023-11-03&sr=d&sdd=2&sp=rl&se=" + time.ToString(Azure::DateTime::DateFormat::Rfc3339) +
                "&sig=" + signature,
            expiry};
  }

  SourceReadContextPtr Context(int64_t deadline = 0) {
    if (!provider->fetch) {
      provider->fetch = [&](const auto& request) {
        ++calls;
        return Lease(request);
      };
    }
    auto result = SourceReadContext::Create({"acct", {kRoot, kOtherRoot}, deadline, 3000}, provider,
                                            [clock = now] { return clock->load(); });
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *result : nullptr;
  }

  std::shared_ptr<AzureFileSystem> FileSystem(SourceReadContextPtr context) {
    AzureOptions options;
    options.account_name = "acct";
    EXPECT_TRUE(options.ConfigureSourceReadContext(std::move(context), transport).ok());
    auto result = AzureFileSystem::Make(options);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *result : nullptr;
  }

  void ExpectStatus(const arrow::Status& status, LoonSourceCredentialStatus expected) {
    auto detail = std::dynamic_pointer_cast<SourceCredentialStatusDetail>(status.detail());
    ASSERT_TRUE(detail) << status;
    EXPECT_EQ(detail->status(), expected);
    const std::map<LoonSourceCredentialStatus, int> codes{
        {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_AUTH_DENIED},
        {LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_AUTH_EXPIRED},
        {LOON_SOURCE_CREDENTIAL_TRANSIENT, LOON_SOURCE_AUTH_TRANSIENT},
        {LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_AUTH_CANCELLED},
        {LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE, LOON_SOURCE_AUTH_INVALID_RESPONSE}};
    auto result = [&]() -> LoonFFIResult { RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, "sig=must-not-be-exposed"); }();
    EXPECT_EQ(result.err_code, codes.at(expected));
    EXPECT_EQ(loon_ffi_is_retryable_errcode(result.err_code), expected == LOON_SOURCE_CREDENTIAL_TRANSIENT);
    EXPECT_EQ(std::string(result.message).find("sig="), std::string::npos);
    loon_ffi_free_result(&result);
  }

  void Send(const SourceSasTokenPolicy& policy,
            const std::string& uri = kFile,
            const Azure::Core::Context& context = {}) {
    Azure::Core::_internal::ClientOptions options;
    options.PerRetryPolicies.emplace_back(policy.Clone());
    options.Transport.Transport = transport;
    options.Retry.MaxRetries = 1;
    options.Retry.RetryDelay = std::chrono::milliseconds(0);
    options.Retry.MaxRetryDelay = std::chrono::milliseconds(0);
    Azure::Core::Http::_internal::HttpPipeline pipeline(options, "test", "1.0.0", {}, {});
    Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Get, Azure::Core::Url(uri));
    const auto original = request.GetUrl().GetAbsoluteUrl();
    try {
      auto response = pipeline.Send(request, context);
      if (auto failure = SourcePolicyResponseStatus(*response))
        throw PolicyFailure{*failure};
    } catch (...) {
      EXPECT_EQ(request.GetUrl().GetAbsoluteUrl(), original);
      throw;
    }
    EXPECT_EQ(request.GetUrl().GetAbsoluteUrl(), original);
  }

  template <typename F>
  void ExpectFailure(F action, LoonSourceCredentialStatus status) {
    try {
      action();
      FAIL() << "Expected typed source credential failure";
    } catch (const PolicyFailure& error) {
      ExpectStatus(error.status, status);
    }
  }
};

TEST_F(SourceSasTokenPolicyTest, NullContextFailsWithoutThrowingFromThePolicy) {
  SourceSasTokenPolicy policy(nullptr);
  ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(transport->requests.empty());
}

TEST_F(SourceSasTokenPolicyTest, OrdinaryAzureErrorsAreNotSourcePolicyFailures) {
  using Azure::Core::Http::HttpStatusCode;
  using Azure::Core::Http::RawResponse;
  RawResponse ordinary(1, 1, HttpStatusCode::BadRequest, "Bad Request");
  ordinary.SetHeader("x-ms-error-code", "InvalidQueryParameterValue");
  EXPECT_FALSE(SourcePolicyResponseStatus(ordinary));
  ordinary.SetHeader("x-loon-source-failure", "denied");
  EXPECT_FALSE(SourcePolicyResponseStatus(ordinary));
  RawResponse malformed(1, 1, HttpStatusCode::BadRequest, "Loon source policy failure");
  const auto status = SourcePolicyResponseStatus(malformed);
  ASSERT_TRUE(status);
  EXPECT_TRUE(status->IsUnknownError());
  EXPECT_FALSE(status->detail());
}

TEST_F(SourceSasTokenPolicyTest, PartitionsByContextRootAndOperationWhileClonesReuseLease) {
  auto context = Context();
  SourceSasTokenPolicy first(context);
  Send(first);
  Send(first);
  EXPECT_EQ(calls, 1);
  Send(first, kList);
  EXPECT_EQ(calls, 2);
  Send(first, "https://acct.blob.core.windows.net/tables/team/other/file");
  EXPECT_EQ(calls, 3);
  SourceSasTokenPolicy second(Context());
  EXPECT_NE(context->identity(), Context()->identity());
  Send(second);
  EXPECT_EQ(calls, 4);
  const auto listed = transport->requests[2];
  EXPECT_EQ(listed.at("prefix"), "team/table/");
  EXPECT_EQ(listed.at("comp"), "list");
}

TEST_F(SourceSasTokenPolicyTest, ActualBlobAndDataLakeClientsRenewBetweenAttemptsAndPreserveFailureType) {
  for (const bool data_lake : {false, true}) {
    for (const auto failure : {LOON_SOURCE_CREDENTIAL_OK, LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED,
                               LOON_SOURCE_CREDENTIAL_TRANSIENT, LOON_SOURCE_CREDENTIAL_CANCELLED,
                               LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
      SCOPED_TRACE(data_lake);
      SCOPED_TRACE(failure);
      now->store(kNow);
      calls.store(0);
      transport = std::make_shared<Transport>();
      provider->fetch = [&](const auto& request) {
        if (++calls == 1) {
          auto lease = Lease(request, "old");
          lease.sas_token += "&skoid=old-owner";
          return lease;
        }
        if (failure != LOON_SOURCE_CREDENTIAL_OK) {
          SourceCredentialResponse result;
          result.status = failure;
          return result;
        }
        return Lease(request, "new");
      };
      transport->respond = [&](size_t attempt) {
        if (attempt == 1) {
          now->fetch_add(121000);
          return Azure::Core::Http::HttpStatusCode::ServiceUnavailable;
        }
        return Azure::Core::Http::HttpStatusCode::NotFound;
      };
      SourceSasTokenPolicy policy(Context());
      auto configure = [&](auto& options) {
        options.PerRetryPolicies.emplace_back(policy.Clone());
        options.Transport.Transport = transport;
        options.Retry.MaxRetries = 1;
        options.Retry.RetryDelay = std::chrono::milliseconds(0);
        options.Retry.MaxRetryDelay = std::chrono::milliseconds(0);
      };
      try {
        if (data_lake) {
          Azure::Storage::Files::DataLake::DataLakeClientOptions options;
          configure(options);
          Azure::Storage::Files::DataLake::DataLakeFileClient client(
              "https://acct.dfs.core.windows.net/tables/team/table/data.parquet", options);
          client.GetProperties();
        } else {
          Azure::Storage::Blobs::BlobClientOptions options;
          configure(options);
          Azure::Storage::Blobs::BlobClient client(kFile, options);
          client.GetProperties();
        }
        FAIL() << "Expected injected terminal response";
      } catch (const Azure::Storage::StorageException& error) {
        if (failure == LOON_SOURCE_CREDENTIAL_OK) {
          EXPECT_EQ(error.StatusCode, Azure::Core::Http::HttpStatusCode::NotFound);
          ASSERT_TRUE(error.RawResponse);
          EXPECT_FALSE(SourcePolicyResponseStatus(*error.RawResponse));
        } else {
          ASSERT_TRUE(error.RawResponse);
          const auto status = SourcePolicyResponseStatus(*error.RawResponse);
          ASSERT_TRUE(status);
          ExpectStatus(*status, failure);
        }
      }
      EXPECT_EQ(calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 5 : 2);
      ASSERT_EQ(transport->requests.size(), failure == LOON_SOURCE_CREDENTIAL_OK ? 2 : 1);
      EXPECT_EQ(transport->requests[0].at("sig"), "old");
      if (failure == LOON_SOURCE_CREDENTIAL_OK) {
        EXPECT_EQ(transport->requests[1].at("sig"), "new");
        EXPECT_FALSE(transport->requests[1].contains("skoid"));
      }
    }
  }
}

TEST_F(SourceSasTokenPolicyTest, TransientFallbackNeverUsesExpiredLeaseAndDenialPoisonsContext) {
  auto failure = LOON_SOURCE_CREDENTIAL_OK;
  provider->fetch = [&](const auto& request) {
    ++calls;
    if (failure == LOON_SOURCE_CREDENTIAL_OK)
      return Lease(request);
    SourceCredentialResponse response;
    response.status = failure;
    return response;
  };
  auto context = Context();
  SourceSasTokenPolicy policy(context);
  Send(policy);
  now->store(kNow + 61000);
  failure = LOON_SOURCE_CREDENTIAL_TRANSIENT;
  Send(policy);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(transport->requests.size(), 2);
  now->store(kNow + 120000);
  ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_TRANSIENT);
  EXPECT_EQ(calls, 6);
  EXPECT_EQ(transport->requests.size(), 2);
  failure = LOON_SOURCE_CREDENTIAL_OK;
  Send(policy);
  now->fetch_add(61000);
  failure = LOON_SOURCE_CREDENTIAL_DENIED;
  ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_DENIED);
  const auto count = calls.load();
  failure = LOON_SOURCE_CREDENTIAL_TRANSIENT;
  ExpectFailure([&] { Send(policy, kList); }, LOON_SOURCE_CREDENTIAL_DENIED);
  EXPECT_EQ(calls, count);
  EXPECT_EQ(transport->requests.size(), 3);
}

TEST_F(SourceSasTokenPolicyTest, CachedLeaseCannotBypassCancellationOrOperationDeadline) {
  auto context = Context();
  SourceSasTokenPolicy policy(context);
  Send(policy);
  context->Cancel();
  ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(transport->requests.size(), 1);
  // The issuer must cap its signed lease to the fixed deadline.
  auto limited = Context(kNow + 120000);
  SourceSasTokenPolicy deadline_policy(limited);
  Send(deadline_policy);
  now->store(kNow + 120000);
  ExpectFailure([&] { Send(deadline_policy); }, LOON_SOURCE_CREDENTIAL_EXPIRED);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(transport->requests.size(), 2);
}

TEST_F(SourceSasTokenPolicyTest, RejectsOutOfScopeAndInjectedOperationsBeforeStorageDispatch) {
  SourceSasTokenPolicy policy(Context());
  for (const auto* uri : {"https://acct.blob.core.windows.net/tables/team/table-other/file",
                          "https://other.blob.core.windows.net/tables/team/table/file",
                          "https://acct.dfs.core.windows.net/tables/?action=getAccessControl",
                          "https://acct.blob.core.windows.net/tables/team/table/file?sig=untrusted"}) {
    ExpectFailure([&] { Send(policy, uri); }, LOON_SOURCE_CREDENTIAL_DENIED);
  }
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(transport->requests.empty());
  for (const auto* injected : {"comp=list", "prefix=other/", "directory=other", "restype=container"}) {
    provider->fetch = [&](const auto& request) {
      ++calls;
      auto result = Lease(request);
      result.sas_token += "&" + std::string(injected);
      return result;
    };
    SourceSasTokenPolicy invalid(Context());
    ExpectFailure([&] { Send(invalid); }, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  }
  EXPECT_EQ(calls, 4);
  EXPECT_TRUE(transport->requests.empty());
}

TEST_F(SourceSasTokenPolicyTest, ConcurrentClonesRetainContextAndOnlyAcquireOneLease) {
  auto context = Context();
  SourceSasTokenPolicy policy(context);
  std::weak_ptr<SourceReadContext> retained = context;
  context.reset();
  std::vector<std::future<void>> workers;
  for (int i = 0; i < 12; ++i) workers.emplace_back(std::async(std::launch::async, [&] { Send(policy); }));
  for (auto& worker : workers) worker.get();
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(transport->requests.size(), 12);
}

TEST_F(SourceSasTokenPolicyTest, ConcurrentWaitersShareTypedFailureAndIndependentRequestsRecover) {
  for (const bool expired_cache : {false, true}) {
    for (const auto failure :
         {LOON_SOURCE_CREDENTIAL_TRANSIENT, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_CANCELLED,
          LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE, LOON_SOURCE_CREDENTIAL_DENIED}) {
      SCOPED_TRACE(expired_cache);
      SCOPED_TRACE(failure);
      now->store(kNow);
      calls.store(0);
      transport = std::make_shared<Transport>();
      std::atomic<bool> failing{false};
      std::promise<void> entered, release;
      auto released = release.get_future().share();
      std::atomic<int> failed_calls{0};
      provider->fetch = [&](const auto& request) {
        ++calls;
        if (!failing.load())
          return Lease(request);
        if (++failed_calls == 1) {
          entered.set_value();
          released.wait();
        }
        SourceCredentialResponse response;
        response.status = failure;
        return response;
      };
      std::atomic<bool> observe_waiters{false};
      std::mutex arrival_mutex;
      std::condition_variable arrived;
      std::map<std::thread::id, int> clock_calls;
      auto context_result = SourceReadContext::Create({"acct", {kRoot}, kNow + 3600000, 3000}, provider, [&] {
        if (observe_waiters.load()) {
          std::lock_guard lock(arrival_mutex);
          ++clock_calls[std::this_thread::get_id()];
          arrived.notify_all();
        }
        return now->load();
      });
      ASSERT_TRUE(context_result.ok());
      SourceSasTokenPolicy policy(*context_result);
      if (expired_cache) {
        Send(policy);
        now->fetch_add(120000);
      }
      failing.store(true);
      auto leader = std::async(std::launch::async, [&] { ExpectFailure([&] { Send(policy); }, failure); });
      entered.get_future().wait();
      observe_waiters.store(true);
      std::vector<std::future<void>> waiters;
      for (int i = 0; i < 7; ++i)
        waiters.emplace_back(std::async(std::launch::async, [&] { ExpectFailure([&] { Send(policy); }, failure); }));
      bool all_waiting;
      {
        std::unique_lock lock(arrival_mutex);
        // Repeated deadline checks prove each caller reached the wait loop
        // while the first provider callback is still gated.
        all_waiting = arrived.wait_for(lock, std::chrono::seconds(5), [&] {
          return clock_calls.size() == waiters.size() &&
                 std::all_of(clock_calls.begin(), clock_calls.end(),
                             [](const auto& entry) { return entry.second >= 8; });
        });
      }
      observe_waiters.store(false);
      release.set_value();
      leader.get();
      for (auto& waiter : waiters) waiter.get();
      EXPECT_TRUE(all_waiting);
      const auto expected = failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1;
      EXPECT_EQ(failed_calls.load(), expected);
      EXPECT_EQ(transport->requests.size(), expired_cache ? 1 : 0);
      failing.store(false);
      if (failure == LOON_SOURCE_CREDENTIAL_DENIED) {
        ExpectFailure([&] { Send(policy); }, failure);
        EXPECT_EQ(calls.load(), expected + (expired_cache ? 1 : 0));
      } else {
        Send(policy);
        EXPECT_EQ(calls.load(), expected + (expired_cache ? 1 : 0) + 1);
      }
    }
  }
}

TEST_F(SourceSasTokenPolicyTest, CancelledWaiterDoesNotWaitForAnotherBlockedProvider) {
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  provider->fetch = [&](const auto& request) {
    ++calls;
    entered.set_value();
    released.wait();
    return Lease(request);
  };
  auto context = Context();
  SourceSasTokenPolicy policy(context);
  auto first =
      std::async(std::launch::async, [&] { ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_CANCELLED); });
  entered.get_future().wait();
  auto second =
      std::async(std::launch::async, [&] { ExpectFailure([&] { Send(policy); }, LOON_SOURCE_CREDENTIAL_CANCELLED); });
  context->Cancel();
  const auto finished = second.wait_for(std::chrono::seconds(1));
  release.set_value();
  first.get();
  second.get();
  EXPECT_EQ(finished, std::future_status::ready);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(transport->requests.empty());
}

TEST_F(SourceSasTokenPolicyTest, SdkCallerCancellationDoesNotCancelOtherReaders) {
  for (const bool cancel_leader : {false, true}) {
    SCOPED_TRACE(cancel_leader);
    calls.store(0);
    transport = std::make_shared<Transport>();
    std::promise<void> entered, release, waiting;
    auto released = release.get_future().share();
    provider->fetch = [&](const auto& request) {
      if (++calls == 1) {
        entered.set_value();
        released.wait();
      }
      return Lease(request);
    };
    std::atomic<bool> observe_waiter{false};
    std::atomic<int> clock_calls{0};
    auto context_result = SourceReadContext::Create({"acct", {kRoot}, kNow + 3600000, 3000}, provider, [&] {
      if (observe_waiter.load() && ++clock_calls == 8)
        waiting.set_value();
      return now->load();
    });
    ASSERT_TRUE(context_result.ok());
    auto source = *context_result;
    SourceSasTokenPolicy policy(source);
    Azure::Core::Context leader_context, waiter_context;
    auto leader = std::async(std::launch::async, [&] {
      if (cancel_leader)
        ExpectFailure([&] { Send(policy, kFile, leader_context); }, LOON_SOURCE_CREDENTIAL_CANCELLED);
      else
        Send(policy, kFile, leader_context);
    });
    entered.get_future().wait();
    observe_waiter.store(true);
    auto waiter = std::async(std::launch::async, [&] {
      if (cancel_leader)
        Send(policy, kFile, waiter_context);
      else
        ExpectFailure([&] { Send(policy, kFile, waiter_context); }, LOON_SOURCE_CREDENTIAL_CANCELLED);
    });
    const auto arrived = waiting.get_future().wait_for(std::chrono::seconds(5));
    observe_waiter.store(false);
    if (cancel_leader) {
      leader_context.Cancel();
    } else {
      waiter_context.Cancel();
      EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    }
    release.set_value();
    leader.get();
    waiter.get();
    EXPECT_EQ(arrived, std::future_status::ready);
    EXPECT_EQ(source->CheckState(), LOON_SOURCE_CREDENTIAL_OK);
    EXPECT_EQ(calls.load(), cancel_leader ? 2 : 1);
    EXPECT_EQ(transport->requests.size(), 1);
    Send(policy);
    EXPECT_EQ(calls.load(), cancel_leader ? 2 : 1);
  }
}

TEST_F(SourceSasTokenPolicyTest, ValidCachedReadDoesNotWaitForAnotherRenewal) {
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  provider->fetch = [&](const auto& request) {
    if (++calls == 2) {
      entered.set_value();
      released.wait();
    }
    return Lease(request);
  };
  SourceSasTokenPolicy policy(Context());
  Send(policy);
  now->fetch_add(61000);
  auto leader = std::async(std::launch::async, [&] { Send(policy); });
  entered.get_future().wait();
  auto cached_reader = std::async(std::launch::async, [&] { Send(policy); });
  const auto finished = cached_reader.wait_for(std::chrono::seconds(1));
  release.set_value();
  leader.get();
  cached_reader.get();
  EXPECT_EQ(finished, std::future_status::ready);
  EXPECT_EQ(calls.load(), 2);
  EXPECT_EQ(transport->requests.size(), 3);
}

TEST_F(SourceSasTokenPolicyTest, NativeFileInfoUsesDirectoryCredentialWithoutRootAclProbe) {
  transport->respond = [](size_t) { return Azure::Core::Http::HttpStatusCode::NotFound; };
  auto fs = FileSystem(Context());
  ASSERT_TRUE(fs);
  const auto info = fs->GetFileInfo("tables/team/table/data.parquet");
  ASSERT_TRUE(info.ok()) << info.status();
  EXPECT_EQ(info->type(), arrow::fs::FileType::NotFound);
  EXPECT_EQ(calls, 1);
  ASSERT_EQ(transport->urls.size(), 1);
  EXPECT_TRUE(transport->urls[0].starts_with(kFile));
  EXPECT_FALSE(transport->requests[0].contains("action"));
}

TEST_F(SourceSasTokenPolicyTest, NativeMetadataListAndRetainedReadsPreserveAllProviderFailures) {
  for (const auto failure :
       {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
        LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    for (const int operation : {0, 1, 2}) {
      SCOPED_TRACE(failure);
      SCOPED_TRACE(operation);
      calls.store(0);
      provider->fetch = [&](const auto& request) {
        ++calls;
        EXPECT_EQ(request.operation, operation == 1 ? LOON_SOURCE_LIST : LOON_SOURCE_READ);
        SourceCredentialResponse response;
        response.status = failure;
        return response;
      };
      auto fs = FileSystem(Context());
      ASSERT_TRUE(fs);
      if (operation == 0) {
        ExpectStatus(fs->GetFileInfo("tables/team/table/data.parquet").status(), failure);
      } else if (operation == 1) {
        arrow::fs::FileSelector selector;
        selector.base_dir = "tables/team/table";
        selector.recursive = true;
        ExpectStatus(fs->GetFileInfo(selector).status(), failure);
      } else {
        arrow::fs::FileInfo info("tables/team/table/data.parquet", arrow::fs::FileType::File);
        info.set_size(16);
        auto file = fs->OpenInputFile(info);
        ASSERT_TRUE(file.ok()) << file.status();
        fs.reset();  // The returned file and SDK policy retain the context.
        char bytes[4];
        ExpectStatus((*file)->ReadAt(0, sizeof(bytes), bytes).status(), failure);
      }
      EXPECT_EQ(calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
      EXPECT_TRUE(transport->requests.empty());
    }
  }
}

TEST_F(SourceSasTokenPolicyTest, NativeScopeAndCancellationGuardKnownSizeAndZeroLengthFastPaths) {
  auto context = Context();
  auto fs = FileSystem(context);
  ASSERT_TRUE(fs);
  for (const auto* path : {"tables", "tables/team/table-other/file", "tables/team/table/../other/file"}) {
    ExpectStatus(fs->GetFileInfo(path).status(), LOON_SOURCE_CREDENTIAL_DENIED);
    arrow::fs::FileInfo info(path, arrow::fs::FileType::File);
    info.set_size(16);
    ExpectStatus(fs->OpenInputFile(info).status(), LOON_SOURCE_CREDENTIAL_DENIED);
  }
  ExpectStatus(fs->CreateDir("tables/team/table/created", true), LOON_SOURCE_CREDENTIAL_DENIED);
  ExpectStatus(fs->OpenOutputStream("tables/team/table/write").status(), LOON_SOURCE_CREDENTIAL_DENIED);
  arrow::fs::FileInfo info("tables/team/table/data.parquet", arrow::fs::FileType::File);
  info.set_size(16);
  auto file = fs->OpenInputFile(info);
  ASSERT_TRUE(file.ok());
  context->Cancel();
  ExpectStatus((*file)->GetSize().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  ExpectStatus((*file)->ReadAt(0, 0).status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(transport->requests.empty());
}

TEST_F(SourceSasTokenPolicyTest, NativeFilesystemEqualityIncludesRuntimeContext) {
  auto context = Context();
  auto first = FileSystem(context);
  auto same = FileSystem(context);
  auto other = FileSystem(Context());
  ASSERT_TRUE(first);
  ASSERT_TRUE(same);
  ASSERT_TRUE(other);
  EXPECT_TRUE(first->Equals(*same));
  EXPECT_FALSE(first->Equals(*other));
}

TEST_F(SourceSasTokenPolicyTest, NativeRetainedReaderRenewsAcrossReadsAndReturnsActualBytes) {
  transport->range_body = true;
  transport->respond = [](size_t) { return Azure::Core::Http::HttpStatusCode::PartialContent; };
  provider->fetch = [&](const auto& request) { return Lease(request, "lease-" + std::to_string(++calls)); };
  auto context = Context();
  auto fs = FileSystem(context);
  ASSERT_TRUE(fs);
  std::weak_ptr<SourceReadContext> retained = context;
  arrow::fs::FileInfo info("tables/team/table/data.parquet", arrow::fs::FileType::File);
  info.set_size(8);
  auto file = fs->OpenInputFile(info);
  ASSERT_TRUE(file.ok());
  fs.reset();
  context.reset();
  for (int round = 0; round < 3; ++round) {
    char bytes[8];
    const auto result = (*file)->ReadAt(0, sizeof(bytes), bytes);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, 8);
    EXPECT_EQ(std::string(bytes, sizeof(bytes)), "abcdefgh");
    EXPECT_EQ(transport->requests.back().at("sig"), "lease-" + std::to_string(round + 1));
    now->fetch_add(121000);
  }
  EXPECT_EQ(calls, 3);
  EXPECT_FALSE(retained.expired());
  ASSERT_TRUE((*file)->Close().ok());
  file = std::shared_ptr<arrow::io::RandomAccessFile>();
  EXPECT_TRUE(retained.expired());
}

}  // namespace
}  // namespace milvus_storage::fs
