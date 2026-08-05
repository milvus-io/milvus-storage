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

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <azure/core/datetime.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/http/policies/policy.hpp>
#include <azure/core/internal/client_options.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/io/body_stream.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/filesystem/azure/azurefs_internal.h"
#include "milvus-storage/filesystem/azure/azure_sas_token_policy.h"

namespace milvus_storage::fs {

using TimePoint = AzureSasTokenPolicy::TimePoint;

static std::string FormatTime(TimePoint time) {
  return Azure::DateTime(time).ToString(Azure::DateTime::DateFormat::Rfc3339);
}

static std::string SuccessResponse(const std::string& account, const std::string& token, TimePoint expiration) {
  const nlohmann::json response = {
      {"success", true},
      {"credentials",
       {
           {"tempAk", account},
           {"tempSk", ""},
           {"sessionToken", token},
           {"expiredAt", FormatTime(expiration)},
       }},
  };
  return response.dump();
}

class MockTransport final : public Azure::Core::Http::HttpTransport {
  public:
  struct RequestRecord {
    std::string method;
    std::string url;
    Azure::Core::CaseInsensitiveMap headers;
    std::string body;
  };

  void Enqueue(Azure::Core::Http::HttpStatusCode status, std::string body) {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.emplace_back(status, std::move(body));
  }

  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       const Azure::Core::Context& context) override {
    auto* stream = request.GetBodyStream();
    std::string body;
    if (stream != nullptr) {
      stream->Rewind();
      const auto bytes = stream->ReadToEnd(context);
      body.assign(bytes.begin(), bytes.end());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    requests_.push_back(RequestRecord{request.GetMethod().ToString(), request.GetUrl().GetAbsoluteUrl(),
                                      request.GetHeaders(), std::move(body)});
    if (responses_.empty()) {
      throw Azure::Core::Http::TransportException("no queued response");
    }
    auto [status, response_body] = std::move(responses_.front());
    responses_.pop_front();
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(1, 1, status, "mock");
    // Match CurlTransport: the response payload remains in BodyStream when
    // HttpTransport::Send is called directly.
    streamed_response_bodies_.emplace_back(response_body.begin(), response_body.end());
    response->SetBodyStream(std::make_unique<Azure::Core::IO::MemoryBodyStream>(streamed_response_bodies_.back()));
    return response;
  }

  std::vector<RequestRecord> Requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }

  private:
  mutable std::mutex mutex_;
  std::deque<std::pair<Azure::Core::Http::HttpStatusCode, std::string>> responses_;
  std::deque<std::vector<uint8_t>> streamed_response_bodies_;
  std::vector<RequestRecord> requests_;
};

static AzureSasBrokerConfig MakeConfig() {
  return AzureSasBrokerConfig{
      .endpoint = "http://credential-broker/v1/credentials/assume-role",
      .region = "westus3",
      .bucket = "container",
      .client_id = "client-id",
      .tenant_id = "tenant-id",
      .account_name = "account",
      .duration_seconds = 3600,
      .request_timeout_ms = 1000,
  };
}

class CapturePolicy final : public Azure::Core::Http::Policies::HttpPolicy {
  public:
  CapturePolicy(bool* called, std::map<std::string, std::string>* query) : called_(called), query_(query) {}

  std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy> Clone() const override {
    return std::make_unique<CapturePolicy>(called_, query_);
  }

  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       Azure::Core::Http::Policies::NextHttpPolicy,
                                                       const Azure::Core::Context&) const override {
    *called_ = true;
    *query_ = request.GetUrl().GetQueryParameters();
    return std::make_unique<Azure::Core::Http::RawResponse>(1, 1, Azure::Core::Http::HttpStatusCode::Ok, "OK");
  }

  private:
  bool* called_;
  std::map<std::string, std::string>* query_;
};

static std::unique_ptr<Azure::Core::Http::RawResponse> SendRequest(
    Azure::Core::Http::Policies::HttpPolicy& policy,
    bool* called,
    std::map<std::string, std::string>* query,
    const std::string& url = "https://account.blob.core.windows.net/container/file") {
  std::vector<std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy>> next_policies;
  // NextHttpPolicy stores the current policy index and dispatches index + 1.
  next_policies.emplace_back(policy.Clone());
  next_policies.emplace_back(std::make_unique<CapturePolicy>(called, query));
  Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Get, Azure::Core::Url(url));
  return policy.Send(request, Azure::Core::Http::Policies::NextHttpPolicy(0, next_policies), Azure::Core::Context());
}

TEST(AzureSasTokenPolicyTest, SendsBrokerContractAndNormalizesToken) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(
      Azure::Core::Http::HttpStatusCode::Ok,
      SuccessResponse("account", "?sv=2026-01-01&sig=encoded%2Bsignature%3D", now + std::chrono::hours(1)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto response = SendRequest(policy, &called, &query);
  EXPECT_EQ(response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_TRUE(called);
  EXPECT_EQ(query.at("sv"), "2026-01-01");
  EXPECT_EQ(query.at("sig"), "encoded%2Bsignature%3D");

  const auto requests = transport->Requests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].method, "POST");
  EXPECT_EQ(requests[0].url, "http://credential-broker/v1/credentials/assume-role");
  EXPECT_EQ(requests[0].headers.at("Content-Type"), "application/json");

  const auto body = nlohmann::json::parse(requests[0].body);
  EXPECT_EQ(body["csp"].get<std::string>(), "azure");
  EXPECT_EQ(body["region"].get<std::string>(), "westus3");
  EXPECT_EQ(body["bucket"].get<std::string>(), "container");
  EXPECT_EQ(body["durationSeconds"].get<int32_t>(), 3600);
  EXPECT_EQ(body["azureClientId"].get<std::string>(), "client-id");
  EXPECT_EQ(body["azureTenantId"].get<std::string>(), "tenant-id");
  EXPECT_EQ(body["azureAccountName"].get<std::string>(), "account");
}

TEST(AzureSasTokenPolicyTest, PerOperationPolicyKeepsSasAcrossSdkRetries) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto broker_transport = std::make_shared<MockTransport>();
  broker_transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                            SuccessResponse("account", "sv=1&sig=retry", now + std::chrono::seconds(30)));
  AzureSasTokenPolicy policy(MakeConfig(), broker_transport, [now] { return now; });

  auto storage_transport = std::make_shared<MockTransport>();
  storage_transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "");
  storage_transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, "");

  Azure::Core::_internal::ClientOptions options;
  options.PerOperationPolicies.emplace_back(policy.Clone());
  options.Retry.MaxRetries = 1;
  options.Retry.RetryDelay = std::chrono::milliseconds(0);
  options.Retry.MaxRetryDelay = std::chrono::milliseconds(0);
  options.Transport.Transport = storage_transport;
  Azure::Core::Http::_internal::HttpPipeline pipeline(options, "test", "1.0.0", {}, {});

  Azure::Core::Http::Request request(
      Azure::Core::Http::HttpMethod::Get,
      Azure::Core::Url("https://account.blob.core.windows.net/container/file?comp=metadata"));
  auto response = pipeline.Send(request, Azure::Core::Context());
  ASSERT_EQ(response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(broker_transport->Requests().size(), 1u);

  const auto storage_requests = storage_transport->Requests();
  ASSERT_EQ(storage_requests.size(), 2u);
  for (const auto& storage_request : storage_requests) {
    const auto query = Azure::Core::Url(storage_request.url).GetQueryParameters();
    EXPECT_EQ(query.at("comp"), "metadata");
    EXPECT_EQ(query.at("sv"), "1");
    EXPECT_EQ(query.at("sig"), "retry");
  }
}

TEST(AzureSasTokenPolicyTest, RefreshFailureReturnsOldTokenAndRetriesEveryRequest) {
  auto now = std::make_shared<TimePoint>(std::chrono::system_clock::from_time_t(1785146400));
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=1&sig=old", *now + std::chrono::seconds(120)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return *now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto initial_response = SendRequest(policy, &called, &query);
  ASSERT_EQ(initial_response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(query.at("sig"), "old");
  *now += std::chrono::seconds(61);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "do not log this body");
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "do not log this body");

  auto first_fallback = SendRequest(policy, &called, &query);
  ASSERT_EQ(first_fallback->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(query.at("sig"), "old");
  auto second_fallback = SendRequest(policy, &called, &query);
  ASSERT_EQ(second_fallback->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(query.at("sig"), "old");
  EXPECT_EQ(transport->Requests().size(), 3u);

  // Past the token's own expiry, the fallback stops. This reverses what this
  // test used to assert, deliberately: an expired SAS is not a usable
  // fallback, because Azure answers it with 401 and that 401 is
  // indistinguishable from a genuinely rejected credential -- so a broker
  // outage came back as Config/never-retry, which is the same misattribution
  // this policy now avoids one hop earlier, just arriving later. Reporting the
  // classified broker failure instead lets the retry succeed once the broker
  // recovers.
  *now += std::chrono::seconds(60);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "expired fallback");
  auto expired_fallback = SendRequest(policy, &called, &query);
  ASSERT_EQ(expired_fallback->GetStatusCode(), Azure::Core::Http::HttpStatusCode::ServiceUnavailable)
      << expired_fallback->GetReasonPhrase();
  EXPECT_EQ(expired_fallback->GetReasonPhrase().find("expired fallback"), std::string::npos)
      << "the broker's response body must not leak into the reason phrase";

  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=2&sig=new", *now + std::chrono::hours(1)));
  auto refreshed = SendRequest(policy, &called, &query);
  ASSERT_EQ(refreshed->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(query.at("sig"), "new");
  EXPECT_EQ(transport->Requests().size(), 5u);
}

TEST(AzureSasTokenPolicyTest, DoesNotRefreshWithMoreThanSixtySecondsRemaining) {
  auto now = std::make_shared<TimePoint>(std::chrono::system_clock::from_time_t(1785146400));
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=1&sig=old", *now + std::chrono::seconds(120)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return *now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto initial_response = SendRequest(policy, &called, &query);
  ASSERT_EQ(initial_response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  *now += std::chrono::seconds(59);
  auto cached = SendRequest(policy, &called, &query);
  ASSERT_EQ(cached->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(query.at("sig"), "old");
  EXPECT_EQ(transport->Requests().size(), 1u);
}

TEST(AzureSasTokenPolicyTest, ClonesShareTokenCache) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=1&sig=shared", now + std::chrono::hours(1)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });
  auto clone = policy.Clone();

  bool original_called = false;
  bool clone_called = false;
  std::map<std::string, std::string> original_query;
  std::map<std::string, std::string> clone_query;
  auto original_response = SendRequest(policy, &original_called, &original_query);
  auto clone_response = SendRequest(*clone, &clone_called, &clone_query);

  EXPECT_EQ(original_response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_EQ(clone_response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_TRUE(original_called);
  EXPECT_TRUE(clone_called);
  EXPECT_EQ(original_query.at("sig"), "shared");
  EXPECT_EQ(clone_query.at("sig"), "shared");
  EXPECT_EQ(transport->Requests().size(), 1u);
}

TEST(AzureSasTokenPolicyTest, RejectsInvalidBrokerResponsesWithoutCachedToken) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  const std::vector<std::pair<std::string, std::string>> invalid_responses = {
      {R"({"success":false})", "returned success=false"},
      {R"({"success":true})", "response schema is invalid"},
      {SuccessResponse("other-account", "sv=1&sig=x", now + std::chrono::hours(1)),
       "returned a token for the wrong account: expected=account, actual=other-account"},
      {SuccessResponse("account", "", now + std::chrono::hours(1)),
       "response field 'credentials.sessionToken' must not be empty"},
      {SuccessResponse("account", "sv=1", now + std::chrono::hours(1)),
       "response field 'credentials.sessionToken' must contain a non-empty 'sig' query parameter"},
      {SuccessResponse("account", "sv=1&sig=", now + std::chrono::hours(1)),
       "response field 'credentials.sessionToken' must contain a non-empty 'sig' query parameter"},
      {R"({"success":true,"credentials":{"tempAk":"account","sessionToken":"sv=1&sig=x","expiredAt":"invalid"}})",
       "returned an invalid expiration time"},
      {SuccessResponse("account", "sv=1&sig=x", now - std::chrono::seconds(1)), "returned an expired token"},
      {R"({not-json})", "returned invalid JSON"},
  };

  for (const auto& [response, expected_error] : invalid_responses) {
    auto transport = std::make_shared<MockTransport>();
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, response);
    AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });
    bool called = false;
    std::map<std::string, std::string> query;
    auto policy_response = SendRequest(policy, &called, &query);
    ASSERT_EQ(policy_response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Unauthorized);
    EXPECT_NE(policy_response->GetReasonPhrase().find(expected_error), std::string::npos)
        << policy_response->GetReasonPhrase();
    EXPECT_FALSE(called);
  }
}

TEST(AzureSasTokenPolicyTest, ConcurrentInitialRequestsOnlyFetchOnce) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=1&sig=shared", now + std::chrono::hours(1)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });

  std::vector<std::thread> threads;
  std::vector<int> status_codes(8, 0);
  std::vector<int> called(8, 0);
  std::vector<std::map<std::string, std::string>> queries(8);
  for (size_t i = 0; i < status_codes.size(); ++i) {
    threads.emplace_back([&policy, &status_codes, &called, &queries, i] {
      bool request_sent = false;
      auto response = SendRequest(policy, &request_sent, &queries[i]);
      status_codes[i] = static_cast<int>(response->GetStatusCode());
      called[i] = request_sent ? 1 : 0;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < status_codes.size(); ++i) {
    EXPECT_EQ(status_codes[i], static_cast<int>(Azure::Core::Http::HttpStatusCode::Ok));
    EXPECT_EQ(called[i], 1);
    EXPECT_EQ(queries[i].at("sig"), "shared");
  }
  EXPECT_EQ(transport->Requests().size(), 1u);
}

TEST(AzureSasTokenPolicyTest, AppendsSasWithoutReplacingOperationQuery) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "?sv=1&sig=encoded%2Bsignature%3D", now + std::chrono::hours(1)));
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto response =
      SendRequest(policy, &called, &query, "https://account.blob.core.windows.net/container/file?comp=metadata");
  EXPECT_EQ(response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::Ok);
  EXPECT_TRUE(called);
  EXPECT_EQ(query.at("comp"), "metadata");
  EXPECT_EQ(query.at("sv"), "1");
  EXPECT_EQ(query.at("sig"), "encoded%2Bsignature%3D");
}

TEST(AzureSasTokenPolicyTest, InitialFetchFailureDoesNotSendAnonymousRequest) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "secret response body");
  AzureSasTokenPolicy policy(MakeConfig(), transport, [now] { return now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto response = SendRequest(policy, &called, &query);
  // 503, not 401. A broker that answered 500 is unwell; it did not reject our
  // credentials. Synthesising 401 here made ClassifyAzureError read it as
  // AccessDenied -- Config, never retried -- so a broker blip looked exactly
  // like a bad key and the one of the two that a retry would have fixed was
  // the one denied a retry. This assertion is the whole point of the split.
  EXPECT_EQ(response->GetStatusCode(), Azure::Core::Http::HttpStatusCode::ServiceUnavailable);
  auto classified = internal::ClassifyAzureError(static_cast<int>(response->GetStatusCode()),
                                                 response->GetHeaders().at("x-ms-error-code"),
                                                 /*transport_failure=*/false);
  ASSERT_TRUE(classified.has_value());
  EXPECT_TRUE(RetryableForExtendStatusCode(*classified));
  EXPECT_NE(response->GetReasonPhrase().find("status_code=500"), std::string::npos);
  EXPECT_EQ(response->GetReasonPhrase().find("secret response body"), std::string::npos);
  EXPECT_FALSE(called);
}

// A cached SAS that is still valid rides out a broker outage. An EXPIRED one
// must not: Azure rejects it with 401, and that rejection reads as "your
// credentials are wrong" for what is actually the broker being down -- the
// same misattribution, arriving one hop later.
TEST(AzureSasTokenPolicyTest, ExpiredCachedTokenIsNotUsedWhenTheBrokerFails) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  // First fetch succeeds with a token that expires almost immediately.
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                     SuccessResponse("account", "sv=1&sig=cached", now + std::chrono::seconds(1)));
  // Second fetch, after it has expired, finds the broker down.
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "broker down");

  auto clock_now = now;
  AzureSasTokenPolicy policy(MakeConfig(), transport, [&clock_now] { return clock_now; });

  bool called = false;
  std::map<std::string, std::string> query;
  auto first = SendRequest(policy, &called, &query);
  ASSERT_TRUE(called) << "the first request should have been signed with the fresh token";

  clock_now = now + std::chrono::hours(1);
  called = false;
  auto second = SendRequest(policy, &called, &query);
  EXPECT_FALSE(called) << "an expired token must not be used to sign a request";
  EXPECT_EQ(second->GetStatusCode(), Azure::Core::Http::HttpStatusCode::ServiceUnavailable)
      << second->GetReasonPhrase();
}

}  // namespace milvus_storage::fs
