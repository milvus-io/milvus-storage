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

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <azure/core/datetime.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/http/transport.hpp>
#include <azure/core/io/body_stream.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_broker_credential_provider.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

namespace milvus_storage {

namespace {

using TimePoint = GcpBrokerCredentialProvider::TimePoint;
using Header = std::optional<std::pair<std::string, std::string>>;

std::string SuccessResponse(const std::string& token, TimePoint expiration) {
  const nlohmann::json response = {
      {"success", true},
      {"credentials",
       {
           {"tempAk", ""},
           {"tempSk", ""},
           {"sessionToken", token},
           {"expiredAt", Azure::DateTime(expiration).ToString(Azure::DateTime::DateFormat::Rfc3339)},
       }},
  };
  return response.dump();
}

class MockTransport final : public Azure::Core::Http::HttpTransport {
  public:
  struct RequestRecord {
    std::string method;
    std::string url;
    std::string body;
  };

  using Headers = std::map<std::string, std::string>;

  void Enqueue(Azure::Core::Http::HttpStatusCode status, std::string body, Headers headers = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.push_back(QueuedResponse{status, std::move(body), std::move(headers)});
  }

  // Called with the zero-based request index after the request is recorded
  // and before the response is produced, outside the transport lock. Tests use
  // it to hold a Broker call open while checking what other callers do.
  void SetOnSend(std::function<void(size_t)> on_send) {
    std::lock_guard<std::mutex> lock(mutex_);
    on_send_ = std::move(on_send);
  }

  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       const Azure::Core::Context& context) override {
    std::string body;
    if (auto* stream = request.GetBodyStream(); stream != nullptr) {
      stream->Rewind();
      const auto bytes = stream->ReadToEnd(context);
      body.assign(bytes.begin(), bytes.end());
    }

    size_t index = 0;
    std::function<void(size_t)> on_send;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      index = requests_.size();
      requests_.push_back(RequestRecord{request.GetMethod().ToString(), request.GetUrl().GetAbsoluteUrl(), body});
      on_send = on_send_;
    }
    if (on_send) {
      on_send(index);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (responses_.empty()) {
      throw Azure::Core::Http::TransportException("no queued response");
    }
    auto queued = std::move(responses_.front());
    responses_.pop_front();
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(1, 1, queued.status, "mock");
    for (const auto& [name, value] : queued.headers) {
      response->SetHeader(name, value);
    }
    bodies_.emplace_back(queued.body.begin(), queued.body.end());
    response->SetBodyStream(std::make_unique<Azure::Core::IO::MemoryBodyStream>(bodies_.back()));
    return response;
  }

  std::vector<RequestRecord> Requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }

  size_t RequestCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_.size();
  }

  private:
  struct QueuedResponse {
    Azure::Core::Http::HttpStatusCode status;
    std::string body;
    Headers headers;
  };

  mutable std::mutex mutex_;
  std::deque<QueuedResponse> responses_;
  // Deque keeps references to earlier bodies valid while new ones are added;
  // the response's MemoryBodyStream reads them after Send returns.
  std::deque<std::vector<uint8_t>> bodies_;
  std::vector<RequestRecord> requests_;
  std::function<void(size_t)> on_send_;
};

GcpBrokerConfig MakeConfig() {
  return GcpBrokerConfig{
      .endpoint = "http://credential-broker/v1/credentials/assume-role",
      .region = "us-central1",
      .bucket = "bucket",
      .target_service_account = "reader@project.iam.gserviceaccount.com",
      .duration_seconds = 3600,
      .request_timeout_ms = 1000,
  };
}

ArrowFileSystemConfig MakeGcpConfig() {
  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderGCP;
  config.address = "storage.googleapis.com";
  config.bucket_name = "bucket";
  config.load_frequency = 900;
  return config;
}

// Polls until `done` returns true or the real-clock timeout elapses.
bool WaitUntil(const std::function<bool()>& done, std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

}  // namespace

TEST(GcpBrokerCredentialProviderTest, SendsBrokerContractAndReturnsBearer) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("ya29.token", now + std::chrono::hours(1)));
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });

  auto header = provider.AuthorizationHeader();
  ASSERT_TRUE(header.has_value());
  EXPECT_EQ(header->first, "Authorization");
  EXPECT_EQ(header->second, "Bearer ya29.token");

  const auto requests = transport->Requests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].method, "POST");
  EXPECT_EQ(requests[0].url, "http://credential-broker/v1/credentials/assume-role");
  const auto body = nlohmann::json::parse(requests[0].body);
  EXPECT_EQ(body["csp"].get<std::string>(), "gcp");
  EXPECT_EQ(body["region"].get<std::string>(), "us-central1");
  EXPECT_EQ(body["bucket"].get<std::string>(), "bucket");
  EXPECT_EQ(body["durationSeconds"].get<int32_t>(), 3600);
  EXPECT_EQ(body["gcpServiceAccountEmail"].get<std::string>(), "reader@project.iam.gserviceaccount.com");
}

TEST(GcpBrokerCredentialProviderTest, CachesTokenUntilSixtySecondsBeforeExpiry) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(10)));
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("second", now + std::chrono::hours(1)));
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [&now] { return now; });

  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  now += std::chrono::minutes(8);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Inside the refresh window the caller that refreshes gets the new token.
  now += std::chrono::seconds(90);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST(GcpBrokerCredentialProviderTest, FailedRefreshKeepsValidTokenWithSingleAttempt) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(2)));
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "{}");
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [&now] { return now; });

  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Refresh fails with a single attempt; the still-valid token is kept.
  now += std::chrono::seconds(90);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 2u);

  // Right after the failure the Broker is left alone; the token still works.
  now += std::chrono::seconds(1);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 2u);

  // Once the backoff is over another single attempt is made (the transport
  // has nothing queued, so it throws) and the valid token is still served.
  now += std::chrono::seconds(2);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 3u);

  // Once the token has expired it is no fallback: the request goes out
  // without Authorization instead, after one more single attempt.
  now += std::chrono::minutes(5);
  EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  EXPECT_EQ(transport->RequestCount(), 4u);
}

TEST(GcpBrokerCredentialProviderTest, InitialFailureRetriesThenReturnsNoHeader) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  for (int i = 0; i < 4; ++i) {
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "{}");
  }
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });

  EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  EXPECT_EQ(transport->RequestCount(), 4u);
}

TEST(GcpBrokerCredentialProviderTest, NegativeCacheSuppressesFetchesAfterFailure) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  for (int i = 0; i < 4; ++i) {
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "{}");
  }
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [&now] { return now; });

  EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  EXPECT_EQ(transport->RequestCount(), 4u);

  // Within the backoff window the failure is returned without a Broker call.
  now += std::chrono::milliseconds(1500);
  EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  EXPECT_EQ(transport->RequestCount(), 4u);

  // After the window the Broker is asked again and a recovery is picked up.
  now += std::chrono::milliseconds(1000);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("recovered", now + std::chrono::hours(1)));
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer recovered");
  EXPECT_EQ(transport->RequestCount(), 5u);
}

TEST(GcpBrokerCredentialProviderTest, ConcurrentInitialRequestsShareFetchRetries) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "unavailable");
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("shared", now + std::chrono::hours(1)));
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });

  // Hold the first Broker call open until every caller has been started, so
  // they all arrive while the fetch is in flight.
  std::promise<void> all_started;
  auto started = all_started.get_future().share();
  transport->SetOnSend([started](size_t index) {
    if (index == 0) {
      started.wait();
    }
  });

  constexpr size_t kCallers = 8;
  std::vector<Header> headers(kCallers);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kCallers; ++i) {
    threads.emplace_back([&provider, &headers, i] { headers[i] = provider.AuthorizationHeader(); });
  }
  all_started.set_value();
  for (auto& thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < kCallers; ++i) {
    ASSERT_TRUE(headers[i].has_value()) << "caller " << i;
    EXPECT_EQ(headers[i]->second, "Bearer shared") << "caller " << i;
  }
  // One failure plus one successful retry, shared by all callers.
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST(GcpBrokerCredentialProviderTest, ConcurrentCallersGetStaleTokenWhileOneRefreshes) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(2)));
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [&now] { return now; });
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  ASSERT_EQ(transport->RequestCount(), 1u);

  // 30 s left: valid, but inside the refresh window.
  now += std::chrono::seconds(90);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("second", now + std::chrono::hours(1)));
  std::promise<void> release_refresh;
  auto released = release_refresh.get_future().share();
  transport->SetOnSend([released](size_t index) {
    if (index == 1) {
      released.wait();
    }
  });

  Header refresher_header;
  std::thread refresher([&] { refresher_header = provider.AuthorizationHeader(); });
  ASSERT_TRUE(WaitUntil([&] { return transport->RequestCount() == 2; }));

  // While the refresh is stuck in the Broker call, other callers must get the
  // cached token right away instead of waiting or fetching.
  constexpr size_t kCallers = 8;
  std::vector<Header> headers(kCallers);
  std::atomic<size_t> finished{0};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kCallers; ++i) {
    threads.emplace_back([&, i] {
      headers[i] = provider.AuthorizationHeader();
      finished.fetch_add(1);
    });
  }
  const bool callers_returned = WaitUntil([&] { return finished.load() == kCallers; });
  release_refresh.set_value();
  for (auto& thread : threads) {
    thread.join();
  }
  refresher.join();

  EXPECT_TRUE(callers_returned) << "callers blocked behind the in-flight refresh";
  for (size_t i = 0; i < kCallers; ++i) {
    ASSERT_TRUE(headers[i].has_value()) << "caller " << i;
    EXPECT_EQ(headers[i]->second, "Bearer first") << "caller " << i;
  }
  ASSERT_TRUE(refresher_header.has_value());
  EXPECT_EQ(refresher_header->second, "Bearer second");
  // Exactly one Broker call for the whole refresh.
  EXPECT_EQ(transport->RequestCount(), 2u);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST(GcpBrokerCredentialProviderTest, LifetimeIsCappedAtRequestedDuration) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto config = MakeConfig();
  config.duration_seconds = 120;
  auto transport = std::make_shared<MockTransport>();
  // The Broker claims one hour; the provider asked for two minutes.
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::hours(1)));
  GcpBrokerCredentialProvider provider(config, transport, [&now] { return now; });

  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  now += std::chrono::seconds(30);
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Past the requested duration the token is treated as expired even though
  // the Broker's expiredAt is still far away.
  now += std::chrono::seconds(91);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("second", now + std::chrono::hours(1)));
  EXPECT_EQ(provider.AuthorizationHeader()->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST(GcpBrokerCredentialProviderTest, OversizedResponseIsRejected) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  constexpr size_t kLimit = 64 * 1024;

  {
    // Body larger than the limit, without a Content-Length header.
    auto transport = std::make_shared<MockTransport>();
    const auto body = SuccessResponse(std::string(kLimit + 1024, 'a'), now + std::chrono::hours(1));
    ASSERT_GT(body.size(), kLimit);
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, body);
    }
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(provider.AuthorizationHeader().has_value());
    EXPECT_EQ(transport->RequestCount(), 4u);
  }
  {
    // Small body, but Content-Length announces more than the limit.
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("small", now + std::chrono::hours(1)),
                         {{"Content-Length", std::to_string(kLimit + 1)}});
    }
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  }
  {
    // Unparsable Content-Length is rejected as well.
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("small", now + std::chrono::hours(1)),
                         {{"Content-Length", "-1"}});
    }
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(provider.AuthorizationHeader().has_value());
  }
  {
    // A large token under the limit with a matching Content-Length passes.
    auto transport = std::make_shared<MockTransport>();
    const auto token = std::string(60 * 1024, 'b');
    const auto body = SuccessResponse(token, now + std::chrono::hours(1));
    ASSERT_LE(body.size(), kLimit);
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, body, {{"Content-Length", std::to_string(body.size())}});
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    auto header = provider.AuthorizationHeader();
    ASSERT_TRUE(header.has_value());
    EXPECT_EQ(header->second, "Bearer " + token);
  }
}

TEST(GcpBrokerCredentialProviderTest, TokenWithDisallowedCharactersIsRejected) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  const std::vector<std::string> bad_tokens = {
      "ya29.tok en", "token\r\nX-Injected: 1", "to=ken", "\"token\"", "token;", "=abc", "tok\xC3\xA9n",
  };
  for (const auto& token : bad_tokens) {
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse(token, now + std::chrono::hours(1)));
    }
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(provider.AuthorizationHeader().has_value()) << token;
  }

  // The full RFC 6750 b64token alphabet with trailing padding is accepted.
  auto transport = std::make_shared<MockTransport>();
  const std::string good_token = "ya29.a0Af-H_~+/AZ09==";
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse(good_token, now + std::chrono::hours(1)));
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
  auto header = provider.AuthorizationHeader();
  ASSERT_TRUE(header.has_value());
  EXPECT_EQ(header->second, "Bearer " + good_token);
}

TEST(GcpBrokerCredentialProviderTest, InvalidResponsesReturnNoHeader) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  const std::vector<std::string> invalid_bodies = {
      "not json",
      R"({"success": false, "msg": "denied"})",
      SuccessResponse("", now + std::chrono::hours(1)),
      SuccessResponse("expired", now - std::chrono::seconds(1)),
      R"({"success": true, "credentials": {"sessionToken": "token", "expiredAt": "not a time"}})",
  };
  for (const auto& body : invalid_bodies) {
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, body);
    }
    GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(provider.AuthorizationHeader().has_value()) << body;
  }
}

TEST(GcpBrokerCredentialProviderTest, TransportExceptionDoesNotEscape) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  // No queued response: every Send throws TransportException.
  auto transport = std::make_shared<MockTransport>();
  GcpBrokerCredentialProvider provider(MakeConfig(), transport, [now] { return now; });
  EXPECT_FALSE(provider.AuthorizationHeader().has_value());
}

TEST(GcpCredentialProviderSelectionTest, CredentialEndpointSelectsBrokerWithoutUseIam) {
  auto config = MakeGcpConfig();
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  config.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";

  auto broker = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(broker.ok()) << broker.status().ToString();
  EXPECT_NE(dynamic_cast<GcpBrokerCredentialProvider*>(broker->get()), nullptr);
  EXPECT_TRUE(broker.ValueOrDie()->IssuesBearerToken());
  EXPECT_EQ(broker.ValueOrDie()->Identity(),
            "broker:http://credential-broker/v1/credentials/assume-role|reader@project.iam.gserviceaccount.com");

  // use_iam selects this process's identity, which then impersonates the
  // target directly.
  config.use_iam = true;
  auto impersonation = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(impersonation.ok()) << impersonation.status().ToString();
  EXPECT_EQ(dynamic_cast<GcpBrokerCredentialProvider*>(impersonation->get()), nullptr);
  EXPECT_EQ(impersonation.ValueOrDie()->Identity(), "impersonate:reader@project.iam.gserviceaccount.com");

  auto missing_target = MakeGcpConfig();
  missing_target.credential_endpoint = config.credential_endpoint;
  EXPECT_TRUE(BuildGcpProviderFromConfig(missing_target).status().IsInvalid());
}

TEST(GcpCredentialProviderSelectionTest, TargetAloneSelectsImpersonation) {
  auto config = MakeGcpConfig();
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";

  auto provider = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider.ok()) << provider.status().ToString();
  EXPECT_EQ(dynamic_cast<GcpBrokerCredentialProvider*>(provider->get()), nullptr);
  EXPECT_TRUE(provider.ValueOrDie()->IssuesBearerToken());

  config.load_frequency = 7200;
  EXPECT_TRUE(BuildGcpProviderFromConfig(config).status().IsInvalid());
}

TEST(GcpCredentialProviderSelectionTest, HmacWithTargetIsAmbiguousUnlessTalon) {
  auto config = MakeGcpConfig();
  config.access_key_id = "ak";
  config.access_key_value = "sk";
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  EXPECT_TRUE(BuildGcpProviderFromConfig(config).status().IsInvalid());

  config.gcp_target_service_account.clear();
  config.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";
  EXPECT_TRUE(BuildGcpProviderFromConfig(config).status().IsInvalid());

  // With Talon the Broker fields describe Talon's identity; the origin keeps HMAC.
  config.talon_mode = TalonMode::Full;
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  auto provider = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider.ok()) << provider.status().ToString();
  EXPECT_FALSE(provider.ValueOrDie()->IssuesBearerToken());
  EXPECT_EQ(provider.ValueOrDie()->Identity(), "hmac:ak");
}

TEST(GcpCredentialProviderSelectionTest, HmacIsNotBearer) {
  auto config = MakeGcpConfig();
  config.access_key_id = "ak";
  config.access_key_value = "sk";
  auto provider = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider.ok()) << provider.status().ToString();
  EXPECT_FALSE(provider.ValueOrDie()->AuthorizationHeader().has_value());
  EXPECT_FALSE(provider.ValueOrDie()->IssuesBearerToken());
  EXPECT_EQ(provider.ValueOrDie()->Identity(), "hmac:ak");
}

}  // namespace milvus_storage
