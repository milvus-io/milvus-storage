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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <azure/core/datetime.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/http/transport.hpp>
#include <azure/core/io/body_stream.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_broker_credential_provider.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_crt_client.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include <aws/s3/model/HeadObjectRequest.h>
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

namespace milvus_storage {

namespace {

using TimePoint = GcpBrokerCredentialProvider::TimePoint;
using Header = arrow::Result<std::pair<std::string, std::string>>;

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

ArrowFileSystemConfig MakeGcpConfig() {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.address = "storage.googleapis.com";
  config.use_ssl = true;
  config.bucket_name = "bucket";
  config.load_frequency = 900;
  return config;
}

ArrowFileSystemConfig MakeConfig() {
  auto config = MakeGcpConfig();
  config.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";
  config.region = "us-central1";
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  config.load_frequency = 3600;
  config.request_timeout_ms = 1000;
  return config;
}

std::shared_ptr<S3CrtClientHolder> MakeBrokerClient(const ArrowFileSystemConfig& config,
                                                    const std::shared_ptr<MockTransport>& transport,
                                                    GcpBrokerCredentialProvider::Clock clock) {
  S3Options options;
  options.cloud_provider = kCloudProviderGCP;
  options.credentials_provider = std::make_shared<GcpBrokerCredentialProvider>(config, transport, std::move(clock));
  options.endpoint_override = config.address;
  options.region = "auto";
  options.scheme = config.use_ssl ? "https" : "http";
  auto client = ClientBuilder<Aws::S3Crt::S3CrtClient>(options).BuildClient().ValueOrDie();
  return client;
}

Header ReadHeader(const std::shared_ptr<S3CrtClientHolder>& client) {
  Aws::Endpoint::AWSEndpoint endpoint;
  endpoint.SetURI(Aws::Http::URI("https://storage.googleapis.com/bucket/key"));
  Aws::S3::Model::HeadObjectRequest request;
  ARROW_ASSIGN_OR_RAISE(auto lease, client->Acquire());
  const auto authorized = GcpCredentialProvider::AuthorizeRead(
      lease->GetCredentialsProvider(), request, std::move(endpoint), "", Aws::Http::HttpMethod::HTTP_HEAD);
  EXPECT_TRUE(authorized.ok()) << authorized.status();
  ARROW_ASSIGN_OR_RAISE(const bool handled, authorized);
  if (!handled)
    return arrow::Status::Invalid("Expected bearer authorization");
  return std::make_pair(std::string("Authorization"),
                        std::string(request.GetAdditionalCustomHeaders().at("authorization").c_str()));
}

class GcpBrokerCredentialProviderTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    const auto status = S3FileSystemProducer(ArrowFileSystemConfig{}).InitS3();
    ASSERT_TRUE(status.ok()) << status;
  }
};
using GcpCredentialProviderSelectionTest = GcpBrokerCredentialProviderTest;

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

TEST_F(GcpBrokerCredentialProviderTest, SendsBrokerContractAndReturnsBearer) {
  for (const bool use_iam : {false, true}) {
    SCOPED_TRACE(use_iam);
    const auto now = std::chrono::system_clock::from_time_t(1785146400);
    auto transport = std::make_shared<MockTransport>();
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                       SuccessResponse("ya29.token", now + std::chrono::hours(1)));
    auto config = MakeConfig();
    config.use_iam = use_iam;
    auto client = MakeBrokerClient(config, transport, [now] { return now; });

    auto header = ReadHeader(client);
    ASSERT_TRUE(header.ok());
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
    if (use_iam) {
      EXPECT_EQ(body.at("credentialType"), "iam");
      EXPECT_FALSE(body.contains("gcpServiceAccountEmail"));
    } else {
      EXPECT_FALSE(body.contains("credentialType"));
      EXPECT_EQ(body.at("gcpServiceAccountEmail"), "reader@project.iam.gserviceaccount.com");
    }
    EXPECT_EQ(ReadHeader(client)->second, "Bearer ya29.token");
    EXPECT_EQ(transport->RequestCount(), 1u);
  }
}

TEST_F(GcpBrokerCredentialProviderTest, CachesTokenUntilSixtySecondsBeforeExpiry) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(10)));
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("second", now + std::chrono::hours(1)));
  auto client = MakeBrokerClient(MakeConfig(), transport, [&now] { return now; });

  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  now += std::chrono::minutes(8);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Inside the refresh window the caller that refreshes gets the new token.
  now += std::chrono::seconds(90);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST_F(GcpBrokerCredentialProviderTest, FailedRefreshKeepsValidTokenWithSingleAttempt) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(2)));
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::InternalServerError, "{}");
  auto client = MakeBrokerClient(MakeConfig(), transport, [&now] { return now; });

  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Refresh fails with a single attempt; the still-valid token is kept.
  now += std::chrono::seconds(90);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 2u);

  // Right after the failure the Broker is left alone; the token still works.
  now += std::chrono::seconds(1);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 2u);

  // Once the backoff is over another single attempt is made (the transport
  // has nothing queued, so it throws) and the valid token is still served.
  now += std::chrono::seconds(2);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 3u);

  // An expired token cannot be a fallback; propagate the refresh failure.
  now += std::chrono::minutes(5);
  EXPECT_FALSE(ReadHeader(client).ok());
  EXPECT_EQ(transport->RequestCount(), 4u);
}

TEST_F(GcpBrokerCredentialProviderTest, InitialFailureReturnsNoHeader) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  for (int i = 0; i < 4; ++i) {
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "{}");
  }
  auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });

  EXPECT_FALSE(ReadHeader(client).ok());
  EXPECT_EQ(transport->RequestCount(), 4u);
}

TEST_F(GcpBrokerCredentialProviderTest, BrokerAccessDeniedUsesInitialRetriesAndBackoff) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  for (const auto status :
       {Azure::Core::Http::HttpStatusCode::Unauthorized, Azure::Core::Http::HttpStatusCode::Forbidden}) {
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) transport->Enqueue(status, "denied");
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok());
    EXPECT_FALSE(ReadHeader(client).ok());
    EXPECT_EQ(transport->RequestCount(), 4u);
  }
}

TEST_F(GcpBrokerCredentialProviderTest, NegativeCacheSuppressesFetchesAfterFailure) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  for (int i = 0; i < 4; ++i) {
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "{}");
  }
  auto client = MakeBrokerClient(MakeConfig(), transport, [&now] { return now; });

  EXPECT_FALSE(ReadHeader(client).ok());
  EXPECT_EQ(transport->RequestCount(), 4u);

  // Within the backoff window the failure is returned without a Broker call.
  now += std::chrono::milliseconds(1500);
  EXPECT_FALSE(ReadHeader(client).ok());
  EXPECT_EQ(transport->RequestCount(), 4u);

  // After the window the Broker is asked again and a recovery is picked up.
  now += std::chrono::milliseconds(1000);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("recovered", now + std::chrono::hours(1)));
  EXPECT_EQ(ReadHeader(client)->second, "Bearer recovered");
  EXPECT_EQ(transport->RequestCount(), 5u);
}

TEST_F(GcpBrokerCredentialProviderTest, InvalidBrokerRequestReturnsNoHeader) {
  auto config = MakeConfig();
  config.gcp_target_service_account = std::string(1, static_cast<char>(0xff));
  auto transport = std::make_shared<MockTransport>();
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto client = MakeBrokerClient(config, transport, [now] { return now; });
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_FALSE(ReadHeader(client).ok());
  }
  EXPECT_EQ(transport->RequestCount(), 0u);
}

#if !defined(_WIN32)
TEST_F(GcpBrokerCredentialProviderTest, AnotherCallerCanRefreshAfterTheOwnerThrows) {
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto recover = [] {
    const auto now = std::chrono::system_clock::from_time_t(1785146400);
    auto transport = std::make_shared<MockTransport>();
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok,
                       SuccessResponse("recovered", now + std::chrono::hours(1)));
    int clock_calls = 0;
    auto client = MakeBrokerClient(MakeConfig(), transport, [&] {
      if (++clock_calls == 2)
        throw std::runtime_error("failed during fetch");
      return now;
    });
    const auto first = ReadHeader(client);
    const auto second = ReadHeader(client);
    return !first.ok() && second.ok() && second->second == "Bearer recovered" ? 0 : 1;
  };
  EXPECT_EXIT((::alarm(5), ::_exit(recover())), ::testing::ExitedWithCode(0), "");
  GTEST_FLAG_SET(death_test_style, previous_style);
}
#endif

TEST_F(GcpBrokerCredentialProviderTest, ConcurrentInitialRequestsShareFetchRetries) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::ServiceUnavailable, "unavailable");
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("shared", now + std::chrono::hours(1)));
  auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });

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
    threads.emplace_back([&client, &headers, i] { headers[i] = ReadHeader(client); });
  }
  all_started.set_value();
  for (auto& thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < kCallers; ++i) {
    ASSERT_TRUE(headers[i].ok()) << "caller " << i;
    EXPECT_EQ(headers[i]->second, "Bearer shared") << "caller " << i;
  }
  // One failure plus one successful retry, shared by all callers.
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST_F(GcpBrokerCredentialProviderTest, ConcurrentCallersGetStaleTokenWhileOneRefreshes) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto transport = std::make_shared<MockTransport>();
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::minutes(2)));
  auto client = MakeBrokerClient(MakeConfig(), transport, [&now] { return now; });
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
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
  std::thread refresher([&] { refresher_header = ReadHeader(client); });
  ASSERT_TRUE(WaitUntil([&] { return transport->RequestCount() == 2; }));

  // While the refresh is stuck in the Broker call, other callers must get the
  // cached token right away instead of waiting or fetching.
  constexpr size_t kCallers = 8;
  std::vector<Header> headers(kCallers);
  std::atomic<size_t> finished{0};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kCallers; ++i) {
    threads.emplace_back([&, i] {
      headers[i] = ReadHeader(client);
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
    ASSERT_TRUE(headers[i].ok()) << "caller " << i;
    EXPECT_EQ(headers[i]->second, "Bearer first") << "caller " << i;
  }
  ASSERT_TRUE(refresher_header.ok());
  EXPECT_EQ(refresher_header->second, "Bearer second");
  // Exactly one Broker call for the whole refresh.
  EXPECT_EQ(transport->RequestCount(), 2u);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST_F(GcpBrokerCredentialProviderTest, LifetimeIsCappedAtRequestedDuration) {
  auto now = std::chrono::system_clock::from_time_t(1785146400);
  auto config = MakeConfig();
  config.load_frequency = 120;
  auto transport = std::make_shared<MockTransport>();
  // The Broker claims one hour; the provider asked for two minutes.
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("first", now + std::chrono::hours(1)));
  auto client = MakeBrokerClient(config, transport, [&now] { return now; });

  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  now += std::chrono::seconds(30);
  EXPECT_EQ(ReadHeader(client)->second, "Bearer first");
  EXPECT_EQ(transport->RequestCount(), 1u);

  // Past the requested duration the token is treated as expired even though
  // the Broker's expiredAt is still far away.
  now += std::chrono::seconds(91);
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("second", now + std::chrono::hours(1)));
  EXPECT_EQ(ReadHeader(client)->second, "Bearer second");
  EXPECT_EQ(transport->RequestCount(), 2u);
}

TEST_F(GcpBrokerCredentialProviderTest, OversizedResponseIsRejected) {
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
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok());
    EXPECT_EQ(transport->RequestCount(), 4u);
  }
  {
    // Small body, but Content-Length announces more than the limit.
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("small", now + std::chrono::hours(1)),
                         {{"Content-Length", std::to_string(kLimit + 1)}});
    }
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok());
  }
  {
    // Unparsable Content-Length is rejected as well.
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse("small", now + std::chrono::hours(1)),
                         {{"Content-Length", "-1"}});
    }
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok());
  }
  {
    // A large token under the limit with a matching Content-Length passes.
    auto transport = std::make_shared<MockTransport>();
    const auto token = std::string(60 * 1024, 'b');
    const auto body = SuccessResponse(token, now + std::chrono::hours(1));
    ASSERT_LE(body.size(), kLimit);
    transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, body, {{"Content-Length", std::to_string(body.size())}});
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    auto header = ReadHeader(client);
    ASSERT_TRUE(header.ok());
    EXPECT_EQ(header->second, "Bearer " + token);
  }
}

TEST_F(GcpBrokerCredentialProviderTest, TokenWithDisallowedCharactersIsRejected) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  const std::vector<std::string> bad_tokens = {
      "ya29.tok en", "token\r\nX-Injected: 1", "to=ken", "\"token\"", "token;", "=abc", "tok\xC3\xA9n",
  };
  for (const auto& token : bad_tokens) {
    auto transport = std::make_shared<MockTransport>();
    for (int i = 0; i < 4; ++i) {
      transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse(token, now + std::chrono::hours(1)));
    }
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok()) << token;
  }

  // The full RFC 6750 b64token alphabet with trailing padding is accepted.
  auto transport = std::make_shared<MockTransport>();
  const std::string good_token = "ya29.a0Af-H_~+/AZ09==";
  transport->Enqueue(Azure::Core::Http::HttpStatusCode::Ok, SuccessResponse(good_token, now + std::chrono::hours(1)));
  auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
  auto header = ReadHeader(client);
  ASSERT_TRUE(header.ok());
  EXPECT_EQ(header->second, "Bearer " + good_token);
}

TEST_F(GcpBrokerCredentialProviderTest, InvalidResponsesReturnNoHeader) {
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
    auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
    EXPECT_FALSE(ReadHeader(client).ok()) << body;
  }
}

TEST_F(GcpBrokerCredentialProviderTest, TransportExceptionDoesNotEscape) {
  const auto now = std::chrono::system_clock::from_time_t(1785146400);
  // No queued response: every Send throws TransportException.
  auto transport = std::make_shared<MockTransport>();
  auto client = MakeBrokerClient(MakeConfig(), transport, [now] { return now; });
  EXPECT_FALSE(ReadHeader(client).ok());
}

TEST_F(GcpCredentialProviderSelectionTest, CredentialEndpointSelectsBrokerInBothModes) {
  auto config = MakeGcpConfig();
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  config.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";

  auto broker = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(broker.ok()) << broker.status().ToString();
  EXPECT_NE(dynamic_cast<GcpBrokerCredentialProvider*>(broker->get()), nullptr);
  EXPECT_TRUE((*broker)->GetAWSCredentials().IsEmpty());

  // Like AWS, IAM selects the Broker's identity and ignores target/static credentials.
  config.use_iam = true;
  auto iam = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(iam.ok()) << iam.status();
  EXPECT_NE(dynamic_cast<GcpBrokerCredentialProvider*>(iam->get()), nullptr);
  S3Options target_options, iam_options;
  target_options.credentials_provider = *broker;
  iam_options.credentials_provider = *iam;
  EXPECT_FALSE(iam_options.Equals(target_options));
  config.gcp_target_service_account.clear();
  config.access_key_id = "unused-key";
  config.access_key_value = "unused-secret";
  auto without_target = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(without_target.ok()) << without_target.status();
  S3Options ignored_target;
  ignored_target.credentials_provider = *without_target;
  EXPECT_TRUE(ignored_target.Equals(iam_options));
  for (const int32_t duration : {899, 3601}) {
    config.load_frequency = duration;
    EXPECT_TRUE(BuildGcpProviderFromConfig(config).status().IsInvalid());
  }
  config.load_frequency = 900;

  config.credential_endpoint.clear();
  auto native_iam = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(native_iam.ok()) << native_iam.status();
  EXPECT_EQ(dynamic_cast<GcpBrokerCredentialProvider*>(native_iam->get()), nullptr);
  EXPECT_TRUE((*native_iam)->GetAWSCredentials().IsEmpty());
  config.access_key_id.clear();
  config.access_key_value.clear();
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";
  auto impersonation = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(impersonation.ok()) << impersonation.status();
  S3Options adc_options, impersonation_options;
  adc_options.credentials_provider = *native_iam;
  impersonation_options.credentials_provider = *impersonation;
  EXPECT_FALSE(impersonation_options.Equals(adc_options));

  auto missing_target = MakeGcpConfig();
  missing_target.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";
  EXPECT_TRUE(BuildGcpProviderFromConfig(missing_target).status().IsInvalid());
}

TEST_F(GcpCredentialProviderSelectionTest, TargetAloneSelectsImpersonation) {
  auto config = MakeGcpConfig();
  config.gcp_target_service_account = "reader@project.iam.gserviceaccount.com";

  auto provider = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider.ok()) << provider.status().ToString();
  EXPECT_EQ(dynamic_cast<GcpBrokerCredentialProvider*>(provider->get()), nullptr);
  EXPECT_TRUE((*provider)->GetAWSCredentials().IsEmpty());

  config.load_frequency = 7200;
  EXPECT_TRUE(BuildGcpProviderFromConfig(config).status().IsInvalid());
}

TEST_F(GcpCredentialProviderSelectionTest, HmacWithTargetIsAmbiguousUnlessTalon) {
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
  EXPECT_EQ((*provider)->GetAWSCredentials().GetAWSAccessKeyId(), "ak");
  EXPECT_EQ((*provider)->GetAWSCredentials().GetAWSSecretKey(), "sk");
}

TEST_F(GcpCredentialProviderSelectionTest, HmacIsNotBearer) {
  auto config = MakeGcpConfig();
  config.access_key_id = "ak";
  config.access_key_value = "sk";
  auto provider = BuildGcpProviderFromConfig(config);
  ASSERT_TRUE(provider.ok()) << provider.status().ToString();
  EXPECT_EQ((*provider)->GetAWSCredentials().GetAWSAccessKeyId(), "ak");
  EXPECT_EQ((*provider)->GetAWSCredentials().GetAWSSecretKey(), "sk");
}

}  // namespace milvus_storage
