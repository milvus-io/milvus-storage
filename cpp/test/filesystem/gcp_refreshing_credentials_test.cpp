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
#include <google/cloud/internal/oauth2_compute_engine_credentials.h>
#include <google/cloud/internal/oauth2_minimal_iam_credentials_rest.h>
#include <google/cloud/internal/http_payload.h>
#include <google/cloud/internal/rest_response.h>
#include <algorithm>
#include <cstring>
#include <future>
#include "milvus-storage/filesystem/gcp/refreshing_credentials.h"

namespace milvus_storage {
namespace {
namespace gc = google::cloud;
namespace gi = google::cloud::oauth2_internal;
namespace gr = google::cloud::rest_internal;
using Time = std::chrono::system_clock::time_point;
using namespace std::chrono_literals;
class Payload final : public gr::HttpPayload {
  public:
  explicit Payload(std::string s) : s_(std::move(s)) {}
  bool HasUnreadData() const override { return pos_ < s_.size(); }
  gc::StatusOr<std::size_t> Read(absl::Span<char> out) override {
    auto n = std::min(out.size(), s_.size() - pos_);
    std::memcpy(out.data(), s_.data() + pos_, n);
    pos_ += n;
    return n;
  }

  private:
  std::string s_;
  size_t pos_ = 0;
};
class Response final : public gr::RestResponse {
  public:
  explicit Response(std::string s) : s_(std::move(s)) {}
  gr::HttpStatusCode StatusCode() const override { return gr::HttpStatusCode::kOk; }
  std::multimap<std::string, std::string> Headers() const override { return {}; }
  std::unique_ptr<gr::HttpPayload> ExtractPayload() && override { return std::make_unique<Payload>(std::move(s_)); }

  private:
  std::string s_;
};
class Delegate final : public gi::Credentials {
  public:
  std::function<gc::StatusOr<gc::AccessToken>(Time)> fn;
  int calls = 0;
  gc::StatusOr<gc::AccessToken> GetToken(Time t) override {
    ++calls;
    return fn(t);
  }
};
gc::StatusOr<gc::AccessToken> Parse(std::string body, Time now) {
  Response r(std::move(body));
  return gi::ParseComputeEngineRefreshResponse(r, now);
}
constexpr auto old_json = R"({"access_token":"OLD","expires_in":600,"token_type":"Bearer"})";

TEST(RefreshingGcpCredentialsTest, InvalidSdkResponsesPreserveOldAndRecover) {
  for (const std::string body : {R"({"access_token":"","expires_in":3600,"token_type":"Bearer"})",
                                 R"({"access_token":123,"expires_in":3600,"token_type":"Bearer"})",
                                 R"({"access_token":"EXPIRED","expires_in":-1,"token_type":"Bearer"})",
                                 R"({"access_token":"BAD TOKEN","expires_in":3600,"token_type":"Bearer"})", "{"}) {
    SCOPED_TRACE(body);
    auto d = std::make_shared<Delegate>();
    Time now{};
    RefreshingGcpCredentials cache(d, [&] { return now; });
    d->fn = [](Time t) { return Parse(old_json, t); };
    ASSERT_EQ(cache.GetToken(now)->token, "OLD");
    now += 400s;
    d->fn = [&](Time t) { return Parse(body, t); };
    auto old = cache.GetToken(now);
    ASSERT_TRUE(old.ok());
    EXPECT_EQ(old->token, "OLD");
    d->fn = [](Time t) { return Parse(R"({"access_token":"NEW","expires_in":3600,"token_type":"Bearer"})", t); };
    EXPECT_EQ(cache.GetToken(now)->token, "NEW");
    EXPECT_EQ(cache.GetToken(now)->token, "NEW");
    EXPECT_EQ(d->calls, 3);
  }
}

TEST(RefreshingGcpCredentialsTest, InitialInvalidResponseReturnsErrorThenRecovers) {
  auto d = std::make_shared<Delegate>();
  Time now{};
  RefreshingGcpCredentials cache(d, [&] { return now; });
  d->fn = [](Time t) { return Parse(R"({"access_token":"","expires_in":3600,"token_type":"Bearer"})", t); };
  EXPECT_FALSE(cache.GetToken(now).ok());
  d->fn = [](Time t) { return Parse(old_json, t); };
  EXPECT_EQ(cache.GetToken(now)->token, "OLD");
}

TEST(RefreshingGcpCredentialsTest, ChecksTimeAfterSlowRefreshForCandidateAndFallback) {
  auto d = std::make_shared<Delegate>();
  Time now{};
  RefreshingGcpCredentials cache(d, [&] { return now; });
  d->fn = [](Time t) { return Parse(old_json, t); };
  ASSERT_EQ(cache.GetToken(now)->token, "OLD");
  now += 400s;
  d->fn = [&](Time t) -> gc::StatusOr<gc::AccessToken> {
    now += 10s;
    return gc::AccessToken{"SHORT", t + 5s};
  };
  EXPECT_EQ(cache.GetToken(now)->token, "OLD");
  d->fn = [&](Time) -> gc::StatusOr<gc::AccessToken> {
    now += 300s;
    throw std::runtime_error("synthetic parse failure");
  };
  EXPECT_FALSE(cache.GetToken(now).ok());
  d->fn = [](Time t) { return Parse(old_json, t); };
  EXPECT_EQ(cache.GetToken(now)->token, "OLD");
}

TEST(RefreshingGcpCredentialsTest, StatusFailureKeepsValidCacheButNotExpired) {
  auto d = std::make_shared<Delegate>();
  Time now{};
  RefreshingGcpCredentials cache(d, [&] { return now; });
  d->fn = [](Time t) { return Parse(old_json, t); };
  ASSERT_EQ(cache.GetToken(now)->token, "OLD");
  d->fn = [](Time) -> gc::StatusOr<gc::AccessToken> { return gc::Status(gc::StatusCode::kUnavailable, "synthetic"); };
  now += 400s;
  EXPECT_EQ(cache.GetToken(now)->token, "OLD");
  now += 201s;
  EXPECT_FALSE(cache.GetToken(now).ok());
}

TEST(RefreshingGcpCredentialsTest, RejectsEmptyAndExpiredImpersonationResponse) {
  auto d = std::make_shared<Delegate>();
  Time now = std::chrono::system_clock::now();
  RefreshingGcpCredentials cache(d, [&] { return now; });
  d->fn = [](Time t) { return Parse(old_json, t); };
  ASSERT_EQ(cache.GetToken(now)->token, "OLD");
  now += 400s;
  for (const auto* body : {R"({"accessToken":"","expireTime":"2099-01-01T00:00:00Z"})",
                           R"({"accessToken":"EXPIRED","expireTime":"2000-01-01T00:00:00Z"})"}) {
    d->fn = [body](Time) {
      Response response(body);
      return gi::ParseGenerateAccessTokenResponse(response, {});
    };
    EXPECT_EQ(cache.GetToken(now)->token, "OLD");
  }
}

TEST(RefreshingGcpCredentialsTest, ConcurrentReadersShareValidatedRefresh) {
  auto d = std::make_shared<Delegate>();
  Time now{};
  RefreshingGcpCredentials cache(d, [&] { return now; });
  d->fn = [](Time t) { return Parse(old_json, t); };
  std::vector<std::future<gc::StatusOr<gc::AccessToken>>> readers;
  for (int i = 0; i != 12; ++i) {
    readers.emplace_back(std::async(std::launch::async, [&] { return cache.GetToken(now); }));
  }
  for (auto& reader : readers) {
    EXPECT_EQ(reader.get()->token, "OLD");
  }
  EXPECT_EQ(d->calls, 1);
}
}  // namespace
}  // namespace milvus_storage
