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

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>
#include <azure/core/http/transport.hpp>

#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

namespace milvus_storage {

struct GcpBrokerConfig {
  std::string endpoint;
  std::string region;
  std::string bucket;
  std::string target_service_account;
  int32_t duration_seconds = 900;
  int64_t request_timeout_ms = 3000;
};

// GcpBrokerCredentialProvider gets OAuth2 access tokens for a target service
// account from a credential Broker, instead of impersonating the target from
// this process. The Broker holds the impersonation grant, so a deployment
// grants roles/iam.serviceAccountTokenCreator to the Broker identity only.
//
// The request uses the Broker protocol of the Azure SAS provider
// (credential_broker.h) with csp=gcp and gcpServiceAccountEmail. The response
// carries the access token in credentials.sessionToken.
//
// AuthorizationHeader() runs inside the AWS SDK HTTP factory callback, which
// must not throw. A token that cannot be obtained yields std::nullopt, so the
// request goes out without Authorization and GCS answers 401/403.
//
// A slow or unavailable Broker must not stall the request threads that share
// one provider. GetToken() therefore never calls the Broker or sleeps while
// holding the mutex, and at most one thread talks to the Broker at a time:
//   - a fresh cached token (more than 60 s left) is returned immediately;
//   - a valid token inside the refresh window is returned immediately by
//     everyone; one caller refreshes it, single attempt, on the way out;
//   - without a valid token, the first caller fetches (with retries only if
//     the provider has never obtained a token) and the others wait on a
//     condition variable for that outcome instead of fetching themselves;
//   - after a failed fetch, callers without a valid token get the failure
//     back immediately for a short negative-cache window instead of hitting
//     the Broker again.
// The injected Clock drives freshness and the negative cache; per-attempt
// request deadlines always use the real clock.
class GcpBrokerCredentialProvider final : public GcpCredentialProvider {
  public:
  using TimePoint = std::chrono::system_clock::time_point;
  using Clock = std::function<TimePoint()>;

  explicit GcpBrokerCredentialProvider(GcpBrokerConfig config,
                                       std::shared_ptr<Azure::Core::Http::HttpTransport> transport = nullptr,
                                       Clock clock = {});
  ~GcpBrokerCredentialProvider() override;

  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override;

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return "broker:" + config_.endpoint + "|" + config_.target_service_account; }

  private:
  struct CachedToken {
    std::string access_token;
    TimePoint expires_at;
  };

  // One Broker round trip on the calling thread, bounded by request_timeout_ms
  // on the real clock. `now` (injected clock) validates and caps the returned
  // lifetime. Never throws.
  arrow::Result<std::string> FetchToken(TimePoint now, TimePoint* expires_at) const;
  // FetchToken plus the initial retry policy when `allow_retries` is set.
  // Sleeps between attempts; must be called without holding mutex_.
  arrow::Result<std::string> FetchTokenWithRetries(bool allow_retries, TimePoint* expires_at) const;
  arrow::Result<std::string> GetToken();

  bool InFailureBackoff(TimePoint now) const;

  GcpBrokerConfig config_;
  std::shared_ptr<Azure::Core::Http::HttpTransport> transport_;
  Clock clock_;

  // Everything below is guarded by mutex_. The mutex is released while the
  // fetching thread talks to the Broker; fetch_in_flight_ marks that window
  // and cv_ wakes the waiters when it closes.
  std::mutex mutex_;
  std::condition_variable cv_;
  // Never reset once set: an expired entry still records that the provider
  // has obtained a token before, which turns off the initial retry policy.
  std::optional<CachedToken> cached_;
  bool fetch_in_flight_ = false;
  // Negative cache: when the last fetch failed and no valid token exists.
  std::optional<TimePoint> last_failure_at_;
  arrow::Status last_failure_ = arrow::Status::OK();
};

}  // namespace milvus_storage
