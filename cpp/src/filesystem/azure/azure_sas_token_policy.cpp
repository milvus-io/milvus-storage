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

#include "milvus-storage/filesystem/azure/azure_sas_token_policy.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <azure/core/datetime.hpp>
#include <azure/core/http/curl_transport.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/io/body_stream.hpp>
#include <azure/core/url.hpp>
#include <nlohmann/json.hpp>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/credential_broker.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::fs {

static constexpr auto kRefreshOffset = std::chrono::seconds(60);
static constexpr int kMaxRetries = 3;

struct AzureSasTokenPolicy::SharedState {
  struct CachedSasToken {
    std::string sas_token;
    TimePoint expires_at;
  };

  AzureSasBrokerConfig config;
  std::shared_ptr<Azure::Core::Http::HttpTransport> transport;
  Clock clock;
  std::mutex cache_mutex;
  std::mutex refresh_mutex;
  std::optional<CachedSasToken> cached;
};

struct AzureSasBrokerToken {
  std::string account_name;
  std::string sas_token;
  std::string expired_at;
};

struct AzureSasBrokerResponse {
  bool success = false;
  AzureSasBrokerToken token;
};

static void from_json(const nlohmann::json& json, AzureSasBrokerToken& token) {
  json.at(credential_broker::kTempAk).get_to(token.account_name);
  json.at(credential_broker::kSessionToken).get_to(token.sas_token);
  json.at(credential_broker::kExpiredAt).get_to(token.expired_at);
}

static void from_json(const nlohmann::json& json, AzureSasBrokerResponse& response) {
  json.at(credential_broker::kSuccess).get_to(response.success);
  if (response.success) {
    json.at(credential_broker::kCredentials).get_to(response.token);
  }
}

AzureSasTokenPolicy::AzureSasTokenPolicy(AzureSasBrokerConfig config,
                                         std::shared_ptr<Azure::Core::Http::HttpTransport> transport,
                                         Clock clock)
    : state_(std::make_shared<SharedState>()) {
  state_->config = std::move(config);
  state_->transport = std::move(transport);
  state_->clock = std::move(clock);
  if (!state_->transport) {
    Azure::Core::Http::CurlTransportOptions options;
    options.ConnectionTimeout = std::chrono::milliseconds(state_->config.request_timeout_ms);
    state_->transport = std::make_shared<Azure::Core::Http::CurlTransport>(options);
  }
  if (!state_->clock) {
    state_->clock = [] { return std::chrono::system_clock::now(); };
  }
}

bool AzureSasTokenPolicy::IsFresh(TimePoint expires_at, TimePoint now) { return expires_at - now > kRefreshOffset; }

arrow::Result<std::string> AzureSasTokenPolicy::FetchSasToken(TimePoint* expires_at, bool* retryable) const {
  *retryable = false;
  nlohmann::json request_json = {
      {credential_broker::kCsp, kCloudProviderAzure},
      {credential_broker::kRegion, state_->config.region},
      {credential_broker::kBucket, state_->config.bucket},
      {credential_broker::kDurationSeconds, state_->config.duration_seconds},
      {credential_broker::kAzureAccountName, state_->config.account_name},
  };
  if (state_->config.use_iam) {
    request_json[credential_broker::kCredentialType] = credential_broker::kIam;
  } else {
    request_json[credential_broker::kAzureClientId] = state_->config.client_id;
    request_json[credential_broker::kAzureTenantId] = state_->config.tenant_id;
  }

  const auto request_body = request_json.dump();
  const std::vector<uint8_t> request_bytes(request_body.begin(), request_body.end());
  Azure::Core::IO::MemoryBodyStream body_stream(request_bytes);

  bool request_completed = false;
  bool response_body_read_completed = false;
  size_t response_size = 0;
  AzureSasBrokerResponse broker_response;
  try {
    Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Post, Azure::Core::Url(state_->config.endpoint),
                                       &body_stream);
    request.SetHeader("Content-Type", "application/json");
    request.SetHeader("Content-Length", std::to_string(request_bytes.size()));

    // The injected clock controls credential freshness only. Azure evaluates
    // context deadlines against the real system clock. Each attempt gets its
    // own request timeout, including reading the response body.
    const auto deadline = Azure::DateTime(std::chrono::system_clock::now() +
                                          std::chrono::milliseconds(state_->config.request_timeout_ms));
    const auto context = Azure::Core::Context().WithDeadline(deadline);
    *retryable = true;  // Transport and body-read failures are transient.
    auto response = state_->transport->Send(request, context);
    request_completed = true;
    if (!response) {
      return arrow::Status::IOError("Azure SAS credential broker returned no HTTP response: endpoint=",
                                    state_->config.endpoint);
    }
    const int status_code = static_cast<int>(response->GetStatusCode());
    if (status_code < 200 || status_code >= 300) {
      *retryable = status_code == 429 || (status_code >= 500 && status_code < 600);
      return arrow::Status::IOError(
          "Azure SAS credential broker returned an HTTP error: endpoint=", state_->config.endpoint,
          ", status_code=", status_code, ", reason=", response->GetReasonPhrase());
    }

    auto response_body = response->GetBody();
    if (auto body_stream = response->ExtractBodyStream()) {
      response_body = body_stream->ReadToEnd(context);
    }
    response_body_read_completed = true;
    *retryable = false;  // Denials and invalid broker responses invalidate the cached SAS.
    response_size = response_body.size();
    const auto response_json = nlohmann::json::parse(response_body.begin(), response_body.end());
    response_json.get_to(broker_response);

    if (!broker_response.success) {
      return arrow::Status::IOError("Azure SAS credential broker returned success=false");
    }

    if (broker_response.token.account_name != state_->config.account_name) {
      return arrow::Status::IOError("Azure SAS credential broker returned a token for the wrong account: expected=",
                                    state_->config.account_name, ", actual=", broker_response.token.account_name);
    }

    auto sas_token = std::move(broker_response.token.sas_token);
    if (!sas_token.empty() && sas_token.front() == '?') {
      sas_token.erase(0, 1);
    }
    if (sas_token.empty()) {
      return arrow::Status::IOError(
          "Azure SAS credential broker response field 'credentials.sessionToken' must not be empty");
    }
    const auto sas_query_parameters = Azure::Core::Url("?" + sas_token).GetQueryParameters();
    const auto signature = sas_query_parameters.find("sig");
    if (signature == sas_query_parameters.end() || signature->second.empty()) {
      return arrow::Status::IOError(
          "Azure SAS credential broker response field 'credentials.sessionToken' must contain a non-empty 'sig' "
          "query parameter");
    }

    const auto parsed = Azure::DateTime::Parse(broker_response.token.expired_at, Azure::DateTime::DateFormat::Rfc3339);
    *expires_at = static_cast<TimePoint>(parsed);
    return sas_token;
  } catch (const nlohmann::json::parse_error& error) {
    return arrow::Status::IOError(
        "Azure SAS credential broker returned invalid JSON: endpoint=", state_->config.endpoint,
        ", parse_error_byte=", error.byte, ", response_size=", response_size);
  } catch (const nlohmann::json::exception& error) {
    return arrow::Status::IOError("Azure SAS credential broker response schema is invalid: ", error.what());
  } catch (const std::exception& error) {
    if (!request_completed) {
      return arrow::Status::IOError("Azure SAS credential broker request failed: endpoint=", state_->config.endpoint,
                                    ", error=", error.what());
    }
    if (!response_body_read_completed) {
      return arrow::Status::IOError("Azure SAS credential broker response body read failed: endpoint=",
                                    state_->config.endpoint, ", error=", error.what());
    }
    return arrow::Status::IOError("Azure SAS credential broker returned an invalid expiration time: expiredAt=",
                                  broker_response.token.expired_at, ", error=", error.what());
  }
}

arrow::Result<std::string> AzureSasTokenPolicy::GetSasToken() const {
  std::unique_lock refresh_lock(state_->refresh_mutex, std::defer_lock);
  {
    std::unique_lock cache_lock(state_->cache_mutex);
    const auto now = state_->clock();
    if (state_->cached && IsFresh(state_->cached->expires_at, now)) {
      return state_->cached->sas_token;
    }
    if (!refresh_lock.try_lock()) {
      // A concurrent refresh may wait on the broker. Valid readers can proceed.
      if (state_->cached && state_->cached->expires_at > now) {
        return state_->cached->sas_token;
      }
      cache_lock.unlock();
      refresh_lock.lock();
      cache_lock.lock();
      if (state_->cached && IsFresh(state_->cached->expires_at, state_->clock())) {
        return state_->cached->sas_token;
      }
    }
  }

  for (int retries = 0;; ++retries) {
    bool retryable = false;
    TimePoint expires_at;
    auto fetched = FetchSasToken(&expires_at, &retryable);
    {
      std::lock_guard cache_lock(state_->cache_mutex);
      // Check both new and cached credentials after the broker round trip.
      const auto now = state_->clock();
      if (fetched.ok()) {
        if (expires_at <= now) {
          state_->cached.reset();
          return arrow::Status::IOError(
              "Azure SAS credential broker returned an expired token: expiredAt=",
              Azure::DateTime(expires_at).ToString(Azure::DateTime::DateFormat::Rfc3339),
              ", current_time=", Azure::DateTime(now).ToString(Azure::DateTime::DateFormat::Rfc3339));
        }
        LOG_STORAGE_DEBUG_ << "Azure SAS credential broker refresh succeeded: endpoint=" << state_->config.endpoint
                           << ", account=" << state_->config.account_name << ", bucket=" << state_->config.bucket
                           << ", region=" << state_->config.region << ", had_cached_sas=" << std::boolalpha
                           << state_->cached.has_value() << ", expires_at="
                           << Azure::DateTime(expires_at).ToString(Azure::DateTime::DateFormat::Rfc3339);
        state_->cached = SharedState::CachedSasToken{std::move(fetched).ValueOrDie(), expires_at};
        return state_->cached->sas_token;
      }
      if (!retryable) {
        state_->cached.reset();
        return fetched.status();
      }
      if (state_->cached && state_->cached->expires_at > now) {
        LOG_STORAGE_WARNING_ << "Azure SAS credential broker refresh failed, using valid cached SAS: "
                             << fetched.status();
        return state_->cached->sas_token;
      }
    }

    if (retries == kMaxRetries) {
      return fetched.status();
    }
    const auto delay = std::chrono::milliseconds(200 << retries);
    LOG_STORAGE_WARNING_ << "Azure SAS credential broker request failed: " << fetched.status()
                         << ", retry=" << retries + 1 << "/" << kMaxRetries << ", retry_delay_ms=" << delay.count();
    std::this_thread::sleep_for(delay);
  }
}

AzureSasTokenPolicy::~AzureSasTokenPolicy() = default;

bool AzureSasTokenPolicy::Equals(const AzureSasTokenPolicy& other) const {
  return state_->config == other.state_->config;
}

std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy> AzureSasTokenPolicy::Clone() const {
  return std::make_unique<AzureSasTokenPolicy>(*this);
}

std::unique_ptr<Azure::Core::Http::RawResponse> AzureSasTokenPolicy::Send(
    Azure::Core::Http::Request& request,
    Azure::Core::Http::Policies::NextHttpPolicy next_policy,
    const Azure::Core::Context& context) const {
  auto sas_token = GetSasToken();
  if (!sas_token.ok()) {
    // Stop the Azure pipeline before it sends an unauthenticated request. The
    // Broker errors are kept in the reason phrase so callers can see the SAS
    // token failure through Azure's RequestFailedException.
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(
        1, 1, Azure::Core::Http::HttpStatusCode::Unauthorized,
        "Azure SAS token unavailable: " + sas_token.status().ToString());
    response->SetHeader("x-ms-error-code", "AuthenticationFailed");
    response->SetHeader("Content-Length", "0");
    return response;
  }

  // SAS authentication is encoded as URL query parameters. Parse the token
  // with Azure::Core::Url and append it without replacing operation parameters
  // already added by the Storage SDK, such as comp or restype.
  const auto sas_query_parameters = Azure::Core::Url("?" + sas_token.ValueOrDie()).GetQueryParameters();
  for (const auto& [key, value] : sas_query_parameters) {
    request.GetUrl().AppendQueryParameter(key, value);
  }
  return next_policy.Send(request, context);
}

}  // namespace milvus_storage::fs
