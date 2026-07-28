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
#include <utility>
#include <vector>

#include <azure/core/datetime.hpp>
#include <azure/core/http/curl_transport.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/io/body_stream.hpp>
#include <azure/core/url.hpp>
#include <nlohmann/json.hpp>

#include "milvus-storage/common/log.h"

namespace milvus_storage::fs {

static constexpr auto kRefreshOffset = std::chrono::seconds(60);

struct AzureSasTokenPolicy::SharedState {
  struct CachedSasToken {
    std::string sas_token;
    TimePoint expires_at;
  };

  AzureSasBrokerConfig config;
  std::shared_ptr<Azure::Core::Http::HttpTransport> transport;
  Clock clock;
  std::mutex mutex;
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
  json.at("tempAk").get_to(token.account_name);
  json.at("sessionToken").get_to(token.sas_token);
  json.at("expiredAt").get_to(token.expired_at);
}

static void from_json(const nlohmann::json& json, AzureSasBrokerResponse& response) {
  json.at("success").get_to(response.success);
  if (response.success) {
    json.at("credentials").get_to(response.token);
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

arrow::Result<std::string> AzureSasTokenPolicy::FetchSasToken(TimePoint now, TimePoint* expires_at) const {
  const nlohmann::json request_json = {
      {"csp", "azure"},
      {"region", state_->config.region},
      {"bucket", state_->config.bucket},
      {"durationSeconds", state_->config.duration_seconds},
      {"azureClientId", state_->config.client_id},
      {"azureTenantId", state_->config.tenant_id},
      {"azureAccountName", state_->config.account_name},
  };

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
    // context deadlines against the real system clock.
    const auto deadline = Azure::DateTime(std::chrono::system_clock::now() +
                                          std::chrono::milliseconds(state_->config.request_timeout_ms));
    const auto context = Azure::Core::Context().WithDeadline(deadline);
    auto response = state_->transport->Send(request, context);
    request_completed = true;
    if (!response) {
      return arrow::Status::IOError("Azure SAS credential broker returned no HTTP response: endpoint=",
                                    state_->config.endpoint);
    }
    const int status_code = static_cast<int>(response->GetStatusCode());
    if (status_code < 200 || status_code >= 300) {
      return arrow::Status::IOError(
          "Azure SAS credential broker returned an HTTP error: endpoint=", state_->config.endpoint,
          ", status_code=", status_code, ", reason=", response->GetReasonPhrase());
    }

    auto response_body = response->GetBody();
    if (auto body_stream = response->ExtractBodyStream()) {
      response_body = body_stream->ReadToEnd(context);
    }
    response_body_read_completed = true;
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

    const auto parsed = Azure::DateTime::Parse(broker_response.token.expired_at, Azure::DateTime::DateFormat::Rfc3339);
    const auto expiration = static_cast<std::chrono::system_clock::time_point>(parsed);
    if (expiration <= now) {
      return arrow::Status::IOError(
          "Azure SAS credential broker returned an expired token: expiredAt=", broker_response.token.expired_at,
          ", current_time=", Azure::DateTime(now).ToString(Azure::DateTime::DateFormat::Rfc3339));
    }

    *expires_at = expiration;
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
  std::lock_guard<std::mutex> lock(state_->mutex);
  const auto now = state_->clock();
  if (state_->cached.has_value() && IsFresh(state_->cached->expires_at, now)) {
    return state_->cached->sas_token;
  }

  TimePoint expires_at;
  const bool had_cached_sas = state_->cached.has_value();
  auto fetched = FetchSasToken(now, &expires_at);
  if (fetched.ok()) {
    state_->cached = SharedState::CachedSasToken{std::move(fetched).ValueOrDie(), expires_at};
    LOG_STORAGE_DEBUG_ << "Azure SAS credential broker refresh succeeded: endpoint=" << state_->config.endpoint
                       << ", account=" << state_->config.account_name << ", bucket=" << state_->config.bucket
                       << ", region=" << state_->config.region << ", had_cached_sas=" << std::boolalpha
                       << had_cached_sas
                       << ", expires_at=" << Azure::DateTime(expires_at).ToString(Azure::DateTime::DateFormat::Rfc3339);
    return state_->cached->sas_token;
  }

  const bool cached_expired = had_cached_sas && state_->cached->expires_at <= now;
  LOG_STORAGE_WARNING_ << "Azure SAS credential broker refresh failed: " << fetched.status().ToString()
                       << ", has_cached_sas=" << std::boolalpha << had_cached_sas
                       << ", cached_expired=" << std::boolalpha << cached_expired;
  if (had_cached_sas) {
    return state_->cached->sas_token;
  }
  return fetched.status();
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
