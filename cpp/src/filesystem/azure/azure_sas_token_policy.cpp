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

#include "milvus-storage/filesystem/azure/azurefs_internal.h"
#include "milvus-storage/common/extend_status.h"
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
      // Classified HERE, at the producer. The policy downstream cannot tell a
      // broker that never answered from a broker that rejected us, and an
      // unclassified failure there defaulted to "credentials are wrong" --
      // permanent, when this is the textbook retriable case.
      return MakeExtendErrorMsg(
          ExtendStatusCode::StorageTransientNetwork,
          "Azure SAS credential broker returned no HTTP response: endpoint=", state_->config.endpoint);
    }
    const int status_code = static_cast<int>(response->GetStatusCode());
    if (status_code < 200 || status_code >= 300) {
      // 408/429/5xx are the broker being unwell, not a verdict about our
      // credentials; only a 401/403 is that. Same split the S3 and Azure
      // filesystems already make on their own responses.
      std::optional<ExtendStatusCode> transient;
      if (status_code == 408) {
        transient = ExtendStatusCode::StorageTransientTimeout;
      } else if (status_code == 429) {
        transient = ExtendStatusCode::StorageTransientThrottling;
      } else if (status_code >= 500) {
        transient = ExtendStatusCode::StorageTransientService;
      }
      if (transient.has_value()) {
        return MakeExtendErrorMsg(
            *transient, "Azure SAS credential broker returned an HTTP error: endpoint=", state_->config.endpoint,
            ", status_code=", status_code, ", reason=", response->GetReasonPhrase());
      }
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
    const auto sas_query_parameters = Azure::Core::Url("?" + sas_token).GetQueryParameters();
    const auto signature = sas_query_parameters.find("sig");
    if (signature == sas_query_parameters.end() || signature->second.empty()) {
      return arrow::Status::IOError(
          "Azure SAS credential broker response field 'credentials.sessionToken' must contain a non-empty 'sig' "
          "query parameter");
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
  } catch (const Azure::Core::Http::TransportException& error) {
    // The request never reached the broker -- connection refused/reset, DNS,
    // TLS. Classified here because the policy downstream can only see the
    // synthetic response it builds, and an unclassified failure there became a
    // 401, i.e. "your credentials are wrong" for a cable that came loose.
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork,
                              "Azure SAS credential broker request failed at the transport: endpoint=",
                              state_->config.endpoint, ", error=", error.what());
  } catch (const Azure::Core::OperationCancelledException& error) {
    // The deadline set on the context above fired.
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientTimeout,
                              "Azure SAS credential broker request timed out: endpoint=", state_->config.endpoint,
                              ", error=", error.what());
  } catch (const std::bad_alloc&) {
    // Ahead of the generic handler. Everything unclassified leaving this
    // function becomes a synthetic 401 downstream -- "your credentials are
    // wrong" -- so an OOM here would be reported as a permanent auth failure.
    // Enforced by cpp/scripts/check_oom_handlers.py.
    return arrow::Status::OutOfMemory("Out of memory fetching an Azure SAS token from the credential broker");
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

  // Re-read the clock. `now` was taken BEFORE the broker request, and that
  // request is exactly the slow thing here -- a token still valid when we
  // asked can have expired by the time we are deciding whether to fall back on
  // it. Using the stale reading let an expired token through, and Azure
  // answers those with 401, which is the credential misattribution this policy
  // exists to avoid.
  const auto decision_time = state_->clock();
  const bool cached_expired = had_cached_sas && state_->cached->expires_at <= decision_time;
  LOG_STORAGE_WARNING_ << "Azure SAS credential broker refresh failed: " << fetched.status().ToString()
                       << ", has_cached_sas=" << std::boolalpha << had_cached_sas
                       << ", cached_expired=" << std::boolalpha << cached_expired;
  // A cached SAS that is still valid is a genuine fallback: the broker is down
  // but the token works. An EXPIRED one is not -- Azure rejects it, and that
  // rejection arrives as 401, i.e. as "your credentials are wrong" for what is
  // actually a broker outage. Report the fetch failure, which is classified.
  if (had_cached_sas && !cached_expired) {
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
    // A broker that timed out, was throttled, or answered 5xx is NOT a rejected
    // credential; reporting both as 401 made them share code 105, so one of the
    // two owners was always wrong -- and it was the retriable one being denied
    // a retry.
    //
    // This policy sits inside the SDK pipeline and cannot return a Status: its
    // only channel is a synthesised HTTP response. That channel is wide enough,
    // though -- ClassifyAzureError already distinguishes 408, 429 and 503 -- so
    // the classification is mapped BACK onto the status code that means it. An
    // earlier version collapsed every transient into 503/ServerBusy, which
    // preserved retriability but relabelled all of them as throttling, sending
    // both metrics and backoff after the wrong thing.
    const auto broker_detail = ExtendStatusDetail::UnwrapStatus(sas_token.status());
    auto status_code = Azure::Core::Http::HttpStatusCode::Unauthorized;
    const char* error_code = "AuthenticationFailed";
    // Guard is load-bearing: without it a non-transient broker failure (a
    // denied credential, say) would fall into the switch's default and be
    // reported as ServiceUnavailable, which reads as a blip to anything
    // downstream that only sees status codes.
    if (broker_detail != nullptr && broker_detail->category() == ErrorCategory::Transient) {
      switch (broker_detail->code()) {
        case ExtendStatusCode::StorageTransientTimeout:
          status_code = Azure::Core::Http::HttpStatusCode::RequestTimeout;
          error_code = "OperationTimedOut";
          break;
        case ExtendStatusCode::StorageTransientThrottling:
          status_code = Azure::Core::Http::HttpStatusCode::TooManyRequests;
          error_code = "ServerBusy";
          break;
        case ExtendStatusCode::StorageTransientNetwork:
          // HTTP has no status meaning "the connection dropped", so the
          // subtype rides in the error code instead: ClassifyAzureError
          // recognises this marker and restores 107 rather than flattening the
          // failure into a service blip. Reported as a 503 so any downstream
          // that only reads status codes still treats it as retriable.
          status_code = Azure::Core::Http::HttpStatusCode::ServiceUnavailable;
          error_code = internal::kSyntheticBrokerNetworkErrorCode;
          break;
        default:
          status_code = Azure::Core::Http::HttpStatusCode::ServiceUnavailable;
          error_code = "ServerUnavailable";
          break;
      }
    }
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(
        1, 1, status_code, "Azure SAS token unavailable: " + sas_token.status().ToString());
    response->SetHeader("x-ms-error-code", error_code);
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
