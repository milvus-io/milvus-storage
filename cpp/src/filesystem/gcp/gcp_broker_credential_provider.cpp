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

#include "milvus-storage/filesystem/gcp/gcp_broker_credential_provider.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <system_error>
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

namespace milvus_storage {

namespace {

// Same freshness window and initial retry policy as the Azure SAS provider.
constexpr auto kRefreshOffset = std::chrono::seconds(60);
constexpr int kMaxInitialRetries = 3;
// After a failed fetch, callers without a valid token get the failure back
// without contacting the Broker again for this long.
constexpr auto kFailureBackoff = std::chrono::seconds(2);
// A Broker response is a small JSON document; anything larger is not read.
constexpr size_t kMaxResponseBytes = 64 * 1024;

struct GcpBrokerToken {
  std::string access_token;
  std::string expired_at;
};

struct GcpBrokerResponse {
  bool success = false;
  GcpBrokerToken token;
};

void from_json(const nlohmann::json& json, GcpBrokerToken& token) {
  json.at(credential_broker::kSessionToken).get_to(token.access_token);
  json.at(credential_broker::kExpiredAt).get_to(token.expired_at);
}

void from_json(const nlohmann::json& json, GcpBrokerResponse& response) {
  json.at(credential_broker::kSuccess).get_to(response.success);
  if (response.success) {
    json.at(credential_broker::kCredentials).get_to(response.token);
  }
}

// RFC 6750 b64token: 1*( ALPHA / DIGIT / "-" / "." / "_" / "~" / "+" / "/" ) *"=".
// Anything else (whitespace, control characters, quotes) must not reach the
// Authorization header.
bool IsB64TokenChar(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.' ||
         c == '_' || c == '~' || c == '+' || c == '/';
}

bool IsB64Token(const std::string& token) {
  size_t i = 0;
  while (i < token.size() && IsB64TokenChar(token[i])) {
    ++i;
  }
  if (i == 0) {
    return false;
  }
  while (i < token.size() && token[i] == '=') {
    ++i;
  }
  return i == token.size();
}

bool ParseContentLength(const std::string& value, uint64_t* content_length) {
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, *content_length);
  return result.ec == std::errc() && result.ptr == end && begin != end;
}

}  // namespace

GcpBrokerCredentialProvider::GcpBrokerCredentialProvider(GcpBrokerConfig config,
                                                         std::shared_ptr<Azure::Core::Http::HttpTransport> transport,
                                                         Clock clock)
    : config_(std::move(config)), transport_(std::move(transport)), clock_(std::move(clock)) {
  if (!transport_) {
    Azure::Core::Http::CurlTransportOptions options;
    options.ConnectionTimeout = std::chrono::milliseconds(config_.request_timeout_ms);
    transport_ = std::make_shared<Azure::Core::Http::CurlTransport>(options);
  }
  if (!clock_) {
    clock_ = [] { return std::chrono::system_clock::now(); };
  }
}

GcpBrokerCredentialProvider::~GcpBrokerCredentialProvider() = default;

arrow::Result<std::string> GcpBrokerCredentialProvider::FetchToken(TimePoint now, TimePoint* expires_at) const {
  const nlohmann::json request_json = {
      {credential_broker::kCsp, kCloudProviderGCP},
      {credential_broker::kRegion, config_.region},
      {credential_broker::kBucket, config_.bucket},
      {credential_broker::kDurationSeconds, config_.duration_seconds},
      {credential_broker::kGcpServiceAccountEmail, config_.target_service_account},
  };

  const auto request_body = request_json.dump();
  const std::vector<uint8_t> request_bytes(request_body.begin(), request_body.end());
  Azure::Core::IO::MemoryBodyStream body_stream(request_bytes);

  bool request_completed = false;
  bool response_body_read_completed = false;
  size_t response_size = 0;
  GcpBrokerResponse broker_response;
  try {
    Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Post, Azure::Core::Url(config_.endpoint),
                                       &body_stream);
    request.SetHeader("Content-Type", "application/json");
    request.SetHeader("Content-Length", std::to_string(request_bytes.size()));

    // The injected clock controls token freshness only. The deadline uses the
    // real clock and also bounds reading the response body.
    const auto deadline =
        Azure::DateTime(std::chrono::system_clock::now() + std::chrono::milliseconds(config_.request_timeout_ms));
    const auto context = Azure::Core::Context().WithDeadline(deadline);
    auto response = transport_->Send(request, context);
    request_completed = true;
    if (!response) {
      return arrow::Status::IOError("GCP credential broker returned no HTTP response: endpoint=", config_.endpoint);
    }
    const int status_code = static_cast<int>(response->GetStatusCode());
    if (status_code < 200 || status_code >= 300) {
      return arrow::Status::IOError("GCP credential broker returned an HTTP error: endpoint=", config_.endpoint,
                                    ", status_code=", status_code, ", reason=", response->GetReasonPhrase());
    }

    // Refuse oversized responses before reading them: first by the announced
    // Content-Length, then by reading at most one byte past the limit.
    const auto& headers = response->GetHeaders();
    if (const auto header = headers.find("Content-Length"); header != headers.end()) {
      uint64_t content_length = 0;
      if (!ParseContentLength(header->second, &content_length)) {
        return arrow::Status::IOError("GCP credential broker returned an invalid Content-Length header: endpoint=",
                                      config_.endpoint, ", content_length=", header->second);
      }
      if (content_length > kMaxResponseBytes) {
        return arrow::Status::IOError("GCP credential broker response is too large: endpoint=", config_.endpoint,
                                      ", content_length=", content_length, ", limit=", kMaxResponseBytes);
      }
    }
    std::vector<uint8_t> response_body;
    if (auto stream = response->ExtractBodyStream()) {
      response_body.resize(kMaxResponseBytes + 1);
      const size_t read = stream->ReadToCount(response_body.data(), response_body.size(), context);
      response_body.resize(read);
    } else {
      response_body = response->GetBody();
    }
    response_body_read_completed = true;
    response_size = response_body.size();
    if (response_size > kMaxResponseBytes) {
      return arrow::Status::IOError("GCP credential broker response body is too large: endpoint=", config_.endpoint,
                                    ", read_at_least=", response_size, ", limit=", kMaxResponseBytes);
    }
    const auto response_json = nlohmann::json::parse(response_body.begin(), response_body.end());
    response_json.get_to(broker_response);

    if (!broker_response.success) {
      return arrow::Status::IOError("GCP credential broker returned success=false: endpoint=", config_.endpoint);
    }
    if (broker_response.token.access_token.empty()) {
      return arrow::Status::IOError(
          "GCP credential broker response field 'credentials.sessionToken' must not be empty");
    }
    if (!IsB64Token(broker_response.token.access_token)) {
      return arrow::Status::IOError(
          "GCP credential broker response field 'credentials.sessionToken' is not a valid bearer token: length=",
          broker_response.token.access_token.size());
    }

    const auto parsed = Azure::DateTime::Parse(broker_response.token.expired_at, Azure::DateTime::DateFormat::Rfc3339);
    const auto expiration = static_cast<std::chrono::system_clock::time_point>(parsed);
    if (expiration <= now) {
      return arrow::Status::IOError(
          "GCP credential broker returned an expired token: expiredAt=", broker_response.token.expired_at,
          ", current_time=", Azure::DateTime(now).ToString(Azure::DateTime::DateFormat::Rfc3339));
    }

    // The Broker cannot grant a longer lifetime than this provider asked for.
    *expires_at = std::min(expiration, now + std::chrono::seconds(config_.duration_seconds));
    return std::move(broker_response.token.access_token);
  } catch (const nlohmann::json::parse_error& error) {
    return arrow::Status::IOError("GCP credential broker returned invalid JSON: endpoint=", config_.endpoint,
                                  ", parse_error_byte=", error.byte, ", response_size=", response_size);
  } catch (const nlohmann::json::exception& error) {
    return arrow::Status::IOError("GCP credential broker response schema is invalid: ", error.what());
  } catch (const std::exception& error) {
    if (!request_completed) {
      return arrow::Status::IOError("GCP credential broker request failed: endpoint=", config_.endpoint,
                                    ", error=", error.what());
    }
    if (!response_body_read_completed) {
      return arrow::Status::IOError("GCP credential broker response body read failed: endpoint=", config_.endpoint,
                                    ", error=", error.what());
    }
    return arrow::Status::IOError("GCP credential broker returned an invalid expiration time: expiredAt=",
                                  broker_response.token.expired_at, ", error=", error.what());
  } catch (...) {
    // Non-std exceptions, or std ones whose type does not resolve across
    // library boundaries: classify by phase so the log stays actionable.
    if (!request_completed) {
      return arrow::Status::IOError("GCP credential broker request failed with an unknown exception: endpoint=",
                                    config_.endpoint);
    }
    if (!response_body_read_completed) {
      return arrow::Status::IOError(
          "GCP credential broker response body read failed with an unknown exception: endpoint=", config_.endpoint);
    }
    return arrow::Status::IOError("GCP credential broker returned an invalid expiration time: expiredAt=",
                                  broker_response.token.expired_at, ", error=unknown exception");
  }
}

arrow::Result<std::string> GcpBrokerCredentialProvider::FetchTokenWithRetries(bool allow_retries,
                                                                              TimePoint* expires_at) const {
  auto fetched = FetchToken(clock_(), expires_at);
  auto retry_delay = std::chrono::milliseconds(100);
  for (int retries = 0; allow_retries && !fetched.ok() && retries < kMaxInitialRetries; ++retries) {
    LOG_STORAGE_WARNING_ << "GCP credential broker fetch failed: " << fetched.status().ToString()
                         << ", endpoint=" << config_.endpoint << ", retry=" << retries + 1 << "/" << kMaxInitialRetries
                         << ", retry_delay_ms=" << retry_delay.count();
    std::this_thread::sleep_for(retry_delay);
    fetched = FetchToken(clock_(), expires_at);
    retry_delay *= 2;
  }
  return fetched;
}

bool GcpBrokerCredentialProvider::InFailureBackoff(TimePoint now) const {
  // A clock that moved backwards ends the backoff early, which costs at most
  // one extra Broker call instead of suppressing fetches indefinitely.
  return last_failure_at_.has_value() && now >= *last_failure_at_ && now - *last_failure_at_ < kFailureBackoff;
}

arrow::Result<std::string> GcpBrokerCredentialProvider::GetToken() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    const auto now = clock_();
    // An expired token is useless to GCS, so only a still-valid one is served
    // or kept as a fallback.
    if (cached_.has_value() && cached_->expires_at > now) {
      // Fresh, or somebody else is already refreshing, or a refresh just
      // failed: serve the cached token and do not touch the Broker.
      if (cached_->expires_at - now > kRefreshOffset || fetch_in_flight_ || InFailureBackoff(now)) {
        return cached_->access_token;
      }
      // Inside the refresh window: this caller refreshes, everyone else keeps
      // getting the cached token meanwhile.
      break;
    }
    if (fetch_in_flight_) {
      // Single-flight: share the outcome of the fetch in progress instead of
      // adding another Broker call.
      cv_.wait(lock, [this] { return !fetch_in_flight_; });
      continue;
    }
    if (InFailureBackoff(now)) {
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(kFailureBackoff - (now - *last_failure_at_));
      return arrow::Status::IOError("GCP credential broker fetch suppressed for ", remaining.count(),
                                    " ms after a failure: ", last_failure_.message());
    }
    break;
  }

  // This thread owns the Broker call. Retries only apply before the provider
  // has ever obtained a token; refreshes stay single attempts.
  fetch_in_flight_ = true;
  const bool allow_retries = !cached_.has_value();
  lock.unlock();
  TimePoint expires_at;
  auto fetched = FetchTokenWithRetries(allow_retries, &expires_at);
  lock.lock();
  fetch_in_flight_ = false;
  if (fetched.ok()) {
    cached_ = CachedToken{std::move(fetched).ValueOrDie(), expires_at};
    last_failure_at_.reset();
    last_failure_ = arrow::Status::OK();
  } else {
    last_failure_at_ = clock_();
    last_failure_ = fetched.status();
  }
  cv_.notify_all();

  if (fetched.ok()) {
    LOG_STORAGE_DEBUG_ << "GCP credential broker refresh succeeded: endpoint=" << config_.endpoint
                       << ", bucket=" << config_.bucket << ", region=" << config_.region
                       << ", expires_at=" << Azure::DateTime(expires_at).ToString(Azure::DateTime::DateFormat::Rfc3339);
    return cached_->access_token;
  }

  const bool has_valid_token = cached_.has_value() && cached_->expires_at > clock_();
  LOG_STORAGE_WARNING_ << "GCP credential broker refresh failed: " << fetched.status().ToString()
                       << ", has_valid_token=" << std::boolalpha << has_valid_token << ", backoff_ms="
                       << std::chrono::duration_cast<std::chrono::milliseconds>(kFailureBackoff).count();
  if (has_valid_token) {
    return cached_->access_token;
  }
  return fetched.status();
}

std::optional<std::pair<std::string, std::string>> GcpBrokerCredentialProvider::AuthorizationHeader() {
  try {
    auto token = GetToken();
    if (!token.ok()) {
      LOG_STORAGE_WARNING_ << "GCP credential broker token unavailable: " << token.status().ToString()
                           << "; request will proceed without Authorization, server will reply 401/403";
      return std::nullopt;
    }
    return std::make_pair(std::string("Authorization"), "Bearer " + token.ValueOrDie());
  } catch (const std::exception& e) {
    LOG_STORAGE_WARNING_ << "GCP credential broker token unavailable: " << e.what();
  } catch (...) {
    LOG_STORAGE_WARNING_ << "GCP credential broker token unavailable: unknown exception";
  }
  return std::nullopt;
}

}  // namespace milvus_storage
