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

#include "milvus-storage/filesystem/s3/provider/AwsCredentialBrokerProvider.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <thread>
#include <utility>

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/crt/Api.h>
#include <aws/crt/DateTime.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"

using ::milvus_storage::fs::internal::ToAwsString;

namespace milvus_storage {

static constexpr int kMaxInitialRetries = 3;

AwsCredentialBrokerProvider::AwsCredentialBrokerProvider(AwsCredentialBrokerConfig config, Clock clock)
    : config_(std::move(config)), clock_(clock ? std::move(clock) : Clock([] { return Aws::Utils::DateTime::Now(); })) {
  // The broker uses its own workload identity; do not probe local AWS metadata.
  Aws::Client::ClientConfiguration options(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  options.region = config_.region.c_str();
  options.connectTimeoutMs = config_.request_timeout_ms;
  options.requestTimeoutMs = config_.request_timeout_ms;
  options.httpRequestTimeoutMs = config_.request_timeout_ms;
  options.maxConnections = 1;
  client_ = Aws::Http::CreateHttpClient(options);
}

Aws::Auth::AWSCredentials AwsCredentialBrokerProvider::GetAWSCredentials() {
  auto result = GetCredentials();
  if (!result.ok()) {
    // AWS's provider interface cannot return a Status. Keep the broker's cause
    // in the log and follow the existing providers' empty-credentials convention.
    LOG_STORAGE_WARNING_ << result.status().ToString();
    return {};
  }
  return std::move(result).ValueUnsafe();
}

arrow::Result<Aws::Auth::AWSCredentials> AwsCredentialBrokerProvider::GetCredentials() {
  // Like the existing STS providers, serialize refresh and recheck the cache
  // under the same lock. All users of this provider share one in-flight fetch.
  std::lock_guard lock(mutex_);
  if (!credentials_.IsEmpty() && credentials_.GetExpiration().Millis() - clock_().Millis() > 60000) {
    return credentials_;
  }
  const bool had_cached_credentials = !credentials_.IsEmpty();
  auto fetched = FetchCredentials();
  auto retry_delay = std::chrono::milliseconds(100);
  // Match Azure SAS: only initial fetches retry; cached refreshes stay single
  // attempts, even after expiration. Each HTTP attempt gets its own timeout.
  for (int retries = 0; !had_cached_credentials && !fetched.ok() && retries < kMaxInitialRetries; ++retries) {
    LOG_STORAGE_WARNING_ << "AWS credential broker initial fetch failed: " << fetched.status().ToString()
                         << ", endpoint=" << config_.endpoint << ", retry=" << retries + 1 << "/" << kMaxInitialRetries
                         << ", retry_delay_ms=" << retry_delay.count();
    std::this_thread::sleep_for(retry_delay);
    fetched = FetchCredentials();
    retry_delay = std::min(retry_delay * 2, std::chrono::milliseconds(1000));
  }
  if (fetched.ok()) {
    credentials_ = std::move(fetched).ValueUnsafe();
    return credentials_;
  }
  if (!credentials_.IsEmpty()) {
    // Match Azure SAS: retain the last credentials even after expiration, and
    // retry refresh on the next request. The service decides whether to accept them.
    LOG_STORAGE_WARNING_ << fetched.status().ToString() << "; using cached AWS credentials, expired=" << std::boolalpha
                         << (credentials_.GetExpiration() <= clock_());
    return credentials_;
  }
  return fetched.status();
}

arrow::Result<Aws::Auth::AWSCredentials> AwsCredentialBrokerProvider::FetchCredentials() const {
  try {
    nlohmann::json payload = {{"csp", "aws"},
                              {"region", config_.region},
                              {"roleArn", config_.role_arn},
                              {"durationSeconds", config_.duration_seconds}};
    if (!config_.external_id.empty()) {
      payload["externalId"] = config_.external_id;
    }
    auto request = Aws::Http::CreateHttpRequest(ToAwsString(config_.endpoint), Aws::Http::HttpMethod::HTTP_POST,
                                                Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
    const auto serialized = payload.dump();
    auto body = Aws::MakeShared<Aws::StringStream>("AwsCredentialBroker");
    *body << serialized;
    request->SetContentType("application/json");
    request->SetContentLength(std::to_string(serialized.size()).c_str());
    request->AddContentBody(body);
    const auto response = client_->MakeRequest(request);
    if (!response) {
      return arrow::Status::IOError("AWS credential broker returned no HTTP response");
    }
    if (response->HasClientError()) {
      return arrow::Status::IOError("AWS credential broker request failed: ",
                                    response->GetClientErrorMessage().c_str());
    }
    const int status = static_cast<int>(response->GetResponseCode());
    const auto json = nlohmann::json::parse(response->GetResponseBody(), nullptr, false);
    if (status < 200 || status >= 300) {
      return arrow::Status::IOError(
          "AWS credential broker HTTP ", status,
          json.is_object() ? ": " + json.value("msg", std::string()) + "; " + json.value("detail", std::string()) : "");
    }
    if (json.is_discarded()) {
      return arrow::Status::IOError("AWS credential broker returned invalid JSON");
    }
    if (!json.at("success").get<bool>()) {
      return arrow::Status::IOError("AWS credential broker returned success=false: ", json.value("msg", std::string()),
                                    "; ", json.value("detail", std::string()));
    }
    const auto& values = json.at("credentials");
    const auto access_key = values.at("tempAk").get<std::string>();
    const auto secret_key = values.at("tempSk").get<std::string>();
    const auto session_token = values.at("sessionToken").get<std::string>();
    const auto expired_at = values.at("expiredAt").get<std::string>();
    if (access_key.empty() || secret_key.empty() || session_token.empty() || expired_at.empty()) {
      return arrow::Status::IOError("AWS credential broker requires non-empty tempAk, tempSk, sessionToken, expiredAt");
    }
    // CRT applies numeric timezone offsets that Aws::Utils::DateTime's parser ignores.
    const Aws::Crt::DateTime parsed(expired_at.c_str(), Aws::Crt::DateFormat::ISO_8601);
    if (!parsed) {
      return arrow::Status::IOError("AWS credential broker returned invalid expiredAt: ",
                                    Aws::Crt::ErrorDebugString(parsed.GetLastError()));
    }
    const Aws::Utils::DateTime expiration(parsed.UnderlyingTimestamp());
    // Check after HTTP completes: a slow fetch must not publish expired credentials.
    if (expiration <= clock_()) {
      return arrow::Status::IOError("AWS credential broker returned expired credentials");
    }
    return Aws::Auth::AWSCredentials(access_key.c_str(), secret_key.c_str(), session_token.c_str(), expiration);
  } catch (const nlohmann::json::exception& error) {
    return arrow::Status::IOError("AWS credential broker response schema is invalid: ", error.what());
  } catch (const std::exception& error) {
    return arrow::Status::IOError("AWS credential broker request failed: ", error.what());
  }
}

bool AwsCredentialBrokerProvider::Equals(const AwsCredentialBrokerProvider& other) const {
  return config_ == other.config_;
}

}  // namespace milvus_storage
