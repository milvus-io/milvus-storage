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

#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <new>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <azure/core/http/http.hpp>
#include <azure/core/url.hpp>

namespace milvus_storage::fs {

namespace {
ExtendStatusCode SourceStatusCode(LoonSourceCredentialStatus status) {
  switch (status) {
    case LOON_SOURCE_CREDENTIAL_DENIED:
      return ExtendStatusCode::SourceAuthDenied;
    case LOON_SOURCE_CREDENTIAL_EXPIRED:
      return ExtendStatusCode::SourceAuthExpired;
    case LOON_SOURCE_CREDENTIAL_TRANSIENT:
      return ExtendStatusCode::SourceAuthTransient;
    case LOON_SOURCE_CREDENTIAL_CANCELLED:
      return ExtendStatusCode::SourceAuthCancelled;
    default:
      return ExtendStatusCode::SourceAuthInvalidResponse;
  }
}
constexpr auto kFailureReason = "Loon source policy failure";
constexpr auto kFailureHeader = "x-loon-source-failure";

std::unique_ptr<Azure::Core::Http::RawResponse> FailureResponse(const arrow::Status& status) {
  // 400 prevents the SDK from multiplying the policy's bounded acquisition
  // retries. This is a local carrier, not an issuer or Azure HTTP response.
  auto response = std::make_unique<Azure::Core::Http::RawResponse>(1, 1, Azure::Core::Http::HttpStatusCode::BadRequest,
                                                                   kFailureReason);
  std::string category = status.IsOutOfMemory() ? "out-of-memory" : "internal";
  if (const auto detail = std::dynamic_pointer_cast<SourceCredentialStatusDetail>(status.detail())) {
    switch (detail->status()) {
      case LOON_SOURCE_CREDENTIAL_DENIED:
        category = "denied";
        break;
      case LOON_SOURCE_CREDENTIAL_EXPIRED:
        category = "expired";
        break;
      case LOON_SOURCE_CREDENTIAL_TRANSIENT:
        category = "transient";
        break;
      case LOON_SOURCE_CREDENTIAL_CANCELLED:
        category = "cancelled";
        break;
      default:
        category = "invalid-response";
        break;
    }
  }
  response->SetHeader(kFailureHeader, category);
  response->SetHeader("x-ms-error-code", "LoonSourcePolicyFailure");
  response->SetHeader("Content-Length", "0");
  return response;
}
}  // namespace

SourceCredentialStatusDetail::SourceCredentialStatusDetail(LoonSourceCredentialStatus status)
    : ExtendStatusDetail(SourceStatusCode(status)), status_(status) {}

std::string SourceCredentialStatusDetail::ToString() const {
  switch (status_) {
    case LOON_SOURCE_CREDENTIAL_DENIED:
      return "source authorization denied";
    case LOON_SOURCE_CREDENTIAL_EXPIRED:
      return "source authorization expired";
    case LOON_SOURCE_CREDENTIAL_TRANSIENT:
      return "source credential provider temporarily unavailable";
    case LOON_SOURCE_CREDENTIAL_CANCELLED:
      return "source operation cancelled";
    default:
      return "invalid source credential response";
  }
}

arrow::Status SourceCredentialArrowStatus(LoonSourceCredentialStatus status) {
  if (status == LOON_SOURCE_CREDENTIAL_OK)
    return arrow::Status::OK();
  auto detail = std::make_shared<SourceCredentialStatusDetail>(status);
  auto result = status == LOON_SOURCE_CREDENTIAL_CANCELLED ? arrow::Status::Cancelled(detail->ToString())
                                                           : arrow::Status::IOError(detail->ToString());
  return result.WithDetail(std::move(detail));
}

std::optional<arrow::Status> SourcePolicyResponseStatus(const Azure::Core::Http::RawResponse& response) {
  if (response.GetStatusCode() != Azure::Core::Http::HttpStatusCode::BadRequest ||
      response.GetReasonPhrase() != kFailureReason) {
    return std::nullopt;
  }
  const auto found = response.GetHeaders().find(kFailureHeader);
  if (found != response.GetHeaders().end()) {
    const auto& category = found->second;
    if (category == "denied")
      return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_DENIED);
    if (category == "expired")
      return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_EXPIRED);
    if (category == "transient")
      return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_TRANSIENT);
    if (category == "cancelled")
      return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_CANCELLED);
    if (category == "invalid-response")
      return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
    if (category == "out-of-memory")
      return arrow::Status::OutOfMemory("Source credential acquisition ran out of memory");
  }
  return arrow::Status::UnknownError("Source credential acquisition failed internally");
}

struct SourceSasTokenPolicy::SharedState {
  struct Lease {
    std::string token;
    int64_t expires_at;
    int64_t refresh_at;
  };
  struct RefreshAttempt {
    std::condition_variable completed;
    bool finished = false;
    bool retry_after_cancellation = false;
    arrow::Status failure;
  };
  struct Entry {
    std::mutex mutex;
    std::shared_ptr<RefreshAttempt> refresh;
    std::optional<Lease> lease;
  };
  explicit SharedState(SourceReadContextPtr source_context) : context(std::move(source_context)) {}
  const SourceReadContextPtr context;
  std::mutex mutex;
  std::map<std::pair<size_t, LoonSourceReadOperation>, std::shared_ptr<Entry>> entries;
};

SourceSasTokenPolicy::SourceSasTokenPolicy(SourceReadContextPtr context)
    : state_(std::make_shared<SharedState>(std::move(context))) {}

SourceSasTokenPolicy::~SourceSasTokenPolicy() = default;

std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy> SourceSasTokenPolicy::Clone() const {
  return std::make_unique<SourceSasTokenPolicy>(*this);
}

arrow::Status SourceSasTokenPolicy::CheckState(const Azure::Core::Context& context) const {
  if (!state_->context)
    return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  ARROW_RETURN_NOT_OK(SourceCredentialArrowStatus(state_->context->CheckState()));
  return SourceCredentialArrowStatus(context.IsCancelled() ? LOON_SOURCE_CREDENTIAL_CANCELLED
                                                           : LOON_SOURCE_CREDENTIAL_OK);
}

arrow::Result<std::string> SourceSasTokenPolicy::GetSasToken(const SourceReadTarget& target,
                                                             const Azure::Core::Context& context) const {
  std::shared_ptr<SharedState::Entry> entry;
  {
    std::lock_guard lock(state_->mutex);
    auto& slot = state_->entries[{target.root_index, target.operation}];
    if (!slot)
      slot = std::make_shared<SharedState::Entry>();
    entry = slot;
  }
  std::shared_ptr<SharedState::RefreshAttempt> refresh;
  for (;;) {
    std::unique_lock lock(entry->mutex);
    ARROW_RETURN_NOT_OK(CheckState(context));
    const auto now = state_->context->NowUnixMillis();
    if (entry->lease && now < entry->lease->refresh_at)
      return entry->lease->token;
    if (!entry->refresh) {
      refresh = std::make_shared<SharedState::RefreshAttempt>();
      entry->refresh = refresh;
      break;
    }
    if (entry->lease && now < entry->lease->expires_at)
      return entry->lease->token;
    auto waiting = entry->refresh;
    while (!waiting->finished) {
      ARROW_RETURN_NOT_OK(CheckState(context));
      waiting->completed.wait_for(lock, std::chrono::milliseconds(10));
    }
    ARROW_RETURN_NOT_OK(CheckState(context));
    if (!waiting->retry_after_cancellation && !waiting->failure.ok())
      return waiting->failure;
    // Recheck the current cache and clock after success. An SDK-local leader
    // cancellation also permits a live waiter to elect another acquisition.
  }

  auto complete = [&](arrow::Status failure, bool retry_after_cancellation) {
    std::lock_guard lock(entry->mutex);
    refresh->failure = std::move(failure);
    refresh->retry_after_cancellation = retry_after_cancellation;
    refresh->finished = true;
    entry->refresh.reset();
    refresh->completed.notify_all();
  };
  auto fetch = [&]() -> arrow::Result<std::string> {
    for (int attempt = 0; attempt < 4; ++attempt) {
      ARROW_RETURN_NOT_OK(CheckState(context));
      auto response = state_->context->Fetch(target.resource_uri, target.operation);
      ARROW_RETURN_NOT_OK(CheckState(context));
      {
        std::lock_guard lock(entry->mutex);
        const auto now = state_->context->NowUnixMillis();
        if (response.status == LOON_SOURCE_CREDENTIAL_OK) {
          if (response.expires_at_unix_ms <= now) {
            entry->lease.reset();
            return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_EXPIRED);
          }
          const auto margin = std::min<int64_t>(60000, (response.expires_at_unix_ms - now) / 2);
          entry->lease = SharedState::Lease{std::move(response.sas_token), response.expires_at_unix_ms,
                                            response.expires_at_unix_ms - margin};
          return entry->lease->token;
        }
        if (response.status != LOON_SOURCE_CREDENTIAL_TRANSIENT) {
          entry->lease.reset();
          return SourceCredentialArrowStatus(response.status);
        }
        if (entry->lease && now < entry->lease->expires_at)
          return entry->lease->token;
        entry->lease.reset();
      }
      if (attempt < 3) {
        // Bound retry backoff while allowing a waiting/cancelled SDK operation to
        // stop promptly. Provider callbacks themselves must honor their deadline.
        for (int tick = 0; tick < (10 << attempt); ++tick) {
          ARROW_RETURN_NOT_OK(CheckState(context));
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    }
    return SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_TRANSIENT);
  };
  // Every completion, including unexpected local failures, releases waiters.
  // Do not turn allocation/internal errors into retryable provider failures or
  // expose arbitrary exception text (which may contain credentials).
  auto result = [&]() -> arrow::Result<std::string> {
    try {
      return fetch();
    } catch (const std::bad_alloc&) {
      return arrow::Status::OutOfMemory("Source credential acquisition ran out of memory");
    } catch (...) {
      return arrow::Status::UnknownError("Source credential acquisition failed internally");
    }
  }();
  const auto detail = std::dynamic_pointer_cast<SourceCredentialStatusDetail>(result.status().detail());
  const bool local_cancel = detail && detail->status() == LOON_SOURCE_CREDENTIAL_CANCELLED && context.IsCancelled();
  complete(result.status(), local_cancel);
  return result;
}

std::unique_ptr<Azure::Core::Http::RawResponse> SourceSasTokenPolicy::Send(
    Azure::Core::Http::Request& request,
    Azure::Core::Http::Policies::NextHttpPolicy next_policy,
    const Azure::Core::Context& context) const {
  auto status = CheckState(context);
  if (!status.ok())
    return FailureResponse(status);
  const auto target =
      state_->context->ResolveHttpRequest(request.GetMethod().ToString(), request.GetUrl().GetAbsoluteUrl());
  if (target.status != LOON_SOURCE_CREDENTIAL_OK)
    return FailureResponse(SourceCredentialArrowStatus(target.status));
  auto token = GetSasToken(target, context);
  if (!token.ok())
    return FailureResponse(token.status());
  status = CheckState(context);
  if (!status.ok())
    return FailureResponse(status);
  const auto parameters = Azure::Core::Url("?" + *token).GetQueryParameters();
  // Restore the unsigned request even on exception. Retrying/reusing a request
  // must never retain a previous lease or expose it through request diagnostics.
  struct RestoreUrl {
    Azure::Core::Url& url;
    Azure::Core::Url original;
    ~RestoreUrl() { url = std::move(original); }
  } restore{request.GetUrl(), request.GetUrl()};
  for (const auto& [key, value] : parameters) {
    if (request.GetUrl().GetQueryParameters().contains(key))
      return FailureResponse(SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE));
    request.GetUrl().AppendQueryParameter(key, value);
  }
  auto response = next_policy.Send(request, context);
  status = CheckState(context);
  if (!status.ok())
    return FailureResponse(status);
  return response;
}

}  // namespace milvus_storage::fs
