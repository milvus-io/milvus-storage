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

#include "milvus-storage/filesystem/gcp/refreshing_credentials.h"

#include <utility>
#include <google/cloud/internal/oauth2_credential_constants.h>
#include <google/cloud/internal/oauth2_error_credentials.h>
#include <google/cloud/internal/oauth2_google_credentials.h>
#include <google/cloud/internal/oauth2_minimal_iam_credentials_rest.h>
#include <google/cloud/internal/oauth2_service_account_credentials.h>
#include <google/cloud/internal/rest_client.h>
#include <google/cloud/storage/oauth2/service_account_credentials.h>
#include "milvus-storage/filesystem/bearer_token.h"

namespace milvus_storage {
namespace {
namespace gc = google::cloud;
namespace oauth = google::cloud::oauth2_internal;

oauth::HttpClientFactory HttpFactory() {
  return [](const gc::Options& options) { return gc::rest_internal::MakeDefaultRestClient("", options); };
}

class ImpersonatedCredentials final : public oauth::Credentials {
  public:
  ImpersonatedCredentials(const std::string& target, int lifetime_seconds)
      : stub_(oauth::MakeMinimalIamCredentialsRestStub(MakeRefreshingGcpDefaultCredentials(), {}, HttpFactory())),
        request_{target, std::chrono::seconds(lifetime_seconds), {oauth::GoogleOAuthScopeCloudPlatform()}, {}} {}
  gc::StatusOr<gc::AccessToken> GetToken(std::chrono::system_clock::time_point) override {
    return stub_->GenerateAccessToken(request_);
  }
  gc::StatusOr<std::string> universe_domain(const gc::Options& options) const override {
    return stub_->universe_domain(options);
  }

  private:
  std::shared_ptr<oauth::MinimalIamCredentialsRest> stub_;
  oauth::GenerateAccessTokenRequest request_;
};
}  // namespace

RefreshingGcpCredentials::RefreshingGcpCredentials(std::shared_ptr<oauth::Credentials> delegate, Clock clock)
    : delegate_(std::move(delegate)),
      clock_(clock ? std::move(clock) : Clock([] { return std::chrono::system_clock::now(); })) {}

gc::StatusOr<gc::AccessToken> RefreshingGcpCredentials::GetToken(TimePoint) {
  std::lock_guard lock(mutex_);
  const auto now = clock_();
  if (IsValidBearerToken(cached_.token) && cached_.expiration - now > oauth::GoogleOAuthAccessTokenExpirationSlack()) {
    return cached_;
  }
  gc::Status failure(gc::StatusCode::kUnavailable, "GCP credential refresh failed");
  try {
    auto candidate = delegate_->GetToken(now);
    if (candidate.ok()) {
      if (IsValidBearerToken(candidate->token) && candidate->expiration > clock_()) {
        cached_ = *std::move(candidate);
        return cached_;
      }
      failure = gc::Status(gc::StatusCode::kUnavailable, "GCP credential refresh returned an invalid or expired token");
    } else {
      // Keep the SDK status code without exposing response/token material in
      // SDK parsing errors. The caller's existing status handling is preserved.
      failure = gc::Status(candidate.status().code(), "GCP credential refresh failed");
    }
  } catch (...) {
    // Some SDK token parsers throw instead of returning Status. Treat both
    // paths identically and never unwind through AWS's no-exceptions callback.
  }
  if (IsValidBearerToken(cached_.token) && cached_.expiration > clock_()) {
    return cached_;
  }
  return failure;
}

gc::StatusOr<std::vector<std::uint8_t>> RefreshingGcpCredentials::SignBlob(const absl::optional<std::string>& account,
                                                                           const std::string& value) const {
  return delegate_->SignBlob(account, value);
}
std::string RefreshingGcpCredentials::AccountEmail() const { return delegate_->AccountEmail(); }
std::string RefreshingGcpCredentials::KeyId() const { return delegate_->KeyId(); }
gc::StatusOr<std::string> RefreshingGcpCredentials::universe_domain() const { return delegate_->universe_domain(); }
gc::StatusOr<std::string> RefreshingGcpCredentials::universe_domain(const gc::Options& options) const {
  return delegate_->universe_domain(options);
}
gc::StatusOr<std::string> RefreshingGcpCredentials::project_id() const { return delegate_->project_id(); }
gc::StatusOr<std::string> RefreshingGcpCredentials::project_id(const gc::Options& options) const {
  return delegate_->project_id(options);
}

std::shared_ptr<oauth::Credentials> MakeRefreshingGcpDefaultCredentials() {
  // Select ADC exactly once. Reconstructing MapCredentials after a bad token
  // could silently switch identity if the environment or ADC file changed.
  auto raw = oauth::GoogleDefaultCredentials({}, HttpFactory());
  if (!raw.ok()) {
    return std::make_shared<oauth::ErrorCredentials>(raw.status());
  }
  return std::make_shared<RefreshingGcpCredentials>(*std::move(raw));
}

std::shared_ptr<oauth::Credentials> MakeRefreshingGcpImpersonatedCredentials(const std::string& target,
                                                                             int lifetime_seconds) {
  return std::make_shared<RefreshingGcpCredentials>(
      std::make_shared<ImpersonatedCredentials>(target, lifetime_seconds));
}

std::shared_ptr<oauth::Credentials> MakeRefreshingGcpServiceAccountCredentials(const std::string& json) {
  // Match the GCS-specific mapping, which disables self-signed JWTs when
  // explicit scopes make them unsuitable for Cloud Storage.
  auto info = gc::storage::oauth2::ParseServiceAccountCredentials(json, "memory");
  if (!info.ok()) {
    return std::make_shared<oauth::ErrorCredentials>(info.status());
  }
  return std::make_shared<RefreshingGcpCredentials>(std::make_shared<oauth::ServiceAccountCredentials>(
      gc::storage::internal::MapServiceAccountCredentialsInfo(*std::move(info)), gc::Options{}, HttpFactory()));
}
}  // namespace milvus_storage
