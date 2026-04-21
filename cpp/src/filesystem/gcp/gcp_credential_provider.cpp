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

#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <google/cloud/credentials.h>
#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/internal/unified_rest_credentials.h>
#include <google/cloud/options.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"

namespace milvus_storage {

namespace {

// GCP IAM caps impersonation token lifetime at 3600s (without an
// `iam.allowServiceAccountCredentialLifetimeExtension` org-policy override).
// Oversize values are rejected at `BuildGcpProviderFromConfig` rather than
// silently clamped so callers see a consistent answer with the Rust Lance
// path (`lance_bridgeimpl::GcpImpersonationConfig::extract`) and with the
// AWS `AssumeRoleConfig::parse` style.
constexpr int kMaxImpersonationTokenLifetime = 3600;
constexpr int kMinImpersonationTokenLifetime = 900;

std::shared_ptr<google::cloud::oauth2_internal::Credentials> MakeInternalCredentials(
    std::shared_ptr<google::cloud::Credentials> public_creds) {
  // MapCredentials returns an oauth2_internal::Credentials wrapping the public
  // credentials; the wrapper owns the state it needs and does its own token
  // caching + refresh via the usual google-cloud-cpp machinery.
  return google::cloud::rest_internal::MapCredentials(*public_creds);
}

// Shared base for both OAuth2-backed providers (VM IAM and Impersonation).
// Wraps a google-cloud-cpp oauth2_internal::Credentials and delegates to its
// built-in token caching + refresh logic.
class OAuth2BearerProvider : public GcpCredentialProvider {
  public:
  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override {
    auto header = google::cloud::oauth2_internal::AuthorizationHeader(*credentials_);
    if (!header.ok()) {
      // On token fetch failure, return nullopt. The factory will issue the
      // request without Authorization and the server will return 401/403,
      // surfacing the auth failure at the call site via the usual AWS SDK path.
      // Log the root cause here so operators can correlate the GCS 401/403
      // with the underlying OAuth2/IAM error (e.g. missing
      // roles/iam.serviceAccountTokenCreator, metadata server unreachable).
      LOG_STORAGE_WARNING_ << "GCP OAuth2 token fetch failed: " << header.status().message()
                           << "; request will proceed without Authorization, server will reply 401/403";
      return std::nullopt;
    }
    return *header;
  }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  protected:
  explicit OAuth2BearerProvider(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials)
      : credentials_(std::move(credentials)) {}

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
};

// VM / ADC identity: ComputeEngineCredentials on GCE, or application default
// credentials elsewhere (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth, etc).
class IamVmProvider : public OAuth2BearerProvider {
  public:
  IamVmProvider() : OAuth2BearerProvider(MakeInternalCredentials(google::cloud::MakeGoogleDefaultCredentials())) {}
};

// Service Account Impersonation: VM/ADC identity impersonates a target SA,
// obtaining short-lived access tokens via iamcredentials.generateAccessToken.
class IamImpersonateProvider : public OAuth2BearerProvider {
  public:
  IamImpersonateProvider(const std::string& target_service_account, int token_lifetime_seconds)
      : OAuth2BearerProvider(
            MakeInternalCredentials(MakeImpersonationPublicCreds(target_service_account, token_lifetime_seconds))) {}

  private:
  static std::shared_ptr<google::cloud::Credentials> MakeImpersonationPublicCreds(const std::string& target_sa,
                                                                                  int token_lifetime_seconds) {
    google::cloud::Options opts;
    if (token_lifetime_seconds > 0) {
      opts.set<google::cloud::AccessTokenLifetimeOption>(std::chrono::seconds(token_lifetime_seconds));
    }
    return google::cloud::MakeImpersonateServiceAccountCredentials(google::cloud::MakeGoogleDefaultCredentials(),
                                                                   target_sa, std::move(opts));
  }
};

// GCS HMAC ak/sk. Non-conditional requests use AWS SigV4 (handled upstream by
// the AWS SDK's signer); conditional writes (x-goog-if-generation-match) are
// re-signed with GOOG4-HMAC-SHA256 since GCS does not accept SigV4 for them.
class HmacProvider : public GcpCredentialProvider {
  public:
  HmacProvider(std::string access_key, std::string secret_key)
      : access_key_(std::move(access_key)), secret_key_(std::move(secret_key)) {}

  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override { return std::nullopt; }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>& request) override {
    // Only re-sign conditional writes; GCS rejects SigV4 Authorization for
    // those and needs GOOG4-HMAC-SHA256 instead.
    if (!request->HasHeader("x-goog-if-generation-match")) {
      return arrow::Status::OK();
    }

    // Strip AWS SigV4 artifacts that would otherwise confuse GCS.
    request->DeleteHeader("Authorization");
    request->DeleteHeader("x-amz-date");
    request->DeleteHeader("x-amz-content-sha256");
    request->DeleteHeader("x-amz-security-token");
    request->DeleteHeader("x-amz-api-version");

    // AWS uses x-amz-meta-*, GCS uses x-goog-meta-*.
    std::vector<std::pair<std::string, std::string>> meta_headers;
    std::vector<std::string> old_keys;
    for (const auto& [key, value] : request->GetHeaders()) {
      std::string lower_key = key;
      std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
      if (lower_key.find("x-amz-meta-") == 0) {
        meta_headers.emplace_back("x-goog-meta-" + key.substr(11), value);
        old_keys.push_back(key);
      }
    }
    for (const auto& k : old_keys) {
      request->DeleteHeader(k.c_str());
    }
    for (const auto& [k, v] : meta_headers) {
      request->SetHeaderValue(k, v);
    }

    if (!auth_signer::googv4::SignRequest(request, access_key_, secret_key_)) {
      return arrow::Status::ExecutionError("GOOG4-HMAC-SHA256 signing failed");
    }
    return arrow::Status::OK();
  }

  private:
  std::string access_key_;
  std::string secret_key_;
};

}  // namespace

arrow::Result<std::shared_ptr<GcpCredentialProvider>> BuildGcpProviderFromConfig(const ArrowFileSystemConfig& config) {
  if (config.use_iam && !config.gcp_target_service_account.empty()) {
    // Reject rather than clamp so this path and the Rust Lance path
    // (`GcpImpersonationConfig::extract`, `[900, 3600]`) reject identically;
    // otherwise `load_frequency=7200` works silently through the filesystem
    // producer but errors out of `open_dataset`.
    if (config.load_frequency < kMinImpersonationTokenLifetime ||
        config.load_frequency > kMaxImpersonationTokenLifetime) {
      return arrow::Status::Invalid("GCP impersonation requires load_frequency in [", kMinImpersonationTokenLifetime,
                                    ", ", kMaxImpersonationTokenLifetime, "], got ", config.load_frequency);
    }
    return std::make_shared<IamImpersonateProvider>(config.gcp_target_service_account, config.load_frequency);
  }
  if (config.use_iam) {
    return std::make_shared<IamVmProvider>();
  }
  if (!config.access_key_id.empty() && !config.access_key_value.empty()) {
    return std::make_shared<HmacProvider>(config.access_key_id, config.access_key_value);
  }
  return arrow::Status::Invalid(
      "GCP filesystem: no credentials configured. Set use_iam=true (VM default SA, or target-SA "
      "impersonation when gcp_target_service_account is also set), or provide both HMAC "
      "access_key_id and access_key_value. Got: use_iam=false, access_key_id=",
      config.access_key_id.empty() ? "(empty)" : "(set)",
      ", access_key_value=", config.access_key_value.empty() ? "(empty)" : "(set)",
      ", gcp_target_service_account=", config.gcp_target_service_account.empty() ? "(empty)" : "(set)");
}

}  // namespace milvus_storage
