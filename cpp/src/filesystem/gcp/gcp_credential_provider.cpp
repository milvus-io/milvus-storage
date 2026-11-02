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
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <google/cloud/credentials.h>
#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/internal/unified_rest_credentials.h>
#include <google/cloud/options.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_broker_credential_provider.h"
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"

namespace milvus_storage {

namespace {

// Accepted range for load_frequency as the requested access-token lifetime
// of impersonated and Broker-minted tokens. GCP IAM generateAccessToken
// rejects lifetimes above 3600s unless the
// `iam.allowServiceAccountCredentialLifetimeExtension` org policy raises the
// cap, and tokens shorter than 900s would be refreshed more often than they
// are useful. Out-of-range values are rejected at `BuildGcpProviderFromConfig`
// rather than silently clamped, so the lifetime a filesystem actually requests
// is always the one its configuration states.
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
    // AuthorizationHeader is called for every GCS request. Confirm that the
    // provider can obtain a token once without logging every object operation.
    std::call_once(success_log_once_, [this] {
      LOG_STORAGE_DEBUG_ << "GCP OAuth2 token obtained successfully: credential_type=" << credential_type_;
    });
    return *header;
  }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return identity_; }

  protected:
  OAuth2BearerProvider(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
                       std::string credential_type,
                       std::string identity)
      : credentials_(std::move(credentials)),
        credential_type_(std::move(credential_type)),
        identity_(std::move(identity)) {}

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
  std::string credential_type_;
  std::string identity_;
  std::once_flag success_log_once_;
};

// VM / ADC identity: ComputeEngineCredentials on GCE, or application default
// credentials elsewhere (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth, etc).
class IamVmProvider : public OAuth2BearerProvider {
  public:
  IamVmProvider()
      : OAuth2BearerProvider(
            MakeInternalCredentials(google::cloud::MakeGoogleDefaultCredentials()), "application_default", "iam") {}
};

// Service Account Impersonation: VM/ADC identity impersonates a target SA,
// obtaining short-lived access tokens via iamcredentials.generateAccessToken.
class IamImpersonateProvider : public OAuth2BearerProvider {
  public:
  IamImpersonateProvider(const std::string& target_service_account, int token_lifetime_seconds)
      : OAuth2BearerProvider(
            MakeInternalCredentials(MakeImpersonationPublicCreds(target_service_account, token_lifetime_seconds)),
            "service_account_impersonation",
            "impersonate:" + target_service_account) {}

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

  bool IssuesBearerToken() const override { return false; }

  std::string Identity() const override { return "hmac:" + access_key_; }

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
  const bool has_hmac = !config.access_key_id.empty() && !config.access_key_value.empty();
  const bool use_broker = !config.use_iam && !config.credential_endpoint.empty();
  const bool has_target = !config.gcp_target_service_account.empty();
  if (has_hmac && (use_broker || has_target)) {
    // With Talon, the Broker fields describe the identity Talon workers use
    // and the HMAC keys stay the origin filesystem's own credential. Without
    // Talon the config asks for two identities at once.
    if (config.talon_mode == TalonMode::Disabled) {
      return arrow::Status::Invalid(
          "GCP filesystem: access_key_id/access_key_value cannot be combined with "
          "gcp_target_service_account or credential_endpoint; choose one credential mode");
    }
    return std::make_shared<HmacProvider>(config.access_key_id, config.access_key_value);
  }
  if (use_broker || has_target) {
    if (use_broker && !has_target) {
      return arrow::Status::Invalid("GCP credential broker requires fs.gcp_target_service_account");
    }
    // Reject rather than clamp: `load_frequency=7200` must fail here instead
    // of silently requesting a 3600s token under a configuration that claims
    // otherwise.
    if (config.load_frequency < kMinImpersonationTokenLifetime ||
        config.load_frequency > kMaxImpersonationTokenLifetime) {
      return arrow::Status::Invalid("GCP impersonation requires load_frequency in [", kMinImpersonationTokenLifetime,
                                    ", ", kMaxImpersonationTokenLifetime, "], got ", config.load_frequency);
    }
    // As for Azure, use_iam=true selects this process's own identity, so the
    // Broker is used only without it.
    if (use_broker) {
      return std::make_shared<GcpBrokerCredentialProvider>(GcpBrokerConfig{
          .endpoint = config.credential_endpoint,
          .region = config.region,
          .bucket = config.bucket_name,
          .target_service_account = config.gcp_target_service_account,
          .duration_seconds = config.load_frequency,
          .request_timeout_ms = config.request_timeout_ms,
      });
    }
    // The target alone selects impersonation; use_iam=true is accepted but
    // not required.
    return std::make_shared<IamImpersonateProvider>(config.gcp_target_service_account, config.load_frequency);
  }
  if (config.use_iam) {
    return std::make_shared<IamVmProvider>();
  }
  if (has_hmac) {
    return std::make_shared<HmacProvider>(config.access_key_id, config.access_key_value);
  }
  return arrow::Status::Invalid(
      "GCP filesystem: no credentials configured. Set use_iam=true (VM default SA), "
      "gcp_target_service_account (target-SA impersonation, through the credential broker when "
      "credential_endpoint is also set), or provide both HMAC access_key_id and access_key_value. "
      "Got: use_iam=false, access_key_id=",
      config.access_key_id.empty() ? "(empty)" : "(set)",
      ", access_key_value=", config.access_key_value.empty() ? "(empty)" : "(set)");
}

}  // namespace milvus_storage
