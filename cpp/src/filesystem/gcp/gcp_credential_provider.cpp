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
#include <cctype>
#include <chrono>
#include <cstdint>
#include <set>
#include <string_view>
#include <tuple>
#include <aws/core/client/AWSClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <folly/Synchronized.h>
#include <smithy/interceptor/Interceptor.h>
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
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
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

using EndpointKey = std::tuple<Aws::Http::Scheme, uint16_t, std::string>;

auto& TrustedEndpoints() {
  // Only primary GCP filesystems add endpoints. Credentials stay on their own clients.
  static folly::Synchronized<std::set<EndpointKey>> endpoints;
  return endpoints;
}

EndpointKey EndpointFromUri(const Aws::Http::URI& uri) {
  std::string host = uri.GetAuthority().c_str();
  std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) { return std::tolower(c); });
  return {uri.GetScheme(), uri.GetPort(), std::move(host)};
}

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
  protected:
  OAuth2BearerProvider(const ArrowFileSystemConfig& config,
                       std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
                       std::string credential_type)
      : GcpCredentialProvider(config),
        credentials_(std::move(credentials)),
        credential_type_(std::move(credential_type)) {}

  private:
  arrow::Result<bool> AuthorizeRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request) override {
    const auto header = google::cloud::oauth2_internal::AuthorizationHeader(*credentials_);
    if (!header.ok()) {
      const auto& error = header.status();
      if (error.code() == google::cloud::StatusCode::kPermissionDenied ||
          error.code() == google::cloud::StatusCode::kUnauthenticated) {
        return MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, error.message());
      }
      return arrow::Status::IOError("GCP OAuth2 token fetch failed: ", error);
    }
    request->SetHeaderValue(header->first.c_str(), header->second.c_str());
    std::call_once(success_log_once_, [this] {
      LOG_STORAGE_DEBUG_ << "GCP OAuth2 token obtained successfully: credential_type=" << credential_type_;
    });
    return true;
  }

  const std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
  const std::string credential_type_;
  std::once_flag success_log_once_;
};

// VM / ADC identity: ComputeEngineCredentials on GCE, or application default
// credentials elsewhere (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth, etc).
class IamVmProvider : public OAuth2BearerProvider {
  public:
  explicit IamVmProvider(const ArrowFileSystemConfig& config)
      : OAuth2BearerProvider(
            config, MakeInternalCredentials(google::cloud::MakeGoogleDefaultCredentials()), "application_default") {}
};

// Service Account Impersonation: VM/ADC identity impersonates a target SA,
// obtaining short-lived access tokens via iamcredentials.generateAccessToken.
class IamImpersonateProvider : public OAuth2BearerProvider {
  public:
  explicit IamImpersonateProvider(const ArrowFileSystemConfig& config)
      : OAuth2BearerProvider(config,
                             MakeInternalCredentials(MakeImpersonationPublicCreds(config.gcp_target_service_account,
                                                                                  config.load_frequency)),
                             "service_account_impersonation") {}

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

// GCS HMAC keys are currently used only for testing. Non-conditional requests use AWS SigV4 (handled upstream by
// the AWS SDK's signer); conditional writes (x-goog-if-generation-match) are
// re-signed with GOOG4-HMAC-SHA256 since GCS does not accept SigV4 for them.
class HmacProvider : public GcpCredentialProvider {
  public:
  explicit HmacProvider(const ArrowFileSystemConfig& config)
      : GcpCredentialProvider(config), access_key_(config.access_key_id), secret_key_(config.access_key_value) {}

  private:
  Aws::Auth::AWSCredentials GetAWSCredentials() override { return {access_key_.c_str(), secret_key_.c_str()}; }

  arrow::Result<bool> AuthorizeRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request) override {
    // Only re-sign conditional writes; GCS rejects SigV4 Authorization for
    // those and needs GOOG4-HMAC-SHA256 instead.
    if (!request->HasHeader("x-goog-if-generation-match")) {
      return false;
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
    return true;
  }

  const std::string access_key_;
  const std::string secret_key_;
};

}  // namespace

GcpCredentialProvider::GcpCredentialProvider(const ArrowFileSystemConfig& config)
    : endpoint_(StorageUri::BuildEndpointUrl(config.address, config.use_ssl).c_str()),
      bucket_(config.bucket_name),
      virtual_host_(Aws::String(bucket_.c_str()) + "." + endpoint_.GetAuthority()),
      config_key_(config.storage_type == "remote" ? config.GetCacheKey() : "") {}

void GcpCredentialProvider::TrustEndpoint(const ArrowFileSystemConfig& config) {
  const Aws::Http::URI uri(StorageUri::BuildEndpointUrl(config.address, config.use_ssl).c_str());
  TrustedEndpoints().wlock()->insert(EndpointFromUri(uri));
}

bool GcpCredentialProvider::MayCarryBearerToken(const Aws::Http::URI& uri) {
  auto endpoint = EndpointFromUri(uri);
  auto& [scheme, port, host] = endpoint;
  constexpr std::string_view suffix = ".googleapis.com";
  if (scheme == Aws::Http::Scheme::HTTPS && host.size() > suffix.size() && host.ends_with(suffix)) {
    return true;
  }
  const auto trusted = TrustedEndpoints().rlock();
  if (trusted->contains(endpoint)) {
    return true;
  }
  // Virtual-host-style requests put the bucket in the first DNS label.
  const auto dot = host.find('.');
  if (dot == std::string::npos || dot == 0) {
    return false;
  }
  host.erase(0, dot + 1);
  return trusted->contains(endpoint);
}

arrow::Result<bool> GcpCredentialProvider::AuthorizeScopedRequest(
    const std::shared_ptr<Aws::Http::HttpRequest>& request, bool bearer) try {
  const auto& uri = request->GetUri();
  const auto& host = uri.GetAuthority();
  if (bearer || !bucket_.empty()) {
    const auto path = uri.GetPath();
    std::string_view bucket(path.data(), path.size());
    if (bucket.starts_with('/'))
      bucket.remove_prefix(1);
    const bool path_style = Aws::Utils::StringUtils::CaselessCompare(host.c_str(), endpoint_.GetAuthority().c_str()) &&
                            bucket.substr(0, bucket.find('/')) == bucket_;
    if (bucket_.empty() || endpoint_.GetScheme() != uri.GetScheme() || endpoint_.GetPort() != uri.GetPort() ||
        (!path_style && !Aws::Utils::StringUtils::CaselessCompare(host.c_str(), virtual_host_.c_str()))) {
      return arrow::Status::Invalid("GCP request is outside the filesystem endpoint and bucket");
    }
  }
  if (bearer && !MayCarryBearerToken(uri)) {
    std::call_once(untrusted_warning_, [&] {
      LOG_STORAGE_WARNING_ << "Withholding GCP bearer token from untrusted endpoint " << host << ":" << uri.GetPort();
    });
    request->DeleteHeader("Authorization");
    return false;
  }
  if (!bearer)
    return AuthorizeRequest(request);
  // Token failures leave the storage service to decide whether anonymous access is allowed.
  try {
    auto result = AuthorizeRequest(request);
    if (result.ok())
      return result;
    LOG_STORAGE_WARNING_ << "GCP bearer token unavailable: " << result.status().ToString();
  } catch (const std::exception& e) {
    LOG_STORAGE_WARNING_ << "GCP bearer token unavailable: " << e.what();
  } catch (...) {
    LOG_STORAGE_WARNING_ << "GCP bearer token unavailable: unknown exception";
  }
  request->DeleteHeader("Authorization");
  return false;
} catch (const std::bad_alloc& e) {
  return arrow::Status::OutOfMemory(e.what());
} catch (const std::exception& e) {
  return arrow::Status::IOError("GCP request authorization failed: ", e.what());
} catch (...) {
  return arrow::Status::UnknownError("GCP request authorization failed");
}

arrow::Result<bool> GcpCredentialProvider::AuthorizeRead(
    const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& credentials,
    Aws::AmazonWebServiceRequest& model,
    Aws::Endpoint::ResolveEndpointOutcome endpoint,
    const std::string& key,
    Aws::Http::HttpMethod method) {
  const auto provider = std::dynamic_pointer_cast<GcpCredentialProvider>(credentials);
  if (!provider)
    return false;
  if (!endpoint.IsSuccess())
    return arrow::Status::Invalid(endpoint.GetError().GetMessage().c_str());
  if (!key.empty())
    endpoint.GetResult().AddPathSegments(key.c_str());
  auto request = std::make_shared<Aws::Http::Standard::StandardHttpRequest>(endpoint.GetResult().GetURI(), method);
  ARROW_ASSIGN_OR_RAISE(const bool authorized, provider->AuthorizeScopedRequest(request, true));
  if (authorized) {
    model.SetAdditionalCustomHeaderValue("Authorization", request->GetAuthorization());
  } else {
    // CRT applies model headers after the global HTTP factory, so an empty value clears the factory's bearer.
    model.SetAdditionalCustomHeaderValue("Authorization", "");
  }
  return authorized;
}

// The SDK invokes this after SigV4 on each modeled request attempt.
class GcpCredentialProvider::RequestInterceptor final : public smithy::interceptor::Interceptor {
  public:
  RequestInterceptor(std::shared_ptr<GcpCredentialProvider> provider, bool bearer)
      : provider_(std::move(provider)), bearer_(bearer) {}
  ModifyRequestOutcome ModifyBeforeSigning(smithy::interceptor::InterceptorContext& context) override {
    return context.GetTransmitRequest();
  }
  ModifyRequestOutcome ModifyBeforeTransmit(smithy::interceptor::InterceptorContext& context) override {
    const auto request = context.GetTransmitRequest();
    const auto result = provider_->AuthorizeScopedRequest(request, bearer_);
    if (!result.ok()) {
      Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::UNKNOWN, "SignatureFailed",
                                                           result.status().message().c_str(), false);
      error.SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
      return error;
    }
    return request;
  }
  ModifyResponseOutcome ModifyBeforeDeserialization(smithy::interceptor::InterceptorContext& context) override {
    return context.GetTransmitResponse();
  }

  private:
  const std::shared_ptr<GcpCredentialProvider> provider_;
  const bool bearer_;
};

bool GcpCredentialProvider::ConfigS3HttpClient(const S3Options& options, Aws::Client::AWSClient& client) {
  if (options.cloud_provider != kCloudProviderGCP)
    return false;
  auto provider = std::dynamic_pointer_cast<GcpCredentialProvider>(options.credentials_provider);
  if (!provider) {
    // Raw GCP S3Options with explicit HMAC keys must also sign conditional PUTs.
    const auto keys = std::dynamic_pointer_cast<Aws::Auth::SimpleAWSCredentialsProvider>(options.credentials_provider);
    if (!keys) {
      return false;
    }
    const auto credentials = keys->GetAWSCredentials();
    if (credentials.IsEmpty())
      return false;
    ArrowFileSystemConfig config;
    config.address = options.endpoint_override;
    config.use_ssl = options.scheme == "https";
    config.access_key_id = credentials.GetAWSAccessKeyId().c_str();
    config.access_key_value = credentials.GetAWSSecretKey().c_str();
    provider = std::make_shared<HmacProvider>(config);
  }
  const bool bearer = provider->GetAWSCredentials().IsEmpty();
  client.AddRequestInterceptor(std::make_shared<RequestInterceptor>(std::move(provider), bearer));
  return bearer;
}

arrow::Result<std::shared_ptr<GcpCredentialProvider>> BuildGcpProviderFromConfig(const ArrowFileSystemConfig& config) {
  const bool has_hmac = !config.access_key_id.empty() && !config.access_key_value.empty();
  const bool use_broker = !config.credential_endpoint.empty();
  const bool has_target = !config.gcp_target_service_account.empty();
  if (has_hmac && (use_broker || has_target) && !(use_broker && config.use_iam)) {
    // With Talon, the Broker fields describe the identity Talon workers use
    // and the HMAC keys stay the origin filesystem's own credential. Without
    // Talon the config asks for two identities at once.
    if (config.talon_mode == TalonMode::Disabled) {
      return arrow::Status::Invalid(
          "GCP filesystem: access_key_id/access_key_value cannot be combined with "
          "gcp_target_service_account or credential_endpoint; choose one credential mode");
    }
    return std::make_shared<HmacProvider>(config);
  } else if (use_broker || has_target) {
    if (use_broker && !config.use_iam && !has_target) {
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
    // Match AWS: the endpoint selects the Broker; use_iam selects its own
    // identity instead of a target service account.
    if (use_broker) {
      return std::make_shared<GcpBrokerCredentialProvider>(config);
    }
    return std::make_shared<IamImpersonateProvider>(config);
  } else if (config.use_iam) {
    return std::make_shared<IamVmProvider>(config);
  } else if (has_hmac) {
    return std::make_shared<HmacProvider>(config);
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
