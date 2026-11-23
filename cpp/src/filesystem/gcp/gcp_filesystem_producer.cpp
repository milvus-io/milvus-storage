// Copyright 2024 Zilliz
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

#include "milvus-storage/filesystem/gcp/gcp_filesystem_producer.h"
#include "filesystem/gcp/gcp_filesystem_producer_internal.h"

#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <aws/core/Aws.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>

#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include <fmt/format.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/tls_http_client.h"

namespace milvus_storage {

namespace {

constexpr const char* kGoogleClientFactoryAllocationTag = "GoogleHttpClientFactory";

const std::unordered_map<std::string, S3LogLevel> kLogLevelMap = {
    {"off", S3LogLevel::Off},   {"fatal", S3LogLevel::Fatal}, {"error", S3LogLevel::Error}, {"warn", S3LogLevel::Warn},
    {"info", S3LogLevel::Info}, {"debug", S3LogLevel::Debug}, {"trace", S3LogLevel::Trace}};

// The provider that GcpFileSystemProducer::Make() binds to the SDK clients it
// builds. The AWS SDK builds each SDK client's HttpClient through the global
// factory while constructing the client, and S3FileSystem::Make constructs its
// clients on the calling thread, so a thread-local slot reaches exactly them.
struct PendingClientCredentials {
  std::shared_ptr<GcpCredentialProvider> provider;
  bool bound = false;
};

thread_local PendingClientCredentials* pending_client_credentials = nullptr;

class ScopedClientCredentials {
  public:
  explicit ScopedClientCredentials(std::shared_ptr<GcpCredentialProvider> provider)
      : pending_{std::move(provider)}, previous_(std::exchange(pending_client_credentials, &pending_)) {}
  ~ScopedClientCredentials() { pending_client_credentials = previous_; }

  ScopedClientCredentials(const ScopedClientCredentials&) = delete;
  ScopedClientCredentials& operator=(const ScopedClientCredentials&) = delete;

  // Whether the GCP factory built a client in this scope. It does not when
  // code outside milvus-storage initialized the AWS SDK with its own factory.
  [[nodiscard]] bool bound() const { return pending_.bound; }

  private:
  PendingClientCredentials pending_;
  PendingClientCredentials* previous_;
};

void WarnUntrustedEndpointOnce(const Aws::Http::URI& uri);

// External filesystems register providers for endpoints taken from user
// specs. Their Bearer tokens may only go to Google APIs over HTTPS or to an
// endpoint the operator configured.
void AttachBearerToken(GcpCredentialProvider& provider, Aws::Http::HttpRequest& request) {
  if (!GcpCredentialRegistry::Instance().MayCarryBearerToken(request.GetUri())) {
    WarnUntrustedEndpointOnce(request.GetUri());
    return;
  }
  // This runs inside the AWS SDK, which is built without exceptions, so an
  // exception must not leave it.
  std::optional<std::pair<std::string, std::string>> header;
  try {
    header = provider.AuthorizationHeader();
  } catch (const std::exception& e) {
    LOG_STORAGE_WARNING_ << "GoogleHttpClientFactory: get GCP authorization header failed: " << e.what();
  } catch (...) {
    LOG_STORAGE_WARNING_ << "GoogleHttpClientFactory: get GCP authorization header failed: unknown exception";
  }
  if (header.has_value() && !header->first.empty()) {
    request.SetHeaderValue(header->first.c_str(), header->second.c_str());
  }
}

// Attaches a filesystem's own Bearer token to its CRT meta requests and native
// transport requests, which no delegator sees.
class GcpRequestAuthenticator final : public RequestAuthenticator {
  public:
  explicit GcpRequestAuthenticator(std::shared_ptr<GcpCredentialProvider> provider) : provider_(std::move(provider)) {}

  void Authenticate(Aws::Http::HttpRequest& request) override {
    // HMAC requests are signed by the AWS signer instead.
    if (provider_->IssuesBearerToken()) {
      AttachBearerToken(*provider_, request);
    }
  }

  [[nodiscard]] bool Equals(const RequestAuthenticator& other) const override {
    const auto* gcp = dynamic_cast<const GcpRequestAuthenticator*>(&other);
    return gcp != nullptr && gcp->provider_->Identity() == provider_->Identity();
  }

  private:
  std::shared_ptr<GcpCredentialProvider> provider_;
};

// HttpClient wrapper with an optional TLS floor. A delegator that
// GcpFileSystemProducer::Make() built is bound to that filesystem's provider;
// any other delegator looks the provider up by request URI. MakeRequest
// attaches the provider's Bearer token, or re-signs conditional writes with
// GOOG4 for HMAC.
class GoogleHttpClientDelegator : public Aws::Http::HttpClient {
  public:
  GoogleHttpClientDelegator(const Aws::Client::ClientConfiguration& config,
                            const std::string& tls_min_version,
                            std::shared_ptr<GcpCredentialProvider> bound_provider)
      : bound_provider_(std::move(bound_provider)) {
    if (!tls_min_version.empty()) {
      underlying_client_ =
          Aws::MakeShared<TlsCurlHttpClient>(kGoogleClientFactoryAllocationTag, config, tls_min_version);
    } else {
      underlying_client_ = Aws::MakeShared<Aws::Http::CurlHttpClient>(kGoogleClientFactoryAllocationTag, config);
    }
  }

  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
      const std::shared_ptr<Aws::Http::HttpRequest>& request,
      Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
      Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
    auto provider = bound_provider_ ? bound_provider_ : GcpCredentialRegistry::Instance().Lookup(request->GetUri());
    if (!provider) {
      // Only an unbound client gets here. Invariant: exactly one cloud
      // provider per process. If the GCP factory is installed, every S3-SDK
      // request should match a registered (endpoint, bucket). A miss here
      // indicates either a URI normalization bug in the registry or a request
      // that escaped Make(). Fail fast so the bug surfaces clearly instead of
      // a puzzling server 403.
      return MakeSignatureErrorResponse(request, fmt::format("No GcpCredentialProvider registered for URI: {}",
                                                             std::string(request->GetUri().GetURIString().c_str())));
    }
    // Bearer filesystems give the SDK anonymous credentials, and the factory
    // attaches nothing to requests built for an HttpClient, so the request
    // carries no Authorization header yet.
    if (provider->IssuesBearerToken()) {
      AttachBearerToken(*provider, *request);
    }
    auto status = provider->MaybeSignConditionalWrite(request);
    if (!status.ok()) {
      return MakeSignatureErrorResponse(request, status.message());
    }
    return underlying_client_->MakeRequest(request, readLimiter, writeLimiter);
  }

  private:
  // Build a standard AWS XML error so the SDK's ErrorMarshaller parses it
  // correctly and surfaces a usable message at the call site.
  static std::shared_ptr<Aws::Http::HttpResponse> MakeSignatureErrorResponse(
      const std::shared_ptr<Aws::Http::HttpRequest>& request, const std::string& message) {
    auto error_response =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>(kGoogleClientFactoryAllocationTag, request);
    error_response->SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
    error_response->GetResponseBody() << fmt::format(
        R"(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>SignatureFailed</Code>
  <Message>{}</Message>
</Error>)",
        message);
    return error_response;
  }

  std::shared_ptr<GcpCredentialProvider> bound_provider_;
  std::shared_ptr<Aws::Http::HttpClient> underlying_client_;
};

// Log each untrusted endpoint once; the gate runs on every request.
void WarnUntrustedEndpointOnce(const Aws::Http::URI& uri) {
  static std::mutex mu;
  static std::unordered_set<std::string> warned;
  std::string endpoint = std::string(uri.GetAuthority().c_str()) + ":" + std::to_string(uri.GetPort());
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!warned.insert(endpoint).second) {
      return;
    }
  }
  LOG_STORAGE_WARNING_ << "GoogleHttpClientFactory: not sending GCP credentials to untrusted endpoint " << endpoint
                       << " (only HTTPS *.googleapis.com and operator-configured endpoints)";
}

// HttpClientFactory. CreateHttpClient binds the provider of the Make() call
// building the client, if any. CreateHttpRequest authenticates only requests
// built in a ScopedRequestOutsideHttpClient (CRT meta requests and native
// transport requests, which no delegator sees): with the filesystem's
// authenticator, or by URI lookup against GcpCredentialRegistry for a
// filesystem built without one.
class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit GoogleHttpClientFactory(std::string tls_min_version) : tls_min_version_(std::move(tls_min_version)) {}

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
    std::shared_ptr<GcpCredentialProvider> bound_provider;
    if (pending_client_credentials != nullptr) {
      bound_provider = pending_client_credentials->provider;
      pending_client_credentials->bound = true;
    }
    return Aws::MakeShared<GoogleHttpClientDelegator>(kGoogleClientFactoryAllocationTag, config, tls_min_version_,
                                                      std::move(bound_provider));
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::String& uri, Aws::Http::HttpMethod method, const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::Http::URI& uri,
      Aws::Http::HttpMethod method,
      const Aws::IOStreamFactory& streamFactory) const override {
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(kGoogleClientFactoryAllocationTag, uri, method);
    request->SetResponseStreamFactory(streamFactory);

    if (!ScopedRequestOutsideHttpClient::Active()) {
      // The delegator authenticates the request.
      return request;
    }
    if (auto* authenticator = ScopedRequestOutsideHttpClient::Authenticator()) {
      authenticator->Authenticate(*request);
      return request;
    }
    auto provider = GcpCredentialRegistry::Instance().Lookup(uri);
    if (!provider) {
      // Same invariant as Delegator::MakeRequest: every GCP URI should have a
      // registered provider. This interface can only return a request (no error
      // channel), so log here; GCS will reject it.
      LOG_STORAGE_ERROR_ << "GoogleHttpClientFactory: no GcpCredentialProvider registered for URI: "
                         << std::string(uri.GetURIString().c_str());
      return request;
    }
    // HMAC requests are signed by the AWS signer instead.
    if (provider->IssuesBearerToken()) {
      AttachBearerToken(*provider, *request);
    }
    return request;
  }

  private:
  std::string tls_min_version_;
};

}  // namespace

namespace gcp_internal {

std::shared_ptr<Aws::Http::HttpClientFactory> MakeGoogleHttpClientFactory(std::string tls_min_version) {
  return Aws::MakeShared<GoogleHttpClientFactory>(kGoogleClientFactoryAllocationTag, std::move(tls_min_version));
}

}  // namespace gcp_internal

arrow::Status GcpFileSystemProducer::InitS3Compat(const ArrowFileSystemConfig& first_config) {
  // FIXME: assumes no cross-cloud-provider usage in practice; if it does happen,
  // this will produce unknown issues (InitializeS3 installs the HttpClientFactory
  // process-globally, and whichever producer runs call_once first wins).
  static std::once_flag s3_init_flag;
  static arrow::Status init_status = arrow::Status::OK();
  std::call_once(s3_init_flag, [&first_config]() {
    S3GlobalOptions global_options;
    auto it = kLogLevelMap.find(first_config.log_level);
    global_options.log_level = it != kLogLevelMap.end() ? it->second : S3LogLevel::Off;

    // tls_min_version only takes effect when use_ssl is enabled. The TLS floor
    // is fixed by the first GCP Make() (the factory is installed once). This
    // matches the pre-refactor behavior; treat it as a separate concern.
    std::string tls_min_ver =
        (first_config.use_ssl && !first_config.tls_min_version.empty()) ? first_config.tls_min_version : "";

    Aws::HttpOptions http_options;
    http_options.httpClientFactory_create_fn = [tls_min_ver]() {
      return gcp_internal::MakeGoogleHttpClientFactory(tls_min_ver);
    };
    global_options.http_options = http_options;
    global_options.override_default_http_options = true;

    auto status = InitializeS3(global_options);
    if (!status.ok()) {
      // Keep the original status code and detail. Stringifying into
      // Status::Invalid turned a credential/network init failure into a
      // data-class error (segcore DataFormatBroken/2024) and destroyed whatever
      // classification the cause carried.
      init_status = arrow::Status(status.code(), "GcpFileSystemProducer failed to initialize S3: " + status.message(),
                                  status.detail());
      return;
    }

    // Register cleanup on exit. atexit handlers must not throw, so log on failure.
    std::atexit([]() {
      auto status = EnsureS3Finalized();
      if (!status.ok()) {
        LOG_STORAGE_ERROR_ << "GcpFileSystemProducer failed to finalize S3: " << status.ToString();
      }
    });
  });
  return init_status;
}

arrow::Result<S3Options> GcpFileSystemProducer::CreateS3Options(bool bearer_token) {
  S3Options options;

  // Three cases:
  // 1. no ssl, verifySSL=false
  // 2. self-signed certificate, verifySSL=false
  // 3. CA-signed certificate, verifySSL=true
  options.scheme = config_.use_ssl ? "https" : "http";

  if (config_.use_ssl && !config_.ssl_ca_cert.empty()) {
    arrow::fs::FileSystemGlobalOptions fs_global_options;
    fs_global_options.tls_ca_file_path = config_.ssl_ca_cert;
    ARROW_RETURN_NOT_OK(arrow::fs::Initialize(fs_global_options));
  }

  options.endpoint_override = config_.address;
  options.force_virtual_addressing = config_.use_virtual_host;

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  options.request_timeout = config_.request_timeout_ms <= 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC
                                                            : config_.request_timeout_ms / 1000.0;
  options.max_connections = config_.max_connections;
  options.crt_queue_sizes = config_.crt_queue_sizes;
  options.multi_part_upload_size = config_.multi_part_upload_size;
  options.cloud_provider = config_.cloud_provider;
  options.background_writes = config_.background_writes;
  options.use_crc32c_checksum = config_.use_crc32c_checksum;
  options.use_crt = config_.s3_use_crt;

  // GCP does not support AssumeRole
  if (!config_.role_arn.empty()) {
    return arrow::Status::Invalid("AssumeRole credentials are only supported for AWS cloud provider, got: ",
                                  config_.cloud_provider);
  }

  if (bearer_token) {
    // GCP+IAM / Impersonation / Broker: the Authorization header is injected by the
    // registry-backed HTTP factory. Use anonymous credentials so the AWS SDK's
    // SigV4 signer does not overwrite the injected Bearer.
    options.ConfigureAnonymousCredentials();
  } else {
    options.ConfigureAccessKey(config_.access_key_id, config_.access_key_value);
  }

  return options;
}

arrow::Result<ArrowFileSystemPtr> GcpFileSystemProducer::Make() {
  ARROW_RETURN_NOT_OK(InitS3Compat(config_));
  ARROW_ASSIGN_OR_RAISE(auto provider, BuildGcpProviderFromConfig(config_));
  return MakeWithProvider(std::move(provider));
}

arrow::Result<ArrowFileSystemPtr> GcpFileSystemProducer::MakeWithProvider(
    std::shared_ptr<GcpCredentialProvider> provider) {
  ARROW_RETURN_NOT_OK(InitS3Compat(config_));
  GcpBucketKey key{NormalizeGcpEndpoint(config_.address, config_.use_ssl), config_.bucket_name};
  auto client_provider = provider;
  // Only the primary filesystem's address is operator configuration;
  // external filesystems (alias set) take theirs from user specs.
  auto registered =
      GcpCredentialRegistry::Instance().Register(std::move(key), provider, config_.alias.empty(), &client_provider);
  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options(provider->IssuesBearerToken()));
  s3_options.request_authenticator = std::make_shared<GcpRequestAuthenticator>(client_provider);
  ArrowFileSystemPtr fs;
  bool bound = false;
  {
    ScopedClientCredentials binding(client_provider);
    ARROW_ASSIGN_OR_RAISE(fs, S3FileSystem::Make(s3_options));
    bound = binding.bound();
  }
  // Bound clients and the authenticator carry the filesystem's own identity
  // on every request, so it may differ from the one registered for its
  // bucket. Clients that an HTTP client factory from outside milvus-storage
  // built find credentials by URI, so they need their own entry.
  if (!registered.ok()) {
    if (!bound) {
      return registered;
    }
    LOG_STORAGE_INFO_ << "GCP filesystem '" << config_.alias << "' on bucket '" << config_.bucket_name
                      << "' authenticates with its own credentials; another identity stays registered for the bucket";
  }
  return fs;
}

}  // namespace milvus_storage
