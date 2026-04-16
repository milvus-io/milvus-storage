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

#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// Stateless HttpClient wrapper. Holds only a reference to the registry and an
// optional TLS floor setting. Per-request credential work is dispatched by URI
// lookup in MakeRequest (for GOOG4 re-signing on conditional writes).
class GoogleHttpClientDelegator : public Aws::Http::HttpClient {
  public:
  GoogleHttpClientDelegator(const Aws::Client::ClientConfiguration& config, const std::string& tls_min_version) {
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
    auto provider = GcpCredentialRegistry::Instance().Lookup(request->GetUri());
    if (!provider) {
      // Invariant: exactly one cloud provider per process. If the GCP factory
      // is installed, every S3-SDK request should match a registered
      // (endpoint, bucket). A miss here indicates either a URI normalization
      // bug in the registry or a request that escaped RegisterIdentity(). Fail
      // fast so the bug surfaces clearly instead of a puzzling server 403.
      return MakeSignatureErrorResponse(request, fmt::format("No GcpCredentialProvider registered for URI: {}",
                                                             std::string(request->GetUri().GetURIString().c_str())));
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

  std::shared_ptr<Aws::Http::HttpClient> underlying_client_;
};

// Stateless HttpClientFactory. Dispatches Authorization header injection by
// URI lookup against GcpCredentialRegistry. Identity itself lives in the
// registry, not in the factory — that's what allows N identities per process.
class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit GoogleHttpClientFactory(std::string tls_min_version) : tls_min_version_(std::move(tls_min_version)) {}

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
    return Aws::MakeShared<GoogleHttpClientDelegator>(kGoogleClientFactoryAllocationTag, config, tls_min_version_);
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

    auto provider = GcpCredentialRegistry::Instance().Lookup(uri);
    if (!provider) {
      // Same invariant as Delegator::MakeRequest: every GCP URI should have a
      // registered provider. This interface can only return a request (no error
      // channel), so log here and let MakeRequest fail the request with a 403.
      LOG_STORAGE_ERROR_ << "GoogleHttpClientFactory: no GcpCredentialProvider registered for URI: "
                         << std::string(uri.GetURIString().c_str());
      return request;
    }
    if (auto header = provider->AuthorizationHeader(); header.has_value()) {
      request->SetHeaderValue(header->first.c_str(), header->second.c_str());
    }
    return request;
  }

  private:
  std::string tls_min_version_;
};

}  // namespace

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
      return Aws::MakeShared<GoogleHttpClientFactory>(kGoogleClientFactoryAllocationTag, tls_min_ver);
    };
    global_options.http_options = http_options;
    global_options.override_default_http_options = true;

    auto status = InitializeS3(global_options);
    if (!status.ok()) {
      init_status = arrow::Status::Invalid("GcpFileSystemProducer failed to initialize S3: ", status.ToString());
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

arrow::Status GcpFileSystemProducer::RegisterIdentity(const ArrowFileSystemConfig& config) {
  ARROW_ASSIGN_OR_RAISE(auto provider, BuildGcpProviderFromConfig(config));
  GcpBucketKey key{NormalizeGcpEndpointHost(config.address), config.bucket_name};
  GcpCredentialRegistry::Instance().Register(std::move(key), std::move(provider));
  return arrow::Status::OK();
}

arrow::Result<S3Options> GcpFileSystemProducer::CreateS3Options() {
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
                                                            : config_.request_timeout_ms / 1000;
  options.max_connections = config_.max_connections;
  options.multi_part_upload_size = config_.multi_part_upload_size;
  options.cloud_provider = config_.cloud_provider;
  options.background_writes = config_.background_writes;
  options.use_crc32c_checksum = config_.use_crc32c_checksum;

  // GCP does not support AssumeRole
  if (!config_.role_arn.empty()) {
    return arrow::Status::Invalid("AssumeRole credentials are only supported for AWS cloud provider, got: ",
                                  config_.cloud_provider);
  }

  if (config_.use_iam) {
    // GCP+IAM / Impersonation: the Authorization header is injected by the
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
  ARROW_RETURN_NOT_OK(RegisterIdentity(config_));

  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ARROW_ASSIGN_OR_RAISE(auto fs, S3FileSystem::Make(s3_options));
  return std::make_shared<FileSystemProxy>(config_.bucket_name, fs);
}

}  // namespace milvus_storage
