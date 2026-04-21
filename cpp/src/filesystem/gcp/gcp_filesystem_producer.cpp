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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/internal/oauth2_google_credentials.h>
#include <google/cloud/internal/rest_client.h>
#include <google/cloud/options.h>
#include <google/cloud/status_or.h>
#include <google/cloud/storage/oauth2/compute_engine_credentials.h>
#include <google/cloud/storage/oauth2/google_credentials.h>

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
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/tls_http_client.h"

namespace milvus_storage {

namespace {

constexpr const char* kGoogleClientFactoryAllocationTag = "GoogleHttpClientFactory";

const std::unordered_map<std::string, S3LogLevel> kLogLevelMap = {
    {"off", S3LogLevel::Off},   {"fatal", S3LogLevel::Fatal}, {"error", S3LogLevel::Error}, {"warn", S3LogLevel::Warn},
    {"info", S3LogLevel::Info}, {"debug", S3LogLevel::Debug}, {"trace", S3LogLevel::Trace}};

// GoogleHttpClientDelegator: Delegation-pattern HttpClient
// Modifies request headers on MakeRequest, then delegates to the underlying CurlHttpClient
class GoogleHttpClientDelegator : public Aws::Http::HttpClient {
  public:
  explicit GoogleHttpClientDelegator(const Aws::Client::ClientConfiguration& config,
                                     bool use_iam,
                                     const std::string& access_key = "",
                                     const std::string& secret_key = "",
                                     const std::string& tls_min_version = "")
      : use_iam_(use_iam), access_key_(access_key), secret_key_(secret_key) {
    // Create underlying CurlHttpClient, optionally with TLS version enforcement
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
    bool is_conditional_write = request->HasHeader("x-goog-if-generation-match");
    bool needs_goog4_signing = !use_iam_ && !access_key_.empty() && !secret_key_.empty() && is_conditional_write;
    if (needs_goog4_signing) {
      auto status = ApplyGoog4Signing(request, access_key_, secret_key_);
      if (!status.ok()) {
        // Standard AWS XML error format so ErrorMarshaller parses it correctly.
        auto error_response =
            Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>(kGoogleClientFactoryAllocationTag, request);
        error_response->SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
        error_response->GetResponseBody() << fmt::format(
            R"(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>SignatureFailed</Code>
  <Message>{}</Message>
</Error>)",
            status.message());
        return error_response;
      }
    }
    return underlying_client_->MakeRequest(request, readLimiter, writeLimiter);
  }

  private:
  // Strip AWS SigV4 headers, rewrite x-amz-meta-* → x-goog-meta-*, and re-sign
  // the request with GOOG4-HMAC-SHA256.
  static arrow::Status ApplyGoog4Signing(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                                         const std::string& access_key,
                                         const std::string& secret_key) {
    // Remove possible AWS signature headers
    request->DeleteHeader("Authorization");
    request->DeleteHeader("x-amz-date");
    request->DeleteHeader("x-amz-content-sha256");
    request->DeleteHeader("x-amz-security-token");
    request->DeleteHeader("x-amz-api-version");

    // AWS uses x-amz-meta-*, GCS uses x-goog-meta-*
    std::vector<std::pair<std::string, std::string>> meta_headers_to_convert;
    std::vector<std::string> old_keys_to_delete;
    for (const auto& [key, value] : request->GetHeaders()) {
      std::string lower_key = key;
      std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);

      if (lower_key.find("x-amz-meta-") == 0) {
        std::string suffix = key.substr(11);  // strip "x-amz-meta-"
        meta_headers_to_convert.emplace_back("x-goog-meta-" + suffix, value);
        old_keys_to_delete.push_back(key);
      }
    }
    for (const auto& old_key : old_keys_to_delete) {
      request->DeleteHeader(old_key.c_str());
    }
    for (const auto& [new_key, value] : meta_headers_to_convert) {
      request->SetHeaderValue(new_key, value);
    }

    if (!milvus_storage::auth_signer::googv4::SignRequest(request, access_key, secret_key)) {
      return arrow::Status::ExecutionError("GOOG4-HMAC-SHA256 signing failed");
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<Aws::Http::HttpClient> underlying_client_;
  bool use_iam_;
  std::string access_key_;
  std::string secret_key_;
};

class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  // Constructor: accepts both credentials (for IAM) and ak/sk (for HMAC)
  GoogleHttpClientFactory(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
                          bool use_iam,
                          const std::string& access_key = "",
                          const std::string& secret_key = "",
                          const std::string& tls_min_version = "")
      : credentials_(std::move(credentials)),
        use_iam_(use_iam),
        access_key_(access_key),
        secret_key_(secret_key),
        tls_min_version_(tls_min_version) {}

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    // Delegator decides whether to apply GOOG4 signing based on use_iam
    return Aws::MakeShared<GoogleHttpClientDelegator>(kGoogleClientFactoryAllocationTag, clientConfiguration, use_iam_,
                                                      access_key_, secret_key_, tls_min_version_);
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

    if (!use_iam_) {
      // HMAC mode: return unsigned request; signing handled by GoogleHttpClientDelegator
      return request;
    }

    // IAM mode: inject OAuth2 Bearer header
    if (!credentials_) {
      throw std::invalid_argument("GoogleHttpClientFactory: credentials_ is nullptr");
    }

    auto auth_header = google::cloud::oauth2_internal::AuthorizationHeader(*credentials_);
    if (!auth_header.ok()) {
      throw std::invalid_argument("GoogleHttpClientFactory: create http request get authorization failed");
    }
    request->SetHeaderValue(auth_header->first.c_str(), auth_header->second.c_str());
    return request;
  }

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
  bool use_iam_;
  std::string access_key_;
  std::string secret_key_;
  std::string tls_min_version_;
};

}  // namespace

arrow::Status GcpFileSystemProducer::InitS3Compat() {
  // FIXME: assumes no cross-cloud-provider usage in practice; if it does happen,
  // this will produce unknown issues (InitializeS3 installs the HttpClientFactory
  // process-globally, and whichever producer runs call_once first wins).
  static std::once_flag s3_init_flag;
  static arrow::Status init_status = arrow::Status::OK();
  std::call_once(s3_init_flag, [this]() {
    S3GlobalOptions global_options;
    auto it = kLogLevelMap.find(config_.log_level);
    global_options.log_level = it != kLogLevelMap.end() ? it->second : S3LogLevel::Off;

    // tls_min_version only takes effect when use_ssl is enabled
    std::string tls_min_ver = (config_.use_ssl && !config_.tls_min_version.empty()) ? config_.tls_min_version : "";

    std::string ak = config_.access_key_id;
    std::string sk = config_.access_key_value;
    bool use_iam = config_.use_iam;

    Aws::HttpOptions http_options;
    http_options.httpClientFactory_create_fn = [ak, sk, use_iam, tls_min_ver]() {
      auto client_factory = [](google::cloud::Options const& opts) {
        return google::cloud::rest_internal::MakeDefaultRestClient("", opts);
      };
      auto credentials = std::make_shared<google::cloud::oauth2_internal::ComputeEngineCredentials>(
          google::cloud::Options{}, std::move(client_factory));
      return Aws::MakeShared<GoogleHttpClientFactory>(kGoogleClientFactoryAllocationTag, credentials, use_iam,
                                                      use_iam ? "" : ak,  // IAM mode does not need ak
                                                      use_iam ? "" : sk,  // IAM mode does not need sk
                                                      tls_min_ver);
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
    // GCP+IAM: authentication is handled by GoogleHttpClientFactory which injects
    // OAuth2 Authorization headers in CreateHttpRequest(). Use anonymous credentials
    // so the AWS SDK's SigV4 signer skips signing and preserves the OAuth2 header.
    options.ConfigureAnonymousCredentials();
  } else {
    options.ConfigureAccessKey(config_.access_key_id, config_.access_key_value);
  }

  return options;
}

arrow::Result<ArrowFileSystemPtr> GcpFileSystemProducer::Make() {
  ARROW_RETURN_NOT_OK(InitS3Compat());

  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ARROW_ASSIGN_OR_RAISE(auto fs, S3FileSystem::Make(s3_options));
  return std::make_shared<FileSystemProxy>(config_.bucket_name, fs);
}

}  // namespace milvus_storage
