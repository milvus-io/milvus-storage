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

#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"

#include <cstdlib>
#include <sstream>

#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

#include <arrow/status.h>
#include <arrow/util/uri.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/provider/AliyunSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/signer.h"

namespace milvus_storage {

static std::unordered_map<std::string, S3LogLevel> LogLevel_Map = {
    {"off", S3LogLevel::Off},   {"fatal", S3LogLevel::Fatal}, {"error", S3LogLevel::Error}, {"warn", S3LogLevel::Warn},
    {"info", S3LogLevel::Info}, {"debug", S3LogLevel::Debug}, {"trace", S3LogLevel::Trace}};

static const char* GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG = "GoogleHttpClientFactory";

// GoogleHttpClientDelegator: Delegation-pattern HttpClient
// Modifies request headers on MakeRequest, then delegates to the underlying CurlHttpClient
class GoogleHttpClientDelegator : public Aws::Http::HttpClient {
  public:
  explicit GoogleHttpClientDelegator(const Aws::Client::ClientConfiguration& config,
                                     bool use_iam,
                                     const std::string& access_key = "",
                                     const std::string& secret_key = "")
      : use_iam_(use_iam), access_key_(access_key), secret_key_(secret_key) {
    // Create underlying CurlHttpClient
    underlying_client_ = Aws::MakeShared<Aws::Http::CurlHttpClient>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, config);
  }

  public:
  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
      const std::shared_ptr<Aws::Http::HttpRequest>& request,
      Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
      Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
    // Check header to see if this is a conditional write
    bool is_conditional_write = request->HasHeader("x-goog-if-generation-match");
    // Decide if GOOG4 signing is needed (HMAC mode + conditional write)
    bool needs_goog4_signing = !use_iam_ && !access_key_.empty() && !secret_key_.empty() && is_conditional_write;
    if (needs_goog4_signing) {
      // Remove possible AWS signature headers
      request->DeleteHeader("Authorization");
      request->DeleteHeader("x-amz-date");
      request->DeleteHeader("x-amz-content-sha256");
      request->DeleteHeader("x-amz-security-token");

      // Convert AWS metadata headers to GCS metadata headers
      // AWS uses x-amz-meta-*, GCS uses x-goog-meta-*
      std::vector<std::pair<std::string, std::string>> meta_headers_to_convert;
      std::vector<std::string> old_keys_to_delete;

      // Collect all x-amz-meta-* headers to convert
      for (const auto& [key, value] : request->GetHeaders()) {
        std::string lower_key = key;
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);

        if (lower_key.find("x-amz-meta-") == 0) {
          // Found x-amz-meta- prefix
          std::string suffix = key.substr(11);  // strip "x-amz-meta-" (11 chars)
          std::string new_key = "x-goog-meta-" + suffix;
          meta_headers_to_convert.emplace_back(new_key, value);
          old_keys_to_delete.push_back(key);
        }
      }

      // Remove old x-amz-meta-* headers, add new x-goog-meta-* headers
      if (!meta_headers_to_convert.empty()) {
        // Delete old headers
        for (const auto& old_key : old_keys_to_delete) {
          request->DeleteHeader(old_key.c_str());
        }

        // Add new headers
        for (const auto& [new_key, value] : meta_headers_to_convert) {
          request->SetHeaderValue(new_key, value);
        }
      }

      // Sign with GOOG4-HMAC-SHA256
      if (!signer::goog4::SignRequest(request, access_key_, secret_key_)) {
        // Signature failed, create error response with FORBIDDEN status
        // Using standard AWS XML error format that will be parsed by ErrorMarshaller
        auto error_response =
            Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, request);
        error_response->SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
        std::string error_body = R"(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>SignatureFailed</Code>
  <Message>Signature failed</Message>
</Error>)";
        error_response->GetResponseBody() << error_body;
        return error_response;
      }
    }
    return underlying_client_->MakeRequest(request, readLimiter, writeLimiter);
  }

  private:
  std::shared_ptr<Aws::Http::HttpClient> underlying_client_;
  bool use_iam_;
  std::string access_key_;
  std::string secret_key_;
};

class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  // Constructor: accepts both credentials (for IAM) and ak/sk (for HMAC)
  explicit GoogleHttpClientFactory(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
                                   bool use_iam,
                                   const std::string& access_key = "",
                                   const std::string& secret_key = "")
      : credentials_(credentials), use_iam_(use_iam), access_key_(access_key), secret_key_(secret_key) {}

  void SetCredentials(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    // Create GoogleHttpClientDelegator with use_iam and ak/sk.
    // Delegator will decide whether to use GOOG4 signing based on use_iam
    return Aws::MakeShared<GoogleHttpClientDelegator>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, clientConfiguration,
                                                      use_iam_, access_key_, secret_key_);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, uri, method);
    request->SetResponseStreamFactory(streamFactory);

    if (!use_iam_) {
      // HMAC mode: directly return unsigned request
      // Actual signing will be handled in GoogleHttpClientDelegator::MakeRequest
      return request;
    }

    // For IAM, add OAuth2 header
    if (!credentials_) {
      throw std::invalid_argument("GoogleHttpClientFactory: credentials_ is nullptr");
    }

    auto auth_header = credentials_->AuthorizationHeader();
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
};

void S3FileSystemProducer::InitS3() {
  if (!IsS3Initialized()) {
    S3GlobalOptions global_options;
    global_options.log_level = LogLevel_Map[config_.log_level];

    if (config_.cloud_provider == kCloudProviderGCP) {
      std::string ak = config_.access_key_id;
      std::string sk = config_.access_key_value;
      bool use_iam = config_.use_iam;

      Aws::HttpOptions http_options;
      http_options.httpClientFactory_create_fn = [ak, sk, use_iam]() {
        auto credentials =
            std::make_shared<google::cloud::oauth2_internal::GOOGLE_CLOUD_CPP_NS::ComputeEngineCredentials>();
        return Aws::MakeShared<GoogleHttpClientFactory>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, credentials,
                                                        use_iam,            // Explicitly pass use_iam flag
                                                        use_iam ? "" : ak,  // IAM mode does not need ak
                                                        use_iam ? "" : sk   // IAM mode does not need sk
        );
      };
      global_options.http_options = http_options;
      global_options.override_default_http_options = true;
    }
    auto status = InitializeS3(global_options);
    if (!status.ok()) {
      throw std::invalid_argument("ArrowFileSystem failed to initialize S3: " + status.ToString());
    }

    // Register cleanup on exit
    std::atexit([]() {
      auto status = EnsureS3Finalized();
      if (!status.ok()) {
        throw std::invalid_argument("ArrowFileSystem failed to finalize S3: " + status.ToString());
      }
    });
  }
}

arrow::Result<S3Options> S3FileSystemProducer::CreateS3Options() {
  S3Options options;
  arrow::util::Uri uri_parser;

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
  if (config_.cloud_provider == kCloudProviderAliyun || config_.cloud_provider == kCloudProviderTencent ||
      config_.cloud_provider == kCloudProviderHuawei) {
    options.force_virtual_addressing = true;
  }

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  options.request_timeout = config_.request_timeout_ms <= 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC
                                                            : config_.request_timeout_ms / 1000;
  options.max_connections = config_.max_connections;
  options.multi_part_upload_size = config_.multi_part_upload_size;
  options.cloud_provider = config_.cloud_provider;

  if (config_.use_iam && config_.cloud_provider != kCloudProviderGCP) {
    auto provider = CreateCredentialsProvider();
    if (!provider) {
      return arrow::Status::Invalid("Unknown credentials provider, cloud provider: ", config_.cloud_provider);
    }
    auto credentials = provider->GetAWSCredentials();
    assert(!credentials.GetAWSAccessKeyId().empty() && "AWS Access Key ID is empty");
    assert(!credentials.GetAWSSecretKey().empty() && "AWS Secret Key is empty");
    assert(!credentials.GetSessionToken().empty() && "AWS Session Token is empty");
    options.credentials_provider = provider;
  } else {
    options.ConfigureAccessKey(config_.access_key_id, config_.access_key_value);
  }

  return options;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateCredentialsProvider() {
  if (config_.cloud_provider == kCloudProviderAWS) {
    return CreateAwsCredentialsProvider();
  }
  if (config_.cloud_provider == kCloudProviderAliyun) {
    return CreateAliyunCredentialsProvider();
  }
  if (config_.cloud_provider == kCloudProviderTencent) {
    return CreateTencentCredentialsProvider();
  }
  if (config_.cloud_provider == kCloudProviderHuawei) {
    return CreateHuaweiCredentialsProvider();
  }
  return nullptr;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateHuaweiCredentialsProvider() {
  static auto provider = Aws::MakeShared<HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider>(
      "HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider");
  return provider;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateAwsCredentialsProvider() {
  static auto provider = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  return provider;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateAliyunCredentialsProvider() {
  static auto provider = Aws::MakeShared<AliyunSTSAssumeRoleWebIdentityCredentialsProvider>(
      "AliyunSTSAssumeRoleWebIdentityCredentialsProvider");
  return provider;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateTencentCredentialsProvider() {
  static auto provider = Aws::MakeShared<TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>(
      "TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider");
  return provider;
}

arrow::Result<ArrowFileSystemPtr> S3FileSystemProducer::Make() {
  InitS3();

  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ARROW_ASSIGN_OR_RAISE(auto fs, S3FileSystem::Make(s3_options));
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
