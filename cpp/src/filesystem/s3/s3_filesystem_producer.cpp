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
#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <utility>

#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
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
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
#include "milvus-storage/filesystem/tls_http_client.h"

namespace milvus_storage {

namespace {

constexpr const char* kTlsFactoryAllocationTag = "TlsHttpClientFactory";

// HttpClientFactory that creates TlsCurlHttpClient instances so the AWS SDK
// honors the configured minimum TLS version for S3-compatible providers.
class TlsHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit TlsHttpClientFactory(const std::string& tls_min_version) : tls_min_version_(tls_min_version) {}

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
#ifdef BUILD_GTEST
    // Enable curl verbose tracing in test builds so that TLS handshake details
    // (e.g. "SSL connection using TLSv1.3 / ...") are routed through the AWS SDK logger.
    auto traced_config = config;
    traced_config.enableHttpClientTrace = true;
    return Aws::MakeShared<TlsCurlHttpClient>(kTlsFactoryAllocationTag, traced_config, tls_min_version_);
#else
    return Aws::MakeShared<TlsCurlHttpClient>(kTlsFactoryAllocationTag, config, tls_min_version_);
#endif
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::String& uri, Aws::Http::HttpMethod method, const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::Http::URI& uri,
      Aws::Http::HttpMethod method,
      const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(kTlsFactoryAllocationTag, uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  private:
  std::string tls_min_version_;
};

}  // namespace

static std::unordered_map<std::string, S3LogLevel> LogLevel_Map = {
    {"off", S3LogLevel::Off},   {"fatal", S3LogLevel::Fatal}, {"error", S3LogLevel::Error}, {"warn", S3LogLevel::Warn},
    {"info", S3LogLevel::Info}, {"debug", S3LogLevel::Debug}, {"trace", S3LogLevel::Trace}};

arrow::Status S3FileSystemProducer::InitS3() {
  static std::once_flag s3_init_flag;
  static arrow::Status init_status = arrow::Status::OK();
  std::call_once(s3_init_flag, [this]() {
    S3GlobalOptions global_options;
    global_options.log_level = LogLevel_Map[config_.log_level];

    // tls_min_version only takes effect when use_ssl is enabled
    std::string tls_min_ver = (config_.use_ssl && !config_.tls_min_version.empty()) ? config_.tls_min_version : "";

    if (!tls_min_ver.empty()) {
      Aws::HttpOptions http_options;
      http_options.httpClientFactory_create_fn = [tls_min_ver]() {
        return Aws::MakeShared<TlsHttpClientFactory>(kTlsFactoryAllocationTag, tls_min_ver);
      };
      global_options.http_options = http_options;
      global_options.override_default_http_options = true;
    }

    auto status = InitializeS3(global_options);
    if (!status.ok()) {
      init_status = arrow::Status::Invalid("S3FileSystemProducer failed to initialize S3: ", status.ToString());
      return;
    }

    // Register cleanup on exit. atexit handlers must not throw, so log on failure.
    std::atexit([]() {
      auto status = EnsureS3Finalized();
      if (!status.ok()) {
        LOG_STORAGE_ERROR_ << "S3FileSystemProducer failed to finalize S3: " << status.ToString();
      }
    });
  });
  return init_status;
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
  options.background_writes = config_.background_writes;
  options.use_crc32c_checksum = config_.use_crc32c_checksum;

  // Credential configuration priority:
  // 1. AssumeRole (role_arn) — AWS only
  // 2. IAM — use provider-specific STS credential providers
  // 3. Explicit access key / secret key
  if (!config_.role_arn.empty()) {
    if (config_.cloud_provider != kCloudProviderAWS) {
      return arrow::Status::Invalid("AssumeRole credentials are only supported for AWS cloud provider, got: ",
                                    config_.cloud_provider);
    }
    LOG_STORAGE_DEBUG_ << "using AssumeRole credentials, role_arn=" << config_.role_arn
                       << ", load_frequency=" << config_.load_frequency;
    options.ConfigureAssumeRoleCredentials(config_.role_arn, config_.session_name, config_.external_id,
                                           config_.load_frequency);
  } else if (config_.use_iam) {
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
  ARROW_RETURN_NOT_OK(InitS3());

  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ARROW_ASSIGN_OR_RAISE(auto fs, S3FileSystem::Make(s3_options));
  return std::make_shared<FileSystemProxy>(config_.bucket_name, fs);
}

}  // namespace milvus_storage
