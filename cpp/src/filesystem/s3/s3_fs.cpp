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

#include "milvus-storage/filesystem/s3/s3_fs.h"

#include <cstdlib>

#include <curl/curl.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
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
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

namespace milvus_storage {

static std::unordered_map<std::string, S3LogLevel> LogLevel_Map = {
    {"off", S3LogLevel::Off},   {"fatal", S3LogLevel::Fatal}, {"error", S3LogLevel::Error}, {"warn", S3LogLevel::Warn},
    {"info", S3LogLevel::Info}, {"debug", S3LogLevel::Debug}, {"trace", S3LogLevel::Trace}};

static const char* GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG = "GoogleHttpClientFactory";
static const char* TLS_FACTORY_ALLOCATION_TAG = "TlsHttpClientFactory";

// Convert tls_min_version string to CURLOPT_SSLVERSION value.
// Returns CURL_SSLVERSION_DEFAULT (0) if the version string is empty or unrecognized.
static long TlsVersionToCurlOpt(const std::string& tls_min_version) {
  if (tls_min_version == "1.0")
    return CURL_SSLVERSION_TLSv1_0;
  if (tls_min_version == "1.1")
    return CURL_SSLVERSION_TLSv1_1;
  if (tls_min_version == "1.2")
    return CURL_SSLVERSION_TLSv1_2;
  if (tls_min_version == "1.3")
    return CURL_SSLVERSION_TLSv1_3;

  return CURL_SSLVERSION_DEFAULT;
}

// CurlHttpClient subclass that enforces a minimum TLS version via CURLOPT_SSLVERSION.
class TlsCurlHttpClient : public Aws::Http::CurlHttpClient {
  public:
  TlsCurlHttpClient(const Aws::Client::ClientConfiguration& config, const std::string& tls_min_version)
      : CurlHttpClient(config), tls_ssl_version_(TlsVersionToCurlOpt(tls_min_version)) {}

  protected:
  void OverrideOptionsOnConnectionHandle(CURL* handle) const override {
    if (tls_ssl_version_ != CURL_SSLVERSION_DEFAULT) {
      curl_easy_setopt(handle, CURLOPT_SSLVERSION, tls_ssl_version_);
    }
  }

  private:
  long tls_ssl_version_;
};

// HttpClientFactory that creates TlsCurlHttpClient instances for non-GCP S3 providers.
class TlsHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit TlsHttpClientFactory(const std::string& tls_min_version) : tls_min_version_(tls_min_version) {}

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
#ifdef BUILD_GTEST
    // Enable curl verbose tracing in test builds so that TLS handshake details
    // (e.g. "SSL connection using TLSv1.3 / ...") are routed through the AWS SDK logger.
    auto traced_config = config;
    traced_config.enableHttpClientTrace = true;
    return Aws::MakeShared<TlsCurlHttpClient>(TLS_FACTORY_ALLOCATION_TAG, traced_config, tls_min_version_);
#else
    return Aws::MakeShared<TlsCurlHttpClient>(TLS_FACTORY_ALLOCATION_TAG, config, tls_min_version_);
#endif
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(TLS_FACTORY_ALLOCATION_TAG, uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  private:
  std::string tls_min_version_;
};

class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit GoogleHttpClientFactory(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
                                   const std::string& tls_min_version = "")
      : credentials_(credentials), tls_min_version_(tls_min_version) {}

  void SetCredentials(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    if (!tls_min_version_.empty()) {
      return Aws::MakeShared<TlsCurlHttpClient>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, clientConfiguration,
                                                tls_min_version_);
    }
    return Aws::MakeShared<Aws::Http::CurlHttpClient>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, clientConfiguration);
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
    auto auth_header = credentials_->AuthorizationHeader();
    if (!auth_header.ok()) {
      throw std::invalid_argument("GoogleHttpClientFactory: create http request get authorization failed");
    }
    request->SetHeaderValue(auth_header->first.c_str(), auth_header->second.c_str());

    return request;
  }

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
  std::string tls_min_version_;
};

void S3FileSystemProducer::InitS3() {
  if (!IsS3Initialized()) {
    S3GlobalOptions global_options;
    global_options.log_level = LogLevel_Map[config_.log_level];

    // tls_min_version only takes effect when use_ssl is enabled
    std::string tls_min_ver = (config_.use_ssl && !config_.tls_min_version.empty()) ? config_.tls_min_version : "";

    if (config_.cloud_provider == "gcp" && config_.use_iam) {
      Aws::HttpOptions http_options;
      http_options.httpClientFactory_create_fn = [tls_min_ver]() {
        auto credentials =
            std::make_shared<google::cloud::oauth2_internal::GOOGLE_CLOUD_CPP_NS::ComputeEngineCredentials>();
        return Aws::MakeShared<GoogleHttpClientFactory>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, credentials, tls_min_ver);
      };
      global_options.http_options = http_options;
      global_options.override_default_http_options = true;
    } else if (!tls_min_ver.empty()) {
      // Non-GCP S3-compatible providers with TLS version override (only when use_ssl=true)
      Aws::HttpOptions http_options;
      http_options.httpClientFactory_create_fn = [tls_min_ver]() {
        return Aws::MakeShared<TlsHttpClientFactory>(TLS_FACTORY_ALLOCATION_TAG, tls_min_ver);
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
  if (config_.cloud_provider == "aliyun" || config_.cloud_provider == "tencent" || config_.cloud_provider == "huawei") {
    options.force_virtual_addressing = true;
  }

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  options.request_timeout = config_.request_timeout_ms <= 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC
                                                            : config_.request_timeout_ms / 1000;
  options.max_connections = config_.max_connections;
  options.cloud_provider = config_.cloud_provider;
  options.use_crc32c_checksum = config_.use_crc32c_checksum;

  // Credential configuration priority:
  // 1. IAM — GCP uses anonymous credentials (auth handled by GoogleHttpClientFactory via OAuth2),
  //          other providers use their respective STS credential providers
  // 2. Explicit access key / secret key
  if (config_.use_iam) {
    if (config_.cloud_provider == "gcp") {
      // GCP+IAM: authentication is handled by GoogleHttpClientFactory which injects
      // OAuth2 Authorization headers in CreateHttpRequest(). Use anonymous credentials
      // so the AWS SDK's SigV4 signer skips signing and preserves the OAuth2 header.
      options.ConfigureAnonymousCredentials();
    } else {
      auto provider = CreateCredentialsProvider();
      if (!provider) {
        return arrow::Status::Invalid("Unknown credentials provider, cloud provider: ", config_.cloud_provider);
      }
      auto credentials = provider->GetAWSCredentials();
      assert(!credentials.GetAWSAccessKeyId().empty() && "AWS Access Key ID is empty");
      assert(!credentials.GetAWSSecretKey().empty() && "AWS Secret Key is empty");
      assert(!credentials.GetSessionToken().empty() && "AWS Session Token is empty");
      options.credentials_provider = provider;
    }
  } else {
    options.ConfigureAccessKey(config_.access_key_id, config_.access_key_value);
  }

  return options;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateCredentialsProvider() {
  if (config_.cloud_provider == "aws") {
    return CreateAwsCredentialsProvider();
  }
  if (config_.cloud_provider == "aliyun") {
    return CreateAliyunCredentialsProvider();
  }
  if (config_.cloud_provider == "tencent") {
    return CreateTencentCredentialsProvider();
  }
  if (config_.cloud_provider == "huawei") {
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
  ARROW_ASSIGN_OR_RAISE(auto fs, MultiPartUploadS3FS::Make(s3_options));
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
