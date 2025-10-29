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
#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/s3/provider/AliyunSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

namespace milvus_storage {

static std::unordered_map<std::string, arrow::fs::S3LogLevel> LogLevel_Map = {
    {"off", arrow::fs::S3LogLevel::Off},     {"fatal", arrow::fs::S3LogLevel::Fatal},
    {"error", arrow::fs::S3LogLevel::Error}, {"warn", arrow::fs::S3LogLevel::Warn},
    {"info", arrow::fs::S3LogLevel::Info},   {"debug", arrow::fs::S3LogLevel::Debug},
    {"trace", arrow::fs::S3LogLevel::Trace}};

static const char* GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG = "GoogleHttpClientFactory";

class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit GoogleHttpClientFactory(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  void SetCredentials(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
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
};

void S3FileSystemProducer::InitS3() {
  if (!IsS3Initialized()) {
    ExtendS3GlobalOptions global_options;
    global_options.log_level = LogLevel_Map[config_.log_level];

    if (config_.cloud_provider == "gcp" && config_.use_iam) {
      Aws::HttpOptions http_options;
      http_options.httpClientFactory_create_fn = []() {
        auto credentials =
            std::make_shared<google::cloud::oauth2_internal::GOOGLE_CLOUD_CPP_NS::ComputeEngineCredentials>();
        return Aws::MakeShared<GoogleHttpClientFactory>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, credentials);
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
  // FIXME: After removing the arrow::fs::S3Options, then we no longer need to call global Aws::InitAPI twice.
  if (!arrow::fs::IsS3Initialized()) {
    arrow::fs::S3GlobalOptions arrow_global_options;
    arrow_global_options.log_level = LogLevel_Map[config_.log_level];
    auto status = arrow::fs::InitializeS3(arrow_global_options);
    if (!status.ok()) {
      throw std::invalid_argument("Arrow S3 initialization failed: " + status.ToString());
    }
    std::atexit([]() {
      auto status = arrow::fs::EnsureS3Finalized();
      if (!status.ok()) {
        throw std::invalid_argument("ArrowFileSystem failed to finalize arrow S3: " + status.ToString());
      }
    });
  }
}

arrow::Result<ExtendedS3Options> S3FileSystemProducer::CreateS3Options() {
  ExtendedS3Options options;
  arrow::util::Uri uri_parser;

  // Three cases:
  // 1. no ssl, verifySSL=false
  // 2. self-signed certificate, verifySSL=false
  // 3. CA-signed certificate, verifySSL=true
  options.scheme = config_.use_ssl ? "https" : "http";

  if (config_.use_ssl && !config_.ssl_ca_cert.empty()) {
    arrow::fs::FileSystemGlobalOptions fs_global_options;
    fs_global_options.tls_ca_file_path = config_.ssl_ca_cert;
    RETURN_ARROW_NOT_OK(arrow::fs::Initialize(fs_global_options));
  }

  options.endpoint_override = config_.address;

  options.force_virtual_addressing = config_.use_virtual_host;
  if (config_.cloud_provider == "aliyun" || config_.cloud_provider == "tencent") {
    options.force_virtual_addressing = true;
  }

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  options.request_timeout = config_.request_timeout_ms <= 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC
                                                            : config_.request_timeout_ms / 1000;

  if (config_.use_iam && config_.cloud_provider != "gcp") {
    auto provider = CreateCredentialsProvider();
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
  if (config_.cloud_provider == "aws") {
    return CreateAwsCredentialsProvider();
  }
  if (config_.cloud_provider == "aliyun") {
    return CreateAliyunCredentialsProvider();
  }
  if (config_.cloud_provider == "tencent") {
    return CreateTencentCredentialsProvider();
  }
  return nullptr;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateAwsCredentialsProvider() {
  static auto provider = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  return provider;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateAliyunCredentialsProvider() {
  static auto provider = Aws::MakeShared<Aws::Auth::AliyunSTSAssumeRoleWebIdentityCredentialsProvider>(
      "AliyunSTSAssumeRoleWebIdentityCredentialsProvider");
  return provider;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateTencentCredentialsProvider() {
  static auto provider = Aws::MakeShared<Aws::Auth::TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>(
      "TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider");
  return provider;
}

arrow::Result<ArrowFileSystemPtr> S3FileSystemProducer::Make() {
  InitS3();

  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ASSIGN_OR_RETURN_NOT_OK(auto fs, MultiPartUploadS3FS::Make(s3_options));
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
