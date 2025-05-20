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

#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/filesystem/s3/AliyunSTSClient.h"
#include "milvus-storage/filesystem/s3/TencentCloudSTSClient.h"
#include "milvus-storage/filesystem/s3/AliyunCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/TencentCloudCredentialsProvider.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

namespace milvus_storage {

void S3FileSystemProducer::InitS3() {
  if (config_.use_custom_part_upload) {
    if (!IsS3Initialized()) {
      S3GlobalOptions global_options;
      global_options.log_level = LogLevel_Map[config_.log_level];

      if (config_.cloud_provider == "gcp" && config_.useIAM) {
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
        throw std::invalid_argument("ArrowFileSystem failed to initialize S3");
      }

      // Register cleanup on exit
      std::atexit([]() {
        auto status = EnsureS3Finalized();
        if (!status.ok()) {
          throw std::invalid_argument("ArrowFileSystem failed to finalize S3");
        }
      });
    }
  } else {
    if (!arrow::fs::IsS3Initialized()) {
      arrow::fs::S3GlobalOptions global_options;
      global_options.log_level = LogLevel_Map[config_.log_level];
      auto status = arrow::fs::InitializeS3(global_options);
      if (!status.ok()) {
        throw std::invalid_argument("ArrowFileSystem failed to initialize S3");
      }
      // Register cleanup on exit
      std::atexit([]() {
        auto status = arrow::fs::EnsureS3Finalized();
        if (!status.ok()) {
          throw std::invalid_argument("ArrowFileSystem failed to finalize S3");
        }
      });
    }
  }
}

Result<arrow::fs::S3Options> S3FileSystemProducer::CreateS3Options() {
  arrow::fs::S3Options options;
  arrow::util::Uri uri_parser;

  // Three cases:
  // 1. no ssl, verifySSL=false
  // 2. self-signed certificate, verifySSL=false
  // 3. CA-signed certificate, verifySSL=true
  options.scheme = config_.useSSL ? "https" : "http";

  if (config_.useSSL && !config_.sslCACert.empty()) {
    arrow::fs::FileSystemGlobalOptions fs_global_options;
    fs_global_options.tls_ca_file_path = config_.sslCACert;
    RETURN_ARROW_NOT_OK(arrow::fs::Initialize(fs_global_options));
  }

  options.endpoint_override = config_.address;

  options.force_virtual_addressing = config_.useVirtualHost;

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  options.request_timeout =
      config_.requestTimeoutMs == 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC : config_.requestTimeoutMs / 1000;

  return options;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> S3FileSystemProducer::CreateCredentialsProvider() {
  if (config_.cloud_provider == "aws") {
    return std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  }
  if (config_.cloud_provider == "aliyun") {
    return Aws::MakeShared<Aws::Auth::AliyunSTSAssumeRoleWebIdentityCredentialsProvider>(
        "AliyunSTSAssumeRoleWebIdentityCredentialsProvider");
  }
  if (config_.cloud_provider == "tencent") {
    return Aws::MakeShared<Aws::Auth::TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>(
        "TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider");
  }
  return nullptr;
}

Result<ArrowFileSystemPtr> S3FileSystemProducer::Make() {
  InitS3();

  auto status = CreateS3Options();
  if (!status.ok()) {
    return Status::ArrowError("cannot create S3 options");
  }
  arrow::fs::S3Options options = status.value();

  if (config_.useIAM && config_.cloud_provider != "gcp") {
    auto provider = CreateCredentialsProvider();
    auto credentials = provider->GetAWSCredentials();
    assert(!credentials.GetAWSAccessKeyId().empty());
    assert(!credentials.GetAWSSecretKey().empty());
    assert(!credentials.GetSessionToken().empty());
    options.credentials_provider = provider;
  } else {
    options.ConfigureAccessKey(config_.access_key_id, config_.access_key_value);
  }

  if (config_.use_custom_part_upload) {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, MultiPartUploadS3FS::Make(options));
    return ArrowFileSystemPtr(fs);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::S3FileSystem::Make(options));
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
