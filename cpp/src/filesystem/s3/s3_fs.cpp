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

void initS3(const ArrowFileSystemConfig& config) {
  if (!arrow::fs::IsS3Initialized()) {
    arrow::fs::S3GlobalOptions global_options;
    global_options.log_level = LogLevel_Map[config.log_level];
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

arrow::fs::S3Options createS3Options(const ArrowFileSystemConfig& config) {
  arrow::fs::S3Options options;
  arrow::util::Uri uri_parser;

  // Three cases:
  // 1. no ssl, verifySSL=false
  // 2. self-signed certificate, verifySSL=false
  // 3. CA-signed certificate, verifySSL=true
  if (config.useSSL) {
    options.scheme = "https";
    if (!config.sslCACert.empty()) {
      arrow::fs::FileSystemGlobalOptions fs_global_options;
      fs_global_options.tls_ca_file_path = config.sslCACert;
      arrow::fs::Initialize(fs_global_options);
    }
    auto uri = "https://" + config.bucket_name + "." + config.address;
    auto status = uri_parser.Parse(uri);
    if (!status.ok()) {
      throw std::invalid_argument("can not parse uri from arrow file system config");
    }
  } else {
    options.scheme = "http";
    auto uri = "http://" + config.bucket_name + "." + config.address;
    auto status = uri_parser.Parse(uri);
    if (!status.ok()) {
      throw std::invalid_argument("can not parse uri from arrow file system config");
    }
  }
  options.endpoint_override = uri_parser.ToString();

  options.force_virtual_addressing = config.useVirtualHost;

  if (!config.region.empty()) {
    options.region = config.region;
  }

  options.request_timeout =
      config.requestTimeoutMs == 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC : config.requestTimeoutMs / 1000;

  return options;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> createCredentialsProvider(const std::string& provider_name) {
  if (provider_name == "aws") {
    return std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  } 
  if (provider_name == "aliyun") {
    return Aws::MakeShared<Aws::Auth::AliyunSTSAssumeRoleWebIdentityCredentialsProvider>(
        "AliyunSTSAssumeRoleWebIdentityCredentialsProvider");
  } 
  if (provider_name == "tencent") {
    return Aws::MakeShared<Aws::Auth::TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>(
        "TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider");
  }
  return nullptr;
}

Result<ArrowFileSystemPtr> S3FileSystemProducer::Make(const ArrowFileSystemConfig& config, std::string* out_path) {
  initS3(config);

  arrow::fs::S3Options options = createS3Options(config);

  // TODO support gcp iam
  if (config.useIAM && config.cloud_provider != "gcp") {
    auto provider = createCredentialsProvider(config.cloud_provider);
    auto credentials = provider->GetAWSCredentials();
    assert(!credentials.GetAWSAccessKeyId().empty());
    assert(!credentials.GetAWSSecretKey().empty());
    assert(!credentials.GetSessionToken().empty());
    options.credentials_provider = provider;
  } else {
    options.ConfigureAccessKey(config.access_key_id, config.access_key_value);
  }

  if (config.use_custom_part_upload) {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, MultiPartUploadS3FS::Make(options));
    return ArrowFileSystemPtr(fs);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::S3FileSystem::Make(options));
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
