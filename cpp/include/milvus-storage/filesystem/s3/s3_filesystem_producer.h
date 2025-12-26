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

#pragma once

#include <aws/core/Aws.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <arrow/util/uri.h>
#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/internal/oauth2_google_credentials.h>
#include <google/cloud/storage/oauth2/compute_engine_credentials.h>
#include <google/cloud/storage/oauth2/google_credentials.h>
#include <google/cloud/status_or.h>
#include <cstdlib>
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

namespace milvus_storage {

class S3FileSystemProducer : public FileSystemProducer {
  public:
  S3FileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  arrow::Result<ArrowFileSystemPtr> Make() override;

  arrow::Result<S3Options> CreateS3Options();

  void InitS3();

  private:
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAwsCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAliyunCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateTencentCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateHuaweiCredentialsProvider();

  private:
  const ArrowFileSystemConfig config_;
};

}  // namespace milvus_storage
