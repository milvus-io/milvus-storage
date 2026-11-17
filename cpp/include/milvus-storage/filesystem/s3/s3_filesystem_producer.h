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
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

namespace milvus_storage {

/// Shared AWS credential-source policy for storage filesystems and external
/// callers such as Milvus ChunkManager.
/// Broker and explicit target-role providers are selected separately.
///
/// IAM mode ignores ambient AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and
/// AWS_SESSION_TOKEN. Select one source at construction: configured STS Web
/// Identity (IRSA), otherwise a configured container endpoint (EKS Pod Identity
/// or ECS task role), otherwise EC2 instance credentials unless the standard
/// AWS_EC2_METADATA_DISABLED equals "true", ignoring case.
/// Web Identity settings retain the SDK's environment/config-profile resolution;
/// no static environment, static-profile or SSO provider is added. Retrieval
/// failure never switches a configured workload identity to the node's role.
/// Construct and destroy this provider within the initialized AWS SDK lifetime.
class AwsCredentialsProviderChain final : public Aws::Auth::AWSCredentialsProviderChain {
  public:
  AwsCredentialsProviderChain();
};

class S3FileSystemProducer : public FileSystemProducer {
  public:
  S3FileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  arrow::Result<ArrowFileSystemPtr> Make() override;

  arrow::Result<S3Options> CreateS3Options();

  arrow::Status InitS3();

  private:
  // Select broker, AssumeRole, IAM, or explicit credentials in priority order.
  arrow::Status ConfigureCredentials(S3Options& options);
  arrow::Status ConfigureAssumeRoleCredentials(S3Options& options) const;
  arrow::Status ConfigureAliyunAssumeRoleCredentials(S3Options& options) const;
  arrow::Status ConfigureVolcengineAssumeRoleCredentials(S3Options& options) const;

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAwsCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAliyunCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateTencentCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateHuaweiCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateVolcengineCredentialsProvider();

  private:
  const ArrowFileSystemConfig config_;
};

}  // namespace milvus_storage
