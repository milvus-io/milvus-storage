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

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include <aws/core/auth/AWSCredentialsProvider.h>

#include "milvus-storage/filesystem/s3/provider/credential_resolution.h"

namespace Aws::Http {
class HttpClient;
}

namespace Aws::STS {
class STSClient;
}

namespace milvus_storage {

/// AWS's default source order with a request-local typed dynamic tail.
///
/// The SDK default chain is kept for the providers which can return usable
/// credentials without this layer guessing their failure (environment,
/// profile/process and SSO).  IRSA, container credentials and IMDS are
/// resolved here because their SDK interfaces discard the HTTP failure and
/// return only an empty AWSCredentials value.
///
/// A declared IRSA or container source is authoritative.  If it fails we do
/// not silently switch the workload identity to IMDS.
class AwsDefaultCredentialsProvider final : public Aws::Auth::AWSCredentialsProvider,
                                            public RequestCredentialsResolver {
  public:
  enum class SourceMode { DefaultChain, WebIdentityOnly };

  struct Dependencies {
    // SDK providers before AssumeRoleWithWebIdentity in the default chain:
    // environment, then profile/process.
    std::vector<std::shared_ptr<Aws::Auth::AWSCredentialsProvider>> before_web_identity;
    // SDK providers after web identity but before container/IMDS (SSO today).
    std::vector<std::shared_ptr<Aws::Auth::AWSCredentialsProvider>> after_web_identity;
    // Optional test seams. Production creates bounded clients when null.
    std::shared_ptr<Aws::STS::STSClient> web_identity_client;
    std::shared_ptr<Aws::Http::HttpClient> metadata_client;
  };

  AwsDefaultCredentialsProvider();
  explicit AwsDefaultCredentialsProvider(SourceMode source_mode);
  explicit AwsDefaultCredentialsProvider(Dependencies dependencies,
                                         SourceMode source_mode = SourceMode::DefaultChain);

  Aws::Auth::AWSCredentials GetAWSCredentials() override;
  [[nodiscard]] arrow::Result<Aws::Auth::AWSCredentials> ResolveForRequest() override;

  private:
  enum class DynamicSource { None, WebIdentity, Container, Imds };

  static Dependencies MakeDefaultDependencies();

  bool CachedDynamicCredentialsAreFresh(DynamicSource source) const;
  arrow::Result<Aws::Auth::AWSCredentials> ResolveWebIdentity();
  arrow::Result<Aws::Auth::AWSCredentials> ResolveContainerCredentials();
  arrow::Result<Aws::Auth::AWSCredentials> ResolveImdsCredentials();
  arrow::Result<Aws::Auth::AWSCredentials> ResolveDynamic(DynamicSource source);

  Dependencies dependencies_;
  SourceMode source_mode_;
  std::mutex mutex_;
  Aws::Auth::AWSCredentials dynamic_credentials_;
  DynamicSource dynamic_source_ = DynamicSource::None;
  arrow::Status last_dynamic_resolution_;
};

}  // namespace milvus_storage
