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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/memory/stl/AWSString.h>

#include "milvus-storage/filesystem/s3/provider/credential_resolution.h"

namespace Aws::STS {
class STSClient;
}

namespace milvus_storage {

/// AWS AssumeRole provider whose failure remains an arrow::Status.
///
/// The AWS SDK provider caches correctly but logs and discards the
/// AssumeRoleOutcome error. This implementation preserves the same request
/// fields, credential lifetime and 60-second refresh grace while returning a
/// request-local typed failure to the S3 holders.
class AwsSTSAssumeRoleCredentialsProvider : public Aws::Auth::AWSCredentialsProvider,
                                            public RequestCredentialsResolver {
  public:
  AwsSTSAssumeRoleCredentialsProvider(const Aws::String& role_arn,
                                      const Aws::String& session_name,
                                      const Aws::String& external_id,
                                      int load_frequency,
                                      std::shared_ptr<Aws::STS::STSClient> sts_client = nullptr,
                                      std::shared_ptr<RequestCredentialsResolver> source_resolver = nullptr);

  Aws::Auth::AWSCredentials GetAWSCredentials() override;
  [[nodiscard]] arrow::Result<Aws::Auth::AWSCredentials> ResolveForRequest() override;

  private:
  bool RefreshRequiredLocked() const;
  void ReloadLocked();

  std::shared_ptr<Aws::STS::STSClient> sts_client_;
  // Present for the production-created STS client (and injectable in tests).
  // Resolve it before AssumeRole so a failed IRSA/IMDS refresh cannot degrade
  // into an unsigned target STS request.
  std::shared_ptr<RequestCredentialsResolver> source_resolver_;
  Aws::Auth::AWSCredentials credentials_;
  arrow::Status last_resolution_;
  Aws::String role_arn_;
  Aws::String session_name_;
  Aws::String external_id_;
  int load_frequency_;
  std::mutex mutex_;
};

}  // namespace milvus_storage
