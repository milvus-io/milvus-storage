// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <arrow/result.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/utils/DateTime.h>

#include "milvus-storage/filesystem/credential_broker.h"

namespace milvus_storage {

// Two authentication modes share the broker endpoint, region, lifetime and timeout:
// - AssumeRole (use_iam=false): csp="aws", required role_arn and optional external_id.
// - Pod IAM (use_iam=true): csp="aws"/"ali"/"tc" for AWS/Aliyun/Tencent. Credentials
//   belong to the broker Pod; role_arn and external_id are ignored and never sent.
struct AwsCredentialBrokerConfig {
  std::string endpoint;
  std::string region;

  // Target identity for AssumeRole only; the IAM factory leaves both fields empty.
  std::string role_arn;
  std::string external_id;

  // Requested lifetime in seconds; filesystem configuration accepts [900, 3600].
  // Cache refresh follows the response's expiredAt rather than this requested duration.
  int32_t duration_seconds = 900;
  // Timeout per HTTP attempt, including each retry of an initial credential fetch.
  int64_t request_timeout_ms = 3000;
  bool use_iam = false;
  std::string csp = credential_broker::kCspAws;

  bool operator==(const AwsCredentialBrokerConfig& other) const = default;
};

// Uses the same synchronous, cached provider contract as STS AssumeRole.
// S3-compatible clouds share the temporary credential response contract.
// IAM selects the broker's own AWS, Aliyun or Tencent identity; otherwise the
// existing AWS STS request selects a target role.
class AwsCredentialBrokerProvider final : public Aws::Auth::AWSCredentialsProvider {
  public:
  using Clock = std::function<Aws::Utils::DateTime()>;
  explicit AwsCredentialBrokerProvider(AwsCredentialBrokerConfig config, Clock clock = {});
  Aws::Auth::AWSCredentials GetAWSCredentials() override;
  arrow::Result<Aws::Auth::AWSCredentials> GetCredentials();
  bool Equals(const AwsCredentialBrokerProvider& other) const;

  private:
  arrow::Result<Aws::Auth::AWSCredentials> FetchCredentials() const;
  const AwsCredentialBrokerConfig config_;
  const Clock clock_;
  std::shared_ptr<Aws::Http::HttpClient> client_;
  std::mutex mutex_;
  Aws::Auth::AWSCredentials credentials_;
};

}  // namespace milvus_storage
