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

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSString.h>

#include "AliyunRAMSTSClient.h"

namespace milvus_storage {

// Credentials provider that bridges:
//     ECS IMDS (the instance's attached RAM role)
//         -> sts:AssumeRole (customer's target role)
//         -> short-lived creds for accessing the customer's OSS bucket.
//
// No ALIBABA_CLOUD_OIDC_* env vars required — the caller identity comes from
// the ECS metadata service. Selected by
// ALIYUN_ROLE_ARN_AUTH_MODE=ram at the dispatch layer; the
// existing OIDC (AssumeRoleWithOIDC) provider is used otherwise so OIDC
// deployments are unaffected.
class AWS_CORE_API AliyunRAMCredentialsProvider : public ::Aws::Auth::AWSCredentialsProvider {
  public:
  AliyunRAMCredentialsProvider(const ::Aws::String& role_arn, const ::Aws::String& role_session_name);

  ::Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();
  bool ExpiresSoon() const;

  ::Aws::UniquePtr<AliyunRAMSTSClient> m_stsClient;
  ::Aws::Auth::AWSCredentials m_credentials;
  ::Aws::String m_roleArn;
  ::Aws::String m_roleSessionName;
};

}  // namespace milvus_storage
