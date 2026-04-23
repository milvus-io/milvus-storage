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
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/internal/AWSHttpResourceClient.h>

namespace milvus_storage {

// STS client for Aliyun sts:AssumeRole using POP v1 (HMAC-SHA1) signing.
// Kept separate from AliyunSTSCredentialsClient (OIDC / AssumeRoleWithOIDC)
// so the two auth paths share no state — changes on the RAM side cannot
// regress the existing OIDC flow. Paired with AliyunRAMCredentialsProvider.
class AWS_CORE_API AliyunRAMSTSClient : public ::Aws::Internal::AWSHttpResourceClient {
  public:
  explicit AliyunRAMSTSClient(const ::Aws::Client::ClientConfiguration& clientConfiguration);

  AliyunRAMSTSClient(const AliyunRAMSTSClient&) = delete;
  AliyunRAMSTSClient& operator=(const AliyunRAMSTSClient&) = delete;
  AliyunRAMSTSClient(AliyunRAMSTSClient&&) = delete;
  AliyunRAMSTSClient& operator=(AliyunRAMSTSClient&&) = delete;

  struct AssumeRoleRequest {
    // Caller identity used to sign the AssumeRole request. When the caller
    // is ECS IMDS, all three fields are populated from the STS creds that
    // IMDS returned for the instance's attached RAM role.
    ::Aws::String callerAccessKeyId;
    ::Aws::String callerAccessKeySecret;
    ::Aws::String callerSecurityToken;
    ::Aws::String roleArn;
    ::Aws::String roleSessionName;
  };

  struct AssumeRoleResult {
    ::Aws::Auth::AWSCredentials creds;
  };

  // Returns empty credentials on failure; errors are logged.
  AssumeRoleResult GetAssumeRoleCredentials(const AssumeRoleRequest& request);

  private:
  ::Aws::String m_endpoint;
};

}  // namespace milvus_storage
