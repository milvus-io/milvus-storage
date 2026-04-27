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

#include "AliyunCredentialsProvider.h"
#include "AliyunRAMSTSClient.h"

namespace milvus_storage {

// Two-step OIDC chain for cross-account OSS access:
//
//   1. AssumeRoleWithOIDC against the *machine-identity* role from env
//      (ALIBABA_CLOUD_ROLE_ARN + ALIBABA_CLOUD_OIDC_PROVIDER_ARN +
//      ALIBABA_CLOUD_OIDC_TOKEN_FILE) — same account, the only shape Aliyun
//      STS accepts for AssumeRoleWithOIDC.
//   2. sts:AssumeRole into the customer-supplied target role using the STS
//      creds from step 1 as caller. This is the only step that crosses
//      accounts; the customer's role trust policy must list the step-1 role
//      as Principal.
//
// The single-step alternative — feeding the customer's role straight into
// AssumeRoleWithOIDC alongside the env's OIDCProviderArn — is what the
// previous code did and Aliyun rejects it with AssumeRolePolicy / ImplicitDeny
// because RoleArn and OIDCProviderArn must share an account.
//
// Mirrors the structure of AliyunRAMCredentialsProvider (IMDS step 1, same
// step 2). Selected for OIDC deployments at the dispatch layer; RAM mode
// (ALIYUN_ROLE_ARN_AUTH_MODE=ram) keeps using AliyunRAMCredentialsProvider.
class AWS_CORE_API AliyunOIDCAssumeRoleChainProvider : public ::Aws::Auth::AWSCredentialsProvider {
  public:
  // `target_external_id` is forwarded to step 2 (sts:AssumeRole). Aliyun's
  // AssumeRoleWithOIDC API itself has no ExternalId concept, so step 1 never
  // sees it. Empty means no ExternalId is sent (the parameter remains
  // optional from the target role's trust policy perspective).
  AliyunOIDCAssumeRoleChainProvider(const ::Aws::String& target_role_arn,
                                    const ::Aws::String& target_session_name,
                                    const ::Aws::String& target_external_id = "");

  ::Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();
  bool ExpiresSoon() const;

  // Step 1: env-driven OIDC. Default-constructed provider reads
  // ALIBABA_CLOUD_ROLE_ARN / ALIBABA_CLOUD_OIDC_TOKEN_FILE / etc. and refreshes
  // itself; we just call GetAWSCredentials() on it each time we reload.
  ::Aws::UniquePtr<AliyunSTSAssumeRoleWebIdentityCredentialsProvider> m_innerOidc;

  // Step 2: cross-account AssumeRole. Same client used by RAM mode.
  ::Aws::UniquePtr<AliyunRAMSTSClient> m_stsClient;

  ::Aws::Auth::AWSCredentials m_credentials;
  ::Aws::String m_targetRoleArn;
  ::Aws::String m_targetSessionName;
  ::Aws::String m_targetExternalId;
};

}  // namespace milvus_storage
