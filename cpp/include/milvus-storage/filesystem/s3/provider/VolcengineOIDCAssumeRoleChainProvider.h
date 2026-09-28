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

#include "VolcengineCredentialsProvider.h"
#include "VolcengineSTSClient.h"

namespace milvus_storage {

// Two-step OIDC chain for cross-account TOS access on Volcengine:
//
//   1. AssumeRoleWithOIDC against the *machine-identity* role from env
//      (VOLCENGINE_OIDC_ROLE_TRN + VOLCENGINE_OIDC_TOKEN_FILE) — this yields
//      credentials for the big account (2100211764) that VKE trusts.
//   2. sts:AssumeRole into the customer-supplied target role (extfs.role_arn,
//      account B) using the step-1 STS creds as caller. This is the only hop
//      that crosses accounts; the customer's role trust policy must list the
//      step-1 (big-account) role as Principal.
//
// The single-step alternative — feeding the customer's role straight into
// AssumeRoleWithOIDC — is what the previous code did, and Volcengine STS
// rejects it whenever the OIDC IdP and RoleTrn live in different accounts (the
// cross-tenant case this provider exists for). use_iam (bucket-policy) callers
// never reach here, so that path is untouched.
//
// Degradation short-circuit: when the target role is empty or identical to the
// step-1 machine role (the single-account case — target role IS the big-account
// role), step 2 is skipped and the step-1 credentials are returned directly.
// This keeps the historical single-account role_arn deployment working without
// requiring the target role to trust itself for sts:AssumeRole.
//
// Mirrors the structure of AliyunOIDCAssumeRoleChainProvider.
class AWS_CORE_API VolcengineOIDCAssumeRoleChainProvider : public ::Aws::Auth::AWSCredentialsProvider {
  public:
  VolcengineOIDCAssumeRoleChainProvider(const ::Aws::String& target_role_trn,
                                        const ::Aws::String& target_session_name);

  ::Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();
  bool ExpiresSoon() const;

  // Step 1: env-driven AssumeRoleWithOIDC. The default-constructed inner
  // provider reads VOLCENGINE_OIDC_ROLE_TRN / VOLCENGINE_OIDC_TOKEN_FILE /
  // VOLCENGINE_OIDC_ROLE_SESSION_NAME and refreshes itself; we just call
  // GetAWSCredentials() on it each reload.
  ::Aws::UniquePtr<VolcengineSTSAssumeRoleWebIdentityCredentialsProvider> m_innerOidc;

  // Step 2: cross-account sts:AssumeRole signed with Volcengine V4.
  ::Aws::UniquePtr<VolcengineSTSCredentialsClient> m_stsClient;

  ::Aws::Auth::AWSCredentials m_credentials;
  ::Aws::String m_targetRoleTrn;
  ::Aws::String m_targetSessionName;
  // step-1 machine role (VOLCENGINE_OIDC_ROLE_TRN), captured at construction so
  // the degradation short-circuit can compare against it without re-reading env.
  ::Aws::String m_step1RoleTrn;
};

}  // namespace milvus_storage
