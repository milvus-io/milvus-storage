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

#include <aws/core/auth/AWSCredentialsProvider.h>

#include "VolcengineSTSClient.h"

namespace milvus_storage {

class VolcengineSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  // Env-only ctor: RoleTrn / token file / session are all read from the
  // VOLCENGINE_OIDC_* environment variables. Backs the use_iam=true path,
  // where VKE injects the machine identity and target role via env.
  // `duration_seconds` is plumbed to VolcengineSTSCredentialsClient and
  // becomes the AssumeRoleWithOIDC DurationSeconds. Callers must supply the
  // active FileSystemConfig::load_frequency; the inner client clamps the
  // value into Volcengine STS's [900, 43200] window.
  explicit VolcengineSTSAssumeRoleWebIdentityCredentialsProvider(int duration_seconds);

  // Per-tenant ctor: RoleTrn (and optionally session name) are supplied by the
  // caller, typically from an external-table spec (extfs.role_arn). The OIDC
  // web-identity token file is still read from VOLCENGINE_OIDC_TOKEN_FILE — it
  // is the VKE-injected machine identity and never appears in a user spec.
  VolcengineSTSAssumeRoleWebIdentityCredentialsProvider(const Aws::String& role_arn,
                                                        const Aws::String& session_name,
                                                        int duration_seconds);
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  // Shared setup for both ctors: validates token file / role arn, defaults the
  // session name, and builds the STS client. Sets m_initialized on success.
  void InitClient();
  void RefreshIfExpired();
  Aws::String CalculateQueryString() const;

  Aws::UniquePtr<VolcengineSTSCredentialsClient> m_client;
  Aws::Auth::AWSCredentials m_credentials;
  Aws::String m_roleArn;
  Aws::String m_tokenFile;
  Aws::String m_sessionName;
  Aws::String m_token;
  int m_durationSeconds;
  bool m_initialized;
  bool ExpiresSoon() const;
};

}  // namespace milvus_storage
