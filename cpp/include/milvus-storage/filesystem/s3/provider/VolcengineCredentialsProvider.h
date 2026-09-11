#pragma once

#include <aws/core/auth/AWSCredentialsProvider.h>

#include "VolcengineSTSClient.h"

namespace milvus_storage {

class VolcengineSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  // Env-only ctor: RoleTrn / token file / session are all read from the
  // VOLCENGINE_OIDC_* environment variables. Backs the use_iam=true path,
  // where VKE injects the machine identity and target role via env.
  VolcengineSTSAssumeRoleWebIdentityCredentialsProvider();

  // Per-tenant ctor: RoleTrn (and optionally session name) are supplied by the
  // caller, typically from an external-table spec (extfs.role_arn). The OIDC
  // web-identity token file is still read from VOLCENGINE_OIDC_TOKEN_FILE — it
  // is the VKE-injected machine identity and never appears in a user spec.
  VolcengineSTSAssumeRoleWebIdentityCredentialsProvider(const Aws::String& role_arn,
                                                        const Aws::String& session_name);
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
  bool m_initialized;
  bool ExpiresSoon() const;
};

}  // namespace milvus_storage
