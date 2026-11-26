#pragma once

#include "TencentCloudCredentialsProvider.h"

namespace milvus_storage {

// TKE OIDC establishes the caller identity; a signed AssumeRole selects the
// explicit data role. Each filesystem owns and refreshes its own target identity.
class AWS_CORE_API TencentCloudOIDCAssumeRoleChainProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  TencentCloudOIDCAssumeRoleChainProvider(const Aws::String& target_role_arn,
                                          const Aws::String& target_session_name,
                                          const Aws::String& target_external_id = "");
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  bool ExpiresSoon() const;
  Aws::UniquePtr<TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider> m_innerOidc;
  Aws::UniquePtr<TencentCloudSTSCredentialsClient> m_stsClient;
  Aws::Auth::AWSCredentials m_credentials;
  Aws::String m_region;
  Aws::String m_targetRoleArn;
  Aws::String m_targetSessionName;
  Aws::String m_targetExternalId;
};

}  // namespace milvus_storage
