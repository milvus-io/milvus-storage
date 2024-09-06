#pragma once
#include "aws/core/auth/AWSCredentials.h"
#include "aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h"
#include "fs/AliyunCredentialsClient.h"

namespace milvus_storage {

class AliyunSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  AliyunSTSAssumeRoleWebIdentityCredentialsProvider();

  /**
   * Retrieves the credentials if found, otherwise returns empty credential set.
   */
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();
  Aws::String CalculateQueryString() const;

  Aws::UniquePtr<AliyunSTSCredentialsClient> m_client;
  Aws::Auth::AWSCredentials m_credentials;
  Aws::String m_roleArn;
  Aws::String m_tokenFile;
  Aws::String m_sessionName;
  Aws::String m_token;
  bool m_initialized;
  bool ExpiresSoon() const;
};
}  // namespace milvus_storage