
#include <aws/core/auth/AWSCredentialsProvider.h>

#include "HuaweiCloudSTSClient.h"

namespace milvus_storage {

class HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider();
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();
  Aws::String CalculateQueryString() const;

  Aws::UniquePtr<HuaweiCloudSTSCredentialsClient> m_client;
  Aws::Auth::AWSCredentials m_credentials;
  Aws::String m_region;
  Aws::String m_providerId;
  Aws::String m_roleArn;
  Aws::String m_tokenFile;
  Aws::String m_sessionName;
  Aws::String m_token;
  bool m_initialized;
  bool ExpiresSoon() const;
};

}  // namespace milvus_storage
