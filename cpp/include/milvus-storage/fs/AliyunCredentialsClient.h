#pragma once

#include <aws/core/internal/AWSHttpResourceClient.h>
namespace milvus_storage {
class AliyunSTSCredentialsClient : public Aws::Internal::AWSHttpResourceClient {
  public:
  /**
   * Initializes the provider to retrieve credentials from STS when it expires.
   */
  explicit AliyunSTSCredentialsClient(const Aws::Client::ClientConfiguration& clientConfiguration);

  AliyunSTSCredentialsClient& operator=(AliyunSTSCredentialsClient& rhs) = delete;
  AliyunSTSCredentialsClient(const AliyunSTSCredentialsClient& rhs) = delete;
  AliyunSTSCredentialsClient& operator=(AliyunSTSCredentialsClient&& rhs) = delete;
  AliyunSTSCredentialsClient(const AliyunSTSCredentialsClient&& rhs) = delete;

  // If you want to make an AssumeRoleWithWebIdentity call to sts. use these classes to pass data to and get info from
  // AliyunSTSCredentialsClient client. If you want to make an AssumeRole call to sts, define the request/result
  // members class/struct like this.
  struct STSAssumeRoleWithWebIdentityRequest {
    Aws::String roleSessionName;
    Aws::String roleArn;
    Aws::String webIdentityToken;
  };

  struct STSAssumeRoleWithWebIdentityResult {
    Aws::Auth::AWSCredentials creds;
  };

  STSAssumeRoleWithWebIdentityResult GetAssumeRoleWithWebIdentityCredentials(
      const STSAssumeRoleWithWebIdentityRequest& request);

  private:
  Aws::String m_endpoint;
  Aws::String m_aliyunOidcProviderArn;  // [aliyun]
};

}  // namespace milvus_storage