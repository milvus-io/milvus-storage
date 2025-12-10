#pragma once

#include <aws/core/internal/AWSHttpResourceClient.h>

namespace milvus_storage {

class AWS_CORE_API HuaweiCloudSTSCredentialsClient : public ::Aws::Internal::AWSHttpResourceClient {
  public:
  explicit HuaweiCloudSTSCredentialsClient(const Aws::Client::ClientConfiguration& clientConfiguration);

  HuaweiCloudSTSCredentialsClient& operator=(HuaweiCloudSTSCredentialsClient& rhs) = delete;
  HuaweiCloudSTSCredentialsClient(const HuaweiCloudSTSCredentialsClient& rhs) = delete;
  HuaweiCloudSTSCredentialsClient& operator=(HuaweiCloudSTSCredentialsClient&& rhs) = delete;
  HuaweiCloudSTSCredentialsClient(const HuaweiCloudSTSCredentialsClient&& rhs) = delete;

  struct STSAssumeRoleWithWebIdentityRequest {
    Aws::String region;
    Aws::String providerId;
    Aws::String webIdentityToken;
    Aws::String roleArn;
    Aws::String roleSessionName;
  };

  struct STSAssumeRoleWithWebIdentityResult {
    Aws::Auth::AWSCredentials creds;
  };

  STSAssumeRoleWithWebIdentityResult GetAssumeRoleWithWebIdentityCredentials(
      const STSAssumeRoleWithWebIdentityRequest& request);

  private:
  Aws::String m_endpoint;
  std::shared_ptr<Aws::Http::HttpClient> m_httpClient;

  struct STSCallResult {
    bool success;
    Aws::Auth::AWSCredentials credentials;
    Aws::String errorMessage;
  };

  STSCallResult callHuaweiCloudSTS(const Aws::String& userToken, const STSAssumeRoleWithWebIdentityRequest& request);
};
}  // namespace milvus_storage
