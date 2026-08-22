#pragma once

#include <atomic>
#include <chrono>
#include <aws/core/auth/AWSCredentialsProvider.h>

#include "HuaweiCloudSTSClient.h"

#include "milvus-storage/filesystem/s3/provider/credential_resolution.h"

namespace milvus_storage {

class HuaweiCloudCredentialsProviderTestHelper;

class HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider,
                                                               public RequestCredentialsResolver {
  friend class HuaweiCloudCredentialsProviderTestHelper;

  public:
  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider();
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  [[nodiscard]] arrow::Result<Aws::Auth::AWSCredentials> ResolveForRequest() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();

  Aws::UniquePtr<HuaweiCloudSTSCredentialsClient> m_client;
  Aws::Auth::AWSCredentials m_credentials;
  // Guarded by m_reloadLock, like m_credentials.
  arrow::Status m_lastResolution;
  Aws::String m_region;
  Aws::String m_providerId;
  Aws::String m_roleArn;
  Aws::String m_tokenFile;
  Aws::String m_sessionName;
  Aws::String m_token;
  bool m_initialized;
  std::atomic<int64_t> m_stsSuccessCount{0};
  std::atomic<int64_t> m_stsFailureCount{0};

  bool ExpiresSoon() const;
};

}  // namespace milvus_storage
