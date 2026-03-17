#pragma once

#include <atomic>
#include <chrono>
#include <aws/core/auth/AWSCredentialsProvider.h>

#include "HuaweiCloudSTSClient.h"

namespace milvus_storage {

class HuaweiCloudCredentialsProviderTestHelper;

class HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  friend class HuaweiCloudCredentialsProviderTestHelper;

  public:
  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider();
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void RefreshIfExpired();

  Aws::UniquePtr<HuaweiCloudSTSCredentialsClient> m_client;
  Aws::Auth::AWSCredentials m_credentials;
  Aws::String m_region;
  Aws::String m_providerId;
  Aws::String m_roleArn;
  Aws::String m_tokenFile;
  Aws::String m_sessionName;
  Aws::String m_token;
  bool m_initialized;
  bool m_lastReloadFailed = false;
  std::chrono::steady_clock::time_point m_lastFailedReloadTime;
  std::atomic<int64_t> m_stsSuccessCount{0};
  std::atomic<int64_t> m_stsFailureCount{0};
  static constexpr int RELOAD_COOLDOWN_SECONDS = 30;
  static constexpr int RELOAD_COOLDOWN_SECONDS_URGENT = 5;

  bool ExpiresSoon() const;
  bool IsInCooldown() const;
};

}  // namespace milvus_storage
