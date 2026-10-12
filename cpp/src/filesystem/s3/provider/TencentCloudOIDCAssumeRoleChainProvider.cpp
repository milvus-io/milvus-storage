#include "milvus-storage/filesystem/s3/provider/TencentCloudOIDCAssumeRoleChainProvider.h"

#include <aws/core/platform/Environment.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {
namespace {
constexpr const char* kLogTag = "TencentCloudOIDCAssumeRoleChainProvider";
constexpr int kRefreshGraceMs = 180 * 1000;
}  // namespace

TencentCloudOIDCAssumeRoleChainProvider::TencentCloudOIDCAssumeRoleChainProvider(const Aws::String& target_role_arn,
                                                                                 const Aws::String& target_session_name,
                                                                                 const Aws::String& target_external_id)
    : m_region(Aws::Environment::GetEnv("TKE_REGION")),
      m_targetRoleArn(target_role_arn),
      m_targetSessionName(target_session_name),
      m_targetExternalId(target_external_id) {
  if (m_targetRoleArn.empty() || m_region.empty() || Aws::Environment::GetEnv("TKE_ROLE_ARN").empty() ||
      Aws::Environment::GetEnv("TKE_WEB_IDENTITY_TOKEN_FILE").empty() ||
      Aws::Environment::GetEnv("TKE_PROVIDER_ID").empty())
    return;
  if (m_targetSessionName.empty())
    m_targetSessionName = Aws::Utils::UUID::RandomUUID();
  m_innerOidc = Aws::MakeUnique<TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider>(kLogTag);
  Aws::Client::ClientConfiguration config(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  config.scheme = Aws::Http::Scheme::HTTPS;
  config.region = m_region;
  m_stsClient = Aws::MakeUnique<TencentCloudSTSCredentialsClient>(kLogTag, config);
}

bool TencentCloudOIDCAssumeRoleChainProvider::ExpiresSoon() const {
  return (m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() < kRefreshGraceMs;
}

Aws::Auth::AWSCredentials TencentCloudOIDCAssumeRoleChainProvider::GetAWSCredentials() {
  if (!m_innerOidc)
    return {};
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (m_credentials.IsExpiredOrEmpty() || ExpiresSoon()) {
    guard.UpgradeToWriterLock();
    if (m_credentials.IsExpiredOrEmpty() || ExpiresSoon())
      Reload();
  }
  // A failed refresh may retain a still-valid target credential, never an
  // expired one or the more privileged machine identity from the first hop.
  return m_credentials.IsExpiredOrEmpty() ? Aws::Auth::AWSCredentials{} : m_credentials;
}

void TencentCloudOIDCAssumeRoleChainProvider::Reload() {
  const auto caller = m_innerOidc->GetAWSCredentials();
  if (caller.IsExpiredOrEmpty() || caller.GetSessionToken().empty())
    return;
  TencentCloudSTSCredentialsClient::STSAssumeRoleRequest request{caller, m_region, m_targetRoleArn, m_targetSessionName,
                                                                 m_targetExternalId};
  auto credentials = m_stsClient->GetAssumeRoleCredentials(request);
  if (!credentials.IsExpiredOrEmpty())
    m_credentials = std::move(credentials);
}
}  // namespace milvus_storage
