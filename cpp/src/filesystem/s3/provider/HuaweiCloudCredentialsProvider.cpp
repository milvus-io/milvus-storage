#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include <fstream>
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudSTSClient.h"
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/client/SpecifiedRetryableErrorsRetryStrategy.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {

static const char STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG[] =
    "MilvusStorage-HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider";
static const int STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD = 180 * 1000;  // huawei cloud support 180s.

HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider()
    : m_initialized(false) {
  m_region = Aws::Environment::GetEnv("HUAWEICLOUD_SDK_REGION");
  m_roleArn = Aws::Environment::GetEnv("HUAWEICLOUD_SDK_PROJECT_ID");
  m_tokenFile = Aws::Environment::GetEnv("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
  m_providerId = Aws::Environment::GetEnv("HUAWEICLOUD_SDK_IDP_ID");
  auto currentTimePoint = std::chrono::high_resolution_clock::now();
  auto nanoseconds = std::chrono::time_point_cast<std::chrono::nanoseconds>(currentTimePoint);
  auto timestamp = nanoseconds.time_since_epoch().count();
  m_sessionName = "huaweicloud-cpp-sdk-" + std::to_string(timestamp / 1000);

  if (m_roleArn.empty() || m_tokenFile.empty() || m_region.empty()) {
    auto profile = Aws::Config::GetCachedConfigProfile(Aws::Auth::GetConfigProfileName());
    m_roleArn = profile.GetRoleArn();
    m_tokenFile = profile.GetValue("web_identity_token_file");
    m_sessionName = profile.GetValue("role_session_name");
  }

  if (m_tokenFile.empty()) {
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "Token file must be specified to use STS AssumeRole "
                       "web identity creds provider.");
    return;  // No need to do further constructing
  } else {
    AWS_LOGSTREAM_DEBUG(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                        "Resolved token_file from profile_config or "
                        "environment variable to be "
                            << m_tokenFile);
  }

  if (m_roleArn.empty()) {
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "RoleArn must be specified to use STS AssumeRole "
                       "web identity creds provider.");
    return;  // No need to do further constructing
  } else {
    AWS_LOGSTREAM_DEBUG(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                        "Resolved role_arn from profile_config or "
                        "environment variable to be "
                            << m_roleArn);
  }

  if (m_region.empty()) {
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "Region must be specified to use STS AssumeRole "
                       "web identity creds provider.");
    return;  // No need to do further constructing
  } else {
    AWS_LOGSTREAM_DEBUG(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                        "Resolved region from profile_config or "
                        "environment variable to be "
                            << m_region);
  }

  if (m_sessionName.empty()) {
    m_sessionName = Aws::Utils::UUID::RandomUUID();
  } else {
    AWS_LOGSTREAM_DEBUG(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                        "Resolved session_name from profile_config or "
                        "environment variable to be "
                            << m_sessionName);
  }

  Aws::Client::ClientConfiguration config;
  config.scheme = Aws::Http::Scheme::HTTPS;
  config.region = m_region;

  Aws::Vector<Aws::String> retryableErrors;
  retryableErrors.push_back("IDPCommunicationError");
  retryableErrors.push_back("InvalidIdentityToken");

  config.retryStrategy = Aws::MakeShared<Aws::Client::SpecifiedRetryableErrorsRetryStrategy>(
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, retryableErrors, 3 /*maxRetries*/);

  m_client = Aws::MakeUnique<HuaweiCloudSTSCredentialsClient>(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
  m_initialized = true;
  AWS_LOGSTREAM_INFO(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                     "Initialized STS AssumeRole with web identity creds provider."
                         << " region=" << m_region << ", tokenFile=" << m_tokenFile << ", providerId=" << m_providerId
                         << ", gracePeriodMs=" << STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD
                         << ", cooldownNormalSec=" << RELOAD_COOLDOWN_SECONDS
                         << ", cooldownUrgentSec=" << RELOAD_COOLDOWN_SECONDS_URGENT);
}

Aws::Auth::AWSCredentials HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
  if (!m_initialized) {
    return Aws::Auth::AWSCredentials();
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  // Do not return fully expired credentials — the caller would get silent
  // auth failures. Return empty credentials instead so the error surfaces
  // immediately rather than after an HTTP round-trip.
  if (!m_credentials.IsEmpty() && m_credentials.IsExpired()) {
    AWS_LOGSTREAM_WARN(
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
        "Cached credentials have fully expired; returning empty credentials to avoid silent auth failures.");
    return Aws::Auth::AWSCredentials();
  }
  return m_credentials;
}

void HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::Reload() {
  if (m_credentials.IsEmpty()) {
    AWS_LOGSTREAM_INFO(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, "Performing initial credential load from STS.");
  } else {
    AWS_LOGSTREAM_INFO(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "Credentials expiring soon, attempting to refresh from STS.");
  }

  Aws::IFStream tokenFile(m_tokenFile.c_str());
  if (tokenFile) {
    Aws::String token((std::istreambuf_iterator<char>(tokenFile)), std::istreambuf_iterator<char>());
    if (!token.empty() && token.back() == '\n') {
      token.pop_back();
    }
    m_token = token;
  } else {
    ++m_stsFailureCount;
    AWS_LOGSTREAM_ERROR(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                        "Can't open token file: " << m_tokenFile << ", sts_success=" << m_stsSuccessCount.load()
                                                  << ", sts_failure=" << m_stsFailureCount.load());
    m_lastReloadFailed = true;
    m_lastFailedReloadTime = std::chrono::steady_clock::now();
    return;
  }
  HuaweiCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{m_region, m_providerId, m_token,
                                                                               m_roleArn, m_sessionName};

  // GetAssumeRoleWithWebIdentityCredentials catches all exceptions internally
  // and returns result.success=false on any failure.
  auto result = m_client->GetAssumeRoleWithWebIdentityCredentials(request);

  const auto& creds = result.creds;

  if (!result.success) {
    ++m_stsFailureCount;
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "STS call failed. has_valid_cached=" << hasExisting << ", retaining existing credentials."
                                                            << " sts_success=" << m_stsSuccessCount.load()
                                                            << ", sts_failure=" << m_stsFailureCount.load());
    m_lastReloadFailed = true;
    m_lastFailedReloadTime = std::chrono::steady_clock::now();
    return;
  }

  if (creds.GetAWSAccessKeyId().empty() || creds.GetAWSSecretKey().empty() || creds.GetSessionToken().empty()) {
    ++m_stsFailureCount;
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "STS returned incomplete credentials (missing ak/sk/token). has_valid_cached="
                           << hasExisting << ", retaining existing credentials."
                           << " sts_success=" << m_stsSuccessCount.load()
                           << ", sts_failure=" << m_stsFailureCount.load());
    m_lastReloadFailed = true;
    m_lastFailedReloadTime = std::chrono::steady_clock::now();
    return;
  }

  ++m_stsSuccessCount;
  m_credentials = creds;
  m_lastReloadFailed = false;
  auto akId = creds.GetAWSAccessKeyId();
  Aws::String akPrefix = akId.length() > 4 ? akId.substr(0, 4) + "***" : akId;
  auto expiresInMs = (creds.GetExpiration() - Aws::Utils::DateTime::Now()).count();
  AWS_LOGSTREAM_INFO(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                     "Successfully retrieved credentials, ak_prefix=" << akPrefix << ", expires_in_ms=" << expiresInMs
                                                                      << ", region=" << m_region
                                                                      << ", sts_success=" << m_stsSuccessCount.load()
                                                                      << ", sts_failure=" << m_stsFailureCount.load());
}

bool HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() <
          STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

bool HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::IsInCooldown() const {
  if (!m_lastReloadFailed) {
    return false;
  }
  // Use shorter cooldown when credentials are empty or expired (urgent),
  // longer cooldown when credentials are still valid (not urgent).
  int cooldownSeconds =
      (m_credentials.IsEmpty() || m_credentials.IsExpired()) ? RELOAD_COOLDOWN_SECONDS_URGENT : RELOAD_COOLDOWN_SECONDS;
  auto elapsed = std::chrono::steady_clock::now() - m_lastFailedReloadTime;
  return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() < cooldownSeconds;
}

void HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::RefreshIfExpired() {
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
    return;
  }

  guard.UpgradeToWriterLock();
  if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
    return;
  }

  if (IsInCooldown()) {
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    AWS_LOGSTREAM_WARN(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                       "Skipping credential reload — in cooldown after previous failure."
                           << " has_valid_cached=" << hasExisting);
    return;
  }

  Reload();
}

}  // namespace milvus_storage
