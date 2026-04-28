#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"

#include "milvus-storage/common/log.h"

#include <fstream>
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudSTSClient.h"
#include <aws/core/platform/Environment.h>
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
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] Token file must be specified to use STS AssumeRole web identity creds "
        "provider.",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return;  // No need to do further constructing
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved token_file from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_tokenFile);
  }

  if (m_roleArn.empty()) {
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] RoleArn must be specified to use STS AssumeRole web identity creds "
        "provider.",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return;  // No need to do further constructing
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved role_arn from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_roleArn);
  }

  if (m_region.empty()) {
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] Region must be specified to use STS AssumeRole web identity creds "
        "provider.",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return;  // No need to do further constructing
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved region from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_region);
  }

  if (m_sessionName.empty()) {
    m_sessionName = Aws::Utils::UUID::RandomUUID();
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved session_name from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_sessionName);
  }

  Aws::Client::ClientConfiguration config;
  config.scheme = Aws::Http::Scheme::HTTPS;
  config.region = m_region;

  Aws::Vector<Aws::String> retryableErrors;
  retryableErrors.emplace_back("IDPCommunicationError");
  retryableErrors.emplace_back("InvalidIdentityToken");

  config.retryStrategy = Aws::MakeShared<Aws::Client::SpecifiedRetryableErrorsRetryStrategy>(
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, retryableErrors, 3 /*maxRetries*/);

  m_client = Aws::MakeUnique<HuaweiCloudSTSCredentialsClient>(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
  m_initialized = true;
  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Initialized STS AssumeRole with web identity creds provider. region={}, "
      "tokenFile={}, providerId={}, gracePeriodMs={}, cooldownNormalSec={}, "
      "cooldownUrgentSec={}",
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_region, m_tokenFile, m_providerId,
      STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD, RELOAD_COOLDOWN_SECONDS, RELOAD_COOLDOWN_SECONDS_URGENT);
}

Aws::Auth::AWSCredentials HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
  if (!m_initialized) {
    return {};
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  // Do not return fully expired credentials — the caller would get silent
  // auth failures. Return empty credentials instead so the error surfaces
  // immediately rather than after an HTTP round-trip.
  if (!m_credentials.IsEmpty() && m_credentials.IsExpired()) {
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] Cached credentials have fully expired; returning empty credentials to "
        "avoid silent auth failures.",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return {};
  }
  return m_credentials;
}

void HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::Reload() {
  if (m_credentials.IsEmpty()) {
    LOG_STORAGE_INFO_ << fmt::format("[{}] Performing initial credential load from STS.",
                                     STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
  } else {
    LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials expiring soon, attempting to refresh from STS.",
                                     STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
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
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Can't open token file: {}, sts_success={}, sts_failure={}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_tokenFile, m_stsSuccessCount.load(),
                                      m_stsFailureCount.load());
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
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] STS call failed. has_valid_cached={}, retaining existing credentials. "
        "sts_success={}, sts_failure={}",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, hasExisting, m_stsSuccessCount.load(), m_stsFailureCount.load());
    m_lastReloadFailed = true;
    m_lastFailedReloadTime = std::chrono::steady_clock::now();
    return;
  }

  if (creds.GetAWSAccessKeyId().empty() || creds.GetAWSSecretKey().empty() || creds.GetSessionToken().empty()) {
    ++m_stsFailureCount;
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] STS returned incomplete credentials (missing ak/sk/token). "
        "has_valid_cached={}, retaining existing credentials. sts_success={}, "
        "sts_failure={}",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, hasExisting, m_stsSuccessCount.load(), m_stsFailureCount.load());
    m_lastReloadFailed = true;
    m_lastFailedReloadTime = std::chrono::steady_clock::now();
    return;
  }

  ++m_stsSuccessCount;
  m_credentials = creds;
  m_lastReloadFailed = false;
  auto expiresInMs = (creds.GetExpiration() - Aws::Utils::DateTime::Now()).count();
  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Successfully retrieved credentials, expires_in_ms={}, region={}, sts_success={}, sts_failure={}",
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, expiresInMs, m_region, m_stsSuccessCount.load(), m_stsFailureCount.load());
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
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] Skipping credential reload — in cooldown after previous failure. "
        "has_valid_cached={}",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, hasExisting);
    return;
  }

  Reload();
}

}  // namespace milvus_storage
