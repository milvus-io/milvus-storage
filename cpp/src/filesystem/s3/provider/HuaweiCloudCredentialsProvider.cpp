#include <chrono>
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
    m_lastResolution = MakeCredentialConfigError("Huawei Cloud web identity token file is not configured");
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
    m_lastResolution = MakeCredentialConfigError("Huawei Cloud project ID is not configured");
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
    m_lastResolution = MakeCredentialConfigError("Huawei Cloud region is not configured");
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] Region must be specified to use STS AssumeRole web identity creds "
        "provider.",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return;  // No need to do further constructing
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved region from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_region);
  }

  if (m_providerId.empty()) {
    m_lastResolution = MakeCredentialConfigError("Huawei Cloud identity provider ID is not configured");
    LOG_STORAGE_WARNING_ << fmt::format("[{}] ProviderId must be specified", STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
    return;
  }

  if (m_sessionName.empty()) {
    m_sessionName = Aws::Utils::UUID::RandomUUID();
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved session_name from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_sessionName);
  }

  Aws::Client::ClientConfiguration config(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  config.scheme = Aws::Http::Scheme::HTTPS;
  config.region = m_region;

  Aws::Vector<Aws::String> retryableErrors;
  retryableErrors.emplace_back("IDPCommunicationError");
  retryableErrors.emplace_back("InvalidIdentityToken");

  // Shared credential retry budget; see credential_resolution.h.
  config.connectTimeoutMs = kCredentialConnectTimeoutMs;
  config.requestTimeoutMs = kCredentialRequestTimeoutMs;
  config.retryStrategy = Aws::MakeShared<Aws::Client::SpecifiedRetryableErrorsRetryStrategy>(
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, retryableErrors, kCredentialRetryAttempts);

  m_client = Aws::MakeUnique<HuaweiCloudSTSCredentialsClient>(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
  m_initialized = true;
  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Initialized STS AssumeRole with web identity creds provider. region={}, "
      "tokenFile={}, providerId={}, gracePeriodMs={}",
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_region, m_tokenFile, m_providerId,
      STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

Aws::Auth::AWSCredentials HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
  if (!m_initialized) {
    return {};
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (ValidateTemporaryCredentials(m_credentials, "Huawei Cloud STS").ok()) {
    return m_credentials;
  }
  return {};
}

arrow::Result<Aws::Auth::AWSCredentials> HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::ResolveForRequest() {
  if (!m_initialized) {
    return m_lastResolution.ok() ? MakeCredentialConfigError("Huawei Cloud credential provider is not initialized")
                                 : m_lastResolution;
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (ExpiresSoon() && !m_lastResolution.ok()) {
    return m_lastResolution;
  }
  auto validation = ValidateTemporaryCredentials(m_credentials, "Huawei Cloud STS");
  if (validation.ok()) {
    return m_credentials;
  }
  return m_lastResolution.ok() ? validation : m_lastResolution;
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
    if (token.empty()) {
      ++m_stsFailureCount;
      m_lastResolution = MakeCredentialConfigError(fmt::format("The OIDC token file {} is empty", m_tokenFile));
      return;
    }
    m_token = token;
  } else {
    ++m_stsFailureCount;
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Can't open token file: {}, sts_success={}, sts_failure={}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_tokenFile, m_stsSuccessCount.load(),
                                      m_stsFailureCount.load());
    // A token file that will not open is the deployment's to fix -- the
    // projected volume is missing or unreadable -- not something a retry
    // outlasts.
    m_lastResolution = MakeCredentialConfigError(fmt::format("Cannot open the OIDC token file {}", m_tokenFile));
    return;
  }
  HuaweiCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{m_region, m_providerId, m_token,
                                                                               m_roleArn, m_sessionName};

  // GetAssumeRoleWithWebIdentityCredentials catches all exceptions internally
  // and returns result.success=false on any failure.
  auto result = m_client->GetAssumeRoleWithWebIdentityCredentials(request);

  const auto& creds = result.creds;

  if (!result.success) {
    m_lastResolution = result.status.ok() ? MakeCredentialResponseError("Huawei Cloud STS call failed") : result.status;
    ++m_stsFailureCount;
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] STS call failed. has_valid_cached={}, retaining existing credentials. "
        "sts_success={}, sts_failure={}",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, hasExisting, m_stsSuccessCount.load(), m_stsFailureCount.load());
    return;
  }

  auto validation = ValidateTemporaryCredentials(creds, "Huawei Cloud STS");
  if (!validation.ok()) {
    m_lastResolution = validation;
    ++m_stsFailureCount;
    bool hasExisting = !m_credentials.IsEmpty() && !m_credentials.IsExpired();
    LOG_STORAGE_WARNING_ << fmt::format(
        "[{}] STS returned incomplete credentials (missing ak/sk/token). "
        "has_valid_cached={}, retaining existing credentials. sts_success={}, "
        "sts_failure={}",
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, hasExisting, m_stsSuccessCount.load(), m_stsFailureCount.load());
    return;
  }

  m_lastResolution = arrow::Status::OK();
  ++m_stsSuccessCount;
  m_credentials = creds;
  auto expiresInMs = (creds.GetExpiration() - Aws::Utils::DateTime::Now()).count();
  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Successfully retrieved credentials, expires_in_ms={}, region={}, sts_success={}, sts_failure={}",
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, expiresInMs, m_region, m_stsSuccessCount.load(), m_stsFailureCount.load());
}

bool HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() <
          STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

void HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider::RefreshIfExpired() {
  // The caller's own budget, started before any waiting. See
  // CredentialAttemptStillWorthMaking: the double-check below handles a leader
  // that succeeded; this handles one that failed, whose empty credentials would
  // otherwise make every queued caller replay the same outage in series.
  const auto started = std::chrono::steady_clock::now();

  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
    return;
  }

  guard.UpgradeToWriterLock();
  if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
    return;
  }

  if (!CredentialAttemptStillWorthMaking(started)) {
    return;
  }

  Reload();
}

}  // namespace milvus_storage
