#include "milvus-storage/filesystem/s3/provider/VolcengineCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/VolcengineSTSClient.h"
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/client/SpecifiedRetryableErrorsRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/CoreErrors.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <limits.h>
#include <mutex>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>

namespace milvus_storage {

static const char STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG[] =
    "VolcengineSTSAssumeRoleWebIdentityCredentialsProvider";
static const int STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD =
    30 * 60 * 1000;  // 30min

VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::
    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider()
    : m_initialized(false) {
    // Env-only mode (use_iam=true path): RoleTrn, token file and session are
    // all provided by the VKE-injected environment.
    m_roleArn = Aws::Environment::GetEnv("VOLCENGINE_OIDC_ROLE_TRN");
    m_tokenFile = Aws::Environment::GetEnv("VOLCENGINE_OIDC_TOKEN_FILE");
    m_sessionName =
        Aws::Environment::GetEnv("VOLCENGINE_OIDC_ROLE_SESSION_NAME");
    InitClient();
}

VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::
    VolcengineSTSAssumeRoleWebIdentityCredentialsProvider(
        const Aws::String& role_arn, const Aws::String& session_name)
    : m_initialized(false) {
    // Per-tenant override (external-table role_arn path): RoleTrn and session
    // come from the caller (extfs), while the OIDC web-identity token file
    // remains the VKE-injected machine identity from the environment.
    m_roleArn = role_arn;
    m_sessionName = session_name;
    m_tokenFile = Aws::Environment::GetEnv("VOLCENGINE_OIDC_TOKEN_FILE");
    InitClient();
}

void
VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::InitClient() {
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

    if (m_sessionName.empty()) {
        m_sessionName = Aws::Utils::UUID::RandomUUID();
    } else {
        AWS_LOGSTREAM_DEBUG(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                            "Resolved session_name from profile_config or "
                            "environment variable to be "
                                << m_sessionName);
    }

    Aws::Client::ClientConfiguration config;
    config.scheme = Aws::Http::Scheme::HTTP;

    Aws::Vector<Aws::String> retryableErrors;
    retryableErrors.push_back("IDPCommunicationError");
    retryableErrors.push_back("InvalidIdentityToken");

    config.retryStrategy =
        Aws::MakeShared<Aws::Client::SpecifiedRetryableErrorsRetryStrategy>(
            STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
            retryableErrors,
            3 /*maxRetries*/);

    m_client = Aws::MakeUnique<VolcengineSTSCredentialsClient>(
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
    m_initialized = true;
    AWS_LOGSTREAM_INFO(
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
        "Creating STS AssumeRole with web identity creds provider.");
}

Aws::Auth::AWSCredentials
VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
    // A valid client means required information like role arn and token file were constructed correctly.
    // We can use this provider to load creds, otherwise, we can just return empty creds.
    if (!m_initialized) {
        return Aws::Auth::AWSCredentials();
    }
    RefreshIfExpired();
    Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
    return m_credentials;
}

void
VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::Reload() {
    AWS_LOGSTREAM_INFO(
        STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
        "Credentials have expired, attempting to renew from STS.");

    Aws::IFStream tokenFile(m_tokenFile.c_str());
    if (tokenFile) {
        Aws::String token((std::istreambuf_iterator<char>(tokenFile)),
                          std::istreambuf_iterator<char>());
        if (!token.empty() && token.back() == '\n') {
            token.pop_back();
        }
        m_token = token;
    } else {
        AWS_LOGSTREAM_ERROR(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                            "Can't open token file: " << m_tokenFile);
        return;
    }
    VolcengineSTSCredentialsClient::
        STSAssumeRoleWithWebIdentityRequest request{
            m_token, m_roleArn, m_sessionName};

    auto result = m_client->GetAssumeRoleWithWebIdentityCredentials(request);
    if (result.creds.IsEmpty()) {
        AWS_LOGSTREAM_ERROR(
            STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
            "Failed to retrieve valid credentials - credentials are empty");
    } else {
        AWS_LOGSTREAM_TRACE(
            STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
            "Successfully retrieved credentials with AWS_ACCESS_KEY: "
                << result.creds.GetAWSAccessKeyId());
        m_credentials = result.creds;
    }
}

bool
VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::ExpiresSoon() const {
    return (
        (m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() <
        STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

void
VolcengineSTSAssumeRoleWebIdentityCredentialsProvider::RefreshIfExpired() {
    Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
    if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
        return;
    }

    guard.UpgradeToWriterLock();
    if (!m_credentials.IsExpiredOrEmpty() && !ExpiresSoon()) {
        return;
    }

    Reload();
}

}  // namespace milvus_storage
