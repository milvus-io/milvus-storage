// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <chrono>
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"

#include "milvus-storage/common/log.h"

#include <fstream>

#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/client/SpecifiedRetryableErrorsRetryStrategy.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {
static const char STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG[] = "TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider";
static const int STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD = 180 * 1000;  // tencent cloud support 180s.

TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider()
    : m_initialized(false) {
  m_region = Aws::Environment::GetEnv("TKE_REGION");
  m_roleArn = Aws::Environment::GetEnv("TKE_ROLE_ARN");
  m_tokenFile = Aws::Environment::GetEnv("TKE_WEB_IDENTITY_TOKEN_FILE");
  m_providerId = Aws::Environment::GetEnv("TKE_PROVIDER_ID");
  auto currentTimePoint = std::chrono::high_resolution_clock::now();
  auto nanoseconds = std::chrono::time_point_cast<std::chrono::nanoseconds>(currentTimePoint);
  auto timestamp = nanoseconds.time_since_epoch().count();
  m_sessionName = "tencentcloud-cpp-sdk-" + std::to_string(timestamp / 1000);

  if (m_roleArn.empty() || m_tokenFile.empty() || m_region.empty()) {
    auto profile = Aws::Config::GetCachedConfigProfile(Aws::Auth::GetConfigProfileName());
    m_roleArn = profile.GetRoleArn();
    m_tokenFile = profile.GetValue("web_identity_token_file");
    m_sessionName = profile.GetValue("role_session_name");
  }

  if (m_tokenFile.empty()) {
    m_lastResolution = MakeCredentialConfigError("Tencent Cloud web identity token file is not configured");
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
    m_lastResolution = MakeCredentialConfigError("Tencent Cloud role ARN is not configured");
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
    m_lastResolution = MakeCredentialConfigError("Tencent Cloud region is not configured");
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
    m_lastResolution = MakeCredentialConfigError("Tencent Cloud identity provider ID is not configured");
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

  m_client = Aws::MakeUnique<TencentCloudSTSCredentialsClient>(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
  m_initialized = true;
  LOG_STORAGE_INFO_ << fmt::format("[{}] Creating STS AssumeRole with web identity creds provider.",
                                   STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
}

Aws::Auth::AWSCredentials TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
  auto result = ResolveForRequest();
  if (!result.ok()) {
    return {};
  }
  return std::move(result).ValueOrDie();
}

arrow::Result<Aws::Auth::AWSCredentials> TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::ResolveForRequest() {
  if (!m_initialized) {
    return m_lastResolution.ok() ? MakeCredentialConfigError("Tencent Cloud credential provider is not initialized")
                                 : m_lastResolution;
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (ExpiresSoon() && !m_lastResolution.ok()) {
    return m_lastResolution;
  }
  auto validation = ValidateTemporaryCredentials(m_credentials, "Tencent Cloud STS");
  if (validation.ok()) {
    return m_credentials;
  }
  return m_lastResolution.ok() ? validation : m_lastResolution;
}

void TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::Reload() {
  LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials have expired, attempting to renew from STS.",
                                   STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);

  Aws::IFStream tokenFile(m_tokenFile.c_str());
  if (tokenFile) {
    Aws::String token((std::istreambuf_iterator<char>(tokenFile)), std::istreambuf_iterator<char>());
    if (!token.empty() && token.back() == '\n') {
      token.pop_back();
    }
    if (token.empty()) {
      m_lastResolution = MakeCredentialConfigError(fmt::format("The OIDC token file {} is empty", m_tokenFile));
      return;
    }
    m_token = token;
  } else {
    // A token file that will not open is the deployment's to fix -- the
    // projected volume is missing or unreadable -- not something a retry
    // outlasts.
    m_lastResolution = MakeCredentialConfigError(fmt::format("Cannot open the OIDC token file {}", m_tokenFile));
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Can't open token file: {}", STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                                      m_tokenFile);
    return;
  }
  TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{m_region, m_providerId, m_token,
                                                                                m_roleArn, m_sessionName};

  auto result = m_client->GetAssumeRoleWithWebIdentityCredentials(request);
  if (!result.status.ok()) {
    m_lastResolution = result.status;
    return;
  }
  auto validation = ValidateTemporaryCredentials(result.creds, "Tencent Cloud STS");
  if (!validation.ok()) {
    m_lastResolution = validation;
    return;
  }
  m_lastResolution = arrow::Status::OK();
  LOG_STORAGE_DEBUG_ << fmt::format("[{}] Successfully retrieved credentials, expiration_count_diff_ms: {}",
                                    STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                                    (result.creds.GetExpiration() - Aws::Utils::DateTime::Now()).count());
  m_credentials = result.creds;
}

bool TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() <
          STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

void TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider::RefreshIfExpired() {
  // The cooldown shares a fast failed refresh with queued callers. The caller
  // budget separately prevents a waiter that already spent too long behind a
  // slow refresh from starting another attempt after the cooldown expires.
  const auto started = std::chrono::steady_clock::now();

  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (!m_credentials.IsEmpty() && !ExpiresSoon()) {
    return;
  }

  guard.UpgradeToWriterLock();
  if (!m_credentials.IsExpiredOrEmpty() && !ExpiresSoon()) {
    return;
  }
  if (!CredentialAttemptStillWorthMaking(started)) {
    return;
  }

  Reload();
}

}  // namespace milvus_storage
