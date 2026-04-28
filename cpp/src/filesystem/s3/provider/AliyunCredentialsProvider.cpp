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

#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"

#include "milvus-storage/common/log.h"

#include <cstdlib>
#include <fstream>
#include <cstring>
#include <climits>

#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/platform/FileSystem.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/client/SpecifiedRetryableErrorsRetryStrategy.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {

static const char STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG[] =
    "AliyunSTSAssumeRoleWebIdentityCredentialsProvider";  // [aliyun]
static const int STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD =
    180 * 1000;  // [aliyun] supports 1800 second at most, here we use their default: 180s -> 180*1k ms

AliyunSTSAssumeRoleWebIdentityCredentialsProvider::AliyunSTSAssumeRoleWebIdentityCredentialsProvider()
    : m_initialized(false) {
  // check environment variables
  // not need in [aliyun]
  // Aws::String tmpRegion = Aws::Environment::GetEnv("AWS_DEFAULT_REGION");
  m_roleArn = Aws::Environment::GetEnv("ALIBABA_CLOUD_ROLE_ARN");           // [aliyun]
  m_tokenFile = Aws::Environment::GetEnv("ALIBABA_CLOUD_OIDC_TOKEN_FILE");  // [aliyun]
  // optional, not existed in [aliyun]
  m_sessionName = Aws::Environment::GetEnv("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  // check profile_config if either m_roleArn or m_tokenFile is not loaded from environment variable
  // region source is not enforced, but we need it to construct sts endpoint, if we can't find from environment, we
  // should check if it's set in config file.
  if (m_roleArn.empty() || m_tokenFile.empty()) {  // || tmpRegion.empty() not need in [aliyun]
    auto profile = Aws::Config::GetCachedConfigProfile(Aws::Auth::GetConfigProfileName());
    // If either of these two were not found from environment, use whatever found for all three in config file
    if (m_roleArn.empty() || m_tokenFile.empty()) {
      m_roleArn = profile.GetRoleArn();
      m_tokenFile = profile.GetValue("web_identity_token_file");
      m_sessionName = profile.GetValue("role_session_name");
    }
  }

  InitializeClient();
}

AliyunSTSAssumeRoleWebIdentityCredentialsProvider::AliyunSTSAssumeRoleWebIdentityCredentialsProvider(
    const Aws::String& role_arn, const Aws::String& session_name)
    : m_initialized(false) {
  m_roleArn = role_arn;
  m_sessionName = session_name;
  // Token file is machine identity and stays in process env; provider ARN is
  // read by AliyunSTSCredentialsClient from env. Caller is authoritative for
  // role_arn / session_name, so no profile-config fallback here.
  m_tokenFile = Aws::Environment::GetEnv("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
  InitializeClient();
}

void AliyunSTSAssumeRoleWebIdentityCredentialsProvider::InitializeClient() {
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

  // not need in [aliyun]
  // if (tmpRegion.empty())
  // {
  //     tmpRegion = Aws::Region::US_EAST_1;
  // }
  // else
  // {
  //     LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved region from profile_config or environment variable to be {}",
  //                                       STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, tmpRegion);
  // }

  if (m_sessionName.empty()) {
    m_sessionName = Aws::Utils::UUID::RandomUUID();
  } else {
    LOG_STORAGE_DEBUG_ << fmt::format("[{}] Resolved session_name from profile_config or environment variable to be {}",
                                      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, m_sessionName);
  }

  Aws::Client::ClientConfiguration config;
  config.scheme = Aws::Http::Scheme::HTTPS;
  // not need in [aliyun]
  // config.region = tmpRegion;

  Aws::Vector<Aws::String> retryableErrors;
  retryableErrors.emplace_back("IDPCommunicationError");
  retryableErrors.emplace_back("InvalidIdentityToken");

  config.retryStrategy = Aws::MakeShared<Aws::Client::SpecifiedRetryableErrorsRetryStrategy>(
      STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, retryableErrors, 3 /*maxRetries*/);

  m_client = Aws::MakeUnique<AliyunSTSCredentialsClient>(STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG, config);
  m_initialized = true;
  LOG_STORAGE_INFO_ << fmt::format("[{}] Creating STS AssumeRole with web identity creds provider.",
                                   STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
}

Aws::Auth::AWSCredentials AliyunSTSAssumeRoleWebIdentityCredentialsProvider::GetAWSCredentials() {
  // A valid client means required information like role arn and token file were constructed correctly.
  // We can use this provider to load creds, otherwise, we can just return empty creds.
  if (!m_initialized) {
    return {};
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  return m_credentials;
}

void AliyunSTSAssumeRoleWebIdentityCredentialsProvider::Reload() {
  LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials have expired, attempting to renew from STS.",
                                   STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);

  Aws::IFStream tokenFile(m_tokenFile.c_str());
  if (tokenFile) {
    Aws::String token((std::istreambuf_iterator<char>(tokenFile)), std::istreambuf_iterator<char>());
    m_token = token;
  } else {
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Can't open token file: {}", STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG,
                                      m_tokenFile);
    return;
  }
  AliyunSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{m_sessionName, m_roleArn, m_token};

  auto result = m_client->GetAssumeRoleWithWebIdentityCredentials(request);
  LOG_STORAGE_TRACE_ << fmt::format("[{}] Successfully retrieved credentials", STS_ASSUME_ROLE_WEB_IDENTITY_LOG_TAG);
  m_credentials = result.creds;
}

bool AliyunSTSAssumeRoleWebIdentityCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() <
          STS_CREDENTIAL_PROVIDER_EXPIRATION_GRACE_PERIOD);
}

void AliyunSTSAssumeRoleWebIdentityCredentialsProvider::RefreshIfExpired() {
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
