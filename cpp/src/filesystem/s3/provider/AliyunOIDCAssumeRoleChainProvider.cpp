// Copyright 2024 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/s3/provider/AliyunOIDCAssumeRoleChainProvider.h"

#include "milvus-storage/common/log.h"

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {

static const char kLogTag[] = "AliyunOIDCAssumeRoleChainProvider";
// Refresh when less than this many ms remain. Matches the OIDC and RAM
// providers so all three rotate on the same schedule.
static const int kRefreshGraceMs = 180 * 1000;

AliyunOIDCAssumeRoleChainProvider::AliyunOIDCAssumeRoleChainProvider(const Aws::String& target_role_arn,
                                                                     const Aws::String& target_session_name,
                                                                     const Aws::String& target_external_id)
    : m_targetRoleArn(target_role_arn),
      m_targetSessionName(target_session_name),
      m_targetExternalId(target_external_id) {
  if (m_targetSessionName.empty()) {
    m_targetSessionName = Aws::Utils::UUID::RandomUUID();
  }

  // Default ctor: reads ALIBABA_CLOUD_ROLE_ARN, ALIBABA_CLOUD_OIDC_TOKEN_FILE,
  // and ALIBABA_CLOUD_ROLE_SESSION_NAME from env. Provider ARN is read inside
  // AliyunSTSCredentialsClient (also from env). The dispatch layer in
  // s3_filesystem_producer.cpp pre-flights ALIBABA_CLOUD_OIDC_TOKEN_FILE and
  // ALIBABA_CLOUD_OIDC_PROVIDER_ARN before constructing this provider, so we
  // do not re-validate here.
  m_innerOidc = Aws::MakeUnique<AliyunSTSAssumeRoleWebIdentityCredentialsProvider>(kLogTag);

  Aws::Client::ClientConfiguration cfg;
  cfg.scheme = Aws::Http::Scheme::HTTPS;
  m_stsClient = Aws::MakeUnique<AliyunRAMSTSClient>(kLogTag, cfg);

  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Created OIDC chain provider for target_role_arn={} session={} external_id_set={}", kLogTag, m_targetRoleArn,
      m_targetSessionName, !m_targetExternalId.empty());
}

Aws::Auth::AWSCredentials AliyunOIDCAssumeRoleChainProvider::GetAWSCredentials() {
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  return m_credentials;
}

bool AliyunOIDCAssumeRoleChainProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() < kRefreshGraceMs);
}

void AliyunOIDCAssumeRoleChainProvider::RefreshIfExpired() {
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

void AliyunOIDCAssumeRoleChainProvider::Reload() {
  LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials missing or expiring; refreshing via OIDC -> AssumeRole.", kLogTag);

  // Step 1: inner provider self-refreshes on call.
  Aws::Auth::AWSCredentials inner = m_innerOidc->GetAWSCredentials();
  if (inner.IsEmpty()) {
    LOG_STORAGE_ERROR_ << fmt::format(
        "[{}] Inner OIDC step returned empty credentials; cannot chain to "
        "target_role_arn={}",
        kLogTag, m_targetRoleArn);
    return;
  }

  // Step 2: cross-account AssumeRole using inner creds as caller. SecurityToken
  // is mandatory because the caller is itself an STS-temporary identity.
  // ExternalId belongs here, not on step 1: AssumeRoleWithOIDC has no
  // ExternalId concept, and the cross-account confused-deputy guard applies
  // to the customer-side hop.
  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = inner.GetAWSAccessKeyId();
  req.callerAccessKeySecret = inner.GetAWSSecretKey();
  req.callerSecurityToken = inner.GetSessionToken();
  req.roleArn = m_targetRoleArn;
  req.roleSessionName = m_targetSessionName;
  req.externalId = m_targetExternalId;
  LOG_STORAGE_INFO_ << fmt::format("[{}] Sending chained AssumeRole request; external_id_set={}", kLogTag,
                                   !req.externalId.empty());

  auto res = m_stsClient->GetAssumeRoleCredentials(req);
  if (res.creds.IsEmpty()) {
    LOG_STORAGE_ERROR_ << fmt::format(
        "[{}] Cross-account AssumeRole returned empty credentials; target_role_arn={} "
        "— check the target role's trust policy lists the OIDC step-1 role as Principal",
        kLogTag, m_targetRoleArn);
    return;
  }
  m_credentials = res.creds;
  LOG_STORAGE_INFO_ << fmt::format("[{}] OIDC chain succeeded; target_role_arn={} expires={}", kLogTag, m_targetRoleArn,
                                   m_credentials.GetExpiration().ToGmtString(Aws::Utils::DateFormat::ISO_8601));
}

}  // namespace milvus_storage
