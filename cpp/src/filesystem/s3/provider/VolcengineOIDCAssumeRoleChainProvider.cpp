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

#include "milvus-storage/filesystem/s3/provider/VolcengineOIDCAssumeRoleChainProvider.h"

#include "milvus-storage/common/log.h"

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/UUID.h>

namespace milvus_storage {

static const char kLogTag[] = "VolcengineOIDCAssumeRoleChainProvider";
// Refresh when less than this many ms remain. Matches the single-step
// Volcengine provider (30min) so both rotate on the same schedule.
static const int kRefreshGraceMs = 30 * 60 * 1000;

VolcengineOIDCAssumeRoleChainProvider::VolcengineOIDCAssumeRoleChainProvider(const Aws::String& target_role_trn,
                                                                             const Aws::String& target_session_name)
    : m_targetRoleTrn(target_role_trn), m_targetSessionName(target_session_name) {
  if (m_targetSessionName.empty()) {
    m_targetSessionName = Aws::Utils::UUID::RandomUUID();
  }

  // step-1 machine role, captured once for the degradation short-circuit.
  m_step1RoleTrn = Aws::Environment::GetEnv("VOLCENGINE_OIDC_ROLE_TRN");

  // Step 1: env-only OIDC provider. Reads VOLCENGINE_OIDC_ROLE_TRN /
  // VOLCENGINE_OIDC_TOKEN_FILE / VOLCENGINE_OIDC_ROLE_SESSION_NAME. The dispatch
  // layer in s3_filesystem_producer.cpp pre-flights the token file and role trn
  // before constructing this provider, so we do not re-validate here.
  m_innerOidc = Aws::MakeUnique<VolcengineSTSAssumeRoleWebIdentityCredentialsProvider>(kLogTag);

  // Step 2: cross-account AssumeRole client, V4-signed over HTTPS.
  Aws::Client::ClientConfiguration cfg(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  cfg.scheme = Aws::Http::Scheme::HTTPS;
  m_stsClient = Aws::MakeUnique<VolcengineSTSCredentialsClient>(kLogTag, cfg);

  LOG_STORAGE_INFO_ << fmt::format(
      "[{}] Created OIDC chain provider; target_role_trn={} step1_role_trn={} session={}", kLogTag, m_targetRoleTrn,
      m_step1RoleTrn, m_targetSessionName);
}

Aws::Auth::AWSCredentials VolcengineOIDCAssumeRoleChainProvider::GetAWSCredentials() {
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  return m_credentials;
}

bool VolcengineOIDCAssumeRoleChainProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() < kRefreshGraceMs);
}

void VolcengineOIDCAssumeRoleChainProvider::RefreshIfExpired() {
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

void VolcengineOIDCAssumeRoleChainProvider::Reload() {
  LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials missing or expiring; refreshing via OIDC -> AssumeRole.", kLogTag);

  // Step 1: inner provider self-refreshes on call.
  Aws::Auth::AWSCredentials inner = m_innerOidc->GetAWSCredentials();
  if (inner.IsEmpty()) {
    LOG_STORAGE_ERROR_ << fmt::format(
        "[{}] Inner OIDC step returned empty credentials; cannot chain to target_role_trn={}", kLogTag,
        m_targetRoleTrn);
    return;
  }

  // Degradation short-circuit for single-account deployments: when the target
  // role is unset, or is the very machine role step 1 already assumed, a second
  // AssumeRole would only succeed if that role trusted itself as Principal —
  // which it does not in the historical single-account setup. Return the step-1
  // credentials directly so use_iam-style single-account role_arn keeps working.
  if (m_targetRoleTrn.empty() || m_targetRoleTrn == m_step1RoleTrn) {
    LOG_STORAGE_INFO_ << fmt::format(
        "[{}] Target role empty or equals step-1 machine role; using step-1 credentials directly (single-account "
        "path). target_role_trn={}",
        kLogTag, m_targetRoleTrn);
    m_credentials = inner;
    return;
  }

  // Step 2: cross-account AssumeRole using inner creds as caller. SecurityToken
  // is mandatory because the caller is itself an STS-temporary identity.
  VolcengineSTSCredentialsClient::STSAssumeRoleRequest req;
  req.callerAccessKeyId = inner.GetAWSAccessKeyId();
  req.callerAccessKeySecret = inner.GetAWSSecretKey();
  req.callerSecurityToken = inner.GetSessionToken();
  req.roleTrn = m_targetRoleTrn;
  req.roleSessionName = m_targetSessionName;
  LOG_STORAGE_INFO_ << fmt::format("[{}] Sending chained AssumeRole request to target_role_trn={}", kLogTag,
                                   m_targetRoleTrn);

  auto res = m_stsClient->GetAssumeRoleCredentials(req);
  if (res.creds.IsEmpty()) {
    LOG_STORAGE_ERROR_ << fmt::format(
        "[{}] Cross-account AssumeRole returned empty credentials; target_role_trn={} "
        "— check the target role's trust policy lists the step-1 machine role as Principal",
        kLogTag, m_targetRoleTrn);
    return;
  }
  m_credentials = res.creds;
  LOG_STORAGE_INFO_ << fmt::format("[{}] OIDC chain succeeded; target_role_trn={} expires={}", kLogTag, m_targetRoleTrn,
                                   m_credentials.GetExpiration().ToGmtString(Aws::Utils::DateFormat::ISO_8601));
}

}  // namespace milvus_storage
