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

#include <chrono>
#include <new>
#include "milvus-storage/filesystem/s3/provider/AliyunRAMCredentialsProvider.h"

#include "milvus-storage/common/log.h"

#include <sstream>
#include <optional>
#include <string>

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

namespace milvus_storage {

static const char kLogTag[] = "AliyunRAMCredentialsProvider";
// Refresh credentials when less than this many ms remain until expiry.
// Matches the OIDC provider.
static const int kRefreshGraceMs = 180 * 1000;

// Aliyun ECS metadata service (link-local, HTTP only).
static const char kImdsHost[] = "http://100.100.100.200";
static const char kImdsRoleListPath[] = "/latest/meta-data/ram/security-credentials/";
static const char kImdsV2TokenPath[] = "/latest/api/token";
static const char kImdsV2TtlHeader[] = "X-aliyun-ecs-metadata-token-ttl-seconds";
static const char kImdsV2TokenHeader[] = "X-aliyun-ecs-metadata-token";
// IMDS caps V2 session token TTL at 6h. We only need the token across the
// two GETs that follow a PUT, so anything positive works; this matches what
// Aliyun's own SDKs request.
static const int kImdsV2TtlSecs = 21600;

namespace {

bool IsTrueEnvironmentVariable(const char* name) {
  return Aws::Utils::StringUtils::CaselessCompare(Aws::Environment::GetEnv(name).c_str(), "true");
}

bool IsImdsV1Disabled() {
  // Alibaba Cloud SDKs have shipped both spellings. Treat either as the same
  // security boundary so a mixed-language deployment cannot accidentally
  // downgrade to an unauthenticated metadata request.
  return IsTrueEnvironmentVariable("ALIBABA_CLOUD_IMDSV1_DISABLED") ||
         IsTrueEnvironmentVariable("ALIBABA_CLOUD_IMDSV1_DISABLE");
}

std::shared_ptr<Aws::Http::HttpClient> MakeImdsHttpClient() {
  Aws::Client::ClientConfiguration cfg(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  cfg.scheme = Aws::Http::Scheme::HTTP;
  // IMDS answers in milliseconds on a healthy VM; a multi-second stall means
  // the service is unreachable (e.g. not an ECS). Short caps keep a broken
  // refresh from blocking callers holding the credentials lock.
  cfg.connectTimeoutMs = kCredentialConnectTimeoutMs;
  cfg.requestTimeoutMs = kImdsRequestTimeoutMs;
  // Deliberately no retryStrategy here: this client is driven by raw
  // MakeRequest calls, which do not consume one. The budget is applied by
  // MakeRequestWithCredentialRetry at each call site instead.
  return Aws::Http::CreateHttpClient(cfg);
}

std::string ReadBody(const std::shared_ptr<Aws::Http::HttpResponse>& resp) {
  if (!resp)
    return {};
  std::ostringstream ss;
  ss << resp->GetResponseBody().rdbuf();
  return ss.str();
}

// A disengaged optional means the endpoint explicitly allows a V1 fallback.
// Transport/timeout/throttling/service failures stay typed errors: sending a
// bare GET after those would both hide the cause and weaken the request.
arrow::Result<std::optional<std::string>> TryImdsV2Token(Aws::Http::HttpClient& http) {
  const std::string url = std::string(kImdsHost) + kImdsV2TokenPath;
  auto req = Aws::Http::CreateHttpRequest(url, Aws::Http::HttpMethod::HTTP_PUT,
                                          Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  req->SetHeaderValue(kImdsV2TtlHeader, std::to_string(kImdsV2TtlSecs).c_str());
  req->SetContentLength("0");
  req->SetUserAgent(Aws::Client::ComputeUserAgentString());

  auto resp = MakeRequestWithCredentialRetry(http, req);
  if (resp == nullptr) {
    return ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode::NO_RESPONSE,
                                         "Aliyun ECS IMDSv2 token request received no response");
  }
  const auto code = resp->GetResponseCode();
  if (code != Aws::Http::HttpResponseCode::OK) {
    // Older/explicitly V1-only metadata endpoints use these responses for an
    // unsupported token method. They are the only safe downgrade signal.
    if (!IsImdsV1Disabled() &&
        (code == Aws::Http::HttpResponseCode::FORBIDDEN || code == Aws::Http::HttpResponseCode::NOT_FOUND ||
         code == Aws::Http::HttpResponseCode::METHOD_NOT_ALLOWED)) {
      return std::nullopt;
    }
    return ClassifyCredentialHttpFailure(code, "Aliyun ECS IMDSv2 token request failed");
  }
  auto token = std::string(Aws::Utils::StringUtils::Trim(ReadBody(resp).c_str()).c_str());
  if (token.empty()) {
    return MakeCredentialResponseError("Aliyun ECS IMDSv2 returned an empty token");
  }
  return std::optional<std::string>(std::move(token));
}

std::shared_ptr<Aws::Http::HttpRequest> MakeImdsGet(const std::string& url, const std::string& v2_token) {
  auto req = Aws::Http::CreateHttpRequest(url, Aws::Http::HttpMethod::HTTP_GET,
                                          Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  if (!v2_token.empty()) {
    req->SetHeaderValue(kImdsV2TokenHeader, v2_token.c_str());
  }
  req->SetUserAgent(Aws::Client::ComputeUserAgentString());
  return req;
}

struct ImdsCreds {
  std::string access_key_id;
  std::string access_key_secret;
  std::string security_token;
  Aws::Utils::DateTime expiration;
};

// `status` receives why this failed, so the caller can tell an ECS without a
// RAM role (fix the deployment) from an IMDS that did not answer (wait).
bool FetchImdsCreds(ImdsCreds& out, arrow::Status& status) {
  // This is the shared containment boundary for all three raw IMDS requests
  // below. HttpClient and the SDK's stream/JSON helpers may throw instead of
  // producing an HTTP response.
  try {
    if (IsTrueEnvironmentVariable("ALIBABA_CLOUD_ECS_METADATA_DISABLED")) {
      status = MakeCredentialConfigError("Aliyun ECS metadata is disabled by ALIBABA_CLOUD_ECS_METADATA_DISABLED");
      return false;
    }

    auto http = MakeImdsHttpClient();
    if (!http) {
      status = MakeCredentialResponseError("Cannot create an HTTP client for ECS IMDS");
      return false;
    }

    // V2 first. Only an explicit unsupported-method response drops back to
    // V1-style bare GETs; a transient PUT failure aborts the refresh.
    auto token_result = TryImdsV2Token(*http);
    if (!token_result.ok()) {
      status = token_result.status();
      LOG_STORAGE_ERROR_ << fmt::format("[{}] {}", kLogTag, status.message());
      return false;
    }
    const std::string v2_token = token_result.ValueOrDie().value_or("");

    auto list_req = MakeImdsGet(std::string(kImdsHost) + kImdsRoleListPath, v2_token);
    auto list_resp = MakeRequestWithCredentialRetry(*http, list_req);
    if (!list_resp || list_resp->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
      status = ClassifyCredentialHttpFailure(
          list_resp ? list_resp->GetResponseCode() : Aws::Http::HttpResponseCode::NO_RESPONSE,
          "ECS IMDS role list request failed; is a RAM role attached to this instance?");
      LOG_STORAGE_ERROR_ << fmt::format("[{}] {}", kLogTag, status.message());
      return false;
    }
    const auto role_name = std::string(Aws::Utils::StringUtils::Trim(ReadBody(list_resp).c_str()).c_str());
    if (role_name.empty()) {
      // IMDS answered 200 with nothing in it: no role is attached, which is a
      // deployment fact and not something a retry changes.
      status = MakeCredentialConfigError("ECS IMDS reports no RAM role attached to this instance");
      LOG_STORAGE_ERROR_ << fmt::format("[{}] {}", kLogTag, status.message());
      return false;
    }

    const std::string creds_url = std::string(kImdsHost) + kImdsRoleListPath + role_name;
    auto creds_req = MakeImdsGet(creds_url, v2_token);
    auto creds_resp = MakeRequestWithCredentialRetry(*http, creds_req);
    if (!creds_resp || creds_resp->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
      status = ClassifyCredentialHttpFailure(
          creds_resp ? creds_resp->GetResponseCode() : Aws::Http::HttpResponseCode::NO_RESPONSE,
          fmt::format("ECS IMDS credentials request failed for role {}", role_name));
      LOG_STORAGE_ERROR_ << fmt::format("[{}] {}", kLogTag, status.message());
      return false;
    }

    Aws::Utils::Json::JsonValue json(ReadBody(creds_resp).c_str());
    if (!json.WasParseSuccessful()) {
      LOG_STORAGE_ERROR_ << fmt::format("[{}] IMDS credentials JSON parse failed: {}", kLogTag, json.GetErrorMessage());
      status = MakeCredentialResponseError("ECS IMDS credentials response is not valid JSON");
      return false;
    }
    auto view = json.View();
    if (!view.KeyExists("AccessKeyId") || !view.KeyExists("AccessKeySecret") || !view.KeyExists("SecurityToken") ||
        !view.KeyExists("Expiration")) {
      LOG_STORAGE_ERROR_ << fmt::format("[{}] IMDS credentials response missing expected fields", kLogTag);
      status = MakeCredentialResponseError("ECS IMDS credentials response is missing expected fields");
      return false;
    }
    out.access_key_id = view.GetString("AccessKeyId");
    out.access_key_secret = view.GetString("AccessKeySecret");
    out.security_token = view.GetString("SecurityToken");
    out.expiration = Aws::Utils::DateTime(Aws::Utils::StringUtils::Trim(view.GetString("Expiration").c_str()).c_str(),
                                          Aws::Utils::DateFormat::ISO_8601);
    Aws::Auth::AWSCredentials credentials(out.access_key_id.c_str(), out.access_key_secret.c_str(),
                                          out.security_token.c_str(), out.expiration);
    status = ValidateTemporaryCredentials(credentials, "Aliyun ECS IMDS");
    if (!status.ok()) {
      return false;
    }
    return true;
  } catch (const std::bad_alloc&) {
    status = MakeCredentialOutOfMemoryError("Aliyun ECS IMDS credential retrieval ran out of memory");
  } catch (const std::exception& e) {
    status = MakeCredentialExceptionError("Aliyun ECS IMDS credential retrieval raised", e);
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Exception during IMDS credential retrieval: {}", kLogTag, e.what());
  } catch (...) {
    status = MakeCredentialUnknownExceptionError("Aliyun ECS IMDS credential retrieval raised");
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Unknown exception during IMDS credential retrieval", kLogTag);
  }
  return false;
}

}  // namespace

AliyunRAMCredentialsProvider::AliyunRAMCredentialsProvider(const Aws::String& role_arn,
                                                           const Aws::String& role_session_name,
                                                           const Aws::String& external_id)
    : m_roleArn(role_arn), m_roleSessionName(role_session_name), m_externalId(external_id) {
  if (m_roleArn.empty()) {
    m_lastResolution = MakeCredentialConfigError("Aliyun target role ARN is not configured");
  }
  if (m_roleSessionName.empty()) {
    m_roleSessionName = Aws::Utils::UUID::RandomUUID();
  }

  Aws::Client::ClientConfiguration cfg(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  cfg.scheme = Aws::Http::Scheme::HTTPS;
  // Explicit, so the credential path takes a bounded and predictable amount of
  // time. The default strategy reads AWS_MAX_ATTEMPTS, which a deployment may
  // have set for object I/O without meaning to make credential resolution block
  // for tens of seconds.
  cfg.connectTimeoutMs = kCredentialConnectTimeoutMs;
  cfg.requestTimeoutMs = kCredentialRequestTimeoutMs;
  cfg.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>(kLogTag, kCredentialRetryAttempts);
  m_stsClient = Aws::MakeUnique<AliyunRAMSTSClient>(kLogTag, cfg);

  LOG_STORAGE_INFO_ << fmt::format("[{}] Created RAM provider for role_arn={} session={} external_id_set={}", kLogTag,
                                   m_roleArn, m_roleSessionName, !m_externalId.empty());
}

Aws::Auth::AWSCredentials AliyunRAMCredentialsProvider::GetAWSCredentials() {
  auto result = ResolveForRequest();
  if (!result.ok()) {
    return {};
  }
  return std::move(result).ValueOrDie();
}

arrow::Result<Aws::Auth::AWSCredentials> AliyunRAMCredentialsProvider::ResolveForRequest() {
  if (m_roleArn.empty()) {
    return m_lastResolution;
  }
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  if (ExpiresSoon() && !m_lastResolution.ok()) {
    return m_lastResolution;
  }
  auto validation = ValidateTemporaryCredentials(m_credentials, "Aliyun AssumeRole");
  if (validation.ok()) {
    return m_credentials;
  }
  return m_lastResolution.ok() ? validation : m_lastResolution;
}

bool AliyunRAMCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() < kRefreshGraceMs);
}

void AliyunRAMCredentialsProvider::RefreshIfExpired() {
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

void AliyunRAMCredentialsProvider::Reload() {
  LOG_STORAGE_INFO_ << fmt::format("[{}] Credentials missing or expiring; refreshing via IMDS → AssumeRole.", kLogTag);

  ImdsCreds imds;
  arrow::Status imds_status;
  if (!FetchImdsCreds(imds, imds_status)) {
    m_lastResolution =
        imds_status.ok() ? MakeCredentialResponseError("Failed to fetch ECS IMDS credentials") : imds_status;
    return;
  }

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = imds.access_key_id.c_str();
  req.callerAccessKeySecret = imds.access_key_secret.c_str();
  req.callerSecurityToken = imds.security_token.c_str();
  req.roleArn = m_roleArn;
  req.roleSessionName = m_roleSessionName;
  req.externalId = m_externalId;
  LOG_STORAGE_INFO_ << fmt::format("[{}] Sending AssumeRole request; external_id_set={}", kLogTag,
                                   !req.externalId.empty());

  auto res = m_stsClient->GetAssumeRoleCredentials(req);
  if (!res.status.ok()) {
    m_lastResolution = res.status;
    LOG_STORAGE_ERROR_ << fmt::format("[{}] AssumeRole failed: {}", kLogTag, m_lastResolution.message());
    return;
  }
  auto validation = ValidateTemporaryCredentials(res.creds, "Aliyun AssumeRole");
  if (!validation.ok()) {
    m_lastResolution = validation;
    LOG_STORAGE_ERROR_ << fmt::format("[{}] AssumeRole returned empty credentials", kLogTag);
    return;
  }
  m_lastResolution = arrow::Status::OK();
  m_credentials = res.creds;
  LOG_STORAGE_INFO_ << fmt::format("[{}] AssumeRole succeeded; expires={}", kLogTag,
                                   m_credentials.GetExpiration().ToGmtString(Aws::Utils::DateFormat::ISO_8601));
}

}  // namespace milvus_storage
