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

#include "milvus-storage/filesystem/s3/provider/AliyunRAMCredentialsProvider.h"

#include <sstream>
#include <string>

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/logging/LogMacros.h>
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

std::shared_ptr<Aws::Http::HttpClient> MakeImdsHttpClient() {
  Aws::Client::ClientConfiguration cfg;
  cfg.scheme = Aws::Http::Scheme::HTTP;
  // IMDS answers in milliseconds on a healthy VM; a multi-second stall means
  // the service is unreachable (e.g. not an ECS). Short caps keep a broken
  // refresh from blocking callers holding the credentials lock.
  cfg.connectTimeoutMs = 2000;
  cfg.requestTimeoutMs = 5000;
  cfg.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>(kLogTag, 0 /*maxRetries*/);
  return Aws::Http::CreateHttpClient(cfg);
}

std::string ReadBody(const std::shared_ptr<Aws::Http::HttpResponse>& resp) {
  if (!resp)
    return {};
  std::ostringstream ss;
  ss << resp->GetResponseBody().rdbuf();
  return ss.str();
}

// Empty return means IMDSv2 is unavailable — caller must fall back to V1.
std::string TryImdsV2Token(Aws::Http::HttpClient& http) {
  const std::string url = std::string(kImdsHost) + kImdsV2TokenPath;
  auto req = Aws::Http::CreateHttpRequest(url, Aws::Http::HttpMethod::HTTP_PUT,
                                          Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  req->SetHeaderValue(kImdsV2TtlHeader, std::to_string(kImdsV2TtlSecs).c_str());
  req->SetContentLength("0");
  req->SetUserAgent(Aws::Client::ComputeUserAgentString());

  auto resp = http.MakeRequest(req);
  if (!resp || resp->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    return {};
  }
  auto token = ReadBody(resp);
  return std::string(Aws::Utils::StringUtils::Trim(token.c_str()).c_str());
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
};

bool FetchImdsCreds(ImdsCreds& out) {
  auto http = MakeImdsHttpClient();
  if (!http)
    return false;

  // V2 first; an empty return drops us back to V1-style bare GETs. Probing
  // order is cheap — one PUT — and lets the same code path cover both
  // "hardened mode" and V1-only instances.
  const std::string v2_token = TryImdsV2Token(*http);

  auto list_req = MakeImdsGet(std::string(kImdsHost) + kImdsRoleListPath, v2_token);
  auto list_resp = http->MakeRequest(list_req);
  if (!list_resp || list_resp->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    AWS_LOGSTREAM_ERROR(kLogTag, "IMDS role list request failed; no RAM role attached to this ECS?");
    return false;
  }
  const auto role_name = std::string(Aws::Utils::StringUtils::Trim(ReadBody(list_resp).c_str()).c_str());
  if (role_name.empty()) {
    AWS_LOGSTREAM_ERROR(kLogTag, "IMDS returned empty role name");
    return false;
  }

  const std::string creds_url = std::string(kImdsHost) + kImdsRoleListPath + role_name;
  auto creds_req = MakeImdsGet(creds_url, v2_token);
  auto creds_resp = http->MakeRequest(creds_req);
  if (!creds_resp || creds_resp->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    AWS_LOGSTREAM_ERROR(kLogTag, "IMDS credentials request failed for role " << role_name);
    return false;
  }

  Aws::Utils::Json::JsonValue json(ReadBody(creds_resp).c_str());
  if (!json.WasParseSuccessful()) {
    AWS_LOGSTREAM_ERROR(kLogTag, "IMDS credentials JSON parse failed: " << json.GetErrorMessage());
    return false;
  }
  auto view = json.View();
  if (!view.KeyExists("AccessKeyId") || !view.KeyExists("AccessKeySecret") || !view.KeyExists("SecurityToken")) {
    AWS_LOGSTREAM_ERROR(kLogTag, "IMDS credentials response missing expected fields");
    return false;
  }
  out.access_key_id = view.GetString("AccessKeyId");
  out.access_key_secret = view.GetString("AccessKeySecret");
  out.security_token = view.GetString("SecurityToken");
  return true;
}

}  // namespace

AliyunRAMCredentialsProvider::AliyunRAMCredentialsProvider(const Aws::String& role_arn,
                                                           const Aws::String& role_session_name,
                                                           const Aws::String& external_id)
    : m_roleArn(role_arn), m_roleSessionName(role_session_name), m_externalId(external_id) {
  if (m_roleSessionName.empty()) {
    m_roleSessionName = Aws::Utils::UUID::RandomUUID();
  }

  Aws::Client::ClientConfiguration cfg;
  cfg.scheme = Aws::Http::Scheme::HTTPS;
  m_stsClient = Aws::MakeUnique<AliyunRAMSTSClient>(kLogTag, cfg);

  AWS_LOGSTREAM_INFO(kLogTag,
                     "Created RAM provider for role_arn=" << m_roleArn << " session=" << m_roleSessionName
                                                          << (m_externalId.empty() ? "" : " (with external_id)"));
}

Aws::Auth::AWSCredentials AliyunRAMCredentialsProvider::GetAWSCredentials() {
  RefreshIfExpired();
  Aws::Utils::Threading::ReaderLockGuard guard(m_reloadLock);
  return m_credentials;
}

bool AliyunRAMCredentialsProvider::ExpiresSoon() const {
  return ((m_credentials.GetExpiration() - Aws::Utils::DateTime::Now()).count() < kRefreshGraceMs);
}

void AliyunRAMCredentialsProvider::RefreshIfExpired() {
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

void AliyunRAMCredentialsProvider::Reload() {
  AWS_LOGSTREAM_INFO(kLogTag, "Credentials missing or expiring; refreshing via IMDS → AssumeRole.");

  ImdsCreds imds;
  if (!FetchImdsCreds(imds)) {
    AWS_LOGSTREAM_ERROR(kLogTag, "Failed to fetch ECS IMDS credentials");
    return;
  }

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = imds.access_key_id.c_str();
  req.callerAccessKeySecret = imds.access_key_secret.c_str();
  req.callerSecurityToken = imds.security_token.c_str();
  req.roleArn = m_roleArn;
  req.roleSessionName = m_roleSessionName;
  req.externalId = m_externalId;

  auto res = m_stsClient->GetAssumeRoleCredentials(req);
  if (res.creds.IsEmpty()) {
    AWS_LOGSTREAM_ERROR(kLogTag, "AssumeRole returned empty credentials");
    return;
  }
  m_credentials = res.creds;
  AWS_LOGSTREAM_INFO(kLogTag, "AssumeRole succeeded; ak="
                                  << m_credentials.GetAWSAccessKeyId() << " expires="
                                  << m_credentials.GetExpiration().ToGmtString(Aws::Utils::DateFormat::ISO_8601));
}

}  // namespace milvus_storage
