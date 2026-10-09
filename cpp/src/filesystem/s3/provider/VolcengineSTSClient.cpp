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

#include "milvus-storage/filesystem/s3/provider/VolcengineSTSClient.h"
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/core/client/AWSClient.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/AWSErrorMarshaller.h>
#include <aws/core/client/CoreErrors.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/core/platform/Environment.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <limits.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>

namespace milvus_storage {
using Aws::Http::HttpClient;
using Aws::Http::HttpRequest;
using Aws::Http::HttpResponseCode;

static const char STS_RESOURCE_CLIENT_LOG_TAG[] = "VolcengineSTSResourceClient";

// Volcengine STS DurationSeconds valid range. Values outside this window are
// clamped at construction time to avoid a hard STS rejection at refresh time.
static const int kMinDurationSeconds = 900;    // 15 min
static const int kMaxDurationSeconds = 43200;  // 12 h

namespace {

// Normalize an RFC3339 timestamp to UTC "...Z" so AWS SDK's ISO_8601 parser
// returns the correct instant. Accepts:
//   - "<datetime>(.frac)?Z"        pass-through (fractional seconds dropped)
//   - "<datetime>(.frac)?[+-]HH:MM" numeric offset, folded to UTC
//   - "<datetime>(.frac)?[+-]HHMM"  colon-less offset (SDK 1.11.842 silently
//                                   mis-parses this form), folded to UTC
// Returns Invalid for any other shape. We intentionally do NOT fall back to
// the SDK's lax ISO_8601 parse, since WasParseSuccessful() returns true for
// several offset-bearing strings whose offset the SDK does not apply.
arrow::Result<std::string> NormalizeToUtcZ(const std::string& s) {
  static const std::regex re(R"(^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})$)");
  std::smatch m;
  if (!std::regex_match(s, m, re)) {
    return arrow::Status::Invalid("invalid RFC3339 timestamp: ", s);
  }
  std::tm tm_in{};
  tm_in.tm_year = std::stoi(m[1]) - 1900;
  tm_in.tm_mon = std::stoi(m[2]) - 1;
  tm_in.tm_mday = std::stoi(m[3]);
  tm_in.tm_hour = std::stoi(m[4]);
  tm_in.tm_min = std::stoi(m[5]);
  tm_in.tm_sec = std::stoi(m[6]);
  std::time_t t = timegm(&tm_in);  // treat captured fields as UTC, then apply offset below
  const std::string off = m[7];
  if (off != "Z") {
    const int sign = (off[0] == '+') ? -1 : 1;  // "+08:00" -> UTC = local - 8h
    const int hh = std::stoi(off.substr(1, 2));
    int mm = 0;
    if (off.size() == 6) {  // +HH:MM
      mm = std::stoi(off.substr(4, 2));
    } else if (off.size() == 5) {  // +HHMM
      mm = std::stoi(off.substr(3, 2));
    }
    t += sign * (hh * 3600 + mm * 60);
  }
  // Thread-safe UTC breakdown: `gmtime` returns a pointer to a process-wide
  // shared tm that concurrent provider refreshes can clobber between here and
  // strftime. Use the reentrant form with a stack buffer.
  std::tm utc{};
  if (gmtime_r(&t, &utc) == nullptr) {
    return arrow::Status::Invalid("gmtime_r failed for timestamp: ", s);
  }
  char buf[32];
  if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
    return arrow::Status::Invalid("strftime failed for timestamp: ", s);
  }
  return std::string(buf);
}

// Volcengine V4 (HMAC-SHA256) signing primitives. Hand-rolled — the AWS SDK's
// AWSAuthV4Signer hard-codes 5 things that Volcengine does differently:
//   1. algorithm token         "AWS4-HMAC-SHA256"      -> "HMAC-SHA256"
//   2. signing-key first HMAC   HMAC("AWS4"+secret,...) -> HMAC(secret,...)  (no "AWS4")
//   3. credential-scope tail    "aws4_request"          -> "request"
//   4. request-time header      X-Amz-Date              -> X-Date
//   5. session-token header     X-Amz-Security-Token    -> X-Security-Token (unsigned)
// so the signature is produced here rather than reusing the SDK signer.
std::vector<uint8_t> HmacSha256(const uint8_t* key, size_t key_len, const std::string& data) {
  std::vector<uint8_t> out(EVP_MAX_MD_SIZE);
  unsigned int len = 0;
  HMAC(EVP_sha256(), key, static_cast<int>(key_len), reinterpret_cast<const unsigned char*>(data.data()), data.size(),
       out.data(), &len);
  out.resize(len);
  return out;
}

std::vector<uint8_t> Sha256(const std::string& data) {
  std::vector<uint8_t> out(EVP_MAX_MD_SIZE);
  unsigned int len = 0;
  EVP_Digest(data.data(), data.size(), out.data(), &len, EVP_sha256(), nullptr);
  out.resize(len);
  return out;
}

std::string HexEncode(const std::vector<uint8_t>& bytes) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0x0f]);
  }
  return out;
}

// RFC 3986 percent-encoding (unreserved: A-Z a-z 0-9 - _ . ~). Uppercase hex,
// space -> %20. Matches what Volcengine V4 expects for query values and the
// form body. '/' is always encoded here (no path segments carry a slash).
std::string UriEncode(const std::string& value) {
  static const char kHex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(value.size() * 3);
  for (unsigned char c : value) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 0x0f]);
    }
  }
  return out;
}

// Volcengine STS returns errors as JSON under ResponseMetadata.Error — the
// stock XmlErrorMarshaller yields a blank AWSError in that case and the SDK
// then discards the response body before the application can inspect it,
// erasing the Error.Code / Error.Message / RequestId that operators need.
// This marshaller reads the body eagerly and preserves those three fields.
class VolcengineJsonErrorMarshaller final : public Aws::Client::JsonErrorMarshaller {
  public:
  Aws::Client::AWSError<Aws::Client::CoreErrors> Marshall(const Aws::Http::HttpResponse& response) const override {
    auto& stream = response.GetResponseBody();
    const auto saved_pos = stream.tellg();
    std::ostringstream oss;
    oss << stream.rdbuf();
    const std::string payload = oss.str();
    stream.clear();
    stream.seekg(saved_pos);

    Aws::Utils::Json::JsonValue root(Aws::String(payload.c_str(), payload.size()));
    if (!root.WasParseSuccessful()) {
      return Aws::Client::JsonErrorMarshaller::Marshall(response);
    }
    const auto view = root.View();
    if (!view.ValueExists("ResponseMetadata")) {
      return Aws::Client::JsonErrorMarshaller::Marshall(response);
    }
    const auto rm = view.GetObject("ResponseMetadata");
    Aws::String code;
    Aws::String message;
    if (rm.ValueExists("Error")) {
      const auto err = rm.GetObject("Error");
      code = err.GetString("Code");
      message = err.GetString("Message");
    }
    Aws::String request_id;
    if (rm.ValueExists("RequestId")) {
      request_id = rm.GetString("RequestId");
    }
    if (code.empty() && message.empty() && request_id.empty()) {
      return Aws::Client::JsonErrorMarshaller::Marshall(response);
    }
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::UNKNOWN, code, message,
                                                         /*isRetryable=*/false);
    error.SetRequestId(request_id);
    error.SetResponseHeaders(response.GetHeaders());
    error.SetResponseCode(response.GetResponseCode());
    return error;
  }
};

// Parse an STS expiry string into an AWS DateTime, tolerating formats the docs
// never actually pin down. Volcengine's AssumeRole / AssumeRoleWithOIDC docs
// only give an *example* value ("2021-04-12T11:57:09+08:00") with no field-level
// format contract, so we must not assume the offset is always +HH:MM. Order:
//   1. NormalizeToUtcZ  — folds a +HH:MM / +HHMM / Z offset to a Z instant;
// If the input is unparseable we DO NOT drop the credentials: the expiry only
// drives our local refresh clock (it is never signed or sent upstream), so
// losing it must never invalidate an otherwise-good AK/SK/token. We fall back
// to a conservative TTL, causing an early proactive re-fetch rather than a
// hard failure. Returns false only when even the fallback cannot be applied.
bool ParseStsExpiry(const Aws::String& rawExpiry, Aws::Auth::AWSCredentials& creds) {
  const Aws::String trimmed = Aws::Utils::StringUtils::Trim(rawExpiry.c_str());

  auto normalized = NormalizeToUtcZ(std::string(trimmed.c_str()));
  if (normalized.ok()) {
    Aws::Utils::DateTime dt(normalized->c_str(), Aws::Utils::DateFormat::ISO_8601);
    if (dt.WasParseSuccessful()) {
      creds.SetExpiration(dt);
      return true;
    }
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                       "ISO-8601 parse rejected normalized STS expiry '" << normalized.ValueOrDie().c_str() << "'");
  } else {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                       "NormalizeToUtcZ rejected STS expiry '" << trimmed << "': " << normalized.status().ToString());
  }

  // Fallback: keep the credentials but assign a conservative TTL so the
  // provider re-fetches well before an unknown real lifetime could lapse. Kept
  // above the chain provider's refresh grace so we still cache for a usable
  // window instead of re-fetching on every single call.
  static const int kFallbackTtlSeconds = 60 * 60;  // 1h
  const Aws::Utils::DateTime fallback = Aws::Utils::DateTime::Now() + std::chrono::seconds(kFallbackTtlSeconds);
  creds.SetExpiration(fallback);
  AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                     "Unparseable STS expiry '" << trimmed << "'; keeping credentials with a " << kFallbackTtlSeconds
                                                << "s fallback TTL to force an early refresh");
  return true;
}

int ClampDurationSeconds(int duration) {
  if (duration < kMinDurationSeconds) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "DurationSeconds=" << duration << " below Volcengine STS minimum "
                                                                       << kMinDurationSeconds << "; clamping");
    return kMinDurationSeconds;
  }
  if (duration > kMaxDurationSeconds) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "DurationSeconds=" << duration << " above Volcengine STS maximum "
                                                                       << kMaxDurationSeconds << "; clamping");
    return kMaxDurationSeconds;
  }
  return duration;
}

}  // namespace

VolcengineSTSCredentialsClient::VolcengineSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration, int duration_seconds)
    : AWSHttpResourceClient(clientConfiguration, STS_RESOURCE_CLIENT_LOG_TAG),
      m_durationSeconds(ClampDurationSeconds(duration_seconds)) {
  // Volcengine STS speaks JSON: a dedicated JSON error marshaller lets us
  // surface Error.Code / Error.Message / RequestId instead of the empty
  // diagnostic produced by the stock XmlErrorMarshaller.
  SetErrorMarshaller(Aws::MakeUnique<VolcengineJsonErrorMarshaller>(STS_RESOURCE_CLIENT_LOG_TAG));

  m_endpoint = Aws::Environment::GetEnv("VOLCENGINE_OIDC_STS_ENDPOINT");
  if (m_endpoint.empty()) {
    m_endpoint = "sts.volcengineapi.com";
  }

  // Volcengine V4 credential scope. Region defaults to cn-beijing (STS is
  // reachable from that region regardless of the bucket's region); service
  // is fixed to "sts". Both feed CredentialScope=<date>/<region>/<service>/request.
  m_region = Aws::Environment::GetEnv("VOLCENGINE_STS_REGION");
  if (m_region.empty()) {
    m_region = "cn-beijing";
  }
  m_service = "sts";

  AWS_LOGSTREAM_INFO(STS_RESOURCE_CLIENT_LOG_TAG,
                     "Creating STS ResourceClient with endpoint: " << m_endpoint << ", region: " << m_region
                                                                   << ", durationSeconds: " << m_durationSeconds);
}

VolcengineSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
VolcengineSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
  Aws::StringStream ss;
  ss << "OIDCToken=" << Aws::Utils::StringUtils::URLEncode(request.webIdentityToken.c_str())
     << "&DurationSeconds=" << Aws::Utils::StringUtils::URLEncode(m_durationSeconds)
     << "&RoleSessionName=" << Aws::Utils::StringUtils::URLEncode(request.roleSessionName.c_str());

  Aws::StringStream urlStream;
  urlStream << "https://" << m_endpoint << "/?Action=AssumeRoleWithOIDC"
            << "&Version=2018-01-01"
            << "&RoleTrn=" << Aws::Utils::StringUtils::URLEncode(request.roleArn.c_str());

  std::shared_ptr<Aws::Http::HttpRequest> httpRequest(Aws::Http::CreateHttpRequest(
      urlStream.str(), Aws::Http::HttpMethod::HTTP_POST, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));

  httpRequest->SetHeaderValue("Host", m_endpoint);
  httpRequest->SetHeaderValue("Content-Type", "application/x-www-form-urlencoded");

  std::shared_ptr<Aws::IOStream> body = Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
  *body << ss.str();

  httpRequest->AddContentBody(body);
  body->seekg(0, body->end);
  auto streamSize = body->tellg();
  body->seekg(0, body->beg);
  Aws::StringStream contentLength;
  contentLength << streamSize;
  httpRequest->SetContentLength(contentLength.str());

  Aws::String credentialsStr = GetResourceWithAWSWebServiceResult(httpRequest).GetPayload();

  // Parse credentials
  STSAssumeRoleWithWebIdentityResult result;
  if (credentialsStr.empty()) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Get an empty credential from sts");
    return result;
  }

  Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
  if (!jsonValue.WasParseSuccessful()) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Failed to parse STS response as JSON");
    return result;
  }
  auto json = jsonValue.View();
  if (!json.ValueExists("Result")) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Get 'Result' node from response failed");
    return result;
  }
  auto resultNode = json.GetObject("Result");

  if (!resultNode.ValueExists("Credentials")) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Get 'Credentials' node from Result failed");
    return result;
  }
  auto credentialsNode = resultNode.GetObject("Credentials");

  result.creds.SetAWSAccessKeyId(credentialsNode.GetString("AccessKeyId"));
  result.creds.SetAWSSecretKey(credentialsNode.GetString("SecretAccessKey"));
  result.creds.SetSessionToken(credentialsNode.GetString("SessionToken"));
  // AssumeRoleWithOIDC reports the expiry as "Expiration". Parse defensively:
  // the docs only give an example value with no format contract.
  ParseStsExpiry(credentialsNode.GetString("Expiration"), result.creds);

  return result;
}

VolcengineSTSCredentialsClient::V4SignResult VolcengineSTSCredentialsClient::SignRequestV4(
    const Aws::String& accessKeyId,
    const Aws::String& secretAccessKey,
    const Aws::String& region,
    const Aws::String& service,
    const Aws::String& host,
    const Aws::String& canonicalQuery,
    const Aws::String& body,
    const Aws::String& contentType,
    const Aws::String& xDate,
    const Aws::String& scopeDate) {
  V4SignResult out;

  const std::string body_str(body.c_str());
  const std::string payload_hash = HexEncode(Sha256(body_str));

  // Canonical headers (sorted, lowercase name, trimmed value). X-Security-Token
  // is deliberately excluded — Volcengine carries the caller's session token in
  // that header but does not include it in the signature.
  std::ostringstream canonical_headers;
  canonical_headers << "content-type:" << contentType.c_str() << '\n'
                    << "host:" << host.c_str() << '\n'
                    << "x-content-sha256:" << payload_hash << '\n'
                    << "x-date:" << xDate.c_str() << '\n';
  const std::string signed_headers = "content-type;host;x-content-sha256;x-date";

  // CanonicalRequest = Method \n URI \n Query \n CanonicalHeaders \n SignedHeaders \n HashedPayload
  std::ostringstream canonical_request;
  canonical_request << "POST" << '\n'
                    << "/" << '\n'
                    << canonicalQuery.c_str() << '\n'
                    << canonical_headers.str() << '\n'
                    << signed_headers << '\n'
                    << payload_hash;

  // CredentialScope = <date>/<region>/<service>/request  (tail is "request",
  // NOT "aws4_request").
  const std::string credential_scope =
      std::string(scopeDate.c_str()) + "/" + region.c_str() + "/" + service.c_str() + "/request";

  // StringToSign = "HMAC-SHA256" \n X-Date \n CredentialScope \n hex(SHA256(CanonicalRequest))
  std::ostringstream string_to_sign;
  string_to_sign << "HMAC-SHA256" << '\n'
                 << xDate.c_str() << '\n'
                 << credential_scope << '\n'
                 << HexEncode(Sha256(canonical_request.str()));

  // Signing-key derivation. First HMAC keys on the raw secret (no "AWS4"
  // prefix); the chain terminates on "request" (not "aws4_request").
  const std::string secret(secretAccessKey.c_str());
  std::vector<uint8_t> k_date =
      HmacSha256(reinterpret_cast<const uint8_t*>(secret.data()), secret.size(), std::string(scopeDate.c_str()));
  std::vector<uint8_t> k_region = HmacSha256(k_date.data(), k_date.size(), std::string(region.c_str()));
  std::vector<uint8_t> k_service = HmacSha256(k_region.data(), k_region.size(), std::string(service.c_str()));
  std::vector<uint8_t> k_signing = HmacSha256(k_service.data(), k_service.size(), "request");
  const std::string signature = HexEncode(HmacSha256(k_signing.data(), k_signing.size(), string_to_sign.str()));

  const std::string authorization = std::string("HMAC-SHA256 Credential=") + accessKeyId.c_str() + "/" +
                                    credential_scope + ", SignedHeaders=" + signed_headers + ", Signature=" + signature;

  out.authorization = authorization.c_str();
  out.signature = signature.c_str();
  out.signedHeaders = signed_headers.c_str();
  out.xContentSha256 = payload_hash.c_str();
  return out;
}

VolcengineSTSCredentialsClient::STSAssumeRoleResult VolcengineSTSCredentialsClient::GetAssumeRoleCredentials(
    const STSAssumeRoleRequest& request) {
  STSAssumeRoleResult result;

  // --- Request layout -----------------------------------------------------
  // Common params (Action, Version) live in the query string; the business
  // params (RoleTrn, RoleSessionName) live in a form-urlencoded body. RoleTrn
  // carries colons and slashes ("trn:iam::<acct>:role/<name>"), and the AWS
  // SDK's URI normaliser would silently re-encode those in the query and
  // break the signature — the exact trap AliyunRAMSTSClient hit. Keeping the
  // tricky value in the body (signed via HashedPayload) sidesteps it, and the
  // query holds only signature-stable alphanumerics.
  std::ostringstream body_stream;
  body_stream << "DurationSeconds=" << m_durationSeconds
              << "&RoleSessionName=" << UriEncode(std::string(request.roleSessionName.c_str()))
              << "&RoleTrn=" << UriEncode(std::string(request.roleTrn.c_str()));
  const std::string body_str = body_stream.str();

  // Canonical query string: sorted "enc(k)=enc(v)" joined by '&'. Action
  // sorts before Version, matching V4's required ascending key order.
  const std::string canonical_query = "Action=AssumeRole&Version=2018-01-01";

  // X-Date is YYYYMMDD'T'HHMMSS'Z' (basic ISO-8601, no separators); the
  // credential-scope date is the YYYYMMDD prefix. Both must come from one
  // clock read so they never straddle a day boundary.
  std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&now, &tm_utc);
  char x_date[32];
  std::strftime(x_date, sizeof(x_date), "%Y%m%dT%H%M%SZ", &tm_utc);
  char scope_date[16];
  std::strftime(scope_date, sizeof(scope_date), "%Y%m%d", &tm_utc);

  const std::string content_type = "application/x-www-form-urlencoded";
  const V4SignResult sig =
      SignRequestV4(request.callerAccessKeyId, request.callerAccessKeySecret, m_region, m_service, m_endpoint,
                    canonical_query.c_str(), body_str.c_str(), content_type.c_str(), x_date, scope_date);

  // --- Send ---------------------------------------------------------------
  Aws::StringStream urlStream;
  urlStream << "https://" << m_endpoint << "/?" << canonical_query.c_str();

  std::shared_ptr<Aws::Http::HttpRequest> httpRequest(Aws::Http::CreateHttpRequest(
      urlStream.str(), Aws::Http::HttpMethod::HTTP_POST, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));

  httpRequest->SetHeaderValue("Host", m_endpoint);
  httpRequest->SetHeaderValue("Content-Type", content_type.c_str());
  httpRequest->SetHeaderValue("X-Date", x_date);
  httpRequest->SetHeaderValue("X-Content-Sha256", sig.xContentSha256);
  httpRequest->SetHeaderValue("Authorization", sig.authorization);
  // Session token of the step-1 caller. Present in the header but not signed.
  if (!request.callerSecurityToken.empty()) {
    httpRequest->SetHeaderValue("X-Security-Token", request.callerSecurityToken);
  }

  std::shared_ptr<Aws::IOStream> body = Aws::MakeShared<Aws::StringStream>(STS_RESOURCE_CLIENT_LOG_TAG);
  *body << body_str;
  httpRequest->AddContentBody(body);
  Aws::StringStream contentLength;
  contentLength << body_str.size();
  httpRequest->SetContentLength(contentLength.str());

  Aws::String credentialsStr = GetResourceWithAWSWebServiceResult(httpRequest).GetPayload();

  if (credentialsStr.empty()) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Get an empty credential from sts AssumeRole");
    return result;
  }

  Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
  if (!jsonValue.WasParseSuccessful()) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Failed to parse AssumeRole response as JSON: " << credentialsStr);
    return result;
  }
  auto json = jsonValue.View();
  if (!json.ValueExists("Result")) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                       "Get 'Result' node from AssumeRole response failed: " << credentialsStr);
    return result;
  }
  auto resultNode = json.GetObject("Result");
  if (!resultNode.ValueExists("Credentials")) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Get 'Credentials' node from AssumeRole Result failed");
    return result;
  }
  auto credentialsNode = resultNode.GetObject("Credentials");

  result.creds.SetAWSAccessKeyId(credentialsNode.GetString("AccessKeyId"));
  result.creds.SetAWSSecretKey(credentialsNode.GetString("SecretAccessKey"));
  result.creds.SetSessionToken(credentialsNode.GetString("SessionToken"));
  // AssumeRole names the expiry "ExpiredTime" whereas AssumeRoleWithOIDC uses
  // "Expiration" — read the former first and fall back to the latter so an
  // empty field never reaches the parser. ParseStsExpiry then tolerates any
  // format drift (the docs only give an example value, no format contract).
  Aws::String expiredTimeStr = credentialsNode.GetString("ExpiredTime");
  if (expiredTimeStr.empty()) {
    expiredTimeStr = credentialsNode.GetString("Expiration");
  }
  ParseStsExpiry(expiredTimeStr, result.creds);

  return result;
}

}  // namespace milvus_storage
