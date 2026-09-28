#include "milvus-storage/filesystem/s3/provider/VolcengineSTSClient.h"
#include "milvus-storage/common/Util.h"
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/CoreErrors.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <limits.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <exception>
#include <ctime>
#include <map>
#include <mutex>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>

namespace milvus_storage {
using Aws::Http::HttpClient;
using Aws::Http::HttpRequest;
using Aws::Http::HttpResponseCode;

static const char STS_RESOURCE_CLIENT_LOG_TAG[] = "VolcengineSTSResourceClient";
static const int DefaultDurationSeconds = 8 * 60 * 60;  // 8h

namespace {

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

// Parse an STS expiry string into an AWS DateTime, tolerating formats the docs
// never actually pin down. Volcengine's AssumeRole / AssumeRoleWithOIDC docs
// only give an *example* value ("2021-04-12T11:57:09+08:00") with no field-level
// format contract, so we must not assume the offset is always +HH:MM. Order:
//   1. NormalizeToUtcZ  — folds a +HH:MM / -HH:MM offset to a Z instant;
//   2. raw ISO_8601     — covers a plain "...Z" or any form the SDK groks;
// If neither yields a usable instant we DO NOT drop the credentials: the expiry
// only drives our local refresh clock (it is never signed or sent upstream), so
// losing it must never invalidate an otherwise-good AK/SK/token. We fall back to
// a conservative TTL, causing an early proactive re-fetch rather than a hard
// failure. Returns false only when even the fallback cannot be applied.
bool ParseStsExpiry(const Aws::String& rawExpiry, Aws::Auth::AWSCredentials& creds) {
  Aws::String trimmed = Aws::Utils::StringUtils::Trim(rawExpiry.c_str());

  // Attempt 1: normalize a numeric offset to UTC Z, then parse.
  try {
    std::string normalized = milvus_storage::NormalizeToUtcZ(std::string(trimmed.c_str()));
    Aws::Utils::DateTime dt(normalized.c_str(), Aws::Utils::DateFormat::ISO_8601);
    if (dt.WasParseSuccessful()) {
      creds.SetExpiration(dt);
      return true;
    }
  } catch (const std::exception& e) {
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                       "NormalizeToUtcZ rejected STS expiry '" << trimmed << "': " << e.what()
                       << "; trying raw ISO-8601 parse");
  }

  // Attempt 2: hand the untouched string to the SDK parser (e.g. plain "...Z"
  // or a fractional-second form NormalizeToUtcZ's strict regex refuses).
  if (!trimmed.empty()) {
    Aws::Utils::DateTime dt(trimmed, Aws::Utils::DateFormat::ISO_8601);
    if (dt.WasParseSuccessful()) {
      creds.SetExpiration(dt);
      return true;
    }
  }

  // Fallback: keep the credentials but assign a conservative TTL so the
  // provider re-fetches well before an unknown real lifetime could lapse. Kept
  // above the chain provider's 30-min refresh grace so we still cache for a
  // usable window instead of re-fetching on every single call.
  static const int kFallbackTtlSeconds = 60 * 60;  // 1h
  Aws::Utils::DateTime fallback = Aws::Utils::DateTime::Now() +
      std::chrono::seconds(kFallbackTtlSeconds);
  creds.SetExpiration(fallback);
  AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                     "Unparseable STS expiry '" << trimmed
                     << "'; keeping credentials with a " << kFallbackTtlSeconds
                     << "s fallback TTL to force an early refresh");
  return true;
}

}  // namespace

VolcengineSTSCredentialsClient::VolcengineSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, STS_RESOURCE_CLIENT_LOG_TAG) {
    SetErrorMarshaller(Aws::MakeUnique<Aws::Client::XmlErrorMarshaller>(
        STS_RESOURCE_CLIENT_LOG_TAG));

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

    AWS_LOGSTREAM_INFO(
        STS_RESOURCE_CLIENT_LOG_TAG,
        "Creating STS ResourceClient with endpoint: " << m_endpoint
            << ", region: " << m_region);
}

VolcengineSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
VolcengineSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
    Aws::StringStream ss;
    ss << "OIDCToken="
       << Aws::Utils::StringUtils::URLEncode(request.webIdentityToken.c_str())
       << "&DurationSeconds="
       << Aws::Utils::StringUtils::URLEncode(DefaultDurationSeconds)
       << "&RoleSessionName="
       << Aws::Utils::StringUtils::URLEncode(request.roleSessionName.c_str());

    Aws::StringStream urlStream;
    urlStream << "https://" << m_endpoint << "/?Action=AssumeRoleWithOIDC"
              << "&Version=2018-01-01"
              << "&RoleTrn="
              << Aws::Utils::StringUtils::URLEncode(request.roleArn.c_str());

    std::shared_ptr<Aws::Http::HttpRequest> httpRequest(
        Aws::Http::CreateHttpRequest(
            urlStream.str(),
            Aws::Http::HttpMethod::HTTP_POST,
            Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));

    httpRequest->SetHeaderValue("Host", m_endpoint);
    httpRequest->SetHeaderValue("Content-Type",
                                "application/x-www-form-urlencoded");

    std::shared_ptr<Aws::IOStream> body =
        Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
    *body << ss.str();

    httpRequest->AddContentBody(body);
    body->seekg(0, body->end);
    auto streamSize = body->tellg();
    body->seekg(0, body->beg);
    Aws::StringStream contentLength;
    contentLength << streamSize;
    httpRequest->SetContentLength(contentLength.str());

    Aws::String credentialsStr =
        GetResourceWithAWSWebServiceResult(httpRequest).GetPayload();

    // Parse credentials
    STSAssumeRoleWithWebIdentityResult result;
    if (credentialsStr.empty()) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get an empty credential from sts");
        return result;
    }

    Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
    if (!jsonValue.WasParseSuccessful()) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Failed to parse STS response as JSON");
        return result;
    }
    auto json = jsonValue.View();
    if (!json.ValueExists("Result")) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get 'Result' node from response failed");
        return result;
    }
    auto resultNode = json.GetObject("Result");

    if (!resultNode.ValueExists("Credentials")) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get 'Credentials' node from Result failed");
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
                                    credential_scope + ", SignedHeaders=" + signed_headers +
                                    ", Signature=" + signature;

  out.authorization = authorization.c_str();
  out.signature = signature.c_str();
  out.signedHeaders = signed_headers.c_str();
  out.xContentSha256 = payload_hash.c_str();
  return out;
}

VolcengineSTSCredentialsClient::STSAssumeRoleResult
VolcengineSTSCredentialsClient::GetAssumeRoleCredentials(
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
    body_stream << "RoleTrn=" << UriEncode(std::string(request.roleTrn.c_str()))
                << "&RoleSessionName=" << UriEncode(std::string(request.roleSessionName.c_str()));
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
    const V4SignResult sig = SignRequestV4(request.callerAccessKeyId, request.callerAccessKeySecret, m_region,
                                           m_service, m_endpoint, canonical_query.c_str(), body_str.c_str(),
                                           content_type.c_str(), x_date, scope_date);

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

    std::shared_ptr<Aws::IOStream> body =
        Aws::MakeShared<Aws::StringStream>(STS_RESOURCE_CLIENT_LOG_TAG);
    *body << body_str;
    httpRequest->AddContentBody(body);
    Aws::StringStream contentLength;
    contentLength << body_str.size();
    httpRequest->SetContentLength(contentLength.str());

    Aws::String credentialsStr =
        GetResourceWithAWSWebServiceResult(httpRequest).GetPayload();

    if (credentialsStr.empty()) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get an empty credential from sts AssumeRole");
        return result;
    }

    Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
    if (!jsonValue.WasParseSuccessful()) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Failed to parse AssumeRole response as JSON: " << credentialsStr);
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
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get 'Credentials' node from AssumeRole Result failed");
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
