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

#include "milvus-storage/filesystem/s3/provider/AliyunRAMSTSClient.h"

#include <cctype>
#include <cstdint>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <aws/core/client/AWSErrorMarshaller.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/core/utils/xml/XmlSerializer.h>

namespace milvus_storage {

static const char kLogTag[] = "AliyunRAMSTSClient";

namespace {

// POP v1 percent-encoding: RFC 3986 unreserved set
// (A-Z / a-z / 0-9 / '-' / '_' / '.' / '~'). Space becomes "%20", never "+".
std::string PopUrlEncode(const std::string& value) {
  std::ostringstream out;
  out.fill('0');
  out << std::hex << std::uppercase;
  for (unsigned char c : value) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out << static_cast<char>(c);
    } else {
      out << '%' << std::setw(2) << static_cast<int>(c);
    }
  }
  return out.str();
}

std::vector<uint8_t> HmacSha1(const std::string& key, const std::string& data) {
  std::vector<uint8_t> out(EVP_MAX_MD_SIZE);
  unsigned int len = 0;
  HMAC(EVP_sha1(), key.data(), static_cast<int>(key.size()), reinterpret_cast<const unsigned char*>(data.data()),
       data.size(), out.data(), &len);
  out.resize(len);
  return out;
}

}  // namespace

AliyunRAMSTSClient::AliyunRAMSTSClient(const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, kLogTag) {
  SetErrorMarshaller(Aws::MakeUnique<Aws::Client::XmlErrorMarshaller>(kLogTag));
  // Aliyun STS is region-agnostic; this endpoint is valid from every region.
  m_endpoint = "https://sts.aliyuncs.com/";
  AWS_LOGSTREAM_INFO(kLogTag, "Creating RAM STS client with endpoint: " << m_endpoint);
}

AliyunRAMSTSClient::AssumeRoleResult AliyunRAMSTSClient::GetAssumeRoleCredentials(const AssumeRoleRequest& request) {
  AssumeRoleResult result;

  // std::map sorts keys ASCII-ascending, which is what POP v1 canonical-query
  // construction needs. Values will be URL-encoded when we emit the string.
  std::map<std::string, std::string> params;
  params["AccessKeyId"] = request.callerAccessKeyId;
  params["Action"] = "AssumeRole";
  params["Format"] = "XML";
  params["RoleArn"] = request.roleArn;
  params["RoleSessionName"] = request.roleSessionName;
  // SecurityToken is mandatory when the caller credentials are themselves
  // temporary (e.g. STS creds from ECS IMDS). Omitting it for long-term
  // AK/SK is also required by Aliyun (they disagree on which set signed).
  if (!request.callerSecurityToken.empty()) {
    params["SecurityToken"] = request.callerSecurityToken;
  }
  // ExternalId is optional and only included when the caller sets it.
  // The target role's trust policy decides whether ExternalId is required;
  // sending an empty value would still flip the request to the
  // "ExternalId-supplied" branch on Aliyun's side, which fails the policy
  // check, so the explicit non-empty guard matters.
  if (!request.externalId.empty()) {
    params["ExternalId"] = request.externalId;
  }
  params["SignatureMethod"] = "HMAC-SHA1";
  // 64-bit random nonce. UUID alone is enough but cheap insurance against
  // correlated clocks on the same host.
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  const Aws::String uuid = Aws::Utils::UUID::RandomUUID();
  Aws::StringStream nonce;
  nonce << dist(gen) << "-" << uuid;
  params["SignatureNonce"] = nonce.str().c_str();
  params["SignatureVersion"] = "1.0";
  params["Timestamp"] = Aws::Utils::DateTime::Now().ToGmtString(Aws::Utils::DateFormat::ISO_8601).c_str();
  params["Version"] = "2015-04-01";

  // Canonical query: sorted "URLEncode(k)=URLEncode(v)" joined by '&'.
  std::ostringstream canonical;
  bool first = true;
  for (const auto& kv : params) {
    if (!first)
      canonical << '&';
    first = false;
    canonical << PopUrlEncode(kv.first) << '=' << PopUrlEncode(kv.second);
  }
  const std::string canonical_query = canonical.str();

  // We sign as POST and send as POST — the whole canonical query lives in the
  // request body (form-urlencoded), with Signature appended. A GET with the
  // same signature would be valid too, but AWS SDK's URI normaliser can re-
  // encode query params and silently break the signature, so POST is safer.
  const std::string string_to_sign = std::string("POST&") + PopUrlEncode("/") + "&" + PopUrlEncode(canonical_query);

  const std::string signing_key = std::string(request.callerAccessKeySecret.c_str()) + "&";
  const std::vector<uint8_t> digest = HmacSha1(signing_key, string_to_sign);

  Aws::Utils::ByteBuffer digest_buf(digest.data(), digest.size());
  const Aws::String signature = Aws::Utils::HashingUtils::Base64Encode(digest_buf);

  const std::string body_str = canonical_query + "&Signature=" + PopUrlEncode(std::string(signature.c_str()));

  std::shared_ptr<Aws::Http::HttpRequest> httpRequest(Aws::Http::CreateHttpRequest(
      m_endpoint, Aws::Http::HttpMethod::HTTP_POST, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));
  httpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());

  auto body = Aws::MakeShared<Aws::StringStream>(kLogTag);
  *body << body_str;
  httpRequest->AddContentBody(body);
  Aws::StringStream content_length;
  content_length << body_str.size();
  httpRequest->SetContentLength(content_length.str());
  httpRequest->SetContentType("application/x-www-form-urlencoded");

  const Aws::String payload = GetResourceWithAWSWebServiceResult(httpRequest).GetPayload();
  if (payload.empty()) {
    AWS_LOGSTREAM_WARN(kLogTag, "Empty AssumeRole response from " << m_endpoint);
    return result;
  }

  // Response shape:
  // <AssumeRoleResponse>
  //   <Credentials>
  //     <AccessKeyId>...</AccessKeyId>
  //     <AccessKeySecret>...</AccessKeySecret>
  //     <SecurityToken>...</SecurityToken>
  //     <Expiration>2023-01-01T12:00:00Z</Expiration>
  //   </Credentials>
  // </AssumeRoleResponse>
  const auto doc = Aws::Utils::Xml::XmlDocument::CreateFromXmlString(payload);
  auto root = doc.GetRootElement();
  auto resultNode = root;
  if (!root.IsNull() && root.GetName() != "AssumeRoleResponse") {
    resultNode = root.FirstChild("AssumeRoleResponse");
  }
  if (resultNode.IsNull()) {
    AWS_LOGSTREAM_WARN(kLogTag, "Unexpected AssumeRole response root: " << payload);
    return result;
  }
  auto credentials = resultNode.FirstChild("Credentials");
  if (credentials.IsNull()) {
    AWS_LOGSTREAM_WARN(kLogTag, "Missing <Credentials> in AssumeRole response: " << payload);
    return result;
  }

  auto ak_node = credentials.FirstChild("AccessKeyId");
  if (!ak_node.IsNull())
    result.creds.SetAWSAccessKeyId(ak_node.GetText());
  auto sk_node = credentials.FirstChild("AccessKeySecret");
  if (!sk_node.IsNull())
    result.creds.SetAWSSecretKey(sk_node.GetText());
  auto tok_node = credentials.FirstChild("SecurityToken");
  if (!tok_node.IsNull())
    result.creds.SetSessionToken(tok_node.GetText());
  auto exp_node = credentials.FirstChild("Expiration");
  if (!exp_node.IsNull()) {
    result.creds.SetExpiration(Aws::Utils::DateTime(Aws::Utils::StringUtils::Trim(exp_node.GetText().c_str()).c_str(),
                                                    Aws::Utils::DateFormat::ISO_8601));
  }
  return result;
}

}  // namespace milvus_storage
