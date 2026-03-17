#include "milvus-storage/filesystem/s3/provider/HuaweiCloudSTSClient.h"
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/DateTime.h>

namespace milvus_storage {
using Aws::Http::HttpClient;
using Aws::Http::HttpRequest;
using Aws::Http::HttpResponseCode;

static const char STS_RESOURCE_CLIENT_LOG_TAG[] = "MilvusStorage-HuaweiCloudSTSResourceClient";

HuaweiCloudSTSCredentialsClient::HuaweiCloudSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, STS_RESOURCE_CLIENT_LOG_TAG) {
  SetErrorMarshaller(Aws::MakeUnique<Aws::Client::XmlErrorMarshaller>(STS_RESOURCE_CLIENT_LOG_TAG));
  m_token_endpoint = "https://iam.{region}.myhuaweicloud.com/v3.0/OS-AUTH/id-token/tokens";
  m_httpClient = Aws::Http::CreateHttpClient(clientConfiguration);
  AWS_LOGSTREAM_INFO(STS_RESOURCE_CLIENT_LOG_TAG, "Creating STS ResourceClient with endpoint: " << m_token_endpoint);
}

HuaweiCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
HuaweiCloudSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
  Aws::StringStream ss;
  ss << R"({
        "auth": {
          "id_token": {
            "id": ")"
     << request.webIdentityToken << R"("
          },
          "scope": {
            "project": {
              "id": ")"
     << request.roleArn << R"("
            }
          }
        }
      })";

  Aws::String endpoint = m_token_endpoint;
  size_t pos = endpoint.find("{region}");
  if (pos != Aws::String::npos) {
    endpoint.replace(pos, 8, request.region);
  }
  std::shared_ptr<Aws::Http::HttpRequest> httpRequest(Aws::Http::CreateHttpRequest(
      endpoint, Aws::Http::HttpMethod::HTTP_POST, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));
  httpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());
  httpRequest->SetHeaderValue("X-Idp-Id", request.providerId);
  httpRequest->SetHeaderValue("Content-Type", "application/json");

  std::shared_ptr<Aws::IOStream> body = Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
  *body << ss.str();
  body->seekg(0, body->end);
  auto streamSize = body->tellg();
  body->seekg(0, body->beg);
  httpRequest->SetContentLength(std::to_string(streamSize));
  httpRequest->AddContentBody(body);
  httpRequest->SetContentType("application/json; charset=utf-8");

  STSAssumeRoleWithWebIdentityResult result;

  try {
    // Stage 1: Get IAM token via OIDC id-token endpoint
    // Note: GetResourceWithAWSWebServiceResult() only treats HTTP 200 as success,
    // so HuaweiCloud's 201 CREATED will produce spurious WARN/ERROR logs from the
    // base class (AWSErrorMarshaller, "Can not retrieve resource"). However, we
    // retain this call to preserve the AWS SDK's built-in retry/backoff mechanism.
    // The response code and headers are still correctly returned for our checks.
    AWS_LOGSTREAM_INFO(STS_RESOURCE_CLIENT_LOG_TAG,
                       "Stage 1: Requesting IAM token from OIDC endpoint, region=" << request.region);
    auto awsResult = GetResourceWithAWSWebServiceResult(httpRequest);
    auto responseCode = awsResult.GetResponseCode();
    if (responseCode != Aws::Http::HttpResponseCode::OK && responseCode != Aws::Http::HttpResponseCode::CREATED) {
      AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                         "Failed to get credentials token from Huawei Cloud "
                         "STS, response code: "
                             << static_cast<int>(responseCode));
      return result;
    }

    auto responseHeaders = awsResult.GetHeaderValueCollection();
    auto subjectTokenIter = responseHeaders.find("x-subject-token");
    if (subjectTokenIter == responseHeaders.end()) {
      AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "No x-subject-token in huawei cloud sts response headers");
      return result;
    }

    // Stage 2: Exchange IAM token for temporary AK/SK credentials
    AWS_LOGSTREAM_INFO(STS_RESOURCE_CLIENT_LOG_TAG,
                       "Stage 1 succeeded. Stage 2: Exchanging IAM token for temporary AK/SK (duration_seconds=7200)");
    const Aws::String subjectToken = subjectTokenIter->second;
    auto stsResult = callHuaweiCloudSTS(subjectToken, request);
    if (!stsResult.success) {
      AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                         "Failed to get credentials from Huawei Cloud STS: " << stsResult.errorMessage);
      return result;
    }

    result.creds = stsResult.credentials;
    result.success = true;
    auto akId = result.creds.GetAWSAccessKeyId();
    Aws::String akPrefix = akId.length() > 4 ? akId.substr(0, 4) + "***" : akId;
    AWS_LOGSTREAM_INFO(
        STS_RESOURCE_CLIENT_LOG_TAG,
        "Stage 2 succeeded. ak_prefix=" << akPrefix << ", expires_in_ms="
                                        << (result.creds.GetExpiration() - Aws::Utils::DateTime::Now()).count());
  } catch (const std::exception& e) {
    result.success = false;
    AWS_LOGSTREAM_ERROR(STS_RESOURCE_CLIENT_LOG_TAG,
                        "Exception during Huawei Cloud STS credential retrieval: " << e.what());
  } catch (...) {
    result.success = false;
    AWS_LOGSTREAM_ERROR(STS_RESOURCE_CLIENT_LOG_TAG, "Unknown exception during Huawei Cloud STS credential retrieval");
  }
  return result;
}

HuaweiCloudSTSCredentialsClient::STSCallResult HuaweiCloudSTSCredentialsClient::callHuaweiCloudSTS(
    const Aws::String& userToken, const STSAssumeRoleWithWebIdentityRequest& request) {
  auto respFactory = []() -> Aws::IOStream* { return Aws::New<Aws::StringStream>("STS_RESPONSE"); };

  Aws::String stsEndpoint = "https://iam." + request.region + ".myhuaweicloud.com/v3.0/OS-CREDENTIAL/securitytokens";
  auto req = Aws::Http::CreateHttpRequest(stsEndpoint, Aws::Http::HttpMethod::HTTP_POST, respFactory);
  req->SetHeaderValue("Content-Type", "application/json;charset=utf8");
  req->SetHeaderValue("Accept", "application/json");
  req->SetHeaderValue("X-Auth-Token", userToken);

  auto body = Aws::MakeShared<Aws::StringStream>("STS_REQUEST");
  *body << R"({
            "auth": {
            "identity": {
                "methods": ["token"],
                "token":{
                    "duration_seconds": 7200
                }
            }
            }
        })";
  body->seekg(0, body->end);
  auto streamSize = body->tellg();
  body->seekg(0, body->beg);

  req->SetContentLength(std::to_string(streamSize));
  req->AddContentBody(body);

  auto resp = m_httpClient->MakeRequest(req);
  STSCallResult result;
  result.success = false;
  if (!resp) {
    result.errorMessage = "Null response from Huawei Cloud STS HTTP request";
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG, "Security token request returned null response");
    return result;
  }
  auto httpResponseCode = resp->GetResponseCode();
  if (httpResponseCode != Aws::Http::HttpResponseCode::OK && httpResponseCode != Aws::Http::HttpResponseCode::CREATED) {
    result.errorMessage = "Huawei Cloud STS security token request failed with HTTP code: " +
                          std::to_string(static_cast<int>(httpResponseCode));
    AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                       "Security token request failed, HTTP code=" << static_cast<int>(httpResponseCode));
    return result;
  }
  std::ostringstream oss;
  oss << resp->GetResponseBody().rdbuf();
  Aws::String credentialsStr = oss.str();
  if (credentialsStr.empty()) {
    result.errorMessage = "Get an empty credential from Huawei Cloud STS";
    return result;
  }
  Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
  auto json = jsonValue.View();
  auto rootNode = json.GetObject("credential");
  if (rootNode.IsNull()) {
    result.errorMessage = "Get credential from STS result failed";
    return result;
  }
  result.credentials.SetAWSAccessKeyId(rootNode.GetString("access"));
  result.credentials.SetAWSSecretKey(rootNode.GetString("secret"));
  result.credentials.SetSessionToken(rootNode.GetString("securitytoken"));

  auto expiresAt = rootNode.GetString("expires_at");
  if (expiresAt.empty()) {
    result.errorMessage = "STS response missing 'expires_at' field, rejecting credentials";
    return result;
  }
  auto parsedExpiration =
      Aws::Utils::DateTime(Aws::Utils::StringUtils::Trim(expiresAt.c_str()).c_str(), Aws::Utils::DateFormat::ISO_8601);
  if (!parsedExpiration.WasParseSuccessful()) {
    result.errorMessage = "STS response 'expires_at' field has invalid format: " + std::string(expiresAt.c_str());
    return result;
  }
  result.credentials.SetExpiration(parsedExpiration);
  result.success = true;
  return result;
}

}  // namespace milvus_storage
