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

#include "milvus-storage/filesystem/s3/provider/TencentCloudSTSClient.h"

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/log.h"

#include <mutex>
#include <new>
#include <sstream>
#include <string_view>

#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/client/AWSError.h>

namespace milvus_storage {
using Aws::Http::HttpClient;
using Aws::Http::HttpRequest;
using Aws::Http::HttpResponseCode;

namespace {

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

arrow::Status ClassifyTencentStsError(std::string_view code, std::string_view message) {
  const auto context = fmt::format("Tencent Cloud STS failed: code={} message={}", code, message);
  if (code == "RequestLimitExceeded" || StartsWith(code, "RequestLimitExceeded.") ||
      code == "InvalidParameter.OverLimit" || code == "Throttling" || StartsWith(code, "Throttling.")) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientThrottling, context);
  }
  if (code == "InternalError" || StartsWith(code, "InternalError.") || code == "ServiceUnavailable" ||
      StartsWith(code, "ServiceUnavailable.") || code == "ResourceUnavailable") {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientService, context);
  }
  if (code == "UnauthorizedOperation" || StartsWith(code, "UnauthorizedOperation.") || code == "AccessDenied" ||
      StartsWith(code, "AccessDenied.") || code == "AuthFailure" || StartsWith(code, "AuthFailure.")) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageAccessDenied, context);
  }
  if (code.empty()) {
    return MakeCredentialResponseError("Tencent Cloud STS returned Error without a Code");
  }
  // InvalidParameter, InvalidRole, InvalidIdentityToken and all other
  // well-formed refusals describe deployment/input configuration. They are
  // not repaired by replaying the object request.
  return MakeCredentialConfigError(context);
}

}  // namespace

static const char STS_RESOURCE_CLIENT_LOG_TAG[] = "TencentCloudSTSResourceClient";  // [tencent cloud]

TencentCloudSTSCredentialsClient::TencentCloudSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, STS_RESOURCE_CLIENT_LOG_TAG) {
  m_rawHttpClient = Aws::Http::CreateHttpClient(clientConfiguration);

  // [tencent cloud]
  m_endpoint = "https://sts.tencentcloudapi.com";

  LOG_STORAGE_INFO_ << fmt::format("[{}] Creating STS ResourceClient with endpoint: {}", STS_RESOURCE_CLIENT_LOG_TAG,
                                   m_endpoint);
}

TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
TencentCloudSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
  STSAssumeRoleWithWebIdentityResult result;
  try {
    // Calculate query string
    Aws::StringStream ss;
    // curl -X POST "https://sts.tencentcloudapi.com"
    // -d "{\"ProviderId\": $ProviderId, \"WebIdentityToken\":
    // $WebIdentityToken,\"RoleArn\":$RoleArn,\"RoleSessionName\":$RoleSessionName,\"DurationSeconds\":7200}" -H
    // "Authorization: SKIP" -H "Content-Type: application/json; charset=utf-8" -H "Host: sts.tencentcloudapi.com" -H
    // "X-TC-Action: AssumeRoleWithWebIdentity" -H "X-TC-Timestamp: $timestamp" -H "X-TC-Version: 2018-08-13" -H
    // "X-TC-Region: $region" -H "X-TC-Token: $token"

    ss << R"({"ProviderId": ")" << request.providerId << R"(", "WebIdentityToken": ")" << request.webIdentityToken
       << R"(", "RoleArn": ")" << request.roleArn << R"(", "RoleSessionName": ")" << request.roleSessionName << R"("})";

    std::shared_ptr<Aws::Http::HttpRequest> httpRequest(Aws::Http::CreateHttpRequest(
        m_endpoint, Aws::Http::HttpMethod::HTTP_POST, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));

    httpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());
    httpRequest->SetHeaderValue("Authorization", "SKIP");
    httpRequest->SetHeaderValue("Host", "sts.tencentcloudapi.com");
    httpRequest->SetHeaderValue("X-TC-Action", "AssumeRoleWithWebIdentity");
    httpRequest->SetHeaderValue("X-TC-Timestamp", std::to_string(Aws::Utils::DateTime::Now().Seconds()));
    httpRequest->SetHeaderValue("X-TC-Version", "2018-08-13");
    httpRequest->SetHeaderValue("X-TC-Region", request.region);
    httpRequest->SetHeaderValue("X-TC-Token", "");

    std::shared_ptr<Aws::IOStream> body = Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
    *body << ss.str();

    httpRequest->AddContentBody(body);
    body->seekg(0, body->end);
    auto streamSize = body->tellg();
    body->seekg(0, body->beg);
    Aws::StringStream contentLength;
    contentLength << streamSize;
    httpRequest->SetContentLength(contentLength.str());
    //    httpRequest->SetContentType("application/x-www-form-urlencoded");
    httpRequest->SetContentType("application/json; charset=utf-8");

    if (m_rawHttpClient == nullptr) {
      result.status = ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode::REQUEST_NOT_MADE,
                                                    "Tencent Cloud STS AssumeRoleWithWebIdentity has no HTTP client");
      return result;
    }
    const auto response = MakeRequestWithCredentialRetry(*m_rawHttpClient, httpRequest);
    if (response == nullptr) {
      result.status =
          ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode::NO_RESPONSE,
                                        "Tencent Cloud STS AssumeRoleWithWebIdentity received no HTTP response");
      return result;
    }
    const auto response_code = response->GetResponseCode();
    if (response_code != Aws::Http::HttpResponseCode::OK) {
      result.status = ClassifyCredentialHttpFailure(
          response_code, fmt::format("Tencent Cloud STS AssumeRoleWithWebIdentity failed (http_status={})",
                                     static_cast<int>(response_code)));
      LOG_STORAGE_WARNING_ << fmt::format("[{}] {}", STS_RESOURCE_CLIENT_LOG_TAG, result.status.message());
      return result;
    }
    Aws::IStreamBufIterator eos;
    Aws::String credentialsStr(Aws::IStreamBufIterator(response->GetResponseBody()), eos);
    if (credentialsStr.empty()) {
      result.status = MakeCredentialResponseError("Tencent Cloud STS returned an empty body");
      LOG_STORAGE_WARNING_ << fmt::format("[{}] {}", STS_RESOURCE_CLIENT_LOG_TAG, result.status.message());
      return result;
    }

    Aws::Utils::Json::JsonValue jsonValue(credentialsStr);
    auto json = jsonValue.View();
    auto rootNode = json.GetObject("Response");
    if (rootNode.IsNull()) {
      LOG_STORAGE_WARNING_ << fmt::format("[{}] Get Response from credential result failed",
                                          STS_RESOURCE_CLIENT_LOG_TAG);
      // A 200 whose body we cannot read does not establish either a transport
      // failure or an access decision, so leave it conservatively unclassified.
      result.status = MakeCredentialResponseError("Tencent Cloud STS response carried no Response object");
      return result;
    }

    if (rootNode.KeyExists("Error")) {
      const auto error_node = rootNode.GetObject("Error");
      result.status = ClassifyTencentStsError(error_node.GetString("Code"), error_node.GetString("Message"));
      LOG_STORAGE_WARNING_ << fmt::format("[{}] {}", STS_RESOURCE_CLIENT_LOG_TAG, result.status.message());
      return result;
    }

    auto credentialsNode = rootNode.GetObject("Credentials");
    if (credentialsNode.IsNull()) {
      LOG_STORAGE_WARNING_ << fmt::format("[{}] Get Credentials from Response failed", STS_RESOURCE_CLIENT_LOG_TAG);
      result.status = MakeCredentialResponseError("Tencent Cloud STS response carried no Credentials");
      return result;
    }
    result.creds.SetAWSAccessKeyId(credentialsNode.GetString("TmpSecretId"));
    result.creds.SetAWSSecretKey(credentialsNode.GetString("TmpSecretKey"));
    result.creds.SetSessionToken(credentialsNode.GetString("Token"));
    result.creds.SetExpiration(
        Aws::Utils::DateTime(Aws::Utils::StringUtils::Trim(rootNode.GetString("Expiration").c_str()).c_str(),
                             Aws::Utils::DateFormat::ISO_8601));

    result.status = ValidateTemporaryCredentials(result.creds, "Tencent Cloud STS AssumeRoleWithWebIdentity");
    if (!result.status.ok()) {
      result.creds = {};
    }

    return result;
  } catch (const std::bad_alloc&) {
    result.status = MakeCredentialOutOfMemoryError("Tencent Cloud STS AssumeRoleWithWebIdentity ran out of memory");
  } catch (const std::exception& e) {
    result.status = MakeCredentialExceptionError("Tencent Cloud STS AssumeRoleWithWebIdentity raised", e);
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Exception during credential retrieval: {}", STS_RESOURCE_CLIENT_LOG_TAG,
                                      e.what());
  } catch (...) {
    result.status = MakeCredentialUnknownExceptionError("Tencent Cloud STS AssumeRoleWithWebIdentity raised");
    LOG_STORAGE_ERROR_ << fmt::format("[{}] Unknown exception during credential retrieval",
                                      STS_RESOURCE_CLIENT_LOG_TAG);
  }
  return result;
}

}  // namespace milvus_storage
