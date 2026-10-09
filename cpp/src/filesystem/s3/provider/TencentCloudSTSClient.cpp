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

#include "milvus-storage/common/log.h"

#include <stdexcept>
#include <vector>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

namespace milvus_storage {
namespace {
constexpr const char* kLogTag = "TencentCloudSTSResourceClient";
constexpr const char* kHost = "sts.tencentcloudapi.com";
constexpr const char* kContentType = "application/json; charset=utf-8";

Aws::String Sha256Hex(const Aws::String& value) {
  return Aws::Utils::HashingUtils::HexEncode(Aws::Utils::HashingUtils::CalculateSHA256(value));
}

std::vector<unsigned char> HmacSha256(const unsigned char* key, size_t key_size, const Aws::String& value) {
  std::vector<unsigned char> result(EVP_MAX_MD_SIZE);
  unsigned int size = 0;
  if (!HMAC(EVP_sha256(), key, static_cast<int>(key_size), reinterpret_cast<const unsigned char*>(value.data()),
            value.size(), result.data(), &size)) {
    throw std::runtime_error("Tencent STS request signing failed");
  }
  result.resize(size);
  return result;
}

Aws::Auth::AWSCredentials ParseCredentials(const Aws::String& body) {
  Aws::Utils::Json::JsonValue value(body);
  if (!value.WasParseSuccessful() || !value.View().IsObject())
    return {};
  auto top = value.View().GetAllObjects();
  if (!top.count("Response") || !top.at("Response").IsObject())
    return {};
  auto response = top.at("Response").GetAllObjects();
  if (response.count("Error") || !response.count("Credentials") || !response.at("Credentials").IsObject())
    return {};
  auto fields = response.at("Credentials").GetAllObjects();
  for (const char* name : {"TmpSecretId", "TmpSecretKey", "Token"}) {
    if (!fields.count(name) || !fields.at(name).IsString() || fields.at(name).AsString().empty())
      return {};
  }
  Aws::Utils::DateTime expiration;
  if (response.count("ExpiredTime") && response.at("ExpiredTime").IsIntegerType()) {
    expiration = Aws::Utils::DateTime(static_cast<double>(response.at("ExpiredTime").AsInt64()));
  } else if (response.count("Expiration") && response.at("Expiration").IsString()) {
    expiration = Aws::Utils::DateTime(response.at("Expiration").AsString(), Aws::Utils::DateFormat::ISO_8601);
  } else {
    return {};
  }
  if (!expiration.WasParseSuccessful() || expiration <= Aws::Utils::DateTime::Now())
    return {};
  Aws::Auth::AWSCredentials credentials(fields.at("TmpSecretId").AsString(), fields.at("TmpSecretKey").AsString(),
                                        fields.at("Token").AsString());
  credentials.SetExpiration(expiration);
  return credentials;
}
}  // namespace

TencentCloudSTSCredentialsClient::TencentCloudSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, kLogTag), m_endpoint("https://sts.tencentcloudapi.com") {
  SetErrorMarshaller(Aws::MakeUnique<Aws::Client::JsonErrorMarshaller>(kLogTag));
}

TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
TencentCloudSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
  Aws::Utils::Json::JsonValue payload;
  payload.WithString("ProviderId", request.providerId)
      .WithString("WebIdentityToken", request.webIdentityToken)
      .WithString("RoleArn", request.roleArn)
      .WithString("RoleSessionName", request.roleSessionName);
  return {SendRequest("AssumeRoleWithWebIdentity", request.region, payload)};
}

Aws::Auth::AWSCredentials TencentCloudSTSCredentialsClient::GetAssumeRoleCredentials(
    const STSAssumeRoleRequest& request) {
  if (request.callerCredentials.IsExpiredOrEmpty() || request.callerCredentials.GetSessionToken().empty() ||
      request.roleArn.empty() || request.roleSessionName.empty() || request.region.empty())
    return {};
  Aws::Utils::Json::JsonValue payload;
  payload.WithString("RoleArn", request.roleArn).WithString("RoleSessionName", request.roleSessionName);
  if (!request.externalId.empty())
    payload.WithString("ExternalId", request.externalId);
  return SendRequest("AssumeRole", request.region, payload, &request.callerCredentials);
}

Aws::Auth::AWSCredentials TencentCloudSTSCredentialsClient::SendRequest(const Aws::String& action,
                                                                        const Aws::String& region,
                                                                        const Aws::Utils::Json::JsonValue& payload,
                                                                        const Aws::Auth::AWSCredentials* caller) {
  const auto body = payload.View().WriteCompact();
  const auto now = Aws::Utils::DateTime::Now();
  const auto timestamp = std::to_string(now.Seconds());
  auto request = Aws::Http::CreateHttpRequest(m_endpoint, Aws::Http::HttpMethod::HTTP_POST,
                                              Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  request->SetContentType(kContentType);
  request->SetHeaderValue("Host", kHost);
  request->SetHeaderValue("X-TC-Action", action);
  request->SetHeaderValue("X-TC-Timestamp", timestamp);
  request->SetHeaderValue("X-TC-Version", "2018-08-13");
  request->SetHeaderValue("X-TC-Region", region);
  if (caller) {
    // Sign exactly the body and content type sent below. TC3 uses a UTC date.
    const auto date = now.ToGmtString("%Y-%m-%d");
    const Aws::String scope = date + "/sts/tc3_request";
    const Aws::String canonical = "POST\n/\n\ncontent-type:" + Aws::String(kContentType) + "\nhost:" + kHost +
                                  "\n\ncontent-type;host\n" + Sha256Hex(body);
    const Aws::String string_to_sign = "TC3-HMAC-SHA256\n" + timestamp + "\n" + scope + "\n" + Sha256Hex(canonical);
    const auto secret = "TC3" + caller->GetAWSSecretKey();
    auto signing = HmacSha256(reinterpret_cast<const unsigned char*>(secret.data()), secret.size(), date);
    signing = HmacSha256(signing.data(), signing.size(), "sts");
    signing = HmacSha256(signing.data(), signing.size(), "tc3_request");
    const auto signature = HmacSha256(signing.data(), signing.size(), string_to_sign);
    request->SetHeaderValue("Authorization", "TC3-HMAC-SHA256 Credential=" + caller->GetAWSAccessKeyId() + "/" + scope +
                                                 ", SignedHeaders=content-type;host, Signature=" +
                                                 Aws::Utils::HashingUtils::HexEncode(
                                                     Aws::Utils::ByteBuffer(signature.data(), signature.size())));
    request->SetHeaderValue("X-TC-Token", caller->GetSessionToken());
  } else {
    request->SetHeaderValue("Authorization", "SKIP");
  }
  auto stream = Aws::MakeShared<Aws::StringStream>(kLogTag);
  *stream << body;
  request->AddContentBody(stream);
  request->SetContentLength(std::to_string(body.size()));
  auto credentials = ParseCredentials(GetResourceWithAWSWebServiceResult(request).GetPayload());
  if (credentials.IsEmpty())
    LOG_STORAGE_WARNING_ << "Tencent STS " << action << " returned no valid credentials";
  return credentials;
}
}  // namespace milvus_storage
