// Copyright 2025 Zilliz
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

#include "milvus-storage/filesystem/azure/azure_cross_tenant_credential.h"

#include <cctype>
#include <cstdio>
#include <stdexcept>

#include <azure/core/datetime.hpp>
#include <azure/core/http/curl_transport.hpp>
#include <azure/core/http/http.hpp>
#include <azure/core/http/raw_response.hpp>
#include <azure/core/io/body_stream.hpp>
#include <azure/core/url.hpp>

#include <folly/json/json.h>

#include "milvus-storage/common/log.h"

namespace milvus_storage::fs {

namespace {

// IMDS endpoint and audience are fixed by the Azure platform; not
// configurable per VM. The audience `api://AzureADTokenExchange` is the
// well-known audience that AAD's `oauth2/v2.0/token` accepts as a
// `client_assertion` for federated identity credentials.
constexpr char kImdsTokenUrl[] = "http://169.254.169.254/metadata/identity/oauth2/token";
constexpr char kImdsApiVersion[] = "2018-02-01";
constexpr char kMiAudience[] = "api://AzureADTokenExchange";

// AAD authority host. Public Azure cloud only; sovereign clouds use
// different hosts but cross-tenant FIC across sovereign boundaries is not a
// supported scenario in this MVP.
constexpr char kAadAuthority[] = "https://login.microsoftonline.com";

// Scope for the customer-tenant Bearer. `.default` asks for whatever
// permissions the customer's App Registration was granted on
// `https://storage.azure.com/`.
constexpr char kStorageScope[] = "https://storage.azure.com/.default";

constexpr char kClientAssertionType[] = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";

/// Percent-encode a value for an `application/x-www-form-urlencoded` body.
/// Keep alnum + `- _ . ~` literal; everything else percent-escaped. Equivalent
/// to RFC 3986 unreserved.
std::string UrlEncode(const std::string& s) {
  std::string out;
  out.reserve(s.size() * 11 / 10 + 2);
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back(static_cast<char>(c));
    } else {
      char buf[4];
      std::snprintf(buf, sizeof(buf), "%%%02X", c);
      out.append(buf, 3);
    }
  }
  return out;
}

/// Read full response body. Use `ExtractBodyStream` rather than `GetBody`
/// because `CurlTransport` may return responses with the body still
/// streaming and `GetBody` would be empty.
std::string ReadResponseBodyAsString(Azure::Core::Http::RawResponse& response, Azure::Core::Context const& context) {
  auto stream = response.ExtractBodyStream();
  std::string body;
  if (stream) {
    auto bytes = stream->ReadToEnd(context);
    body.assign(bytes.begin(), bytes.end());
  }
  // Some transport configurations populate `m_body` directly instead of
  // setting a stream; fall back to that.
  if (body.empty()) {
    auto const& raw = response.GetBody();
    body.assign(raw.begin(), raw.end());
  }
  return body;
}

}  // namespace

AzureCrossTenantCredential::AzureCrossTenantCredential(std::string tenant_id, std::string client_id)
    : Azure::Core::Credentials::TokenCredential("AzureCrossTenantCredential"),
      tenant_id_(std::move(tenant_id)),
      client_id_(std::move(client_id)) {}

Azure::Core::Credentials::AccessToken AzureCrossTenantCredential::GetToken(
    Azure::Core::Credentials::TokenRequestContext const& /*tokenRequestContext*/,
    Azure::Core::Context const& context) const {
  // Fast path: cached token still has comfortable lifetime.
  {
    std::lock_guard<std::mutex> g(cache_mu_);
    if (cached_) {
      // Both sides as Azure::DateTime so the time_point template params line
      // up; DateTime inherits from system_clock::time_point but operator-
      // doesn't pick up the unrelated raw time_point on the RHS.
      auto remaining = cached_->ExpiresOn - Azure::DateTime(std::chrono::system_clock::now());
      if (remaining > kRefreshOffset) {
        return *cached_;
      }
    }
  }

  // Slow path: fetch fresh. We deliberately don't hold cache_mu_ across the
  // HTTP calls — concurrent misses may both fetch, but only one bearer
  // wins the cache slot. That is bounded waste; holding the mutex would
  // serialize unrelated FS operations behind a 200-500ms AAD round-trip.
  Azure::Core::Http::CurlTransport transport;
  std::string mi_assertion;
  std::pair<std::string, Azure::DateTime> exchanged;
  try {
    mi_assertion = FetchManagedIdentityAssertion(transport, context);
    exchanged = ExchangeForStorageBearer(transport, mi_assertion, context);
  } catch (const std::exception& e) {
    throw Azure::Core::Credentials::AuthenticationException(std::string("AzureCrossTenantCredential failed: ") +
                                                            e.what());
  }

  Azure::Core::Credentials::AccessToken fresh;
  fresh.Token = std::move(exchanged.first);
  fresh.ExpiresOn = exchanged.second;

  {
    std::lock_guard<std::mutex> g(cache_mu_);
    cached_ = fresh;
  }
  return fresh;
}

std::string AzureCrossTenantCredential::FetchManagedIdentityAssertion(Azure::Core::Http::HttpTransport& transport,
                                                                      Azure::Core::Context const& context) const {
  Azure::Core::Url url(kImdsTokenUrl);
  url.AppendQueryParameter("api-version", kImdsApiVersion);
  url.AppendQueryParameter("resource", kMiAudience);

  Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Get, url);
  request.SetHeader("Metadata", "true");

  auto response = transport.Send(request, context);
  if (!response) {
    throw std::runtime_error("IMDS token request returned no response");
  }
  auto status = response->GetStatusCode();
  if (status != Azure::Core::Http::HttpStatusCode::Ok) {
    auto body = ReadResponseBodyAsString(*response, context);
    throw std::runtime_error("IMDS token request failed status=" + std::to_string(static_cast<int>(status)) +
                             " body=" + body);
  }

  auto body = ReadResponseBodyAsString(*response, context);
  folly::dynamic parsed;
  try {
    parsed = folly::parseJson(body);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("IMDS response was not valid JSON: ") + e.what());
  }
  auto access_token = parsed.getDefault("access_token", "").asString();
  if (access_token.empty()) {
    throw std::runtime_error("IMDS response missing access_token");
  }
  return access_token;
}

std::pair<std::string, Azure::DateTime> AzureCrossTenantCredential::ExchangeForStorageBearer(
    Azure::Core::Http::HttpTransport& transport,
    std::string const& mi_assertion,
    Azure::Core::Context const& context) const {
  // POST to https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
  // with form-urlencoded body. AAD requires:
  //   client_id={customer_app}
  //   scope=https://storage.azure.com/.default
  //   grant_type=client_credentials
  //   client_assertion={MI assertion JWT}
  //   client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer
  std::string token_url = std::string(kAadAuthority) + "/" + tenant_id_ + "/oauth2/v2.0/token";
  Azure::Core::Url url(token_url);

  std::string body_str;
  body_str.reserve(2048);
  body_str.append("client_id=").append(UrlEncode(client_id_));
  body_str.append("&scope=").append(UrlEncode(kStorageScope));
  body_str.append("&grant_type=client_credentials");
  body_str.append("&client_assertion_type=").append(UrlEncode(kClientAssertionType));
  body_str.append("&client_assertion=").append(UrlEncode(mi_assertion));

  std::vector<uint8_t> body_bytes(body_str.begin(), body_str.end());
  Azure::Core::IO::MemoryBodyStream body_stream(body_bytes.data(), body_bytes.size());

  Azure::Core::Http::Request request(Azure::Core::Http::HttpMethod::Post, url, &body_stream);
  request.SetHeader("Content-Type", "application/x-www-form-urlencoded");
  request.SetHeader("Content-Length", std::to_string(body_bytes.size()));

  auto response = transport.Send(request, context);
  if (!response) {
    throw std::runtime_error("AAD token exchange returned no response");
  }
  auto status = response->GetStatusCode();
  if (status != Azure::Core::Http::HttpStatusCode::Ok) {
    auto err_body = ReadResponseBodyAsString(*response, context);
    throw std::runtime_error(
        "AAD token exchange failed (the customer's App Registration likely has no "
        "Federated Identity Credential trusting our MI, or the audience/issuer/subject "
        "in the FIC don't match) tenant=" +
        tenant_id_ + " client_id=" + client_id_ + " status=" + std::to_string(static_cast<int>(status)) +
        " body=" + err_body);
  }

  auto resp_body = ReadResponseBodyAsString(*response, context);
  folly::dynamic parsed;
  try {
    parsed = folly::parseJson(resp_body);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("AAD response was not valid JSON: ") + e.what());
  }
  auto access_token = parsed.getDefault("access_token", "").asString();
  if (access_token.empty()) {
    throw std::runtime_error("AAD response missing access_token");
  }
  // expires_in is seconds-from-now. AAD always sends an int here for
  // client_credentials grants.
  int64_t expires_in = parsed.getDefault("expires_in", 3600).asInt();
  auto expires_at = std::chrono::system_clock::now() + std::chrono::seconds(expires_in);
  return {std::move(access_token), Azure::DateTime(expires_at)};
}

}  // namespace milvus_storage::fs
