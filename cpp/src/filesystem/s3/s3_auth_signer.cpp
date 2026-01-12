// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/s3/s3_auth_signer.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include <openssl/hmac.h>
#include <openssl/sha.h>

namespace milvus_storage::auth_signer::googv4 {

// SHA256 hash (returns hex string)
static std::string SHA256Hex(const std::string& data) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(data.c_str()), data.length(), hash);

  std::stringstream ss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
  }
  return ss.str();
}

// HMAC-SHA256
static std::vector<uint8_t> HMACSHA256(const std::vector<uint8_t>& key, const std::string& data) {
  std::vector<uint8_t> result(EVP_MAX_MD_SIZE);
  unsigned int len = 0;

  HMAC(EVP_sha256(), key.data(), key.size(), reinterpret_cast<const unsigned char*>(data.c_str()), data.length(),
       result.data(), &len);

  result.resize(len);
  return result;
}

// Convert byte array to hex string
static std::string BytesToHex(const std::vector<uint8_t>& bytes) {
  std::stringstream ss;
  for (uint8_t byte : bytes) {
    ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  }
  return ss.str();
}

// URL encode
static std::string URLEncode(const std::string& value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << std::hex;

  for (char c : value) {
    // Keep alphanumeric and other safe characters
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      escaped << c;
      continue;
    }
    // Any other characters are percent-encoded
    escaped << std::uppercase;
    escaped << '%' << std::setw(2) << int((unsigned char)c);
    escaped << std::nouppercase;
  }

  return escaped.str();
}

// Build canonicalized query string
static std::string BuildCanonicalQueryString(const Aws::Http::URI& uri) {
  auto query_params = uri.GetQueryStringParameters();
  if (query_params.empty()) {
    return "";
  }

  std::vector<std::pair<std::string, std::string>> sorted_params;
  for (const auto& [key, val] : query_params) {
    sorted_params.emplace_back(key, val);
  }
  std::sort(sorted_params.begin(), sorted_params.end());

  std::vector<std::string> parts;
  for (const auto& [key, val] : sorted_params) {
    parts.push_back(URLEncode(key) + "=" + URLEncode(val));
  }

  std::string result;
  for (size_t i = 0; i < parts.size(); i++) {
    if (i > 0)
      result += "&";
    result += parts[i];
  }
  return result;
}

// Build canonical headers
static std::string BuildCanonicalHeaders(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  std::unordered_map<std::string, std::string> headers_map;

  for (const auto& [key, val] : request->GetHeaders()) {
    std::string lower_key = key;
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
    headers_map[lower_key] = val;
  }

  std::stringstream ss;
  for (const auto& [key, val] : headers_map) {
    ss << key << ":" << val << "\n";
  }
  return ss.str();
}

// Get signed headers list
static std::string GetSignedHeaders(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  std::vector<std::string> headers;
  for (const auto& [key, val] : request->GetHeaders()) {
    std::string lower_key = key;
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
    headers.push_back(lower_key);
  }
  std::sort(headers.begin(), headers.end());

  std::string result;
  for (size_t i = 0; i < headers.size(); i++) {
    if (i > 0)
      result += ";";
    result += headers[i];
  }
  return result;
}

// Calculate payload hash
static std::string CalculatePayloadHash(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  const static std::string EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  auto& body_stream = request->GetContentBody();
  if (!body_stream || !body_stream->good()) {
    return EMPTY_SHA256;
  }

  // Read body content
  std::stringstream buffer;
  buffer << body_stream->rdbuf();
  std::string body_content = buffer.str();

  // Reset stream
  body_stream->clear();
  if (!body_stream->seekg(0, std::ios::beg)) {
    // Failed to reset stream position, return empty to indicate failure
    return "";
  }

  if (body_content.empty()) {
    return EMPTY_SHA256;
  }

  return SHA256Hex(body_content);
}

// Build canonical request
static std::string BuildCanonicalRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  // HTTP Method
  std::string method;
  switch (request->GetMethod()) {
    case Aws::Http::HttpMethod::HTTP_GET:
      method = "GET";
      break;
    case Aws::Http::HttpMethod::HTTP_POST:
      method = "POST";
      break;
    case Aws::Http::HttpMethod::HTTP_PUT:
      method = "PUT";
      break;
    case Aws::Http::HttpMethod::HTTP_DELETE:
      method = "DELETE";
      break;
    case Aws::Http::HttpMethod::HTTP_HEAD:
      method = "HEAD";
      break;
    default:
      method = "GET";
      break;
  }

  // Canonical URI
  std::string canonical_uri = request->GetUri().GetPath();
  if (canonical_uri.empty()) {
    canonical_uri = "/";
  }

  // Canonical Query String
  std::string canonical_query = BuildCanonicalQueryString(request->GetUri());

  // Canonical Headers
  std::string canonical_headers = BuildCanonicalHeaders(request);

  // Signed Headers
  std::string signed_headers = GetSignedHeaders(request);

  // Payload Hash
  std::string payload_hash = request->GetHeaderValue("x-goog-content-sha256").c_str();

  return method + "\n" + canonical_uri + "\n" + canonical_query + "\n" + canonical_headers + "\n" + signed_headers +
         "\n" + payload_hash;
}

// Calculate GOOG4 signature
static std::string CalculateGoog4Signature(const std::string& secret_key,
                                           const std::string& date_stamp,
                                           const std::string& string_to_sign) {
  // Key derivation (similar to AWS Signature V4)
  std::string key_data = "GOOG4" + secret_key;
  std::vector<uint8_t> k_date = HMACSHA256(std::vector<uint8_t>(key_data.begin(), key_data.end()), date_stamp);
  std::vector<uint8_t> k_region = HMACSHA256(k_date, "auto");
  std::vector<uint8_t> k_service = HMACSHA256(k_region, "storage");
  std::vector<uint8_t> k_signing = HMACSHA256(k_service, "goog4_request");

  // Calculate signature
  std::vector<uint8_t> signature = HMACSHA256(k_signing, string_to_sign);
  return BytesToHex(signature);
}

// Sign request using GOOG4-HMAC-SHA256
bool SignRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                 const std::string& access_key,
                 const std::string& secret_key) {
  // Get current time
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now;
#ifdef _WIN32
  gmtime_s(&tm_now, &time_t_now);
#else
  gmtime_r(&time_t_now, &tm_now);
#endif

  std::stringstream date_stamp_ss, time_stamp_ss;
  date_stamp_ss << std::put_time(&tm_now, "%Y%m%d");
  time_stamp_ss << std::put_time(&tm_now, "%Y%m%dT%H%M%SZ");
  std::string date_stamp = date_stamp_ss.str();
  std::string time_stamp = time_stamp_ss.str();

  // Calculate and set payload hash
  std::string payload_hash = CalculatePayloadHash(request);
  // Check if payload hash calculation failed (empty string indicates failure)
  if (payload_hash.empty()) {
    return false;
  }
  request->SetHeaderValue("x-goog-content-sha256", payload_hash);

  // Add required headers
  request->SetHeaderValue("x-goog-date", time_stamp);
  if (!request->HasHeader("host")) {
    request->SetHeaderValue("host", request->GetUri().GetAuthority());
  }

  // Step 1: Create canonical request
  std::string canonical_request = BuildCanonicalRequest(request);

  // Step 2: Create string to sign
  std::string credential_scope = date_stamp + "/auto/storage/goog4_request";
  std::string hashed_canonical_request = SHA256Hex(canonical_request);
  std::string string_to_sign =
      "GOOG4-HMAC-SHA256\n" + time_stamp + "\n" + credential_scope + "\n" + hashed_canonical_request;

  // Step 3: Calculate signature
  std::string signature = CalculateGoog4Signature(secret_key, date_stamp, string_to_sign);

  // Step 4: Add Authorization header
  std::string signed_headers = GetSignedHeaders(request);
  std::string authorization = "GOOG4-HMAC-SHA256 Credential=" + access_key + "/" + credential_scope +
                              ", SignedHeaders=" + signed_headers + ", Signature=" + signature;
  request->SetHeaderValue("Authorization", authorization);

  return true;
}

}  // namespace signer::goog4
