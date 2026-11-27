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

#pragma once

#include <string>

#include <aws/core/Aws.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <curl/curl.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>

namespace milvus_storage {

inline bool OpenSSLFipsEnabled() {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
  return EVP_default_properties_is_fips_enabled(OSSL_LIB_CTX_get0_global_default()) == 1;
#else
  return false;
#endif
}

// Convert tls_min_version string to CURLOPT_SSLVERSION value.
// Returns CURL_SSLVERSION_DEFAULT (0) if the version string is empty or unrecognized.
inline long TlsVersionToCurlOpt(const std::string& tls_min_version) {
  // An explicit CURLOPT_SSLVERSION replaces the floor from OpenSSL's configuration.
  if ((tls_min_version == "1.0" || tls_min_version == "1.1") && OpenSSLFipsEnabled()) {
    return CURL_SSLVERSION_TLSv1_2;
  }
  if (tls_min_version == "1.0") {
    return CURL_SSLVERSION_TLSv1_0;
  }
  if (tls_min_version == "1.1") {
    return CURL_SSLVERSION_TLSv1_1;
  }
  if (tls_min_version == "1.2") {
    return CURL_SSLVERSION_TLSv1_2;
  }
  if (tls_min_version == "1.3") {
    return CURL_SSLVERSION_TLSv1_3;
  }

  return CURL_SSLVERSION_DEFAULT;
}

// CurlHttpClient subclass that enforces a minimum TLS version via CURLOPT_SSLVERSION.
class TlsCurlHttpClient : public Aws::Http::CurlHttpClient {
  public:
  TlsCurlHttpClient(const Aws::Client::ClientConfiguration& config, const std::string& tls_min_version)
      : CurlHttpClient(config), tls_ssl_version_(TlsVersionToCurlOpt(tls_min_version)) {}

  protected:
  void OverrideOptionsOnConnectionHandle(CURL* handle) const override {
    if (tls_ssl_version_ != CURL_SSLVERSION_DEFAULT) {
      curl_easy_setopt(handle, CURLOPT_SSLVERSION, tls_ssl_version_);
    }
  }

  private:
  long tls_ssl_version_;
};

}  // namespace milvus_storage
