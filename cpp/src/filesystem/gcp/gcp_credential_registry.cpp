// Copyright 2026 Zilliz
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

#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace milvus_storage {

namespace {

std::string FirstPathSegment(const std::string& path) {
  // path from Aws::Http::URI::GetPath starts with '/'; skip it.
  size_t start = (!path.empty() && path.front() == '/') ? 1 : 0;
  size_t end = path.find('/', start);
  if (end == std::string::npos) {
    return path.substr(start);
  }
  return path.substr(start, end - start);
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

GcpEndpointKey EndpointFromUri(const Aws::Http::URI& uri) {
  return {uri.GetScheme(), uri.GetPort(), ToLower(uri.GetAuthority().c_str())};
}

bool IsGoogleApisHost(const std::string& host) {
  constexpr std::string_view kGoogleApisSuffix = ".googleapis.com";
  return host.size() > kGoogleApisSuffix.size() && host.ends_with(kGoogleApisSuffix);
}

}  // namespace

GcpEndpointKey NormalizeGcpEndpoint(const std::string& address, bool use_ssl) {
  std::string endpoint = address;
  if (!endpoint.starts_with("http://") && !endpoint.starts_with("https://")) {
    endpoint.insert(0, use_ssl ? "https://" : "http://");
  }
  return EndpointFromUri(Aws::Http::URI(endpoint.c_str()));
}

GcpCredentialRegistry& GcpCredentialRegistry::Instance() {
  static GcpCredentialRegistry instance;
  return instance;
}

arrow::Status GcpCredentialRegistry::Register(GcpBucketKey key,
                                              std::shared_ptr<GcpCredentialProvider> provider,
                                              bool trusted_endpoint) {
  key.endpoint.host = ToLower(std::move(key.endpoint.host));
  std::lock_guard<std::mutex> lock(mu_);
  auto it = providers_.find(key);
  if (it != providers_.end()) {
    auto& entry = it->second;
    if (entry.provider->Identity() != provider->Identity()) {
      if (entry.trusted && !trusted_endpoint) {
        return arrow::Status::Invalid("GCP bucket '", key.bucket_name, "' at ", key.endpoint.host,
                                      " is this process's own storage and authenticates with a different "
                                      "identity; an external filesystem cannot use it");
      }
      if (!trusted_endpoint) {
        return arrow::Status::Invalid("GCP bucket '", key.bucket_name, "' at ", key.endpoint.host,
                                      " is already registered with a different identity in this process; "
                                      "one bucket maps to one identity");
      }
      // Operator configuration replaces a user-supplied identity.
      entry.provider = std::move(provider);
    }
    entry.trusted = entry.trusted || trusted_endpoint;
  } else {
    providers_.emplace(key, Entry{std::move(provider), trusted_endpoint});
  }
  if (trusted_endpoint) {
    trusted_endpoints_.insert(key.endpoint);
  }
  return arrow::Status::OK();
}

std::shared_ptr<GcpCredentialProvider> GcpCredentialRegistry::Lookup(const Aws::Http::URI& uri) const {
  auto endpoint = EndpointFromUri(uri);
  std::string path = uri.GetPath().c_str();

  auto find = [this](const GcpBucketKey& key) -> std::shared_ptr<GcpCredentialProvider> {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = providers_.find(key);
    return it == providers_.end() ? nullptr : it->second.provider;
  };

  // Path-style: endpoint = request endpoint, bucket = first path segment.
  auto seg = FirstPathSegment(path);
  if (!seg.empty()) {
    if (auto p = find({endpoint, seg})) {
      return p;
    }
  }

  // Virtual-host-style: bucket = first subdomain, endpoint host = rest of host.
  auto dot = endpoint.host.find('.');
  if (dot != std::string::npos && dot > 0) {
    auto subdomain = endpoint.host.substr(0, dot);
    endpoint.host.erase(0, dot + 1);
    if (auto p = find({endpoint, subdomain})) {
      return p;
    }
  }

  return nullptr;
}

bool GcpCredentialRegistry::MayCarryBearerToken(const Aws::Http::URI& uri) const {
  auto endpoint = EndpointFromUri(uri);
  // A Bearer token is a reusable credential, so Google hosts get it over TLS
  // only. Operator-configured endpoints decide their own scheme.
  if (endpoint.scheme == Aws::Http::Scheme::HTTPS && IsGoogleApisHost(endpoint.host)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (trusted_endpoints_.contains(endpoint)) {
    return true;
  }
  // Virtual-host-style: the first label is the bucket.
  auto dot = endpoint.host.find('.');
  if (dot == std::string::npos || dot == 0) {
    return false;
  }
  endpoint.host.erase(0, dot + 1);
  return trusted_endpoints_.contains(endpoint);
}

}  // namespace milvus_storage
