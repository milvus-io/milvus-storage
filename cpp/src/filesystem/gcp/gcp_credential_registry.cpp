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

}  // namespace

std::string NormalizeGcpEndpointHost(const std::string& address) {
  std::string host = address;

  // Strip scheme.
  auto scheme_pos = host.find("://");
  if (scheme_pos != std::string::npos) {
    host.erase(0, scheme_pos + 3);
  }

  // Strip trailing slash(es).
  while (!host.empty() && host.back() == '/') {
    host.pop_back();
  }

  return ToLower(host);
}

GcpCredentialRegistry& GcpCredentialRegistry::Instance() {
  static GcpCredentialRegistry instance;
  return instance;
}

void GcpCredentialRegistry::Register(GcpBucketKey key, std::shared_ptr<GcpCredentialProvider> provider) {
  key.endpoint_host = ToLower(std::move(key.endpoint_host));
  std::lock_guard<std::mutex> lock(mu_);
  providers_[std::move(key)] = std::move(provider);
}

std::shared_ptr<GcpCredentialProvider> GcpCredentialRegistry::Lookup(const Aws::Http::URI& uri) const {
  std::string host = ToLower(uri.GetAuthority().c_str());
  std::string path = uri.GetPath().c_str();

  auto find = [this](const GcpBucketKey& key) -> std::shared_ptr<GcpCredentialProvider> {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = providers_.find(key);
    return it == providers_.end() ? nullptr : it->second;
  };

  // Path-style: endpoint = host, bucket = first path segment.
  auto seg = FirstPathSegment(path);
  if (!seg.empty()) {
    if (auto p = find({host, seg})) {
      return p;
    }
  }

  // Virtual-host-style: bucket = first subdomain, endpoint = rest of host.
  auto dot = host.find('.');
  if (dot != std::string::npos && dot > 0) {
    auto subdomain = host.substr(0, dot);
    auto rest = host.substr(dot + 1);
    if (auto p = find({rest, subdomain})) {
      return p;
    }
  }

  return nullptr;
}

}  // namespace milvus_storage
