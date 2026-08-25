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

#include "milvus-storage/common/extend_status.h"

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

arrow::Result<std::shared_ptr<GcpCredentialRegistration>> GcpCredentialRegistry::Register(
    GcpBucketKey key, std::shared_ptr<GcpCredentialRegistration> registration) {
  if (registration == nullptr || registration->provider_ == nullptr) {
    return arrow::Status::Invalid("GCP credential registration and provider must not be null");
  }

  key.endpoint.host = ToLower(std::move(key.endpoint.host));
  std::lock_guard<std::mutex> lock(mu_);

  // Weak entries do not retain credentials. Opportunistically remove expired
  // keys so repeated creation/destruction of filesystems cannot grow the map.
  for (auto it = registrations_.begin(); it != registrations_.end();) {
    if (it->second.expired()) {
      it = registrations_.erase(it);
    } else {
      ++it;
    }
  }

  if (auto it = registrations_.find(key); it != registrations_.end()) {
    auto existing = it->second.lock();
    if (existing != nullptr) {
      if (existing->identity_ != registration->identity_) {
        return MakeExtendErrorMsg(ExtendStatusCode::StorageConfigInvalid,
                                  "GCP credential identity conflict for endpoint=", key.endpoint.host,
                                  ", port=", std::to_string(key.endpoint.port), ", bucket=", key.bucket_name);
      }
      return existing;
    }
  }

  registrations_[std::move(key)] = registration;
  return registration;
}

std::shared_ptr<GcpCredentialProvider> GcpCredentialRegistry::Lookup(const Aws::Http::URI& uri) const {
  auto endpoint = EndpointFromUri(uri);
  std::string path = uri.GetPath().c_str();

  auto find = [this](const GcpBucketKey& key) -> std::shared_ptr<GcpCredentialProvider> {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = registrations_.find(key);
    if (it == registrations_.end()) {
      return nullptr;
    }
    auto registration = it->second.lock();
    return registration == nullptr ? nullptr : registration->provider_;
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

}  // namespace milvus_storage
