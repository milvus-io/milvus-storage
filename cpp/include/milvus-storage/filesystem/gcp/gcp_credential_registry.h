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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <aws/core/http/URI.h>

#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

namespace milvus_storage {

// Key used to map an outgoing GCP request to a credential provider.
//
// endpoint_host is the bare host (scheme stripped, no trailing slash).
// Examples after normalization:
//   - "storage.googleapis.com"
//   - "custom-gcs.internal:8443"
//
// bucket_name is the GCS bucket the request targets.
struct GcpBucketKey {
  std::string endpoint_host;
  std::string bucket_name;

  bool operator==(const GcpBucketKey& other) const noexcept {
    return endpoint_host == other.endpoint_host && bucket_name == other.bucket_name;
  }
};

struct GcpBucketKeyHash {
  size_t operator()(const GcpBucketKey& k) const noexcept {
    return std::hash<std::string>{}(k.endpoint_host) ^ (std::hash<std::string>{}(k.bucket_name) << 1);
  }
};

// Normalize a filesystem config's `address` field into an endpoint host for
// use as a GcpBucketKey. Strips scheme (http://, https://) and trailing '/'.
std::string NormalizeGcpEndpointHost(const std::string& address);

// Process-wide registry mapping (endpoint_host, bucket) → credential provider.
//
// The GCP HTTP client factory and delegator are installed once globally (AWS
// SDK constraint via InitializeS3 + call_once). They are stateless and look
// up the per-request provider from this registry by inspecting each request's
// URI. Identities are registered per GcpFileSystemProducer::Make() call, so
// any number of GCP identities can coexist in one process as long as each
// (endpoint, bucket) pair maps to exactly one identity.
class GcpCredentialRegistry {
  public:
  static GcpCredentialRegistry& Instance();

  // Register or replace the provider for a (endpoint, bucket) pair.
  //
  // Registration is idempotent: a second Register with the same key silently
  // replaces the prior provider. In practice this only happens when the same
  // bucket is configured via both `fs.*` and an `extfs.<ns>.*` slot, or when
  // the same config Make()s twice — in both cases the identity is identical.
  // Same bucket + different identity is not a supported configuration.
  void Register(GcpBucketKey key, std::shared_ptr<GcpCredentialProvider> provider);

  // Look up the provider for an outgoing request URI. Tries both path-style
  // (host + first path segment) and virtual-host-style (first subdomain +
  // remaining host) interpretations. Returns nullptr if no match.
  std::shared_ptr<GcpCredentialProvider> Lookup(const Aws::Http::URI& uri) const;

  private:
  GcpCredentialRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<GcpBucketKey, std::shared_ptr<GcpCredentialProvider>, GcpBucketKeyHash> providers_;
};

}  // namespace milvus_storage
