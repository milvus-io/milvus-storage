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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <aws/core/http/URI.h>

#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"

namespace milvus_storage {

// Canonical endpoint identity used for registry lookups. The port is always
// the effective port, including 80/443 when it was omitted from the address.
struct GcpEndpointKey {
  Aws::Http::Scheme scheme;
  uint16_t port;
  std::string host;

  bool operator==(const GcpEndpointKey& other) const noexcept {
    return scheme == other.scheme && port == other.port && host == other.host;
  }
};

// Key used to map an outgoing GCP request to a credential provider.
struct GcpBucketKey {
  GcpEndpointKey endpoint;
  std::string bucket_name;

  bool operator==(const GcpBucketKey& other) const noexcept {
    return endpoint == other.endpoint && bucket_name == other.bucket_name;
  }
};

struct GcpBucketKeyHash {
  size_t operator()(const GcpBucketKey& k) const noexcept {
    size_t hash = std::hash<int>{}(static_cast<int>(k.endpoint.scheme));
    hash ^= std::hash<uint16_t>{}(k.endpoint.port) << 1;
    hash ^= std::hash<std::string>{}(k.endpoint.host) << 2;
    hash ^= std::hash<std::string>{}(k.bucket_name) << 3;
    return hash;
  }
};

// Normalize a filesystem config's address using an explicit scheme when
// present, or use_ssl otherwise.
GcpEndpointKey NormalizeGcpEndpoint(const std::string& address, bool use_ssl);

// Process-wide registry mapping (endpoint, bucket) → credential provider.
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
  // (request endpoint + first path segment) and virtual-host-style (first
  // subdomain + remaining endpoint host) interpretations. Returns nullptr if
  // no match.
  std::shared_ptr<GcpCredentialProvider> Lookup(const Aws::Http::URI& uri) const;

  private:
  GcpCredentialRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<GcpBucketKey, std::shared_ptr<GcpCredentialProvider>, GcpBucketKeyHash> providers_;
};

}  // namespace milvus_storage
