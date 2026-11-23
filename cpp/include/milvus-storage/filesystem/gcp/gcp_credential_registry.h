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
#include <unordered_set>

#include <arrow/status.h>
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

struct GcpEndpointKeyHash {
  size_t operator()(const GcpEndpointKey& k) const noexcept {
    size_t hash = std::hash<int>{}(static_cast<int>(k.scheme));
    hash ^= std::hash<uint16_t>{}(k.port) << 1;
    hash ^= std::hash<std::string>{}(k.host) << 2;
    return hash;
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
// The GCP HTTP client factory is installed once globally (AWS SDK constraint
// via InitializeS3 + call_once). The SDK clients that
// GcpFileSystemProducer::Make() builds are bound to that filesystem's
// provider, and their CRT meta requests and native transport requests carry it
// through S3Options::request_authenticator. Requests whose HttpClient the GCP
// factory did not build (an HTTP client factory installed outside
// milvus-storage), and requests of filesystems built without Make(), look up
// their provider here by request URI. The registry also records the trusted
// endpoints that decide where Bearer tokens may go.
class GcpCredentialRegistry {
  public:
  static GcpCredentialRegistry& Instance();

  // Register the provider for a (endpoint, bucket) pair.
  //
  // One (endpoint, bucket) maps to one registered identity per process. A
  // second Register with the same key keeps the existing provider when the
  // identity (GcpCredentialProvider::Identity) is the same, and fails with
  // Invalid when it differs. A trusted (operator) registration may replace an
  // untrusted one; an untrusted registration never replaces a trusted one.
  // GcpFileSystemProducer::Make() still builds a filesystem whose registration
  // fails when its SDK clients are bound, because every request then carries
  // its own identity.
  //
  // trusted_endpoint marks key.endpoint as configured by the operator (the
  // primary filesystem's own address), so Bearer tokens may be sent to it even
  // when it is not an HTTPS Google API host. Endpoints of external filesystems
  // come from user specs and must not be marked.
  //
  // On success, a non-null registered receives the provider the entry holds:
  // the existing one when the identity is the same, so its cached token is
  // reused, and provider otherwise.
  arrow::Status Register(GcpBucketKey key,
                         std::shared_ptr<GcpCredentialProvider> provider,
                         bool trusted_endpoint = false,
                         std::shared_ptr<GcpCredentialProvider>* registered = nullptr);

  // Look up the provider for an outgoing request URI. Tries both path-style
  // (request endpoint + first path segment) and virtual-host-style (first
  // subdomain + remaining endpoint host) interpretations. Returns nullptr if
  // no match.
  std::shared_ptr<GcpCredentialProvider> Lookup(const Aws::Http::URI& uri) const;

  // Whether a request to this URI may carry a Bearer token: it is HTTPS to a
  // Google API host (*.googleapis.com), or its endpoint (path-style) or parent
  // endpoint (virtual-host-style) was registered as trusted.
  bool MayCarryBearerToken(const Aws::Http::URI& uri) const;

  private:
  GcpCredentialRegistry() = default;

  struct Entry {
    std::shared_ptr<GcpCredentialProvider> provider;
    bool trusted;
  };

  mutable std::mutex mu_;
  std::unordered_map<GcpBucketKey, Entry, GcpBucketKeyHash> providers_;
  std::unordered_set<GcpEndpointKey, GcpEndpointKeyHash> trusted_endpoints_;
};

}  // namespace milvus_storage
