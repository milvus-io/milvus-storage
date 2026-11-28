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

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace milvus_storage {

// A source identity, not a credential. Blob and DFS service names identify the
// same account only when the complete endpoint suffix also matches. Paths use
// decoded components, so authorization never depends on a raw string prefix.
struct AzureSourceResource {
  std::string account;
  std::string endpoint_suffix;
  std::string container;
  std::vector<std::string> path;

  // Accepts explicit HTTPS Blob/DFS URLs and secure ABFSS URLs. Relative paths,
  // credentials/query strings and ambiguous path encodings are rejected. Milvus
  // and Lance shorthand URIs must first be resolved using the selected storage
  // configuration; this parser never invents an account or endpoint for them.
  static std::optional<AzureSourceResource> Parse(std::string_view uri, std::string_view expected_account);

  [[nodiscard]] std::string CanonicalUri() const;
  [[nodiscard]] bool Contains(const AzureSourceResource& resource) const;
  bool operator==(const AzureSourceResource&) const = default;
};

// Decode the resource addressed by an unsigned SDK read/list request. Listing
// addresses the container URL but selects a directory through query parameters.
// This is request classification only: callers must still match the returned
// resource against context roots and check context state before attaching SAS.
struct AzureSourceHttpRequest {
  AzureSourceResource resource;
  bool is_list = false;

  // Unsupported operations, writes, existing credentials, duplicate query keys
  // and Blob prefixes without a directory boundary are rejected.
  static std::optional<AzureSourceHttpRequest> Parse(std::string_view method,
                                                     std::string_view unsigned_uri,
                                                     std::string_view expected_account);
};

// Resolves storage-library aliases only after selecting the filesystem's exact
// account, endpoint suffix and container. This binding supplies missing URI
// fields; it does not grant access. Check the result against the context roots
// before I/O, and retain the result before legacy URI conversions erase fields.
class AzureSourceStorageBinding {
  public:
  static std::optional<AzureSourceStorageBinding> Create(std::string_view account,
                                                         std::string_view endpoint_suffix,
                                                         std::string_view container);

  // Explicit HTTPS/ABFSS, Milvus azure://endpoint/container/key and
  // az://endpoint/container/key, or Lance az://container/key. Explicit fields
  // must match this binding exactly; a different authority is never replaced.
  [[nodiscard]] std::optional<AzureSourceResource> ResolveUri(std::string_view uri) const;

  // FileSystemProxy receives decoded, container-relative keys. Encode each
  // component exactly once instead of interpreting percent escapes as a URI.
  // This is a separate API so untrusted relative URIs cannot silently acquire
  // an ambient account/container.
  [[nodiscard]] std::optional<AzureSourceResource> ResolvePath(std::string_view path) const;

  private:
  explicit AzureSourceStorageBinding(AzureSourceResource container) : container_(std::move(container)) {}
  AzureSourceResource container_;
};

// Choose the most specific explicitly authorized root. Independent roots do not
// become authority for their common parent. The returned index also partitions
// directory-scoped credential caches within one runtime context.
std::optional<size_t> FindSourceRoot(const std::vector<AzureSourceResource>& roots,
                                     const AzureSourceResource& resource);

}  // namespace milvus_storage
