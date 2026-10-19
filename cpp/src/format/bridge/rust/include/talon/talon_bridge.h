// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/result.h>
#include <arrow/util/future.h>

#include "milvus-storage/properties.h"
#include "rust/cxx.h"
#include "rust-bridge/talon/talon_bridge.h"

namespace milvus_storage::talon {

/// Optional object metadata supplied by storage when opening a Talon reader.
/// The version is the origin ETag without surrounding quotes. An empty version
/// skips ETag matching for immutable objects whose size is already known.
struct TalonObjectStat {
  uint64_t size;
  std::string version;
};

struct TalonLoadRequest {
  std::string uri;
  uint64_t size;
  std::string version;
};
struct TalonLoadResult {
  uint64_t size;
  uint64_t blocks;
};
struct TalonLoadFailure {
  uint64_t index;
  bool uncertain;
  std::string error;
};
struct TalonBatchLoadResult {
  arrow::Status status;
  std::vector<TalonLoadResult> results;
  std::vector<TalonLoadFailure> failed_files;
};

/// Owns a Rust Talon object reader and exposes only Arrow-native results to C++ callers.
class TalonObjectReader final {
  public:
  TalonObjectReader(TalonObjectReader&&) noexcept = default;
  TalonObjectReader& operator=(TalonObjectReader&&) noexcept = default;

  TalonObjectReader(const TalonObjectReader&) = delete;
  TalonObjectReader& operator=(const TalonObjectReader&) = delete;

  /// Returns the cached or caller-provided object size, or -1 when unknown.
  int64_t KnownSize() const;

  /// Resolves object metadata on Talon's shared Tokio runtime.
  arrow::Future<int64_t> StatAsync() const;

  /// Reads into caller-owned memory on Talon's shared Tokio runtime.
  /// `dst` must remain valid until the returned future completes.
  arrow::Future<int64_t> ReadAtAsync(uint64_t offset, uint64_t length, uint8_t* dst) const;

  private:
  friend class TalonClient;

  explicit TalonObjectReader(rust::Box<ffi::TalonObjectReader> impl) : impl_(std::move(impl)) {}

  rust::Box<ffi::TalonObjectReader> impl_;
};

/// Owns a Rust Talon client and converts every fallible FFI operation to Arrow errors.
class TalonClient final {
  public:
  static arrow::Result<std::shared_ptr<TalonClient>> Make(const std::string& coordinator,
                                                          uint32_t block_size,
                                                          uint32_t max_idle_per_addr);

  static arrow::Result<std::shared_ptr<TalonClient>> Make(const api::Properties& properties);
  /// Raw operations use Talon directly, independently of filesystem routing mode.
  /// The caller retains dst until completion; present size-zero metadata is a known empty file.
  arrow::Future<int64_t> ReadAsync(const std::string& uri,
                                   uint64_t offset,
                                   uint64_t length,
                                   uint8_t* dst,
                                   const std::optional<TalonObjectStat>& initial_stat = std::nullopt) const;
  arrow::Result<int64_t> Read(const std::string& uri,
                              uint64_t offset,
                              uint64_t length,
                              uint8_t* dst,
                              const std::optional<TalonObjectStat>& initial_stat = std::nullopt) const;
  arrow::Future<TalonObjectStat> StatAsync(const std::string& uri) const;
  arrow::Result<TalonObjectStat> Stat(const std::string& uri) const;
  /// Uses caller-provided size/version without HEAD; an empty version skips version matching.
  arrow::Future<TalonLoadResult> LoadAsync(const TalonLoadRequest& request) const;
  arrow::Result<TalonLoadResult> Load(const TalonLoadRequest& request) const;
  /// Native batch errors stay in the value's status, preserving ordered failed_files.
  /// An outer future error denotes a bridge failure. Partial failure has no results array.
  arrow::Future<TalonBatchLoadResult> BatchLoadAsync(const std::vector<TalonLoadRequest>& requests) const;
  arrow::Result<TalonBatchLoadResult> BatchLoad(const std::vector<TalonLoadRequest>& requests) const;

  arrow::Result<TalonObjectReader> OpenObject(const std::string& cloud_provider,
                                              const std::string& bucket,
                                              const std::string& key,
                                              const std::optional<TalonObjectStat>& initial_stat) const;

  private:
  explicit TalonClient(rust::Box<ffi::TalonClient> impl) : impl_(std::move(impl)) {}

  rust::Box<ffi::TalonClient> impl_;
};

}  // namespace milvus_storage::talon
