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

#include <arrow/result.h>
#include <arrow/util/future.h>

#include "rust/cxx.h"
#include "rust-bridge/talon_bridge.h"

namespace milvus_storage::talon {

namespace internal {
// Convert the typed Rust callback result without parsing its diagnostic text.
arrow::Result<int64_t> ToArrowIoResult(const char* operation,
                                       int32_t error_code,
                                       uint64_t value,
                                       const char* error_msg);
}  // namespace internal

/// Optional object metadata supplied by storage when opening a Talon reader.
/// An empty version is valid for the version-independent GetRange path.
struct TalonObjectStat {
  uint64_t size;
  std::string version;
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
  static arrow::Result<std::shared_ptr<TalonClient>> Make(const std::string& coordinator, uint32_t block_size);

  arrow::Result<TalonObjectReader> OpenObject(const std::string& cloud_provider,
                                              const std::string& bucket,
                                              const std::string& key,
                                              const std::optional<TalonObjectStat>& initial_stat) const;

  private:
  explicit TalonClient(rust::Box<ffi::TalonClient> impl) : impl_(std::move(impl)) {}

  rust::Box<ffi::TalonClient> impl_;
};

}  // namespace milvus_storage::talon
