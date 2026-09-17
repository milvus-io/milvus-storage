// Copyright 2025 Zilliz
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
#include <string_view>
#include <unordered_map>

#include <arrow/result.h>
#include <arrow/status.h>

#include "rust/cxx.h"

namespace milvus_storage {

inline constexpr std::string_view kFFIErrorCodeMarker = "__LOON_FFI_ERRCODE__=";

/// Converts a raw Rust/FFI error message to an Arrow status, using an embedded
/// FFI error-code marker when present and falling back to IOError otherwise.
arrow::Status MakeBridgeErrorStatus(std::string_view message);

/// Converts a raw Rust/FFI error message to an Arrow status and prepends the
/// supplied operation context while preserving the mapped status detail.
arrow::Status MakeBridgeErrorStatus(std::string_view context, std::string_view message);

/// Adds operation context to an Arrow status. Statuses carrying an FFI
/// error-code marker are decoded; other statuses retain their code and detail.
arrow::Status MakeBridgeErrorStatus(std::string_view context, const arrow::Status& status);

using RustErrorMapper = arrow::Status (*)(std::string_view);

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(Fn&& fn) {
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeBridgeErrorStatus(e.what());
  }
}

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(RustErrorMapper error_mapper, Fn&& fn) {
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& e) {
    return error_mapper(e.what());
  }
}

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(std::string_view context, Fn&& fn) {
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeBridgeErrorStatus(context, e.what());
  }
}

template <typename Fn>
arrow::Status CatchRustStatus(Fn&& fn) {
  try {
    fn();
    return arrow::Status::OK();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeBridgeErrorStatus(e.what());
  }
}

template <typename Fn>
arrow::Status CatchRustStatus(RustErrorMapper error_mapper, Fn&& fn) {
  try {
    fn();
    return arrow::Status::OK();
  } catch (const rust::cxxbridge1::Error& e) {
    return error_mapper(e.what());
  }
}

template <typename Fn>
arrow::Status CatchRustStatus(std::string_view context, Fn&& fn) {
  try {
    fn();
    return arrow::Status::OK();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeBridgeErrorStatus(context, e.what());
  }
}

/// Releases an Arrow C Data Interface object on scope exit unless ownership
/// has been transferred and Disarm() has been called. T must provide a
/// `release(T*)` callback field, such as ArrowSchema, ArrowArray, or
/// ArrowArrayStream.
template <typename T>
class ArrowCDataReleaseGuard {
  public:
  explicit ArrowCDataReleaseGuard(T* data) : data_(data) {}

  ~ArrowCDataReleaseGuard() {
    if (armed_ && data_ != nullptr && data_->release != nullptr) {
      data_->release(data_);
    }
  }

  void Disarm() { armed_ = false; }

  private:
  T* data_;
  bool armed_ = true;
};

inline void ConvertStorageOptions(const std::unordered_map<std::string, std::string>& storage_options,
                                  rust::Vec<rust::String>& keys,
                                  rust::Vec<rust::String>& values) {
  for (const auto& [k, v] : storage_options) {
    keys.push_back(rust::String(k.data(), k.length()));
    values.push_back(rust::String(v.data(), v.length()));
  }
}

}  // namespace milvus_storage
