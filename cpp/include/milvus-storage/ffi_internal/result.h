// Copyright 2023 Zilliz
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

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_c.h"

#include <arrow/status.h>
#include <arrow/util/io_util.h>

#include <cerrno>
#include <exception>
#include <optional>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <cassert>
#include <iostream>

#define RETURN_SUCCESS()                         \
  do {                                           \
    return LoonFFIResult{LOON_SUCCESS, nullptr}; \
  } while (0)

#define RETURN_EXCEPTION(...)                                                             \
  do {                                                                                    \
    return CreateFFIResult(LOON_GOT_EXCEPTION, __func__, " Got exception: ", ##__VA_ARGS__); \
  } while (0)

#define RETURN_ERROR(code, ...)                    \
  do {                                             \
    return CreateFFIResult((code), ##__VA_ARGS__); \
  } while (0)

#define RETURN_ARROW_ERROR(status, fallback, ...)                               \
  do {                                                                          \
    auto ffi_status__ = (status);                                               \
    auto ffi_err_code__ = FFIErrorCodeFromExtendStatus(ffi_status__, fallback); \
    RETURN_ERROR(ffi_err_code__, ##__VA_ARGS__);                                \
  } while (0)

#define RETURN_ARROW_ERROR_IF(status, fallback, ...)                              \
  do {                                                                            \
    auto ffi_status__ = (status);                                                 \
    if (!ffi_status__.ok()) {                                                     \
      auto ffi_err_code__ = FFIErrorCodeFromExtendStatus(ffi_status__, fallback); \
      RETURN_ERROR(ffi_err_code__, ##__VA_ARGS__);                                \
    }                                                                             \
  } while (0)

#define RETURN_UNREACHABLE() RETURN_ERROR(LOON_UNREACHABLE_ERROR);

std::string error_to_string(int code);

namespace milvus_storage::ffi_internal {

inline int FFIErrorCodeFromExtendStatusCode(milvus_storage::ExtendStatusCode code, int fallback) {
  if (milvus_storage::ExtendStatusCodeFromInt(static_cast<int>(code)).has_value()) {
    return static_cast<int>(code);
  }
  return fallback;
}

inline std::optional<milvus_storage::ExtendStatusCode> ExtendStatusCodeFromFFIErrorCode(int err_code) {
  return milvus_storage::ExtendStatusCodeFromInt(err_code);
}

}  // namespace milvus_storage::ffi_internal

inline int FFIErrorCodeFromExtendStatus(const arrow::Status& status, int fallback = LOON_ARROW_ERROR) {
  auto detail = milvus_storage::ExtendStatusDetail::UnwrapStatus(status);
  if (detail) {
    return milvus_storage::ffi_internal::FFIErrorCodeFromExtendStatusCode(detail->code(), fallback);
  }

  // Allocation failures are not storage failures: LOON_MEMORY_ERROR (2) maps to
  // segcore's MemAllocateFailed (2034), so milvus's memory-pressure handling can
  // see "not enough memory" across the FFI boundary. Keep it away from
  // call-site fallbacks such as LOON_SOURCE_INVALID, which describes source
  // availability, but do not expose a separate retryable OOM category.
  if (status.IsOutOfMemory()) {
    return LOON_MEMORY_ERROR;
  }

  if (arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    return LOON_FILE_NOT_FOUND;
  }

  // NotImplemented has exactly one meaning everywhere in this library: the
  // capability is absent. It needs no per-call-site interpretation, and every
  // entry point that hand-rolled it (`if (status.IsNotImplemented()) fallback =
  // LOON_NOT_SUPPORT`) was reimplementing this line. Unlike Invalid -- which
  // covers caller input, internal invariants and unparsable persisted bytes
  // alike -- it can be mapped centrally without guessing.
  if (status.IsNotImplemented()) {
    return LOON_NOT_SUPPORT;
  }
  return fallback;
}

inline std::optional<milvus_storage::ExtendStatusCode> ExtendStatusCodeFromFFIErrorCode(int err_code) {
  return milvus_storage::ffi_internal::ExtendStatusCodeFromFFIErrorCode(err_code);
}

/// \brief Classify terminal failures while resolving or reading an external
/// source supplied to an external-table entry point.
///
/// This is deliberately an availability verdict, not an attribution verdict:
/// an assumed role, impersonated service account, or credential broker can
/// involve both the caller and the deployment, and a generic AccessDenied does
/// not say which step failed. The external-table API therefore reports one
/// stable outcome -- SourceInvalid -- for missing, denied, or unusable sources.
/// Retryable transport/service failures and data-format failures retain the
/// producing layer's classification.
inline int ExternalSourceErrorCodeFromStatus(const arrow::Status& status, int fallback = LOON_ARROW_ERROR) {
  auto code = FFIErrorCodeFromExtendStatus(status, fallback);
  switch (code) {
    case LOON_STORAGE_NOT_FOUND:
    case LOON_FILE_NOT_FOUND:
    case LOON_STORAGE_ACCESS_DENIED:
    case LOON_STORAGE_CONFIG_INVALID:
    case LOON_STORAGE_BUCKET_NOT_FOUND:
      return LOON_SOURCE_INVALID;

    default:
      return code;
  }
}

/// Same contract as RETURN_ARROW_ERROR_IF, for a status produced while
/// resolving or reading an external source. See ExternalSourceErrorCodeFromStatus().
#define RETURN_EXTERNAL_SOURCE_ERROR_IF(status, fallback, ...)                         \
  do {                                                                                 \
    auto ffi_status__ = (status);                                                      \
    if (!ffi_status__.ok()) {                                                          \
      auto ffi_err_code__ = ExternalSourceErrorCodeFromStatus(ffi_status__, fallback); \
      RETURN_ERROR(ffi_err_code__, ##__VA_ARGS__);                                     \
    }                                                                                  \
  } while (0)

// The place every ERROR result is materialized, so the place that must
// not throw: every caller is either about to cross the C ABI or already inside
// a catch block doing so, and an exception here is undefined behaviour.
//
// Formatting an error allocates and can itself fail while an exception is being
// reported. The code still crosses the boundary; only the message is given up.
// loon_ffi_free_result is free(), which accepts null, and
// loon_ffi_get_errmsg's consumers already handle a null message.
template <typename... Args>
LoonFFIResult CreateFFIResult(int code, Args&&... args) noexcept {
  LoonFFIResult result;
  assert(code != LOON_SUCCESS);
  result.err_code = code;
  result.message = nullptr;

  try {
    std::ostringstream ss;
    ss << "ERROR: " << error_to_string(code) << "(code " << code << ") details: ";
    if constexpr (sizeof...(Args) > 0) {
      (ss << ... << std::forward<Args>(args));
    } else {
      ss << "<no details>";
    }
    result.message = strdup(ss.str().c_str());
  } catch (...) {
    // Keep the code, drop the message. Better a terse error than no error --
    // and infinitely better than throwing across the C ABI.
  }

  return result;
}
