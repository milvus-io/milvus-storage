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

/// The code for the exception currently being handled.
///
/// Call only from inside a catch block. Everything is LOON_GOT_EXCEPTION -- our
/// bug, Permanent, never retry -- except std::bad_alloc, which is memory
/// pressure: another node, or this one later, may have the memory.
///
/// This exists because the two halves of "we ran out of memory" disagreed. An
/// arrow OutOfMemory status already mapped to LOON_MEMORY_ERROR and came back
/// retriable; the identical condition arriving as a thrown std::bad_alloc hit
/// the catch-all at every FFI entry point and came back Permanent. Same event,
/// opposite instruction, decided by which layer happened to notice it.
///
/// Rethrowing to recover the type is the only way to see it: the catch blocks
/// bind std::exception& and RETURN_EXCEPTION is handed a string. Doing it here
/// fixes all ~55 entry points without touching one of them.
inline int FFIExceptionErrorCode() {
  try {
    auto current = std::current_exception();
    if (!current) {
      return LOON_GOT_EXCEPTION;
    }
    std::rethrow_exception(current);
  } catch (const std::bad_alloc&) {
    return LOON_MEMORY_ERROR;
  } catch (...) {
  }
  return LOON_GOT_EXCEPTION;
}

#define RETURN_EXCEPTION(...)                                                                     \
  do {                                                                                            \
    return CreateFFIResult(FFIExceptionErrorCode(), __func__, " Got exception: ", ##__VA_ARGS__); \
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

  // An allocation failure needs no ExtendStatusDetail to be classified: arrow
  // already says OutOfMemory, and LOON_MEMORY_ERROR is Transient. Without this
  // branch an OOM fell to the call site's fallback -- Permanent in general, and
  // User at the entry points that pass LOON_SOURCE_INVALID.
  if (status.IsOutOfMemory()) {
    return LOON_MEMORY_ERROR;
  }

  if (arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    return LOON_FILE_NOT_FOUND;
  }
  return fallback;
}

inline std::optional<milvus_storage::ExtendStatusCode> ExtendStatusCodeFromFFIErrorCode(int err_code) {
  return milvus_storage::ffi_internal::ExtendStatusCodeFromFFIErrorCode(err_code);
}

/// \brief Classify a failure that happened while touching a location the USER
/// named, re-tagging the two conditions whose owner depends on the call site.
///
/// The same object-store condition means opposite things depending on whose
/// path it is. A missing object on an internally generated path is a system
/// failure (a GC race, lost data, stale metadata) that an operator must look
/// at; a missing bucket that the user typed into an external-source URI is a
/// user error that no retry and no operator can fix. The storage layer that
/// raises the error cannot tell the two apart -- only the entry point knows
/// where the path came from, so ONLY entry points that accept a user-supplied
/// location may call this.
///
/// Everything else keeps the classification the producing layer attached.
/// \param location_is_user_supplied whether the location that failed is one the
///        caller named -- an absolute URI or explicit extfs.* properties --
///        rather than a relative path the deployment's own configuration
///        resolves. Only the former may re-tag a configuration failure as the
///        user's.
inline int UserSourceErrorCodeFromStatus(const arrow::Status& status,
                                         int fallback = LOON_ARROW_ERROR,
                                         bool location_is_user_supplied = true) {
  auto code = FFIErrorCodeFromExtendStatus(status, fallback);
  switch (code) {
    // The object the user named is not there, or their credentials for it were
    // rejected.
    case LOON_AWS_ERROR_NOT_FOUND:
    case LOON_FILE_NOT_FOUND:
    case LOON_AWS_ERROR_ACCESS_DENIED:
      return LOON_SOURCE_INVALID;

    // The location spec itself does not work: its URI/extfs properties are
    // unusable, or its bucket is not there.
    //
    // Gated on the location actually being one the user named. A RELATIVE
    // external path is resolved against the deployment's default fs.*
    // configuration, so a missing fs.azure_tenant_id surfaced here as
    // LOON_SOURCE_INVALID -- telling the caller their DDL was wrong about a
    // setting only an operator can reach. The retry verdict was the same
    // either way; the person sent to fix it was not.
    case LOON_STORAGE_CONFIG_INVALID:
    case LOON_AWS_ERROR_BUCKET_NOT_FOUND:
      return location_is_user_supplied ? LOON_SOURCE_INVALID : code;

    default:
      return code;
  }
}

/// Same contract as RETURN_ARROW_ERROR_IF, for a status produced while reaching
/// a user-supplied location. See UserSourceErrorCodeFromStatus().
#define RETURN_USER_SOURCE_ERROR_IF(status, fallback, ...)                         \
  do {                                                                             \
    auto ffi_status__ = (status);                                                  \
    if (!ffi_status__.ok()) {                                                      \
      auto ffi_err_code__ = UserSourceErrorCodeFromStatus(ffi_status__, fallback); \
      RETURN_ERROR(ffi_err_code__, ##__VA_ARGS__);                                 \
    }                                                                              \
  } while (0)

/// Same, for entry points whose location may be relative -- in which case the
/// deployment's own fs.* configuration is what resolved it, and a
/// configuration failure is the operator's rather than the caller's.
#define RETURN_USER_SOURCE_ERROR_IF_AT(status, fallback, user_supplied, ...)                      \
  do {                                                                                            \
    auto ffi_status__ = (status);                                                                 \
    if (!ffi_status__.ok()) {                                                                     \
      auto ffi_err_code__ = UserSourceErrorCodeFromStatus(ffi_status__, fallback, user_supplied); \
      RETURN_ERROR(ffi_err_code__, ##__VA_ARGS__);                                                \
    }                                                                                             \
  } while (0)

// The place every ERROR result is materialized, so the place that must
// not throw: every caller is either about to cross the C ABI or already inside
// a catch block doing so, and an exception here is undefined behaviour.
//
// The case that makes this real is the one this function is most likely to hit:
// reporting an out-of-memory error. Formatting the message allocates, and under
// genuine memory pressure that allocation can itself throw bad_alloc -- from
// inside the handler for the first one. The code still crosses the boundary;
// only the message is given up. loon_ffi_free_result is free(), which accepts
// null, and loon_ffi_get_errmsg's consumers already handle a null message.
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
