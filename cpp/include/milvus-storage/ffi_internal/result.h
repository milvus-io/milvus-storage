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

#define RETURN_EXCEPTION(...)                                                                  \
  do {                                                                                         \
    return CreateFFIResult((LOON_GOT_EXCEPTION), __func__, " Got exception: ", ##__VA_ARGS__); \
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
  if (!detail) {
    return fallback;
  }

  return milvus_storage::ffi_internal::FFIErrorCodeFromExtendStatusCode(detail->code(), fallback);
}

inline std::optional<milvus_storage::ExtendStatusCode> ExtendStatusCodeFromFFIErrorCode(int err_code) {
  return milvus_storage::ffi_internal::ExtendStatusCodeFromFFIErrorCode(err_code);
}

template <typename... Args>
LoonFFIResult CreateFFIResult(int code, Args&&... args) {
  LoonFFIResult result;
  std::ostringstream ss;
  assert(code != LOON_SUCCESS);

  ss << "ERROR: " << error_to_string(code) << "(code " << code << ") details: ";
  if constexpr (sizeof...(Args) > 0) {
    (ss << ... << std::forward<Args>(args));
  } else {
    ss << "<no details>";
  }

  result.err_code = code;
  result.message = strdup(ss.str().c_str());

  return result;
}
