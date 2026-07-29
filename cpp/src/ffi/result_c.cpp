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

#include <string>
#include <cassert>
#include <unordered_map>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"

namespace {

// The one place every FFI error code is described. Both halves are generated
// from the tables in ffi_error_code.h, so a code cannot be named here and
// classified differently in extend_status.cpp.
struct LoonErrorMetadata {
  int code;
  const char* name;
  int category;
  const char* s3_code;
};

constexpr LoonErrorMetadata kLoonErrorMetadata[] = {
#define MILVUS_STORAGE_INTERNAL_ERROR_ENTRY(name, code, symbol, category, s3_code) {(code), name, (category), s3_code},
    LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_INTERNAL_ERROR_ENTRY)
#undef MILVUS_STORAGE_INTERNAL_ERROR_ENTRY
#define MILVUS_STORAGE_EXTEND_ERROR_ENTRY(name, code, symbol, category, s3_code) {(code), #name, (category), s3_code},
        LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_EXTEND_ERROR_ENTRY)
#undef MILVUS_STORAGE_EXTEND_ERROR_ENTRY
};

const LoonErrorMetadata* FindLoonErrorMetadata(int code) {
  for (const auto& metadata : kLoonErrorMetadata) {
    if (metadata.code == code) {
      return &metadata;
    }
  }
  return nullptr;
}

constexpr const char* kUnknownErrorName = "Unknown error(undefined)";

}  // namespace

extern "C" {

extern FFI_EXPORT const int loon_errcode_success = LOON_SUCCESS;

#define MILVUS_STORAGE_ERRCODE_CONSTANT(name, code, symbol, category, s3_code) \
  extern FFI_EXPORT const int loon_errcode_##symbol = (code);
LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_ERRCODE_CONSTANT)
LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_ERRCODE_CONSTANT)
#undef MILVUS_STORAGE_ERRCODE_CONSTANT

extern FFI_EXPORT const int loon_error_category_unknown = LOON_ERROR_CATEGORY_UNKNOWN;
extern FFI_EXPORT const int loon_error_category_user = LOON_ERROR_CATEGORY_USER;
extern FFI_EXPORT const int loon_error_category_transient = LOON_ERROR_CATEGORY_TRANSIENT;
extern FFI_EXPORT const int loon_error_category_permanent = LOON_ERROR_CATEGORY_PERMANENT;

}  // extern "C"

std::string error_to_string(int code) {
  if (code == LOON_SUCCESS) {
    return "Success";
  }
  if (const auto* metadata = FindLoonErrorMetadata(code); metadata != nullptr) {
    return metadata->name;
  }
  return kUnknownErrorName;
}

int loon_ffi_is_success(LoonFFIResult* result) {
  assert(result);
  return result->err_code == LOON_SUCCESS;
}

const char* loon_ffi_get_errmsg(LoonFFIResult* result) {
  assert(result);
  if (loon_ffi_is_success(result)) {
    return nullptr;
  }
  return result->message;
}

void loon_ffi_free_result(LoonFFIResult* result) {
  assert(result);
  free(result->message);
}

int loon_ffi_error_category(int err_code) {
  if (const auto* metadata = FindLoonErrorMetadata(err_code); metadata != nullptr) {
    return metadata->category;
  }
  return LOON_ERROR_CATEGORY_UNKNOWN;
}

// Retriability is derived, never stored separately: an error is worth retrying
// exactly when it is transient. Unknown codes are non-retriable by omission.
int loon_ffi_is_retryable_errcode(int err_code) {
  return loon_ffi_error_category(err_code) == LOON_ERROR_CATEGORY_TRANSIENT;
}

const char* loon_ffi_error_name(int err_code) {
  if (err_code == LOON_SUCCESS) {
    return "Success";
  }
  if (const auto* metadata = FindLoonErrorMetadata(err_code); metadata != nullptr) {
    return metadata->name;
  }
  return kUnknownErrorName;
}
