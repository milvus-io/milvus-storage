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
};

constexpr LoonErrorMetadata kLoonErrorMetadata[] = {
#define MILVUS_STORAGE_INTERNAL_ERROR_ENTRY(name, code, symbol, category) {(code), name, (category)},
    LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_INTERNAL_ERROR_ENTRY)
#undef MILVUS_STORAGE_INTERNAL_ERROR_ENTRY
#define MILVUS_STORAGE_EXTEND_ERROR_ENTRY(name, code, symbol, category) {(code), #name, (category)},
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

#define MILVUS_STORAGE_ERRCODE_CONSTANT(name, code, symbol, category) \
  extern FFI_EXPORT const int loon_errcode_##symbol = (code);
LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_ERRCODE_CONSTANT)
LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_ERRCODE_CONSTANT)
#undef MILVUS_STORAGE_ERRCODE_CONSTANT

extern FFI_EXPORT const int loon_errcode_aws_no_such_upload = LOON_AWS_ERROR_NO_SUCH_UPLOAD;
extern FFI_EXPORT const int loon_errcode_aws_conflict = LOON_AWS_ERROR_CONFLICT;
extern FFI_EXPORT const int loon_errcode_aws_precondition_failed = LOON_AWS_ERROR_PRECONDITION_FAILED;
extern FFI_EXPORT const int loon_errcode_aws_not_found = LOON_AWS_ERROR_NOT_FOUND;
extern FFI_EXPORT const int loon_errcode_aws_access_denied = LOON_AWS_ERROR_ACCESS_DENIED;
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
  if (result->message == nullptr) {
    // CreateFFIResult gives up the message rather than throw while reporting an
    // exception. Callers feed this into printf-style formatting, where null is
    // undefined behaviour, so return static text instead.
    return "(error message unavailable: formatting failed)";
  }
  return result->message;
}

void loon_ffi_free_result(LoonFFIResult* result) {
  assert(result);
  free(result->message);
  // Freeing must be idempotent. The documented consumer pattern is "free on
  // every path", and every retry/classification loop that follows it has at
  // least one shape -- an early `continue`, a caller freeing what a helper
  // already freed -- where the same result is released twice. free() tolerates
  // null; it does not tolerate a stale pointer. Nulling here also turns a
  // use-after-free in loon_ffi_get_errmsg into the static "unavailable" text.
  result->message = nullptr;
}

int loon_ffi_error_category(int err_code) {
  if (const auto* metadata = FindLoonErrorMetadata(err_code); metadata != nullptr) {
    return metadata->category;
  }
  return LOON_ERROR_CATEGORY_UNKNOWN;
}

int loon_ffi_is_retryable_errcode(int err_code) {
  return loon_ffi_error_category(err_code) == LOON_ERROR_CATEGORY_RETRYABLE;
}
