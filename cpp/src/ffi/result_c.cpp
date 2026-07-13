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

extern "C" {

extern FFI_EXPORT const int loon_errcode_success = LOON_SUCCESS;
extern FFI_EXPORT const int loon_errcode_invalid_args = LOON_INVALID_ARGS;
extern FFI_EXPORT const int loon_errcode_memory = LOON_MEMORY_ERROR;
extern FFI_EXPORT const int loon_errcode_arrow = LOON_ARROW_ERROR;
extern FFI_EXPORT const int loon_errcode_logical = LOON_LOGICAL_ERROR;
extern FFI_EXPORT const int loon_errcode_got_exception = LOON_GOT_EXCEPTION;
extern FFI_EXPORT const int loon_errcode_unreachable = LOON_UNREACHABLE_ERROR;
extern FFI_EXPORT const int loon_errcode_invalid_properties = LOON_INVALID_PROPERTIES;
extern FFI_EXPORT const int loon_errcode_fault_inject = LOON_FAULT_INJECT_ERROR;
extern FFI_EXPORT const int loon_errcode_not_support = LOON_NOT_SUPPORT;
extern FFI_EXPORT const int loon_errcode_file_not_found = LOON_FILE_NOT_FOUND;
extern FFI_EXPORT const int loon_errcode_aws_no_such_upload = LOON_AWS_ERROR_NO_SUCH_UPLOAD;
extern FFI_EXPORT const int loon_errcode_aws_conflict = LOON_AWS_ERROR_CONFLICT;
extern FFI_EXPORT const int loon_errcode_aws_precondition_failed = LOON_AWS_ERROR_PRECONDITION_FAILED;
extern FFI_EXPORT const int loon_errcode_aws_not_found = LOON_AWS_ERROR_NOT_FOUND;
extern FFI_EXPORT const int loon_errcode_aws_access_denied = LOON_AWS_ERROR_ACCESS_DENIED;
extern FFI_EXPORT const int loon_errcode_aws_non_retryable = LOON_AWS_ERROR_NON_RETRYABLE;
extern FFI_EXPORT const int loon_errcode_transient_network = LOON_TRANSIENT_NETWORK;
extern FFI_EXPORT const int loon_errcode_transient_timeout = LOON_TRANSIENT_TIMEOUT;
extern FFI_EXPORT const int loon_errcode_transient_throttling = LOON_TRANSIENT_THROTTLING;
extern FFI_EXPORT const int loon_errcode_transient_service = LOON_TRANSIENT_SERVICE;
extern FFI_EXPORT const int loon_errcode_txn_exhausted_retry = LOON_TXN_EXHAUSTED_RETRY;
extern FFI_EXPORT const int loon_errcode_txn_resolution_failed = LOON_TXN_RESOLUTION_FAILED;

}  // extern "C"

std::string error_to_string(int code) {
  static const std::unordered_map<int, std::string> error_strings = {
      {LOON_SUCCESS, "Success"},
      {LOON_INVALID_ARGS, "Invalid args"},
      {LOON_MEMORY_ERROR, "Memory allocation failed"},
      {LOON_ARROW_ERROR, "Internal error"},
      {LOON_LOGICAL_ERROR, "Logical error"},
      {LOON_GOT_EXCEPTION, "Got exception"},
      {LOON_UNREACHABLE_ERROR, "Unreachable code"},
      {LOON_INVALID_PROPERTIES, "Invalid properties"},
      {LOON_FAULT_INJECT_ERROR, "Fault injection error"},
      {LOON_NOT_SUPPORT, "Not supported"},
      {LOON_FILE_NOT_FOUND, "File not found"},
      {LOON_AWS_ERROR_NO_SUCH_UPLOAD, "AwsErrorNoSuchUpload"},
      {LOON_AWS_ERROR_CONFLICT, "AwsErrorConflict"},
      {LOON_AWS_ERROR_PRECONDITION_FAILED, "AwsErrorPreConditionFailed"},
      {LOON_AWS_ERROR_NOT_FOUND, "AwsErrorNotFound"},
      {LOON_AWS_ERROR_ACCESS_DENIED, "AwsErrorAccessDenied"},
      {LOON_AWS_ERROR_NON_RETRYABLE, "AwsErrorNonRetryable"},
      {LOON_TRANSIENT_NETWORK, "StorageTransientNetwork"},
      {LOON_TRANSIENT_TIMEOUT, "StorageTransientTimeout"},
      {LOON_TRANSIENT_THROTTLING, "StorageTransientThrottling"},
      {LOON_TRANSIENT_SERVICE, "StorageTransientService"},
      {LOON_TXN_EXHAUSTED_RETRY, "TxnExhaustedRetry"},
      {LOON_TXN_RESOLUTION_FAILED, "TxnResolutionFailed"},
  };

  if (auto it = error_strings.find(code); it != error_strings.end()) {
    return it->second;
  }
  return "Unknown error(undefined)";
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

int loon_ffi_is_retryable_errcode(int err_code) {
  auto code = milvus_storage::ExtendStatusCodeFromInt(err_code);
  return code.has_value() && milvus_storage::DefaultRetryableForExtendStatusCode(*code);
}
