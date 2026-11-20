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

#ifndef LOON_SOURCE_READ_CONTEXT_C
#define LOON_SOURCE_READ_CONTEXT_C

#include <stddef.h>
#include <stdint.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_filesystem_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Runtime-only handle. Never serialize this value or place it in properties. */
typedef uintptr_t LoonSourceReadContextHandle;

/** Provider outcomes, not LoonFFIResult error codes. */
typedef enum LoonSourceCredentialStatus {
  LOON_SOURCE_CREDENTIAL_OK = 0,
  LOON_SOURCE_CREDENTIAL_DENIED = 1,
  LOON_SOURCE_CREDENTIAL_EXPIRED = 2,
  LOON_SOURCE_CREDENTIAL_TRANSIENT = 3,
  LOON_SOURCE_CREDENTIAL_CANCELLED = 4,
  LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE = 5,
} LoonSourceCredentialStatus;

typedef enum LoonSourceReadOperation {
  LOON_SOURCE_READ = 0,
  LOON_SOURCE_LIST = 1,
} LoonSourceReadOperation;

typedef enum LoonSourceCredentialKind {
  LOON_SOURCE_CREDENTIAL_KIND_UNSPECIFIED = 0,
  LOON_SOURCE_CREDENTIAL_KIND_AZURE_SAS = 1,
} LoonSourceCredentialKind;

/** Borrowed fields remain valid only until the fetch callback returns. */
typedef struct LoonSourceCredentialRequest {
  uint32_t abi_version;
  size_t struct_size;
  const char* resource_uri;
  size_t resource_uri_length;
  /** Selected authorized root; the returned directory SAS must bind this root. */
  const char* scope_root_uri;
  size_t scope_root_uri_length;
  LoonSourceReadOperation operation;
  /** Finite UTC deadline, including the context's operation deadline if set. */
  int64_t request_deadline_unix_ms;
  /** Zero if unset. Issued SAS must not outlive a fixed operation deadline. */
  int64_t operation_deadline_unix_ms;
} LoonSourceCredentialRequest;

/**
 * Callback-owned output. Storage copies successful output before release_result.
 * Storage initializes status to INVALID_RESPONSE before invoking fetch.
 * No provider error text is accepted: failures expose only the typed outcome.
 */
typedef struct LoonSourceCredentialResult {
  uint32_t abi_version;
  size_t struct_size;
  LoonSourceCredentialStatus status;
  LoonSourceCredentialKind credential_kind;
  const char* account_name;
  size_t account_name_length;
  const char* sas_token;
  size_t sas_token_length;
  /** Must equal the requested authorized root after canonicalization. */
  const char* scope_root_uri;
  size_t scope_root_uri_length;
  int64_t expires_at_unix_ms;
  void* result_userdata;
} LoonSourceCredentialResult;

/**
 * Callbacks must be thread-safe and must not throw across the C ABI.
 * fetch must honor request_deadline_unix_ms. Native code cannot forcibly stop a
 * callback that ignores its deadline. Late results are rejected, and userdata
 * stays alive until the callback has returned.
 * release_result is called once after every fetch invocation, on success or
 * failure. It releases callback-owned result buffers, not the result structure.
 * destroy_userdata is called once after the last retained context reference and
 * in-flight callback has been released. No callback may run after destruction.
 */
typedef struct LoonSourceCredentialCallbacks {
  uint32_t abi_version;
  size_t struct_size;
  void* userdata;
  void (*fetch)(void* userdata, const LoonSourceCredentialRequest* request, LoonSourceCredentialResult* result);
  void (*release_result)(void* userdata, LoonSourceCredentialResult* result);
  void (*destroy_userdata)(void* userdata);
} LoonSourceCredentialCallbacks;

/**
 * Immutable source scope supplied by a trusted engine, not arbitrary client
 * properties. Strings are copied during construction. Root URI normalization
 * and supported Azure URI forms are part of the context implementation contract.
 */
typedef struct LoonSourceReadContextOptions {
  uint32_t abi_version;
  size_t struct_size;
  const char* account_name;
  const char* const* allowed_root_uris;
  size_t allowed_root_count;
  /** Zero means no fixed operation deadline; individual requests remain bounded. */
  int64_t operation_deadline_unix_ms;
  int64_t request_timeout_ms;
} LoonSourceReadContextOptions;

/**
 * Creates an owned context. On success storage owns callback userdata; on failure
 * ownership remains with the caller. The output is zero on failure.
 * Version 1 accepts Azure SAS credentials only.
 */
FFI_EXPORT LoonFFIResult loon_source_read_context_create(const LoonSourceReadContextOptions* options,
                                                         const LoonSourceCredentialCallbacks* callbacks,
                                                         LoonSourceReadContextHandle* out_handle);

/** Creates an independently releasable handle retaining the same context. */
FFI_EXPORT LoonFFIResult loon_source_read_context_retain(LoonSourceReadContextHandle handle,
                                                         LoonSourceReadContextHandle* out_handle);

/** Cancels future source I/O and discards results of in-flight acquisitions. */
FFI_EXPORT void loon_source_read_context_cancel(LoonSourceReadContextHandle handle);

/** Releases this handle. Retained readers and cloned handles remain owners. */
FFI_EXPORT void loon_source_read_context_release(LoonSourceReadContextHandle handle);

/** Context-aware entry points reject a zero context instead of falling back. */
FFI_EXPORT LoonFFIResult loon_reader_new_with_context(const LoonColumnGroups* column_groups,
                                                      struct ArrowSchema* schema,
                                                      const char* const* needed_columns,
                                                      size_t num_columns,
                                                      const LoonProperties* properties,
                                                      LoonSourceReadContextHandle context,
                                                      LoonReaderHandle* out_handle);

FFI_EXPORT LoonFFIResult loon_exttable_explore_with_context(const char** columns,
                                                            size_t col_lens,
                                                            const char* format,
                                                            const char* base_dir,
                                                            const char* explore_dir,
                                                            const LoonProperties* properties,
                                                            LoonSourceReadContextHandle context,
                                                            uint64_t* out_num_of_files,
                                                            char** out_column_groups_file_path);

FFI_EXPORT LoonFFIResult loon_exttable_get_file_info_with_context(const char* format,
                                                                  const char* file_path,
                                                                  const LoonProperties* properties,
                                                                  LoonSourceReadContextHandle context,
                                                                  uint64_t* out_num_of_rows);

FFI_EXPORT LoonFFIResult loon_filesystem_get_with_context(const LoonProperties* properties,
                                                          const char* path,
                                                          uint32_t path_len,
                                                          LoonSourceReadContextHandle context,
                                                          FileSystemHandle* out_handle);

#ifdef __cplusplus
}
#endif

#endif  // LOON_SOURCE_READ_CONTEXT_C
