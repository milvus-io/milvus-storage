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

#ifndef LOON_FIU_C
#define LOON_FIU_C

#include "milvus-storage/ffi_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file ffi_fiu_c.h
 * @brief Fault Injection FFI Interface
 *
 * This header provides FFI functions for fault injection using libfiu.
 * Fault injection is used for testing error handling and recovery scenarios.
 *
 * Example usage (Python):
 * @code
 * # Enable fault point to fail once (one_time=1)
 * loon_fiu_enable(loon_fiu_key_writer_flush_fail, len, 1)
 *
 * # The next flush will fail
 * writer.flush()  # raises IOError
 *
 * # Retry should succeed (one_time auto-disables)
 * writer.flush()  # succeeds
 *
 * # Disable all fault points
 * loon_fiu_disable_all()
 * @endcode
 */

// ==================== Fault Point Names ====================
// Use these constants instead of hardcoding fault point names

// --- Writer fault points ---
/** Fault point: Fail during Writer write batch operation */
FFI_EXPORT extern const char* loon_fiukey_writer_write_fail;

/** Fault point: Fail during Writer flush operation */
FFI_EXPORT extern const char* loon_fiukey_writer_flush_fail;

/** Fault point: Fail during Writer close operation */
FFI_EXPORT extern const char* loon_fiukey_writer_close_fail;

// --- Reader fault points (low-level) ---
/** Fault point: Fail during ColumnGroup read (get_chunk/get_chunks) */
FFI_EXPORT extern const char* loon_fiukey_column_group_read_fail;

/** Fault point: Fail during take rows operation */
FFI_EXPORT extern const char* loon_fiukey_take_rows_fail;

/** Fault point: Fail during ChunkReader read operation (FFI layer) */
FFI_EXPORT extern const char* loon_fiukey_chunk_reader_read_fail;

/** Fault point: Fail during Reader open operation (FFI layer) */
FFI_EXPORT extern const char* loon_fiukey_reader_open_fail;

// --- Transaction/Manifest fault points ---
/** Fault point: Fail during Transaction/Manifest commit */
FFI_EXPORT extern const char* loon_fiukey_manifest_commit_fail;

/** Fault point: Fail during Manifest read operation */
FFI_EXPORT extern const char* loon_fiukey_manifest_read_fail;

/** Fault point: Fail during Manifest write/serialize operation */
FFI_EXPORT extern const char* loon_fiukey_manifest_write_fail;

// --- Filesystem fault points ---
/** Fault point: Fail during FileSystem OpenOutputStream operation */
FFI_EXPORT extern const char* loon_fiukey_fs_open_output_fail;

/** Fault point: Fail during FileSystem OpenInputFile operation */
FFI_EXPORT extern const char* loon_fiukey_fs_open_input_fail;

// --- S3 Filesystem fault points ---
/** Fault point: Fail during S3 CreateMultipartUpload operation */
FFI_EXPORT extern const char* loon_fiukey_s3fs_create_upload_fail;

/** Fault point: Fail during S3 multipart upload UploadPart operation */
FFI_EXPORT extern const char* loon_fiukey_s3fs_part_upload_fail;

/** Fault point: Fail during S3 multipart upload CompleteMultipartUpload operation */
FFI_EXPORT extern const char* loon_fiukey_s3fs_complete_upload_fail;

/** Fault point: Fail during S3 ObjectInputFile Read operation */
FFI_EXPORT extern const char* loon_fiukey_s3fs_read_fail;

/** Fault point: Fail during S3 ObjectInputFile ReadAt operation */
FFI_EXPORT extern const char* loon_fiukey_s3fs_readat_fail;

// --- ColumnGroup fault points ---
/** Fault point: Fail during ColumnGroup write operation */
FFI_EXPORT extern const char* loon_fiukey_column_group_write_fail;

/**
 * Enable a fault injection point.
 *
 * @param name The name of the fault point to enable.
 * @param name_len The length of the fault point name.
 * @param one_time If non-zero, the fault triggers only once (FIU_ONETIME).
 *                 If zero, the fault triggers forever until disabled.
 * @return result of FFI (success or error if FIU not enabled)
 */
FFI_EXPORT LoonFFIResult loon_fiu_enable(const char* name, uint32_t name_len, int one_time);

/**
 * Disable a specific fault injection point.
 *
 * @param name The name of the fault point to disable.
 * @param name_len The length of the fault point name.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_fiu_disable(const char* name, uint32_t name_len);

/**
 * Disable all active fault injection points.
 *
 * This should be called in test cleanup to ensure all fault points
 * are disabled after a test completes.
 */
FFI_EXPORT void loon_fiu_disable_all(void);

/**
 * Check if fault injection support is compiled in.
 *
 * @return 1 if FIU is enabled, 0 otherwise.
 */
FFI_EXPORT int loon_fiu_is_enabled(void);

#ifdef __cplusplus
}
#endif

#endif  // LOON_FIU_C
