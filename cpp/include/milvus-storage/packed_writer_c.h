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

#ifndef MILVUS_STORAGE_PACKED_WRITER_C_H_
#define MILVUS_STORAGE_PACKED_WRITER_C_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "milvus-storage/ffi_c.h"

// Opaque handle for the packed record-batch writer.
typedef uintptr_t LoonPackedWriterHandle;

/**
 * @brief Create a PackedRecordBatchWriter that writes one parquet file per
 *        column group at the caller-specified paths.
 *
 * Mirrors what milvus's `internal/storagev2/packed/packed_writer.go` does on
 * the Milvus side. Unlike `loon_writer_new` (which lets the Loon WriterImpl
 * pick UUID-suffixed paths under a `_data/` prefix), this entry point is for
 * callers that need exact path control — e.g. the spark backfill writer that
 * must produce `{rootPath}/insert_log/{coll}/{part}/{seg}/{fieldId}/{logId}`.
 *
 * Column groups are passed as a CSR-style flat encoding:
 *   - `group_offsets[i]..group_offsets[i+1]` is the slice of `group_indices`
 *     that belongs to group `i`.
 *   - `group_offsets` length = `num_groups + 1`.
 *
 * @param paths           Output parquet path per group (length `num_groups`).
 * @param num_groups      Number of column groups (also number of paths).
 * @param group_offsets   CSR offsets, length `num_groups + 1`.
 * @param group_indices   Flat array of column indices into `schema`.
 * @param total_indices   Total length of `group_indices`.
 * @param schema          Arrow schema describing all input columns (consumed
 *                        via Arrow C Data Interface).
 * @param properties      LoonProperties carrying filesystem + storage config.
 * @param buffer_size     Max in-memory buffer (0 → DEFAULT_WRITE_BUFFER_SIZE).
 * @param out_handle      Output handle (caller must call
 *                        `loon_packed_writer_destroy`).
 */
FFI_EXPORT LoonFFIResult loon_packed_writer_new(const char* const* paths,
                                                int32_t num_groups,
                                                const int32_t* group_offsets,
                                                const int32_t* group_indices,
                                                int32_t total_indices,
                                                struct ArrowSchema* schema,
                                                const LoonProperties* properties,
                                                int64_t buffer_size,
                                                LoonPackedWriterHandle* out_handle);

/**
 * @brief Write one Arrow record batch (containing all columns from the schema
 *        passed at writer-new time). Internally split per column group.
 */
FFI_EXPORT LoonFFIResult loon_packed_writer_write(LoonPackedWriterHandle handle, struct ArrowArray* array);

/**
 * @brief Close the writer and flush all column-group files to storage. After
 *        this returns successfully, every parquet path passed at writer-new
 *        time exists with milvus-storage's full KV metadata
 *        (`row_group_metadata`, `storage_version`, `group_field_id_list`).
 */
FFI_EXPORT LoonFFIResult loon_packed_writer_close(LoonPackedWriterHandle handle);

/**
 * @brief Destroy the writer handle and release all resources.
 */
FFI_EXPORT void loon_packed_writer_destroy(LoonPackedWriterHandle handle);

#ifdef __cplusplus
}
#endif

#endif  // MILVUS_STORAGE_PACKED_WRITER_C_H_
