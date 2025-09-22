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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef READER_C
#define READER_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "milvus-storage/result_c.h"

// TODO: move out this header
#include <arrow/c/abi.h>

// ==================== ReadProperties C Interface ====================

/// C struct for a single property key-value pair
typedef struct {
  char* key;    ///< Property key (caller owns memory)
  char* value;  ///< Property value (caller owns memory)
} ReadProperty;

/// C struct for read properties (array of key-value pairs)
typedef struct {
  ReadProperty* properties;  ///< Array of property key-value pairs (caller owns memory)
  size_t count;              ///< Number of properties in the array
} ReadProperties;

/**
 * @brief Creates read properties from key-value arrays
 *
 * @param keys Array of property keys
 * @param values Array of property values
 * @param count Number of key-value pairs
 * @param properties Output parameter for created properties (caller must free)
 */
FFIResult read_properties_create(const char* const* keys,
                                 const char* const* values,
                                 size_t count,
                                 ReadProperties* properties);

/**
 * @brief Gets a property value by key
 *
 * @param properties Properties to search
 * @param key Property key to find
 * @return Property value if found, NULL otherwise (do not free)
 */
const char* read_properties_get(const ReadProperties* properties, const char* key);

/**
 * @brief Frees memory allocated for ReadProperties
 *
 * @param properties Properties to free
 */
void read_properties_free(ReadProperties* properties);

// ==================== ChunkReader C Interface ====================

/// Opaque handle for ChunkReader
typedef uintptr_t ChunkReaderHandle;

/**
 * @brief Maps row indices to their corresponding chunk indices
 *
 * @param reader ChunkReader handle
 * @param row_indices Array of global row indices to map
 * @param num_indices Number of indices in the array
 * @param chunk_indices Output array of chunk indices (caller must free)
 * @param num_chunk_indices Output number of chunk indices
 * @return 0 on success, others is error code
 */
FFIResult get_chunk_indices(ChunkReaderHandle reader,
                            const int64_t* row_indices,
                            size_t num_indices,
                            int64_t** chunk_indices,
                            size_t* num_chunk_indices);

/**
 * @brief Retrieves a single chunk by its index
 *
 * @param reader ChunkReader handle
 * @param chunk_index Zero-based index of the chunk to retrieve
 * @param array Output array of RecordBatch (caller must free)
 * @return 0 on success, others is error code
 */
FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, struct ArrowArray* out_array);

/**
 * @brief Retrieves multiple chunks by their indices with optional parallel processing
 *
 * @param reader ChunkReader handle
 * @param chunk_indices Array of chunk indices to retrieve
 * @param num_indices Number of indices in the array
 * @param parallelism Number of threads to use for parallel reading
 * @param arrays Output array of RecordBatch handles (caller must free)
 * @param num_arrays Output number of record batches
 * @return 0 on success, others is error code
 */
FFIResult get_chunks(ChunkReaderHandle reader,
                     const int64_t* chunk_indices,
                     size_t num_indices,
                     int64_t parallelism,
                     struct ArrowArray** arrays,
                     size_t* num_arrays);

/**
 * @brief Destroys a ChunkReader
 *
 * @param reader ChunkReader handle to destroy
 */
void chunk_reader_destroy(ChunkReaderHandle reader);

// ==================== Reader C Interface ====================

/// Opaque handle for Reader
typedef uintptr_t ReaderHandle;

/**
 * @brief Creates a new Reader for a milvus storage dataset
 *
 * @param fs Filesystem interface handle
 * @param manifest Dataset manifest handle
 * @param schema Arrow schema handle
 * @param needed_columns Array of column names to read (NULL for all columns)
 * @param num_columns Number of columns in needed_columns array
 * @param properties Read configuration properties
 * @param out_handle Output (caller must call `reader_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFIResult reader_new(char* manifest,
                     struct ArrowSchema* schema,
                     const char* const* needed_columns,
                     size_t num_columns,
                     const ReadProperties* properties,
                     ReaderHandle* out_handle);

/**
 * @brief Performs a full table scan with optional filtering and buffering
 *
 * Creates a RecordBatchReader for sequential reading of the entire dataset.
 * The reader automatically handles column group coordination and provides
 * efficient streaming access to large datasets.
 *
 * @param reader Reader handle
 * @param predicate Filter expression string for row-level filtering (NULL or empty disables filtering)
 * @param batch_size Maximum number of rows per record batch for memory management
 * @param buffer_size Target buffer size in bytes for internal I/O buffering
 * @param out_array_stream Output the ArrowArrayStream (caller must call `out_array_stream->release()`)
 * @return 0 on success, others is error code
 */
FFIResult get_record_batch_reader(ReaderHandle reader,
                                  const char* predicate,
                                  int64_t batch_size,
                                  int64_t buffer_size,
                                  struct ArrowArrayStream* out_array_stream);

/**
 * @brief Get a chunk reader for a specific column group
 *
 * @param reader Reader handle
 * @param column_group_id ID of the column group to read from
 * @param out_handle Output (caller must call `chunk_reader_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle);

/**
 * @brief Extracts specific rows by their global indices with parallel processing
 *
 * Efficiently retrieves rows at the specified global indices from across all
 * column groups in the dataset. This method is optimized for random access
 * patterns and supports parallel I/O for improved performance.
 *
 * @param reader Reader handle
 * @param row_indices Array of global row indices to extract
 * @param num_indices Number of indices in the array
 * @param parallelism Number of threads to use for parallel chunk reading
 * @param out_arrays RecordBatch handle with requested rows (caller must call `out_arrays->release`)
 * @return 0 on success, others is error code
 */
FFIResult take(ReaderHandle reader,
               const int64_t* row_indices,
               size_t num_indices,
               int64_t parallelism,
               struct ArrowArray* out_arrays);

/**
 * @brief Destroys a Reader
 *
 * @param reader Reader handle to destroy
 */
void reader_destroy(ReaderHandle reader);

#endif  // READER_C

#ifdef __cplusplus
}
#endif
