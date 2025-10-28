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

#ifndef LOON_FFI_C
#define LOON_FFI_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <arrow/c/abi.h>

// ==================== Result C Interface ====================
#define LOON_SUCCESS 0
#define LOON_INVALID_ARGS 1
#define LOON_MEMORY_ERROR 2
#define LOON_ARROW_ERROR 3
#define LOON_LOGICAL_ERROR 4
#define LOON_GOT_EXCEPTION 5
#define LOON_UNREACHABLE_ERROR 6
#define LOON_INVALID_PROPERTIES 7
#define LOON_ERRORCODE_MAX 8

// usage example(caller must free the message string):
//
// FFIResult result = SomeFFIFunction(...);
// if (!IsSuccess(&result)) {
//    printf("Error: %s\n", GetErrorMessage(&result));
//    ... // handle error, e.g. log result.message
//    FreeFFIResult(&result); // free the message string
// }
typedef struct ffi_result {
  int err_code;
  char* message;
} FFIResult;

// check result is success
int IsSuccess(FFIResult* result);

// get the error message, return NULL if success
const char* GetErrorMessage(FFIResult* result);

// free the message string inside FFIResult
void FreeFFIResult(FFIResult* result);

// ==================== End of Result C Interface ====================

// ==================== Properties C Interface ====================

/// C struct for a single property key-value pair
typedef struct Property {
  char* key;    ///< Property key (caller owns memory)
  char* value;  ///< Property value (caller owns memory)
} Property;

/// C struct for read properties (array of key-value pairs)
typedef struct Properties {
  Property* properties;
  size_t count;
} Properties;

/**
 * @brief Creates read properties from key-value arrays
 *
 * @param keys Array of property keys
 * @param values Array of property values
 * @param count Number of key-value pairs
 * @param properties Output parameter for created properties (caller must free)
 */
FFIResult properties_create(const char* const* keys, const char* const* values, size_t count, Properties* properties);

/**
 * @brief Gets a property value by key
 *
 * @param properties Properties to search
 * @param key Property key to find
 * @return Property value if found, NULL otherwise (do not free)
 */
const char* properties_get(const Properties* properties, const char* key);

/**
 * @brief Frees memory allocated for Properties
 *
 * @param properties Properties to free
 */
void properties_free(Properties* properties);

// ==================== End of Properties C Interface ====================

// ==================== Writer C Interface ====================
typedef uintptr_t WriterHandle;

/**
 * @brief Creates a new Writer for a milvus storage dataset
 *
 * @param base_path Base path in the filesystem to write data
 * @param schema Arrow schema handle
 * @param properties configuration properties
 * @param out_handle Output (caller must call `writer_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFIResult writer_new(const char* base_path,
                     struct ArrowSchema* schema,
                     const Properties* properties,
                     WriterHandle* out_handle);

/**
 * @brief Writes a record batch to the dataset
 * @param handle Writer handle
 * @param array Arrow array representing the record batch to write
 * @return 0 on success, others is error code
 */
FFIResult writer_write(WriterHandle handle, struct ArrowArray* array);

/**
 * @brief Flushes the buffer to the storage
 * @param handle Writer handle
 * @return 0 on success, others is error code
 */
FFIResult writer_flush(WriterHandle handle);

/**
 * @brief Closes the writer and returns the manifest
 * @param handle Writer handle
 * @param out_manifest Output manifest JSON buffer (caller must call `free_manifest` to free)
 * @param out_manifest_size Size of the output manifest string
 * @return 0 on success, others is error code
 */
FFIResult writer_close(WriterHandle handle, char** out_manifest, size_t* out_manifest_size);

/**
 * @brief Destroys a Writer
 *
 * @param handle Writer handle to destroy
 */
void writer_destroy(WriterHandle handle);

/**
 * @brief Frees a manifest buffer allocated by writer_close
 *
 * @param manifest Manifest buffer to free
 */
void free_manifest(char* manifest);

// ==================== End of Writer C Interface ====================

// ==================== ChunkReader C Interface ====================

/// Opaque handle for ChunkReader
typedef uintptr_t ChunkReaderHandle;

/**
 * @brief Maps row indices to their corresponding chunk indices
 *
 * @param reader ChunkReader handle
 * @param row_indices Array of global row indices to map
 * @param num_indices Number of indices in the array
 * @param chunk_indices Output array of chunk indices (caller must call `free_chunk_indices` to free)
 * @param num_chunk_indices Output number of chunk indices
 * @return 0 on success, others is error code
 */
FFIResult get_chunk_indices(ChunkReaderHandle reader,
                            const int64_t* row_indices,
                            size_t num_indices,
                            int64_t** chunk_indices,
                            size_t* num_chunk_indices);

/**
 * @brief Frees a chunk indices array allocated by get_chunk_indices
 *
 * @param chunk_indices Chunk indices array to free
 */
void free_chunk_indices(int64_t* chunk_indices);

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
 * @brief Frees an array of ArrowArray allocated by get_chunks
 *
 * @param arrays Array of ArrowArray to free
 * @param num_arrays Number of arrays in the array
 */
void free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);

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
                     const Properties* properties,
                     ReaderHandle* out_handle);

/**
 *
 */
void reader_set_keyretriever(ReaderHandle reader, const char* (*key_retriever)(const char* metadata));

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

// ==================== End of Reader C Interface ====================

#endif  // LOON_FFI_C

#ifdef __cplusplus
}
#endif
