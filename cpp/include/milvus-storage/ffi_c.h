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

// Visibility macro for FFI exports when building Python bindings
#if defined(__GNUC__) || defined(__clang__)
#define FFI_EXPORT __attribute__((visibility("default")))
#else
#define FFI_EXPORT
#endif

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
FFI_EXPORT int IsSuccess(FFIResult* result);

// get the error message, return NULL if success
FFI_EXPORT const char* GetErrorMessage(FFIResult* result);

// free the message string inside FFIResult
FFI_EXPORT void FreeFFIResult(FFIResult* result);

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
FFI_EXPORT FFIResult properties_create(const char* const* keys,
                                       const char* const* values,
                                       size_t count,
                                       Properties* properties);

/**
 * @brief Gets a property value by key
 *
 * @param properties Properties to search
 * @param key Property key to find
 * @return Property value if found, NULL otherwise (do not free)
 */
FFI_EXPORT const char* properties_get(const Properties* properties, const char* key);

/**
 * @brief Frees memory allocated for Properties
 *
 * @param properties Properties to free
 */
FFI_EXPORT void properties_free(Properties* properties);

// ==================== End of Properties C Interface ====================

// ==================== ColumnGroups C Interface ====================
typedef struct CColumnGroupFile {
  const char* path;
  int64_t start_index;
  int64_t end_index;

  // producer-specific data
  uint8_t* metadata;
  uint64_t metadata_size;
} CColumnGroupFile;

typedef struct CColumnGroup {
  const char** columns;
  uint32_t num_of_columns;
  const char* format;

  CColumnGroupFile* files;
  uint32_t num_of_files;
} CColumnGroup;

typedef struct CColumnGroups {
  CColumnGroup* column_group_array;
  uint32_t num_of_column_groups;
} CColumnGroups;

/**
 * @brief C structure representing delta logs
 */
typedef struct DeltaLogs {
  const char** delta_log_paths;
  uint32_t* delta_log_num_entries;
  uint32_t num_delta_logs;
} DeltaLogs;

/**
 * @brief C structure representing stats
 */
typedef struct StatsLog {
  const char** stat_keys;
  const char*** stat_files;
  uint32_t* stat_file_counts;
  uint32_t num_stats;
} StatsLog;

/**
 * @brief C structure representing a Manifest
 */
typedef struct CManifest {
  // Embedded ColumnGroups
  CColumnGroups column_groups;

  // Delta logs (PRIMARY_KEY type only)
  DeltaLogs delta_logs;

  // Stats
  StatsLog stats;
} CManifest;

/**
 * @brief Destroys a CManifest and frees all allocated memory
 *
 * @param manifest CManifest to destroy (can be null)
 */
FFI_EXPORT void manifest_destroy(CManifest* manifest);

/**
 * @brief Generate column groups from external files
 *
 * @param columns Array of column names
 * @param col_lens Number of columns
 * @param format Format of the files
 * @param paths Array of file paths
 * @param start_indices Array of start indices
 * @param end_indices Array of end indices
 * @param file_lens Number of files
 * @param out_column_groups Output parameter for generated CColumnGroups (function allocates and returns pointer)
 *                          Caller must call column_groups_destroy to free allocated memory
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult column_groups_create(const char** columns,
                                          size_t col_lens,
                                          char* format,
                                          char** paths,
                                          int64_t* start_indices,
                                          int64_t* end_indices,
                                          size_t file_lens,
                                          CColumnGroups** out_column_groups);

/**
 * @brief Destroys a CColumnGroups and frees all allocated memory
 *
 * @param cgroups CColumnGroups to destroy (can be null)
 */
FFI_EXPORT void column_groups_destroy(CColumnGroups* cgroups);

// ==================== End of ColumnGroups C Interface ====================

// ==================== ThreadPool C Interface ====================
/**
 * @brief Initialize the thread pool singleton
 *        If current singleton thread pool is not null, update the number of threads
 *        Otherwise, create a new thread pool singleton
 *
 * @param num_of_thread Number of threads in the thread pool
 */
FFI_EXPORT FFIResult thread_pool_singleton(size_t num_of_thread);

/**
 * @brief Release the thread pool singleton
 *        If current singleton thread pool is not null, waiting
 *        all threads join and release the thread pool singleton
 */
FFI_EXPORT void thread_pool_singleton_release();

// ==================== End of ThreadPool C Interface ====================

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
FFI_EXPORT FFIResult writer_new(const char* base_path,
                                struct ArrowSchema* schema,
                                const Properties* properties,
                                WriterHandle* out_handle);

/**
 * @brief Writes a record batch to the dataset
 * @param handle Writer handle
 * @param array Arrow array representing the record batch to write
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult writer_write(WriterHandle handle, struct ArrowArray* array);

/**
 * @brief Flushes the buffer to the storage
 * @param handle Writer handle
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult writer_flush(WriterHandle handle);

/**
 * @brief Closes the writer and returns the columngroups
 * @param handle Writer handle
 * @param out_columngroups Output CColumnGroups structure (function allocates and returns pointer)
 *                         Caller must call column_groups_destroy to free allocated memory
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult writer_close(
    WriterHandle handle, char** meta_keys, char** meta_vals, uint16_t meta_len, CColumnGroups** out_columngroups);

/**
 * @brief Destroys a Writer
 *
 * @param handle Writer handle to destroy
 */
FFI_EXPORT void writer_destroy(WriterHandle handle);

/**
 * @brief Frees a column groups buffer allocated by writer_close
 *
 * @param c_str buffer to free
 */
FFI_EXPORT void free_cstr(char* c_str);

// ==================== End of Writer C Interface ====================

// ==================== ChunkReader C Interface ====================

/// Opaque handle for ChunkReader
typedef uintptr_t ChunkReaderHandle;

// Metadata type flags(maximum 32 bits)
#define LOON_CHUNK_METADATA_ESTIMATED_MEMORY 0x01
#define LOON_CHUNK_METADATA_NUMOFROWS 0x02
#define LOON_CHUNK_METADATA_ALL (LOON_CHUNK_METADATA_ESTIMATED_MEMORY | LOON_CHUNK_METADATA_NUMOFROWS)

// Chunk metadata struct
typedef struct ChunkMetadata {
  // metadata type, 32bits is enough
  uint32_t metadata_type;

  /* clang-format off */
  // metadata content
  union result_u {
    uint64_t estimated_memsz;
    uint64_t number_of_rows;
  } *data;
  /* clang-format on */

  uint64_t number_of_chunks;
} ChunkMetadata;

typedef struct ChunkMetadatas {
  ChunkMetadata* metadatas;
  uint8_t metadatas_size;
} ChunkMetadatas;

/**
 * @brief Get the total number of chunks in the ChunkReader
 *
 * @param chunk_reader ChunkReader handle
 * @param out_number_of_chunks Output total number of chunks
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult get_number_of_chunks(ChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks);

/**
 * @brief Get chunk metadata for a specific column group
 *
 * @param reader Reader handle
 * @param out_chunk_metadata Output chunk metadata (caller must call `free_chunk_metadata` to free)
 */
FFI_EXPORT FFIResult get_chunk_metadatas(ChunkReaderHandle reader,
                                         uint32_t metadata_type,
                                         ChunkMetadatas* out_chunk_metadata);

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
FFI_EXPORT FFIResult get_chunk_indices(ChunkReaderHandle reader,
                                       const int64_t* row_indices,
                                       size_t num_indices,
                                       int64_t** chunk_indices,
                                       size_t* num_chunk_indices);

/**
 * @brief Frees a chunk indices array allocated by get_chunk_indices
 *
 * @param chunk_indices Chunk indices array to free
 */
FFI_EXPORT void free_chunk_indices(int64_t* chunk_indices);

/**
 * @brief Retrieves a single chunk by its index
 *
 * @param reader ChunkReader handle
 * @param chunk_index Zero-based index of the chunk to retrieve
 * @param array Output array of RecordBatch (caller must free)
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, struct ArrowArray* out_array);

/**
 * @brief Retrieves multiple chunks by their indices with optional parallel processing
 *
 * @param reader ChunkReader handle
 * @param chunk_indices Array of chunk indices to retrieve
 * @param num_indices Number of indices in the array
 * @param parallelism Number of parallel threads to use for I/O
 * @param arrays Output array of RecordBatch handles (caller must free)
 * @param num_arrays Output number of record batches
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult get_chunks(ChunkReaderHandle reader,
                                const int64_t* chunk_indices,
                                size_t num_indices,
                                size_t parallelism,
                                struct ArrowArray** arrays,
                                size_t* num_arrays);

/**
 * @brief Frees an array of ArrowArray allocated by get_chunks
 *
 * @param arrays Array of ArrowArray to free
 * @param num_arrays Number of arrays in the array
 */
FFI_EXPORT void free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);

/**
 * @brief Frees a ChunkMetadatas allocated by `get_chunk_metadata`
 *
 * @param chunk_metadata ChunkMetadatas to free
 */
FFI_EXPORT void free_chunk_metadatas(ChunkMetadatas* chunk_metadata);

/**
 * @brief Destroys a ChunkReader
 *
 * @param reader ChunkReader handle to destroy
 */
FFI_EXPORT void chunk_reader_destroy(ChunkReaderHandle reader);

// ==================== Reader C Interface ====================

/// Opaque handle for Reader
typedef uintptr_t ReaderHandle;

/**
 * @brief Creates a new Reader for a milvus storage dataset
 *
 * @param fs Filesystem interface handle
 * @param columngroups Dataset column groups handle
 * @param schema Arrow schema handle
 * @param needed_columns Array of column names to read (NULL for all columns)
 * @param num_columns Number of columns in needed_columns array
 * @param thread_pool Thread pool, must be point of folly::ThreadPoolExecutor
 * @param properties Read configuration properties
 * @param out_handle Output (caller must call `reader_destroy` to destory the handle)
 * @return 0 on success, others is error code
 *
 * @note thread_pool must be point of folly::ThreadPoolExecutor,
 *       If caller do pass a invalid memory or valid memory with
 *       invalid vtable ptr, it will cause the crash. Also caller
 *       must ensure the memory of thread_pool is valid during the
 *       lifetime of reader.
 */
FFI_EXPORT FFIResult reader_new(const CColumnGroups* column_groups,
                                struct ArrowSchema* schema,
                                const char* const* needed_columns,
                                size_t num_columns,
                                const Properties* properties,
                                ReaderHandle* out_handle);

/**
 * @brief Sets a key retriever callback for dynamic key retrieval
 * use to the KMS(key management system) integration
 */
FFI_EXPORT void reader_set_keyretriever(ReaderHandle reader, const char* (*key_retriever)(const char* metadata));

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
FFI_EXPORT FFIResult get_record_batch_reader(ReaderHandle reader,
                                             const char* predicate,
                                             struct ArrowArrayStream* out_array_stream);

/**
 * @brief Get a chunk reader for a specific column group.
 *        The chunk reader is opened after call this function,
 *        means the file FOOTER HAS BEEN READ!
 *
 * @param reader Reader handle
 * @param column_group_id ID of the column group to read from
 * @param out_handle Output (caller must call `chunk_reader_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle);

/**
 * @brief Extracts specific rows by their global indices with parallel processing
 *
 * Efficiently retrieves rows at the specified global indices from across all
 * column groups in the dataset. This method is optimized for random access
 * patterns and supports parallel I/O for improved performance.
 *
 * @param reader Reader handle
 * @param row_indices Array of global row indices to extract, MUST be uniqued and sorted
 * @param num_indices Number of indices in the array
 * @param parallelism Number of parallel threads to use for I/O
 * @param out_arrays Output array of RecordBatch handles (caller must call `free_chunk_arrays` to free)
 * @param num_arrays Number of record batches in the output array
 * @return 0 on success, others is error code
 */
FFI_EXPORT FFIResult take(ReaderHandle reader,
                          const int64_t* row_indices,
                          size_t num_indices,
                          size_t parallelism,
                          struct ArrowArray** out_arrays,
                          size_t* num_arrays);

/**
 * @brief Destroys a Reader
 *
 * @param reader Reader handle to destroy
 */
FFI_EXPORT void reader_destroy(ReaderHandle reader);

// ==================== End of Reader C Interface ====================

// ==================== Manifest C Interface ====================
typedef uintptr_t TransactionHandle;

#define LOON_TRANSACTION_RESOLVE_FAIL 0
#define LOON_TRANSACTION_RESOLVE_MERGE 1
#define LOON_TRANSACTION_RESOLVE_OVERWRITE 2

/**
 * @brief Opens a transaction at the specified base path
 *
 * @param base_path Base path in the filesystem for the transaction
 * @param properties configuration properties
 * @param read_version Version to read (<0 means fetch greatest version)
 * @param retry_limit Maximum number of retry attempts on commit conflicts (default: 1)
 * @param out_handle Output transaction handle
 * @return result of FFI
 */
FFIResult transaction_begin(const char* base_path,
                            const Properties* properties,
                            int64_t read_version,
                            uint32_t retry_limit,
                            TransactionHandle* out_handle);

/**
 * @brief get the manifest of the transaction
 *
 * @param handle Transaction handle
 * @param out_manifest Output CManifest structure (function allocates and returns pointer)
 *                     Caller must call manifest_destroy to free allocated memory
 * @return result of FFI
 */
FFIResult transaction_get_manifest(TransactionHandle handle, CManifest** out_manifest);

/**
 * @brief Get the read version of the transaction
 *
 * @param handle Transaction handle
 * @param out_read_version Output read version number
 * @return result of FFI
 */
FFIResult transaction_get_read_version(TransactionHandle handle, int64_t* out_read_version);

/**
 * @brief Commits the transaction with the provided manifest
 *
 * @param handle Transaction handle
 * @param resolve_id The resolve strategy, more info see LOON_TRANSACTION_RESOLVE_*
 * @param in_manifest The new manifest handle need updated
 *                    Input NULL if current transaction have not any write operation
 * @param out_committed_version Output committed version (valid only if commit succeeds)
 * @return result of FFI
 */
FFIResult transaction_commit(TransactionHandle handle, int64_t* out_committed_version);

/**
 * @brief Destroys a Transaction
 *
 * @param handle Transaction handle to destroy
 */
void transaction_destroy(TransactionHandle handle);

/**
 * @brief Add a new column group to the transaction updates
 *
 * @param handle Transaction handle
 * @param column_group CColumnGroup structure to add
 * @return result of FFI
 */
FFIResult transaction_add_column_group(TransactionHandle handle, const CColumnGroup* column_group);

/**
 * @brief Append files to existing column groups in the transaction updates
 *
 * @param handle Transaction handle
 * @param column_groups CColumnGroups structure containing files to append
 * @return result of FFI
 */
FFIResult transaction_append_files(TransactionHandle handle, const CColumnGroups* column_groups);

/**
 * @brief Add a delta log to the transaction updates
 *
 * @param handle Transaction handle
 * @param path Relative path to the delta log file
 * @param num_entries Number of entries in the delta log
 * @return result of FFI
 * @note Type is hardcoded to PRIMARY_KEY internally
 */
FFIResult transaction_add_delta_log(TransactionHandle handle, const char* path, int64_t num_entries);

/**
 * @brief Add a stat entry to the transaction updates
 *
 * @param handle Transaction handle
 * @param key Stat key (e.g., "pk.delete", "bloomfilter", "bm25")
 * @param files Array of file paths for this stat
 * @param files_len Number of files in the array
 * @return result of FFI
 */
FFIResult transaction_update_stat(TransactionHandle handle,
                                  const char* key,
                                  const char* const* files,
                                  size_t files_len);

/**
 * @brief Cleans the global filesystem cache
 *
 * This function clears the LRUCache used for storing ArrowFileSystem instances.
 * Useful for testing or when resetting the environment.
 */
FFI_EXPORT void close_filesystems();

// ==================== End of Manifest C Interface ====================

#endif  // LOON_FFI_C

#ifdef __cplusplus
}
#endif
