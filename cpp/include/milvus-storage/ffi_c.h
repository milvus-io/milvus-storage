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
#include "arrow/c/abi.h"

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
// LoonFFIResult result = SomeFFIFunction(...);
// if (!loon_ffi_is_success(&result)) {
//    printf("Error: %s\n", loon_ffi_get_errmsg(&result));
//    ... // handle error, e.g. log result.message
//    loon_ffi_free_result(&result); // free the message string
// }
typedef struct LoonFFIResult {
  int err_code;
  char* message;
} LoonFFIResult;

// check result is success
FFI_EXPORT int loon_ffi_is_success(LoonFFIResult* result);

// get the error message, return NULL if success
FFI_EXPORT const char* loon_ffi_get_errmsg(LoonFFIResult* result);

// free the message string inside LoonFFIResult
FFI_EXPORT void loon_ffi_free_result(LoonFFIResult* result);

// ==================== End of Result C Interface ====================

// ==================== Properties C Interface ====================

// --- Global property definitions ---
extern const char* loon_properties_format;

// --- Export FS property keys ---
extern const char* loon_properties_fs_address;
extern const char* loon_properties_fs_bucket_name;
extern const char* loon_properties_fs_access_key_id;
extern const char* loon_properties_fs_access_key_value;
extern const char* loon_properties_fs_root_path;
extern const char* loon_properties_fs_storage_type;
extern const char* loon_properties_fs_cloud_provider;
extern const char* loon_properties_fs_iam_endpoint;
extern const char* loon_properties_fs_log_level;
extern const char* loon_properties_fs_region;
extern const char* loon_properties_fs_use_ssl;
extern const char* loon_properties_fs_ssl_ca_cert;
extern const char* loon_properties_fs_use_iam;
extern const char* loon_properties_fs_use_virtual_host;
extern const char* loon_properties_fs_request_timeout_ms;
extern const char* loon_properties_fs_gcp_native_without_auth;
extern const char* loon_properties_fs_gcp_credential_json;
extern const char* loon_properties_fs_use_custom_part_upload;
extern const char* loon_properties_fs_max_connections;
extern const char* loon_properties_fs_multi_part_upload_size;

// --- Export Writer property keys ---
extern const char* loon_properties_writer_policy;
extern const char* loon_properties_writer_schema_base_patterns;
extern const char* loon_properties_writer_size_base_macs;
extern const char* loon_properties_writer_size_base_mcig;
extern const char* loon_properties_writer_buffer_size;
extern const char* loon_properties_writer_file_rolling_size;
extern const char* loon_properties_writer_compression;
extern const char* loon_properties_writer_compression_level;
extern const char* loon_properties_writer_enable_dictionary;
extern const char* loon_properties_writer_enc_enable;
extern const char* loon_properties_writer_enc_key;
extern const char* loon_properties_writer_enc_meta;
extern const char* loon_properties_writer_enc_algorithm;
extern const char* loon_properties_writer_vortex_enable_statistics;

// --- Export Reader property keys ---
extern const char* loon_properties_reader_record_batch_max_rows;
extern const char* loon_properties_reader_record_batch_max_size;
extern const char* loon_properties_reader_vortex_chunk_rows;

// --- Export Transaction property keys ---
extern const char* loon_properties_transaction_commit_num_retries;

/// C struct for a single property key-value pair
typedef struct LoonProperty {
  char* key;    ///< Property key (caller owns memory)
  char* value;  ///< Property value (caller owns memory)
} LoonProperty;

/// C struct for read properties (array of key-value pairs)
typedef struct LoonProperties {
  LoonProperty* properties;
  size_t count;
} LoonProperties;

/**
 * @brief Creates read properties from key-value arrays
 *
 * @param keys Array of property keys
 * @param values Array of property values
 * @param count Number of key-value pairs
 * @param properties Output parameter for created properties (caller must free)
 */
FFI_EXPORT LoonFFIResult loon_properties_create(const char* const* keys,
                                                const char* const* values,
                                                size_t count,
                                                LoonProperties* properties);

/**
 * @brief Gets a property value by key
 *
 * @param properties Properties to search
 * @param key Property key to find
 * @return Property value if found, NULL otherwise (do not free)
 */
FFI_EXPORT const char* loon_properties_get(const LoonProperties* properties, const char* key);

/**
 * @brief Frees memory allocated for Properties
 *
 * @param properties Properties to free
 */
FFI_EXPORT void loon_properties_free(LoonProperties* properties);

// ==================== End of Properties C Interface ====================

// ==================== ColumnGroups C Interface ====================
typedef struct LoonColumnGroupFile {
  const char* path;
  int64_t start_index;
  int64_t end_index;

  // producer-specific data
  uint8_t* metadata;
  uint64_t metadata_size;
} LoonColumnGroupFile;

typedef struct LoonColumnGroup {
  const char** columns;
  uint32_t num_of_columns;
  const char* format;

  LoonColumnGroupFile* files;
  uint32_t num_of_files;
} LoonColumnGroup;

typedef struct LoonColumnGroups {
  LoonColumnGroup* column_group_array;
  uint32_t num_of_column_groups;
} LoonColumnGroups;

/**
 * @brief C structure representing delta logs
 */
typedef struct LoonDeltaLogs {
  const char** delta_log_paths;
  uint32_t* delta_log_num_entries;
  uint32_t num_delta_logs;
} LoonDeltaLogs;

/**
 * @brief C structure representing stats
 */
typedef struct LoonStatsLog {
  const char** stat_keys;
  const char*** stat_files;
  uint32_t* stat_file_counts;
  uint32_t num_stats;
} LoonStatsLog;

/**
 * @brief C structure representing a Manifest
 */
typedef struct LoonManifest {
  // Embedded ColumnGroups
  LoonColumnGroups column_groups;

  // Delta logs (PRIMARY_KEY type only)
  LoonDeltaLogs delta_logs;

  // Stats
  LoonStatsLog stats;
} LoonManifest;

/**
 * @brief Destroys a CManifest and frees all allocated memory
 *
 * @param manifest CManifest to destroy (can be null)
 */
FFI_EXPORT void loon_manifest_destroy(LoonManifest* manifest);

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
 * @param out_column_groups Output parameter for generated LoonColumnGroups (function allocates and returns pointer)
 *                          Caller must call `loon_column_groups_destroy` to free allocated memory
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_column_groups_create(const char** columns,
                                                   size_t col_lens,
                                                   char* format,
                                                   char** paths,
                                                   int64_t* start_indices,
                                                   int64_t* end_indices,
                                                   size_t file_lens,
                                                   LoonColumnGroups** out_column_groups);

/**
 * @brief Destroys a LoonColumnGroups and frees all allocated memory
 *
 * @param cgroups LoonColumnGroups to destroy (can be null)
 */
FFI_EXPORT void loon_column_groups_destroy(LoonColumnGroups* cgroups);

// ==================== End of ColumnGroups C Interface ====================

// ==================== ThreadPool C Interface ====================
/**
 * @brief Initialize the thread pool singleton
 *        If current singleton thread pool is not null, update the number of threads
 *        Otherwise, create a new thread pool singleton
 *
 * @param num_of_thread Number of threads in the thread pool
 */
FFI_EXPORT LoonFFIResult loon_thread_pool_singleton(size_t num_of_thread);

/**
 * @brief Release the thread pool singleton
 *        If current singleton thread pool is not null, waiting
 *        all threads join and release the thread pool singleton
 */
FFI_EXPORT void loon_thread_pool_singleton_release();

// ==================== End of ThreadPool C Interface ====================

// ==================== Writer C Interface ====================
typedef uintptr_t LoonWriterHandle;

/**
 * @brief Creates a new Writer for a milvus storage dataset
 *
 * @param base_path Base path in the filesystem to write data
 * @param schema Arrow schema handle
 * @param properties configuration properties
 * @param out_handle Output (caller must call `writer_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_writer_new(const char* base_path,
                                         struct ArrowSchema* schema,
                                         const LoonProperties* properties,
                                         LoonWriterHandle* out_handle);

/**
 * @brief Writes a record batch to the dataset
 * @param handle Writer handle
 * @param array Arrow array representing the record batch to write
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_writer_write(LoonWriterHandle handle, struct ArrowArray* array);

/**
 * @brief Flushes the buffer to the storage
 * @param handle Writer handle
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_writer_flush(LoonWriterHandle handle);

/**
 * @brief Closes the writer and returns the columngroups
 * @param handle Writer handle
 * @param out_columngroups Output LoonColumnGroups structure (function allocates and returns pointer)
 *                         Caller must call `loon_column_groups_destroy` to free allocated memory
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_writer_close(LoonWriterHandle handle,
                                           char** meta_keys,
                                           char** meta_vals,
                                           uint16_t meta_len,
                                           LoonColumnGroups** out_columngroups);

/**
 * @brief Destroys a Writer
 *
 * @param handle Writer handle to destroy
 */
FFI_EXPORT void loon_writer_destroy(LoonWriterHandle handle);

/**
 * @brief Frees a column groups buffer allocated by writer_close
 *
 * @param c_str buffer to free
 */
FFI_EXPORT void loon_free_cstr(char* c_str);

// ==================== End of Writer C Interface ====================

// ==================== ChunkReader C Interface ====================

/// Opaque handle for ChunkReader
typedef uintptr_t LoonChunkReaderHandle;

// Metadata type flags(maximum 32 bits)
#define LOON_CHUNK_METADATA_ESTIMATED_MEMORY 0x01
#define LOON_CHUNK_METADATA_NUMOFROWS 0x02
#define LOON_CHUNK_METADATA_ALL (LOON_CHUNK_METADATA_ESTIMATED_MEMORY | LOON_CHUNK_METADATA_NUMOFROWS)

// Chunk metadata struct
typedef struct LoonChunkMetadata {
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
} LoonChunkMetadata;

typedef struct LoonChunkMetadatas {
  LoonChunkMetadata* metadatas;
  uint8_t metadatas_size;
} LoonChunkMetadatas;

/**
 * @brief Get the total number of chunks in the ChunkReader
 *
 * @param chunk_reader ChunkReader handle
 * @param out_number_of_chunks Output total number of chunks
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_get_number_of_chunks(LoonChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks);

/**
 * @brief Get chunk metadata for a specific column group
 *
 * @param reader Reader handle
 * @param out_chunk_metadata Output chunk metadata (caller must call `free_chunk_metadata` to free)
 */
FFI_EXPORT LoonFFIResult loon_get_chunk_metadatas(LoonChunkReaderHandle reader,
                                                  uint32_t metadata_type,
                                                  LoonChunkMetadatas* out_chunk_metadata);

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
FFI_EXPORT LoonFFIResult loon_get_chunk_indices(LoonChunkReaderHandle reader,
                                                const int64_t* row_indices,
                                                size_t num_indices,
                                                int64_t** chunk_indices,
                                                size_t* num_chunk_indices);

/**
 * @brief Frees a chunk indices array allocated by get_chunk_indices
 *
 * @param chunk_indices Chunk indices array to free
 */
FFI_EXPORT void loon_free_chunk_indices(int64_t* chunk_indices);

/**
 * @brief Retrieves a single chunk by its index
 *
 * @param reader ChunkReader handle
 * @param chunk_index Zero-based index of the chunk to retrieve
 * @param array Output array of RecordBatch (caller must free)
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_get_chunk(LoonChunkReaderHandle reader,
                                        int64_t chunk_index,
                                        struct ArrowArray* out_array);

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
FFI_EXPORT LoonFFIResult loon_get_chunks(LoonChunkReaderHandle reader,
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
FFI_EXPORT void loon_free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);

/**
 * @brief Frees a ChunkMetadatas allocated by `loon_get_chunk_metadatas`
 *
 * @param chunk_metadata ChunkMetadatas to free
 */
FFI_EXPORT void loon_free_chunk_metadatas(LoonChunkMetadatas* chunk_metadata);

/**
 * @brief Destroys a ChunkReader
 *
 * @param reader ChunkReader handle to destroy
 */
FFI_EXPORT void loon_chunk_reader_destroy(LoonChunkReaderHandle reader);

// ==================== Reader C Interface ====================

/// Opaque handle for Reader
typedef uintptr_t LoonReaderHandle;

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
FFI_EXPORT LoonFFIResult loon_reader_new(const LoonColumnGroups* column_groups,
                                         struct ArrowSchema* schema,
                                         const char* const* needed_columns,
                                         size_t num_columns,
                                         const LoonProperties* properties,
                                         LoonReaderHandle* out_handle);

/**
 * @brief Sets a key retriever callback for dynamic key retrieval
 * use to the KMS(key management system) integration
 */
FFI_EXPORT void loon_reader_set_keyretriever(LoonReaderHandle reader,
                                             const char* (*key_retriever)(const char* metadata));

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
FFI_EXPORT LoonFFIResult loon_get_record_batch_reader(LoonReaderHandle reader,
                                                      const char* predicate,
                                                      struct ArrowArrayStream* out_array_stream);

/**
 * @brief Get a chunk reader for a specific column group.
 *        The chunk reader is opened after call this function,
 *        means the file FOOTER HAS BEEN READ!
 *
 * @param reader Reader handle
 * @param column_group_id ID of the column group to read from
 * @param out_handle Output (caller must call `loon_chunk_reader_destroy` to destory the handle)
 * @return 0 on success, others is error code
 */
FFI_EXPORT LoonFFIResult loon_get_chunk_reader(LoonReaderHandle reader,
                                               int64_t column_group_id,
                                               LoonChunkReaderHandle* out_handle);

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
FFI_EXPORT LoonFFIResult loon_take(LoonReaderHandle reader,
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
FFI_EXPORT void loon_reader_destroy(LoonReaderHandle reader);

// ==================== End of Reader C Interface ====================

// ==================== Manifest C Interface ====================
typedef uintptr_t LoonTransactionHandle;

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
FFI_EXPORT LoonFFIResult loon_transaction_begin(const char* base_path,
                                                const LoonProperties* properties,
                                                int64_t read_version,
                                                uint32_t retry_limit,
                                                LoonTransactionHandle* out_handle);

/**
 * @brief get the manifest of the transaction
 *
 * @param handle Transaction handle
 * @param out_manifest Output CManifest structure (function allocates and returns pointer)
 *                     Caller must call manifest_destroy to free allocated memory
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_transaction_get_manifest(LoonTransactionHandle handle, LoonManifest** out_manifest);

/**
 * @brief Get the read version of the transaction
 *
 * @param handle Transaction handle
 * @param out_read_version Output read version number
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_transaction_get_read_version(LoonTransactionHandle handle, int64_t* out_read_version);

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
FFI_EXPORT LoonFFIResult loon_transaction_commit(LoonTransactionHandle handle, int64_t* out_committed_version);

/**
 * @brief Destroys a Transaction
 *
 * @param handle Transaction handle to destroy
 */
FFI_EXPORT void loon_transaction_destroy(LoonTransactionHandle handle);

/**
 * @brief Add a new column group to the transaction updates
 *
 * @param handle Transaction handle
 * @param column_group LoonColumnGroup structure to add
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_transaction_add_column_group(LoonTransactionHandle handle,
                                                           const LoonColumnGroup* column_group);

/**
 * @brief Append files to existing column groups in the transaction updates
 *
 * @param handle Transaction handle
 * @param column_groups LoonColumnGroups structure containing files to append
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_transaction_append_files(LoonTransactionHandle handle,
                                                       const LoonColumnGroups* column_groups);

/**
 * @brief Add a delta log to the transaction updates
 *
 * @param handle Transaction handle
 * @param path Relative path to the delta log file
 * @param num_entries Number of entries in the delta log
 * @return result of FFI
 * @note Type is hardcoded to PRIMARY_KEY internally
 */
FFI_EXPORT LoonFFIResult loon_transaction_add_delta_log(LoonTransactionHandle handle,
                                                        const char* path,
                                                        int64_t num_entries);

/**
 * @brief Add a stat entry to the transaction updates
 *
 * @param handle Transaction handle
 * @param key Stat key (e.g., "pk.delete", "bloomfilter", "bm25")
 * @param files Array of file paths for this stat
 * @param files_len Number of files in the array
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_transaction_update_stat(LoonTransactionHandle handle,
                                                      const char* key,
                                                      const char* const* files,
                                                      size_t files_len);

// ==================== End of Manifest C Interface ====================

// ==================== Filesystem C Interface ====================

typedef uintptr_t FileSystemHandle;
typedef uintptr_t FileSystemWriterHandle;
typedef uintptr_t FileSystemReaderHandle;

/**
 * Metadata key-value pair for filesystem operations.
 */
typedef struct LoonFileSystemMeta {
  char* key;
  char* value;
} LoonFileSystemMeta;

/**
 * Get a filesystem from cache or create a new one.
 * This function uses FilesystemCache internally to manage filesystem instances.
 *
 * @param properties The properties of the filesystem.
 * @param path Optional path to determine filesystem (empty = default filesystem).
 *             If path is provided with a scheme (e.g., "s3://bucket/key"), it will
 *             try to resolve external filesystem (extfs.*) configurations.
 * @param path_len The length of the path string (0 if path is NULL or empty).
 * @param out_handle The output filesystem instance handle.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get(const LoonProperties* properties,
                                             const char* path,
                                             uint32_t path_len,
                                             FileSystemHandle* out_handle);

/**
 * @brief Destroy a filesystem handle
 *
 * This function releases the resources associated with a filesystem handle.
 * After calling this function, the handle becomes invalid and should not be used.
 *
 * @param handle The filesystem handle to destroy
 */
FFI_EXPORT void loon_filesystem_destroy(FileSystemHandle handle);

/**
 * @brief Cleans the global filesystem cache
 *
 * This function clears the LRUCache used for storing ArrowFileSystem instances.
 * Useful for testing or when resetting the environment.
 */
FFI_EXPORT void loon_close_filesystems();

/**
 * Open a outputstream for a file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to write.
 * @param path_len The length of the path.
 * @param meta_array The metadata array.
 * @param num_of_meta The number of metadata.
 * @param out_handle The output writer instance.
 *
 * The metadata will be passed into the `OpenOutputStream`.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_open_writer(FileSystemHandle handle,
                                                     const char* path_ptr,
                                                     uint32_t path_len,
                                                     const LoonFileSystemMeta* meta_array,
                                                     uint32_t num_of_meta,
                                                     FileSystemWriterHandle* out_handle);

/**
 * Write data to the outputstream.
 *
 * @param handle The outputstream instance.
 * @param data The data to write.
 * @param size The size of the data.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_writer_write(FileSystemWriterHandle handle,
                                                      const uint8_t* data,
                                                      uint64_t size);

/**
 * Flush the outputstream.
 *
 * @param handle The outputstream instance.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_writer_flush(FileSystemWriterHandle handle);

/**
 * Close the outputstream.
 *
 * @param handle The outputstream instance.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_writer_close(FileSystemWriterHandle handle);

/**
 * Destroy the outputstream.
 *
 * @param handle The outputstream instance.
 */
FFI_EXPORT void loon_filesystem_writer_destroy(FileSystemWriterHandle handle);

/**
 * Get the size of the file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param out_size The size of the file.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_file_info(FileSystemHandle handle,
                                                       const char* path_ptr,
                                                       uint32_t path_len,
                                                       uint64_t* out_size);

/**
 * Get the content of the file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param offset The start position of the file.
 * @param nbytes The number of bytes to read
 * @param out_data The buffer to read bytes into
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_read_file(FileSystemHandle handle,
                                                   const char* path_ptr,
                                                   uint32_t path_len,
                                                   uint64_t offset,
                                                   uint64_t nbytes,
                                                   uint8_t* out_data);

/**
 * Open a inputstream for a file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_open_reader(FileSystemHandle handle,
                                                     const char* path_ptr,
                                                     uint32_t path_len,
                                                     FileSystemReaderHandle* out_reader_ptr);

/**
 * Read data from the inputstream.
 *
 * @param handle The inputstream instance.
 * @param offset The start position of the file.
 * @param nbytes The number of bytes to read
 * @param out_data The buffer to read bytes into.
 *                 The `out_data` must be allocated by caller, the allocated size must GE `nbytes`.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_reader_readat(FileSystemReaderHandle handle,
                                                       uint64_t offset,
                                                       uint64_t nbytes,
                                                       uint8_t* out_data);

/**
 * Close the inputstream.
 *
 * @param handle The inputstream instance.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_reader_close(FileSystemReaderHandle handle);

/**
 * Destroy the inputstream.
 *
 * @param handle The inputstream instance.
 */
FFI_EXPORT void loon_filesystem_reader_destroy(FileSystemReaderHandle handle);

/**
 * Initialize the ArrowFileSystemSingleton.
 *
 * @param properties The properties of the filesystem.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_initialize_filesystem_singleton(const LoonProperties* properties);

/**
 * Get the handle of the filesystem singleton.
 *
 * @param out_handle The output filesystem singleton handle.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_get_filesystem_singleton_handle(FileSystemHandle* out_handle);

/**
 * Get file statistics including size and metadata.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param out_size The output file size.
 * @param out_meta_array The output metadata array (caller must free using loon_filesystem_free_meta_array).
 * @param out_meta_count The output metadata count.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_file_stats(FileSystemHandle handle,
                                                        const char* path_ptr,
                                                        uint32_t path_len,
                                                        uint64_t* out_size,
                                                        LoonFileSystemMeta** out_meta_array,
                                                        uint32_t* out_meta_count);

/**
 * Free metadata array returned by loon_filesystem_get_file_stats.
 *
 * @param meta_array The metadata array to free.
 * @param meta_count The number of metadata entries in the array.
 */
FFI_EXPORT void loon_filesystem_free_meta_array(LoonFileSystemMeta* meta_array, uint32_t meta_count);

/**
 * Read entire file content into memory.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param out_data The output buffer containing file data (caller must free).
 * @param out_size The output size of the data.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_read_file_all(
    FileSystemHandle handle, const char* path_ptr, uint32_t path_len, uint8_t** out_data, uint64_t* out_size);

/**
 * Write data to a file with optional metadata.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to write.
 * @param path_len The length of the path.
 * @param data The data to write.
 * @param data_size The size of the data.
 * @param meta_array The metadata array (optional, can be NULL if meta_count is 0).
 * @param meta_count The number of metadata pairs.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_write_file(FileSystemHandle handle,
                                                    const char* path_ptr,
                                                    uint32_t path_len,
                                                    const uint8_t* data,
                                                    uint64_t data_size,
                                                    const LoonFileSystemMeta* meta_array,
                                                    uint32_t meta_count);

/**
 * Delete a file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to delete.
 * @param path_len The length of the path.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_delete_file(FileSystemHandle handle, const char* path_ptr, uint32_t path_len);

/**
 * Get path information (existence, type, timestamps).
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path to check.
 * @param path_len The length of the path.
 * @param out_exists Output flag indicating if path exists.
 * @param out_is_dir Output flag indicating if path is a directory (can be NULL).
 * @param out_mtime_ns Output modification time in nanoseconds (can be NULL).
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_path_info(FileSystemHandle handle,
                                                       const char* path_ptr,
                                                       uint32_t path_len,
                                                       bool* out_exists,
                                                       bool* out_is_dir,
                                                       int64_t* out_mtime_ns);

/**
 * Create a directory.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the directory to create.
 * @param path_len The length of the path.
 * @param recursive If true, create parent directories as needed.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_create_dir(FileSystemHandle handle,
                                                    const char* path_ptr,
                                                    uint32_t path_len,
                                                    bool recursive);

/**
 * List directory contents.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the directory to list.
 * @param path_len The length of the path.
 * @param recursive If true, list recursively.
 * @param out_paths Output array of paths (caller must free each string and array).
 * @param out_path_lens Output array of path lengths.
 * @param out_is_dirs Output array of directory flags.
 * @param out_sizes Output array of file sizes.
 * @param out_mtime_ns Output array of modification times in nanoseconds.
 * @param out_count Output count of entries.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_list_dir(FileSystemHandle handle,
                                                  const char* path_ptr,
                                                  uint32_t path_len,
                                                  bool recursive,
                                                  char*** out_paths,
                                                  uint32_t** out_path_lens,
                                                  bool** out_is_dirs,
                                                  uint64_t** out_sizes,
                                                  int64_t** out_mtime_ns,
                                                  uint32_t* out_count);

// ==================== End of Filesystem C Interface ====================

// ==================== Filesystem Metrics C Interface ====================

/**
 * C structure representing filesystem metrics snapshot.
 */
typedef struct LoonFilesystemMetricsSnapshot {  // NOLINT
  int64_t read_count;
  int64_t write_count;
  int64_t read_bytes;
  int64_t write_bytes;
  int64_t get_file_info_count;
  int64_t create_dir_count;
  int64_t delete_dir_count;
  int64_t delete_file_count;
  int64_t move_count;
  int64_t copy_file_count;
  int64_t failed_count;
  // S3-specific metrics
  int64_t multi_part_upload_created;
  int64_t multi_part_upload_finished;
} LoonFilesystemMetricsSnapshot;  // NOLINT

/**
 * Get metrics from a filesystem handle.
 * Returns metrics if the filesystem is observable, otherwise returns error.
 *
 * @param handle The filesystem instance handle.
 * @param out_metrics The output metrics snapshot structure (caller allocates).
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_metrics(FileSystemHandle handle,
                                                     LoonFilesystemMetricsSnapshot* out_metrics);

/**
 * Reset all metrics for a filesystem.
 *
 * @param handle The filesystem instance handle.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_reset_metrics(FileSystemHandle handle);

// ==================== End of Filesystem Metrics C Interface ====================

// ==================== External Table C Interface ====================

/**
 * @brief Import external files into a dataset
 *
 * @param columns Array of column names
 * @param col_lens Array of column names sizes
 * @param format The file format type (currently only "parquet" is supported)
 * @param base_dir base directory path
 * @param dir_path directory path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_num_of_files output number of files
 * @param out_column_groups_file_path output column groups file path, need call `loon_free_cstr` to free memory
 * @return LoonFFIResult
 */
FFI_EXPORT LoonFFIResult loon_exttable_explore(const char** columns,
                                               size_t col_lens,
                                               const char* format,
                                               const char* base_dir,
                                               const char* explore_dir,
                                               const LoonProperties* properties,
                                               uint64_t* out_num_of_files,
                                               char** out_column_groups_file_path);

/**
 * @brief Get file info
 *
 * @param format The file format type (currently only "parquet" is supported)
 * @param file_path file path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_num_of_rows output number of rows
 * @param out_schema output schema
 * @return LoonFFIResult
 */
FFI_EXPORT LoonFFIResult loon_exttable_get_file_info(const char* format,
                                                     const char* file_path,
                                                     const LoonProperties* properties,
                                                     uint64_t* out_num_of_rows);

/**
 * @brief Read manifest from file
 *
 * @param manifest_file_path manifest file path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_manifest output manifest (includes column groups, delta logs, and stats)
 * @return LoonFFIResult
 */
FFI_EXPORT LoonFFIResult loon_exttable_read_manifest(const char* manifest_file_path,
                                                     const LoonProperties* properties,
                                                     LoonManifest** out_manifest);

// ==================== End of External Table C Interface ====================

#endif  // LOON_FFI_C

#ifdef __cplusplus
}
#endif
