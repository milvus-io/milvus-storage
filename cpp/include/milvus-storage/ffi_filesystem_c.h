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

#ifndef LOON_FILESYSTEM_C
#define LOON_FILESYSTEM_C

#include <stdbool.h>
#include <stdint.h>

#include "milvus-storage/ffi_c.h"

// ============================================================================
// Filesystem Property Key Accessor Functions
// ============================================================================
// These functions return the property key strings for use in FFI bindings
// where C++ macros may not be accessible. All returned pointers are to static
// strings and do not need to be freed by the caller.

const char* loon_property_fs_storage_type(void);
const char* loon_property_fs_root_path(void);
const char* loon_property_fs_address(void);
const char* loon_property_fs_bucket_name(void);
const char* loon_property_fs_region(void);
const char* loon_property_fs_access_key_id(void);
const char* loon_property_fs_access_key_value(void);
const char* loon_property_fs_use_iam(void);
const char* loon_property_fs_iam_endpoint(void);
const char* loon_property_fs_gcp_native_without_auth(void);
const char* loon_property_fs_gcp_credential_json(void);
const char* loon_property_fs_use_ssl(void);
const char* loon_property_fs_ssl_ca_cert(void);
const char* loon_property_fs_use_virtual_host(void);
const char* loon_property_fs_request_timeout_ms(void);
const char* loon_property_fs_max_connections(void);
const char* loon_property_fs_use_custom_part_upload(void);
const char* loon_property_fs_multi_part_upload_size(void);
const char* loon_property_fs_log_level(void);
const char* loon_property_fs_cloud_provider(void);

typedef uintptr_t FileSystemHandle;
typedef uintptr_t FileSystemWriterHandle;
typedef uintptr_t FileSystemReaderHandle;

/**
 * Create a filesystem instance.
 *
 * @param properties The properties of the filesystem.
 * @param out_handle The output filesystem instance.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_create(const LoonProperties* properties, FileSystemHandle* out_handle);

/**
 * Destroy a filesystem instance.
 *
 * @param handle The filesystem instance to destroy.
 */
void loon_filesystem_destroy(FileSystemHandle handle);

/**
 * Open a outputstream for a file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to write.
 * @param path_len The length of the path.
 * @param meta_keys The metadata keys.
 * @param meta_values The metadata values.
 * @param num_of_meta The number of metadata.
 * @param out_handle The output writer instance.
 *
 * The metadata will be passed into the `OpenOutputStream`.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_open_writer(FileSystemHandle handle,
                                          const char* path_ptr,
                                          uint32_t path_len,
                                          const char** meta_keys,
                                          const char** meta_values,
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
LoonFFIResult loon_filesystem_writer_write(FileSystemWriterHandle handle, const uint8_t* data, uint64_t size);

/**
 * Flush the outputstream.
 *
 * @param handle The outputstream instance.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_writer_flush(FileSystemWriterHandle handle);

/**
 * Close the outputstream.
 *
 * @param handle The outputstream instance.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_writer_close(FileSystemWriterHandle handle);

/**
 * Destroy the outputstream.
 *
 * @param handle The outputstream instance.
 */
void loon_filesystem_writer_destroy(FileSystemWriterHandle handle);

/**
 * Get the size of the file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param out_size The size of the file.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_get_file_info(FileSystemHandle handle,
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
LoonFFIResult loon_filesystem_read_file(FileSystemHandle handle,
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
LoonFFIResult loon_filesystem_open_reader(FileSystemHandle handle,
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
LoonFFIResult loon_filesystem_reader_readat(FileSystemReaderHandle handle,
                                            uint64_t offset,
                                            uint64_t nbytes,
                                            uint8_t* out_data);

/**
 * Close the inputstream.
 *
 * @param handle The inputstream instance.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_reader_close(FileSystemReaderHandle handle);

/**
 * Destroy the inputstream.
 *
 * @param handle The inputstream instance.
 */
void loon_filesystem_reader_destroy(FileSystemReaderHandle handle);

/**
 * Initialize the ArrowFileSystemSingleton.
 *
 * @param properties The properties of the filesystem.
 * @return result of FFI
 */
LoonFFIResult loon_initialize_filesystem_singleton(const LoonProperties* properties);

/**
 * Get the handle of the filesystem singleton.
 *
 * @param out_handle The output filesystem singleton handle.
 * @return result of FFI
 */
LoonFFIResult loon_get_filesystem_singleton_handle(FileSystemHandle* out_handle);

/**
 * Get file statistics including size and metadata.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file.
 * @param path_len The length of the path.
 * @param out_size The output file size.
 * @param out_meta_keys The output metadata keys array (caller must free).
 * @param out_meta_values The output metadata values array (caller must free).
 * @param out_meta_count The output metadata count.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_get_file_stats(FileSystemHandle handle,
                                             const char* path_ptr,
                                             uint32_t path_len,
                                             uint64_t* out_size,
                                             char*** out_meta_keys,
                                             char*** out_meta_values,
                                             uint32_t* out_meta_count);

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
LoonFFIResult loon_filesystem_read_file_all(
    FileSystemHandle handle, const char* path_ptr, uint32_t path_len, uint8_t** out_data, uint64_t* out_size);

/**
 * Write data to a file with optional metadata.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to write.
 * @param path_len The length of the path.
 * @param data The data to write.
 * @param data_size The size of the data.
 * @param meta_keys The metadata keys (optional, can be NULL if meta_count is 0).
 * @param meta_values The metadata values (optional, can be NULL if meta_count is 0).
 * @param meta_count The number of metadata pairs.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_write_file(FileSystemHandle handle,
                                         const char* path_ptr,
                                         uint32_t path_len,
                                         const uint8_t* data,
                                         uint64_t data_size,
                                         const char** meta_keys,
                                         const char** meta_values,
                                         uint32_t meta_count);

/**
 * Delete a file.
 *
 * @param handle The filesystem instance.
 * @param path_ptr The path of the file to delete.
 * @param path_len The length of the path.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_delete_file(FileSystemHandle handle, const char* path_ptr, uint32_t path_len);

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
LoonFFIResult loon_filesystem_get_path_info(FileSystemHandle handle,
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
LoonFFIResult loon_filesystem_create_dir(FileSystemHandle handle,
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
LoonFFIResult loon_filesystem_list_dir(FileSystemHandle handle,
                                       const char* path_ptr,
                                       uint32_t path_len,
                                       bool recursive,
                                       char*** out_paths,
                                       uint32_t** out_path_lens,
                                       bool** out_is_dirs,
                                       uint64_t** out_sizes,
                                       int64_t** out_mtime_ns,
                                       uint32_t* out_count);

#endif  // LOON_FILESYSTEM_C

#ifdef __cplusplus
}
#endif
