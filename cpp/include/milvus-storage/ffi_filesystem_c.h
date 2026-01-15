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
 * @param out_meta_array The output metadata array (caller must free using loon_filesystem_free_meta_array).
 * @param out_meta_count The output metadata count.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_get_file_stats(FileSystemHandle handle,
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
void loon_filesystem_free_meta_array(LoonFileSystemMeta* meta_array, uint32_t meta_count);

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
 * @param meta_array The metadata array (optional, can be NULL if meta_count is 0).
 * @param meta_count The number of metadata pairs.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_write_file(FileSystemHandle handle,
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
