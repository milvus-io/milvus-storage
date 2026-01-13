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
#include <stddef.h>
#include <stdint.h>
#include "milvus-storage/ffi_c.h"

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

#endif  // LOON_FILESYSTEM_C

#ifdef __cplusplus
}
#endif
