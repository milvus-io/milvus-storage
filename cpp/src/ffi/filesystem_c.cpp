// Copyright 2024 Zilliz
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
#include "milvus-storage/ffi_filesystem_c.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/properties.h"

using namespace milvus_storage::api;
using namespace milvus_storage;

LoonFFIResult loon_filesystem_get(const ::LoonProperties* properties,
                                  const char* path,
                                  uint32_t path_len,
                                  FileSystemHandle* out_handle) {
  try {
    if (!properties || !out_handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "properties and out_handle must not be null");
    }

    milvus_storage::api::Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    std::string path_str;
    if (path && path_len > 0) {
      path_str = std::string(path, path_len);
    }

    auto& cache = FilesystemCache::getInstance();
    auto fs_result = cache.get(properties_map, path_str);
    if (!fs_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_result.status().ToString());
    }

    auto fs_wrapper = std::make_unique<FileSystemWrapper>(fs_result.ValueOrDie());
    *out_handle = reinterpret_cast<FileSystemHandle>(fs_wrapper.release());

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_get. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_filesystem_destroy(FileSystemHandle handle) {
  if (handle) {
    auto* wrapper = reinterpret_cast<FileSystemWrapper*>(handle);
    delete wrapper;
  }
}

void loon_close_filesystems() {
  auto& fs_cache = milvus_storage::FilesystemCache::getInstance();
  fs_cache.clean();
}

LoonFFIResult loon_filesystem_open_writer(FileSystemHandle handle,
                                          const char* path_ptr,
                                          uint32_t path_len,
                                          const LoonFileSystemMeta* meta_array,
                                          uint32_t num_of_meta,
                                          FileSystemWriterHandle* out_writer_ptr) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_writer_ptr) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: handle, path_ptr, path_len, and out_writer_ptr must not be null");
    }

    if (num_of_meta > 0 && !meta_array) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: meta_array must not be null if current num_of_meta > 0");
    }

    // build metadata if passed
    std::shared_ptr<const arrow::KeyValueMetadata> metadatas = nullptr;
    if (num_of_meta > 0) {
      std::vector<std::string> keys(num_of_meta);
      std::vector<std::string> values(num_of_meta);
      for (size_t i = 0; i < num_of_meta; i++) {
        if (!meta_array[i].key || !meta_array[i].value) {
          RETURN_ERROR(LOON_INVALID_ARGS, "The meta_array[", i, "].key or value is nullptr");
        }
        keys[i] = std::string(meta_array[i].key);
        values[i] = std::string(meta_array[i].value);
      }

      metadatas = arrow::KeyValueMetadata::Make(keys, values);
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto output_stream_result = fs->OpenOutputStream(path, metadatas);
    if (!output_stream_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, output_stream_result.status().ToString());
    }

    auto output_stream = output_stream_result.ValueOrDie();
    auto wrapper = std::make_unique<OutputStreamWrapper>(output_stream);
    *out_writer_ptr = reinterpret_cast<FileSystemWriterHandle>(wrapper.release());

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_open_writer. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_writer_write(FileSystemWriterHandle handle, const uint8_t* data, uint64_t size) {
  try {
    if (!handle || !data || size == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, data, and size must not be null");
    }

    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(handle)->get();
    auto write_status = output_stream->Write(data, size);
    if (!write_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, write_status.ToString());
    }
    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_writer_write. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_writer_flush(FileSystemWriterHandle handle) {
  try {
    if (!handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
    }

    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(handle)->get();
    auto flush_result = output_stream->Flush();
    if (!flush_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, flush_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_writer_flush. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_writer_close(FileSystemWriterHandle handle) {
  try {
    if (!handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
    }

    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(handle)->get();
    auto close_result = output_stream->Close();
    if (!close_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, close_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_writer_close. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

void loon_filesystem_writer_destroy(FileSystemWriterHandle handle) {
  if (handle) {
    auto* wrapper = reinterpret_cast<OutputStreamWrapper*>(handle);
    delete wrapper;
  }
}

LoonFFIResult loon_filesystem_get_file_info(FileSystemHandle handle,
                                            const char* path_ptr,
                                            uint32_t path_len,
                                            uint64_t* out_size) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_size) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, path_len, and out_size must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto info_result = fs->GetFileInfo(path);
    if (!info_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Fail to get file info, [path=", path,
                   "] details: ", info_result.status().ToString());
    }

    auto info = info_result.ValueOrDie();
    *out_size = static_cast<uint64_t>(info.size());

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_get_file_info. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_read_file(FileSystemHandle handle,
                                        const char* path_ptr,
                                        uint32_t path_len,
                                        uint64_t offset,
                                        uint64_t nbytes,
                                        uint8_t* out_data) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_data || nbytes == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: handle, path_ptr, path_len, out_data, and nbytes must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> input_file_result;
    std::shared_ptr<arrow::io::RandomAccessFile> input_file;

    arrow::Result<int64_t> read_result;
    int64_t read_size = 0;

    input_file_result = fs->OpenInputFile(path);
    if (!input_file_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Fail to open input stream, [path=", path,
                   "] details: ", input_file_result.status().ToString());
    }

    input_file = input_file_result.ValueOrDie();

    read_result = input_file->ReadAt(offset, nbytes, out_data);
    if (!read_result.ok()) {
      // won't fail, no need check
      (void)input_file->Close();
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to read object data, [path=", path, ", offset=", offset, ", size=", nbytes,
                   "] details: ", read_result.status().ToString());
    }

    read_size = read_result.ValueOrDie();

    if (read_size != nbytes) {
      (void)input_file->Close();
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Read size mismatch, expected size=", nbytes, ", actual size=", read_size,
                   ", [path=", path, ", offset=", offset, "]");
    }

    // close the inputstream
    auto close_result = input_file->Close();
    if (!close_result.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to close inputstream, details: ", close_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_read_file. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_open_reader(FileSystemHandle handle,
                                          const char* path_ptr,
                                          uint32_t path_len,
                                          FileSystemReaderHandle* out_reader_ptr) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_reader_ptr) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: handle, path_ptr, path_len, and out_reader_ptr must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto input_file_result = fs->OpenInputFile(path);
    if (!input_file_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, input_file_result.status().ToString());
    }

    auto input_file = input_file_result.ValueOrDie();

    auto wrapper = std::make_unique<RandomAccessFileWrapper>(input_file);
    *out_reader_ptr = reinterpret_cast<FileSystemReaderHandle>(wrapper.release());

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_open_reader. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_reader_readat(FileSystemReaderHandle handle,
                                            uint64_t offset,
                                            uint64_t nbytes,
                                            uint8_t* out_data) {
  try {
    if (!handle || !out_data || nbytes == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, out_data, and nbytes must not be null");
    }

    auto input_file = reinterpret_cast<RandomAccessFileWrapper*>(handle)->get();
    auto read_result = input_file->ReadAt(offset, nbytes, out_data);
    if (!read_result.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to read object data, [offset=", offset, ", size=", nbytes,
                   "] details: ", read_result.status().ToString());
    }

    auto read_size = read_result.ValueOrDie();
    if (read_size != nbytes) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Read size mismatch, expected size=", nbytes, ", actual size=", read_size,
                   ", [offset=", offset, "]");
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_reader_readat. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

/**
 * Close the inputstream.
 *
 * @param handle The inputstream instance.
 * @return result of FFI
 */
LoonFFIResult loon_filesystem_reader_close(FileSystemReaderHandle handle) {
  try {
    if (!handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
    }

    auto input_file = reinterpret_cast<RandomAccessFileWrapper*>(handle)->get();
    auto close_result = input_file->Close();
    if (!close_result.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to close inputstream, details: ", close_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_reader_close. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_filesystem_reader_destroy(FileSystemReaderHandle handle) {
  if (handle) {
    auto* wrapper = reinterpret_cast<RandomAccessFileWrapper*>(handle);
    delete wrapper;
  }
}

LoonFFIResult loon_initialize_filesystem_singleton(const ::LoonProperties* properties) {
  try {
    if (!properties) {
      RETURN_ERROR(LOON_INVALID_ARGS, "properties is null");
    }

    milvus_storage::api::Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    ArrowFileSystemConfig fs_config;
    auto fs_status = ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
    if (!fs_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_status.ToString());
    }

    // Initialize the singleton with the config
    ArrowFileSystemSingleton::GetInstance().Init(fs_config);

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in initialize_filesystem_singleton. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_get_filesystem_singleton_handle(FileSystemHandle* out_handle) {
  try {
    if (!out_handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "out_handle is null");
    }

    // Get the filesystem from singleton
    auto fs = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
    if (!fs) {
      RETURN_ERROR(LOON_LOGICAL_ERROR,
                   "Filesystem singleton not initialized. Call initialize_filesystem_singleton first.");
    }

    // Wrap it and return as handle
    auto fs_wrapper = std::make_unique<FileSystemWrapper>(fs);
    auto raw_fs_wrapper = reinterpret_cast<FileSystemHandle>(fs_wrapper.release());
    assert(raw_fs_wrapper);
    *out_handle = raw_fs_wrapper;

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_get_filesystem_singleton_handle. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_get_file_stats(FileSystemHandle handle,
                                             const char* path_ptr,
                                             uint32_t path_len,
                                             uint64_t* out_size,
                                             LoonFileSystemMeta** out_meta_array,
                                             uint32_t* out_meta_count) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_size) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, path_len, and out_size must not be null");
    }

    // Initialize outputs
    *out_size = 0;
    if (out_meta_array)
      *out_meta_array = nullptr;
    if (out_meta_count)
      *out_meta_count = 0;

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Open input file to read size and metadata
    auto input_result = fs->OpenInputFile(path);
    if (!input_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to open input file: ", input_result.status().ToString());
    }
    auto input_file = input_result.ValueOrDie();

    // Get file size
    auto size_result = input_file->GetSize();
    if (!size_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to get file size: ", size_result.status().ToString());
    }
    *out_size = static_cast<uint64_t>(size_result.ValueOrDie());

    // Read metadata if requested
    if (out_meta_array && out_meta_count) {
      auto metadata_result = input_file->ReadMetadata();
      if (metadata_result.ok()) {
        auto metadata = metadata_result.ValueOrDie();
        if (metadata && metadata->size() > 0) {
          const auto& keys = metadata->keys();
          const auto& values = metadata->values();
          uint32_t count = static_cast<uint32_t>(keys.size());

          // Allocate array of LoonFileSystemMeta structs
          *out_meta_array = (LoonFileSystemMeta*)malloc(count * sizeof(LoonFileSystemMeta));

          if (!*out_meta_array) {
            RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate memory for metadata array");
          }

          // Initialize all pointers to nullptr
          for (uint32_t i = 0; i < count; i++) {
            (*out_meta_array)[i].key = nullptr;
            (*out_meta_array)[i].value = nullptr;
          }

          // Copy key-value pairs
          for (uint32_t i = 0; i < count; i++) {
            (*out_meta_array)[i].key = strdup(keys[i].c_str());
            (*out_meta_array)[i].value = strdup(values[i].c_str());

            if (!(*out_meta_array)[i].key || !(*out_meta_array)[i].value) {
              // Clean up on error
              for (uint32_t j = 0; j <= i; j++) {
                if ((*out_meta_array)[j].key)
                  free((*out_meta_array)[j].key);
                if ((*out_meta_array)[j].value)
                  free((*out_meta_array)[j].value);
              }
              free(*out_meta_array);
              *out_meta_array = nullptr;
              RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to duplicate metadata strings");
            }
          }

          *out_meta_count = count;
        }
      }
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    // Clean up on error
    if (out_size)
      *out_size = 0;
    if (out_meta_array && *out_meta_array) {
      for (uint32_t i = 0; i < (out_meta_count && *out_meta_count ? *out_meta_count : 0); i++) {
        if ((*out_meta_array)[i].key)
          free((*out_meta_array)[i].key);
        if ((*out_meta_array)[i].value)
          free((*out_meta_array)[i].value);
      }
      free(*out_meta_array);
      *out_meta_array = nullptr;
    }
    if (out_meta_count)
      *out_meta_count = 0;
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_get_file_stats. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_filesystem_free_meta_array(LoonFileSystemMeta* meta_array, uint32_t meta_count) {
  if (!meta_array) {
    return;
  }

  // Free each key and value string
  for (uint32_t i = 0; i < meta_count; i++) {
    if (meta_array[i].key) {
      free(meta_array[i].key);
    }
    if (meta_array[i].value) {
      free(meta_array[i].value);
    }
  }

  // Free the array itself
  free(meta_array);
}

LoonFFIResult loon_filesystem_read_file_all(
    FileSystemHandle handle, const char* path_ptr, uint32_t path_len, uint8_t** out_data, uint64_t* out_size) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_data || !out_size) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: handle, path_ptr, path_len, out_data, and out_size must not be null");
    }

    *out_data = nullptr;
    *out_size = 0;

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Open input file
    auto input_result = fs->OpenInputFile(path);
    if (!input_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to open input file: ", input_result.status().ToString());
    }
    auto input_file = input_result.ValueOrDie();

    // Get file size
    auto size_result = input_file->GetSize();
    if (!size_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to get file size: ", size_result.status().ToString());
    }
    auto file_size = size_result.ValueOrDie();

    // Allocate memory for file content
    *out_size = static_cast<uint64_t>(file_size);
    *out_data = (uint8_t*)malloc(file_size);
    if (!*out_data) {
      *out_size = 0;
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate memory for file data");
    }

    auto read_result = input_file->Read(static_cast<int64_t>(file_size));
    if (!read_result.ok()) {
      free(*out_data);
      *out_data = nullptr;
      *out_size = 0;
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to read file: ", read_result.status().ToString());
    }
    auto buffer = read_result.ValueOrDie();

    // Copy data to output
    std::memcpy(*out_data, buffer->data(), file_size);

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    if (out_data && *out_data) {
      free(*out_data);
      *out_data = nullptr;
    }
    if (out_size) {
      *out_size = 0;
    }
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_read_file_all. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_write_file(FileSystemHandle handle,
                                         const char* path_ptr,
                                         uint32_t path_len,
                                         const uint8_t* data,
                                         uint64_t data_size,
                                         const LoonFileSystemMeta* meta_array,
                                         uint32_t meta_count) {
  try {
    if (!handle || !path_ptr || path_len == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, and path_len must not be null");
    }

    if (data_size > 0 && !data) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Data cannot be null if data_size > 0");
    }

    if (meta_count > 0 && !meta_array) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Metadata array must not be null if meta_count > 0");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Build metadata if provided
    std::shared_ptr<const arrow::KeyValueMetadata> metadata = nullptr;
    if (meta_count > 0) {
      std::vector<std::string> keys(meta_count);
      std::vector<std::string> values(meta_count);
      for (uint32_t i = 0; i < meta_count; i++) {
        if (!meta_array[i].key || !meta_array[i].value) {
          RETURN_ERROR(LOON_INVALID_ARGS, "Metadata key or value is null at index ", i);
        }
        keys[i] = std::string(meta_array[i].key);
        values[i] = std::string(meta_array[i].value);
      }
      metadata = arrow::KeyValueMetadata::Make(keys, values);
    }

    // Open output stream
    auto stream_result = fs->OpenOutputStream(path, metadata);
    if (!stream_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to open output stream: ", stream_result.status().ToString());
    }
    auto output_stream = stream_result.ValueOrDie();

    // Write data
    if (data_size > 0) {
      auto write_status = output_stream->Write(data, data_size);
      if (!write_status.ok()) {
        (void)output_stream->Close();  // Try to close, ignore errors
        RETURN_ERROR(LOON_ARROW_ERROR, "Failed to write data: ", write_status.ToString());
      }
    }

    // Close the stream
    auto close_status = output_stream->Close();
    if (!close_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to close output stream: ", close_status.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_write_file. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_delete_file(FileSystemHandle handle, const char* path_ptr, uint32_t path_len) {
  try {
    if (!handle || !path_ptr || path_len == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, and path_len must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Verify file exists before deletion
    auto file_info_result = fs->GetFileInfo(path);
    if (!file_info_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to get file info: ", file_info_result.status().ToString());
    }
    auto file_info = file_info_result.ValueOrDie();

    if (file_info.type() == arrow::fs::FileType::NotFound) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "File not found: ", path);
    }

    if (file_info.type() != arrow::fs::FileType::File) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Path is not a file: ", path);
    }

    // Delete the file
    auto delete_status = fs->DeleteFile(path);
    if (!delete_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to delete file: ", delete_status.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_delete_file. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_get_path_info(FileSystemHandle handle,
                                            const char* path_ptr,
                                            uint32_t path_len,
                                            bool* out_exists,
                                            bool* out_is_dir,
                                            int64_t* out_mtime_ns) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_exists) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, path_len, and out_exists must not be null");
    }

    // Initialize outputs
    *out_exists = false;
    if (out_is_dir)
      *out_is_dir = false;
    if (out_mtime_ns)
      *out_mtime_ns = 0;

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Get file info
    auto file_info_result = fs->GetFileInfo(path);
    if (!file_info_result.ok()) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Failed to get file info: ", file_info_result.status().ToString());
    }

    auto file_info = file_info_result.ValueOrDie();

    // Check if path exists
    if (file_info.type() == arrow::fs::FileType::NotFound) {
      RETURN_ERROR(LOON_INVALID_ARGS, "File not found: ", path);
    }

    // Path exists
    *out_exists = true;

    // Check if it's a directory
    if (out_is_dir) {
      *out_is_dir = (file_info.type() == arrow::fs::FileType::Directory);
    }

    // Get modification time
    if (out_mtime_ns) {
      auto mtime = file_info.mtime();
      if (mtime.time_since_epoch().count() > 0) {
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch());
        *out_mtime_ns = duration_ns.count();
      }
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    if (out_exists)
      *out_exists = false;
    if (out_is_dir)
      *out_is_dir = false;
    if (out_mtime_ns)
      *out_mtime_ns = 0;
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_get_path_info. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_create_dir(FileSystemHandle handle,
                                         const char* path_ptr,
                                         uint32_t path_len,
                                         bool recursive) {
  try {
    if (!handle || !path_ptr || path_len == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, path_ptr, and path_len must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Create directory
    auto create_status = fs->CreateDir(path, recursive);
    if (!create_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to create directory: ", create_status.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_create_dir. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_list_dir(FileSystemHandle handle,
                                       const char* path_ptr,
                                       uint32_t path_len,
                                       bool recursive,
                                       char*** out_paths,
                                       uint32_t** out_path_lens,
                                       bool** out_is_dirs,
                                       uint64_t** out_sizes,
                                       int64_t** out_mtime_ns,
                                       uint32_t* out_count) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_paths || !out_path_lens || !out_is_dirs || !out_sizes ||
        !out_mtime_ns || !out_count) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: all output parameters must not be null");
    }

    // Initialize outputs
    *out_paths = nullptr;
    *out_path_lens = nullptr;
    *out_is_dirs = nullptr;
    *out_sizes = nullptr;
    *out_mtime_ns = nullptr;
    *out_count = 0;

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    std::string path(path_ptr, path_len);

    // Create FileSelector
    arrow::fs::FileSelector selector;
    selector.base_dir = path;
    selector.recursive = recursive;
    selector.allow_not_found = false;

    // Get file info list
    auto file_info_result = fs->GetFileInfo(selector);
    if (!file_info_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to list directory: ", file_info_result.status().ToString());
    }

    auto file_infos = file_info_result.ValueOrDie();
    uint32_t count = static_cast<uint32_t>(file_infos.size());

    if (count == 0) {
      RETURN_SUCCESS();
    }

    // Allocate arrays
    *out_paths = (char**)malloc(count * sizeof(char*));
    *out_path_lens = (uint32_t*)malloc(count * sizeof(uint32_t));
    *out_is_dirs = (bool*)malloc(count * sizeof(bool));
    *out_sizes = (uint64_t*)malloc(count * sizeof(uint64_t));
    *out_mtime_ns = (int64_t*)malloc(count * sizeof(int64_t));

    if (!*out_paths || !*out_path_lens || !*out_is_dirs || !*out_sizes || !*out_mtime_ns) {
      if (*out_paths)
        free(*out_paths);
      if (*out_path_lens)
        free(*out_path_lens);
      if (*out_is_dirs)
        free(*out_is_dirs);
      if (*out_sizes)
        free(*out_sizes);
      if (*out_mtime_ns)
        free(*out_mtime_ns);
      *out_paths = nullptr;
      *out_path_lens = nullptr;
      *out_is_dirs = nullptr;
      *out_sizes = nullptr;
      *out_mtime_ns = nullptr;
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate memory for list directory results");
    }

    // Fill arrays with file info
    for (uint32_t i = 0; i < count; i++) {
      const auto& file_info = file_infos[i];

      // Copy path
      (*out_paths)[i] = strdup(file_info.path().c_str());
      if (!(*out_paths)[i]) {
        // Clean up on error
        for (uint32_t j = 0; j < i; j++) {
          free((*out_paths)[j]);
        }
        free(*out_paths);
        free(*out_path_lens);
        free(*out_is_dirs);
        free(*out_sizes);
        free(*out_mtime_ns);
        *out_paths = nullptr;
        *out_path_lens = nullptr;
        *out_is_dirs = nullptr;
        *out_sizes = nullptr;
        *out_mtime_ns = nullptr;
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to duplicate path string");
      }

      (*out_path_lens)[i] = static_cast<uint32_t>(file_info.path().length());
      (*out_is_dirs)[i] = (file_info.type() == arrow::fs::FileType::Directory);
      (*out_sizes)[i] =
          (file_info.type() == arrow::fs::FileType::Directory) ? 0 : static_cast<uint64_t>(file_info.size());

      auto mtime = file_info.mtime();
      if (mtime.time_since_epoch().count() > 0) {
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch());
        (*out_mtime_ns)[i] = duration_ns.count();
      } else {
        (*out_mtime_ns)[i] = 0;
      }
    }

    *out_count = count;

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    // Clean up on error
    if (out_paths && *out_paths && out_count && *out_count > 0) {
      for (uint32_t i = 0; i < *out_count; i++) {
        free((*out_paths)[i]);
      }
      free(*out_paths);
      *out_paths = nullptr;
    }
    if (out_path_lens && *out_path_lens) {
      free(*out_path_lens);
      *out_path_lens = nullptr;
    }
    if (out_is_dirs && *out_is_dirs) {
      free(*out_is_dirs);
      *out_is_dirs = nullptr;
    }
    if (out_sizes && *out_sizes) {
      free(*out_sizes);
      *out_sizes = nullptr;
    }
    if (out_mtime_ns && *out_mtime_ns) {
      free(*out_mtime_ns);
      *out_mtime_ns = nullptr;
    }
    if (out_count)
      *out_count = 0;
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in loon_filesystem_list_dir. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}
