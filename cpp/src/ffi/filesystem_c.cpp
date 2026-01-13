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

#include <cstdint>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

using namespace milvus_storage::api;
using namespace milvus_storage;

LoonFFIResult loon_filesystem_create(const ::LoonProperties* properties, FileSystemHandle* out_fs_ptr) {
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

    auto fs_result = CreateArrowFileSystem(fs_config);
    if (!fs_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_result.status().ToString());
    }

    auto fs_wrapper = std::make_unique<FileSystemWrapper>(fs_result.ValueOrDie());
    auto raw_fs_wrapper = reinterpret_cast<FileSystemHandle>(fs_wrapper.release());
    assert(raw_fs_wrapper);
    *out_fs_ptr = raw_fs_wrapper;

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in filesystem_create. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_filesystem_destroy(FileSystemHandle ptr) {
  if (ptr) {
    delete reinterpret_cast<FileSystemWrapper*>(ptr);
  }
}

LoonFFIResult loon_filesystem_open_writer(FileSystemHandle handle,
                                          const char* path_ptr,
                                          uint32_t path_len,
                                          const char** meta_keys,
                                          const char** meta_values,
                                          uint32_t num_of_meta,
                                          FileSystemWriterHandle* out_writer_ptr) {
  try {
    if (!handle || !path_ptr || path_len == 0 || !out_writer_ptr) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: handle, path_ptr, path_len, and out_writer_ptr must not be null");
    }

    if (num_of_meta > 0 && (!meta_keys || !meta_values)) {
      RETURN_ERROR(LOON_INVALID_ARGS,
                   "Invalid arguments: meta_keys and meta_values must not be null if current num_of_meta > 0");
    }

    // build metadata if passed
    std::shared_ptr<const arrow::KeyValueMetadata> metadatas = nullptr;
    if (num_of_meta > 0) {
      std::vector<std::string> keys(num_of_meta);
      std::vector<std::string> values(num_of_meta);
      for (size_t i = 0; i < num_of_meta; i++) {
        if (!meta_keys[i] || !meta_values[i]) {
          RETURN_ERROR(LOON_INVALID_ARGS, "The meta_keys or meta_values is nullptr [index=", i, "]");
        }
        keys[i] = std::string(meta_keys[i]);
        values[i] = std::string(meta_values[i]);
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
