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
#include <cstdint>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

extern "C" {
typedef void* FileSystemHandle;
typedef void* FileSystemWriterHandle;
typedef void* FileSystemConfig;

// C-ABI: writer API
FFIResult fscpp_open_writer(FileSystemHandle ptr,
                            const uint8_t* path_ptr,
                            uint64_t path_len,
                            FileSystemWriterHandle* out_writer_ptr);
FFIResult fscpp_write(FileSystemWriterHandle ptr, const uint8_t* data, uint64_t size);
FFIResult fscpp_flush(FileSystemWriterHandle ptr);
FFIResult fscpp_close(FileSystemWriterHandle ptr);
void fscpp_destroy_writer(FileSystemWriterHandle ptr);

// C-ABI: reader API
FFIResult fscpp_head_object(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len, uint64_t* out_size);
FFIResult fscpp_get_object(FileSystemHandle ptr,
                           const uint8_t* path_ptr,
                           uint64_t path_len,
                           uint64_t start,
                           uint64_t out_size,
                           uint8_t* out_data);
};

FFIResult fscpp_open_writer(FileSystemHandle ptr,
                            const uint8_t* path_ptr,
                            uint64_t path_len,
                            FileSystemWriterHandle* out_writer_ptr) {
  try {
    OutputStreamWrapper* wrapper = nullptr;
    assert(ptr != nullptr);
    assert(path_ptr != nullptr && path_len > 0);
    assert(out_writer_ptr != nullptr);

    auto fs = reinterpret_cast<FileSystemWrapper*>(ptr)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto output_stream_result = fs->OpenOutputStream(path);
    if (!output_stream_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, output_stream_result.status().ToString());
    }

    auto output_stream = output_stream_result.ValueOrDie();
    wrapper = new OutputStreamWrapper(output_stream);
    *out_writer_ptr = reinterpret_cast<FileSystemWriterHandle>(wrapper);

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_open_writer. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult fscpp_write(FileSystemWriterHandle ptr, const uint8_t* data, uint64_t size) {
  try {
    assert(ptr != nullptr);
    assert(data != nullptr && size > 0);

    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(ptr)->get();
    auto write_status = output_stream->Write(data, size);
    if (!write_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, write_status.ToString());
    }
    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_write. details: ", e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult fscpp_flush(FileSystemWriterHandle ptr) {
  try {
    assert(ptr != nullptr);

    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(ptr)->get();
    auto flush_result = output_stream->Flush();
    if (!flush_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, flush_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_flush. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

FFIResult fscpp_close(FileSystemWriterHandle ptr) {
  try {
    assert(ptr != nullptr);
    auto output_stream = reinterpret_cast<OutputStreamWrapper*>(ptr)->get();
    auto close_result = output_stream->Close();
    if (!close_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, close_result.ToString());
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_close. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

void fscpp_destroy_writer(FileSystemWriterHandle ptr) {
  if (ptr) {
    auto* wrapper = reinterpret_cast<OutputStreamWrapper*>(ptr);
    delete wrapper;
  }
}

FFIResult fscpp_head_object(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len, uint64_t* out_size) {
  try {
    assert(ptr != nullptr);
    auto fs = reinterpret_cast<FileSystemWrapper*>(ptr)->get();
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
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_close. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}

FFIResult fscpp_get_object(FileSystemHandle ptr,
                           const uint8_t* path_ptr,
                           uint64_t path_len,
                           uint64_t start,
                           uint64_t out_size,
                           uint8_t* out_data) {
  try {
    assert(ptr != nullptr);
    assert(path_ptr != nullptr && path_len > 0);
    assert(out_data != nullptr && out_size > 0);

    auto fs = reinterpret_cast<FileSystemWrapper*>(ptr)->get();
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> input_file_result;
    std::shared_ptr<arrow::io::RandomAccessFile> input_file;

    arrow::Result<int64_t> read_result;
    std::shared_ptr<arrow::Buffer> out_buffer = nullptr;
    int64_t read_size = 0;

    input_file_result = fs->OpenInputFile(path);
    if (!input_file_result.ok()) {
      out_data = nullptr;
      out_size = 0;
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to open input stream, [path=", path,
                   "] details: ", input_file_result.status().ToString());
    }

    input_file = input_file_result.ValueOrDie();

    read_result = input_file->ReadAt(start, out_size, out_data);
    if (!read_result.ok()) {
      // won't fail, no need check
      (void)input_file->Close();
      out_data = nullptr;
      out_size = 0;
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Fail to read object data, [path=", path, ", start=", start, ", size=", out_size,
                   "] details: ", read_result.status().ToString());
    }

    read_size = read_result.ValueOrDie();

    if (read_size != out_size) {
      (void)input_file->Close();
      out_data = nullptr;
      out_size = 0;
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Read size mismatch, expected size=", out_size, ", actual size=", read_size,
                   ", [path=", path, ", start=", start, "]");
    }

    // won't fail close here, no need check
    (void)input_file->Close();
    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Got exception in fscpp_get_object. details: ", e.what());
  }
  RETURN_UNREACHABLE();
}
