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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/macro.h"
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/type.h>
#include <arrow/table.h>
#include <iostream>
#include <arrow/util/key_value_metadata.h>
#include <cstdint>

namespace milvus_storage {
arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(arrow::fs::FileSystem& fs,
                                                                               const std::string& file_path) {
  ARROW_ASSIGN_OR_RAISE(auto file, fs.OpenInputFile(file_path));

  std::unique_ptr<parquet::arrow::FileReader> file_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(file, arrow::default_memory_pool(), &file_reader));
  return std::move(file_reader);
}

arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(
    arrow::fs::FileSystem& fs, const std::string& file_path, const parquet::ReaderProperties& read_properties) {
  ARROW_ASSIGN_OR_RAISE(auto file, fs.OpenInputFile(file_path));
  parquet::arrow::FileReaderBuilder builder;
  std::unique_ptr<parquet::arrow::FileReader> reader;

  // Create a completely independent deep copy of read_properties to prevent any mutations
  // This ensures the original read_properties object remains unchanged
  auto read_properties_copy = std::make_shared<parquet::ReaderProperties>(read_properties);
  if (read_properties.file_decryption_properties()) {
    auto deep_copied_decryption = read_properties.file_decryption_properties()->DeepClone();
    read_properties_copy->file_decryption_properties(std::move(deep_copied_decryption));
  }

  ARROW_RETURN_NOT_OK(builder.Open(std::move(file), *read_properties_copy));
  ARROW_RETURN_NOT_OK(builder.memory_pool(arrow::default_memory_pool())
                          ->properties(parquet::default_arrow_reader_properties())
                          ->Build(&reader));
  return std::move(reader);
}

size_t GetRecordBatchMemorySize(const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  if (!record_batch) {
    return 0;
  }
  size_t total_size = 0;
  for (const auto& column : record_batch->columns()) {
    total_size += GetArrowArrayMemorySize(column);
  }
  return total_size;
}

size_t GetArrowArrayMemorySize(const std::shared_ptr<arrow::Array>& array) {
  if (!array || !array->data()) {
    return 0;
  }
  size_t total_size = 0;
  for (const auto& buffer : array->data()->buffers) {
    if (buffer) {
      total_size += buffer->size();
    }
  }
  return total_size;
}

size_t GetTableMemorySize(const std::shared_ptr<arrow::Table>& table) {
  size_t total_size = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto chunked_array = table->column(i);
    for (int j = 0; j < chunked_array->num_chunks(); ++j) {
      total_size += GetArrowArrayMemorySize(chunked_array->chunk(j));
    }
  }
  return total_size;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ConvertTableToRecordBatch(
    const std::shared_ptr<arrow::Table>& table) {
  if (!table) {
    return arrow::Status::Invalid("Input table is null");
  }

  if (table->num_rows() == 0) {
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(table->schema(), 0, empty_arrays);
  }

  return table->CombineChunksToBatch();
}

arrow::Result<std::string> GetEnvVar(const char* name) {
#ifdef _WIN32
  // On Windows, getenv() reads an early copy of the process' environment
  // which doesn't get updated when SetEnvironmentVariable() is called.
  constexpr int32_t bufsize = 2000;
  char c_str[bufsize];
  auto res = GetEnvironmentVariableA(name, c_str, bufsize);
  if (res >= bufsize) {
    return arrow::Status::CapacityError("environment variable value too long");
  } else if (res == 0) {
    return arrow::Status::KeyError("environment variable undefined");
  }
  return std::string(c_str);
#else
  char* c_str = getenv(name);
  if (c_str == nullptr) {
    return arrow::Status::KeyError("environment variable undefined");
  }
  return std::string(c_str);
#endif
}

arrow::Result<std::string> GetEnvVar(const std::string& name) { return GetEnvVar(name.c_str()); }

}  // namespace milvus_storage
