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

#include <cstdint>
#include <iostream>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/type.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/record_batch.h>

#include "milvus-storage/common/macro.h"

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

size_t CalculateArrayDataMemorySize(const std::shared_ptr<arrow::ArrayData>& array_data,
                                    std::unordered_set<const void*>& seen_buffers) {
  if (!array_data)
    return 0;
  size_t size = 0;

  for (const auto& buffer : array_data->buffers) {
    if (buffer && buffer->data()) {
      if (seen_buffers.insert(buffer->data()).second) {
        size += buffer->size();
      }
    }
  }

  for (const auto& child : array_data->child_data) {
    size += CalculateArrayDataMemorySize(child, seen_buffers);
  }

  return size;
}

size_t CalculateArrayMemorySize(const std::shared_ptr<arrow::Array>& array,
                                std::unordered_set<const void*>& seen_buffers) {
  if (!array)
    return 0;
  return CalculateArrayDataMemorySize(array->data(), seen_buffers);
}

size_t GetArrowArrayMemorySize(const std::shared_ptr<arrow::Array>& array) {
  std::unordered_set<const void*> seen_buffers;
  return CalculateArrayMemorySize(array, seen_buffers);
}

size_t GetRecordBatchMemorySize(const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  std::unordered_set<const void*> seen_buffers;
  size_t total_size = 0;

  for (int i = 0; i < record_batch->num_columns(); ++i) {
    total_size += CalculateArrayMemorySize(record_batch->column(i), seen_buffers);
  }
  return total_size;
}

size_t GetTableMemorySize(const std::shared_ptr<arrow::Table>& table) {
  std::unordered_set<const void*> seen_buffers;
  size_t total_size = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto chunked_array = table->column(i);
    for (int j = 0; j < chunked_array->num_chunks(); ++j) {
      total_size += CalculateArrayMemorySize(chunked_array->chunk(j), seen_buffers);
    }
  }
  return total_size;
}

static inline bool WillCombineChunksCopy(const std::shared_ptr<arrow::Table>& table) {
  for (int i = 0; i < table->num_columns(); ++i) {
    if (table->column(i)->num_chunks() > 1) {
      return true;
    }
  }
  return false;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ConvertTableToRecordBatch(const std::shared_ptr<arrow::Table>& table,
                                                                             bool allow_concat) {
  assert(table && table->num_rows() != 0);

  if (!allow_concat && WillCombineChunksCopy(table)) {
    return arrow::Status::Invalid("Current table has multiple chunks, which will trigger copy when combining chunks");
  }

  return table->CombineChunksToBatch();
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ConvertTableToRecordBatchs(
    const std::shared_ptr<arrow::Table>& table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  std::shared_ptr<arrow::RecordBatch> rb;
  arrow::TableBatchReader table_reader(table);

  while (true) {
    ARROW_RETURN_NOT_OK(table_reader.ReadNext(&rb));
    if (!rb) {
      break;
    }
    result.emplace_back(rb);
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::Table>> CopySelectedRows(const std::shared_ptr<arrow::Table>& table,
                                                              const std::vector<int64_t>& indices) {
  // wrap indices to Int64Array
  auto index_array = std::make_shared<arrow::Int64Array>(indices.size(), arrow::Buffer::Wrap(indices));

  arrow::compute::ExecContext context;
  arrow::compute::TakeOptions options = arrow::compute::TakeOptions::Defaults();

  // apply take to each column
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
  for (int i = 0; i < table->num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto datum, arrow::compute::Take(table->column(i), index_array, options, &context));

    if (datum.kind() == arrow::Datum::CHUNKED_ARRAY) {
      new_columns.emplace_back(datum.chunked_array());
    } else {
      new_columns.emplace_back(std::make_shared<arrow::ChunkedArray>(datum.make_array()));
    }
  }

  // create new table
  return arrow::Table::Make(table->schema(), new_columns, indices.size());
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
