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

#include "milvus-storage/format/format_reader.h"

#include <set>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include "milvus-storage/reader.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::api {

// ==================== ParquetFormatReader Implementation ====================

ParquetFormatReader::ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                         std::shared_ptr<ColumnGroup> column_group,
                                         std::shared_ptr<arrow::Schema> schema,
                                         const ReadProperties& properties)
    : fs_(std::move(fs)),
      column_group_(std::move(column_group)),
      schema_(std::move(schema)),
      properties_(properties),
      initialized_(false) {}

arrow::Status ParquetFormatReader::initialize(std::shared_ptr<ColumnGroup> column_group,
                                              const std::vector<std::string>& needed_columns) {
  if (initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader already initialized");
  }

  column_group_ = column_group;
  needed_columns_ = needed_columns;

  if (!column_group_) {
    return arrow::Status::Invalid("No column group provided");
  }

  // Create parquet reader for this column group file
  ARROW_ASSIGN_OR_RAISE(auto input_stream, fs_->OpenInputFile(column_group->path));

  // Create Arrow parquet reader
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input_stream, arrow::default_memory_pool(), &parquet_reader));

  parquet_reader_ = std::move(parquet_reader);

  initialized_ = true;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ParquetFormatReader::get_record_batch_reader(
    const std::string& predicate, int64_t batch_size, int64_t buffer_size) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  if (!parquet_reader_) {
    return arrow::Status::Invalid("No parquet reader available");
  }

  // Get the parquet reader
  auto& reader = parquet_reader_;

  // Create a record batch reader that reads all row groups
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;

  // Get all row group indices to ensure we read all data
  int num_row_groups = reader->parquet_reader()->metadata()->num_row_groups();
  std::vector<int> row_group_indices;
  for (int i = 0; i < num_row_groups; i++) {
    row_group_indices.push_back(i);
  }

  // Read all row groups
  ARROW_RETURN_NOT_OK(reader->GetRecordBatchReader(row_group_indices, &batch_reader));

  return batch_reader;
}

arrow::Result<std::shared_ptr<ChunkReader>> ParquetFormatReader::get_chunk_reader() {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  if (!column_group_) {
    return arrow::Status::Invalid("No column group available");
  }

  // Create ChunkReader for this column group
  return std::make_shared<ChunkReader>(fs_, column_group_, needed_columns_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetFormatReader::take(const std::vector<int64_t>& row_indices,
                                                                             int64_t parallelism) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices cannot be empty");
  }

  if (!parquet_reader_) {
    return arrow::Status::Invalid("No parquet reader available");
  }

  // Get the parquet reader
  auto& reader = parquet_reader_;

  // Read all data first
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));

  if (!table || table->num_rows() == 0) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(schema_, 0, empty_arrays);
  }

  // Create an indices array for the take operation
  arrow::Int64Builder builder;
  for (int64_t index : row_indices) {
    ARROW_RETURN_NOT_OK(builder.Append(index));
  }
  ARROW_ASSIGN_OR_RAISE(auto indices_array, builder.Finish());

  // Use Arrow compute API to take specific rows
  ARROW_ASSIGN_OR_RAISE(auto taken_table, arrow::compute::Take(table, indices_array));

  // Convert result back to RecordBatch
  auto result_table = taken_table.table();
  ARROW_ASSIGN_OR_RAISE(auto combined_table, result_table->CombineChunks());

  if (combined_table->num_rows() == 0) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(combined_table->schema(), 0, empty_arrays);
  }

  // Convert to RecordBatch
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (int i = 0; i < combined_table->num_columns(); ++i) {
    auto column = combined_table->column(i);
    if (column->num_chunks() > 0) {
      arrays.push_back(column->chunk(0));
    } else {
      return arrow::Status::Invalid("Column has no chunks");
    }
  }

  return arrow::RecordBatch::Make(combined_table->schema(), arrays[0]->length(), arrays);
}

}  // namespace milvus_storage::api
