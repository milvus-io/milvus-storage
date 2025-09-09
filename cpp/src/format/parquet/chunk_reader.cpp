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

#include <memory>
#include <string>
#include <algorithm>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>

#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/chunk_reader.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/status.h"

namespace milvus_storage::api {

ParquetChunkReader::ParquetChunkReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       const std::string& path,
                                       parquet::ReaderProperties reader_props,
                                       const std::vector<std::string>& needed_columns)
    : ChunkReader(fs, path, needed_columns) {
  auto status = init(fs, path, reader_props);
  if (!status.ok()) {
    LOG_STORAGE_ERROR_ << "Error initializing file reader: " << status.ToString();
    throw std::runtime_error(status.ToString());
  }
}

Status ParquetChunkReader::init(std::shared_ptr<arrow::fs::FileSystem> fs,
                                const std::string& path,
                                parquet::ReaderProperties reader_props) {
  // Open the file
  auto result = MakeArrowFileReader(*fs_, file_path_, reader_props);
  if (!result.ok()) {
    return Status::ReaderError("Error making file reader:" + result.status().ToString());
  }
  file_reader_ = std::move(result.value());

  auto metadata = file_reader_->parquet_reader()->metadata();
  ASSIGN_OR_RETURN_NOT_OK(file_metadata_, PackedFileMetadata::Make(metadata));

  std::shared_ptr<arrow::Schema> file_schema;
  auto status = file_reader_->GetSchema(&file_schema);
  if (!status.ok()) {
    return Status::ReaderError("Failed to get schema from file: " + status.ToString());
  }
  schema_ = file_schema;

  // Convert needed column names to column indices
  std::vector<int> column_indices;
  if (ChunkReader::needed_columns_.empty()) {
    for (int i = 0; i < schema_->num_fields(); ++i) {
      column_indices.push_back(i);
    }
  } else {
    for (const auto& col_name : ChunkReader::needed_columns_) {
      int col_index = schema_->GetFieldIndex(col_name);
      if (col_index >= 0) {
        column_indices.push_back(col_index);
      } else {
        return Status::InvalidArgument("Column " + col_name + " not found in schema for file: " + file_path_);
      }
    }
  }
  needed_column_indices_ = column_indices;

  // Precompute number of rows until each chunk for efficient binary search for get_chunk_indices
  auto row_group_metadata = file_metadata_->GetRowGroupMetadataVector();
  num_rows_until_chunk_.reserve(row_group_metadata.size() + 1);
  num_rows_until_chunk_.push_back(0);  // First element is always 0
  for (size_t i = 0; i < row_group_metadata.size(); ++i) {
    num_rows_until_chunk_.push_back(num_rows_until_chunk_.back() + row_group_metadata.Get(i).row_num());
  }

  return Status::OK();
}

std::shared_ptr<arrow::Schema> ParquetChunkReader::schema() const { return schema_; }

arrow::Result<std::vector<int64_t>> ParquetChunkReader::get_chunk_indices(
    const std::vector<int64_t>& row_indices) const {
  if (num_rows_until_chunk_.empty()) {
    return arrow::Status::Invalid("Chunk row counts not initialized");
  }

  std::vector<int64_t> chunk_indices;
  chunk_indices.reserve(row_indices.size());

  int64_t total_rows = num_rows_until_chunk_.back();

  for (int64_t row_index : row_indices) {
    if (row_index < 0) {
      return arrow::Status::Invalid("Row index cannot be negative: " + std::to_string(row_index));
    }

    if (row_index >= total_rows) {
      return arrow::Status::Invalid("Row index " + std::to_string(row_index) + " is out of range. File has " +
                                    std::to_string(total_rows) + " rows");
    }
    auto it = std::upper_bound(num_rows_until_chunk_.begin(), num_rows_until_chunk_.end(), row_index);
    int64_t chunk_index = std::distance(num_rows_until_chunk_.begin(), it) - 1;

    chunk_indices.push_back(chunk_index);
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetChunkReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  if (!file_reader_) {
    return arrow::Status::Invalid("File reader not initialized");
  }

  std::shared_ptr<arrow::Table> table;
  auto status = file_reader_->ReadRowGroup(static_cast<int>(chunk_index), needed_column_indices_, &table);
  if (!status.ok()) {
    return status;
  }

  if (!table) {
    return arrow::Status::Invalid("Failed to read row group " + std::to_string(chunk_index));
  }
  return ConvertTableToRecordBatch(table);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetChunkReader::get_chunk_range(
    int64_t start_chunk_index, int64_t chunk_count) const {
  // Validate chunk indices
  for (int64_t i = 0; i < chunk_count; ++i) {
    ARROW_RETURN_NOT_OK(validate_chunk_index(start_chunk_index + i));
  }

  if (!file_reader_) {
    return arrow::Status::Invalid("File reader not initialized");
  }

  if (chunk_count <= 0) {
    return std::vector<std::shared_ptr<arrow::RecordBatch>>();
  }

  // Use ParquetReader's ReadRowGroups for optimal performance with continuous chunks
  std::vector<int> row_group_indices;
  row_group_indices.reserve(chunk_count);
  for (int64_t i = 0; i < chunk_count; ++i) {
    row_group_indices.push_back(static_cast<int>(start_chunk_index + i));
  }

  std::shared_ptr<arrow::Table> table;
  auto status = file_reader_->ReadRowGroups(row_group_indices, needed_column_indices_, &table);
  if (!status.ok()) {
    return status;
  }

  if (!table) {
    return arrow::Status::Invalid("Failed to read row groups");
  }

  // Convert the combined table to individual record batches per chunk
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  result.reserve(chunk_count);

  int64_t current_row_offset = 0;
  for (int64_t i = 0; i < chunk_count; ++i) {
    int64_t chunk_idx = start_chunk_index + i;
    auto chunk_row_count_result = get_chunk_row_num(chunk_idx);
    if (!chunk_row_count_result.ok()) {
      return chunk_row_count_result.status();
    }
    int64_t chunk_row_count = chunk_row_count_result.ValueOrDie();

    // Slice the table to get this chunk's data
    auto chunk_table = table->Slice(current_row_offset, chunk_row_count);
    auto batch_result = ConvertTableToRecordBatch(chunk_table);
    if (!batch_result.ok()) {
      return batch_result.status();
    }

    result.push_back(batch_result.ValueOrDie());
    current_row_offset += chunk_row_count;
  }

  return result;
}

arrow::Result<int64_t> ParquetChunkReader::get_chunk_size(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  if (!file_metadata_) {
    return arrow::Status::Invalid("File metadata not initialized");
  }

  return file_metadata_->GetRowGroupMetadataVector().Get(chunk_index).memory_size();
}

arrow::Result<int64_t> ParquetChunkReader::get_chunk_row_num(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  if (!file_metadata_) {
    return arrow::Status::Invalid("File metadata not initialized");
  }

  return file_metadata_->GetRowGroupMetadataVector().Get(chunk_index).row_num();
}

arrow::Status ParquetChunkReader::validate_chunk_index(int64_t chunk_index) const {
  if (chunk_index < 0) {
    return arrow::Status::Invalid("Chunk index cannot be negative: " + std::to_string(chunk_index));
  }

  if (!file_metadata_) {
    return arrow::Status::Invalid("File metadata not initialized");
  }

  auto row_group_metadata = file_metadata_->GetRowGroupMetadataVector();
  if (chunk_index >= static_cast<int64_t>(row_group_metadata.size())) {
    return arrow::Status::Invalid("Chunk index " + std::to_string(chunk_index) + " is out of range. File has " +
                                  std::to_string(row_group_metadata.size()) + " chunks");
  }

  return arrow::Status::OK();
}

}  // namespace milvus_storage::api
