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
#include <utility>

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

#include "milvus-storage/format/parquet/reader.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::parquet {

arrow::Status ParquetChunkReader::init() {
  if (!file_readers_.empty()) {
    return arrow::Status::OK();
  }
  // Open files and read metadata
  size_t file_rows = 0;
  size_t file_row_groups = 0;
  for (size_t i = 0; i < paths_.size(); ++i) {
    auto result = MakeArrowFileReader(*fs_, paths_[i], reader_props_);
    if (!result.ok()) {
      return arrow::Status::Invalid("Error making file reader:" + result.status().ToString());
    }
    file_readers_.push_back(std::move(result.value()));

    auto metadata = file_readers_[i]->parquet_reader()->metadata();
    auto metadata_result = PackedFileMetadata::Make(metadata);
    if (!metadata_result.ok()) {
      return arrow::Status::Invalid("Error making file metadata:" + metadata_result.status().ToString());
    }
    file_metadatas_.push_back(metadata_result.value());

    // Calculate number of rows until each chunk for efficient binary search for get_chunk_indices
    // TODO: lazily read row group metadata.
    auto row_group_metadata = file_metadatas_[i]->GetRowGroupMetadataVector();

    size_t rows = 0;
    for (size_t j = 0; j < row_group_metadata.size(); ++j) {
      auto size = row_group_metadata.Get(j).memory_size();
      rows += row_group_metadata.Get(j).row_num();
      row_group_indices_.push_back(RowGroupIndex{i, j, rows, size, file_row_groups + j, file_rows + rows});
    }
    file_rows += rows;
    file_row_groups += row_group_metadata.size();
  }

  if (schema_ == nullptr) {
    std::shared_ptr<arrow::Schema> file_schema;
    auto status = file_readers_[0]->GetSchema(&file_schema);
    if (!status.ok()) {
      return status;
    }
    schema_ = file_schema;
  }

  // Convert needed column names to column indices
  std::vector<int> column_indices;
  if (needed_columns_.empty()) {
    for (int i = 0; i < schema_->num_fields(); ++i) {
      column_indices.push_back(i);
    }
  } else {
    for (const auto& col_name : needed_columns_) {
      int col_index = schema_->GetFieldIndex(col_name);
      if (col_index >= 0) {
        column_indices.push_back(col_index);
      } else {
        return arrow::Status::Invalid("Column " + col_name + " not found in schema");
      }
    }
  }
  needed_column_indices_ = column_indices;
  return arrow::Status::OK();
}

arrow::Result<std::vector<int64_t>> ParquetChunkReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }

  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(row_group_indices_.begin(), row_group_indices_.end(), row_index,
                               [](int64_t a, const RowGroupIndex& b) { return a < b.row_index; });
    auto chunk_index = std::distance(row_group_indices_.begin(), it);
    chunk_indices.push_back(chunk_index);
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetChunkReader::get_chunk(int64_t chunk_index) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }
  if (chunk_index < 0 || chunk_index >= row_group_indices_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(row_group_indices_.size()));
  }
  auto row_group_index = row_group_indices_[chunk_index];
  std::shared_ptr<arrow::Table> table;
  status = file_readers_[row_group_index.file_index]->ReadRowGroup(row_group_index.row_group_index_in_file,
                                                                   needed_column_indices_, &table);
  if (!status.ok()) {
    return status;
  }

  if (!table) {
    return arrow::Status::Invalid("Failed to read row group " + std::to_string(chunk_index));
  }
  return ConvertTableToRecordBatch(table);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> result;

  auto read_file = [&](size_t file_index, int from_include, int64_t to_not_include) -> arrow::Status {
    // read row groups from last file
    auto reader = file_readers_[file_index];
    std::shared_ptr<arrow::Table> table;
    std::vector<int> current_row_group_indices;
    for (int64_t j = from_include; j < to_not_include; ++j) {
      current_row_group_indices.push_back(row_group_indices_[j].row_group_index_in_file);
    }
    auto status = reader->ReadRowGroups(current_row_group_indices, needed_column_indices_, &table);
    if (!status.ok()) {
      return status;
    }
    // TODO: ConvertTableToRecordBatch require copy, is not efficient
    auto batch_result = ConvertTableToRecordBatch(table);
    if (!batch_result.ok()) {
      return batch_result.status();
    }
    result.push_back(batch_result.ValueOrDie());
    return arrow::Status::OK();
  };

  // loop through chunk_indices and read row groups
  auto file_index = -1;
  auto chunk_index_from = -1;
  for (int64_t i = 0; i < chunk_indices.size(); ++i) {
    auto row_group_index = row_group_indices_[i];

    if (i == 0) {
      file_index = row_group_index.file_index;
      chunk_index_from = 0;
      continue;
    }

    if (row_group_index.file_index != file_index) {
      status = read_file(file_index, chunk_index_from, i);
      if (!status.ok()) {
        return status;
      }
      file_index = row_group_index.file_index;
      chunk_index_from = i;
    }
  }

  // read row groups from last file
  status = read_file(file_index, chunk_index_from, chunk_indices.size());
  if (!status.ok()) {
    return status;
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetChunkReader::take(const std::vector<int64_t>& row_indices) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }

  ARROW_ASSIGN_OR_RAISE(auto chunk_indices, get_chunk_indices(row_indices));

  std::set<int64_t> unique_chunks(chunk_indices.begin(), chunk_indices.end());
  std::vector<int64_t> chunks_to_read(unique_chunks.begin(), unique_chunks.end());
  ARROW_ASSIGN_OR_RAISE(auto chunks, get_chunks(chunks_to_read));

  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_map;
  for (size_t i = 0; i < chunks_to_read.size(); ++i) {
    chunk_map[chunks_to_read[i]] = chunks[i];
  }

  // extract data for each target row
  std::vector<std::shared_ptr<arrow::RecordBatch>> row_slices;
  for (size_t i = 0; i < row_indices.size(); ++i) {
    auto global_row = row_indices[i];
    auto chunk_idx = chunk_indices[i];
    auto chunk = chunk_map[chunk_idx];
    // calculate local row number in chunk
    int64_t chunk_start = chunk_idx > 0 ? row_group_indices_[chunk_idx - 1].row_index : 0;
    int64_t local_row = global_row - chunk_start;

    // extract this row
    row_slices.push_back(chunk->Slice(local_row, 1));
  }

  // collapse row slices
  auto combined_table = arrow::Table::FromRecordBatches(row_slices);
  if (!combined_table.ok()) {
    return combined_table.status();
  }
  auto combined_batch = combined_table.ValueOrDie()->CombineChunksToBatch();
  if (!combined_batch.ok()) {
    return combined_batch.status();
  }
  return combined_batch.ValueOrDie();
}

arrow::Result<int64_t> ParquetChunkReader::get_chunk_size(int64_t chunk_index) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }
  if (chunk_index < 0 || chunk_index >= row_group_indices_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(row_group_indices_.size()));
  }
  return row_group_indices_[chunk_index].size;
}

arrow::Result<int64_t> ParquetChunkReader::get_chunk_rows(int64_t chunk_index) {
  auto status = init();
  if (!status.ok()) {
    return status;
  }
  if (chunk_index < 0 || chunk_index >= row_group_indices_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(row_group_indices_.size()));
  }
  auto file_index = row_group_indices_[chunk_index].file_index;
  auto row_group_index_in_file = row_group_indices_[chunk_index].row_group_index_in_file;
  return file_metadatas_[file_index]->GetRowGroupMetadataVector().Get(row_group_index_in_file).row_num();
}

}  // namespace milvus_storage::parquet
