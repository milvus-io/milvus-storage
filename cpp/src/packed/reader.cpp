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

#include "milvus-storage/packed/reader.h"
#include <arrow/array/array_base.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/properties.h>
#include <memory>
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/packed/chunk_manager.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/serde.h"
#include "milvus-storage/common/path_util.h"

namespace milvus_storage {

PackedRecordBatchReader::PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                                                 const std::string& file_path,
                                                 const std::shared_ptr<arrow::Schema> origin_schema,
                                                 const std::set<int>& needed_columns,
                                                 const int64_t buffer_size)
    : file_path_(file_path),
      origin_schema_(origin_schema),
      buffer_available_(buffer_size),
      memory_limit_(buffer_size),
      row_limit_(0),
      absolute_row_position_(0),
      read_count_(0) {
  init(fs, file_path_, origin_schema_, needed_columns, buffer_size);
}

void PackedRecordBatchReader::init(arrow::fs::FileSystem& fs,
                                   const std::string& file_path,
                                   const std::shared_ptr<arrow::Schema> origin_schema,
                                   const std::set<int>& needed_columns,
                                   const int64_t buffer_size) {
  // init needed schema
  auto status = initNeededSchema(needed_columns, origin_schema);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  // init column offsets
  status = initColumnOffsets(fs, needed_columns, origin_schema->num_fields());
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  // init arrow file readers
  for (auto i : needed_paths_) {
    auto result = MakeArrowFileReader(fs, ConcatenateFilePath(file_path_, std::to_string(i)));
    if (!result.ok()) {
      LOG_STORAGE_ERROR_ << "Error making file reader " << i << ":" << result.status().ToString();
      throw std::runtime_error(result.status().ToString());
    }
    file_readers_.emplace_back(std::move(result.value()));
  }

  // init uncompressed row group sizes from metadata
  for (int i = 0; i < file_readers_.size(); ++i) {
    auto metadata = file_readers_[i]->parquet_reader()->metadata()->key_value_metadata();

    auto row_group_size_meta = metadata->Get(ROW_GROUP_SIZE_META_KEY);
    if (!row_group_size_meta.ok()) {
      LOG_STORAGE_ERROR_ << "row group size meta not found in file " << i;
      throw std::runtime_error(row_group_size_meta.status().ToString());
    }
    row_group_sizes_.push_back(PackedMetaSerde::DeserializeRowGroupSizes(row_group_size_meta.ValueOrDie()));
    LOG_STORAGE_DEBUG_ << " file " << i << " metadata size: " << file_readers_[i]->parquet_reader()->metadata()->size();
  }

  // Initialize table states and chunk manager
  column_group_states_.resize(file_readers_.size(), ColumnGroupState(0, -1, 0));
  chunk_manager_ = std::make_unique<ChunkManager>(needed_column_offsets_, 0);
  // tables are referrenced by column_offsets, so it's size is of paths's size.
  tables_.resize(needed_paths_.size(), std::queue<std::shared_ptr<arrow::Table>>());
}

Status PackedRecordBatchReader::initColumnOffsets(arrow::fs::FileSystem& fs,
                                                  const std::set<int>& needed_columns,
                                                  size_t num_fields) {
  std::string path = ConcatenateFilePath(file_path_, std::to_string(0));
  auto reader = MakeArrowFileReader(fs, path);
  if (!reader.ok()) {
    return Status::ReaderError("can not open file reader");
  }
  auto metadata = reader.value()->parquet_reader()->metadata()->key_value_metadata();
  auto column_offset_meta = metadata->Get(COLUMN_OFFSETS_META_KEY);
  if (!column_offset_meta.ok()) {
    return Status::ReaderError("can not find column offset meta");
  }
  auto group_indices = PackedMetaSerde::DeserializeColumnOffsets(column_offset_meta.ValueOrDie());
  std::vector<ColumnOffset> offsets(num_fields);
  for (int path_index = 0; path_index < group_indices.size(); path_index++) {
    for (int col_index = 0; col_index < group_indices[path_index].size(); col_index++) {
      int origin_col = group_indices[path_index][col_index];
      offsets[origin_col] = ColumnOffset(path_index, col_index);
    }
  }
  for (int col : needed_columns) {
    needed_paths_.emplace(offsets[col].path_index);
    needed_column_offsets_.push_back(offsets[col]);
  }
  return Status::OK();
}

Status PackedRecordBatchReader::initNeededSchema(const std::set<int>& needed_columns, const std::shared_ptr<arrow::Schema> schema) {
  std::vector<std::shared_ptr<arrow::Field>> needed_fields;

  for (int col : needed_columns) {
    if (col < 0 || col >= schema->num_fields()) {
      return Status::ReaderError("Specified column index" + std::to_string(col) +
        " is out of bounds. Schema has " + std::to_string(schema->num_fields()) + " fields."
      );
    }
    needed_fields.push_back(schema->field(col));
  }
  needed_schema_ = std::make_shared<arrow::Schema>(needed_fields);
  return Status::OK();
}

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return needed_schema_; }

arrow::Status PackedRecordBatchReader::advanceBuffer() {
  std::vector<std::vector<int>> rgs_to_read(file_readers_.size());
  size_t plan_buffer_size = 0;

  // Advances to the next row group in a specific file reader and calculates the required buffer size.
  auto advance_row_group = [&](int i) -> int64_t {
    auto& reader = file_readers_[i];
    int rg = column_group_states_[i].row_group_offset + 1;
    int num_row_groups = reader->parquet_reader()->metadata()->num_row_groups();
    if (rg >= num_row_groups) {
      // No more row groups. It means we're done or there is an error.
      LOG_STORAGE_DEBUG_ << "No more row groups in file " << i << " total row groups " << num_row_groups;
      return -1;
    }
    int64_t rg_size = row_group_sizes_[i][rg];
    if (plan_buffer_size + rg_size >= buffer_available_) {
      LOG_STORAGE_DEBUG_ << "buffer is full now " << i;
      return -1;
    }
    rgs_to_read[i].push_back(rg);
    const auto metadata = reader->parquet_reader()->metadata()->RowGroup(rg);
    plan_buffer_size += rg_size;
    column_group_states_[i].addMemorySize(rg_size);
    column_group_states_[i].setRowGroupOffset(rg);
    column_group_states_[i].addRowOffset(metadata->num_rows());
    return rg_size;
  };

  // Fill in tables that have no rows available
  int drained_index = -1;
  for (int i = 0; i < file_readers_.size(); ++i) {
    if (column_group_states_[i].row_offset > row_limit_) {
      continue;
    }
    buffer_available_ += column_group_states_[i].memory_size;
    column_group_states_[i].resetMemorySize();
    if (advance_row_group(i) < 0) {
      LOG_STORAGE_DEBUG_ << "No more row groups in file " << i;
      drained_index = i;
      break;
    }
    chunk_manager_->ResetChunkState(i);
  }

  if (drained_index >= 0) {
    if (plan_buffer_size == 0) {
      // If nothing to fill, it must be done
      return arrow::Status::OK();
    } else {
      // Otherwise, the rows are not match, there is something wrong with the files.
      return arrow::Status::Invalid("File broken at index " + std::to_string(drained_index));
    }
  }

  // Fill in tables if we have enough buffer size
  // find the lowest offset table and advance it
  RowOffsetMinHeap sorted_offsets;
  for (int i = 0; i < file_readers_.size(); ++i) {
    sorted_offsets.emplace(i, column_group_states_[i].row_offset);
  }
  while (true) {
    int i = sorted_offsets.top().first;
    if (advance_row_group(i) < 0) {
      break;
    }
    sorted_offsets.pop();
    sorted_offsets.emplace(i, column_group_states_[i].row_offset);
  }

  // Conduct read and update buffer size
  for (int i = 0; i < file_readers_.size(); ++i) {
    if (rgs_to_read[i].empty()) {
      continue;
    }
    read_count_++;
    LOG_STORAGE_DEBUG_ << "File reader " << i << " advanced to row group " << rgs_to_read[i].back();
    column_group_states_[i].read_times++;
    std::shared_ptr<arrow::Table> read_table = nullptr;
    RETURN_NOT_OK(file_readers_[i]->ReadRowGroups(rgs_to_read[i], &read_table));
    tables_[i].push(std::move(read_table));
  }
  buffer_available_ -= plan_buffer_size;
  row_limit_ = sorted_offsets.top().second;
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  if (absolute_row_position_ >= row_limit_) {
    RETURN_NOT_OK(advanceBuffer());
    if (absolute_row_position_ >= row_limit_) {
      *out = nullptr;
      return arrow::Status::OK();
    }
  }

  // Determine the maximum contiguous slice across all tables
  auto batch_data = chunk_manager_->SliceChunksByMaxContiguousSlice(row_limit_ - absolute_row_position_, tables_);
  absolute_row_position_ += chunk_manager_->GetChunkSize();
  *out = arrow::RecordBatch::Make(needed_schema_, chunk_manager_->GetChunkSize(), std::move(batch_data));
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::Close() {
  LOG_STORAGE_DEBUG_ << "PackedRecordBatchReader::Close(), total read " << read_count_ << " times";
  for (int i = 0; i < column_group_states_.size(); ++i) {
    LOG_STORAGE_DEBUG_ << "File reader " << i << " read " << column_group_states_[i].read_times << " times";
  }
  read_count_ = 0;
  column_group_states_.clear();
  tables_.clear();
  file_readers_.clear();
  chunk_manager_.release();
  return arrow::Status::OK();
}

}  // namespace milvus_storage