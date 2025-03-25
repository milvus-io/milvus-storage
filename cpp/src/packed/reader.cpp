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

#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/array/array_base.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <parquet/properties.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/packed/chunk_manager.h"
#include "milvus-storage/packed/reader.h"

namespace milvus_storage {

PackedRecordBatchReader::PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 std::vector<std::string>& paths,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 int64_t buffer_size)
    : buffer_available_(buffer_size),
      memory_limit_(buffer_size),
      row_limit_(0),
      absolute_row_position_(0),
      read_count_(0) {
  init(fs, paths, schema, buffer_size);
}

Status PackedRecordBatchReader::init(std::shared_ptr<arrow::fs::FileSystem> fs,
                                     std::vector<std::string>& paths,
                                     std::shared_ptr<arrow::Schema> schema,
                                     int64_t buffer_size) {
  // read first file metadata to get field id mapping and do schema matching
  RETURN_NOT_OK(schemaMatching(fs, schema, paths));

  // init arrow file readers and metadata list
  for (auto path : needed_paths_) {
    auto result = MakeArrowFileReader(*fs, path);
    if (!result.ok()) {
      return Status::ArrowError("Error making file reader with path " + path + ":" + result.status().ToString());
    }
    auto file_reader = std::move(result.value());
    auto metadata = file_reader->parquet_reader()->metadata();
    ASSIGN_OR_RETURN_NOT_OK(auto file_metadata, PackedFileMetadata::Make(metadata));
    metadata_list_.push_back(std::move(file_metadata));
    file_readers_.push_back(std::move(file_reader));
  }

  // Initialize table states and chunk manager
  column_group_states_.resize(file_readers_.size(), ColumnGroupState(0, -1, 0));
  chunk_manager_ = std::make_unique<ChunkManager>(needed_column_offsets_, 0);
  // tables are referrenced by column_offsets, so it's size is of paths's size.
  for (int i = 0; i < needed_paths_.size(); i++) {
    tables_.push_back(std::queue<std::shared_ptr<arrow::Table>>());
  }
  return Status::OK();
}

Status PackedRecordBatchReader::schemaMatching(std::shared_ptr<arrow::fs::FileSystem> fs,
                                               std::shared_ptr<arrow::Schema> schema,
                                               std::vector<std::string>& paths) {
  // read first file metadata to get field id mapping
  auto result = MakeArrowFileReader(*fs, paths[0]);
  if (!result.ok()) {
    return Status::ArrowError("Error making file reader with path " + paths[0] + ":" + result.status().ToString());
  }
  auto parquet_metadata = result.value()->parquet_reader()->metadata();
  ASSIGN_OR_RETURN_NOT_OK(auto metadata, PackedFileMetadata::Make(parquet_metadata));

  // parse field id list from schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Field>> needed_fields;
  auto status = FieldIDList::Make(schema);
  if (!status.ok()) {
    return Status::MetadataParseError("Error getting field id list from schema: " + schema->ToString());
  }
  field_id_list_ = status.value();

  // schema matching
  field_id_mapping_ = metadata->GetFieldIDMapping();
  for (int i = 0; i < field_id_list_.size(); ++i) {
    FieldID field_id = field_id_list_.Get(i);
    if (field_id_mapping_.find(field_id) != field_id_mapping_.end()) {
      auto column_offset = field_id_mapping_[field_id];
      needed_column_offsets_.push_back(column_offset);
      needed_paths_.emplace(paths[column_offset.path_index]);
      fields.push_back(schema->field(i));
      needed_fields.push_back(schema->field(i));
    } else {
      // mark nullable if the field can not be found in the file, in case the reader schema is not marked
      fields.push_back(schema->field(i)->WithNullable(true));
    }
  }
  needed_schema_ = std::make_shared<arrow::Schema>(needed_fields);
  schema_ = std::make_shared<arrow::Schema>(fields);
  return Status::OK();
}

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return schema_; }

std::shared_ptr<PackedFileMetadata> PackedRecordBatchReader::file_metadata(int i) {
  if (i < 0 || i >= metadata_list_.size()) {
    return nullptr;
  }
  return metadata_list_[i];
}

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
    int64_t rg_size = metadata_list_[i]->GetRowGroupSize(rg);
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
  int64_t chunk_size = chunk_manager_->GetChunkSize();
  absolute_row_position_ += chunk_size;
  std::shared_ptr<arrow::RecordBatch> batch =
      arrow::RecordBatch::Make(needed_schema_, chunk_size, std::move(batch_data));

  // match schema and fill null columns
  int batch_index = 0;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (int i = 0; i < field_id_list_.size(); ++i) {
    FieldID field_id = field_id_list_.Get(i);
    if (field_id_mapping_.find(field_id) != field_id_mapping_.end()) {
      arrays.push_back(std::move(batch->column(batch_index++)));
    } else {
      auto null_array = arrow::MakeArrayOfNull(schema_->field(i)->type(), chunk_size).ValueOrDie();
      arrays.push_back(std::move(null_array));
    }
  }
  *out = arrow::RecordBatch::Make(schema_, chunk_size, arrays);
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
  metadata_list_.clear();
  return arrow::Status::OK();
}

}  // namespace milvus_storage