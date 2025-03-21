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

#include <numeric>

#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>

#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/status.h"

namespace milvus_storage {

FileRecordBatchReader::FileRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                             const FieldID field_id,
                                             const std::string& path,
                                             const int64_t buffer_size) {
  auto status = init(fs, path, buffer_size);
  if (!status.ok()) {
    LOG_STORAGE_ERROR_ << "Error initializing file reader: " << status.ToString();
    throw std::runtime_error(status.ToString());
  }
  std::shared_ptr<arrow::Schema> arrow_schema = schema();
  FieldIDList field_ids = FieldIDList::Make(arrow_schema).value();
  for (int i = 0; i < field_ids.size(); i++) {
    if (field_ids.Get(i) == field_id) {
      needed_columns_ = {i};
      break;
    }
  }
  if (needed_columns_.empty()) {
    LOG_STORAGE_ERROR_ << "Field id " << field_id << " not found in schema: " << field_id;
    throw std::runtime_error("Field id not found in schema");
  }
}

FileRecordBatchReader::FileRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                             const std::string& path,
                                             const int64_t buffer_size) {
  auto status = init(fs, path, buffer_size);
  if (!status.ok()) {
    LOG_STORAGE_ERROR_ << "Error initializing file reader: " << status.ToString();
    throw std::runtime_error(status.ToString());
  }
  needed_columns_.resize(metadata_->schema()->num_columns());
  std::iota(needed_columns_.begin(), needed_columns_.end(), 0);
}

Status FileRecordBatchReader::init(std::shared_ptr<arrow::fs::FileSystem> fs,
                                   const std::string& path,
                                   const int64_t buffer_size) {
  buffer_size_limit_ = buffer_size;
  auto result = MakeArrowFileReader(*fs, path);
  if (!result.ok()) {
    return Status::ReaderError("Error making file reader:" + result.status().ToString());
  }
  file_reader_ = std::move(result.value());

  metadata_ = file_reader_->parquet_reader()->metadata();
  auto row_group_sizes_meta = metadata_->key_value_metadata()->Get(ROW_GROUP_SIZE_META_KEY);
  if (!row_group_sizes_meta.ok()) {
    return Status::ReaderError("Row group size metadata not found in file: " + path);
  }
  row_group_sizes_ = std::move(RowGroupSizeVector::Deserialize(row_group_sizes_meta.ValueOrDie()));
  return Status::OK();
}

RowGroupSizeVector FileRecordBatchReader::GetRowGroupSizes() { return row_group_sizes_; }

std::shared_ptr<arrow::Schema> FileRecordBatchReader::schema() const {
  std::shared_ptr<arrow::Schema> arrow_schema;
  auto status = parquet::arrow::FromParquetSchema(metadata_->schema(), &arrow_schema);
  if (!status.ok()) {
    LOG_STORAGE_ERROR_ << "can not get arrow schema from parquet schema: " << status.message();
    throw std::runtime_error(status.message());
  }
  return arrow_schema;
}

Status FileRecordBatchReader::SetRowGroupOffsetAndCount(int row_group_offset, int row_group_num) {
  if (row_group_offset < 0 || row_group_num <= 0) {
    return Status::InvalidArgument("please provide row group offset and row group num");
  }
  if (row_group_offset >= row_group_sizes_.size() || row_group_offset + row_group_num > row_group_sizes_.size()) {
    std::string error_msg =
        "Row group range exceeds total number of row groups: " + std::to_string(row_group_sizes_.size());
    return Status::InvalidArgument(error_msg);
  }
  rg_start_ = row_group_offset;
  rg_end_ = row_group_offset + row_group_num - 1;
  ;
  return Status::OK();
}

arrow::Status FileRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  if (rg_start_ == -1 || rg_start_ > rg_end_ || rg_start_ >= row_group_sizes_.size()) {
    LOG_STORAGE_WARNING_ << "Please set row group offset and count before reading next.";
    rg_start_ = -1;
    rg_end_ = -1;
    *out = nullptr;
    return arrow::Status::OK();
  }
  std::vector<int> rgs_to_read;
  size_t buffer_size = 0;

  while (rg_start_ <= rg_end_ && buffer_size + row_group_sizes_.Get(rg_start_) <= buffer_size_limit_) {
    rgs_to_read.push_back(rg_start_);
    buffer_size += row_group_sizes_.Get(rg_start_);
    rg_start_++;
  }

  if (rgs_to_read.empty()) {
    rg_start_ = -1;
    rg_end_ = -1;
    *out = nullptr;
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Table> table = nullptr;
  auto status = file_reader_->ReadRowGroups(rgs_to_read, needed_columns_, &table);
  if (!status.ok()) {
    *out = nullptr;
    return status;
  }
  *out = table->CombineChunksToBatch().ValueOrDie();
  return arrow::Status::OK();
}

arrow::Status FileRecordBatchReader::Close() {
  LOG_STORAGE_DEBUG_ << "FileRecordBatchReader closed after reading " << read_count_ << " times.";
  file_reader_ = nullptr;
  row_group_sizes_.clear();
  return arrow::Status::OK();
}

ParquetFileReader::ParquetFileReader(std::unique_ptr<parquet::arrow::FileReader> reader) : reader_(std::move(reader)) {}

Result<std::shared_ptr<arrow::RecordBatch>> GetRecordAtOffset(arrow::RecordBatchReader* reader, int64_t offset) {
  int64_t skipped = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  do {
    RETURN_ARROW_NOT_OK(reader->ReadNext(&batch));
    skipped += batch->num_rows();
  } while (skipped < offset);

  auto offset_batch = offset - skipped + batch->num_rows();
  // zero-copy slice
  return batch->Slice(offset_batch, 1);
}

// TODO: support projection
Result<std::shared_ptr<arrow::Table>> ParquetFileReader::ReadByOffsets(std::vector<int64_t>& offsets) {
  std::sort(offsets.begin(), offsets.end());

  auto num_row_groups = reader_->parquet_reader()->metadata()->num_row_groups();
  int rg_idx = 0;
  int64_t total_skipped = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::unique_ptr<arrow::RecordBatchReader> rg_reader;

  for (auto& offset : offsets) {
    // skip row groups
    // TODO: to make read more efficient, we should find offsets belonged to a row group and read together.
    while (rg_idx < num_row_groups) {
      auto row_group_meta = reader_->parquet_reader()->metadata()->RowGroup(rg_idx);
      auto row_group_num_rows = row_group_meta->num_rows();
      if (row_group_num_rows + total_skipped > offset) {
        break;
      }
      rg_idx++;
      total_skipped += row_group_num_rows;
      rg_reader = nullptr;
    }

    if (rg_idx >= num_row_groups) {
      break;
    }

    if (rg_reader == nullptr) {
      RETURN_ARROW_NOT_OK(reader_->GetRecordBatchReader({rg_idx}, &rg_reader));
    }

    auto row_group_offset = offset - total_skipped;
    ASSIGN_OR_RETURN_NOT_OK(auto batch, GetRecordAtOffset(rg_reader.get(), row_group_offset))
    batches.push_back(batch);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(batches));
  return res;
}
}  // namespace milvus_storage
