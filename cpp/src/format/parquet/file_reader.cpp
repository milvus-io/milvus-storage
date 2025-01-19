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

#include "milvus-storage/format/parquet/file_reader.h"

#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/type_fwd.h>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>
#include "arrow/table.h"
#include "common/macro.h"
#include "common/serde.h"
#include "common/log.h"
#include "common/arrow_util.h"

namespace milvus_storage {

FileRecordBatchReader::FileRecordBatchReader(arrow::fs::FileSystem& fs,
                                             const std::string& path,
                                             const std::shared_ptr<arrow::Schema>& schema,
                                             const int64_t buffer_size,
                                             const size_t row_group_offset,
                                             const size_t row_group_num)
    : schema_(schema), row_group_offset_(row_group_offset), buffer_size_(buffer_size) {
  auto result = MakeArrowFileReader(fs, path);
  if (!result.ok()) {
    LOG_STORAGE_ERROR_ << "Error making file reader:" << result.status().ToString();
    throw std::runtime_error(result.status().ToString());
  }
  file_reader_ = std::move(result.value());

  auto metadata = file_reader_->parquet_reader()->metadata()->key_value_metadata()->Get(ROW_GROUP_SIZE_META_KEY);
  if (!metadata.ok()) {
    LOG_STORAGE_ERROR_ << "Metadata not found in file: " << path;
    throw std::runtime_error(metadata.status().ToString());
  }
  auto all_row_group_sizes = PackedMetaSerde::DeserializeRowGroupSizes(metadata.ValueOrDie());
  if (row_group_offset >= all_row_group_sizes.size()) {
    std::string error_msg =
        "Row group offset exceeds total number of row groups. "
        "Row group offset: " +
        std::to_string(row_group_offset) + ", Total row groups: " + std::to_string(all_row_group_sizes.size());
    LOG_STORAGE_ERROR_ << error_msg;
    throw std::out_of_range(error_msg);
  }
  size_t end_offset = std::min(row_group_offset + row_group_num, all_row_group_sizes.size());
  row_group_sizes_.assign(all_row_group_sizes.begin() + row_group_offset, all_row_group_sizes.begin() + end_offset);
}

std::shared_ptr<arrow::Schema> FileRecordBatchReader::schema() const { return schema_; }

arrow::Status FileRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  std::vector<int> rgs_to_read;
  size_t buffer_size = 0;

  while (current_row_group_ < row_group_sizes_.size() &&
         buffer_size + row_group_sizes_[current_row_group_] <= buffer_size_) {
    rgs_to_read.push_back(current_row_group_ + row_group_offset_);
    buffer_size += row_group_sizes_[current_row_group_];
    current_row_group_++;
  }

  if (rgs_to_read.empty()) {
    *out = nullptr;
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Table> table = nullptr;
  RETURN_NOT_OK(file_reader_->ReadRowGroups(rgs_to_read, &table));
  *out = table->CombineChunksToBatch().ValueOrDie();
  return arrow::Status::OK();
}

arrow::Status FileRecordBatchReader::Close() {
  LOG_STORAGE_DEBUG_ << "FileRecordBatchReader closed after reading " << read_count_ << " times.";
  file_reader_ = nullptr;
  schema_ = nullptr;
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
  int current_row_group_idx = 0;
  int64_t total_skipped = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::unique_ptr<arrow::RecordBatchReader> current_row_group_reader;

  for (auto& offset : offsets) {
    // skip row groups
    // TODO: to make read more efficient, we should find offsets belonged to a row group and read together.
    while (current_row_group_idx < num_row_groups) {
      auto row_group_meta = reader_->parquet_reader()->metadata()->RowGroup(current_row_group_idx);
      auto row_group_num_rows = row_group_meta->num_rows();
      if (row_group_num_rows + total_skipped > offset) {
        break;
      }
      current_row_group_idx++;
      total_skipped += row_group_num_rows;
      current_row_group_reader = nullptr;
    }

    if (current_row_group_idx >= num_row_groups) {
      break;
    }

    if (current_row_group_reader == nullptr) {
      RETURN_ARROW_NOT_OK(reader_->GetRecordBatchReader({current_row_group_idx}, &current_row_group_reader));
    }

    auto row_group_offset = offset - total_skipped;
    ASSIGN_OR_RETURN_NOT_OK(auto batch, GetRecordAtOffset(current_row_group_reader.get(), row_group_offset))
    batches.push_back(batch);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(batches));
  return res;
}
}  // namespace milvus_storage
