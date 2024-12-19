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

#include "packed/mem_record_reader.h"
#include <parquet/properties.h>
#include <memory>
#include <limits>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/util/logging.h>
#include "packed/utils/serde.h"
#include "common/log.h"
#include "common/arrow_util.h"

namespace milvus_storage {

MemRecordBatchReader::MemRecordBatchReader(arrow::fs::FileSystem& fs,
                                           const std::string& path,
                                           const std::shared_ptr<arrow::Schema>& schema,
                                           const size_t row_group_offset,
                                           const size_t row_group_num,
                                           const int64_t buffer_size)
    : schema_(schema), row_group_offset_(row_group_offset), buffer_size_(buffer_size) {
  auto result = MakeArrowFileReader(fs, path);
  if (!result.ok()) {
    LOG_STORAGE_ERROR_ << "Error making file reader:" << result.status().ToString();
    throw std::runtime_error(result.status().ToString());
  }
  file_reader_ = std::move(result.value());

  auto metadata = file_reader_->parquet_reader()->metadata()->key_value_metadata()->Get(ROW_GROUP_SIZE_META_KEY);
  if (!metadata.ok()) {
    LOG_STORAGE_ERROR_ << "metadata not found in file: " << path;
    throw std::runtime_error(metadata.status().ToString());
  }
  auto all_row_group_sizes = PackedMetaSerde::deserialize(metadata.ValueOrDie());
  cout << "all_row_group_sizes: " << all_row_group_sizes.size() << endl;
  cout << "row_group_offset: " << row_group_offset << endl;
  cout << "row_group_num: " << row_group_num << endl;
  cout << "end_offset: " << row_group_offset + row_group_num << endl;
  size_t end_offset = row_group_offset + row_group_num;
  if (row_group_offset >= all_row_group_sizes.size() || end_offset > all_row_group_sizes.size()) {
    LOG_STORAGE_ERROR_ << "Row group range exceeds total number of row groups";
    throw std::out_of_range("Row group range exceeds total number of row groups");
  }
  row_group_sizes_.assign(all_row_group_sizes.begin() + row_group_offset, all_row_group_sizes.begin() + end_offset);
}

std::shared_ptr<arrow::Schema> MemRecordBatchReader::schema() const { return schema_; }

arrow::Status MemRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
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

arrow::Status MemRecordBatchReader::Close() {
  LOG_STORAGE_DEBUG_ << "MemRecordBatchReader closed after reading " << read_count_ << " times.";
  file_reader_ = nullptr;
  schema_ = nullptr;
  row_group_sizes_.clear();
  return arrow::Status::OK();
}

}  // namespace milvus_storage
