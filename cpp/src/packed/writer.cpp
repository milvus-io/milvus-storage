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

#include "milvus-storage/packed/writer.h"
#include <cstddef>
#include <numeric>
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/column_group_writer.h"
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/path_util.h"

namespace milvus_storage {

PackedRecordBatchWriter::PackedRecordBatchWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 std::vector<std::string>& paths,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 StorageConfig& storage_config,
                                                 std::vector<std::vector<int>>& column_groups,
                                                 size_t buffer_size)
    : buffer_size_(buffer_size), group_indices_(column_groups), splitter_(column_groups), current_memory_usage_(0) {
  if (paths.size() != group_indices_.size()) {
    LOG_STORAGE_ERROR_ << "Mismatch between paths number and column groups number: " << paths.size() << " vs "
                       << group_indices_.size();
    return;
  }

  splitter_ = IndicesBasedSplitter(group_indices_);
  for (size_t i = 0; i < paths.size(); ++i) {
    auto sub_schema = getSubSchema(schema, group_indices_[i]);
    auto writer = std::make_unique<ColumnGroupWriter>(i, sub_schema, fs, paths[i], storage_config, group_indices_[i]);
    auto status = writer->Init();
    if (status.ok()) {
      group_writers_.emplace_back(std::move(writer));
    } else {
      LOG_STORAGE_ERROR_ << "Failed to initialize writer for column group " << i << ": " << status.ToString();
    }
  }
}

Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  size_t next_batch_size = GetRecordBatchMemorySize(record);
  if (next_batch_size > buffer_size_) {
    LOG_STORAGE_WARNING_ << "Batch size " << next_batch_size << " exceeds memory limit " << buffer_size_;
  }
  return writeWithSplitIndex(record, next_batch_size);
}

Status PackedRecordBatchWriter::writeWithSplitIndex(const std::shared_ptr<arrow::RecordBatch>& record,
                                                    size_t next_batch_size) {
  std::vector<ColumnGroup> column_groups = splitter_.Split(record);

  // Flush column groups until there's enough room for the new column groups
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + next_batch_size >= buffer_size_ && !max_heap_.empty()) {
    LOG_STORAGE_DEBUG_ << "Current memory usage: " << current_memory_usage_ / 1024 / 1024 << " MB, "
                       << ", flushing column group: " << max_heap_.top().first;
    auto max_group = max_heap_.top();
    max_heap_.pop();
    current_memory_usage_ -= max_group.second;

    ColumnGroupWriter* writer = group_writers_[max_group.first].get();
    RETURN_NOT_OK(writer->Flush());
  }

  // After flushing, add the new column groups if memory usage allows
  for (const ColumnGroup& group : column_groups) {
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(group.group_id(), group.GetMemoryUsage());
    ColumnGroupWriter* writer = group_writers_[group.group_id()].get();
    RETURN_NOT_OK(writer->Write(group.GetRecordBatch(0)));
  }
  return balanceMaxHeap();
}

Status PackedRecordBatchWriter::Close() {
  // flush all remaining column groups before closing
  return flushRemainingBuffer();
}

Status PackedRecordBatchWriter::flushRemainingBuffer() {
  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    ColumnGroupWriter* writer = group_writers_[max_group.first].get();

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    RETURN_NOT_OK(writer->Flush());
    current_memory_usage_ -= max_group.second;
  }
  for (auto& writer : group_writers_) {
    RETURN_NOT_OK(writer->WriteColumnOffsetsMeta(group_indices_));
    RETURN_NOT_OK(writer->Close());
  }
  return Status::OK();
}

Status PackedRecordBatchWriter::balanceMaxHeap() {
  std::map<GroupId, size_t> group_map;
  while (!max_heap_.empty()) {
    auto pair = max_heap_.top();
    max_heap_.pop();
    group_map[pair.first] += pair.second;
  }
  for (auto& pair : group_map) {
    max_heap_.emplace(pair.first, pair.second);
  }
  group_map.clear();
  return Status::OK();
}

std::shared_ptr<arrow::Schema> PackedRecordBatchWriter::getSubSchema(const std::shared_ptr<arrow::Schema>& schema,
                                                                     const std::vector<int>& column_indices) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int index : column_indices) {
    fields.push_back(schema->field(index));
  }
  return arrow::schema(fields);
}

}  // namespace milvus_storage
