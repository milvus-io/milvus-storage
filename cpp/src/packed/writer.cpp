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

#include "packed/writer.h"
#include <cstddef>
#include "common/log.h"
#include "common/status.h"
#include "packed/column_group.h"
#include "packed/column_group_writer.h"
#include "packed/splitter/indices_based_splitter.h"
#include "packed/splitter/size_based_splitter.h"
#include "common/fs_util.h"

namespace milvus_storage {

PackedRecordBatchWriter::PackedRecordBatchWriter(size_t memory_limit,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 arrow::fs::FileSystem& fs,
                                                 std::string& file_path,
                                                 parquet::WriterProperties& props)
    : memory_limit_(memory_limit),
      schema_(std::move(schema)),
      fs_(fs),
      file_path_(file_path),
      props_(props),
      splitter_({}),
      current_memory_usage_(0) {}

Status PackedRecordBatchWriter::Init(const std::shared_ptr<arrow::RecordBatch>& record) {
  // split first batch into column groups
  std::vector<ColumnGroup> groups = SizeBasedSplitter(record->num_columns()).Split(record);

  // init column group writer and
  // put column groups into max heap
  std::vector<std::vector<int>> group_indices;
  GroupId group_id = 0;
  for (const ColumnGroup& group : groups) {
    std::string group_path = file_path_ + "/" + std::to_string(group_id);
    auto writer = std::make_unique<ColumnGroupWriter>(group_id, group.Schema(), fs_, group_path, props_,
                                                      group.GetOriginColumnIndices());
    auto status = writer->Init();
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to init column group writer: " << status.ToString();
      return status;
    }
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(group_id, group.GetMemoryUsage());
    status = writer->Write(group.GetRecordBatch(0));
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to write column group: " << group_id << ", " << status.ToString();
      return status;
    }
    group_indices.emplace_back(group.GetOriginColumnIndices());
    group_writers_.emplace_back(std::move(writer));
    group_id++;
  }
  splitter_ = IndicesBasedSplitter(group_indices);

  // check memory usage limit
  size_t min_memory_limit = group_id * (DEFAULT_MAX_ROW_GROUP_SIZE + ARROW_PART_UPLOAD_SIZE);
  if (memory_limit_ < min_memory_limit) {
    return Status::InvalidArgument("Please provide at least " + std::to_string(min_memory_limit / 1024 / 1024) +
                                   " MB of memory for packed writer.");
  }
  memory_limit_ -= min_memory_limit;
  return balanceMaxHeap();
}

Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  std::vector<ColumnGroup> column_groups = splitter_.Split(record);

  // Calculate the total memory usage of the new column groups
  size_t new_memory_usage = 0;
  for (const ColumnGroup& group : column_groups) {
    new_memory_usage += group.GetMemoryUsage();
  }

  // Flush column groups until there's enough room for the new column groups
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + new_memory_usage >= memory_limit_ && !max_heap_.empty()) {
    LOG_STORAGE_DEBUG_ << "Current memory usage: " << current_memory_usage_
                       << ", flushing column group: " << max_heap_.top().first;
    auto max_group = max_heap_.top();
    current_memory_usage_ -= max_group.second;

    ColumnGroupWriter* writer = group_writers_[max_group.first].get();
    max_heap_.pop();
    auto status = writer->Flush();
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to flush column group: " << max_group.first << ", " << status.ToString();
      return status;
    }
  }

  // After flushing, add the new column groups if memory usage allows
  for (const ColumnGroup& group : column_groups) {
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(group.group_id(), group.GetMemoryUsage());
    ColumnGroupWriter* writer = group_writers_[group.group_id()].get();
    auto status = writer->Write(group.GetRecordBatch(0));
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to write column group: " << group.group_id() << ", " << status.ToString();
      return status;
    }
  }
  return balanceMaxHeap();
}

Status PackedRecordBatchWriter::Close() {
  // flush all remaining column groups before closing'
  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    ColumnGroupWriter* writer = group_writers_[max_group.first].get();

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    auto status = writer->Flush();
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to flush column group: " << max_group.first << ", " << status.ToString();
      return status;
    }
    current_memory_usage_ -= max_group.second;
  }
  for (auto& writer : group_writers_) {
    auto status = writer->Close();
    if (!status.ok()) {
      LOG_STORAGE_ERROR_ << "Failed to close column group writer: " << status.ToString();
      return status;
    }
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

}  // namespace milvus_storage
