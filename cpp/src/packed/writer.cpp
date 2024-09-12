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
#include <numeric>
#include "common/log.h"
#include "common/macro.h"
#include "common/status.h"
#include "packed/column_group.h"
#include "packed/column_group_writer.h"
#include "packed/splitter/indices_based_splitter.h"
#include "packed/splitter/size_based_splitter.h"
#include "packed/utils/config.h"
#include "filesystem/fs.h"
#include "common/arrow_util.h"

namespace milvus_storage {

PackedRecordBatchWriter::PackedRecordBatchWriter(size_t memory_limit,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 arrow::fs::FileSystem& fs,
                                                 const std::string& file_path,
                                                 parquet::WriterProperties& props)
    : memory_limit_(memory_limit),
      schema_(std::move(schema)),
      fs_(fs),
      file_path_(file_path),
      props_(props),
      splitter_({}),
      current_memory_usage_(0),
      size_split_done_(false) {}

Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  size_t next_batch_size = GetRecordBatchMemorySize(record);
  if (next_batch_size > memory_limit_) {
    return Status::InvalidArgument("Provided record batch size exceeds memory limit");
  }
  if (!size_split_done_) {
    if (current_memory_usage_ + next_batch_size < memory_limit_ / 2 || buffered_batches_.empty()) {
      buffered_batches_.push_back(record);
      current_memory_usage_ += next_batch_size;
      return Status::OK();
    } else {
      size_split_done_ = true;
      RETURN_NOT_OK(splitAndWriteFirstBuffer());
    }
  }
  return writeWithSplitIndex(record, next_batch_size);
}

Status PackedRecordBatchWriter::splitAndWriteFirstBuffer() {
  std::vector<ColumnGroup> groups =
      SizeBasedSplitter(buffered_batches_[0]->num_columns()).SplitRecordBatches(buffered_batches_);
  std::vector<std::vector<int>> group_indices;
  for (GroupId i = 0; i < groups.size(); ++i) {
    auto& group = groups[i];
    std::string group_path = file_path_ + "/" + std::to_string(i);
    auto writer =
        std::make_unique<ColumnGroupWriter>(i, group.Schema(), fs_, group_path, props_, group.GetOriginColumnIndices());
    RETURN_NOT_OK(writer->Init());
    for (auto& batch : group.GetRecordBatches()) {
      RETURN_NOT_OK(writer->Write(group.GetRecordBatch(0)));
    }

    max_heap_.emplace(i, group.GetMemoryUsage());
    group_indices.emplace_back(group.GetOriginColumnIndices());
    group_writers_.emplace_back(std::move(writer));
  }
  splitter_ = IndicesBasedSplitter(group_indices);

  // check memory usage limit
  size_t min_memory_limit = groups.size() * MIN_BUFFER_SIZE_PER_FILE;
  if (memory_limit_ < min_memory_limit) {
    return Status::InvalidArgument("Please provide at least " + std::to_string(min_memory_limit / 1024 / 1024) +
                                   " MB of memory for packed writer.");
  }
  memory_limit_ -= min_memory_limit;
  return balanceMaxHeap();
}

Status PackedRecordBatchWriter::writeWithSplitIndex(const std::shared_ptr<arrow::RecordBatch>& record,
                                                    size_t next_batch_size) {
  std::vector<ColumnGroup> column_groups = splitter_.Split(record);

  // Flush column groups until there's enough room for the new column groups
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + next_batch_size >= memory_limit_ && !max_heap_.empty()) {
    LOG_STORAGE_DEBUG_ << "Current memory usage: " << current_memory_usage_
                       << ", flushing column group: " << max_heap_.top().first;
    auto max_group = max_heap_.top();
    current_memory_usage_ -= max_group.second;

    ColumnGroupWriter* writer = group_writers_[max_group.first].get();
    max_heap_.pop();
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
  if (!size_split_done_ && !buffered_batches_.empty()) {
    std::string group_path = file_path_ + "/" + std::to_string(0);
    std::vector<int> indices(buffered_batches_[0]->num_columns());
    std::iota(std::begin(indices), std::end(indices), 0);
    auto writer =
        std::make_unique<ColumnGroupWriter>(0, buffered_batches_[0]->schema(), fs_, group_path, props_, indices);
    RETURN_NOT_OK(writer->Init());
    for (auto& batch : buffered_batches_) {
      RETURN_NOT_OK(writer->Write(batch));
    }
    RETURN_NOT_OK(writer->Flush());
    RETURN_NOT_OK(writer->Close());
    return Status::OK();
  }
  // flush all remaining column groups before closing'
  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    ColumnGroupWriter* writer = group_writers_[max_group.first].get();

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    RETURN_NOT_OK(writer->Flush());
    current_memory_usage_ -= max_group.second;
  }
  for (auto& writer : group_writers_) {
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

}  // namespace milvus_storage
