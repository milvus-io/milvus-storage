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

#pragma once

#include <arrow/record_batch.h>
#include <queue>
#include "common/status.h"

namespace milvus_storage {

typedef size_t GroupId;

class ColumnGroup {
  public:
  ColumnGroup(GroupId group_id, const std::vector<int>& origin_column_indices);

  ColumnGroup(GroupId group_id,
              const std::vector<int>& origin_column_indices,
              const std::shared_ptr<arrow::RecordBatch>& batch);

  ~ColumnGroup();

  size_t size() const;

  GroupId group_id() const;

  Status AddRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch);

  Status Merge(const ColumnGroup& other);

  std::shared_ptr<arrow::Table> Table() const;

  std::shared_ptr<arrow::Schema> Schema() const;

  std::shared_ptr<arrow::RecordBatch> GetRecordBatch(size_t index) const;

  std::vector<std::shared_ptr<arrow::RecordBatch>> GetRecordBatches() const;

  int GetRecordBatchNum() const;

  std::vector<int> GetOriginColumnIndices() const;

  size_t GetMemoryUsage() const;

  std::vector<size_t> GetRecordMemoryUsages() const;

  Status Clear();

  int GetTotalRows() const { return total_rows_; }

  private:
  GroupId group_id_;
  std::vector<size_t> batch_memory_usage_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  size_t memory_usage_;
  std::vector<int> origin_column_indices_;
  int total_rows_;
};

struct ColumnGroupState {
  int64_t row_offset;
  int64_t row_group_offset;
  int64_t memory_size;
  int read_times;

  ColumnGroupState(int64_t row_offset, int64_t row_group_offset, int64_t memory_size)
      : row_offset(row_offset), row_group_offset(row_group_offset), memory_size(memory_size), read_times(0) {}

  void addRowOffset(int64_t row_offset) { this->row_offset += row_offset; }

  void setRowGroupOffset(int64_t row_group_offset) { this->row_group_offset = row_group_offset; }

  void addMemorySize(int64_t memory_size) { this->memory_size += memory_size; }

  void resetMemorySize() { this->memory_size = 0; }
};

}  // namespace milvus_storage