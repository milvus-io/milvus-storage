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

#include "writer/column_group.h"
#include "common/arrow_util.h"
#include "arrow/table.h"
#include "common/status.h"

namespace milvus_storage {

ColumnGroup::ColumnGroup(GroupId group_id, const std::vector<int>& origin_column_indices)
    : group_id_(group_id), origin_column_indices_(origin_column_indices), memory_usage_(0) {}

ColumnGroup::ColumnGroup(GroupId group_id,
                         const std::vector<int>& origin_column_indices,
                         const std::shared_ptr<arrow::RecordBatch>& batch)
    : group_id_(group_id), origin_column_indices_(origin_column_indices) {
  AddRecordBatch(batch);
}

ColumnGroup::~ColumnGroup() {}

size_t ColumnGroup::size() const { return batches_.size(); }

GroupId ColumnGroup::group_id() const { return group_id_; }

Status ColumnGroup::AddRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!batch) {
    return Status::WriterError("ColumnGroup::AddRecordBatch: batch is null");
  }
  batches_.push_back(batch);
  memory_usage_ += GetRecordBatchMemorySize(batch);
  return Status::OK();
}

std::shared_ptr<arrow::Table> ColumnGroup::Table() const {
  auto result = arrow::Table::FromRecordBatches(batches_);
  if (result.ok()) {
    return result.ValueOrDie();
  } else {
    throw std::runtime_error(result.status().message());
  }
}

std::shared_ptr<arrow::RecordBatch> ColumnGroup::GetRecordBatch(size_t index) const { return batches_[index]; }

int ColumnGroup::GetRecordBatchNum() const { return batches_.size(); }

std::vector<int> ColumnGroup::GetOriginColumnIndices() const { return origin_column_indices_; }

size_t ColumnGroup::GetMemoryUsage() const { return memory_usage_; }

Status ColumnGroup::Clear() {
  batches_.clear();
  memory_usage_ = 0;
}

}  // namespace milvus_storage
