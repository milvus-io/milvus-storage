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

#include "packed/splitter/size_based_splitter.h"
#include "common/arrow_util.h"
#include "packed/column_group.h"
#include <stdexcept>

namespace milvus_storage {

SizeBasedSplitter::SizeBasedSplitter(size_t max_group_size) : max_group_size_(max_group_size) {}

void SizeBasedSplitter::Init() {}

std::vector<ColumnGroup> SizeBasedSplitter::Split(const std::shared_ptr<arrow::RecordBatch>& record) {
  if (!record) {
    throw std::invalid_argument("RecordBatch is null");
  }
  std::vector<ColumnGroup> column_groups;
  std::vector<int> small_group_indices;
  GroupId group_id = 0;
  for (int i = 0; i < record->num_columns(); ++i) {
    std::shared_ptr<arrow::Array> column = record->column(i);
    if (!column) {
      throw std::runtime_error("Column is null");
    }
    size_t avg_size = GetArrowArrayMemorySize(column) / record->num_rows();

    if (small_group_indices.size() >= max_group_size_) {
      AddColumnGroup(record, column_groups, small_group_indices, group_id);
    }

    if (avg_size >= SPLIT_THRESHOLD) {
      std::vector<int> indices = {i};
      AddColumnGroup(record, column_groups, indices, group_id);
    } else {
      small_group_indices.push_back(i);
    }
  }

  AddColumnGroup(record, column_groups, small_group_indices, group_id);
  return column_groups;
}

void SizeBasedSplitter::AddColumnGroup(const std::shared_ptr<arrow::RecordBatch>& record,
                                       std::vector<ColumnGroup>& column_groups,
                                       std::vector<int>& indices,
                                       GroupId& group_id) {
  if (indices.empty() || !record) {
    return;
  }
  auto batch = record->SelectColumns(indices).ValueOrDie();
  column_groups.push_back(ColumnGroup(group_id++, indices, batch));
  indices.clear();
}

}  // namespace milvus_storage
