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
#include "common/log.h"
#include "common/macro.h"
#include "packed/column_group.h"
#include <stdexcept>
#include <arrow/table.h>
#include <arrow/array/concatenate.h>
#include <iostream>

namespace milvus_storage {

SizeBasedSplitter::SizeBasedSplitter(size_t max_group_size, int pk_index, int ts_index)
    : max_group_size_(max_group_size), pk_index_(pk_index), ts_index_(ts_index) {}

void SizeBasedSplitter::Init() {}

std::vector<ColumnGroup> SizeBasedSplitter::SplitRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
  if (batches.empty()) {
    return {};
  }
  // calculate column sizes and rows
  std::vector<size_t> column_sizes(batches[0]->num_columns(), 0);
  std::vector<int64_t> column_rows(batches[0]->num_columns(), 0);
  for (const auto& record : batches) {
    for (int i = 0; i < record->num_columns(); ++i) {
      std::shared_ptr<arrow::Array> column = record->column(i);
      column_sizes[i] += GetArrowArrayMemorySize(column);
      column_rows[i] += record->num_rows();
    }
  }

  // split column indices into small and large groups
  std::vector<int> small_group = initializeSmallGroupIndices();
  std::vector<int> large_group;
  std::vector<std::vector<int>> small_groups;
  std::vector<std::vector<int>> large_groups;
  for (int i = 0; i < column_sizes.size(); ++i) {
    if (i == pk_index_ || i == ts_index_) {
      continue;
    }
    size_t avg_size = column_sizes[i] / column_rows[i];
    if (small_group.size() >= max_group_size_) {
      small_groups.push_back(small_group);
      small_group.clear();
    }
    if (large_group.size() >= max_group_size_) {
      large_groups.push_back(large_group);
      large_group.clear();
    }
    if (avg_size >= SPLIT_THRESHOLD) {
      large_group.push_back({i});
    } else {
      small_group.push_back(i);
    }
  }
  if (!small_group.empty()) {
    small_groups.push_back(small_group);
  }
  if (!large_group.empty()) {
    large_groups.push_back(large_group);
  }

  // merge small and large groups
  std::vector<std::vector<int>> group_indices = small_groups;
  group_indices.insert(group_indices.end(), large_groups.begin(), large_groups.end());

  // create column groups
  std::vector<ColumnGroup> column_groups;
  for (auto& record : batches) {
    for (GroupId group_id = 0; group_id < group_indices.size(); ++group_id) {
      auto batch = record->SelectColumns(group_indices[group_id]).ValueOrDie();
      if (column_groups.size() < group_indices.size()) {
        column_groups.emplace_back(group_id, group_indices[group_id], batch);
      } else {
        column_groups[group_id].AddRecordBatch(batch);
      }
    }
  }
  return column_groups;
}

std::vector<ColumnGroup> SizeBasedSplitter::Split(const std::shared_ptr<arrow::RecordBatch>& record) {
  return SplitRecordBatches({record});
}

// Forced to add pk and ts to the small group
std::vector<int> SizeBasedSplitter::initializeSmallGroupIndices() {
  std::vector<int> small_group_indices;
  small_group_indices.push_back(pk_index_);
  small_group_indices.push_back(ts_index_);
  return small_group_indices;
}

}  // namespace milvus_storage
