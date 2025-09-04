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

#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/packed/column_group.h"

namespace milvus_storage {

IndicesBasedSplitter::IndicesBasedSplitter(const std::vector<std::vector<int>>& column_indices)
    : column_indices_(column_indices) {}

std::vector<ColumnGroup> IndicesBasedSplitter::Split(const std::shared_ptr<arrow::RecordBatch>& record) {
  std::vector<ColumnGroup> column_groups;

  for (GroupId group_id = 0; group_id < column_indices_.size(); group_id++) {
    auto batch = record->SelectColumns(column_indices_[group_id]).ValueOrDie();
    column_groups.emplace_back(group_id, column_indices_[group_id], batch);
  }

  return column_groups;
}

}  // namespace milvus_storage
