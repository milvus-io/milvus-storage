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
#include "writer/splitter/size_based_splitter.h"
#include <arrow/record_batch.h>
#include <memory>

namespace milvus_storage {
namespace writer {

SizeBasedSplitter::SizeBasedSplitter(size_t max_group_size) : max_group_size_(max_group_size) {}

void SizeBasedSplitter::Init() {}

std::vector<std::shared_ptr<arrow::RecordBatch>> SizeBasedSplitter::Split(
    const std::shared_ptr<arrow::RecordBatch>& record) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> column_groups;
  std::vector<int> small_group_indices;

  for (int i = 0; i < record->num_columns(); ++i) {
    std::shared_ptr<arrow::Array> column = record->column(i);
    size_t avg_size = SizeBasedSplitter::GetColumnMemorySize(column) / record->num_rows();
    if (small_group_indices.size() >= max_group_size_) {
      std::shared_ptr<arrow::RecordBatch> column_group = record->SelectColumns(small_group_indices).ValueOrDie();
      column_groups.push_back(column_group);
      small_group_indices.clear();
    }

    if (avg_size > SPLIT_THRESHOLD) {
      std::shared_ptr<arrow::RecordBatch> column_group = record->SelectColumns({i}).ValueOrDie();
      column_groups.push_back(column_group);
    } else {
      small_group_indices.push_back(i);
    }
  }

  if (!small_group_indices.empty()) {
    std::shared_ptr<arrow::RecordBatch> column_group = record->SelectColumns(small_group_indices).ValueOrDie();
    column_groups.push_back(column_group);
  }

  return column_groups;
}

size_t SizeBasedSplitter::GetColumnMemorySize(const std::shared_ptr<arrow::Array>& array) {
  size_t total_size = 0;
  for (const auto& buffer : array->data()->buffers) {
    if (buffer) {
      total_size += buffer->size();
    }
  }
  return total_size;
}

}  // namespace writer
}  // namespace milvus_storage
