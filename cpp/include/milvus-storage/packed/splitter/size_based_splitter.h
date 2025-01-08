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

#include "splitter_plugin.h"
#include "common/result.h"

namespace milvus_storage {

class SizeBasedSplitter : public SplitterPlugin {
  public:
  /*
   * @brief SizeBasedSplitter is a splitter plugin that splits record batches into column groups based on the size of
   * each column.
   */
  explicit SizeBasedSplitter(size_t max_group_size);

  void Init() override;

  std::vector<ColumnGroup> SplitRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches);

  std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) override;

  size_t max_group_size_;
  static constexpr size_t SPLIT_THRESHOLD = 1024;  // 1K
};

}  // namespace milvus_storage
