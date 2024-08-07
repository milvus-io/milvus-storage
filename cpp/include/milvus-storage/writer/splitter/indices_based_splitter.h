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

#include <vector>
#include <map>
#include "splitter_plugin.h"

namespace milvus_storage {

class IndicesBasedSplitter : public SplitterPlugin {
  public:
  explicit IndicesBasedSplitter(const std::vector<std::vector<int>>& column_indices);

  void Init() override;

  std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) override;

  private:
  std::vector<std::vector<int>> column_indices_;
};

}  // namespace milvus_storage
