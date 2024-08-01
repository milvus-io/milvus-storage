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

namespace milvus_storage {
namespace writer {

class SizeBasedSplitter : public SplitterPlugin {
  public:
  explicit SizeBasedSplitter(size_t max_group_size);

  void Init() override;

  std::vector<std::shared_ptr<arrow::RecordBatch>> Split(const std::shared_ptr<arrow::RecordBatch>& record) override;

  private:
  size_t max_group_size_;
  static constexpr size_t SPLIT_THRESHOLD = 1024;  // 1K

  size_t GetColumnMemorySize(const std::shared_ptr<arrow::Array>& array);
};

}  // namespace writer
}  // namespace milvus_storage
