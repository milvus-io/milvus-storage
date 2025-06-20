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
#include <memory>
#include <arrow/record_batch.h>
#include <milvus-storage/packed/column_group.h>

namespace milvus_storage {

class SplitterPlugin {
  public:
  virtual ~SplitterPlugin() = default;

  virtual void Init() = 0;

  // Split the input record batch into multiple groups of columns
  virtual std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) = 0;
};

}  // namespace milvus_storage
