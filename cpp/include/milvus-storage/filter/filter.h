// Copyright 2023 Zilliz
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

#include <bitset>
#include <memory>
#include "arrow/record_batch.h"
#include "milvus-storage/common/constants.h"
#include "parquet/statistics.h"
#include "milvus-storage/common/macro.h"

namespace milvus_storage {

using filter_mask = std::bitset<kReadBatchSize>;
class Filter {
  public:
  using FilterSet = std::vector<Filter*>;
  explicit Filter(std::string column_name) : column_name_(std::move(column_name)) {}

  virtual bool CheckStatistics(::parquet::Statistics*) = 0;

  std::string get_column_name() { return column_name_; };

  virtual arrow::Status Apply(arrow::Array* col_data, filter_mask& bitset) = 0;

  static arrow::Status ApplyFilter(const std::shared_ptr<arrow::RecordBatch>& record_batch,
                                   const FilterSet& filters,
                                   filter_mask& bitset) {
    for (auto& filter : filters) {
      auto col_data = record_batch->GetColumnByName(filter->get_column_name());
      RETURN_NOT_OK(filter->Apply(col_data.get(), bitset));
    }
    return arrow::Status::OK();
  }

  virtual ~Filter(){};

  protected:
  std::string column_name_;
};
}  // namespace milvus_storage
