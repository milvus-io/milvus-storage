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
#include "arrow/array/array_base.h"
#include "filter.h"
#include "value.h"

namespace milvus_storage {

enum ComparisonType {
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
};

class ConstantFilter : public Filter {
  public:
  ConstantFilter(ComparisonType comparison_type, std::string column_name, Value value);

  bool CheckStatistics(parquet::Statistics*) override;

  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  template <typename StatisticsType>
  bool CheckMinMax(StatisticsType* statistics);

  template <typename ArrayType, typename T = typename ArrayType::TypeClass>
  void ApplyFilter(const ArrayType* array, filter_mask& bitset);

  ComparisonType comparison_type_;
  Value value_;
};
}  // namespace milvus_storage
