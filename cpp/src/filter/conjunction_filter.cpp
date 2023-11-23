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

#include "filter/conjunction_filter.h"
#include "common/macro.h"

namespace milvus_storage {

bool ConjunctionOrFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (!filter->CheckStatistics(stats)) {
      return false;
    }
  }
  return true;
}

Status ConjunctionOrFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  filter_mask or_bitset;
  for (auto& filter : filters_) {
    filter_mask bitset_cloned = bitset;
    RETURN_NOT_OK(filter->Apply(col_data, bitset_cloned));
    or_bitset &= bitset_cloned;
  }
  bitset |= or_bitset;
  return Status::OK();
}

bool ConjunctionAndFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (filter->CheckStatistics(stats)) {
      return true;
    }
  }
  return false;
}

Status ConjunctionAndFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  for (auto& filter : filters_) {
    RETURN_NOT_OK(filter->Apply(col_data, bitset));
  }
  return Status::OK();
}
}  // namespace milvus_storage
