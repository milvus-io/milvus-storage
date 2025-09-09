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

#include <queue>
#include <vector>

namespace milvus_storage {

/**
 * @brief Template comparator for maintaining row offset ordering
 *
 * @tparam OffsetType The type for offset values (int or int64_t)
 */
template <typename OffsetType>
struct RowOffsetComparator {
  bool operator()(const std::pair<int, OffsetType>& a, const std::pair<int, OffsetType>& b) const {
    return a.second > b.second;
  }
};

/**
 * @brief Type alias for row offset min heap with int offsets
 */
using RowOffsetMinHeap =
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, RowOffsetComparator<int>>;

}  // namespace milvus_storage