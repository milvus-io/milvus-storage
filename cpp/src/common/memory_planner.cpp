// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cstddef>
#include <algorithm>
#include "milvus-storage/common/memory_planner.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

size_t max_row_groups_per_block(size_t memory_limit, size_t parallel_degree) {
  size_t memory_per_task = memory_limit / parallel_degree;
  return memory_per_task / DEFAULT_MAX_ROW_GROUP_SIZE;
}

std::vector<RowGroupBlock> split_row_groups(const std::vector<int64_t>& input_row_groups,
                                            size_t max_row_groups_per_block) {
  std::vector<RowGroupBlock> blocks;
  if (input_row_groups.empty()) {
    return blocks;
  }
  // Sort the row groups
  std::vector<int64_t> sorted_row_groups = input_row_groups;
  std::sort(sorted_row_groups.begin(), sorted_row_groups.end());

  int64_t current_start = sorted_row_groups[0];
  int64_t current_count = 1;

  for (size_t i = 1; i < sorted_row_groups.size(); ++i) {
    if (current_count < max_row_groups_per_block) {
      if (sorted_row_groups[i] < current_start + max_row_groups_per_block) {
        current_count = sorted_row_groups[i] - current_start + 1;
        continue;
      }
    }

    blocks.push_back({current_start, current_count});
    current_start = sorted_row_groups[i];
    current_count = 1;
  }

  if (current_count > 0) {
    blocks.push_back({current_start, current_count});
  }
  return blocks;
}

}  // namespace milvus_storage