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
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace milvus_storage {

struct RowGroupBlock {
  int64_t offset;  // Start offset of the row group block
  int64_t count;   // Number of row groups in this block

  bool operator==(const RowGroupBlock& other) const { return offset == other.offset && count == other.count; }
};

// Calculate the maximum number of row groups allowed per block
size_t max_row_groups_per_block(size_t memory_limit, size_t parallel_degree);

// Split row groups into blocks of appropriate size
std::vector<RowGroupBlock> split_row_groups(const std::vector<int64_t>& input_row_groups,
                                            size_t max_row_groups_per_block);

}  // namespace milvus_storage