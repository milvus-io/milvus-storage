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
#include "gtest/gtest.h"
#include "milvus-storage/common/serde.h"
#include <cstdint>

namespace milvus_storage {

class SerdeTest : public testing::Test {};

TEST_F(SerdeTest, TestColumnOffsetSerde) {
  std::vector<std::vector<int64_t>> column_offsets = {{0, 1, 2}, {3, 4}, {5, 6, 7, 8}};

  std::string serialized = PackedMetaSerde::SerializeColumnOffsets(column_offsets);
  EXPECT_EQ(serialized, "0,1,2;3,4;5,6,7,8");

  std::vector<std::vector<int64_t>> deserialized = PackedMetaSerde::DeserializeColumnOffsets(serialized);
  EXPECT_EQ(deserialized, column_offsets);

  // Test case: Empty input
  std::vector<std::vector<int64_t>> empty_offsets = {};
  serialized = PackedMetaSerde::SerializeColumnOffsets(empty_offsets);
  EXPECT_EQ(serialized, "");

  deserialized = PackedMetaSerde::DeserializeColumnOffsets(serialized);
  EXPECT_TRUE(deserialized.empty());

  // Test case: Single group
  std::vector<std::vector<int64_t>> single_group = {{1, 2, 3}};
  serialized = PackedMetaSerde::SerializeColumnOffsets(single_group);
  EXPECT_EQ(serialized, "1,2,3");

  deserialized = PackedMetaSerde::DeserializeColumnOffsets(serialized);
  EXPECT_EQ(deserialized, single_group);

  // Test case: Single column in each group
  std::vector<std::vector<int64_t>> single_column_groups = {{0}, {1}, {2}};
  serialized = PackedMetaSerde::SerializeColumnOffsets(single_column_groups);
  EXPECT_EQ(serialized, "0;1;2");

  deserialized = PackedMetaSerde::DeserializeColumnOffsets(serialized);
  EXPECT_EQ(deserialized, single_column_groups);
}
}  // namespace milvus_storage