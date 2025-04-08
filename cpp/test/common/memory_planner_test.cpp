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

#include <gtest/gtest.h>
#include "milvus-storage/common/memory_planner.h"

namespace milvus_storage {

TEST(SplitRowGroupsTest, EmptyInput) {
  std::vector<int64_t> empty_input;
  auto result = split_row_groups(empty_input, 2);
  EXPECT_TRUE(result.empty());
}

TEST(SplitRowGroupsTest, SingleRowGroup) {
  std::vector<int64_t> input = {1};
  auto result = split_row_groups(input, 2);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 1}));
}

TEST(SplitRowGroupsTest, ContinuousWithinLimit) {
  std::vector<int64_t> input = {1, 2, 3, 4};
  auto result = split_row_groups(input, 4);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 4}));
}

TEST(SplitRowGroupsTest, ContinuousExceedLimit) {
  std::vector<int64_t> input = {1, 2, 3, 4, 5};
  auto result = split_row_groups(input, 3);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 3}));
  EXPECT_EQ(result[1], (RowGroupBlock{4, 2}));
}

TEST(SplitRowGroupsTest, NonContinuous_SmallGap) {
  std::vector<int64_t> input = {1, 3, 5, 7};
  auto result = split_row_groups(input, 3);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 3}));
  EXPECT_EQ(result[1], (RowGroupBlock{5, 3}));
}

TEST(SplitRowGroupsTest, NonContinuous_LargeGap) {
  std::vector<int64_t> input = {1, 10, 20, 30};
  auto result = split_row_groups(input, 3);
  EXPECT_EQ(result.size(), 4);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 1}));
  EXPECT_EQ(result[1], (RowGroupBlock{10, 1}));
  EXPECT_EQ(result[2], (RowGroupBlock{20, 1}));
  EXPECT_EQ(result[3], (RowGroupBlock{30, 1}));
}

TEST(SplitRowGroupsTest, Mixed_ContinuousAndNonContinuous) {
  std::vector<int64_t> input = {1, 2, 4, 5, 6, 8};
  auto result = split_row_groups(input, 3);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 2}));
  EXPECT_EQ(result[1], (RowGroupBlock{4, 3}));
  EXPECT_EQ(result[2], (RowGroupBlock{8, 1}));
}

TEST(SplitRowGroupsTest, Mixed_GapsAndContinuous) {
  std::vector<int64_t> input = {1, 3, 4, 7, 8, 9};
  auto result = split_row_groups(input, 4);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 4}));
  EXPECT_EQ(result[1], (RowGroupBlock{7, 3}));
}

TEST(SplitRowGroupsTest, EdgeCase_SingleBlock) {
  std::vector<int64_t> input = {1, 2, 3, 4, 5};
  auto result = split_row_groups(input, 5);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (RowGroupBlock{1, 5}));
}

TEST(SplitRowGroupsTest, EdgeCase_EachRowGroupSeparate) {
  std::vector<int64_t> input = {1, 3, 5, 7, 9};
  auto result = split_row_groups(input, 1);
  EXPECT_EQ(result.size(), 5);
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], (RowGroupBlock{input[i], 1}));
  }
}

TEST(MaxRowGroupsPerBlockTest, Basic) { EXPECT_EQ(max_row_groups_per_block(128 * 1024 * 1024, 8), 16); }

}  // namespace milvus_storage