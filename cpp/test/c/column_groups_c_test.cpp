// Copyright 2025 Zilliz
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
#include <vector>
#include <cstring>
#include "milvus-storage/c/column_groups_c.h"

namespace milvus_storage {

class CColumnGroupsTest : public testing::Test {};

TEST_F(CColumnGroupsTest, TestCColumnGroups) {
  CColumnGroups cgs = NewCColumnGroups();
  int group1[] = {2, 4, 5};
  int group2[] = {0, 1};
  int group3[] = {3, 6, 7, 8};

  int* test_groups[] = {group1, group2, group3};
  int group_sizes[] = {3, 2, 4};

  for (int i = 0; i < 3; i++) {
    AddCColumnGroup(cgs, test_groups[i], group_sizes[i]);
  }

  auto vv = static_cast<std::vector<std::vector<int>>*>(cgs);
  ASSERT_EQ(vv->size(), 3);

  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(vv->at(i).size(), group_sizes[i]);
    for (int j = 0; j < group_sizes[i]; j++) {
      EXPECT_EQ(vv->at(i)[j], test_groups[i][j]);
    }
  }

  FreeCColumnGroups(cgs);
}
}  // namespace milvus_storage