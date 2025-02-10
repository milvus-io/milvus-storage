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
#include "milvus-storage/c/paths_c.h"

namespace milvus_storage {

class CPathsTest : public testing::Test {};

TEST_F(CPathsTest, TestCPaths) {
  CPaths paths = NewCPaths();
  const char* strs[] = {"colID/parID/segID/0/logID0", "colID/parID/segID/1/logID1", "colID/parID/segID/2/logID2"};
  for (size_t i = 0; i < 3; i++) {
    AddPathToCPaths(paths, strs[i], std::strlen(strs[i]));
  }

  auto v = static_cast<std::vector<std::string>*>(paths);
  for (size_t i = 0; i < v->size(); i++) {
    EXPECT_EQ(strs[i], v->at(i));
  }

  FreeCPaths(paths);
}
}  // namespace milvus_storage