// Copyright 2026 Zilliz
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

// Coverage for the V2 (non-manifest) LoonColumnGroups builder.
//
// Goal: if the LoonColumnGroup / LoonColumnGroupFile layout evolves, this
// test locks the build→destroy contract so that any missed allocation step
// shows up as a failure or an ASAN leak, rather than as silent UB under the
// spark-connector.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "milvus-storage/common/config.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/v2_column_groups_builder.h"

namespace milvus_storage::test {

TEST(V2ColumnGroupsBuilder, BuildsExpectedLayout) {
  std::vector<std::vector<std::string>> cols = {
      {"100", "101"},
      {"200"},
  };
  std::vector<std::vector<std::string>> files = {
      {"g0/f0.parquet", "g0/f1.parquet"},
      {"g1/f0.parquet"},
  };
  std::vector<std::vector<int64_t>> rcs = {
      {10, 25},
      {7},
  };

  LoonColumnGroups* cgs = BuildLoonColumnGroups(cols, files, rcs);
  ASSERT_NE(cgs, nullptr);
  ASSERT_EQ(cgs->num_of_column_groups, 2u);
  ASSERT_NE(cgs->column_group_array, nullptr);

  const LoonColumnGroup& g0 = cgs->column_group_array[0];
  ASSERT_EQ(g0.num_of_columns, 2u);
  EXPECT_STREQ(g0.columns[0], "100");
  EXPECT_STREQ(g0.columns[1], "101");
  EXPECT_STREQ(g0.format, LOON_FORMAT_PARQUET);
  ASSERT_EQ(g0.num_of_files, 2u);
  EXPECT_STREQ(g0.files[0].path, "g0/f0.parquet");
  EXPECT_EQ(g0.files[0].start_index, 0);
  EXPECT_EQ(g0.files[0].end_index, 10);
  EXPECT_STREQ(g0.files[1].path, "g0/f1.parquet");
  EXPECT_EQ(g0.files[1].start_index, 10);
  EXPECT_EQ(g0.files[1].end_index, 35);
  EXPECT_EQ(g0.files[0].num_properties, 0u);
  EXPECT_EQ(g0.files[0].property_keys, nullptr);
  EXPECT_EQ(g0.files[0].property_values, nullptr);

  const LoonColumnGroup& g1 = cgs->column_group_array[1];
  ASSERT_EQ(g1.num_of_columns, 1u);
  EXPECT_STREQ(g1.columns[0], "200");
  EXPECT_STREQ(g1.format, LOON_FORMAT_PARQUET);
  ASSERT_EQ(g1.num_of_files, 1u);
  EXPECT_STREQ(g1.files[0].path, "g1/f0.parquet");
  EXPECT_EQ(g1.files[0].start_index, 0);
  EXPECT_EQ(g1.files[0].end_index, 7);

  // Ownership handed back via the standard destroy. If a new field were
  // added to the structs and left unallocated by the builder,
  // loon_column_groups_destroy would double-free or leak (ASAN catches both).
  loon_column_groups_destroy(cgs);
}

TEST(V2ColumnGroupsBuilder, DestroyTolerantOfNull) { loon_column_groups_destroy(nullptr); }

TEST(V2ColumnGroupsBuilder, RejectsMismatchedOuterLengths) {
  std::vector<std::vector<std::string>> cols = {{"a"}};
  std::vector<std::vector<std::string>> files = {{"f.parquet"}, {"f2.parquet"}};
  std::vector<std::vector<int64_t>> rcs = {{1}};
  EXPECT_THROW(BuildLoonColumnGroups(cols, files, rcs), std::invalid_argument);
}

TEST(V2ColumnGroupsBuilder, RejectsZeroGroups) {
  EXPECT_THROW(BuildLoonColumnGroups({}, {}, {}), std::invalid_argument);
}

TEST(V2ColumnGroupsBuilder, RejectsEmptyGroup) {
  std::vector<std::vector<std::string>> cols = {{}};
  std::vector<std::vector<std::string>> files = {{"f.parquet"}};
  std::vector<std::vector<int64_t>> rcs = {{1}};
  EXPECT_THROW(BuildLoonColumnGroups(cols, files, rcs), std::invalid_argument);

  std::vector<std::vector<std::string>> cols2 = {{"a"}};
  std::vector<std::vector<std::string>> files2 = {{}};
  std::vector<std::vector<int64_t>> rcs2 = {{}};
  EXPECT_THROW(BuildLoonColumnGroups(cols2, files2, rcs2), std::invalid_argument);
}

TEST(V2ColumnGroupsBuilder, RejectsRowCountLengthMismatch) {
  std::vector<std::vector<std::string>> cols = {{"a"}};
  std::vector<std::vector<std::string>> files = {{"f0.parquet", "f1.parquet"}};
  std::vector<std::vector<int64_t>> rcs = {{5}};
  EXPECT_THROW(BuildLoonColumnGroups(cols, files, rcs), std::invalid_argument);
}

}  // namespace milvus_storage::test
