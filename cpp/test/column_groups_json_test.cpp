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

#include <gtest/gtest.h>
#include <sstream>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/config.h"

#include "test_util.h"

using namespace milvus_storage::api;

class ColumnGroupsJsonTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Create test column groups
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = {"id", "name", "age"};
    cg1->files = {{.path = "/data/cg1_part1.parquet"}, {.path = "/data/cg1_part2.parquet"}};
    cg1->format = LOON_FORMAT_PARQUET;

    auto cg2 = std::make_shared<ColumnGroup>();
    cg2->columns = {"embedding", "metadata"};
    cg2->files = {{.path = "/data/cg2_vectors.vortex"}};
    cg2->format = LOON_FORMAT_VORTEX;

    std::vector<std::shared_ptr<ColumnGroup>> column_groups = {cg1, cg2};
    test_cgs_ = std::make_shared<ColumnGroups>(std::move(column_groups));
  }

  std::shared_ptr<ColumnGroups> test_cgs_;
};

TEST_F(ColumnGroupsJsonTest, SerializeDeserialize) {
  // Serialize to JSON
  ASSERT_AND_ASSIGN(auto json_str, test_cgs_->serialize());

  EXPECT_FALSE(json_str.empty());
  EXPECT_NE(json_str.find("\"column_groups\""), std::string::npos);

  // Deserialize from JSON
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(json_str));
  const auto& groups = deserialized_cgs->get_all();
  EXPECT_EQ(groups.size(), 2);

  // Check first column group
  EXPECT_EQ(groups[0]->columns.size(), 3);
  EXPECT_EQ(groups[0]->columns[0], "id");
  EXPECT_EQ(groups[0]->columns[1], "name");
  EXPECT_EQ(groups[0]->columns[2], "age");
  EXPECT_EQ(groups[0]->files.size(), 2);
  EXPECT_EQ(groups[0]->format, LOON_FORMAT_PARQUET);

  // Check second column group
  EXPECT_EQ(groups[1]->columns.size(), 2);
  EXPECT_EQ(groups[1]->columns[0], "embedding");
  EXPECT_EQ(groups[1]->columns[1], "metadata");
  EXPECT_EQ(groups[1]->files.size(), 1);
  EXPECT_EQ(groups[1]->format, LOON_FORMAT_VORTEX);
}

TEST_F(ColumnGroupsJsonTest, EmptyColumnGroups) {
  // Test empty column groups
  std::vector<std::shared_ptr<ColumnGroup>> column_groups = {};
  auto empty_cgs = std::make_shared<ColumnGroups>(column_groups);

  ASSERT_AND_ASSIGN(auto json_str, empty_cgs->serialize());

  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(json_str));

  EXPECT_EQ(deserialized_cgs->get_all().size(), 0);
}

TEST_F(ColumnGroupsJsonTest, ColumnLookup) {
  // Serialize and deserialize
  std::ostringstream output;
  ASSERT_AND_ASSIGN(auto json_str, test_cgs_->serialize());
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(json_str));

  // Test column lookup functionality
  auto name_cg = deserialized_cgs->get_column_group("name");
  ASSERT_NE(name_cg, nullptr);
  EXPECT_EQ(name_cg->format, LOON_FORMAT_PARQUET);

  auto embedding_cg = deserialized_cgs->get_column_group("embedding");
  ASSERT_NE(embedding_cg, nullptr);
  EXPECT_EQ(embedding_cg->format, LOON_FORMAT_VORTEX);

  auto missing_cg = deserialized_cgs->get_column_group("nonexistent");
  EXPECT_EQ(missing_cg, nullptr);
}

TEST_F(ColumnGroupsJsonTest, InvalidJson) {
  // Test deserialization with invalid JSON
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_NOT_OK(deserialized_cgs->deserialize("invalid json"));
  EXPECT_EQ(deserialized_cgs->get_all().size(), 0);
}
