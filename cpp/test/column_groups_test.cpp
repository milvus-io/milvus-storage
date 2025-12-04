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
#include <cstdint>
#include <sstream>
#include <random>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/config.h"

#include "test_env.h"

using namespace milvus_storage::api;

class ColumnGroupsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Create test column groups
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = {"id", "name", "age"};
    // Initialize files using brace initialization (aggregates)
    // ColumnGroupFile has path, start_index, end_index.
    // Optional members default to nullopt.
    cg1->files = {{"/data/cg1_part1.parquet"}, {"/data/cg1_part2.parquet"}};
    cg1->format = LOON_FORMAT_PARQUET;

    auto cg2 = std::make_shared<ColumnGroup>();
    cg2->columns = {"embedding", "metadata"};
    cg2->files = {{"/data/cg2_vectors.vortex"}};
    cg2->format = LOON_FORMAT_VORTEX;

    std::vector<std::shared_ptr<ColumnGroup>> column_groups = {cg1, cg2};
    test_cgs_ = std::make_shared<ColumnGroups>(std::move(column_groups));
  }

  std::shared_ptr<ColumnGroups> test_cgs_;
};

TEST_F(ColumnGroupsTest, SerializeDeserialize) {
  // Serialize to Avro
  ASSERT_AND_ASSIGN(auto avro_str, test_cgs_->serialize());

  EXPECT_FALSE(avro_str.empty());

  // Deserialize from Avro
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(avro_str));
  const auto& groups = deserialized_cgs->get_all();
  const auto& expected_groups = test_cgs_->get_all();

  EXPECT_EQ(groups.size(), expected_groups.size());

  for (size_t i = 0; i < groups.size(); ++i) {
    EXPECT_EQ(groups[i]->columns, expected_groups[i]->columns);
    EXPECT_EQ(groups[i]->format, expected_groups[i]->format);

    ASSERT_EQ(groups[i]->files.size(), expected_groups[i]->files.size());
    for (size_t j = 0; j < groups[i]->files.size(); ++j) {
      EXPECT_EQ(groups[i]->files[j].path, expected_groups[i]->files[j].path);
      EXPECT_EQ(groups[i]->files[j].start_index, expected_groups[i]->files[j].start_index);
      EXPECT_EQ(groups[i]->files[j].end_index, expected_groups[i]->files[j].end_index);
    }
  }
}

TEST_F(ColumnGroupsTest, EmptyColumnGroups) {
  // Test empty column groups
  std::vector<std::shared_ptr<ColumnGroup>> column_groups = {};
  auto empty_cgs = std::make_shared<ColumnGroups>(column_groups);

  ASSERT_AND_ASSIGN(auto avro_str, empty_cgs->serialize());

  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(avro_str));

  EXPECT_EQ(deserialized_cgs->get_all().size(), 0);
}

TEST_F(ColumnGroupsTest, ColumnLookup) {
  // Serialize and deserialize
  ASSERT_AND_ASSIGN(auto avro_str, test_cgs_->serialize());
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(avro_str));

  // Test column lookup functionality
  const auto& expected_groups = test_cgs_->get_all();
  if (!expected_groups.empty() && !expected_groups[0]->columns.empty()) {
    std::string test_col = expected_groups[0]->columns[0];
    auto cg = deserialized_cgs->get_column_group(test_col);
    ASSERT_NE(cg, nullptr);
    EXPECT_EQ(cg->format, expected_groups[0]->format);
  }

  auto missing_cg = deserialized_cgs->get_column_group("nonexistent_column_name_xyz");
  EXPECT_EQ(missing_cg, nullptr);
}

TEST_F(ColumnGroupsTest, InvalidAvro) {
  // Test deserialization with empty string
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  // Empty string is generally invalid for Avro binary decoding if it expects data
  // but our implementation might handle it or throw EOF.
  // Let's just ensure it doesn't crash.
  auto status = deserialized_cgs->deserialize("");
  // Depending on implementation, might return Invalid or just empty.
  // Currently checking if it survives.
  EXPECT_FALSE(status.ok());

  // Test with garbage data
  status = deserialized_cgs->deserialize("garbage_data_12345");
  EXPECT_FALSE(status.ok());
}

TEST_F(ColumnGroupsTest, TestPrivateData) {
  uint8_t private_data[] = {0x01, 0x02, 0x03, 0x04};
  auto pvec = std::vector<uint8_t>(private_data, private_data + sizeof(private_data));
  auto cg1 = std::make_shared<ColumnGroup>();
  cg1->columns = {"test_column"};
  cg1->files.emplace_back(ColumnGroupFile{
      .path = "test_path",
      .private_data = pvec,
  });
  cg1->format = LOON_FORMAT_PARQUET;

  std::vector<std::shared_ptr<ColumnGroup>> column_groups = {cg1};
  auto test_cgs = std::make_shared<ColumnGroups>(std::move(column_groups));

  ASSERT_AND_ASSIGN(auto avro_str, test_cgs->serialize());
  auto deserialized_cgs = std::make_shared<ColumnGroups>();
  ASSERT_STATUS_OK(deserialized_cgs->deserialize(avro_str));

  auto deserialized_cg = deserialized_cgs->get_column_group("test_column");
  ASSERT_EQ(deserialized_cg->files[0].private_data,
            std::vector<uint8_t>(private_data, private_data + sizeof(private_data)));
}
