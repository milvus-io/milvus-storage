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
#include "milvus-storage/manifest.h"
#include "milvus-storage/manifest_json.h"

using namespace milvus_storage::api;
using milvus_storage::JsonManifestSerDe;

class ManifestJsonTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Create test column groups
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = {"id", "name", "age"};
    cg1->paths = {"/data/cg1_part1.parquet", "/data/cg1_part2.parquet"};
    cg1->format = "parquet";

    auto cg2 = std::make_shared<ColumnGroup>();
    cg2->columns = {"embedding", "metadata"};
    cg2->paths = {"/data/cg2_vectors.lance"};
    cg2->format = "lance";

    std::vector<std::shared_ptr<ColumnGroup>> column_groups = {cg1, cg2};
    test_manifest_ = std::make_shared<Manifest>(std::move(column_groups), 42);
  }

  std::shared_ptr<Manifest> test_manifest_;
  milvus_storage::JsonManifestSerDe serializer_;
};

TEST_F(ManifestJsonTest, SerializeDeserialize) {
  // Serialize to JSON
  std::ostringstream output;
  ASSERT_TRUE(serializer_.Serialize(test_manifest_, output));

  std::string json_str = output.str();
  EXPECT_FALSE(json_str.empty());
  EXPECT_NE(json_str.find("\"version\": 42"), std::string::npos);
  EXPECT_NE(json_str.find("\"column_groups\""), std::string::npos);

  // Deserialize from JSON
  std::istringstream input(json_str);
  auto deserialized_manifest = serializer_.Deserialize(input);

  // Verify data integrity
  EXPECT_EQ(deserialized_manifest->version(), 42);

  const auto& groups = deserialized_manifest->get_column_groups();
  EXPECT_EQ(groups.size(), 2);

  // Check first column group
  EXPECT_EQ(groups[0]->columns.size(), 3);
  EXPECT_EQ(groups[0]->columns[0], "id");
  EXPECT_EQ(groups[0]->columns[1], "name");
  EXPECT_EQ(groups[0]->columns[2], "age");
  EXPECT_EQ(groups[0]->paths.size(), 2);
  EXPECT_EQ(groups[0]->format, "parquet");

  // Check second column group
  EXPECT_EQ(groups[1]->columns.size(), 2);
  EXPECT_EQ(groups[1]->columns[0], "embedding");
  EXPECT_EQ(groups[1]->columns[1], "metadata");
  EXPECT_EQ(groups[1]->paths.size(), 1);
  EXPECT_EQ(groups[1]->format, "lance");
}

TEST_F(ManifestJsonTest, EmptyManifest) {
  // Test empty manifest
  std::vector<std::shared_ptr<ColumnGroup>> column_groups = {};
  auto empty_manifest = std::make_shared<Manifest>(column_groups, 0);

  std::ostringstream output;
  ASSERT_TRUE(serializer_.Serialize(empty_manifest, output));

  std::istringstream input(output.str());
  auto deserialized_manifest = std::make_shared<Manifest>(column_groups, 1);
  deserialized_manifest = serializer_.Deserialize(input);

  EXPECT_EQ(deserialized_manifest->version(), 0);
  EXPECT_EQ(deserialized_manifest->get_column_groups().size(), 0);
}

TEST_F(ManifestJsonTest, ColumnLookup) {
  // Serialize and deserialize
  std::ostringstream output;
  ASSERT_TRUE(serializer_.Serialize(test_manifest_, output));

  std::istringstream input(output.str());
  auto deserialized_manifest = serializer_.Deserialize(input);

  // Test column lookup functionality
  auto name_cg = deserialized_manifest->get_column_group("name");
  ASSERT_NE(name_cg, nullptr);
  EXPECT_EQ(name_cg->format, "parquet");

  auto embedding_cg = deserialized_manifest->get_column_group("embedding");
  ASSERT_NE(embedding_cg, nullptr);
  EXPECT_EQ(embedding_cg->format, "lance");

  auto missing_cg = deserialized_manifest->get_column_group("nonexistent");
  EXPECT_EQ(missing_cg, nullptr);
}

TEST_F(ManifestJsonTest, InvalidJson) {
  // Test deserialization with invalid JSON
  std::istringstream invalid_input("invalid json");
  auto manifest = serializer_.Deserialize(invalid_input);
  EXPECT_TRUE(manifest == nullptr);
}
