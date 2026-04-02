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

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "iceberg_bridge.h"

namespace milvus_storage::iceberg {

class IcebergBridgeTest : public ::testing::Test {};

// PlanFiles should throw IcebergException for a nonexistent local metadata file
TEST_F(IcebergBridgeTest, PlanFilesNonexistentLocalMetadata) {
  std::unordered_map<std::string, std::string> opts;
  EXPECT_THROW(PlanFiles("/nonexistent/path/v1.metadata.json", 1, opts), IcebergException);
}

// PlanFiles should throw IcebergException for an empty metadata location
TEST_F(IcebergBridgeTest, PlanFilesEmptyMetadataLocation) {
  std::unordered_map<std::string, std::string> opts;
  EXPECT_THROW(PlanFiles("", 1, opts), IcebergException);
}

// PlanFiles should throw IcebergException with an invalid snapshot id
// even if the metadata file does not exist
TEST_F(IcebergBridgeTest, PlanFilesInvalidSnapshotId) {
  std::unordered_map<std::string, std::string> opts;
  EXPECT_THROW(PlanFiles("file:///nonexistent/metadata.json", -999, opts), IcebergException);
}

// Verify IcebergException carries a descriptive message
TEST_F(IcebergBridgeTest, ExceptionMessageIsDescriptive) {
  std::unordered_map<std::string, std::string> opts;
  try {
    PlanFiles("/nonexistent/v1.metadata.json", 1, opts);
    FAIL() << "Expected IcebergException";
  } catch (const IcebergException& e) {
    std::string msg = e.what();
    EXPECT_FALSE(msg.empty()) << "Exception message should not be empty";
  }
}

// IcebergFileInfo default construction
TEST_F(IcebergBridgeTest, FileInfoStructConstruction) {
  IcebergFileInfo info;
  info.data_file_path = "s3://bucket/table/data/file.parquet";
  info.record_count = 1000;
  info.delete_metadata_json = {'{', '}'};

  EXPECT_EQ(info.data_file_path, "s3://bucket/table/data/file.parquet");
  EXPECT_EQ(info.record_count, 1000);
  EXPECT_EQ(info.delete_metadata_json.size(), 2);
}

// Empty delete metadata
TEST_F(IcebergBridgeTest, FileInfoEmptyDeleteMetadata) {
  IcebergFileInfo info;
  info.data_file_path = "data.parquet";
  info.record_count = 100;

  EXPECT_TRUE(info.delete_metadata_json.empty());
}

}  // namespace milvus_storage::iceberg
