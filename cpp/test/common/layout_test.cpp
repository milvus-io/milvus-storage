// Copyright 2024 Zilliz
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

#include <unistd.h>
#include <filesystem>

#include <arrow/api.h>

#include "milvus-storage/common/layout.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "test_env.h"

namespace milvus_storage {

using namespace milvus_storage::api;

class FileLayoutTest : public ::testing::Test {
  protected:
  void SetUp() override {
    base_path_ = "layout_test";
    new_base_path_ = "new_layout_test";
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // no need create dir in new_base_path_, just delete it
    ASSERT_STATUS_OK(DeleteTestDir(fs_, new_base_path_));
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override {
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(DeleteTestDir(fs_, new_base_path_));
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  std::string new_base_path_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  Properties properties_;
};

TEST_F(FileLayoutTest, CheckLayoutCreation) {
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());

  ASSERT_AND_ASSIGN(auto transaction, api::transaction::Transaction::Open(fs_, base_path_));
  transaction->AppendFiles(*column_groups);
  ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
  ASSERT_GT(committed_version, 0);

  // Verify Directory Structure
  arrow::fs::FileSelector selector;
  selector.base_dir = base_path_;
  selector.recursive = true;
  ASSERT_AND_ASSIGN(auto file_infos, fs_->GetFileInfo(selector));

  bool found_metadata_dir = false;
  bool found_data_dir = false;
  bool found_manifest = false;
  bool found_data_file = false;

  // Construct expected paths using constants
  std::string metadata_path = get_manifest_path(base_path_);
  std::string data_path = get_data_path(base_path_);
  std::string manifest_prefix = base_path_ + "/" + kMetadataPath + kManifestFileNamePrefix;

  auto path_eq = [](const std::string& path, const std::string& expected) {
    std::filesystem::path lp(path);
    std::filesystem::path rp(expected);
    return (lp / "").lexically_normal() == (rp / "").lexically_normal();
  };

  for (const auto& info : file_infos) {
    // Use full path matching if possible, or Ensure trailing slashes
    std::string path = info.path();
    found_metadata_dir |= (info.type() == arrow::fs::FileType::Directory && path_eq(path, metadata_path));
    found_data_dir |= (info.type() == arrow::fs::FileType::Directory && path_eq(path, data_path));
    found_manifest |= (info.type() == arrow::fs::FileType::File && path.find(manifest_prefix) == 0);

    // Check if file is in data dir
    // Ensure we only match files INSIDE data dir, not the dir itself
    if (path.find(data_path) == 0 && path != data_path && info.type() == arrow::fs::FileType::File) {
      found_data_file = true;
      // check if file name starts with {group_id}_{uuid} (no column_group_ prefix)
      std::string filename = info.base_name();
      EXPECT_EQ(filename.find("column_group_"), std::string::npos)
          << "File name should not start with column_group_: " << filename;
    }
  }

  EXPECT_TRUE(found_metadata_dir) << "Metadata directory not found at " << metadata_path;
  EXPECT_TRUE(found_data_dir) << "Data directory not found at " << data_path;
  EXPECT_TRUE(found_manifest) << "Manifest file not found starting with " << manifest_prefix;
  EXPECT_TRUE(found_data_file) << "No data file found in " << data_path;
}

TEST_F(FileLayoutTest, TestChangeBasePath) {
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());

  // write manifest
  {
    ASSERT_AND_ASSIGN(auto transaction, api::transaction::Transaction::Open(fs_, base_path_));
    transaction->AppendFiles(*column_groups);
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_GT(committed_version, 0);
  }

  auto verify_read = [&](auto cgs) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    ASSERT_NE(reader, nullptr);
    ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
    ASSERT_NE(batch_reader, nullptr);

    ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
    ASSERT_STATUS_OK(batch_reader->Close());
    ASSERT_AND_ASSIGN(auto combined_batch, table->CombineChunksToBatch());
    ASSERT_STATUS_OK(ValidateRowAlignment(combined_batch));
  };

  verify_read(column_groups);

  ASSERT_STATUS_OK(MoveTestBasePath(fs_, base_path_, new_base_path_));
  std::shared_ptr<ColumnGroups> new_column_groups;

  // read manifest in new_base_path_
  {
    ASSERT_AND_ASSIGN(auto transaction, api::transaction::Transaction::Open(fs_, new_base_path_));
    ASSERT_AND_ASSIGN(auto new_manifest, transaction->GetManifest());
    new_column_groups = std::make_shared<ColumnGroups>(new_manifest->columnGroups());
  }

  verify_read(new_column_groups);
}

// ---------- Layout Path Function Unit Tests (no filesystem needed) ----------

class LayoutPathTest : public ::testing::Test {};

TEST_F(LayoutPathTest, TestLayoutPaths) {
  // get_data_filename — format and uniqueness
  {
    auto filename = get_data_filename(0, "parquet");
    EXPECT_EQ(filename.substr(0, 2), "0_");
    EXPECT_NE(filename.find(".parquet"), std::string::npos);

    auto filename_vortex = get_data_filename(5, "vortex");
    EXPECT_EQ(filename_vortex.substr(0, 2), "5_");
    EXPECT_NE(filename_vortex.find(".vortex"), std::string::npos);

    // UUIDs should differ
    EXPECT_NE(get_data_filename(0, "parquet"), get_data_filename(0, "parquet"));
  }

  // get_manifest_filename
  {
    EXPECT_EQ(get_manifest_filename(1), "manifest-1.avro");
    EXPECT_EQ(get_manifest_filename(42), "manifest-42.avro");
  }

  // get_manifest_path / get_manifest_filepath
  {
    EXPECT_NE(get_manifest_path("/base").find("_metadata"), std::string::npos);

    auto path = get_manifest_filepath("/base", 3);
    EXPECT_NE(path.find("_metadata"), std::string::npos);
    EXPECT_NE(path.find("manifest-3.avro"), std::string::npos);
  }

  // get_data_path / get_data_filepath
  {
    EXPECT_NE(get_data_path("/base").find("_data"), std::string::npos);

    auto path = get_data_filepath("/base", 0, "parquet");
    EXPECT_NE(path.find("_data"), std::string::npos);
    EXPECT_NE(path.find(".parquet"), std::string::npos);

    auto path_by_name = get_data_filepath("/base", "myfile.parquet");
    EXPECT_NE(path_by_name.find("_data"), std::string::npos);
    EXPECT_NE(path_by_name.find("myfile.parquet"), std::string::npos);
  }

  // get_delta_path / get_delta_filepath
  {
    EXPECT_NE(get_delta_path("/base").find("_delta"), std::string::npos);

    auto path = get_delta_filepath("/base", "delta_001.log");
    EXPECT_NE(path.find("_delta"), std::string::npos);
    EXPECT_NE(path.find("delta_001.log"), std::string::npos);
  }

  // get_stats_path / get_stats_filepath
  {
    EXPECT_NE(get_stats_path("/base").find("_stats"), std::string::npos);

    auto path = get_stats_filepath("/base", "bloom.bin");
    EXPECT_NE(path.find("_stats"), std::string::npos);
    EXPECT_NE(path.find("bloom.bin"), std::string::npos);
  }
}

}  // namespace milvus_storage
