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
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <unistd.h>

#include "milvus-storage/common/file_layout.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/transaction/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "test_env.h"

namespace milvus_storage {

class FileLayoutTest : public ::testing::Test {
  protected:
  void SetUp() override {
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    base_path_ = "/tmp/milvus_storage_layout_test_" + std::to_string(getpid());
    if (fs_->GetFileInfo(base_path_).ValueOrDie().type() != arrow::fs::FileType::NotFound) {
      ASSERT_STATUS_OK(fs_->DeleteDirContents(base_path_));
      ASSERT_STATUS_OK(fs_->DeleteDir(base_path_));
    }
    ASSERT_STATUS_OK(fs_->CreateDir(base_path_));

    schema_ = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("data", arrow::float64())});

    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties_, "/", base_path_));
  }

  void TearDown() override {
    ASSERT_STATUS_OK(fs_->DeleteDirContents(base_path_));
    ASSERT_STATUS_OK(fs_->DeleteDir(base_path_));
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;
};

TEST_F(FileLayoutTest, CheckLayoutCreation) {
  auto policy_res = api::ColumnGroupPolicy::create_column_group_policy(properties_, schema_);
  ASSERT_STATUS_OK(policy_res.status());
  auto policy = std::move(policy_res.ValueOrDie());
  auto writer = api::Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write some data
  auto id_builder = std::make_shared<arrow::Int64Builder>();
  auto data_builder = std::make_shared<arrow::DoubleBuilder>();
  ASSERT_STATUS_OK(id_builder->Append(1));
  ASSERT_STATUS_OK(data_builder->Append(1.0));
  auto id_array = id_builder->Finish().ValueOrDie();
  auto data_array = data_builder->Finish().ValueOrDie();
  auto batch = arrow::RecordBatch::Make(schema_, 1, {id_array, data_array});

  ASSERT_STATUS_OK(writer->write(batch));
  auto close_result = writer->close({}, {});
  ASSERT_STATUS_OK(close_result.status());
  auto column_groups = close_result.ValueOrDie();

  auto transaction = std::make_shared<api::transaction::TransactionImpl<api::ColumnGroups>>(properties_, base_path_);
  ASSERT_STATUS_OK(transaction->begin());
  auto commit_res = transaction->commit(column_groups, api::transaction::UpdateType::APPENDFILES,
                                        api::transaction::TransResolveStrategy::RESOLVE_FAIL);
  ASSERT_STATUS_OK(commit_res.status());
  ASSERT_TRUE(commit_res.ValueOrDie().success) << "Commit failed: " << commit_res.ValueOrDie().failed_message;

  // Verify Directory Structure
  arrow::fs::FileSelector selector;
  selector.base_dir = base_path_;
  selector.recursive = true;
  auto file_infos = fs_->GetFileInfo(selector).ValueOrDie();

  bool found_metadata_dir = false;
  bool found_data_dir = false;
  bool found_manifest = false;
  bool found_data_file = false;

  // Construct expected paths using constants
  std::string metadata_path = base_path_ + "/" + kMetadataDir;
  std::string data_path = base_path_ + "/" + kDataDir;
  // kManifestFilePrefix is relative like "_metadata/manifest-", so we need to construct absolute search string
  std::string manifest_prefix = base_path_ + "/" + kManifestFilePrefix;

  for (const auto& info : file_infos) {
    // Use full path matching if possible, or Ensure trailing slashes
    std::string path = info.path();
    if (path == metadata_path && info.type() == arrow::fs::FileType::Directory) {
      found_metadata_dir = true;
    }
    if (path == data_path && info.type() == arrow::fs::FileType::Directory) {
      found_data_dir = true;
    }
    if (path.find(manifest_prefix) == 0 && info.type() == arrow::fs::FileType::File) {
      found_manifest = true;
    }
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

}  // namespace milvus_storage