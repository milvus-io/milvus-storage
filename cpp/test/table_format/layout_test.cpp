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

#include <arrow/filesystem/localfs.h>

#include "milvus-storage/table_format/layout.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

TEST(LayoutTest, MetadataFilepath) {
  EXPECT_EQ(GetCollMetadataFilepath("base", 3), "base/_metadata/3.metadata.avro");
  EXPECT_EQ(GetCollMetadataFilepath("base", 1), "base/_metadata/1.metadata.avro");
  EXPECT_EQ(GetCollMetadataFilepath("/root/col", 42), "/root/col/_metadata/42.metadata.avro");
}

TEST(LayoutTest, MetadataFilename) {
  EXPECT_EQ(GetCollMetadataFilename(1), "1.metadata.avro");
  EXPECT_EQ(GetCollMetadataFilename(100), "100.metadata.avro");
}

TEST(LayoutTest, ManifestListFilepath) {
  std::string uuid = "abc-123-def";
  EXPECT_EQ(GetManifestListFilepath("base", uuid), "base/_manifests/abc-123-def.avro");
}

TEST(LayoutTest, SegmentManifestFilepath) {
  EXPECT_EQ(GetSegmentManifestFilepath("base", "seg-uuid.avro"), "base/_manifests/seg-uuid.avro");
}

TEST(LayoutTest, GenerateUniqueId) {
  auto id1 = GenerateUniqueId();
  auto id2 = GenerateUniqueId();
  EXPECT_EQ(id1.size(), 16u);
  EXPECT_EQ(id2.size(), 16u);
  EXPECT_NE(id1, id2);
}

class LayoutVersionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    base_path_ = milvus_storage::GetTestBasePath("layout-test");
    ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_)); }

  arrow::Status CreateEmptyFile(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto out, fs_->OpenOutputStream(path));
    return out->Close();
  }

  milvus_storage::ArrowFileSystemPtr fs_;
  std::string base_path_;
};

TEST_F(LayoutVersionTest, GetLatestMetadataVersion) {
  std::string metadata_dir = GetCollMetadataDir(base_path_);
  ASSERT_STATUS_OK(fs_->CreateDir(metadata_dir));

  ASSERT_STATUS_OK(CreateEmptyFile(GetCollMetadataFilepath(base_path_, 1)));
  ASSERT_STATUS_OK(CreateEmptyFile(GetCollMetadataFilepath(base_path_, 2)));
  ASSERT_STATUS_OK(CreateEmptyFile(GetCollMetadataFilepath(base_path_, 3)));

  ASSERT_AND_ASSIGN(auto version, GetLatestMetadataVersion(fs_, base_path_));
  EXPECT_EQ(version, 3);
}

TEST_F(LayoutVersionTest, GetLatestMetadataVersionEmpty) {
  ASSERT_AND_ASSIGN(auto version, GetLatestMetadataVersion(fs_, base_path_));
  EXPECT_EQ(version, 0);
}

TEST_F(LayoutVersionTest, GetLatestMetadataVersionSkipsNonMatching) {
  std::string metadata_dir = GetCollMetadataDir(base_path_);
  ASSERT_STATUS_OK(fs_->CreateDir(metadata_dir));

  ASSERT_STATUS_OK(CreateEmptyFile(GetCollMetadataFilepath(base_path_, 1)));
  ASSERT_STATUS_OK(CreateEmptyFile(GetCollMetadataFilepath(base_path_, 5)));
  // Create a non-matching file
  ASSERT_STATUS_OK(CreateEmptyFile(metadata_dir + "/random-file.txt"));

  ASSERT_AND_ASSIGN(auto version, GetLatestMetadataVersion(fs_, base_path_));
  EXPECT_EQ(version, 5);
}

}  // namespace milvus_storage::api::table_format
