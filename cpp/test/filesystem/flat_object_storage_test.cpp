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

#include <arrow/filesystem/localfs.h>
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/filesystem/flat_object_storage.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"

#include "test_env.h"

namespace milvus_storage {

namespace {

arrow::Status WriteObject(const ArrowFileSystemPtr& fs, const std::string& path, const std::string& content) {
  ARROW_ASSIGN_OR_RAISE(auto out, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(out->Write(content.data(), static_cast<int64_t>(content.size())));
  return out->Close();
}

bool Contains(const std::vector<arrow::fs::FileInfo>& infos, const std::string& path) {
  return std::any_of(infos.begin(), infos.end(), [&](const arrow::fs::FileInfo& info) { return info.path() == path; });
}

}  // namespace

// ============================================================================
// Local backend: exercises the free-helper fallback path and the
// FileSystemProxy's NotImplemented delegation (the local base filesystem does
// not implement FlatObjectStorage). Always runs.
// ============================================================================
class FlatObjectStorageLocalTest : public ::testing::Test {
  protected:
  void SetUp() override {
    api::SetValue(properties_, PROPERTY_FS_STORAGE_TYPE, "local");
    api::SetValue(properties_, PROPERTY_FS_ROOT_PATH, "/tmp/milvus-storage-test");
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("flat-object-local-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_, /*allow_missing=*/true));
    ASSERT_STATUS_OK(fs_->CreateDir(base_path_, true));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_, /*allow_missing=*/true)); }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string base_path_;
};

TEST_F(FlatObjectStorageLocalTest, ProxyReportsNotImplementedForLocalBase) {
  // FileSystemProxy implements FlatObjectStorage but delegates to its base
  // filesystem. A local base does not implement FlatObjectStorage, so the proxy
  // reports NotImplemented and the free helpers fall back to Arrow semantics.
  // (GetFileSystem("local") returns a bare LocalFileSystemWrapper, not a proxy,
  // so construct the proxy explicitly to exercise its delegation.)
  auto base = std::make_shared<arrow::fs::LocalFileSystem>();
  auto proxy = std::make_shared<FileSystemProxy>(base_path_, base);
  auto flat = std::dynamic_pointer_cast<FlatObjectStorage>(proxy);
  ASSERT_NE(flat, nullptr);
  EXPECT_TRUE(flat->ListObjectsByPrefix("x").status().IsNotImplemented());
  EXPECT_TRUE(flat->DeleteObject("x").IsNotImplemented());
  EXPECT_TRUE(flat->ObjectExists("x").status().IsNotImplemented());
}

TEST_F(FlatObjectStorageLocalTest, ObjectExistsFallback) {
  const std::string path = base_path_ + "/exists.txt";
  ASSERT_STATUS_OK(WriteObject(fs_, path, "hello"));

  ASSERT_AND_ASSIGN(auto exists, ObjectExists(fs_, path));
  EXPECT_TRUE(exists);

  ASSERT_AND_ASSIGN(auto missing, ObjectExists(fs_, base_path_ + "/nope.txt"));
  EXPECT_FALSE(missing);

  // A directory is not an object.
  ASSERT_AND_ASSIGN(auto dir_is_obj, ObjectExists(fs_, base_path_));
  EXPECT_FALSE(dir_is_obj);
}

TEST_F(FlatObjectStorageLocalTest, DeleteObjectFallbackIsIdempotent) {
  const std::string path = base_path_ + "/to-delete.txt";
  ASSERT_STATUS_OK(WriteObject(fs_, path, "data"));

  ASSERT_STATUS_OK(DeleteObject(fs_, path));
  ASSERT_AND_ASSIGN(auto exists, ObjectExists(fs_, path));
  EXPECT_FALSE(exists);

  // Deleting a missing object is success.
  ASSERT_STATUS_OK(DeleteObject(fs_, path));
}

TEST_F(FlatObjectStorageLocalTest, ListObjectsByPrefixFallbackMatchesRawPrefix) {
  ASSERT_STATUS_OK(WriteObject(fs_, base_path_ + "/list_a1.txt", "1"));
  ASSERT_STATUS_OK(WriteObject(fs_, base_path_ + "/list_a2.txt", "2"));
  ASSERT_STATUS_OK(WriteObject(fs_, base_path_ + "/list_b1.txt", "3"));

  // Mid-segment prefix matches list_a* only.
  ASSERT_AND_ASSIGN(auto a_infos, ListObjectsByPrefix(fs_, base_path_ + "/list_a"));
  EXPECT_EQ(a_infos.size(), 2u);
  EXPECT_TRUE(Contains(a_infos, base_path_ + "/list_a1.txt"));
  EXPECT_TRUE(Contains(a_infos, base_path_ + "/list_a2.txt"));
  EXPECT_FALSE(Contains(a_infos, base_path_ + "/list_b1.txt"));

  // Broader prefix matches all three.
  ASSERT_AND_ASSIGN(auto all_infos, ListObjectsByPrefix(fs_, base_path_ + "/list_"));
  EXPECT_EQ(all_infos.size(), 3u);
}

// ============================================================================
// Remote backend: exercises the native S3 FlatObjectStorage implementation.
// Runs only against a cloud/minio environment (STORAGE_TYPE=remote), matching
// the existing cloud filesystem tests.
// ============================================================================
class FlatObjectStorageRemoteTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "remote FlatObjectStorage tests require a cloud/minio environment";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    prefix_ = "flat-object-remote-test";
  }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string prefix_;
};

TEST_F(FlatObjectStorageRemoteTest, ListObjectsByPrefixIsRawNotDirectory) {
  ASSERT_STATUS_OK(WriteObject(fs_, prefix_ + "/seg/one.txt", "1"));
  ASSERT_STATUS_OK(WriteObject(fs_, prefix_ + "/seg/two.txt", "2"));
  // Sibling object that shares the "seg" mid-segment prefix but is not under seg/.
  ASSERT_STATUS_OK(WriteObject(fs_, prefix_ + "/segment.txt", "3"));

  // Raw object-key prefix "…/seg" matches the two seg/ objects AND segment.txt,
  // which a directory-scoped listing could not do.
  ASSERT_AND_ASSIGN(auto seg_infos, ListObjectsByPrefix(fs_, prefix_ + "/seg"));
  EXPECT_TRUE(Contains(seg_infos, prefix_ + "/seg/one.txt"));
  EXPECT_TRUE(Contains(seg_infos, prefix_ + "/seg/two.txt"));
  EXPECT_TRUE(Contains(seg_infos, prefix_ + "/segment.txt"));

  // Boundary prefix "…/seg/" matches only the objects under seg/.
  ASSERT_AND_ASSIGN(auto boundary_infos, ListObjectsByPrefix(fs_, prefix_ + "/seg/"));
  EXPECT_TRUE(Contains(boundary_infos, prefix_ + "/seg/one.txt"));
  EXPECT_TRUE(Contains(boundary_infos, prefix_ + "/seg/two.txt"));
  EXPECT_FALSE(Contains(boundary_infos, prefix_ + "/segment.txt"));

  ASSERT_STATUS_OK(DeleteObject(fs_, prefix_ + "/seg/one.txt"));
  ASSERT_STATUS_OK(DeleteObject(fs_, prefix_ + "/seg/two.txt"));
  ASSERT_STATUS_OK(DeleteObject(fs_, prefix_ + "/segment.txt"));
}

TEST_F(FlatObjectStorageRemoteTest, ObjectExistsHasExactObjectSemantics) {
  ASSERT_STATUS_OK(WriteObject(fs_, prefix_ + "/dir/child.txt", "x"));

  ASSERT_AND_ASSIGN(auto child_exists, ObjectExists(fs_, prefix_ + "/dir/child.txt"));
  EXPECT_TRUE(child_exists);

  // "…/dir" has a descendant (and possibly a directory marker) but is not
  // itself an object: exact-key HeadObject semantics report it as missing.
  ASSERT_AND_ASSIGN(auto dir_is_obj, ObjectExists(fs_, prefix_ + "/dir"));
  EXPECT_FALSE(dir_is_obj);

  ASSERT_AND_ASSIGN(auto missing, ObjectExists(fs_, prefix_ + "/dir/absent.txt"));
  EXPECT_FALSE(missing);

  ASSERT_STATUS_OK(DeleteObject(fs_, prefix_ + "/dir/child.txt"));
}

TEST_F(FlatObjectStorageRemoteTest, DeleteObjectIsIdempotent) {
  const std::string path = prefix_ + "/deletable.txt";
  ASSERT_STATUS_OK(WriteObject(fs_, path, "y"));
  ASSERT_AND_ASSIGN(auto exists, ObjectExists(fs_, path));
  ASSERT_TRUE(exists);

  ASSERT_STATUS_OK(DeleteObject(fs_, path));
  ASSERT_AND_ASSIGN(auto gone, ObjectExists(fs_, path));
  EXPECT_FALSE(gone);

  // Deleting a missing object is success (no HeadObject, no error).
  ASSERT_STATUS_OK(DeleteObject(fs_, path));
}

}  // namespace milvus_storage
