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

// Cloud filesystem tests that work for all providers (S3, Azure, GCS, etc.)

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <thread>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/filesystem/upload_sizable.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/fs.h"

#include "test_env.h"

namespace milvus_storage {

// ============================================================================
// Cloud-env tests (existing)
// ============================================================================

class CloudFsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "S3 tests skipped in non-cloud environment";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
  }

  milvus_storage::api::Properties properties_;
  ArrowFileSystemPtr fs_;
};

TEST_F(CloudFsTest, ConditionalWrite) {
  auto provider = GetEnvVar("CLOUD_PROVIDER");
  if (provider.ok() && provider.ValueOrDie() == "azure") {
    GTEST_SKIP()
        << "Azure conditional write has different semantics (fail on open, not close), see ConditionalWriteAzure";
  }
  std::string file_to = "/test_conditional_write.txt";

  // Ensure source file does not exist
  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  // Create source file
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
    // check file exists, it should be a file
    ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(file_to));
    ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
  }

  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_NOT_OK(output_stream->Close());
  }

  (void)fs_->DeleteFile(file_to);

  // Test conditional write in output_stream close
  {
    std::shared_ptr<arrow::Buffer> buffer1 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());
    std::shared_ptr<arrow::Buffer> buffer2 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream1, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream1->Write(buffer1));

    ASSERT_AND_ASSIGN(auto output_stream2, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream2->Write(buffer2));

    ASSERT_STATUS_OK(output_stream1->Close());
    auto write_status = output_stream2->Close();
    ASSERT_FALSE(write_status.ok());
  }
}

// Azure conditional write fails at open time (Init creates blob with IfNoneMatch),
// unlike S3 which fails at close time.
TEST_F(CloudFsTest, ConditionalWriteAzure) {
  auto provider = GetEnvVar("CLOUD_PROVIDER");
  if (!provider.ok() || provider.ValueOrDie() != "azure") {
    GTEST_SKIP() << "Azure-specific conditional write test";
  }
  std::string file_to = "/test_conditional_write_azure.txt";

  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  // First conditional write should succeed
  {
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.data()), content1.size());
    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto out, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());

    ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(file_to));
    ASSERT_EQ(info.type(), arrow::fs::FileType::File);
  }

  // Second conditional write should fail at open (blob already exists)
  {
    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    auto result = conditional_fs->OpenConditionalOutputStream(file_to, nullptr);
    ASSERT_FALSE(result.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
    ASSERT_NE(detail, nullptr);
    ASSERT_EQ(detail->code(), ExtendStatusCode::AwsErrorPreConditionFailed);
  }

  // Original content should be preserved
  {
    ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(file_to));
    ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
    EXPECT_EQ(read_buf->ToString(), content1);
  }

  (void)fs_->DeleteFile(file_to);
}

TEST_F(CloudFsTest, TestMetadata) {
  // predefined metadata
  {
    std::string file_to = "/predefined_metadata.txt";
    (void)fs_->DeleteFile(file_to);
    std::string content = "This is a test file for metadata.";

    auto kvmeta = arrow::KeyValueMetadata::Make({"Content-Language"}, {"zh-CN"});
    ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_to, kvmeta));
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.c_str()), content.size());

    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }

  // custom metadata
  {
    std::string file_to = "/custom_metadata.txt";
    (void)fs_->DeleteFile(file_to);
    std::string content = "This is a test file for custom metadata.";
    auto kvmeta = arrow::KeyValueMetadata::Make({"Content-Disposition"}, {"inline"});
    ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_to, kvmeta));
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.c_str()), content.size());
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }
}

// ============================================================================
// background_writes cloud-env tests
// ============================================================================

TEST_F(CloudFsTest, BackgroundWritesConcurrent) {
  constexpr int kNumThreads = 10;
  const std::string base_dir = "/test_background_writes";

  auto run_concurrent_writes = [&](bool background_writes) {
    // Clear the global fs cache to force re-creation with new config
    FilesystemCache::getInstance().clean();

    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    api::SetValue(properties, PROPERTY_FS_BACKGROUND_WRITES, background_writes ? "true" : "false");

    ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));

    std::string dir = base_dir + (background_writes ? "/bg_true" : "/bg_false");
    (void)fs->DeleteDirContents(dir, true);
    ASSERT_STATUS_OK(fs->CreateDir(dir));

    std::vector<std::thread> threads;
    std::vector<arrow::Status> statuses(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back([&, i]() {
        std::string path = dir + "/file_" + std::to_string(i) + ".txt";
        std::string content = "thread_" + std::to_string(i) + "_data";

        auto out_result = fs->OpenOutputStream(path);
        if (!out_result.ok()) {
          statuses[i] = out_result.status();
          return;
        }
        auto out = out_result.ValueOrDie();
        auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
        statuses[i] = out->Write(buf);
        if (statuses[i].ok()) {
          statuses[i] = out->Close();
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    for (int i = 0; i < kNumThreads; ++i) {
      EXPECT_TRUE(statuses[i].ok()) << "Thread " << i << " failed: " << statuses[i].ToString();
    }

    // Verify all files exist and content is correct
    for (int i = 0; i < kNumThreads; ++i) {
      std::string path = dir + "/file_" + std::to_string(i) + ".txt";
      std::string expected = "thread_" + std::to_string(i) + "_data";

      ASSERT_AND_ASSIGN(auto input, fs->OpenInputStream(path));
      ASSERT_AND_ASSIGN(auto buf, input->Read(expected.size()));
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(buf->data()), buf->size()), expected)
          << "Content mismatch for thread " << i;
    }

    // Cleanup
    (void)fs->DeleteDirContents(dir, true);
  };

  // 1. background_writes = true
  run_concurrent_writes(true);

  // 2. background_writes = false
  run_concurrent_writes(false);

  (void)fs_->DeleteDirContents(base_dir, true);
}

// ============================================================================
// use_crc32c_checksum cloud-env tests
// ============================================================================

TEST_F(CloudFsTest, Crc32cChecksumWriteAndRead) {
  auto provider = GetEnvVar("CLOUD_PROVIDER");
  if (!provider.ok() || provider.ValueOrDie() != "aws") {
    GTEST_SKIP() << "CRC32C checksum is S3-specific, only runs with CLOUD_PROVIDER=aws";
  }
  const std::string base_dir = "/test_crc32c_checksum";

  auto run_with_checksum = [&](bool use_crc32c) {
    FilesystemCache::getInstance().clean();

    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    api::SetValue(properties, PROPERTY_FS_USE_CRC32C_CHECKSUM, use_crc32c ? "true" : "false");
    // Use the minimum S3 part size (5MB) to trigger multipart upload with less data
    api::SetValue(properties, PROPERTY_FS_MULTI_PART_UPLOAD_SIZE, "5242880");

    ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));

    std::string dir = base_dir + (use_crc32c ? "/crc32c_on" : "/crc32c_off");
    (void)fs->DeleteDirContents(dir, true);

    // 1. CreateDir - exercises CreateEmptyDir (PutObjectRequest with CRC32C)
    ASSERT_STATUS_OK(fs->CreateDir(dir));
    std::string subdir = dir + "/subdir";
    ASSERT_STATUS_OK(fs->CreateDir(subdir));

    // 2. Single PutObject - small file write (PutObjectRequest via Upload template)
    std::string path = dir + "/test_file.txt";
    std::string content = "Hello, CRC32C checksum test!";
    {
      ASSERT_AND_ASSIGN(auto out, fs->OpenOutputStream(path));
      auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
      ASSERT_STATUS_OK(out->Write(buf));
      ASSERT_STATUS_OK(out->Close());
    }

    // Read back and verify
    {
      ASSERT_AND_ASSIGN(auto input, fs->OpenInputStream(path));
      ASSERT_AND_ASSIGN(auto buf, input->Read(content.size()));
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(buf->data()), buf->size()), content);
    }

    // 3. Multipart upload - write a buffer larger than part size to trigger multipart
    {
      std::string mp_path = dir + "/multipart_file.bin";
      const int64_t kTotalSize = 15LL * 1024 * 1024;  // 15MB, guarantees multiple parts
      std::string large_content(kTotalSize, 'A');
      // Fill with a pattern so we can verify integrity
      for (int64_t i = 0; i < kTotalSize; ++i) {
        large_content[i] = static_cast<char>('A' + (i % 26));
      }
      {
        ASSERT_AND_ASSIGN(auto out, fs->OpenOutputStream(mp_path));
        auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(large_content.data()),
                                                   large_content.size());
        ASSERT_STATUS_OK(out->Write(buf));
        ASSERT_STATUS_OK(out->Close());
      }
      // Read back and verify
      {
        ASSERT_AND_ASSIGN(auto input, fs->OpenInputStream(mp_path));
        ASSERT_AND_ASSIGN(auto buf, input->Read(kTotalSize));
        ASSERT_EQ(buf->size(), kTotalSize);
        EXPECT_EQ(std::string(reinterpret_cast<const char*>(buf->data()), buf->size()), large_content);
      }
    }

    // 5. CopyObject - exercises CopyObjectRequest with CRC32C
    std::string copy_path = dir + "/test_file_copy.txt";
    ASSERT_STATUS_OK(fs->CopyFile(path, copy_path));
    {
      ASSERT_AND_ASSIGN(auto input, fs->OpenInputStream(copy_path));
      ASSERT_AND_ASSIGN(auto buf, input->Read(content.size()));
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(buf->data()), buf->size()), content);
    }

    // 6. DeleteDirContents - exercises DeleteObjectsRequest with CRC32C (batch delete)
    // Create several files then batch-delete them
    for (int i = 0; i < 3; ++i) {
      std::string p = subdir + "/del_" + std::to_string(i) + ".txt";
      ASSERT_AND_ASSIGN(auto out, fs->OpenOutputStream(p));
      std::string data = "delete_me_" + std::to_string(i);
      auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(data.data()), data.size());
      ASSERT_STATUS_OK(out->Write(buf));
      ASSERT_STATUS_OK(out->Close());
    }
    ASSERT_STATUS_OK(fs->DeleteDirContents(subdir));

    // Cleanup
    (void)fs->DeleteDirContents(dir, true);
  };

  // 1. use_crc32c_checksum = true
  run_with_checksum(true);

  // 2. use_crc32c_checksum = false
  run_with_checksum(false);

  (void)fs_->DeleteDirContents(base_dir, true);
}

// ============================================================================
// Generic cloud filesystem tests (work for S3, Azure, GCS, etc.)
// Ported from Arrow v23 azurefs_test.cc and extended.
// ============================================================================

TEST_F(CloudFsTest, WriteAndReadSmallFile) {
  std::string path = "/test_small_rw.txt";
  std::string content = "hello cloud storage";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->ToString(), content);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, WriteAndReadLargeFile) {
  // 12MB exceeds the 10MB Azure block buffer and S3 multipart thresholds
  std::string path = "/test_large_rw.bin";
  (void)fs_->DeleteFile(path);

  const int64_t size = 12 * 1024 * 1024;
  std::string content(size, '\0');
  std::mt19937 rng(42);
  for (auto& c : content) {
    c = static_cast<char>(rng() % 256);
  }

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputFile(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(size + 1));
  EXPECT_EQ(read_buf->size(), size);
  EXPECT_EQ(read_buf->ToString(), content);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, WriteMultipleChunks) {
  std::string path = "/test_multi_chunk.txt";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  std::string part1 = "first part, ";
  std::string part2 = "second part, ";
  std::string part3 = "third part.";
  ASSERT_STATUS_OK(out->Write(part1.data(), part1.size()));
  ASSERT_STATUS_OK(out->Write(part2.data(), part2.size()));
  ASSERT_STATUS_OK(out->Write(part3.data(), part3.size()));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->ToString(), part1 + part2 + part3);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, ReadInSmallBuffers) {
  std::string path = "/test_small_buffers.txt";
  std::string content = "0123456789abcdef";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  std::string result;
  std::shared_ptr<arrow::Buffer> chunk;
  do {
    ASSERT_AND_ASSIGN(chunk, in->Read(4));
    result.append(chunk->ToString());
  } while (chunk && chunk->size() != 0);
  EXPECT_EQ(result, content);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, GetFileInfo) {
  std::string path = "/test_file_info.txt";
  (void)fs_->DeleteFile(path);

  // Not found
  ASSERT_AND_ASSIGN(auto info_before, fs_->GetFileInfo(path));
  EXPECT_EQ(info_before.type(), arrow::fs::FileType::NotFound);

  // Create
  std::string content = "file info test data";
  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  // Found
  ASSERT_AND_ASSIGN(auto info_after, fs_->GetFileInfo(path));
  EXPECT_EQ(info_after.type(), arrow::fs::FileType::File);
  EXPECT_EQ(info_after.size(), static_cast<int64_t>(content.size()));

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, DeleteFile) {
  std::string path = "/test_delete_file.txt";
  std::string content = "to be deleted";

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_STATUS_OK(fs_->DeleteFile(path));

  ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(path));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(CloudFsTest, OpenInputStreamNotFound) {
  auto result = fs_->OpenInputStream("/nonexistent_file_12345.txt");
  ASSERT_FALSE(result.ok());
}

TEST_F(CloudFsTest, OpenInputFileRandomRead) {
  std::string path = "/test_random_read.txt";
  std::string content = "0123456789abcdefghijklmnopqrstuvwxyz";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path));

  // ReadAt from middle
  ASSERT_AND_ASSIGN(auto buf1, file->ReadAt(10, 6));
  EXPECT_EQ(buf1->ToString(), "abcdef");

  // ReadAt from beginning
  ASSERT_AND_ASSIGN(auto buf2, file->ReadAt(0, 4));
  EXPECT_EQ(buf2->ToString(), "0123");

  // ReadAt beyond end
  ASSERT_AND_ASSIGN(auto buf3, file->ReadAt(30, 100));
  EXPECT_EQ(buf3->ToString(), "uvwxyz");

  // GetSize
  ASSERT_AND_ASSIGN(auto size, file->GetSize());
  EXPECT_EQ(size, static_cast<int64_t>(content.size()));

  // Seek + Read
  ASSERT_STATUS_OK(file->Seek(26));
  ASSERT_AND_ASSIGN(auto buf4, file->Read(10));
  EXPECT_EQ(buf4->ToString(), "qrstuvwxyz");

  // Tell
  ASSERT_AND_ASSIGN(auto pos, file->Tell());
  EXPECT_EQ(pos, 36);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, OverwriteExistingFile) {
  std::string path = "/test_overwrite.txt";
  (void)fs_->DeleteFile(path);

  // Write v1
  {
    std::string content = "version 1";
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());
  }

  // Overwrite with v2
  {
    std::string content = "version 2 is longer";
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());
  }

  // Read should get v2
  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->ToString(), "version 2 is longer");

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, CopyFile) {
  std::string src = "/test_copy_src.txt";
  std::string dst = "/test_copy_dst.txt";
  std::string content = "copy me";
  (void)fs_->DeleteFile(src);
  (void)fs_->DeleteFile(dst);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(src, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_STATUS_OK(fs_->CopyFile(src, dst));

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(dst));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->ToString(), content);

  (void)fs_->DeleteFile(src);
  (void)fs_->DeleteFile(dst);
}

TEST_F(CloudFsTest, CopyFileSourceNotFound) {
  auto status = fs_->CopyFile("/nonexistent_src_12345.txt", "/nonexistent_dst_12345.txt");
  ASSERT_FALSE(status.ok());
}

TEST_F(CloudFsTest, CreateAndDeleteDir) {
  std::string dir = "/test_dir_ops";
  (void)fs_->DeleteDir(dir);

  ASSERT_STATUS_OK(fs_->CreateDir(dir, false));
  ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(dir));
  EXPECT_EQ(info.type(), arrow::fs::FileType::Directory);

  ASSERT_STATUS_OK(fs_->DeleteDir(dir));
}

TEST_F(CloudFsTest, CreateDirRecursive) {
  std::string dir = "/test_recursive_dir/sub1/sub2";
  (void)fs_->DeleteDir("/test_recursive_dir");

  ASSERT_STATUS_OK(fs_->CreateDir(dir, true));
  ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(dir));
  EXPECT_EQ(info.type(), arrow::fs::FileType::Directory);

  (void)fs_->DeleteDir("/test_recursive_dir");
}

TEST_F(CloudFsTest, DeleteDirContents) {
  std::string dir = "/test_delete_contents";
  std::string file1 = dir + "/f1.txt";
  std::string file2 = dir + "/f2.txt";
  (void)fs_->DeleteDir(dir);

  ASSERT_STATUS_OK(fs_->CreateDir(dir, false));

  for (const auto& f : {file1, file2}) {
    std::string c = "data";
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(f, nullptr));
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(c.data()), c.size());
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());
  }

  ASSERT_STATUS_OK(fs_->DeleteDirContents(dir, false));

  ASSERT_AND_ASSIGN(auto info1, fs_->GetFileInfo(file1));
  EXPECT_EQ(info1.type(), arrow::fs::FileType::NotFound);
  ASSERT_AND_ASSIGN(auto info2, fs_->GetFileInfo(file2));
  EXPECT_EQ(info2.type(), arrow::fs::FileType::NotFound);

  (void)fs_->DeleteDir(dir);
}

TEST_F(CloudFsTest, GetFileInfoSelector) {
  std::string dir = "/test_selector";
  (void)fs_->DeleteDir(dir);
  ASSERT_STATUS_OK(fs_->CreateDir(dir, false));

  // Create files
  for (const auto& name : {"a.txt", "b.txt", "c.txt"}) {
    std::string path = std::string(dir) + "/" + name;
    std::string content = name;
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());
  }

  // Create subdirectory with file
  std::string subdir = dir + "/subdir";
  ASSERT_STATUS_OK(fs_->CreateDir(subdir, false));
  {
    std::string path = subdir + "/d.txt";
    std::string content = "d";
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
    auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    ASSERT_STATUS_OK(out->Write(buf));
    ASSERT_STATUS_OK(out->Close());
  }

  // Non-recursive selector
  {
    arrow::fs::FileSelector selector;
    selector.base_dir = dir;
    selector.recursive = false;
    ASSERT_AND_ASSIGN(auto infos, fs_->GetFileInfo(selector));
    // Should have at least 3 files + 1 subdir
    EXPECT_GE(infos.size(), 4u);
  }

  // Recursive selector
  {
    arrow::fs::FileSelector selector;
    selector.base_dir = dir;
    selector.recursive = true;
    ASSERT_AND_ASSIGN(auto infos, fs_->GetFileInfo(selector));
    // Should have 3 files + 1 subdir + 1 file in subdir = at least 5
    EXPECT_GE(infos.size(), 5u);
  }

  (void)fs_->DeleteDir(dir);
}

TEST_F(CloudFsTest, WriteWithHttpHeaders) {
  std::string path = "/test_http_headers.txt";
  (void)fs_->DeleteFile(path);
  std::string content = "http header test";

  auto kvmeta =
      arrow::KeyValueMetadata::Make({"Content-Type", "Cache-Control"}, {"application/octet-stream", "no-cache"});
  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, kvmeta));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(path));
  EXPECT_EQ(info.type(), arrow::fs::FileType::File);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, OpenOutputStreamWithUploadSize) {
  std::string path = "/test_upload_size.txt";
  (void)fs_->DeleteFile(path);

  auto sizable_fs = std::dynamic_pointer_cast<UploadSizable>(fs_);
  ASSERT_NE(sizable_fs, nullptr);

  std::string content = "upload size test content";
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_AND_ASSIGN(auto out, sizable_fs->OpenOutputStreamWithUploadSize(path, nullptr, 5 * 1024 * 1024));
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->ToString(), content);

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, GetMetrics) {
  auto observable_fs = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable_fs, nullptr);
  // Should not crash; may return nullptr for providers that don't implement it yet
  observable_fs->GetMetrics();
}

TEST_F(CloudFsTest, OpenInputFileClosed) {
  std::string path = "/test_closed_input.txt";
  std::string content = "closed test";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  auto buf = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
  ASSERT_STATUS_OK(out->Write(buf));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path));
  ASSERT_STATUS_OK(file->Close());
  ASSERT_TRUE(file->closed());

  // Operations on closed file should fail
  auto read_result = file->Read(10);
  ASSERT_FALSE(read_result.ok());

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, OpenOutputStreamClosed) {
  std::string path = "/test_closed_output.txt";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  ASSERT_STATUS_OK(out->Close());
  ASSERT_TRUE(out->closed());

  // Write to closed stream should fail
  auto write_result = out->Write("data", 4);
  ASSERT_FALSE(write_result.ok());

  (void)fs_->DeleteFile(path);
}

TEST_F(CloudFsTest, WriteEmptyFile) {
  std::string path = "/test_empty_file.txt";
  (void)fs_->DeleteFile(path);

  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, nullptr));
  ASSERT_STATUS_OK(out->Close());

  ASSERT_AND_ASSIGN(auto in, fs_->OpenInputStream(path));
  ASSERT_AND_ASSIGN(auto read_buf, in->Read(1024));
  EXPECT_EQ(read_buf->size(), 0);

  ASSERT_AND_ASSIGN(auto info, fs_->GetFileInfo(path));
  EXPECT_EQ(info.type(), arrow::fs::FileType::File);
  EXPECT_EQ(info.size(), 0);

  (void)fs_->DeleteFile(path);
}

}  // namespace milvus_storage
