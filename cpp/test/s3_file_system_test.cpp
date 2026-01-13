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

#include <memory>
#include <type_traits>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/filesystem_extend.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"

#include "test_env.h"

namespace milvus_storage {
class LocalFsTest : public ::testing::Test {};

class S3FsTest : public ::testing::Test {
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

TEST_F(S3FsTest, ConditionalWrite) {
  std::string file_to = "/test_conditional_write.txt";

  // Ensure source file does not exist
  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  // Create source file
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());

    ASSERT_AND_ASSIGN(auto output_stream, open_condition_write_output_stream(fs_, file_to));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }

  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    ASSERT_AND_ASSIGN(auto output_stream, open_condition_write_output_stream(fs_, file_to));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    auto write_status = output_stream->Close();

    ASSERT_FALSE(write_status.ok());
    auto extend_status = ExtendStatusDetail::UnwrapStatus(write_status);
    ASSERT_NE(extend_status, nullptr);
    ASSERT_TRUE(extend_status->code() == ExtendStatusCode::AwsErrorPreConditionFailed ||
                extend_status->code() == ExtendStatusCode::AwsErrorConflict);
  }

  (void)fs_->DeleteFile(file_to);

  // Test conditional write in output_stream close
  {
    std::shared_ptr<arrow::Buffer> buffer1 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());
    std::shared_ptr<arrow::Buffer> buffer2 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    ASSERT_AND_ASSIGN(auto output_stream1, open_condition_write_output_stream(fs_, file_to));
    ASSERT_STATUS_OK(output_stream1->Write(buffer1));

    ASSERT_AND_ASSIGN(auto output_stream2, open_condition_write_output_stream(fs_, file_to));
    ASSERT_STATUS_OK(output_stream2->Write(buffer2));

    ASSERT_STATUS_OK(output_stream1->Close());
    auto write_status = output_stream2->Close();
    ASSERT_FALSE(write_status.ok());
    auto extend_status = ExtendStatusDetail::UnwrapStatus(write_status);
    ASSERT_NE(extend_status, nullptr);
    ASSERT_TRUE(extend_status->code() == ExtendStatusCode::AwsErrorPreConditionFailed ||
                extend_status->code() == ExtendStatusCode::AwsErrorConflict);
  }
}

TEST_F(S3FsTest, TestMetadata) {
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

TEST_F(S3FsTest, TestExtendErrorInFs) {
  Aws::Client::AWSError<Aws::S3::S3Errors> test_err(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                    Aws::Client::RetryableType::NOT_RETRYABLE, "AwsErrorNoSuchUpload",
                                                    "Just for test");

  auto status = fs::internal::ErrorToStatus("test", test_err);
  ASSERT_STATUS_NOT_OK(status);
  auto extend_status = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(extend_status, nullptr);
  ASSERT_EQ(extend_status->code(), ExtendStatusCode::AwsErrorNoSuchUpload);
  ASSERT_TRUE(status.ToString().find(extend_status->ToString()) != std::string::npos);
}

// Regression case revealed that:
// root-path always generated relative paths, even those starting with "/".
TEST_F(LocalFsTest, TestRootPath) {
  auto boost_rmdir = [](const std::string& path) {
    boost::filesystem::path dir_path(path);
    if (boost::filesystem::exists(dir_path)) {
      boost::filesystem::remove_all(dir_path);
    }
  };

  auto boost_create_dir = [](const std::string& path) {
    boost::filesystem::path dir_path(path);
    if (!boost::filesystem::exists(dir_path)) {
      boost::filesystem::create_directories(dir_path);
    }
  };

  std::string abs_path = "/tmp/test-localfs/";
  std::string rel_path = "./test-localfs/";

  std::string abs_exist_path = "/tmp/test-exist-localfs/";
  std::string rel_exist_path = "./test-exist-localfs/";

  boost_rmdir(abs_path);
  boost_rmdir(rel_path);

  boost_rmdir(abs_exist_path);
  boost_rmdir(rel_exist_path);
  boost_create_dir(abs_exist_path);
  boost_create_dir(rel_exist_path);

  std::vector<std::string> paths = {
      abs_path,
      rel_path,
      abs_exist_path,
      rel_exist_path,
  };

  for (const auto& root_path : paths) {
    ArrowFileSystemConfig config;
    config.storage_type = "local";
    config.root_path = root_path;
    std::string write_content = "This is a test file.";

    ASSERT_AND_ASSIGN(auto local_fs, CreateArrowFileSystem(config));
    ASSERT_AND_ASSIGN(auto output_stream, local_fs->OpenOutputStream("test.txt"));
    ASSERT_STATUS_OK(output_stream->Write(write_content.c_str(), write_content.size()));
    ASSERT_STATUS_OK(output_stream->Close());

    ASSERT_TRUE(boost::filesystem::exists(root_path));
    ASSERT_TRUE(boost::filesystem::exists(root_path + "/test.txt"));

    ASSERT_AND_ASSIGN(auto input_stream, local_fs->OpenInputStream("test.txt"));
    ASSERT_AND_ASSIGN(auto read_buffer, input_stream->Read(write_content.size()));
    auto read_content = read_buffer->ToString();
    ASSERT_STATUS_OK(input_stream->Close());
    ASSERT_EQ(write_content, read_content);
  }
}

}  // namespace milvus_storage