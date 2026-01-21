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

#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>

#include <memory>
#include <type_traits>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/filesystem/upload_sizable.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"

#include "test_env.h"

namespace milvus_storage {

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

}  // namespace milvus_storage