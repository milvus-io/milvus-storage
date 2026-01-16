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
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <arrow/status.h>
#include <arrow/testing/gtest_util.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "test_env.h"

namespace milvus_storage::test {

class S3ClientMetricsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "S3 tests skipped in non-cloud environment";
    }

    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    ASSERT_AND_ASSIGN(arrowfs_, GetFileSystem(properties));

    base_path_ = GetTestBasePath("s3client-metrics");
    ASSERT_STATUS_OK(CreateTestDir(arrowfs_, base_path_));
  }

  void TearDown() override {
    // Clean up test files
    if (IsCloudEnv()) {
      ASSERT_STATUS_OK(DeleteTestDir(arrowfs_, base_path_));
    }
  }

  // Helper method to generate a unique test file name
  std::string GenerateTestFileName() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "test-metrics-" + std::to_string(timestamp) + ".txt";
  }

  protected:
  ArrowFileSystemPtr arrowfs_;
  std::string base_path_;
};

TEST_F(S3ClientMetricsTest, TestMetricsAfterFileOperations) {
  // Get initial metrics
  auto observable = milvus_storage::GetUnderlyingFileSystem<Observable>(arrowfs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 0);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 0);
  EXPECT_EQ(metrics->GetWriteCount(), 0);
  EXPECT_EQ(metrics->GetReadCount(), 0);

  // Generate a unique test file name
  std::string test_file_name = GenerateTestFileName();

  // Create some test data (large enough to trigger multipart upload)
  std::string test_data(11 * 1024 * 1024, 'A');  // 11MB of data

  // Write the file (this should trigger multipart upload for large files)
  // 5MB part size is the minimum part size for multipart upload
  auto fs = milvus_storage::GetUnderlyingFileSystem<UploadSizable>(arrowfs_);
  ASSERT_NE(fs, nullptr);
  auto write_result = fs->OpenOutputStreamWithUploadSize(base_path_ + test_file_name, nullptr, 10 * 1024 * 1024);
  ASSERT_OK_AND_ASSIGN(auto output_stream, write_result);

  auto write_status = output_stream->Write(test_data.data());
  ASSERT_STATUS_OK(write_status);

  auto close_status = output_stream->Close();
  ASSERT_STATUS_OK(close_status);

  // Get metrics after operations
  metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(1, metrics->GetMultiPartUploadCreated());
  EXPECT_EQ(1, metrics->GetMultiPartUploadFinished());
  EXPECT_EQ(2, metrics->GetWriteCount());
  EXPECT_EQ(test_data.size(), metrics->GetWriteBytes());

  // Download the file
  auto download_result = arrowfs_->OpenInputStream(base_path_ + test_file_name);
  ASSERT_OK_AND_ASSIGN(auto input_stream, download_result);
  std::vector<uint8_t> buffer(test_data.size());
  ASSERT_OK_AND_ASSIGN(auto read_size, input_stream->Read(test_data.size(), buffer.data()));

  ASSERT_STATUS_OK(input_stream->Close());
  EXPECT_EQ(test_data, std::string(reinterpret_cast<char*>(buffer.data()), test_data.size()));

  // Get metrics after operations
  metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  EXPECT_EQ(1, metrics->GetReadCount());
  EXPECT_EQ(test_data.size(), metrics->GetReadBytes());
}

}  // namespace milvus_storage::test