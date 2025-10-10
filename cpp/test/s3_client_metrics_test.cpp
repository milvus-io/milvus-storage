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

#include <arrow/status.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>

#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"
#include "test_util.h"
#include "arrow/testing/gtest_util.h"

namespace milvus_storage {

class S3ClientMetricsTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    // Check if environment variables are set
    S3ClientMetricsTest::ACCESS_KEY = GetEnvVar("ACCESS_KEY");
    S3ClientMetricsTest::SECRET_KEY = GetEnvVar("SECRET_KEY");
    S3ClientMetricsTest::ADDRESS = GetEnvVar("ADDRESS");
    S3ClientMetricsTest::BUCKET_NAME = GetEnvVar("BUCKET_NAME");
    S3ClientMetricsTest::REGION = GetEnvVar("REGION");

    if (S3ClientMetricsTest::ACCESS_KEY.empty() || S3ClientMetricsTest::SECRET_KEY.empty() ||
        S3ClientMetricsTest::ADDRESS.empty() || S3ClientMetricsTest::BUCKET_NAME.empty() ||
        S3ClientMetricsTest::REGION.empty()) {
      GTEST_SKIP() << "S3 credentials not set. Please set environment variables:\n"
                   << "ACCESS_KEY, SECRET_KEY, ADDRESS, BUCKET_NAME, REGION";
    }
    // Initialize S3 once for the entire test suite
    ExtendS3GlobalOptions global_options;
    global_options.log_level = arrow::fs::S3LogLevel::Off;  // Disable logging for tests
    auto status = InitializeS3(global_options);
    S3ClientMetricsTest::s3_initialized = status.ok();
    if (!status.ok()) {
      GTEST_SKIP() << "S3 initialization failed: " << status.ToString();
    }
  }

  static void TearDownTestSuite() {
    // Clean up S3 when all tests are done
    auto status = FinalizeS3();
    if (!status.ok()) {
      GTEST_SKIP() << "S3 finalization failed: " << status.ToString();
    }
  }

  void SetUp() override {
    if (!S3ClientMetricsTest::s3_initialized) {
      GTEST_SKIP() << "S3 not initialized";
    }
  }

  void TearDown() override {
    // Clean up any test files
    if (s3fs_) {
      CleanupTestFiles();
    }
    // Don't finalize S3 here - let it stay initialized for subsequent tests
  }

  // Helper method to get environment variable
  static std::string GetEnvVar(const std::string& var_name) {
    const char* value = std::getenv(var_name.c_str());
    return value ? std::string(value) : std::string();
  }

  // Helper method to create a real S3FS instance using environment credentials
  arrow::Result<std::shared_ptr<MultiPartUploadS3FS>> CreateRealS3FS() {
    if (s3fs_) {
      auto metrics_result = s3fs_->GetMetrics();
      if (!metrics_result.ok()) {
        return arrow::Status::Invalid("Failed to get metrics");
      }
      metrics_result.ValueUnsafe()->Reset();
      return s3fs_;  // Return cached instance
    }

    ExtendedS3Options options;
    options.region = S3ClientMetricsTest::REGION;
    options.endpoint_override = S3ClientMetricsTest::ADDRESS;
    options.ConfigureAccessKey(S3ClientMetricsTest::ACCESS_KEY, S3ClientMetricsTest::SECRET_KEY);
    options.scheme = "https";

    return MultiPartUploadS3FS::Make(options);
  }

  // Helper method to generate a unique test file path
  std::string GenerateTestFilePath() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "test-metrics-" + std::to_string(timestamp) + ".txt";
  }

  // Helper method to clean up test files
  void CleanupTestFiles() {
    for (const auto& file_path : test_files_) {
      auto status = s3fs_->DeleteFile(file_path);
      // Ignore errors during cleanup
    }
    test_files_.clear();
  }

  // Helper method to track test files for cleanup
  void TrackTestFile(const std::string& file_path) { test_files_.push_back(file_path); }

  protected:
  static std::string ACCESS_KEY;
  static std::string SECRET_KEY;
  static std::string ADDRESS;
  static std::string BUCKET_NAME;
  static std::string REGION;
  static bool s3_initialized;
  std::shared_ptr<MultiPartUploadS3FS> s3fs_;
  std::vector<std::string> test_files_;
};

// Define static member variables
std::string S3ClientMetricsTest::ACCESS_KEY;
std::string S3ClientMetricsTest::SECRET_KEY;
std::string S3ClientMetricsTest::ADDRESS;
std::string S3ClientMetricsTest::BUCKET_NAME;
std::string S3ClientMetricsTest::REGION;
bool S3ClientMetricsTest::s3_initialized;

TEST_F(S3ClientMetricsTest, TestMetricsAfterFileOperations) {
  ASSERT_OK_AND_ASSIGN(auto s3fs, CreateRealS3FS());

  // Get initial metrics
  ASSERT_OK_AND_ASSIGN(auto metrics, s3fs->GetMetrics());

  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 0);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 0);
  EXPECT_EQ(metrics->GetUploadCount(), 0);
  EXPECT_EQ(metrics->GetDownloadCount(), 0);

  // Generate a unique test file path
  std::string test_file_path = BUCKET_NAME + "/" + GenerateTestFilePath();
  TrackTestFile(test_file_path);

  // Create some test data (large enough to trigger multipart upload)
  std::string test_data(6 * 1024 * 1024, 'A');  // 6MB of data

  // Write the file (this should trigger multipart upload for large files)
  // 5MB part size is the minimum part size for multipart upload
  auto write_result = s3fs->OpenOutputStreamWithUploadSize(test_file_path, nullptr, 5 * 1024 * 1024);
  ASSERT_OK_AND_ASSIGN(auto output_stream, write_result);

  auto write_status = output_stream->Write(test_data.data());
  ASSERT_STATUS_OK(write_status);

  auto close_status = output_stream->Close();
  ASSERT_STATUS_OK(close_status);

  // Get metrics after operations
  ASSERT_OK_AND_ASSIGN(metrics, s3fs->GetMetrics());
  EXPECT_EQ(1, metrics->GetMultiPartUploadCreated());
  EXPECT_EQ(1, metrics->GetMultiPartUploadFinished());
  EXPECT_EQ(2, metrics->GetUploadCount());
  EXPECT_EQ(test_data.size(), metrics->GetUploadBytes());

  // Download the file
  auto download_result = s3fs->OpenInputStream(test_file_path);
  ASSERT_OK_AND_ASSIGN(auto input_stream, download_result);
  std::vector<uint8_t> buffer(test_data.size());
  ASSERT_OK_AND_ASSIGN(auto read_size, input_stream->Read(test_data.size(), buffer.data()));

  ASSERT_STATUS_OK(input_stream->Close());
  EXPECT_EQ(test_data, std::string(reinterpret_cast<char*>(buffer.data()), test_data.size()));

  // Get metrics after operations
  ASSERT_OK_AND_ASSIGN(metrics, s3fs->GetMetrics());

  EXPECT_EQ(1, metrics->GetDownloadCount());
  EXPECT_EQ(test_data.size(), metrics->GetDownloadBytes());
}

}  // namespace milvus_storage