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

#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <numeric>
#include "arrow/io/memory.h"
#include "arrow/status.h"
#include "arrow/result.h"
#include "arrow/buffer.h"
#include "test_util.h"
#include <cstdlib>
#include "milvus-storage/filesystem/fs.h"
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/operations.hpp>

#define ASSERT_OK(expr) ASSERT_TRUE((expr).ok())

namespace milvus_storage {

// Environment variables to configure the S3 test environment
static const char* kEnvAccessKey = "ACCESS_KEY";
static const char* kEnvSecretKey = "SECRET_KEY";
static const char* kEnvAddress = "ADDRESS";
static const char* kEnvCloudProvider = "CLOUD_PROVIDER";
static const char* kEnvBucketName = "BUCKET_NAME";
static const char* kEnvRegion = "REGION";

class S3ConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* access_key = std::getenv(kEnvAccessKey);
        const char* secret_key = std::getenv(kEnvSecretKey);
        const char* address = std::getenv(kEnvAddress);
        const char* cloud_provider = std::getenv(kEnvCloudProvider);
        const char* bucket_name = std::getenv(kEnvBucketName);
        const char* region = std::getenv(kEnvRegion);

        path_ = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
        auto conf = ArrowFileSystemConfig();
        conf.storage_type = "local";
        conf.root_path = path_.string();

        if (cloud_provider != nullptr) {
            conf.cloud_provider = std::string(cloud_provider);
            conf.root_path = boost::filesystem::unique_path().string();
            path_ = conf.root_path;
            conf.use_custom_part_upload = true;
            conf.storage_type = "remote";
            conf.requestTimeoutMs = 10000;
            conf.useSSL = false;
            conf.log_level = "debug";
            conf.region = std::string(region);
            conf.address = std::string(address);
            conf.bucket_name = std::string(bucket_name);
            conf.useVirtualHost = false;
            
            // not use iam
            if (access_key != nullptr && secret_key != nullptr) {
                conf.useIAM = false;
                conf.access_key_id = std::string(access_key);
                conf.access_key_value = std::string(secret_key);
            } else {
                conf.useIAM = true;
                conf.access_key_id = "";
                conf.access_key_value = "";
            }
        }

        ArrowFileSystemSingleton::GetInstance().Init(conf);
        fs_ = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
    }

    void TearDown() override {
        boost::filesystem::remove_all(path_);
        ArrowFileSystemSingleton::GetInstance().Release();
    }

    ArrowFileSystemPtr fs_;
    boost::filesystem::path path_;
};

TEST_F(S3ConcurrencyTest, TestMaxConnections) {
    const int num_threads = 50; // More than max connections (25)
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::vector<std::chrono::steady_clock::time_point> start_times(num_threads);
    std::vector<std::chrono::steady_clock::time_point> end_times(num_threads);

    // Function to write a file
    auto write_file = [&](int file_id) {
        std::string bucket_name = std::getenv(kEnvBucketName) ? std::getenv(kEnvBucketName) : "oss-test-01";
        std::string path = bucket_name + "/concurrent-test-file-" + std::to_string(file_id);
        
        // Wait for all threads to be ready
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return ready; });
        }

        // Record start time
        start_times[file_id] = std::chrono::steady_clock::now();

        // Write file with some content
        auto out_result = fs_->OpenOutputStream(path, nullptr);
        ASSERT_TRUE(out_result.ok()) << "Failed to create output stream: " << out_result.status().ToString();
        auto out_stream = std::move(out_result).ValueOrDie();
        
        // Write a larger content to make the operation take some time
        std::string content(1024 * 1024, 'x'); // 1MB of data
        ASSERT_OK(out_stream->Write(content.data(), content.size()));
        ASSERT_OK(out_stream->Close());

        // Record end time
        end_times[file_id] = std::chrono::steady_clock::now();
    };

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(write_file, i);
    }

    // Start all threads simultaneously
    {
        std::unique_lock<std::mutex> lock(mtx);
        ready = true;
        cv.notify_all();
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Analyze the timing of operations
    auto first_start = *std::min_element(start_times.begin(), start_times.end());
    auto last_end = *std::max_element(end_times.begin(), end_times.end());
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(last_end - first_start);

    // Calculate average duration per operation
    std::vector<std::chrono::milliseconds> durations;
    for (int i = 0; i < num_threads; i++) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_times[i] - start_times[i]);
        durations.push_back(duration);
    }
    auto avg_duration = std::accumulate(durations.begin(), durations.end(), std::chrono::milliseconds(0)) / num_threads;

    // If operations are truly concurrent, total duration should be less than
    // num_threads * avg_duration (which would be the case if operations were sequential)
    EXPECT_LT(total_duration.count(), (num_threads * avg_duration.count()) / 2)
        << "Total duration: " << total_duration.count() << "ms, "
        << "Average duration: " << avg_duration.count() << "ms, "
        << "Expected sequential duration: " << (num_threads * avg_duration.count()) << "ms";
}

} // namespace milvus_storage 