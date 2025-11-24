// Copyright 2023 Zilliz
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

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "test_util.h"

namespace milvus_storage {

class S3FsMultiThreadTest : public ::testing::Test {
  protected:
  void SetUp() override {
    std::string cloud_provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("");
    std::string storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    std::string access_key_id = GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("");
    std::string access_key_value = GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("");
    std::string address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
    std::string bucket_name = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    std::string region = GetEnvVar(ENV_VAR_REGION).ValueOr("");

    if (cloud_provider.empty() || storage_type.empty() || address.empty() || bucket_name.empty()) {
      GTEST_SKIP() << "S3 credentials not set. Please set environment variables:\n"
                   << "CLOUD_PROVIDER, STORAGE_TYPE, ACCESS_KEY, SECRET_KEY, ADDRESS, BUCKET_NAME, REGION";
    }

    conf_.cloud_provider = cloud_provider;
    conf_.root_path = "/";
    conf_.storage_type = storage_type;
    conf_.request_timeout_ms = 1000;
    conf_.use_ssl = false;
    conf_.log_level = "debug";
    conf_.region = region;
    conf_.address = address;
    conf_.bucket_name = bucket_name;
    conf_.use_virtual_host = false;
    // not use iam
    if (!access_key_id.empty() && !access_key_value.empty()) {
      conf_.use_iam = false;
      conf_.access_key_id = access_key_id;
      conf_.access_key_value = access_key_value;
    } else {
      conf_.use_iam = true;
      conf_.access_key_id = "";
      conf_.access_key_value = "";
      // azure should provide access key
      if (conf_.cloud_provider == "azure") {
        conf_.access_key_id = access_key_id;
      }
    }
  }

  void TearDown() override {
    // Clean up: Release the singleton after each test
    ArrowFileSystemSingleton::GetInstance().Release();
  }

  ArrowFileSystemConfig conf_;
};

// Test concurrent Init calls
TEST_F(S3FsMultiThreadTest, ConcurrentInit) {
  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  std::atomic<int> exception_count{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, &success_count, &exception_count]() {
      try {
        ArrowFileSystemSingleton::GetInstance().Init(conf_);
        success_count++;
      } catch (const std::exception& e) {
        exception_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All threads should complete without crashes
  ASSERT_EQ(success_count + exception_count, num_threads);

  // Verify singleton is initialized
  auto fs = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
  ASSERT_NE(fs, nullptr);
}

// Test concurrent GetInstance calls
TEST_F(S3FsMultiThreadTest, ConcurrentGetInstance) {
  // First initialize the singleton
  ArrowFileSystemSingleton::GetInstance().Init(conf_);

  constexpr int num_threads = 20;
  std::vector<std::thread> threads;
  std::vector<ArrowFileSystemPtr> fs_pointers(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(
        [this, &fs_pointers, i]() { fs_pointers[i] = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All pointers should be non-null
  for (const auto& fs : fs_pointers) {
    ASSERT_NE(fs, nullptr);
  }

  // All pointers should point to the same instance
  for (size_t i = 1; i < fs_pointers.size(); ++i) {
    ASSERT_EQ(fs_pointers[0].get(), fs_pointers[i].get());
  }
}

// Test concurrent Release calls
TEST_F(S3FsMultiThreadTest, ConcurrentRelease) {
  // First initialize the singleton
  ArrowFileSystemSingleton::GetInstance().Init(conf_);

  // Verify it's initialized
  auto fs_before = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
  ASSERT_NE(fs_before, nullptr);

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([]() { ArrowFileSystemSingleton::GetInstance().Release(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // After release, the filesystem should be null
  auto fs_after = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
  ASSERT_EQ(fs_after, nullptr);
}

}  // namespace milvus_storage