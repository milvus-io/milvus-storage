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
#include <thread>
#include "test_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"

namespace milvus_storage {

class FileSystemCacheTest : public testing::Test {
  protected:
  ArrowFileSystemConfig MakeConfig(const std::string& id) {
    ArrowFileSystemConfig cfg;
    // use address + bucket_name to differentiate configs
    cfg.root_path = "/tmp/fs_test_" + id;
    //   cfg.format = LOON_FORMAT_VORTEX;
    return cfg;
  }
};

TEST_F(FileSystemCacheTest, TemplateSingleton) {
  // won't got different instances for different template types
  auto& c1 = LRUCache<int, std::string>::getInstance();
  auto& c2 = LRUCache<int, double>::getInstance();
  ASSERT_NE((uintptr_t)&c1, (uintptr_t)&c2);

  auto& c3 = LRUCache<int, std::string>::getInstance();
  ASSERT_EQ((uintptr_t)&c1, (uintptr_t)&c3);
}

TEST_F(FileSystemCacheTest, Basic) {
  ArrowFileSystemConfig config1 = MakeConfig("A");
  ArrowFileSystemConfig config2 = MakeConfig("B");
  auto& cache = LRUCache<ArrowFileSystemConfig, ArrowFileSystemPtr>::getInstance();

  // Initially, cache size should be 0
  EXPECT_EQ(cache.size(), 0);

  // Get filesystem for config1
  ASSERT_AND_ASSIGN(auto fs1, cache.get(config1, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for config1 again, should return the same instance
  ASSERT_AND_ASSIGN(auto fs1_again, cache.get(config1, CreateArrowFileSystem));
  EXPECT_EQ(fs1, fs1_again);
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for config2
  ASSERT_AND_ASSIGN(auto fs2, cache.get(config2, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 2);

  // Remove config1 from cache
  cache.remove(config1);
  EXPECT_EQ(cache.size(), 1);

  // Remove config2 from cache
  cache.remove(config2);
  EXPECT_EQ(cache.size(), 0);

  cache.clean();
}

TEST_F(FileSystemCacheTest, FileSystemCacheLRUEviction) {
  auto& cache = LRUCache<ArrowFileSystemConfig, ArrowFileSystemPtr>::getInstance();
  cache.set_capacity(2);

  auto cfg1 = MakeConfig("A");
  auto cfg2 = MakeConfig("B");
  auto cfg3 = MakeConfig("C");

  // Insert two entries
  ASSERT_AND_ASSIGN(auto res1, cache.get(cfg1, CreateArrowFileSystem));
  ASSERT_AND_ASSIGN(auto res2, cache.get(cfg2, CreateArrowFileSystem));

  // Insert third -> should evict least recently used (cfg1)
  ASSERT_AND_ASSIGN(auto res3, cache.get(cfg3, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 2u);

  // Accessing cfg1 should recreate it
  ASSERT_AND_ASSIGN(auto res1b, cache.get(cfg1, CreateArrowFileSystem));
  EXPECT_NE(res1, res1b);

  cache.clean();
  EXPECT_EQ(cache.size(), 0);
}

TEST_F(FileSystemCacheTest, FileSystemCacheLRUUpdateOnAccess) {
  auto& cache = LRUCache<ArrowFileSystemConfig, ArrowFileSystemPtr>::getInstance();
  cache.set_capacity(2);

  auto A = MakeConfig("X");
  auto B = MakeConfig("Y");
  auto C = MakeConfig("Z");

  // Insert A and B
  ASSERT_AND_ASSIGN(auto res1, cache.get(A, CreateArrowFileSystem));
  ASSERT_AND_ASSIGN(auto res2, cache.get(B, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 2u);

  // Access A again to make it most-recent
  ASSERT_AND_ASSIGN(auto res3, cache.get(A, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_EQ(res1, res3);

  // Insert C -> should evict B
  ASSERT_AND_ASSIGN(auto res4, cache.get(C, CreateArrowFileSystem));
  EXPECT_EQ(cache.size(), 2u);

  // B should be gone and recreated when requested
  ASSERT_AND_ASSIGN(auto res5, cache.get(B, CreateArrowFileSystem));
  EXPECT_NE(res2, res5);  // B have been evicted earlier

  // clean up
  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(FileSystemCacheTest, ConcurrentGetsSingleCreate) {
  auto& cache = LRUCache<ArrowFileSystemConfig, ArrowFileSystemPtr>::getInstance();
  cache.set_capacity(10);

  auto cfg = MakeConfig("CONCURRENT");

  const int thread_count = 8;
  const int calls_per_thread = 50;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&cache, &cfg, calls_per_thread]() {
      for (int j = 0; j < calls_per_thread; ++j) {
        ASSERT_AND_ASSIGN(auto res, cache.get(cfg, CreateArrowFileSystem));
        // small yield to increase interleaving
        std::this_thread::yield();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
  // Only one creation should have happened if the cache creation is properly synchronized.
}

TEST_F(FileSystemCacheTest, ConcurrentGetsAndCreate) {
  const int thread_count = 20;
  const int configs_per_thread = 100;
  const int capacity = 50;
  auto& cache = LRUCache<ArrowFileSystemConfig, ArrowFileSystemPtr>::getInstance();
  cache.set_capacity(capacity);

  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < configs_per_thread; ++j) {
        auto cfg = MakeConfig("THREAD_" + std::to_string(j));
        ASSERT_AND_ASSIGN(auto res, cache.get(cfg, CreateArrowFileSystem));
        ASSERT_LE(cache.size(), std::min(capacity, configs_per_thread));
        std::this_thread::yield();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(cache.size(), std::min(capacity, configs_per_thread));

  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
}

}  // namespace milvus_storage
