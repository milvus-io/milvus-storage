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
#include <thread>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "test_env.h"

namespace milvus_storage {

class FileSystemCacheTest : public testing::Test {
  protected:
  void SetUp() override {
    // clean cache before each test
    // other tests may have polluted the cache
    FilesystemCache::getInstance().clean();
  }

  void TearDown() override { FilesystemCache::getInstance().clean(); }

  // Helper to create properties for an external filesystem with given ID
  milvus_storage::api::Properties MakeProperties(const std::string& id) {
    milvus_storage::api::Properties props;
    // Use extfs.* prefix to create external filesystem configurations
    // Each ID gets a unique address and bucket, ensuring unique cache keys
    props["extfs." + id + ".root_path"] = static_cast<std::string>("/tmp/fs_test_" + id);
    props["extfs." + id + ".storage_type"] = std::string("local");
    props["extfs." + id + ".bucket_name"] = static_cast<std::string>("bucket_" + id);
    props["extfs." + id + ".address"] = static_cast<std::string>("localhost_" + id);
    return props;
  }
};

TEST_F(FileSystemCacheTest, LRUCacheInstantiation) {
  // LRUCache is now a pure library class that can be instantiated
  LRUCache<int, std::string> c1;
  LRUCache<int, double> c2;
  LRUCache<int, std::string> c3(32);  // Custom capacity

  // Each instance is independent
  ASSERT_NE((uintptr_t)&c1, (uintptr_t)&c2);
  ASSERT_NE((uintptr_t)&c1, (uintptr_t)&c3);

  // Test that they work independently
  c1.set_capacity(10);
  c3.set_capacity(20);

  // Test put and get
  c1.put(1, "value_1");
  auto s1 = c1.get(1);
  EXPECT_TRUE(s1.has_value());
  EXPECT_EQ(s1.value(), "value_1");

  c2.put(10, 15.0);
  auto d1 = c2.get(10);
  EXPECT_TRUE(d1.has_value());
  EXPECT_EQ(d1.value(), 15.0);

  // Test get on non-existent key returns std::nullopt
  auto s2 = c1.get(999);
  EXPECT_FALSE(s2.has_value());
}

TEST_F(FileSystemCacheTest, Basic) {
  auto props1 = MakeProperties("A");
  auto props2 = MakeProperties("B");
  auto& cache = FilesystemCache::getInstance();

  // Initially, cache size should be 0
  EXPECT_EQ(cache.size(), 0);

  // Get filesystem for props1 with matching path
  ASSERT_AND_ASSIGN(auto fs1, cache.get(props1, "s3://localhost_A/bucket_A/file.parquet"));
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for props1 again, should return the same instance
  ASSERT_AND_ASSIGN(auto fs1_again, cache.get(props1, "s3://localhost_A/bucket_A/file2.parquet"));
  EXPECT_EQ(fs1, fs1_again);
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for props2 with matching path
  ASSERT_AND_ASSIGN(auto fs2, cache.get(props2, "s3://localhost_B/bucket_B/file.parquet"));
  EXPECT_EQ(cache.size(), 2);

  // Cache entries persist across different property instances with same values
  EXPECT_EQ(cache.size(), 2);

  cache.clean();
  EXPECT_EQ(cache.size(), 0);
}

TEST_F(FileSystemCacheTest, CacheKeyIncludesCredentialIdentity) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "s3.amazonaws.com";
  base.bucket_name = "shared-bucket";
  base.role_arn = "arn:aws:iam::111111111111:role/source";
  base.session_name = "source-session";
  base.external_id = "source-external";

  auto different_role = base;
  different_role.role_arn = "arn:aws:iam::222222222222:role/target";
  EXPECT_NE(base.GetCacheKey(), different_role.GetCacheKey());

  auto different_external_id = base;
  different_external_id.external_id = "target-external";
  EXPECT_NE(base.GetCacheKey(), different_external_id.GetCacheKey());

  ArrowFileSystemConfig static_credentials = base;
  static_credentials.role_arn.clear();
  static_credentials.session_name.clear();
  static_credentials.external_id.clear();
  static_credentials.access_key_id = "ak-a";
  static_credentials.access_key_value = "sk-a";

  auto different_secret = static_credentials;
  different_secret.access_key_value = "sk-b";
  EXPECT_NE(static_credentials.GetCacheKey(), different_secret.GetCacheKey());

  EXPECT_EQ(static_credentials.GetCacheKey(), static_credentials.GetCacheKey());
}

TEST_F(FileSystemCacheTest, CacheKeyIgnoresUnusedCredentialFields) {
  ArrowFileSystemConfig assumed_role;
  assumed_role.storage_type = "remote";
  assumed_role.cloud_provider = kCloudProviderAWS;
  assumed_role.address = "s3.amazonaws.com";
  assumed_role.bucket_name = "shared-bucket";
  assumed_role.role_arn = "arn:aws:iam::111111111111:role/source";
  assumed_role.session_name = "source-session";
  assumed_role.external_id = "source-external";
  assumed_role.access_key_id = "ak-a";
  assumed_role.access_key_value = "sk-a";

  auto different_static_credentials = assumed_role;
  different_static_credentials.access_key_id = "ak-b";
  different_static_credentials.access_key_value = "sk-b";
  different_static_credentials.use_iam = true;
  EXPECT_EQ(assumed_role.GetCacheKey(), different_static_credentials.GetCacheKey());

  ArrowFileSystemConfig gcp_impersonation;
  gcp_impersonation.storage_type = "remote";
  gcp_impersonation.cloud_provider = kCloudProviderGCP;
  gcp_impersonation.address = "storage.googleapis.com";
  gcp_impersonation.bucket_name = "shared-bucket";
  gcp_impersonation.use_iam = true;
  gcp_impersonation.gcp_target_service_account = "source@project.iam.gserviceaccount.com";
  gcp_impersonation.access_key_id = "ak-a";
  gcp_impersonation.access_key_value = "sk-a";
  gcp_impersonation.load_frequency = 1800;

  auto different_static_credentials_for_gcp = gcp_impersonation;
  different_static_credentials_for_gcp.access_key_id = "ak-b";
  different_static_credentials_for_gcp.access_key_value = "sk-b";
  EXPECT_EQ(gcp_impersonation.GetCacheKey(), different_static_credentials_for_gcp.GetCacheKey());

  auto different_role_arn_for_gcp = gcp_impersonation;
  different_role_arn_for_gcp.role_arn = "arn:aws:iam::111111111111:role/unused";
  different_role_arn_for_gcp.session_name = "unused-session";
  different_role_arn_for_gcp.external_id = "unused-external";
  EXPECT_EQ(gcp_impersonation.GetCacheKey(), different_role_arn_for_gcp.GetCacheKey());
}

TEST_F(FileSystemCacheTest, GcpCacheKeyUsesImpersonationIdentityOnly) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderGCP;
  base.address = "storage.googleapis.com";
  base.bucket_name = "shared-bucket";
  base.use_iam = true;
  base.gcp_target_service_account = "source@project.iam.gserviceaccount.com";
  base.load_frequency = 1800;

  auto different_target_sa = base;
  different_target_sa.gcp_target_service_account = "target@project.iam.gserviceaccount.com";
  EXPECT_NE(base.GetCacheKey(), different_target_sa.GetCacheKey());
}

TEST_F(FileSystemCacheTest, CacheKeyIgnoresAlias) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "s3.amazonaws.com";
  base.bucket_name = "shared-bucket";
  base.access_key_id = "ak";
  base.access_key_value = "sk";

  auto with_alias = base;
  with_alias.alias = "external-a";
  EXPECT_EQ(base.GetCacheKey(), with_alias.GetCacheKey());
}

TEST_F(FileSystemCacheTest, FilesystemCacheSeparatesSameBucketWithDifferentCredentials) {
  auto make_props = [](const std::string& access_key, const std::string& secret_key) {
    api::Properties props;
    props[PROPERTY_FS_ADDRESS] = std::string("s3.amazonaws.com");
    props[PROPERTY_FS_BUCKET_NAME] = std::string("shared-bucket");
    props[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
    props[PROPERTY_FS_CLOUD_PROVIDER] = std::string(kCloudProviderAWS);
    props[PROPERTY_FS_ACCESS_KEY_ID] = access_key;
    props[PROPERTY_FS_ACCESS_KEY_VALUE] = secret_key;
    return props;
  };

  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(10);

  auto props_a = make_props("ak", "sk-a");
  auto props_b = make_props("ak", "sk-b");

  ASSERT_AND_ASSIGN(auto fs_a, cache.get(props_a));
  ASSERT_AND_ASSIGN(auto fs_a_again, cache.get(props_a));
  ASSERT_AND_ASSIGN(auto fs_b, cache.get(props_b));

  EXPECT_EQ(fs_a, fs_a_again);
  EXPECT_NE(fs_a, fs_b);
  EXPECT_EQ(cache.size(), 2u);
}

TEST_F(FileSystemCacheTest, FileSystemCacheLRUEviction) {
  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(2);

  auto props1 = MakeProperties("A");
  auto props2 = MakeProperties("B");
  auto props3 = MakeProperties("C");

  // Insert two entries
  ASSERT_AND_ASSIGN(auto res1, cache.get(props1, "s3://localhost_A/bucket_A/file.parquet"));
  ASSERT_AND_ASSIGN(auto res2, cache.get(props2, "s3://localhost_B/bucket_B/file.parquet"));

  // Insert third -> should evict least recently used (props1)
  ASSERT_AND_ASSIGN(auto res3, cache.get(props3, "s3://localhost_C/bucket_C/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // Accessing props1 should recreate it
  ASSERT_AND_ASSIGN(auto res1b, cache.get(props1, "s3://localhost_A/bucket_A/file.parquet"));
  EXPECT_NE(res1, res1b);

  cache.clean();
  EXPECT_EQ(cache.size(), 0);
}

TEST_F(FileSystemCacheTest, FileSystemCacheLRUUpdateOnAccess) {
  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(2);

  auto A = MakeProperties("X");
  auto B = MakeProperties("Y");
  auto C = MakeProperties("Z");

  // Insert A and B
  ASSERT_AND_ASSIGN(auto res1, cache.get(A, "s3://localhost_X/bucket_X/file.parquet"));
  ASSERT_AND_ASSIGN(auto res2, cache.get(B, "s3://localhost_Y/bucket_Y/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // Access A again to make it most-recent
  ASSERT_AND_ASSIGN(auto res3, cache.get(A, "s3://localhost_X/bucket_X/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_EQ(res1, res3);

  // Insert C -> should evict B
  ASSERT_AND_ASSIGN(auto res4, cache.get(C, "s3://localhost_Z/bucket_Z/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // B should be gone and recreated when requested
  ASSERT_AND_ASSIGN(auto res5, cache.get(B, "s3://localhost_Y/bucket_Y/file.parquet"));
  EXPECT_NE(res2, res5);  // B have been evicted earlier

  // clean up
  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(FileSystemCacheTest, LRUCacheRemove) {
  LRUCache<int, std::string> cache(10);

  cache.put(1, "a");
  cache.put(2, "b");
  cache.put(3, "c");
  EXPECT_EQ(cache.size(), 3u);

  // Remove an existing key
  cache.remove(2);
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_FALSE(cache.get(2).has_value());

  // Remaining entries are still accessible
  EXPECT_EQ(cache.get(1).value(), "a");
  EXPECT_EQ(cache.get(3).value(), "c");

  // Remove a non-existent key — should be a no-op
  cache.remove(999);
  EXPECT_EQ(cache.size(), 2u);

  // Remove all remaining entries one by one
  cache.remove(1);
  cache.remove(3);
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(FileSystemCacheTest, LRUCacheSetCapacityEvicts) {
  LRUCache<int, std::string> cache(5);

  cache.put(1, "a");
  cache.put(2, "b");
  cache.put(3, "c");
  cache.put(4, "d");
  cache.put(5, "e");
  EXPECT_EQ(cache.size(), 5u);

  // Shrink capacity below current size — should evict LRU entries (oldest first)
  // LRU order (front→back): 5, 4, 3, 2, 1. Evicting from back removes 1, then 2.
  cache.set_capacity(3);
  EXPECT_EQ(cache.size(), 3u);

  // Oldest entries (1, 2) should be evicted
  EXPECT_FALSE(cache.get(1).has_value());
  EXPECT_FALSE(cache.get(2).has_value());

  // Newest entries (3, 4, 5) should survive
  EXPECT_EQ(cache.get(3).value(), "c");
  EXPECT_EQ(cache.get(4).value(), "d");
  EXPECT_EQ(cache.get(5).value(), "e");

  // Shrink to 0 — evicts everything
  cache.set_capacity(0);
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(FileSystemCacheTest, ConcurrentGetsSingleCreate) {
  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(10);

  auto props = MakeProperties("CONCURRENT");
  std::string path = "s3://localhost_CONCURRENT/bucket_CONCURRENT/file.parquet";

  const int thread_count = 8;
  const int calls_per_thread = 50;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  std::atomic<bool> failed{false};
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&cache, &props, &path, calls_per_thread, &failed]() {
      for (int j = 0; j < calls_per_thread; ++j) {
        auto result = cache.get(props, path);
        if (!result.ok()) {
          failed.store(true);
          return;
        }
        // small yield to increase interleaving
        std::this_thread::yield();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  EXPECT_FALSE(failed.load()) << "cache.get() failed inside a thread";

  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(FileSystemCacheTest, ConcurrentGetsAndCreate) {
  const int thread_count = 20;
  const int props_per_thread = 100;
  const int capacity = 50;
  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(capacity);

  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  std::atomic<bool> failed{false};
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < props_per_thread; ++j) {
        auto id = "THREAD_" + std::to_string(j);
        auto props = MakeProperties(id);
        std::string path = "s3://localhost_" + id + "/bucket_" + id + "/file.parquet";
        auto result = cache.get(props, path);
        if (!result.ok()) {
          failed.store(true);
          return;
        }
        EXPECT_LE(cache.size(), static_cast<size_t>(std::min(capacity, props_per_thread)));
        std::this_thread::yield();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  EXPECT_FALSE(failed.load()) << "cache.get() failed inside a thread";

  EXPECT_EQ(cache.size(), static_cast<size_t>(std::min(capacity, props_per_thread)));

  cache.clean();
  EXPECT_EQ(cache.size(), 0u);
}

}  // namespace milvus_storage
