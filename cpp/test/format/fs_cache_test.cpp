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
#include <string>
#include <thread>
#include <vector>

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

TEST_F(FileSystemCacheTest, LRUCacheList) {
  LRUCache<int, std::string> cache(2);
  cache.put(1, "a");
  cache.put(2, "b");

  auto entries = cache.list();
  ASSERT_EQ(entries.size(), 2u);

  bool found_a = false;
  bool found_b = false;
  for (const auto& [key, value] : entries) {
    found_a = found_a || (key == 1 && value == "a");
    found_b = found_b || (key == 2 && value == "b");
  }
  EXPECT_TRUE(found_a);
  EXPECT_TRUE(found_b);

  // Listing must not update recency. Entry 1 remains the least recently used.
  cache.put(3, "c");
  EXPECT_FALSE(cache.get(1).has_value());
  EXPECT_EQ(cache.get(2).value(), "b");
  EXPECT_EQ(cache.get(3).value(), "c");
}

TEST_F(FileSystemCacheTest, Basic) {
  auto props1 = MakeProperties("A");
  auto props2 = MakeProperties("B");
  auto& cache = FilesystemCache::getInstance();

  // Initially, cache size should be 0
  EXPECT_EQ(cache.size(), 0);

  // Get filesystem for props1 with matching path
  ASSERT_AND_ASSIGN(
      auto fs1, cache.get(std::make_shared<const api::Properties>(props1), "s3://localhost_A/bucket_A/file.parquet"));
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for props1 again, should return the same instance
  ASSERT_AND_ASSIGN(auto fs1_again, cache.get(std::make_shared<const api::Properties>(props1),
                                              "s3://localhost_A/bucket_A/file2.parquet"));
  EXPECT_EQ(fs1, fs1_again);
  EXPECT_EQ(cache.size(), 1);

  // Get filesystem for props2 with matching path
  ASSERT_AND_ASSIGN(
      auto fs2, cache.get(std::make_shared<const api::Properties>(props2), "s3://localhost_B/bucket_B/file.parquet"));
  EXPECT_EQ(cache.size(), 2);

  // Cache entries persist across different property instances with same values
  EXPECT_EQ(cache.size(), 2);

  cache.clean();
  EXPECT_EQ(cache.size(), 0);
}

TEST_F(FileSystemCacheTest, List) {
  auto props1 = MakeProperties("LIST_A");
  auto props2 = MakeProperties("LIST_B");
  auto& cache = FilesystemCache::getInstance();

  ASSERT_AND_ASSIGN(auto config1, cache.resolve_config(std::make_shared<const api::Properties>(props1),
                                                       "s3://localhost_LIST_A/bucket_LIST_A/file.parquet"));
  ASSERT_AND_ASSIGN(auto config2, cache.resolve_config(std::make_shared<const api::Properties>(props2),
                                                       "s3://localhost_LIST_B/bucket_LIST_B/file.parquet"));

  ASSERT_AND_ASSIGN(auto fs1, cache.get(std::make_shared<const api::Properties>(props1),
                                        "s3://localhost_LIST_A/bucket_LIST_A/file.parquet"));
  ASSERT_AND_ASSIGN(auto fs2, cache.get(std::make_shared<const api::Properties>(props2),
                                        "s3://localhost_LIST_B/bucket_LIST_B/file.parquet"));

  auto entries = cache.list();
  ASSERT_EQ(entries.size(), 2u);

  bool found_fs1 = false;
  bool found_fs2 = false;
  for (const auto& [display_key, fs] : entries) {
    if (fs == fs1) {
      EXPECT_EQ(display_key, "file:///tmp/fs_test_LIST_A#" + config1.GetCacheKey());
      found_fs1 = true;
    }
    if (fs == fs2) {
      EXPECT_EQ(display_key, "file:///tmp/fs_test_LIST_B#" + config2.GetCacheKey());
      found_fs2 = true;
    }
  }
  EXPECT_TRUE(found_fs1);
  EXPECT_TRUE(found_fs2);

  cache.remove(config1.GetCacheKey());
  cache.remove(config2.GetCacheKey());
  EXPECT_TRUE(cache.list().empty());
}

TEST_F(FileSystemCacheTest, ListReturnsRemoteDisplayKey) {
  api::Properties props;
  props[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  props[PROPERTY_FS_CLOUD_PROVIDER] = std::string(kCloudProviderAWS);
  props[PROPERTY_FS_ADDRESS] = std::string("minio.example.com:9000");
  props[PROPERTY_FS_BUCKET_NAME] = std::string("display-bucket");
  props[PROPERTY_FS_ACCESS_KEY_ID] = std::string("ak");
  props[PROPERTY_FS_ACCESS_KEY_VALUE] = std::string("sk");

  auto& cache = FilesystemCache::getInstance();
  ASSERT_AND_ASSIGN(auto config, cache.resolve_config(std::make_shared<const api::Properties>(props)));
  ASSERT_AND_ASSIGN(auto fs, cache.get(std::make_shared<const api::Properties>(props)));

  auto entries = cache.list();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].first, "minio.example.com:9000/display-bucket#" + config.GetCacheKey());
  EXPECT_EQ(entries[0].second, fs);
}

TEST_F(FileSystemCacheTest, ListReturnsTalonDisplayKey) {
  api::Properties props;
  props[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  props[PROPERTY_FS_CLOUD_PROVIDER] = std::string(kCloudProviderAWS);
  props[PROPERTY_FS_ADDRESS] = std::string("minio.example.com:9000");
  props[PROPERTY_FS_BUCKET_NAME] = std::string("display-bucket");
  props[PROPERTY_FS_ACCESS_KEY_ID] = std::string("ak");
  props[PROPERTY_FS_ACCESS_KEY_VALUE] = std::string("sk");
  props[PROPERTY_FS_TALON_MODE] = static_cast<uint32_t>(TalonMode::Full);
  props[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:7000");
  props[PROPERTY_FS_TALON_BLOCK_SIZE] = uint32_t{8U * 1024U * 1024U};

  auto& cache = FilesystemCache::getInstance();
  ASSERT_AND_ASSIGN(auto config, cache.resolve_config(std::make_shared<const api::Properties>(props)));
  ASSERT_AND_ASSIGN(auto fs, cache.get(std::make_shared<const api::Properties>(props)));

  auto entries = cache.list();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].first, "minio.example.com:9000/display-bucket?talon=127.0.0.1:7000#" + config.GetCacheKey());
  EXPECT_EQ(entries[0].second, fs);
}

TEST_F(FileSystemCacheTest, ListUsesNullForEmptyRemoteAddress) {
  api::Properties props;
  props[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  props[PROPERTY_FS_CLOUD_PROVIDER] = std::string(kCloudProviderAWS);
  props[PROPERTY_FS_ADDRESS] = std::string("");
  props[PROPERTY_FS_BUCKET_NAME] = std::string("display-bucket");
  props[PROPERTY_FS_ACCESS_KEY_ID] = std::string("ak");
  props[PROPERTY_FS_ACCESS_KEY_VALUE] = std::string("sk");

  auto& cache = FilesystemCache::getInstance();
  ASSERT_AND_ASSIGN(auto config, cache.resolve_config(std::make_shared<const api::Properties>(props)));
  ASSERT_TRUE(config.address.empty());
  ASSERT_AND_ASSIGN(auto fs, cache.get(std::make_shared<const api::Properties>(props)));

  auto entries = cache.list();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].first, "<null>/display-bucket#" + config.GetCacheKey());
  EXPECT_EQ(entries[0].second, fs);
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

  auto different_session = base;
  different_session.session_name = "target-session";
  EXPECT_NE(base.GetCacheKey(), different_session.GetCacheKey());

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

TEST_F(FileSystemCacheTest, S3CompatibleIamAndStaticCredentialsHaveSeparateCacheKeys) {
  for (const auto* provider : {kCloudProviderAWS, kCloudProviderAliyun, kCloudProviderTencent, kCloudProviderHuawei,
                               kCloudProviderVolcengine}) {
    SCOPED_TRACE(provider);
    ArrowFileSystemConfig iam;
    iam.storage_type = "remote";
    iam.cloud_provider = provider;
    iam.bucket_name = "bucket";
    iam.use_iam = true;
    auto unused_keys = iam;
    unused_keys.access_key_id = "unused-ak";
    unused_keys.access_key_value = "unused-sk";
    EXPECT_EQ(iam.GetCacheKey(), unused_keys.GetCacheKey());

    auto static_keys = unused_keys;
    static_keys.use_iam = false;
    EXPECT_NE(iam.GetCacheKey(), static_keys.GetCacheKey());
    auto different_ak = static_keys;
    different_ak.access_key_id = "other-ak";
    EXPECT_NE(static_keys.GetCacheKey(), different_ak.GetCacheKey());
    auto different_sk = static_keys;
    different_sk.access_key_value = "other-sk";
    EXPECT_NE(static_keys.GetCacheKey(), different_sk.GetCacheKey());
  }
}

TEST_F(FileSystemCacheTest, GcpHmacDefaultIamAndImpersonationHaveSeparateCacheKeys) {
  ArrowFileSystemConfig hmac;
  hmac.storage_type = "remote";
  hmac.cloud_provider = kCloudProviderGCP;
  hmac.bucket_name = "bucket";
  hmac.access_key_id = "hmac-id";
  hmac.access_key_value = "hmac-key";
  auto different_key = hmac;
  different_key.access_key_value = "other-key";
  EXPECT_NE(hmac.GetCacheKey(), different_key.GetCacheKey());
  auto different_id = hmac;
  different_id.access_key_id = "other-id";
  EXPECT_NE(hmac.GetCacheKey(), different_id.GetCacheKey());
  auto iam = hmac;
  iam.use_iam = true;
  EXPECT_NE(hmac.GetCacheKey(), iam.GetCacheKey());
  auto impersonation = iam;
  impersonation.gcp_target_service_account = "reader@example.iam.gserviceaccount.com";
  EXPECT_NE(iam.GetCacheKey(), impersonation.GetCacheKey());
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
  // Talon Broker IAM ignores the target SA, but native GCP fallback still uses it.
  base.talon_mode = different_target_sa.talon_mode = TalonMode::Full;
  base.credential_endpoint = different_target_sa.credential_endpoint = "https://broker/credentials";
  EXPECT_NE(base.GetCacheKey(), different_target_sa.GetCacheKey());
}

TEST_F(FileSystemCacheTest, AwsBrokerCacheKeyUsesBrokerIdentity) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;
  config.region = "us-west-2";
  config.role_arn = "arn:aws:iam::123456789012:role/reader";
  config.credential_endpoint = "http://broker/";
  auto changed = config;
  changed.credential_endpoint = "http://other-broker/";
  EXPECT_NE(config.GetCacheKey(), changed.GetCacheKey());
  changed = config;
  changed.external_id = "another-tenant";
  EXPECT_NE(config.GetCacheKey(), changed.GetCacheKey());
  changed = config;
  changed.role_arn = "arn:aws:iam::123456789012:role/other";
  EXPECT_NE(config.GetCacheKey(), changed.GetCacheKey());
  changed = config;
  changed.session_name = "unused";
  changed.access_key_id = "unused";
  changed.access_key_value = "unused";
  EXPECT_EQ(config.GetCacheKey(), changed.GetCacheKey());
  changed = config;
  changed.use_iam = true;
  EXPECT_NE(config.GetCacheKey(), changed.GetCacheKey());
  auto native_iam = changed;
  native_iam.credential_endpoint.clear();
  EXPECT_EQ(changed.GetCacheKey(), native_iam.GetCacheKey());
  changed = config;
  changed.credential_endpoint.clear();
  EXPECT_NE(config.GetCacheKey(), changed.GetCacheKey());
}

TEST_F(FileSystemCacheTest, AzureBrokerCacheKeyUsesBrokerIdentity) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAzure;
  base.address = "core.windows.net";
  base.bucket_name = "container";
  base.region = "westus3";
  base.access_key_id = "account";
  base.azure_client_id = "client-a";
  base.azure_tenant_id = "tenant-a";
  base.credential_endpoint = "http://credential-broker/v1/credentials/assume-role";
  base.load_frequency = 3600;

  auto different_client = base;
  different_client.azure_client_id = "client-b";
  EXPECT_NE(base.GetCacheKey(), different_client.GetCacheKey());

  auto different_tenant = base;
  different_tenant.azure_tenant_id = "tenant-b";
  EXPECT_NE(base.GetCacheKey(), different_tenant.GetCacheKey());

  auto different_endpoint = base;
  different_endpoint.credential_endpoint = "https://other-broker/v1/credentials/assume-role";
  EXPECT_NE(base.GetCacheKey(), different_endpoint.GetCacheKey());

  auto different_static_secret = base;
  different_static_secret.access_key_value = "unused";
  EXPECT_EQ(base.GetCacheKey(), different_static_secret.GetCacheKey());

  auto iam = base;
  iam.use_iam = true;
  EXPECT_NE(base.GetCacheKey(), iam.GetCacheKey());
  auto different_broker = iam;
  different_broker.azure_client_id = "unused-client";
  different_broker.azure_tenant_id = "unused-tenant";
  different_broker.access_key_value = "unused-secret";
  EXPECT_EQ(iam.GetCacheKey(), different_broker.GetCacheKey());
  different_broker.credential_endpoint = "https://other-broker/credentials";
  EXPECT_EQ(iam.GetCacheKey(), different_broker.GetCacheKey());
  auto no_broker = iam;
  no_broker.credential_endpoint.clear();
  no_broker.azure_client_id.clear();
  no_broker.azure_tenant_id.clear();
  EXPECT_EQ(iam.GetCacheKey(), no_broker.GetCacheKey());
  auto different_account = iam;
  different_account.access_key_id = "otheraccount";
  EXPECT_NE(iam.GetCacheKey(), different_account.GetCacheKey());
  iam.talon_mode = different_broker.talon_mode = TalonMode::Full;
  EXPECT_NE(iam.GetCacheKey(), different_broker.GetCacheKey());
  auto ignored_targets = iam;
  ignored_targets.azure_client_id = "unused-client";
  ignored_targets.azure_tenant_id = "unused-tenant";
  ignored_targets.role_arn = "unused-role";
  ignored_targets.external_id = "unused-external-id";
  ignored_targets.gcp_target_service_account = "unused@example.com";
  EXPECT_EQ(iam.GetCacheKey(), ignored_targets.GetCacheKey());

  auto account_key = no_broker;
  account_key.use_iam = false;
  account_key.access_key_value = "key-a";
  EXPECT_NE(no_broker.GetCacheKey(), account_key.GetCacheKey());
  auto other_key = account_key;
  other_key.access_key_value = "key-b";
  EXPECT_NE(account_key.GetCacheKey(), other_key.GetCacheKey());
}

// For GCP without use_iam, credential_endpoint selects the native Broker
// identity, so GetCacheKey hashes it whether or not Talon is enabled. Only
// Talon-specific settings stay out of the origin filesystem's identity.
TEST_F(FileSystemCacheTest, GcpBrokerEndpointIsPartOfOriginIdentityRegardlessOfTalon) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.talon_mode = TalonMode::Full;
  base.credential_endpoint = "https://broker/assume-role";
  base.cloud_provider = kCloudProviderGCP;
  base.use_iam = false;
  base.gcp_target_service_account = "reader-a@example.com";
  auto different = base;
  different.use_iam = true;
  EXPECT_NE(base.GetCacheKey(), different.GetCacheKey());
  different = base;
  different.gcp_target_service_account = "reader-b@example.com";
  EXPECT_NE(base.GetCacheKey(), different.GetCacheKey());
  different = base;
  different.credential_endpoint = "https://other-broker/assume-role";
  EXPECT_NE(base.GetCacheKey(), different.GetCacheKey());
  // The endpoint is a credential selector, not part of the display string.
  EXPECT_EQ(base.ToString().find("https://broker"), std::string::npos);

  // Disabling Talon does not merge the two Broker identities: the endpoint
  // still picks the native GcpBrokerCredentialProvider.
  auto base_native = base;
  auto different_native = different;
  base_native.talon_mode = TalonMode::Disabled;
  different_native.talon_mode = TalonMode::Disabled;
  EXPECT_NE(base_native.GetCacheKey(), different_native.GetCacheKey());

  // Talon-only settings never change the origin identity.
  auto talon_only = base_native;
  talon_only.talon_coordinator = "talon-coordinator.other:9000";
  talon_only.talon_namespace = "other-namespace";
  talon_only.talon_block_size = base_native.talon_block_size * 2;
  EXPECT_EQ(base_native.GetCacheKey(), talon_only.GetCacheKey());

  // With use_iam the native filesystem uses this process's identity and the
  // endpoint describes only Talon's Broker request, so it drops out of the
  // origin identity once Talon is disabled.
  auto iam_a = base_native;
  iam_a.use_iam = true;
  auto iam_b = different_native;
  iam_b.use_iam = true;
  EXPECT_EQ(iam_a.GetCacheKey(), iam_b.GetCacheKey());
}

TEST_F(FileSystemCacheTest, TalonBrokerIamSeparatesClientsWithTheSameOriginRole) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.talon_mode = TalonMode::Full;
  base.credential_endpoint = "https://broker/credentials";
  base.role_arn = "arn:aws:iam::123456789012:role/reader";
  auto iam = base;
  iam.use_iam = true;
  EXPECT_NE(base.GetCacheKey(), iam.GetCacheKey());
  // With Talon disabled, use_iam selects native STS instead of the Broker.
  base.talon_mode = iam.talon_mode = TalonMode::Disabled;
  EXPECT_NE(base.GetCacheKey(), iam.GetCacheKey());
}

TEST_F(FileSystemCacheTest, TalonNamespaceUsesConfiguredIdentityIndependentlyOfBroker) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.talon_mode = TalonMode::Full;
  auto other = base;
  other.talon_namespace = "default";
  EXPECT_NE(base.GetCacheKey(), other.GetCacheKey());
  other.talon_namespace = "tenant-a";
  EXPECT_NE(base.GetCacheKey(), other.GetCacheKey());
  base.talon_namespace = "tenant-b";
  EXPECT_NE(base.GetCacheKey(), other.GetCacheKey());
  base.talon_mode = other.talon_mode = TalonMode::Disabled;
  EXPECT_EQ(base.GetCacheKey(), other.GetCacheKey());
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

TEST_F(FileSystemCacheTest, CacheKeyIgnoresLanceIoParallelism) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "s3.amazonaws.com";
  base.bucket_name = "shared-bucket";
  base.access_key_id = "ak";
  base.access_key_value = "sk";

  auto different_parallelism = base;
  different_parallelism.lance_io_parallelism = 256;
  EXPECT_EQ(base.GetCacheKey(), different_parallelism.GetCacheKey());
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

  ASSERT_AND_ASSIGN(auto fs_a, cache.get(std::make_shared<const api::Properties>(props_a)));
  ASSERT_AND_ASSIGN(auto fs_a_again, cache.get(std::make_shared<const api::Properties>(props_a)));
  ASSERT_AND_ASSIGN(auto fs_b, cache.get(std::make_shared<const api::Properties>(props_b)));

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
  ASSERT_AND_ASSIGN(
      auto res1, cache.get(std::make_shared<const api::Properties>(props1), "s3://localhost_A/bucket_A/file.parquet"));
  ASSERT_AND_ASSIGN(
      auto res2, cache.get(std::make_shared<const api::Properties>(props2), "s3://localhost_B/bucket_B/file.parquet"));

  // Insert third -> should evict least recently used (props1)
  ASSERT_AND_ASSIGN(
      auto res3, cache.get(std::make_shared<const api::Properties>(props3), "s3://localhost_C/bucket_C/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // Accessing props1 should recreate it
  ASSERT_AND_ASSIGN(
      auto res1b, cache.get(std::make_shared<const api::Properties>(props1), "s3://localhost_A/bucket_A/file.parquet"));
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
  ASSERT_AND_ASSIGN(auto res1,
                    cache.get(std::make_shared<const api::Properties>(A), "s3://localhost_X/bucket_X/file.parquet"));
  ASSERT_AND_ASSIGN(auto res2,
                    cache.get(std::make_shared<const api::Properties>(B), "s3://localhost_Y/bucket_Y/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // Access A again to make it most-recent
  ASSERT_AND_ASSIGN(auto res3,
                    cache.get(std::make_shared<const api::Properties>(A), "s3://localhost_X/bucket_X/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_EQ(res1, res3);

  // Insert C -> should evict B
  ASSERT_AND_ASSIGN(auto res4,
                    cache.get(std::make_shared<const api::Properties>(C), "s3://localhost_Z/bucket_Z/file.parquet"));
  EXPECT_EQ(cache.size(), 2u);

  // B should be gone and recreated when requested
  ASSERT_AND_ASSIGN(auto res5,
                    cache.get(std::make_shared<const api::Properties>(B), "s3://localhost_Y/bucket_Y/file.parquet"));
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
        auto result = cache.get(std::make_shared<const api::Properties>(props), path);
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

TEST_F(FileSystemCacheTest, ConcurrentSameKeyGetsReturnSingleCachedInstance) {
  auto& cache = FilesystemCache::getInstance();
  cache.set_capacity(10);

  auto props = MakeProperties("SINGLE_CREATE");
  std::string path = "s3://localhost_SINGLE_CREATE/bucket_SINGLE_CREATE/file.parquet";

  const int thread_count = 256;
  std::vector<std::thread> threads;
  std::vector<ArrowFileSystemPtr> results(thread_count);
  std::vector<std::string> errors(thread_count);
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};

  threads.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&cache, &props, &path, &results, &errors, &ready, &start, i]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      auto result = cache.get(std::make_shared<const api::Properties>(props), path);
      if (!result.ok()) {
        errors[i] = result.status().ToString();
        return;
      }
      results[i] = result.ValueOrDie();
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  for (const auto& error : errors) {
    EXPECT_TRUE(error.empty()) << error;
  }

  ASSERT_NE(results[0], nullptr);
  for (const auto& fs : results) {
    EXPECT_EQ(results[0], fs);
  }
  EXPECT_EQ(cache.size(), 1u);
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
        auto result = cache.get(std::make_shared<const api::Properties>(props), path);
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
