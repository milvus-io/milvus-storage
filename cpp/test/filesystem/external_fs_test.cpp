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
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"

namespace milvus_storage::test {

class ExternalFilesystemTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Clear any existing registrations before each test
    auto& cache = FilesystemCache::getInstance();
    cache.clean();
  }

  void TearDown() override {
    // Clean up after each test
    auto& cache = FilesystemCache::getInstance();
    cache.clean();
  }
};

// ==================== StorageUri Tests ====================

TEST_F(ExternalFilesystemTest, StorageUriParseRelativePath) {
  // Test relative paths - should return empty scheme, address, and bucket_name
  std::vector<std::string> relative_paths = {"relative/path/to/file.parquet", "file.parquet", "dir/subdir/file.txt",
                                             "/absolute/local/path.txt"};

  for (const auto& path : relative_paths) {
    auto result = StorageUri::Parse(path);
    ASSERT_TRUE(result.ok()) << "Failed to parse: " << path << " - " << result.status().ToString();

    auto uri = result.ValueOrDie();
    EXPECT_EQ(uri.scheme, "") << "Path: " << path;
    EXPECT_EQ(uri.address, "") << "Path: " << path;
    EXPECT_EQ(uri.bucket_name, "") << "Path: " << path;
    EXPECT_EQ(uri.key, path) << "Path: " << path;
  }
}

TEST_F(ExternalFilesystemTest, StorageUriParseS3Simple) {
  auto result = StorageUri::Parse("s3://my-bucket/path/to/file.parquet");
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  auto uri = result.ValueOrDie();
  EXPECT_EQ(uri.scheme, "s3");
  EXPECT_EQ(uri.address, "my-bucket");  // Host is treated as address
  EXPECT_EQ(uri.bucket_name, "path");   // First path component is bucket
  EXPECT_EQ(uri.key, "to/file.parquet");
}

TEST_F(ExternalFilesystemTest, StorageUriParseS3WithEndpoint) {
  auto result = StorageUri::Parse("s3://s3.us-west-2.amazonaws.com/prod-bucket/data/file.parquet");
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  auto uri = result.ValueOrDie();
  EXPECT_EQ(uri.scheme, "s3");
  EXPECT_EQ(uri.address, "s3.us-west-2.amazonaws.com");
  EXPECT_EQ(uri.bucket_name, "prod-bucket");
  EXPECT_EQ(uri.key, "data/file.parquet");
}

TEST_F(ExternalFilesystemTest, StorageUriParseCustomScheme) {
  auto result = StorageUri::Parse("custom://endpoint.example.com/bucket/path/to/object");
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  auto uri = result.ValueOrDie();
  EXPECT_EQ(uri.scheme, "custom");
  EXPECT_EQ(uri.address, "endpoint.example.com");
  EXPECT_EQ(uri.bucket_name, "bucket");
  EXPECT_EQ(uri.key, "path/to/object");
}

TEST_F(ExternalFilesystemTest, StorageUriParseInvalid) {
  // Missing bucket in URI with scheme
  auto result1 = StorageUri::Parse("s3://");
  EXPECT_FALSE(result1.ok());

  // Empty after scheme
  auto result2 = StorageUri::Parse("s3:///");
  EXPECT_FALSE(result2.ok());

  // With simplified parsing: host is always address, bucket comes from path
  // s3://my-bucket has no path, so it's invalid (missing bucket)
  auto result = StorageUri::Parse("s3://my-bucket");
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid());
}

// ==================== External Filesystem Property Tests ====================

TEST_F(ExternalFilesystemTest, ExternalFsRejectInvalidFormat) {
  // Test 1: Missing property name after fs name
  {
    api::Properties props;
    props["extfs.prod"] = std::string("value");

    // With lazy evaluation, provide a path with scheme to trigger extraction
    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(props, "s3://test.com/bucket/key");
    EXPECT_FALSE(fs_result.ok());
    EXPECT_TRUE(fs_result.status().IsInvalid());
    EXPECT_NE(fs_result.status().ToString().find("Invalid external filesystem property format"), std::string::npos);
  }

  // Test 2: Empty filesystem name
  {
    api::Properties props;
    props["extfs..address"] = std::string("value");

    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(props, "s3://test.com/bucket/key");
    EXPECT_FALSE(fs_result.ok());
    EXPECT_TRUE(fs_result.status().IsInvalid());
    EXPECT_NE(fs_result.status().ToString().find("Empty external filesystem name"), std::string::npos);
  }

  // Test 3: Empty property name
  {
    api::Properties props;
    props["extfs.prod."] = std::string("value");

    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(props, "s3://test.com/bucket/key");
    EXPECT_FALSE(fs_result.ok());
    EXPECT_TRUE(fs_result.status().IsInvalid());
    EXPECT_NE(fs_result.status().ToString().find("Empty property name"), std::string::npos);
  }
}

// ==================== External Filesystem Tests ====================

TEST_F(ExternalFilesystemTest, ExternalFsAllowDifferentBuckets) {
  api::Properties props;

  // Add default filesystem properties
  props[PROPERTY_FS_STORAGE_TYPE] = std::string("local");
  props[PROPERTY_FS_ROOT_PATH] = std::string("/tmp/test");

  // First external filesystem
  props["extfs.prod.address"] = std::string("s3.amazonaws.com");
  props["extfs.prod.bucket_name"] = std::string("prod-bucket");
  props["extfs.prod.access_key_id"] = std::string("PROD_KEY");
  props["extfs.prod.access_key_value"] = std::string("PROD_SECRET");
  props["extfs.prod.storage_type"] = std::string("remote");
  props["extfs.prod.cloud_provider"] = std::string(kCloudProviderAWS);

  // Second external filesystem with same address but DIFFERENT bucket
  props["extfs.backup.address"] = std::string("s3.amazonaws.com");
  props["extfs.backup.bucket_name"] = std::string("backup-bucket");  // Different bucket - OK
  props["extfs.backup.access_key_id"] = std::string("BACKUP_KEY");
  props["extfs.backup.access_key_value"] = std::string("BACKUP_SECRET");
  props["extfs.backup.storage_type"] = std::string("remote");
  props["extfs.backup.cloud_provider"] = std::string(kCloudProviderAWS);

  // Should succeed - different buckets are allowed
  auto fs_result = milvus_storage::FilesystemCache::getInstance().get(props, "");
  ASSERT_TRUE(fs_result.ok()) << fs_result.status().ToString();
  fs_result = milvus_storage::FilesystemCache::getInstance().get(
      props, "s3://s3.amazonaws.com/backup-bucket/data/file.parquet");
  ASSERT_TRUE(fs_result.ok()) << fs_result.status().ToString();
  fs_result =
      milvus_storage::FilesystemCache::getInstance().get(props, "s3://s3.amazonaws.com/prod-bucket/data/file.parquet");
  ASSERT_TRUE(fs_result.ok()) << fs_result.status().ToString();
}

// ==================== Integration Tests ====================

TEST_F(ExternalFilesystemTest, IntegrationExternalFsWithPath) {
  api::Properties props;

  // Add default filesystem properties
  props[PROPERTY_FS_STORAGE_TYPE] = std::string("local");
  props[PROPERTY_FS_ROOT_PATH] = std::string("/tmp/test");

  // Add external filesystem for prod-bucket
  props["extfs.prod.address"] = std::string("s3.amazonaws.com");
  props["extfs.prod.bucket_name"] = std::string("prod-bucket");
  props["extfs.prod.storage_type"] = std::string("remote");
  props["extfs.prod.cloud_provider"] = std::string(kCloudProviderAWS);
  props["extfs.prod.access_key_id"] = std::string("test");
  props["extfs.prod.access_key_value"] = std::string("test");

  // Get filesystem with matching URI - should use extfs.prod config
  auto fs_result =
      milvus_storage::FilesystemCache::getInstance().get(props, "s3://s3.amazonaws.com/prod-bucket/data/file.parquet");
  ASSERT_TRUE(fs_result.ok()) << fs_result.status().ToString();

  // Get filesystem with non-matching URI - should use default fs
  auto fs_default = milvus_storage::FilesystemCache::getInstance().get(props, "");
  ASSERT_TRUE(fs_default.ok()) << fs_default.status().ToString();
}

TEST_F(ExternalFilesystemTest, IntegrationMultipleExternalFs) {
  api::Properties props;

  // Production filesystem for prod-data bucket
  props["extfs.prod.address"] = std::string("s3.us-west-2.amazonaws.com");
  props["extfs.prod.bucket_name"] = std::string("prod-data");
  props["extfs.prod.storage_type"] = std::string("remote");
  props["extfs.prod.cloud_provider"] = std::string(kCloudProviderAWS);
  props["extfs.prod.access_key_id"] = std::string("prod_key");
  props["extfs.prod.access_key_value"] = std::string("prod_secret");

  // Backup filesystem for backup-data bucket
  props["extfs.backup.address"] = std::string("s3.us-east-1.amazonaws.com");
  props["extfs.backup.bucket_name"] = std::string("backup-data");
  props["extfs.backup.storage_type"] = std::string("remote");
  props["extfs.backup.cloud_provider"] = std::string(kCloudProviderAWS);
  props["extfs.backup.access_key_id"] = std::string("backup_key");
  props["extfs.backup.access_key_value"] = std::string("backup_secret");

  // Get filesystem with prod bucket URI
  auto fs_prod = milvus_storage::FilesystemCache::getInstance().get(
      props, "s3://s3.us-west-2.amazonaws.com/prod-data/file.parquet");
  ASSERT_TRUE(fs_prod.ok()) << fs_prod.status().ToString();

  // Get filesystem with backup bucket URI
  auto fs_backup = milvus_storage::FilesystemCache::getInstance().get(
      props, "s3://s3.us-east-1.amazonaws.com/backup-data/file.parquet");
  ASSERT_TRUE(fs_backup.ok()) << fs_backup.status().ToString();
}

// ==================== Type Conversion Tests ====================

TEST_F(ExternalFilesystemTest, ExtractExternalFsPropertiesTypeConversion) {
  api::Properties props;

  // Set ALL extfs.* properties as strings (simulating user input)
  // String properties
  props["extfs.myfs.address"] = std::string("s3.us-west-2.amazonaws.com");
  props["extfs.myfs.bucket_name"] = std::string("my-bucket");
  props["extfs.myfs.access_key_id"] = std::string("AKIAIOSFODNN7EXAMPLE");
  props["extfs.myfs.access_key_value"] = std::string("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  props["extfs.myfs.root_path"] = std::string("/data/storage");
  props["extfs.myfs.storage_type"] = std::string("remote");
  props["extfs.myfs.cloud_provider"] = std::string(kCloudProviderAWS);
  props["extfs.myfs.iam_endpoint"] = std::string("https://iam.amazonaws.com");
  props["extfs.myfs.log_level"] = std::string("info");
  props["extfs.myfs.region"] = std::string("us-west-2");
  props["extfs.myfs.ssl_ca_cert"] = std::string("/etc/ssl/certs/ca.pem");
  props["extfs.myfs.gcp_credential_json"] = std::string("{\"type\":\"service_account\"}");
  props["extfs.myfs.role_arn"] = std::string("arn:aws:iam::123456789012:role/myrole");
  props["extfs.myfs.session_name"] = std::string("my-session");
  props["extfs.myfs.external_id"] = std::string("ext-id-123");
  props["extfs.myfs.tls_min_version"] = std::string("1.2");

  // Bool properties (as string "true"/"false" - this is the bug scenario)
  props["extfs.myfs.use_ssl"] = std::string("true");
  props["extfs.myfs.use_iam"] = std::string("true");
  props["extfs.myfs.use_virtual_host"] = std::string("true");
  props["extfs.myfs.gcp_native_without_auth"] = std::string("true");
  props["extfs.myfs.use_custom_part_upload"] = std::string("false");
  props["extfs.myfs.background_writes"] = std::string("false");
  props["extfs.myfs.use_crc32c_checksum"] = std::string("true");

  // Numeric properties (as strings)
  props["extfs.myfs.request_timeout_ms"] = std::string("5000");
  props["extfs.myfs.max_connections"] = std::string("200");
  props["extfs.myfs.multi_part_upload_size"] = std::string("20971520");
  props["extfs.myfs.load_frequency"] = std::string("1800");

  // resolve_config should succeed and produce correctly typed values
  auto config_result =
      FilesystemCache::resolve_config(props, "s3://s3.us-west-2.amazonaws.com/my-bucket/data/file.parquet");
  ASSERT_TRUE(config_result.ok()) << config_result.status().ToString();

  auto config = config_result.ValueOrDie();

  // Verify string properties
  EXPECT_EQ(config.address, "s3.us-west-2.amazonaws.com");
  EXPECT_EQ(config.bucket_name, "my-bucket");
  EXPECT_EQ(config.access_key_id, "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(config.access_key_value, "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(config.root_path, "/data/storage");
  EXPECT_EQ(config.storage_type, "remote");
  EXPECT_EQ(config.cloud_provider, kCloudProviderAWS);
  EXPECT_EQ(config.iam_endpoint, "https://iam.amazonaws.com");
  EXPECT_EQ(config.log_level, "info");
  EXPECT_EQ(config.region, "us-west-2");
  EXPECT_EQ(config.ssl_ca_cert, "/etc/ssl/certs/ca.pem");
  EXPECT_EQ(config.gcp_credential_json, "{\"type\":\"service_account\"}");
  EXPECT_EQ(config.role_arn, "arn:aws:iam::123456789012:role/myrole");
  EXPECT_EQ(config.session_name, "my-session");
  EXPECT_EQ(config.external_id, "ext-id-123");
  EXPECT_EQ(config.tls_min_version, "1.2");

  // Verify bool properties (these would crash with bad_variant_access before the fix)
  EXPECT_EQ(config.use_ssl, true);
  EXPECT_EQ(config.use_iam, true);
  EXPECT_EQ(config.use_virtual_host, true);
  EXPECT_EQ(config.gcp_native_without_auth, true);
  EXPECT_EQ(config.use_custom_part_upload, false);
  EXPECT_EQ(config.background_writes, false);
  EXPECT_EQ(config.use_crc32c_checksum, true);

  // Verify numeric properties
  EXPECT_EQ(config.request_timeout_ms, 5000);
  EXPECT_EQ(config.max_connections, 200u);
  EXPECT_EQ(config.multi_part_upload_size, 20971520);
  EXPECT_EQ(config.load_frequency, 1800);

  // Verify alias
  EXPECT_EQ(config.alias, "myfs");
}

TEST_F(ExternalFilesystemTest, ExtractExternalFsRejectsUndefinedProperty) {
  api::Properties props;

  // Set required properties
  props["extfs.myfs.address"] = std::string("s3.amazonaws.com");
  props["extfs.myfs.bucket_name"] = std::string("my-bucket");
  props["extfs.myfs.storage_type"] = std::string("remote");

  // Set an undefined property that doesn't map to any registered fs.* key
  props["extfs.myfs.nonexistent_key"] = std::string("some_value");

  auto config_result = FilesystemCache::resolve_config(props, "s3://s3.amazonaws.com/my-bucket/data/file.parquet");
  EXPECT_FALSE(config_result.ok());
  EXPECT_TRUE(config_result.status().IsInvalid());
}

}  // namespace milvus_storage::test