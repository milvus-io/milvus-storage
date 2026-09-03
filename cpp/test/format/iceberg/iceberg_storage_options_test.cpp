// Copyright 2025 Zilliz
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
#include <filesystem>

#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "test_env.h"

namespace milvus_storage::iceberg::test {

class IcebergStorageOptionsTest : public ::testing::Test {};

static ArrowFileSystemConfig MakeAwsConfig() {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;
  config.access_key_id = "AKIAIOSFODNN7EXAMPLE";
  config.access_key_value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  config.region = "us-west-2";
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;
  return config;
}

TEST_F(IcebergStorageOptionsTest, ReaderOptionsRemoteContainOnlyFilesystemIdentity) {
  auto config = MakeAwsConfig();
  config.bucket_name = "tenant-bucket";

  auto options = ToReaderOptions(config);

  EXPECT_EQ(options, (std::unordered_map<std::string, std::string>{{"milvus_fs_is_local", "false"},
                                                                   {"milvus_fs_bucket", "tenant-bucket"}}));
  EXPECT_EQ(options.count("cloud_provider"), 0);
  EXPECT_EQ(options.count("s3.access-key-id"), 0);
  EXPECT_EQ(options.count("s3.secret-access-key"), 0);
  EXPECT_EQ(options.count("s3.endpoint"), 0);
}

TEST_F(IcebergStorageOptionsTest, ReaderOptionsLocalNormalizeRoot) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "relative-iceberg-root";

  auto options = ToReaderOptions(config);
  auto expected = (std::filesystem::current_path() / config.root_path).lexically_normal().string();

  EXPECT_EQ(options.at("milvus_fs_is_local"), "true");
  EXPECT_EQ(options.at("milvus_fs_root_path"), expected);
  EXPECT_EQ(options.count("milvus_fs_bucket"), 0);
}

TEST_F(IcebergStorageOptionsTest, AwsKeys) {
  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(MakeAwsConfig()));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["s3.access-key-id"], "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(opts["s3.secret-access-key"], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(opts["s3.region"], "us-west-2");
  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("aws_access_key_id"), 0);
}

TEST_F(IcebergStorageOptionsTest, AwsIamDoesNotForwardStaticCredentials) {
  auto config = MakeAwsConfig();
  config.use_iam = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("s3.access-key-id"), 0);
  EXPECT_EQ(opts.count("s3.secret-access-key"), 0);
  EXPECT_EQ(opts["s3.region"], "us-west-2");
  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
}

TEST_F(IcebergStorageOptionsTest, AzureKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "myaccountkey";

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["adls.account-name"], "myaccount");
  EXPECT_EQ(opts["adls.account-key"], "myaccountkey");
  EXPECT_EQ(opts.count("azure_storage_account_name"), 0);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsAzureCredentialBroker) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "must-not-be-forwarded";
  config.bucket_name = "mycontainer";
  config.region = "westus3";
  config.address = "core.windows.net";
  config.use_iam = true;
  config.azure_client_id = "client-id";
  config.azure_tenant_id = "tenant-id";
  config.azure_credential_endpoint = "http://credential-broker/v1/credentials/assume-role";
  config.load_frequency = 3600;
  config.request_timeout_ms = 5000;

  auto result = ToWriterOptions(config);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, AliyunKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.access_key_id = "LTAI5tExample";
  config.access_key_value = "OSSSecretExample";
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["oss.access-key-id"], "LTAI5tExample");
  EXPECT_EQ(opts["oss.access-key-secret"], "OSSSecretExample");
  EXPECT_EQ(opts["oss.endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
  EXPECT_EQ(opts.count("oss.role-arn"), 0);
  EXPECT_EQ(opts.count("oss.role-session-name"), 0);
}

TEST_F(IcebergStorageOptionsTest, AliyunIamDoesNotForwardStaticCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.use_iam = true;
  config.access_key_id = "must-not-be-forwarded";
  config.access_key_value = "must-not-be-forwarded";
  config.region = "cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("oss.access-key-id"), 0);
  EXPECT_EQ(opts.count("oss.access-key-secret"), 0);
  EXPECT_EQ(opts["oss.region"], "cn-hangzhou");
  EXPECT_EQ(opts["oss.endpoint"], "http://oss-cn-hangzhou.aliyuncs.com");
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsGcpImpersonation) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = true;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";
  config.load_frequency = 1800;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsGcpImpersonationWithoutIam) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = false;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsGcpHmacCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.access_key_id = "GOOGACCESSKEY";
  config.access_key_value = "secret";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, GcpIamUsesDefaultCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_TRUE(opts.empty());
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsAwsAssumeRole) {
  auto config = MakeAwsConfig();
  config.role_arn = "arn:aws:iam::123456789012:role/test-role";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsAliyunRole) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.role_arn = "acs:ram::123456789012:role/test-role";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsUnsupportedProvider) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderTencent;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsUnknownProvider) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = "unknown";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, WriterRejectsMissingCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(IcebergStorageOptionsTest, LocalWriterOptionsAreEmpty) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_TRUE(opts.empty());
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(IcebergStorageOptionsTest, BareEndpointUsesHttpWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "localhost:9000";
  config.use_ssl = false;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["s3.endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(IcebergStorageOptionsTest, BareEndpointUsesHttpsWhenSslEnabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

TEST_F(IcebergStorageOptionsTest, ExplicitHttpEndpointIsPreserved) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "http://localhost:9000";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["s3.endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(IcebergStorageOptionsTest, ExplicitHttpsEndpointIsPreservedWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "https://s3.us-west-2.amazonaws.com";
  config.use_ssl = false;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

}  // namespace milvus_storage::iceberg::test
