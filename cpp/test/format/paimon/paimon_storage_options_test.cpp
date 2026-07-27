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

#include "milvus-storage/format/paimon/paimon_common.h"

namespace milvus_storage::paimon::test {
namespace {

ArrowFileSystemConfig MakeAwsConfig() {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;
  config.access_key_id = "access-key";
  config.access_key_value = "secret-key";
  config.region = "us-west-2";
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;
  return config;
}

TEST(PaimonStorageOptionsTest, AwsStaticCredentialsAndAddressing) {
  auto options = ToStorageOptions(MakeAwsConfig());
  EXPECT_EQ(options.at("s3.access-key"), "access-key");
  EXPECT_EQ(options.at("s3.secret-key"), "secret-key");
  EXPECT_EQ(options.at("s3.region"), "us-west-2");
  EXPECT_EQ(options.at("s3.endpoint"), "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(options.at("s3.path-style-access"), "true");

  auto virtual_host = MakeAwsConfig();
  virtual_host.use_virtual_host = true;
  EXPECT_EQ(ToStorageOptions(virtual_host).at("s3.path-style-access"), "false");
}

TEST(PaimonStorageOptionsTest, AwsAssumeRoleDoesNotLeakStaticCredentials) {
  auto config = MakeAwsConfig();
  config.role_arn = "arn:aws:iam::123456789012:role/paimon-reader";
  config.session_name = "milvus-external-table";
  config.external_id = "tenant-a";

  auto options = ToStorageOptions(config);
  EXPECT_EQ(options.at("s3.assumed.role.arn"), config.role_arn);
  EXPECT_EQ(options.at("s3.assumed.role.session.name"), config.session_name);
  EXPECT_EQ(options.at("s3.assumed.role.externalId"), config.external_id);
  EXPECT_EQ(options.count("s3.access-key"), 0);
  EXPECT_EQ(options.count("s3.secret-key"), 0);
}

TEST(PaimonStorageOptionsTest, AwsIamUsesDefaultCredentialChain) {
  auto config = MakeAwsConfig();
  config.use_iam = true;
  auto options = ToStorageOptions(config);
  EXPECT_EQ(options.count("s3.access-key"), 0);
  EXPECT_EQ(options.count("s3.secret-key"), 0);
}

TEST(PaimonStorageOptionsTest, EndpointSchemeIsNormalized) {
  auto config = MakeAwsConfig();
  config.address = "localhost:9000";
  config.use_ssl = false;
  EXPECT_EQ(ToStorageOptions(config).at("s3.endpoint"), "http://localhost:9000");

  config.address = "https://s3.us-west-2.amazonaws.com";
  EXPECT_EQ(ToStorageOptions(config).at("s3.endpoint"), "https://s3.us-west-2.amazonaws.com");
}

TEST(PaimonStorageOptionsTest, LocalIsEmpty) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  EXPECT_TRUE(ToStorageOptions(config).empty());
}

TEST(PaimonStorageOptionsTest, AliyunStaticCredentialsAndAssumeRoleFailsClosed) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  config.use_ssl = true;
  config.access_key_id = "oss-access-key";
  config.access_key_value = "oss-secret-key";

  auto options = ToStorageOptions(config);
  EXPECT_EQ(options.at("fs.oss.endpoint"), "https://oss-cn-hangzhou.aliyuncs.com");
  EXPECT_EQ(options.at("fs.oss.accessKeyId"), config.access_key_id);
  EXPECT_EQ(options.at("fs.oss.accessKeySecret"), config.access_key_value);

  config.role_arn = "acs:ram::123456789012:role/paimon-reader";
  EXPECT_THROW(ToStorageOptions(config), std::runtime_error);
}

TEST(PaimonStorageOptionsTest, AzureAccountKey) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.address = "account.dfs.core.windows.net";
  config.use_ssl = true;
  config.access_key_id = "account";
  config.access_key_value = "account-key";

  auto options = ToStorageOptions(config);
  EXPECT_EQ(options.at("azure.endpoint"), "https://account.dfs.core.windows.net");
  EXPECT_EQ(options.at("azure.account-name"), "account");
  EXPECT_EQ(options.at("azure.account-key"), "account-key");
}

TEST(PaimonStorageOptionsTest, GcpStaticAnonymousAndImpersonationFailsClosed) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.address = "storage.googleapis.com";
  config.use_ssl = true;
  config.gcp_credential_json = R"({"type":"service_account"})";
  config.gcp_native_without_auth = true;

  auto options = ToStorageOptions(config);
  EXPECT_EQ(options.at("gcs.endpoint"), "https://storage.googleapis.com");
  EXPECT_EQ(options.at("gcs.credential"), config.gcp_credential_json);
  EXPECT_EQ(options.at("gcs.allow-anonymous"), "true");

  config.gcp_target_service_account = "target@project.iam.gserviceaccount.com";
  EXPECT_THROW(ToStorageOptions(config), std::runtime_error);
}

TEST(PaimonStorageOptionsTest, UnsupportedProviderFailsClosed) {
  auto config = MakeAwsConfig();
  config.cloud_provider = kCloudProviderTencent;
  EXPECT_THROW(ToStorageOptions(config), std::runtime_error);
}

TEST(PaimonStorageOptionsTest, MilvusUriConversion) {
  EXPECT_EQ(ToStandardUri("s3://localhost:9000/bucket/table"), "s3://bucket/table");
  EXPECT_EQ(ToMilvusUri("s3://bucket/table/data.parquet", "localhost:9000"),
            "s3://localhost:9000/bucket/table/data.parquet");
  EXPECT_EQ(ToMilvusUri("/tmp/table/data.parquet", "localhost:9000"), "/tmp/table/data.parquet");
  EXPECT_EQ(ToMilvusUri("s3://bucket/table/data.parquet", ""), "s3://bucket/table/data.parquet");
}

}  // namespace
}  // namespace milvus_storage::paimon::test
