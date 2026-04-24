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
#include "milvus-storage/format/iceberg/iceberg_common.h"

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
  return config;
}

TEST_F(IcebergStorageOptionsTest, AwsKeys) {
  auto opts = ToStorageOptions(MakeAwsConfig());

  EXPECT_EQ(opts["s3.access-key-id"], "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(opts["s3.secret-access-key"], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(opts["s3.region"], "us-west-2");
  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("aws_access_key_id"), 0);
}

TEST_F(IcebergStorageOptionsTest, AzureKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "myaccountkey";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["adls.account-name"], "myaccount");
  EXPECT_EQ(opts["adls.account-key"], "myaccountkey");
  EXPECT_EQ(opts.count("azure_storage_account_name"), 0);
}

TEST_F(IcebergStorageOptionsTest, AliyunKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.access_key_id = "LTAI5tExample";
  config.access_key_value = "OSSSecretExample";
  config.address = "oss-cn-hangzhou.aliyuncs.com";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["oss.access-key-id"], "LTAI5tExample");
  EXPECT_EQ(opts["oss.access-key-secret"], "OSSSecretExample");
  EXPECT_EQ(opts["oss.endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
  // Static-AKSK branch must NOT emit role_arn keys — otherwise the Rust side
  // would incorrectly route through AliyunOssStorage's AssumeRole path.
  EXPECT_EQ(opts.count("oss.role-arn"), 0);
  EXPECT_EQ(opts.count("oss.role-session-name"), 0);
}

TEST_F(IcebergStorageOptionsTest, AliyunArnRole) {
  // Per-tenant AssumeRoleWithOIDC. Must emit endpoint/region + role_arn +
  // session_name, and must NOT emit AK/SK (reqsign's static-creds loader
  // would otherwise take precedence over the OIDC path; see the module-level
  // comment in aliyun_oss_provider.rs).
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.role_arn = "acs:ram::111111111111:role/tenant-A";
  config.session_name = "tenant-A-session";
  config.region = "cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  // Even if the caller populates AK/SK alongside role_arn, the iceberg
  // branch must drop them — a caller mistake here should not bypass OIDC.
  config.access_key_id = "LTAI-should-be-ignored";
  config.access_key_value = "aksk-should-be-ignored";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["oss.endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
  EXPECT_EQ(opts["oss.region"], "cn-hangzhou");
  EXPECT_EQ(opts["oss.role-arn"], "acs:ram::111111111111:role/tenant-A");
  EXPECT_EQ(opts["oss.role-session-name"], "tenant-A-session");
  EXPECT_EQ(opts.count("oss.access-key-id"), 0);
  EXPECT_EQ(opts.count("oss.access-key-secret"), 0);
}

TEST_F(IcebergStorageOptionsTest, LocalEmpty) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  EXPECT_TRUE(ToStorageOptions(config).empty());
}

TEST_F(IcebergStorageOptionsTest, GcpImpersonation) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["gcs.service-account"], "target-sa@customer-project.iam.gserviceaccount.com");
}

TEST_F(IcebergStorageOptionsTest, GcpDefaultCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  EXPECT_TRUE(ToStorageOptions(config).empty());
}

}  // namespace milvus_storage::iceberg::test
