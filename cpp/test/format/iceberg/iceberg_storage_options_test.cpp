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
  config.use_ssl = true;
  return config;
}

TEST_F(IcebergStorageOptionsTest, AwsKeys) {
  auto opts = ToStorageOptions(MakeAwsConfig());

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAWS);
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

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAzure);
  EXPECT_EQ(opts["adls.account-name"], "myaccount");
  EXPECT_EQ(opts["adls.account-key"], "myaccountkey");
  EXPECT_EQ(opts.count("azure_storage_account_name"), 0);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(IcebergStorageOptionsTest, AzureCredentialBrokerKeysExcludeFallbackCredentials) {
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

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["adls.account-name"], "myaccount");
  EXPECT_EQ(opts["adls.endpoint-suffix"], "core.windows.net");
  EXPECT_EQ(opts["azure_broker_endpoint"], "http://credential-broker/v1/credentials/assume-role");
  EXPECT_EQ(opts["azure_broker_client_id"], "client-id");
  EXPECT_EQ(opts["azure_broker_tenant_id"], "tenant-id");
  EXPECT_EQ(opts["azure_broker_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_broker_region"], "westus3");
  EXPECT_EQ(opts["azure_broker_bucket"], "mycontainer");
  EXPECT_EQ(opts["azure_broker_duration_seconds"], "3600");
  EXPECT_EQ(opts["azure_broker_request_timeout_ms"], "5000");
  EXPECT_EQ(opts.count("adls.account-key"), 0);
  EXPECT_EQ(opts.count("adls.client-id"), 0);
  EXPECT_EQ(opts.count("adls.tenant-id"), 0);
  EXPECT_EQ(opts.count("adls.sas-token"), 0);
  EXPECT_EQ(opts["milvus_fs_cache_key"], config.GetCacheKey());
}

TEST_F(IcebergStorageOptionsTest, AliyunKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.access_key_id = "LTAI5tExample";
  config.access_key_value = "OSSSecretExample";
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAliyun);
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
  config.external_id = "tenant-A-ext";
  config.region = "cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  config.use_ssl = true;
  // Even if the caller populates AK/SK alongside role_arn, the iceberg
  // branch must drop them — a caller mistake here should not bypass OIDC.
  config.access_key_id = "LTAI-should-be-ignored";
  config.access_key_value = "aksk-should-be-ignored";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["oss.endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
  EXPECT_EQ(opts["oss.region"], "cn-hangzhou");
  EXPECT_EQ(opts["oss.role-arn"], "acs:ram::111111111111:role/tenant-A");
  EXPECT_EQ(opts["oss.role-session-name"], "tenant-A-session");
  EXPECT_EQ(opts["oss.external-id"], "tenant-A-ext");
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
  config.use_iam = true;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";
  config.load_frequency = 1800;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  EXPECT_EQ(opts["gcs.service-account"], "target-sa@customer-project.iam.gserviceaccount.com");
  EXPECT_EQ(opts["gcp_credential_refresh_secs"], "1800");
  EXPECT_EQ(opts["milvus_fs_cache_key"], config.GetCacheKey());
}

TEST_F(IcebergStorageOptionsTest, GcpTargetServiceAccountRequiresIam) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts.size(), 1);
  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(IcebergStorageOptionsTest, GcpDefaultCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  auto opts = ToStorageOptions(config);
  EXPECT_EQ(opts.size(), 1);
  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(IcebergStorageOptionsTest, BareEndpointUsesHttpWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "localhost:9000";
  config.use_ssl = false;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["s3.endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(IcebergStorageOptionsTest, BareEndpointUsesHttpsWhenSslEnabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

TEST_F(IcebergStorageOptionsTest, ExplicitHttpEndpointIsPreserved) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "http://localhost:9000";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["s3.endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(IcebergStorageOptionsTest, ExplicitHttpsEndpointIsPreservedWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "https://s3.us-west-2.amazonaws.com";
  config.use_ssl = false;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["s3.endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

}  // namespace milvus_storage::iceberg::test
