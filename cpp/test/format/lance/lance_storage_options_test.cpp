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

#include <cstdlib>
#include <gtest/gtest.h>
#include "milvus-storage/format/lance/lance_common.h"
#include "test_env.h"

namespace milvus_storage::lance::test {

// RAII helper to temporarily clear USE_AZURITE so tests are isolated from
// ambient shell env (e.g. when someone has `source scripts/azurite_env.sh`).
class ScopedUnsetAzurite {
  public:
  ScopedUnsetAzurite() {
    const char* v = std::getenv("USE_AZURITE");
    if (v != nullptr) {
      saved_ = v;
      had_ = true;
      unsetenv("USE_AZURITE");
    }
  }
  ~ScopedUnsetAzurite() {
    if (had_) {
      setenv("USE_AZURITE", saved_.c_str(), 1);
    }
  }

  private:
  std::string saved_;
  bool had_ = false;
};

class LanceStorageOptionsTest : public ::testing::Test {};

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

TEST_F(LanceStorageOptionsTest, AwsKeys) {
  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(MakeAwsConfig()));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["aws_access_key_id"], "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(opts["aws_secret_access_key"], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(opts["aws_region"], "us-west-2");
  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("s3.access-key-id"), 0);
  EXPECT_EQ(opts.count("lance_io_parallelism"), 0);
}

TEST_F(LanceStorageOptionsTest, AwsIamDoesNotForwardStaticCredentials) {
  auto config = MakeAwsConfig();
  config.use_iam = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("aws_access_key_id"), 0);
  EXPECT_EQ(opts.count("aws_secret_access_key"), 0);
  EXPECT_EQ(opts["aws_region"], "us-west-2");
  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
}

TEST_F(LanceStorageOptionsTest, AzureKeys) {
  ScopedUnsetAzurite no_azurite;
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "myaccountkey";
  config.address = "core.windows.net";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["azure_storage_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_storage_account_key"], "myaccountkey");
  EXPECT_EQ(opts["azure_endpoint"], "https://myaccount.blob.core.windows.net");
  EXPECT_EQ(opts.count("adls.account-name"), 0);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(LanceStorageOptionsTest, WriterRejectsAzureCredentialBroker) {
  ScopedUnsetAzurite no_azurite;
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "must-not-be-forwarded";
  config.bucket_name = "mycontainer";
  config.region = "westus3";
  config.address = "core.windows.net";
  config.use_ssl = true;
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

TEST_F(LanceStorageOptionsTest, AliyunKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.access_key_id = "LTAI5tExample";
  config.access_key_value = "OSSSecretExample";
  config.region = "oss-cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts["oss_access_key_id"], "LTAI5tExample");
  EXPECT_EQ(opts["oss_secret_access_key"], "OSSSecretExample");
  EXPECT_EQ(opts["oss_region"], "oss-cn-hangzhou");
  EXPECT_EQ(opts["oss_endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
}

TEST_F(LanceStorageOptionsTest, AliyunIamDoesNotForwardStaticCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.use_iam = true;
  config.access_key_id = "must-not-be-forwarded";
  config.access_key_value = "must-not-be-forwarded";
  config.region = "oss-cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts.count("oss_access_key_id"), 0);
  EXPECT_EQ(opts.count("oss_secret_access_key"), 0);
  EXPECT_EQ(opts["oss_region"], "oss-cn-hangzhou");
  EXPECT_EQ(opts["oss_endpoint"], "http://oss-cn-hangzhou.aliyuncs.com");
}

TEST_F(LanceStorageOptionsTest, WriterRejectsGcpImpersonation) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = true;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";
  config.load_frequency = 1800;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsGcpImpersonationWithoutIam) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = false;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsGcpHmacCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = false;
  config.access_key_id = "GOOGACCESSKEY";
  config.access_key_value = "secret";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsAwsAssumeRole) {
  auto config = MakeAwsConfig();
  config.role_arn = "arn:aws:iam::123456789012:role/test-role";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsAliyunRole) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.role_arn = "acs:ram::123456789012:role/test-role";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsUnsupportedProvider) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderTencent;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsUnknownProvider) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = "unknown";

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, WriterRejectsMissingCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;

  auto result = ToWriterOptions(config);
  EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
}

TEST_F(LanceStorageOptionsTest, GcpIamUsesDefaultCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  // No gcp_target_service_account → no impersonation keys; lance-io falls back
  // to the default credential chain (VM metadata).
  EXPECT_EQ(opts.count("cloud_provider"), 0);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
  EXPECT_EQ(opts.count("lance_io_parallelism"), 0);
  EXPECT_EQ(opts.count("gcp_target_service_account"), 0);
  EXPECT_EQ(opts.count("gcp_credential_refresh_secs"), 0);
}

TEST_F(LanceStorageOptionsTest, LocalWriterOptionsAreEmpty) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.lance_io_parallelism = 17;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_TRUE(opts.empty());
}

TEST_F(LanceStorageOptionsTest, AimdRatesAreForwarded) {
  auto config = MakeAwsConfig();

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));
  EXPECT_EQ(opts["lance_aimd_initial_rate"], "2000");
  EXPECT_EQ(opts["lance_aimd_max_rate"], "5000");

  config.iops_initial_rate = 4000;
  config.iops_max_rate = 5000;
  ASSERT_AND_ASSIGN(opts, ToWriterOptions(config));
  EXPECT_EQ(opts["lance_aimd_initial_rate"], "4000");
  EXPECT_EQ(opts["lance_aimd_max_rate"], "5000");

  config.iops_max_rate = 0;
  ASSERT_AND_ASSIGN(opts, ToWriterOptions(config));
  EXPECT_EQ(opts["lance_aimd_max_rate"], "0");
}

TEST_F(LanceStorageOptionsTest, ReadOptionsContainNoCredentials) {
  auto config = MakeAwsConfig();

  auto options = ToReaderOptions(config);

  EXPECT_EQ(options["milvus_fs_cache_key"], config.GetCacheKey());
  EXPECT_EQ(options["milvus_fs_is_local"], "false");
  EXPECT_EQ(options["lance_io_parallelism"], "64");
  EXPECT_EQ(options["lance_aimd_initial_rate"], "2000");
  EXPECT_EQ(options["lance_aimd_max_rate"], "5000");
  EXPECT_EQ(options.count("aws_access_key_id"), 0);
  EXPECT_EQ(options.count("aws_secret_access_key"), 0);
  EXPECT_EQ(options.count("aws_role_arn"), 0);
  EXPECT_EQ(options.count("gcp_target_service_account"), 0);
  EXPECT_EQ(options.count("azure_storage_account_key"), 0);
  EXPECT_EQ(options.count("oss_access_key_id"), 0);
}

TEST_F(LanceStorageOptionsTest, BareEndpointUsesHttpWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "localhost:9000";
  config.use_ssl = false;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["aws_endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(LanceStorageOptionsTest, BareEndpointUsesHttpsWhenSslEnabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

TEST_F(LanceStorageOptionsTest, ExplicitHttpEndpointIsPreserved) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "http://localhost:9000";
  config.use_ssl = true;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["aws_endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(LanceStorageOptionsTest, ExplicitHttpsEndpointIsPreservedWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "https://s3.us-west-2.amazonaws.com";
  config.use_ssl = false;

  ASSERT_AND_ASSIGN(auto opts, ToWriterOptions(config));

  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

}  // namespace milvus_storage::lance::test
