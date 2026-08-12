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
  auto opts = ToStorageOptions(MakeAwsConfig());

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAWS);
  EXPECT_EQ(opts["aws_access_key_id"], "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(opts["aws_secret_access_key"], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(opts["aws_region"], "us-west-2");
  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("s3.access-key-id"), 0);
  EXPECT_EQ(opts["milvus_lance_io_parallelism"], "64");
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

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAzure);
  EXPECT_EQ(opts["azure_storage_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_storage_account_key"], "myaccountkey");
  EXPECT_EQ(opts["azure_endpoint"], "https://myaccount.blob.core.windows.net");
  EXPECT_EQ(opts.count("adls.account-name"), 0);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
}

TEST_F(LanceStorageOptionsTest, AzureCredentialBrokerKeysExcludeFallbackCredentials) {
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

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["azure_storage_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_broker_endpoint"], "http://credential-broker/v1/credentials/assume-role");
  EXPECT_EQ(opts["azure_broker_client_id"], "client-id");
  EXPECT_EQ(opts["azure_broker_tenant_id"], "tenant-id");
  EXPECT_EQ(opts["azure_broker_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_broker_region"], "westus3");
  EXPECT_EQ(opts["azure_broker_bucket"], "mycontainer");
  EXPECT_EQ(opts["azure_broker_duration_seconds"], "3600");
  EXPECT_EQ(opts["azure_broker_request_timeout_ms"], "5000");
  EXPECT_EQ(opts.count("azure_storage_account_key"), 0);
  EXPECT_EQ(opts.count("azure_storage_sas_token"), 0);
  EXPECT_EQ(opts["milvus_fs_cache_key"], config.GetCacheKey());
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

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderAliyun);
  EXPECT_EQ(opts["oss_access_key_id"], "LTAI5tExample");
  EXPECT_EQ(opts["oss_secret_access_key"], "OSSSecretExample");
  EXPECT_EQ(opts["oss_region"], "oss-cn-hangzhou");
  EXPECT_EQ(opts["oss_endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
}

TEST_F(LanceStorageOptionsTest, GcpImpersonation) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.use_iam = true;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";
  config.load_frequency = 1800;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  // Bridge-private keys; not forwarded to lance-io / object_store.
  EXPECT_EQ(opts["gcp_target_service_account"], "target-sa@customer-project.iam.gserviceaccount.com");
  EXPECT_EQ(opts["gcp_credential_refresh_secs"], "1800");
  EXPECT_EQ(opts["milvus_fs_cache_key"], config.GetCacheKey());
}

TEST_F(LanceStorageOptionsTest, GcpTargetServiceAccountRequiresIam) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.gcp_target_service_account = "target-sa@customer-project.iam.gserviceaccount.com";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts.size(), 2);
  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
  EXPECT_EQ(opts["milvus_lance_io_parallelism"], "64");
}

TEST_F(LanceStorageOptionsTest, GcpDefaultCredentials) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;

  auto opts = ToStorageOptions(config);

  // No gcp_target_service_account → no impersonation keys; lance-io falls back
  // to the default credential chain (VM metadata).
  EXPECT_EQ(opts.size(), 2);
  EXPECT_EQ(opts["cloud_provider"], kCloudProviderGCP);
  EXPECT_EQ(opts.count("milvus_fs_cache_key"), 0);
  EXPECT_EQ(opts["milvus_lance_io_parallelism"], "64");
}

TEST_F(LanceStorageOptionsTest, LocalContainsOnlyLanceIoParallelism) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.lance_io_parallelism = 17;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts.size(), 1);
  EXPECT_EQ(opts["milvus_lance_io_parallelism"], "17");
}

TEST_F(LanceStorageOptionsTest, BareEndpointUsesHttpWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "localhost:9000";
  config.use_ssl = false;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["aws_endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(LanceStorageOptionsTest, BareEndpointUsesHttpsWhenSslEnabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "s3.us-west-2.amazonaws.com";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

TEST_F(LanceStorageOptionsTest, ExplicitHttpEndpointIsPreserved) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "http://localhost:9000";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["aws_endpoint"], "http://localhost:9000");
  EXPECT_EQ(opts["allow_http"], "true");
}

TEST_F(LanceStorageOptionsTest, ExplicitHttpsEndpointIsPreservedWhenSslDisabled) {
  ArrowFileSystemConfig config = MakeAwsConfig();
  config.address = "https://s3.us-west-2.amazonaws.com";
  config.use_ssl = false;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("allow_http"), 0);
}

}  // namespace milvus_storage::lance::test
