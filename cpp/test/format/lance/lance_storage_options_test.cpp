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
#include "milvus-storage/format/lance/lance_common.h"

namespace milvus_storage::lance::test {

class LanceStorageOptionsTest : public ::testing::Test {};

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

TEST_F(LanceStorageOptionsTest, AwsKeys) {
  auto opts = ToStorageOptions(MakeAwsConfig());

  EXPECT_EQ(opts["aws_access_key_id"], "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(opts["aws_secret_access_key"], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(opts["aws_region"], "us-west-2");
  EXPECT_EQ(opts["aws_endpoint"], "https://s3.us-west-2.amazonaws.com");
  EXPECT_EQ(opts.count("s3.access-key-id"), 0);
}

TEST_F(LanceStorageOptionsTest, AzureKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "myaccount";
  config.access_key_value = "myaccountkey";
  config.address = ".blob.core.windows.net";
  config.use_ssl = true;

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["azure_storage_account_name"], "myaccount");
  EXPECT_EQ(opts["azure_storage_account_key"], "myaccountkey");
  EXPECT_NE(opts.find("azure_endpoint"), opts.end());
  EXPECT_EQ(opts.count("adls.account-name"), 0);
}

TEST_F(LanceStorageOptionsTest, AliyunKeys) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAliyun;
  config.access_key_id = "LTAI5tExample";
  config.access_key_value = "OSSSecretExample";
  config.region = "oss-cn-hangzhou";
  config.address = "oss-cn-hangzhou.aliyuncs.com";

  auto opts = ToStorageOptions(config);

  EXPECT_EQ(opts["oss_access_key_id"], "LTAI5tExample");
  EXPECT_EQ(opts["oss_secret_access_key"], "OSSSecretExample");
  EXPECT_EQ(opts["oss_region"], "oss-cn-hangzhou");
  EXPECT_EQ(opts["oss_endpoint"], "https://oss-cn-hangzhou.aliyuncs.com");
}

TEST_F(LanceStorageOptionsTest, LocalEmpty) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  EXPECT_TRUE(ToStorageOptions(config).empty());
}

}  // namespace milvus_storage::lance::test
