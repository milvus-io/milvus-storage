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
#include <gmock/gmock.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/constants.h"

namespace milvus_storage {

class S3FileSystemProducerTest : public ::testing::Test {
  protected:
  ArrowFileSystemConfig config;

  void SetUp() override {
    config.cloud_provider = "aws";
    config.useIAM = true;
    config.access_key_id = "test-access-key";
    config.access_key_value = "test-secret-key";
    config.bucket_name = "test-bucket";
    config.address = "s3.amazonaws.com";
    config.region = "us-east-1";
    config.use_custom_part_upload = true;
  }
};

TEST_F(S3FileSystemProducerTest, TestCreateS3Options_NotUseSSL) {
  // config.useVirtualHost = true;
  auto producer = std::make_shared<S3FileSystemProducer>(config);
  auto options = producer->CreateS3Options().value();
  EXPECT_EQ(options.scheme, "http");
  EXPECT_EQ(options.endpoint_override, "s3.amazonaws.com");
  EXPECT_EQ(options.region, config.region);
  EXPECT_EQ(options.force_virtual_addressing, false);
  EXPECT_EQ(options.request_timeout, 3);
}

TEST_F(S3FileSystemProducerTest, TestCreateS3Options_UseSSL) {
  config.requestTimeoutMs = 0;
  config.useSSL = true;
  config.useVirtualHost = true;
  auto producer = std::make_shared<S3FileSystemProducer>(config);
  auto options = producer->CreateS3Options().value();
  EXPECT_EQ(options.scheme, "https");
  EXPECT_EQ(options.endpoint_override, "s3.amazonaws.com");
  EXPECT_EQ(options.region, config.region);
  EXPECT_EQ(options.force_virtual_addressing, true);
  EXPECT_EQ(options.request_timeout, DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC);
}

TEST_F(S3FileSystemProducerTest, TestS3FileSystemProducer_MakeWithoutIAM) {
  config.useIAM = false;
  auto producer = std::make_shared<S3FileSystemProducer>(config);
  auto fs = producer->Make();
  EXPECT_TRUE(fs.ok());
}

TEST_F(S3FileSystemProducerTest, TestS3FileSystemProducer_MakeWithIAM) {
  config.useIAM = true;
  auto producer = std::make_shared<S3FileSystemProducer>(config);
  auto fs = producer->Make();
  EXPECT_TRUE(fs.ok());
}

}  // namespace milvus_storage