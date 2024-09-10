#include <arrow/filesystem/s3fs.h>
#include "arrow/filesystem/azurefs.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "common/log.h"
#include "gtest/gtest.h"
#include "arrow/result.h"
#include "arrow/buffer.h"

TEST(fs, aliyun) {
  // this test case is just for aliyun access testing, please don't run it
  GTEST_SKIP();

  const std::string access_key = "";
  const std::string secret_key = "";
  arrow::fs::S3GlobalOptions global_options;
  auto status = arrow::fs::InitializeS3(global_options);
  EXPECT_TRUE(status.ok());
  auto key = "bucket/test.txt";
  arrow::fs::S3Options options;
  options.endpoint_override = "https://bucket.oss-cn-hangzhou.aliyuncs.com";
  options.ConfigureAccessKey(access_key, secret_key);

  auto fs = arrow::fs::S3FileSystem::Make(options).ValueOrDie();
  auto stream = fs->OpenOutputStream(key).ValueOrDie();
  status = stream->Write("hello world");
  EXPECT_TRUE(status.ok());
  status = stream->Close();
  EXPECT_TRUE(status.ok());

  status = arrow::fs::FinalizeS3();
  EXPECT_TRUE(status.ok());

  auto istream = fs->OpenInputStream(key).ValueOrDie();
  auto buffer = istream->Read(11).ValueOrDie();
  EXPECT_EQ("hello world", buffer->ToString());

  status = stream->Close();
  EXPECT_TRUE(status.ok());
  auto file = fs->OpenInputFile(key).ValueOrDie();
  buffer = file->ReadAt(6, 5).ValueOrDie();
  EXPECT_EQ("world", buffer->ToString());

  status = arrow::fs::FinalizeS3();
  EXPECT_TRUE(status.ok());
}

TEST(fs, azure) {
  // this test case is just for azure access testing, please don't run it
  GTEST_SKIP();

  arrow::fs::AzureOptions options;
  options.account_name = "";
  auto status = options.ConfigureAccountKeyCredential("");
  EXPECT_TRUE(status.ok());
  auto fs = arrow::fs::AzureFileSystem::Make(options).ValueOrDie();

  status = fs->CreateDir("sby");
  EXPECT_TRUE(status.ok());
  auto stream = fs->OpenOutputStream("sby/dd").ValueOrDie();
  status = stream->Write("hello world");
  EXPECT_TRUE(status.ok());

  auto istream = fs->OpenInputStream("sby/dd").ValueOrDie();
  auto buffer = istream->Read(11).ValueOrDie();
  EXPECT_EQ("hello world", buffer->ToString());
}