// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#ifdef WITH_CRT
#include <chrono>
#include <random>
#include <set>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/thread_pool.h>
#include <aws/s3/model/ListObjectsV2Result.h>
#include <gtest/gtest.h>
#include "filesystem/s3/native_s3_operations.h"
#include "milvus-storage/filesystem/async_output_stream.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "test_env.h"

namespace milvus_storage {
namespace {
template <typename T>
arrow::Result<T> WaitCloud(arrow::Future<T> future) {
  if (!future.Wait(60))
    return arrow::Status::IOError("Timed out waiting for native cloud request");
  return future.result();
}
}  // namespace

// Each test uses a fresh prefix in an existing bucket. A stopped I/O executor
// makes SDK executor fallback fail instead of disguising it as native coverage.
class S3NativeCloudTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv() || GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("") == "azure")
      GTEST_SKIP() << "Requires a configured S3-compatible cloud";
    api::Properties properties;
    ASSERT_OK(InitTestProperties(properties));
    ASSERT_OK_AND_ASSIGN(config_, GetFileSystemConfig(properties));
    config_.s3_crt_async_read = true;
    config_.multi_part_upload_size = 5 * 1024 * 1024;
    config_.max_connections = 4;
    config_.request_timeout_ms = 30000;
    config_.log_level = "off";
    // Initialize through the real producer, including the GCP HTTP factory and
    // credential registry. Keep this normal executor instance for cleanup only.
    ASSERT_OK_AND_ASSIGN(cleanup_, CreateArrowFileSystem(config_));
    ASSERT_OK_AND_ASSIGN(options_, S3FileSystemProducer(config_).CreateS3Options());
    options_.connect_timeout = 10;
    ASSERT_OK_AND_ASSIGN(executor_, arrow::internal::ThreadPool::Make(1));
    ASSERT_OK(executor_->Shutdown());
    ASSERT_OK_AND_ASSIGN(auto raw, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get())));
    prefix_ = "pr693-native-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
              std::to_string(std::random_device{}());
    fs_ = std::make_shared<FileSystemProxy>(config_.bucket_name + "/" + prefix_, raw);
    std::cout << "Native cloud test prefix: " << prefix_ << std::endl;
  }
  void TearDown() override {
    if (cleanup_ && !prefix_.empty()) {
      auto status = cleanup_->DeleteDirContents(prefix_, true);
      EXPECT_TRUE(status.ok()) << status;
    }
  }
  arrow::Status Put(const std::string& path, const std::string& data) {
    ARROW_ASSIGN_OR_RAISE(auto out, fs_->OpenOutputStream(path));
    ARROW_RETURN_NOT_OK(out->Write(arrow::Buffer::FromString(data)));
    return WaitCloud(out->CloseAsync()).status();
  }
  arrow::Result<arrow::fs::FileInfo> Stat(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto infos, WaitCloud(fs_->GetFileInfoAsync({path})));
    return infos.at(0);
  }
  ArrowFileSystemConfig config_;
  S3Options options_;
  FileSystemPtr cleanup_;
  FileSystemPtr fs_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  std::string prefix_;
};

TEST_F(S3NativeCloudTest, PutMetadataRangeReadAndOverwrite) {
  const std::string path = "nested/hello #?+% 中文";
  auto metadata = arrow::key_value_metadata({"Content-Type"}, {"text/plain"});
  ASSERT_OK_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, metadata));
  ASSERT_NE(std::dynamic_pointer_cast<AsyncOutputStream>(out), nullptr);
  ASSERT_OK(out->Write(arrow::Buffer::FromString("abcdef")));
  ASSERT_OK(WaitCloud(out->CloseAsync()));
  ASSERT_OK_AND_ASSIGN(auto info, Stat(path));
  EXPECT_EQ(info.size(), 6);
  ASSERT_OK_AND_ASSIGN(auto file, WaitCloud(fs_->OpenInputFileAsync(path)));
  ASSERT_NE(std::dynamic_pointer_cast<NonBlockingRandomAccessFile>(file), nullptr);
  ASSERT_OK_AND_ASSIGN(auto headers, WaitCloud(file->ReadMetadataAsync({})));
  ASSERT_OK_AND_ASSIGN(auto content_type, headers->Get("Content-Type"));
  EXPECT_EQ(content_type, "text/plain");
  ASSERT_OK_AND_ASSIGN(auto data, WaitCloud(file->ReadAsync({}, 2, 3)));
  EXPECT_EQ(data->ToString(), "cde");
  ASSERT_OK(file->Close());
  ASSERT_OK_AND_ASSIGN(file, WaitCloud(fs_->OpenInputFileAsync(info)));
  ASSERT_OK_AND_ASSIGN(data, WaitCloud(file->ReadAsync({}, 0, 6)));
  EXPECT_EQ(data->ToString(), "abcdef");
  ASSERT_OK(file->Close());
  for (const bool by_info : {false, true}) {
    ASSERT_OK_AND_ASSIGN(auto stream,
                         WaitCloud(by_info ? fs_->OpenInputStreamAsync(info) : fs_->OpenInputStreamAsync(path)));
    ASSERT_OK_AND_ASSIGN(auto streamed, stream->Read(6));
    EXPECT_EQ(streamed->ToString(), "abcdef");
    ASSERT_OK(stream->Close());
  }
  ASSERT_OK_AND_ASSIGN(auto batch, WaitCloud(fs_->GetFileInfoAsync({path, "nested", "missing"})));
  ASSERT_EQ(batch.size(), 3);
  EXPECT_EQ(batch[0].type(), arrow::fs::FileType::File);
  EXPECT_EQ(batch[1].type(), arrow::fs::FileType::Directory);
  EXPECT_EQ(batch[2].type(), arrow::fs::FileType::NotFound);
  ASSERT_OK(Put(path, "replacement"));
  ASSERT_OK_AND_ASSIGN(info, Stat(path));
  EXPECT_EQ(info.size(), 11);
  ASSERT_OK(Put("empty", ""));
  ASSERT_OK_AND_ASSIGN(info, Stat("empty"));
  EXPECT_EQ(info.size(), 0);
  ASSERT_OK_AND_ASSIGN(info, Stat("missing"));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(S3NativeCloudTest, ListingAndContinuationPages) {
  for (int i = 0; i < 5; ++i) ASSERT_OK(Put("list/" + std::to_string(i), "data"));
  // Force real service pagination without creating thousands of objects.
  auto io = arrow::io::IOContext(executor_.get());
  ASSERT_OK_AND_ASSIGN(auto sdk, ClientBuilder<S3Client>(options_).BuildClient(io));
  ASSERT_OK_AND_ASSIGN(auto crt, ClientBuilder<Aws::S3Crt::S3CrtClient>(options_).BuildClient(io));
  ASSERT_OK_AND_ASSIGN(auto transport, NativeS3Transport::Make(options_, sdk, crt));
  ASSERT_OK_AND_ASSIGN(auto operations, MakeNativeS3Operations(io, transport));
  std::string token;
  std::set<std::string> keys;
  int pages = 0;
  do {
    ASSERT_LT(pages++, 10);
    ASSERT_OK_AND_ASSIGN(
        auto response, WaitCloud(operations->ListAsync(config_.bucket_name + "/" + prefix_ + "/list", token, true, 2)));
    ASSERT_TRUE(response.ToStatus().ok()) << response.ToStatus();
    auto xml = Aws::Utils::Xml::XmlDocument::CreateFromXmlString(response.body.c_str());
    ASSERT_TRUE(xml.WasParseSuccessful());
    Aws::S3::Model::ListObjectsV2Result page(
        Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>(std::move(xml), response.headers));
    EXPECT_EQ(std::string(page.GetPrefix().c_str()), prefix_ + "/list/");
    for (const auto& object : page.GetContents()) EXPECT_TRUE(keys.emplace(object.GetKey().c_str()).second);
    if (!page.GetIsTruncated())
      break;
    auto next = std::string(page.GetNextContinuationToken().c_str());
    ASSERT_FALSE(next.empty());
    ASSERT_NE(next, token);
    token = std::move(next);
  } while (true);
  EXPECT_GE(pages, 3);
  EXPECT_EQ(keys.size(), 5);
  arrow::fs::FileSelector selector;
  selector.base_dir = "list";
  selector.recursive = true;
  auto generator = fs_->GetFileInfoGenerator(selector);
  size_t count = 0;
  for (;;) {
    ASSERT_OK_AND_ASSIGN(auto page, WaitCloud(generator()));
    if (page.empty())
      break;
    count += page.size();
  }
  EXPECT_EQ(count, 5);
}

TEST_F(S3NativeCloudTest, MultipartFlushCloseAndAbort) {
  const std::string part(5 * 1024 * 1024, 'a');
  ASSERT_OK_AND_ASSIGN(auto out, fs_->OpenOutputStream("multipart"));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(out);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(out->Write(arrow::Buffer::FromString(part)));
  ASSERT_OK(WaitCloud(async->FlushAsync()));
  ASSERT_OK(out->Write(arrow::Buffer::FromString("tail")));
  ASSERT_OK(WaitCloud(out->CloseAsync()));
  ASSERT_OK_AND_ASSIGN(auto info, Stat("multipart"));
  EXPECT_EQ(info.size(), part.size() + 4);
  ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile(info));
  ASSERT_OK_AND_ASSIGN(auto bytes, WaitCloud(file->ReadAsync({}, part.size() - 2, 6)));
  EXPECT_EQ(bytes->ToString(), "aatail");
  ASSERT_OK(file->Close());
  ASSERT_OK_AND_ASSIGN(auto sized, fs_->OpenOutputStreamWithUploadSize("sized", nullptr, 6 * 1024 * 1024));
  ASSERT_NE(std::dynamic_pointer_cast<AsyncOutputStream>(sized), nullptr);
  ASSERT_OK(sized->Write(arrow::Buffer::FromString(part)));
  ASSERT_OK(WaitCloud(sized->CloseAsync()));
  ASSERT_OK_AND_ASSIGN(info, Stat("sized"));
  EXPECT_EQ(info.size(), part.size());
  for (const bool multipart : {false, true}) {
    auto path = multipart ? "abort-multipart" : "abort-buffer";
    ASSERT_OK_AND_ASSIGN(out, fs_->OpenOutputStream(path));
    async = std::dynamic_pointer_cast<AsyncOutputStream>(out);
    ASSERT_NE(async, nullptr);
    ASSERT_OK(out->Write(arrow::Buffer::FromString(multipart ? part : "buffer")));
    if (multipart)
      ASSERT_OK(WaitCloud(async->FlushAsync()));
    ASSERT_OK(WaitCloud(async->AbortAsync()));
    ASSERT_OK_AND_ASSIGN(info, Stat(path));
    EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
  }
}

TEST_F(S3NativeCloudTest, ConditionalPutAndMultipartPreserveExistingObject) {
  if (config_.cloud_provider == kCloudProviderHuawei)
    GTEST_SKIP() << "Existing OBS conditional-output API is not implemented";
  for (const auto size : {7, 5 * 1024 * 1024}) {
    auto path = "conditional-" + std::to_string(size) + " #?+% 中文";
    ASSERT_OK_AND_ASSIGN(auto out, fs_->OpenConditionalOutputStream(path, nullptr));
    ASSERT_OK(out->Write(arrow::Buffer::FromString(std::string(size, 'a'))));
    auto created = WaitCloud(out->CloseAsync());
    if (config_.cloud_provider == kCloudProviderGCP && size >= config_.multi_part_upload_size) {
      ASSERT_TRUE(created.status().IsNotImplemented()) << created.status();
      ASSERT_OK_AND_ASSIGN(auto absent, Stat(path));
      EXPECT_EQ(absent.type(), arrow::fs::FileType::NotFound);
      continue;
    }
    ASSERT_OK(created);
    for (const auto conflict_size : {7, 5 * 1024 * 1024}) {
      ASSERT_OK_AND_ASSIGN(auto conflict, fs_->OpenConditionalOutputStream(path, nullptr));
      ASSERT_OK(conflict->Write(arrow::Buffer::FromString(std::string(conflict_size, 'b'))));
      auto rejected = WaitCloud(conflict->CloseAsync());
      EXPECT_FALSE(rejected.ok());
      if (config_.cloud_provider == kCloudProviderGCP && conflict_size >= config_.multi_part_upload_size)
        EXPECT_TRUE(rejected.status().IsNotImplemented()) << rejected.status();
      auto async = std::dynamic_pointer_cast<AsyncOutputStream>(conflict);
      ASSERT_NE(async, nullptr);
      ASSERT_OK(WaitCloud(async->AbortAsync()));
      ASSERT_OK_AND_ASSIGN(auto info, Stat(path));
      EXPECT_EQ(info.size(), size);
      ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile(info));
      ASSERT_OK_AND_ASSIGN(auto bytes, WaitCloud(file->ReadAsync({}, 0, 7)));
      EXPECT_EQ(bytes->ToString(), "aaaaaaa");
      ASSERT_OK(file->Close());
    }
  }
}
}  // namespace milvus_storage
#endif
