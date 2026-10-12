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
    ASSERT_TRUE(response.IsSuccess()) << response.GetError();
    Aws::S3::Model::ListObjectsV2Result page(response.GetResult());
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

// Use only legacy synchronous entry points, with both runtime backends.
class S3SyncCloudTest : public ::testing::TestWithParam<bool> {
  protected:
  void SetUp() override {
    if (!IsCloudEnv() || GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("") == "azure")
      GTEST_SKIP() << "Requires a configured S3-compatible cloud";
    api::Properties properties;
    ASSERT_OK(InitTestProperties(properties));
    ASSERT_OK_AND_ASSIGN(config_, GetFileSystemConfig(properties));
    config_.s3_crt_async_read = GetParam();
    config_.multi_part_upload_size = 5 * 1024 * 1024;
    config_.max_connections = 4;
    config_.request_timeout_ms = 30000;
    config_.log_level = "off";
    ASSERT_OK_AND_ASSIGN(parent_, CreateArrowFileSystem(config_));
    prefix_ = "pr693-sync-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
              std::to_string(std::random_device{}());
    proxy_ = std::make_shared<FileSystemProxy>(prefix_, parent_);
    fs_ = proxy_;
    std::cout << "Sync cloud test prefix: " << prefix_ << " crt=" << GetParam() << std::endl;
  }
  void TearDown() override {
    if (parent_ && !prefix_.empty()) {
      auto status = parent_->DeleteDirContents(prefix_, true);
      EXPECT_TRUE(status.ok()) << status;
    }
  }
  arrow::Status Put(const std::string& path, const std::string& value) {
    ARROW_ASSIGN_OR_RAISE(auto out, fs_->OpenOutputStream(path));
    ARROW_RETURN_NOT_OK(out->Write(value.data(), value.size()));
    return out->Close();
  }
  ArrowFileSystemConfig config_;
  FileSystemPtr parent_, proxy_;
  ArrowFileSystemPtr fs_;
  std::string prefix_;
};

TEST_P(S3SyncCloudTest, ReadWriteMetadataAndClosedHandles) {
  const std::string path = "nested/hello #?+% 中文";
  auto metadata = arrow::key_value_metadata({"Content-Type"}, {"text/plain"});
  ASSERT_OK_AND_ASSIGN(auto out, fs_->OpenOutputStream(path, metadata));
  ASSERT_OK(out->Write("abc", 3));
  ASSERT_OK(out->Write(arrow::Buffer::FromString("def")));
  ASSERT_OK_AND_ASSIGN(auto position, out->Tell());
  EXPECT_EQ(position, 6);
  ASSERT_OK(out->Flush());
  ASSERT_OK(out->Close());
  ASSERT_OK(out->Close());
  EXPECT_TRUE(out->closed());
  EXPECT_FALSE(out->Write("x", 1).ok());
  EXPECT_FALSE(out->Flush().ok());
  ASSERT_OK_AND_ASSIGN(auto info, fs_->GetFileInfo(path));
  EXPECT_EQ(info.size(), 6);
  for (const bool by_info : {false, true}) {
    ASSERT_OK_AND_ASSIGN(auto file, by_info ? fs_->OpenInputFile(info) : fs_->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());
    EXPECT_EQ(size, 6);
    ASSERT_OK_AND_ASSIGN(auto headers, file->ReadMetadata());
    ASSERT_OK_AND_ASSIGN(auto content_type, headers->Get("Content-Type"));
    EXPECT_EQ(content_type, "text/plain");
    ASSERT_OK_AND_ASSIGN(auto range, file->ReadAt(2, 3));
    EXPECT_EQ(range->ToString(), "cde");
    ASSERT_OK(file->Seek(4));
    ASSERT_OK_AND_ASSIGN(auto tail, file->Read(10));
    EXPECT_EQ(tail->ToString(), "ef");
    ASSERT_OK(file->Close());
    EXPECT_FALSE(file->Read(1).ok());
    ASSERT_OK_AND_ASSIGN(auto stream, by_info ? fs_->OpenInputStream(info) : fs_->OpenInputStream(path));
    ASSERT_OK_AND_ASSIGN(auto all, stream->Read(6));
    EXPECT_EQ(all->ToString(), "abcdef");
    ASSERT_OK(stream->Close());
  }
  auto other_config = config_;
  other_config.s3_crt_async_read = !GetParam();
  ASSERT_OK_AND_ASSIGN(auto other, CreateArrowFileSystem(other_config));
  ASSERT_OK_AND_ASSIGN(auto input, other->OpenInputFile(prefix_ + "/" + path));
  ASSERT_OK_AND_ASSIGN(auto data, input->Read(6));
  EXPECT_EQ(data->ToString(), "abcdef");
  ASSERT_OK(input->Close());
  ASSERT_OK(Put(path, "replacement"));
  ASSERT_OK_AND_ASSIGN(info, fs_->GetFileInfo(path));
  EXPECT_EQ(info.size(), 11);
  ASSERT_OK(Put("empty", ""));
  ASSERT_OK_AND_ASSIGN(info, fs_->GetFileInfo("empty"));
  EXPECT_EQ(info.size(), 0);
  ASSERT_OK_AND_ASSIGN(auto batch, fs_->GetFileInfo(std::vector<std::string>{path, "nested", "missing"}));
  ASSERT_EQ(batch.size(), 3);
  EXPECT_EQ(batch[0].type(), arrow::fs::FileType::File);
  EXPECT_EQ(batch[1].type(), arrow::fs::FileType::Directory);
  EXPECT_EQ(batch[2].type(), arrow::fs::FileType::NotFound);
  // Opening is lazy; missing-object errors must surface when performing I/O.
  auto missing = fs_->OpenInputFile("missing");
  if (missing.ok()) {
    EXPECT_FALSE((*missing)->GetSize().ok());
    EXPECT_FALSE((*missing)->ReadAt(0, 1).ok());
    ASSERT_OK((*missing)->Close());
  }
}

TEST_P(S3SyncCloudTest, ListingCopyMoveAndDirectoryCleanup) {
  ASSERT_OK(fs_->CreateDir("dir/sub", true));
  ASSERT_OK(Put("dir/a", "first"));
  ASSERT_OK(Put("dir/sub/b", "second"));
  arrow::fs::FileSelector selector;
  selector.base_dir = "dir";
  selector.recursive = true;
  ASSERT_OK_AND_ASSIGN(auto listed, fs_->GetFileInfo(selector));
  std::set<std::string> files;
  for (const auto& info : listed)
    if (info.IsFile())
      files.insert(info.path());
  EXPECT_EQ(files, (std::set<std::string>{"dir/a", "dir/sub/b"}));
  ASSERT_OK(fs_->CopyFile("dir/a", "dir/copy"));
  ASSERT_OK(fs_->Move("dir/copy", "dir/moved"));
  ASSERT_OK_AND_ASSIGN(auto info, fs_->GetFileInfo("dir/copy"));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile("dir/moved"));
  ASSERT_OK_AND_ASSIGN(auto bytes, file->Read(10));
  EXPECT_EQ(bytes->ToString(), "first");
  ASSERT_OK(file->Close());
  ASSERT_OK(fs_->DeleteFile("dir/moved"));
  ASSERT_OK(fs_->DeleteDirContents("dir", false));
  ASSERT_OK_AND_ASSIGN(listed, fs_->GetFileInfo(selector));
  EXPECT_TRUE(listed.empty());
  ASSERT_OK(fs_->DeleteDir("dir"));
  ASSERT_OK_AND_ASSIGN(info, fs_->GetFileInfo("dir"));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_P(S3SyncCloudTest, MultipartFlushCloseAndAbort) {
  const std::string part(5 * 1024 * 1024, 'a');
  ASSERT_OK_AND_ASSIGN(auto out, proxy_->OpenOutputStreamWithUploadSize("multipart", nullptr, part.size()));
  ASSERT_OK(out->Write(arrow::Buffer::FromString(part)));
  ASSERT_OK(out->Flush());
  ASSERT_OK(out->Write("tail", 4));
  ASSERT_OK(out->Close());
  ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile("multipart"));
  ASSERT_OK_AND_ASSIGN(auto bytes, file->Read(part.size() + 4));
  EXPECT_EQ(bytes->ToString(), part + "tail");
  ASSERT_OK(file->Close());
  for (const bool multipart : {false, true}) {
    const auto path = multipart ? "abort-multipart" : "abort-buffer";
    ASSERT_OK_AND_ASSIGN(out, fs_->OpenOutputStream(path));
    ASSERT_OK(out->Write(arrow::Buffer::FromString(multipart ? part : "buffer")));
    ASSERT_OK(out->Flush());
    ASSERT_OK(out->Abort());
    EXPECT_TRUE(out->closed());
    ASSERT_OK_AND_ASSIGN(auto info, fs_->GetFileInfo(path));
    EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
  }
}

TEST_P(S3SyncCloudTest, ConditionalWritesAndProviderLimits) {
  if (config_.cloud_provider == kCloudProviderHuawei) {
    EXPECT_TRUE(proxy_->OpenConditionalOutputStream("conditional", nullptr).status().IsNotImplemented());
    return;
  }
  for (const auto size : {7, 5 * 1024 * 1024}) {
    // GCP conditional multipart is outside the successful compatibility matrix.
    // Check the native path's explicit rejection without claiming SDK support.
    if (config_.cloud_provider == kCloudProviderGCP && !GetParam() && size > 7)
      continue;
    const auto path = "conditional-" + std::to_string(size) + " #?+% 中文";
    if (config_.cloud_provider == kCloudProviderGCP && size > 7)
      ASSERT_OK(Put(path, "original"));
    ASSERT_OK_AND_ASSIGN(auto out, proxy_->OpenConditionalOutputStream(path, nullptr));
    ASSERT_OK(out->Write(arrow::Buffer::FromString(std::string(size, 'a'))));
    auto status = out->Close();
    if (config_.cloud_provider == kCloudProviderGCP && size > 7) {
      ASSERT_TRUE(status.IsNotImplemented()) << status;
      ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile(path));
      ASSERT_OK_AND_ASSIGN(auto bytes, file->Read(16));
      EXPECT_EQ(bytes->ToString(), "original");
      ASSERT_OK(file->Close());
      continue;
    }
    ASSERT_OK(status);
    ASSERT_OK_AND_ASSIGN(auto conflict, proxy_->OpenConditionalOutputStream(path, nullptr));
    auto rejected = conflict->Write(arrow::Buffer::FromString(std::string(size, 'b')));
    const bool rejected_on_write = !rejected.ok();
    if (!rejected_on_write)
      rejected = conflict->Close();
    EXPECT_FALSE(rejected.ok());
    std::cout << "Conditional conflict reported by " << (rejected_on_write ? "Write" : "Close") << std::endl;
    ASSERT_OK(conflict->Abort());
    ASSERT_OK_AND_ASSIGN(auto file, fs_->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto bytes, file->Read(size));
    EXPECT_EQ(bytes->ToString(), std::string(size, 'a'));
    ASSERT_OK(file->Close());
  }
}

INSTANTIATE_TEST_SUITE_P(SdkAndCrt,
                         S3SyncCloudTest,
                         ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Crt" : "Sdk"; });
}  // namespace milvus_storage
#endif
