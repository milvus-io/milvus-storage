// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include <arrow/testing/gtest_util.h>
#include <arrow/util/thread_pool.h>
#include <arrow/filesystem/localfs.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <future>
#include "milvus-storage/filesystem/async_filesystem.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

namespace milvus_storage {
template <class T>
arrow::Result<T> Await(arrow::Future<T> future) {
  return future.result();
}
TEST(NativeS3Capability, UnsupportedProviderDoesNotFallback) {
  ASSERT_OK_AND_ASSIGN(auto executor, arrow::internal::ThreadPool::Make(1));
  auto local = std::make_shared<arrow::fs::LocalFileSystem>();
  auto result = MakeAsyncFileSystem(local, arrow::io::IOContext(executor.get()));
  EXPECT_TRUE(result.status().IsNotImplemented());
}

#ifdef WITH_CRT
class NativeS3Test : public ::testing::Test {
  protected:
  static void SetUpTestSuite() { ASSERT_OK(EnsureS3Initialized()); }
  void SetUp() override {
    const char* endpoint = std::getenv("STORAGE_NATIVE_S3_ENDPOINT");
    if (!endpoint)
      GTEST_SKIP() << "Run cpp/scripts/test_native_s3.py for isolated S3 fixture";
    ASSERT_OK(EnsureS3Initialized());
    ASSERT_OK_AND_ASSIGN(executor_, arrow::internal::ThreadPool::Make(1));
    options_ = S3Options::FromAccessKey("fixture", "fixture-secret");
    options_.scheme = "http";
    options_.region = "us-east-1";
    options_.endpoint_override = endpoint;
    options_.cloud_provider = "aws";
    options_.use_crt_async_reads = true;
    options_.max_connections = 1;
    options_.allow_bucket_creation = true;
    options_.use_crc32c_checksum = true;
    ASSERT_OK_AND_ASSIGN(sync_, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get())));
    if (std::getenv("STORAGE_NATIVE_S3_REAL"))
      ASSERT_OK(sync_->CreateDir("bucket", true));
    const auto* prefix = std::getenv("STORAGE_NATIVE_S3_PREFIX");
    auto subtree = std::make_shared<FileSystemProxy>(prefix ? prefix : "bucket/root", sync_);
    ASSERT_OK_AND_ASSIGN(fs_, MakeAsyncFileSystem(subtree, arrow::io::IOContext(executor_.get())));
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> Read(const std::string& path, int64_t offset, int64_t size) {
    return sync_->OpenInputFileAsync("bucket/root/" + path).Then([this, offset, size](std::shared_ptr<arrow::io::RandomAccessFile> file) {
      auto native = std::dynamic_pointer_cast<NonBlockingRandomAccessFile>(file);
      return native->GetSizeAsync().Then([this, file, offset, size](int64_t) {
        return file->ReadAsync(arrow::io::IOContext(executor_.get()), offset, size);
      });
    });
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> Metadata(const std::string& path) {
    return sync_->OpenInputFileAsync("bucket/root/" + path).Then([this](std::shared_ptr<arrow::io::RandomAccessFile> file) {
      return file->ReadMetadataAsync(arrow::io::IOContext(executor_.get()));
    });
  }
  void TearDown() override {
    fs_.reset();
    sync_.reset();
    if (executor_ && !executor_stopped_)
      ASSERT_OK(executor_->Shutdown());
  }
  template <class T>
  void CompletesWithoutBlockingWorker(std::function<arrow::Future<T>()> start) {
    auto result = arrow::Future<T>::Make();
    std::promise<void> heartbeat;
    ASSERT_OK(executor_->Spawn([start, result] {
      start().AddCallback([result](const arrow::Result<T>& value) mutable { result.MarkFinished(value); });
    }));
    ASSERT_OK(executor_->Spawn([&heartbeat] { heartbeat.set_value(); }));
    EXPECT_EQ(heartbeat.get_future().wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
    EXPECT_FALSE(result.is_finished());
    ASSERT_TRUE(result.Wait(5));
    ASSERT_OK(result.status());
  }
  S3Options options_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  std::shared_ptr<S3FileSystem> sync_;
  std::shared_ptr<AsyncFileSystem> fs_;
  bool executor_stopped_ = false;
};

TEST_F(NativeS3Test, HeadAndReadUseCallerExecutorWithoutNetworkWait) {
  CompletesWithoutBlockingWorker<arrow::fs::FileInfo>([this] { return fs_->GetFileInfoAsync("slow"); });
  auto subtree = std::make_shared<arrow::fs::SubTreeFileSystem>("", fs_);
  auto view = std::make_shared<FileSystemProxy>("", subtree);
  CompletesWithoutBlockingWorker<arrow::fs::FileInfoVector>(
      [view] { return view->GetFileInfoAsync(std::vector<std::string>{"slow"}); });
  CompletesWithoutBlockingWorker<std::shared_ptr<arrow::Buffer>>([this] { return Read("slow", 0, 3); });
}

TEST_F(NativeS3Test, MetadataRangesAndSubtreePaths) {
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs_->GetFileInfoAsync("hello #?+% 中文")));
  EXPECT_EQ(info.path(), "hello #?+% 中文");
  EXPECT_EQ(info.size(), 6);
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("hello #?+% 中文", 2, 20)));
  EXPECT_EQ(data->ToString(), "cdef");
  ASSERT_OK_AND_ASSIGN(auto metadata, Await(Metadata("hello #?+% 中文")));
  ASSERT_OK_AND_ASSIGN(auto size, metadata->Get("Content-Length"));
  EXPECT_EQ(size, "6");
  EXPECT_TRUE(Read("slow", -1, 1).status().IsInvalid());
  ASSERT_OK_AND_ASSIGN(auto empty, Await(Read("hello #?+% 中文", 6, 1)));
  EXPECT_EQ(empty->size(), 0);
  EXPECT_FALSE(Read("hello #?+% 中文", 7, 1).status().ok());
}

TEST_F(NativeS3Test, ListPaginatesAndDetectsMissingDirectory) {
  arrow::fs::FileSelector select;
  select.base_dir = "pages";
  select.recursive = true;
  auto generator = fs_->GetFileInfoGenerator(select);
  std::vector<std::string> paths;
  for (;;) {
    ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
    if (page.empty())
      break;
    for (auto& info : page) paths.push_back(info.path());
  }
  EXPECT_EQ(paths, (std::vector<std::string>{"pages/0", "pages/1", "pages/2", "pages/3", "pages/4"}));
  ASSERT_OK_AND_ASSIGN(auto dir, Await(fs_->GetFileInfoAsync("pages")));
  EXPECT_EQ(dir.type(), arrow::fs::FileType::Directory);
  ASSERT_OK_AND_ASSIGN(auto missing, Await(fs_->GetFileInfoAsync("missing")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(NativeS3Test, ListSubmissionDoesNotBlockWorker) {
  arrow::fs::FileSelector select;
  select.base_dir = "slow-list";
  auto generator = fs_->GetFileInfoGenerator(select);
  CompletesWithoutBlockingWorker<arrow::fs::FileInfoVector>(generator);
}

TEST_F(NativeS3Test, HttpErrorsAndMalformedPagination) {
  EXPECT_FALSE(fs_->GetFileInfoAsync("denied").status().ok());
  EXPECT_FALSE(Read("denied", 0, 1).status().ok());
  arrow::fs::FileSelector select;
  select.base_dir = "bad-token";
  auto generator = fs_->GetFileInfoGenerator(select);
  EXPECT_FALSE(generator().status().ok());
}

TEST_F(NativeS3Test, RetainsRequestAfterFilesystemIsReleased) {
  auto future = Read("slow", 0, 6);
  fs_.reset();
  sync_.reset();
  ASSERT_TRUE(future.Wait(5));
  ASSERT_OK_AND_ASSIGN(auto data, future.result());
  EXPECT_EQ(data->ToString(), "abcdef");
}

TEST_F(NativeS3Test, RejectedCompletionExecutorPreservesResult) {
  // Shutdown before submission forces the completion-dispatch fallback.
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs_->GetFileInfoAsync("hello #?+% 中文")));
  EXPECT_EQ(info.size(), 6);
}

TEST_F(NativeS3Test, RejectedCompletionCanReleaseTheLastNativeTransportOwner) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  auto pending = fs_->GetFileInfoAsync("slow");
  fs_.reset();
  sync_.reset();
  ASSERT_TRUE(pending.Wait(5));
  ASSERT_OK_AND_ASSIGN(auto info, pending.result());
  EXPECT_EQ(info.size(), 6);
}

TEST_F(NativeS3Test, RejectedBatchCompletionDoesNotRetainTheSdkCrtHolder) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  auto pending = fs_->GetFileInfoAsync(std::vector<std::string>{"slow", "hello #?+% 中文"});
  fs_.reset();
  sync_.reset();
  ASSERT_TRUE(pending.Wait(5));
  ASSERT_OK_AND_ASSIGN(auto infos, pending.result());
  ASSERT_EQ(infos.size(), 2);
  EXPECT_EQ(infos[0].size(), 6);
  EXPECT_EQ(infos[1].size(), 6);
}
TEST_F(NativeS3Test, RootListingSkipsEmptyContinuationPage) {
  ASSERT_OK_AND_ASSIGN(auto root, MakeAsyncFileSystem(sync_, arrow::io::IOContext(executor_.get())));
  arrow::fs::FileSelector selector;
  auto generator = root->GetFileInfoGenerator(selector);
  ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
  ASSERT_EQ(page.size(), 1);
  EXPECT_EQ(page.front().path(), "bucket");
  ASSERT_OK_AND_ASSIGN(page, Await(generator()));
  EXPECT_TRUE(page.empty());
  EXPECT_TRUE(root->GetFileInfoAsync("bucket/../file").status().IsInvalid());
  EXPECT_TRUE(fs_->GetFileInfoAsync("../file").status().IsInvalid());
}

TEST_F(NativeS3Test, WritesConditionalMetadataAndOwnedBuffer) {
  AsyncWriteOptions options;
  options.if_absent = true;
  options.metadata = arrow::key_value_metadata({"content-type", "x-amz-meta-test"}, {"text/plain", "kept"});
  auto pending = fs_->WriteAsync("write #?+% 中文", arrow::Buffer::FromString("hello"), options);
  ASSERT_OK_AND_ASSIGN(auto version, Await(pending));
  EXPECT_FALSE(version.etag.empty());
  ASSERT_OK_AND_ASSIGN(auto metadata, Await(fs_->ReadMetadataAsync("write #?+% 中文")));
  ASSERT_OK_AND_ASSIGN(auto value, metadata->Get("x-amz-meta-test"));
  EXPECT_EQ(value, "kept");
  EXPECT_FALSE(fs_->WriteAsync("write #?+% 中文", arrow::Buffer::FromString("bad"), options).status().ok());
  options.if_absent = false;
  options.if_match = version.etag;
  ASSERT_OK(fs_->WriteAsync("write #?+% 中文", arrow::Buffer::FromString("updated"), options).status());
  options.if_match = "\"wrong-etag\"";
  EXPECT_FALSE(fs_->WriteAsync("write #?+% 中文", arrow::Buffer::FromString("bad"), options).status().ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(fs_->ReadAsync("write #?+% 中文", 0, 20)));
  EXPECT_EQ(data->ToString(), "updated");
  ASSERT_OK(fs_->WriteAsync("empty-write", arrow::Buffer::FromString("")).status());
}
TEST_F(NativeS3Test, WritesAndMultipartUseNativeRequests) {
  CompletesWithoutBlockingWorker<AsyncObjectVersion>(
      [this] { return fs_->WriteAsync("slow-write", arrow::Buffer::FromString("owned")); });
}
TEST_F(NativeS3Test, MultipartCompleteAndAbort) {
  ASSERT_OK_AND_ASSIGN(auto id, Await(fs_->CreateMultipartUploadAsync("multipart")));
  ASSERT_OK_AND_ASSIGN(
      auto first,
      Await(fs_->UploadPartAsync("multipart", id, 1, arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a')))));
  ASSERT_OK_AND_ASSIGN(auto last, Await(fs_->UploadPartAsync("multipart", id, 2, arrow::Buffer::FromString("tail"))));
  EXPECT_FALSE(fs_->CompleteMultipartUploadAsync("multipart", id, {last, first}).status().ok());
  ASSERT_OK_AND_ASSIGN(auto version, Await(fs_->CompleteMultipartUploadAsync("multipart", id, {first, last})));
  EXPECT_FALSE(version.etag.empty());
  ASSERT_OK_AND_ASSIGN(auto data, Await(fs_->ReadAsync("multipart", 5 * 1024 * 1024 - 2, 10)));
  EXPECT_EQ(data->ToString(), "aatail");
  ASSERT_OK_AND_ASSIGN(auto abort_id, Await(fs_->CreateMultipartUploadAsync("aborted")));
  ASSERT_OK(fs_->AbortMultipartUploadAsync("aborted", abort_id).status());
  EXPECT_FALSE(fs_->UploadPartAsync("aborted", abort_id, 1, arrow::Buffer::FromString("bad")).status().ok());
}
TEST_F(NativeS3Test, MultipartEmbeddedErrorIsNotSuccess) {
  ASSERT_OK_AND_ASSIGN(auto id, Await(fs_->CreateMultipartUploadAsync("error-complete")));
  ASSERT_OK_AND_ASSIGN(auto part,
                       Await(fs_->UploadPartAsync("error-complete", id, 1, arrow::Buffer::FromString("data"))));
  EXPECT_FALSE(fs_->CompleteMultipartUploadAsync("error-complete", id, {part}).status().ok());
  ASSERT_OK(fs_->AbortMultipartUploadAsync("error-complete", id).status());
}
TEST_F(NativeS3Test, RejectedCompletionPreservesSuccessfulWrite) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  ASSERT_OK_AND_ASSIGN(auto version, Await(fs_->WriteAsync("rejected-dispatch", arrow::Buffer::FromString("durable"))));
  EXPECT_FALSE(version.etag.empty());
  ASSERT_OK_AND_ASSIGN(auto data, Await(fs_->ReadAsync("rejected-dispatch", 0, 7)));
  EXPECT_EQ(data->ToString(), "durable");
}
#endif
}  // namespace milvus_storage
