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
    options_.use_crt_async_reads = false;
    ASSERT_OK_AND_ASSIGN(sync_, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get())));
    auto subtree = std::make_shared<FileSystemProxy>("bucket/root", sync_);
    ASSERT_OK_AND_ASSIGN(fs_, MakeAsyncFileSystem(subtree, arrow::io::IOContext(executor_.get())));
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
  CompletesWithoutBlockingWorker<std::shared_ptr<arrow::Buffer>>([this] { return fs_->ReadAsync("slow", 0, 3); });
}

TEST_F(NativeS3Test, MetadataRangesAndSubtreePaths) {
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs_->GetFileInfoAsync("hello #?+% 中文")));
  EXPECT_EQ(info.path(), "hello #?+% 中文");
  EXPECT_EQ(info.size(), 6);
  ASSERT_OK_AND_ASSIGN(auto data, Await(fs_->ReadAsync("hello #?+% 中文", 2, 20)));
  EXPECT_EQ(data->ToString(), "cdef");
  ASSERT_OK_AND_ASSIGN(auto metadata, Await(fs_->ReadMetadataAsync("hello #?+% 中文")));
  ASSERT_OK_AND_ASSIGN(auto size, metadata->Get("content-length"));
  EXPECT_EQ(size, "6");
  EXPECT_TRUE(fs_->ReadAsync("slow", -1, 1).status().IsInvalid());
  ASSERT_OK_AND_ASSIGN(auto empty, Await(fs_->ReadAsync("hello #?+% 中文", 6, 1)));
  EXPECT_EQ(empty->size(), 0);
  EXPECT_FALSE(fs_->ReadAsync("hello #?+% 中文", 7, 1).status().ok());
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
  EXPECT_FALSE(fs_->ReadAsync("denied", 0, 1).status().ok());
  arrow::fs::FileSelector select;
  select.base_dir = "bad-token";
  auto generator = fs_->GetFileInfoGenerator(select);
  EXPECT_FALSE(generator().status().ok());
}

TEST_F(NativeS3Test, RetainsRequestAfterFilesystemIsReleased) {
  auto future = fs_->ReadAsync("slow", 0, 6);
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
TEST_F(NativeS3Test, RootListingSkipsEmptyContinuationPage) {
  ASSERT_OK_AND_ASSIGN(auto root, MakeAsyncFileSystem(sync_, arrow::io::IOContext(executor_.get())));
  arrow::fs::FileSelector selector;
  auto generator = root->GetFileInfoGenerator(selector);
  ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
  ASSERT_EQ(page.size(), 1);
  EXPECT_EQ(page.front().path(), "bucket");
  ASSERT_OK_AND_ASSIGN(page, Await(generator()));
  EXPECT_TRUE(page.empty());
  EXPECT_TRUE(root->ReadAsync("bucket/../file", 0, 1).status().IsInvalid());
  EXPECT_TRUE(fs_->GetFileInfoAsync("../file").status().IsInvalid());
}
#endif
}  // namespace milvus_storage
