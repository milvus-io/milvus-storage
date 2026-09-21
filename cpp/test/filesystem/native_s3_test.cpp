// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include <arrow/testing/gtest_util.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <future>
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/async_output_stream.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

namespace milvus_storage {
template <class T>
arrow::Result<T> Await(arrow::Future<T> future) {
  return future.result();
}
#ifdef WITH_CRT
// Single-object assertions use the existing Arrow stat API with one path.
arrow::Future<arrow::fs::FileInfo> Stat(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::string& path) {
  return fs->GetFileInfoAsync(std::vector<std::string>{path})
      .Then([](arrow::fs::FileInfoVector infos) -> arrow::Result<arrow::fs::FileInfo> {
        if (infos.size() != 1)
          return arrow::Status::Invalid("Expected one stat result");
        return std::move(infos.front());
      });
}

class NativeS3Test : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    ArrowFileSystemConfig config;
    ASSERT_OK(S3FileSystemProducer(config).InitS3());
  }
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
    options_.multi_part_upload_size = 5 * 1024 * 1024;
    ASSERT_OK_AND_ASSIGN(sync_, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get())));
    if (std::getenv("STORAGE_NATIVE_S3_REAL"))
      ASSERT_OK(sync_->CreateDir("bucket", true));
    const auto* prefix = std::getenv("STORAGE_NATIVE_S3_PREFIX");
    prefix_ = prefix ? prefix : "bucket/root";
    fs_ = std::make_shared<FileSystemProxy>(prefix_, sync_);
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> Read(const std::string& path, int64_t offset, int64_t size) {
    return fs_->OpenInputFileAsync(path).Then([this, offset, size](std::shared_ptr<arrow::io::RandomAccessFile> file) {
      auto native = std::dynamic_pointer_cast<NonBlockingRandomAccessFile>(file);
      return native->GetSizeAsync().Then([this, file, offset, size](int64_t) {
        return file->ReadAsync(arrow::io::IOContext(executor_.get()), offset, size);
      });
    });
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> Metadata(const std::string& path) {
    return fs_->OpenInputFileAsync(path).Then([this](std::shared_ptr<arrow::io::RandomAccessFile> file) {
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
    ASSERT_OK(executor_->Spawn([start, result]() mutable {
      start().AddCallback([result](const arrow::Result<T>& value) mutable { result.MarkFinished(value); });
    }));
    ASSERT_OK(executor_->Spawn([&heartbeat] { heartbeat.set_value(); }));
    EXPECT_EQ(heartbeat.get_future().wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
    EXPECT_FALSE(result.is_finished());
    ASSERT_TRUE(result.Wait(5));
    ASSERT_OK(result.status());
  }
  arrow::Future<> Write(const std::string& path,
                        std::shared_ptr<arrow::Buffer> data,
                        std::shared_ptr<const arrow::KeyValueMetadata> metadata = nullptr) {
    ARROW_ASSIGN_OR_RAISE(auto stream, fs_->OpenOutputStream(path, metadata));
    ARROW_RETURN_NOT_OK(stream->Write(data));
    return stream->CloseAsync();
  }
  std::string prefix_ = "bucket/root";
  S3Options options_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  std::shared_ptr<S3FileSystem> sync_;
  std::shared_ptr<FileSystemProxy> fs_;
  bool executor_stopped_ = false;
};

TEST_F(NativeS3Test, SameInstanceSupportsSyncAndArrowAsyncCalls) {
  ASSERT_OK_AND_ASSIGN(auto before, fs_->GetFileInfo("hello #?+% 中文"));
  ASSERT_OK_AND_ASSIGN(auto after, Await(Stat(fs_, "hello #?+% 中文")));
  EXPECT_EQ(before, after);
  std::shared_ptr<arrow::fs::FileSystem> arrow_fs = fs_;
  ASSERT_OK_AND_ASSIGN(auto batch,
                       Await(arrow_fs->GetFileInfoAsync(std::vector<std::string>{"hello #?+% 中文", "missing"})));
  ASSERT_EQ(batch.size(), 2);
  EXPECT_EQ(batch[0], after);
  EXPECT_EQ(batch[1].type(), arrow::fs::FileType::NotFound);
  auto nested = std::make_shared<FileSystemProxy>("pages", fs_);
  ASSERT_OK_AND_ASSIGN(auto nested_info, Await(Stat(nested, "a")));
  EXPECT_EQ(nested_info.path(), "a");
}

TEST_F(NativeS3Test, FactoryAndCacheReturnTheSameSyncAsyncHandle) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = "aws";
  config.bucket_name = "bucket";
  config.address = options_.endpoint_override;
  config.access_key_id = "fixture";
  config.access_key_value = "fixture-secret";
  config.region = "us-east-1";
  ASSERT_OK_AND_ASSIGN(auto fs, CreateArrowFileSystem(config));
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs, "root/hello #?+% 中文")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::File);
  ASSERT_OK_AND_ASSIGN(auto sync_info, fs->GetFileInfo(info.path()));
  EXPECT_EQ(info, sync_info);
  api::Properties properties;
  properties[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  properties[PROPERTY_FS_ADDRESS] = options_.endpoint_override;
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  properties[PROPERTY_FS_ACCESS_KEY_ID] = std::string("fixture");
  properties[PROPERTY_FS_ACCESS_KEY_VALUE] = std::string("fixture-secret");
  properties[PROPERTY_FS_REGION] = std::string("us-east-1");
  properties[PROPERTY_FS_USE_SSL] = false;
  auto& cache = FilesystemCache::getInstance();
  ASSERT_OK_AND_ASSIGN(auto cached, cache.get(properties));
  ASSERT_OK_AND_ASSIGN(auto cached_again, cache.get(properties));
  EXPECT_EQ(cached.get(), cached_again.get());
  ASSERT_OK_AND_ASSIGN(auto cached_info, Await(Stat(cached, info.path())));
  EXPECT_EQ(cached_info, info);
  cache.clean();
}

TEST_F(NativeS3Test, HeadAndReadUseCallerExecutorWithoutNetworkWait) {
  CompletesWithoutBlockingWorker<arrow::fs::FileInfo>([this] { return Stat(fs_, "slow"); });
  auto subtree = std::make_shared<arrow::fs::SubTreeFileSystem>("", fs_);
  auto view = std::make_shared<FileSystemProxy>("", subtree);
  CompletesWithoutBlockingWorker<arrow::fs::FileInfoVector>(
      [view] { return view->GetFileInfoAsync(std::vector<std::string>{"slow"}); });
  CompletesWithoutBlockingWorker<std::shared_ptr<arrow::Buffer>>([this] { return Read("slow", 0, 3); });
}

TEST_F(NativeS3Test, MetadataRangesAndSubtreePaths) {
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "hello #?+% 中文")));
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
  ASSERT_OK_AND_ASSIGN(auto dir, Await(Stat(fs_, "pages")));
  EXPECT_EQ(dir.type(), arrow::fs::FileType::Directory);
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "missing")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(NativeS3Test, ListSubmissionDoesNotBlockWorker) {
  arrow::fs::FileSelector select;
  select.base_dir = "slow-list";
  auto generator = fs_->GetFileInfoGenerator(select);
  CompletesWithoutBlockingWorker<arrow::fs::FileInfoVector>(generator);
}

TEST_F(NativeS3Test, HttpErrorsAndMalformedPagination) {
  EXPECT_FALSE(Stat(fs_, "denied").status().ok());
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
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "hello #?+% 中文")));
  EXPECT_EQ(info.size(), 6);
}

TEST_F(NativeS3Test, RejectedCompletionCanReleaseTheLastNativeTransportOwner) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  auto pending = Stat(fs_, "slow");
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
  auto root = sync_;
  arrow::fs::FileSelector selector;
  auto generator = root->GetFileInfoGenerator(selector);
  ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
  ASSERT_EQ(page.size(), 1);
  EXPECT_EQ(page.front().path(), "bucket");
  ASSERT_OK_AND_ASSIGN(page, Await(generator()));
  EXPECT_TRUE(page.empty());
  EXPECT_TRUE(Stat(root, "bucket/../file").status().IsInvalid());
  EXPECT_TRUE(Stat(fs_, "../file").status().IsInvalid());
}

TEST_F(NativeS3Test, ExistingOutputFactoriesOpenWithoutIo) {
  auto options = options_;
  options.endpoint_override = "127.0.0.1:1";
  ASSERT_OK_AND_ASSIGN(auto s3, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
  auto proxy = std::make_shared<FileSystemProxy>("bucket/root", s3);
  std::shared_ptr<arrow::fs::FileSystem> fs = proxy;
  ASSERT_OK_AND_ASSIGN(auto stream, fs->OpenOutputStream("no-network"));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(async->FlushAsync().status());
  ASSERT_OK(async->AbortAsync().status());
  ASSERT_OK_AND_ASSIGN(auto conditional, proxy->OpenConditionalOutputStream("no-network", nullptr));
  auto conditional_async = std::dynamic_pointer_cast<AsyncOutputStream>(conditional);
  ASSERT_NE(conditional_async, nullptr);
  ASSERT_OK(conditional_async->AbortAsync().status());
  // The sized factory uses its argument, not the default part size in options.
  EXPECT_TRUE(s3->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1).status().IsInvalid());
  ASSERT_OK_AND_ASSIGN(auto sized, s3->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 6 * 1024 * 1024));
  auto sized_async = std::dynamic_pointer_cast<AsyncOutputStream>(sized);
  ASSERT_NE(sized_async, nullptr);
  ASSERT_OK(sized_async->AbortAsync().status());
}

TEST_F(NativeS3Test, ExistingConditionalOutputStreamUsesNativeWrites) {
  CompletesWithoutBlockingWorker<arrow::internal::Empty>([this]() -> arrow::Future<> {
    ARROW_ASSIGN_OR_RAISE(auto stream, fs_->OpenConditionalOutputStream("slow-conditional", nullptr));
    ARROW_RETURN_NOT_OK(stream->Write(arrow::Buffer::FromString("original")));
    return stream->CloseAsync();
  });
  ASSERT_OK_AND_ASSIGN(auto conflicting, fs_->OpenConditionalOutputStream("slow-conditional", nullptr));
  ASSERT_OK(conflicting->Write(arrow::Buffer::FromString("replacement")));
  EXPECT_FALSE(conflicting->CloseAsync().status().ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("slow-conditional", 0, 20)));
  EXPECT_EQ(data->ToString(), "original");
}

TEST_F(NativeS3Test, AsyncWriteAndSyncReadShareOneHandle) {
  ASSERT_OK(Write("same-handle", arrow::Buffer::FromString("data")).status());
  ASSERT_OK_AND_ASSIGN(auto reader, fs_->OpenInputFile("same-handle"));
  ASSERT_OK_AND_ASSIGN(auto data, reader->Read(4));
  EXPECT_EQ(data->ToString(), "data");
  ASSERT_OK_AND_ASSIGN(auto info, fs_->GetFileInfo("same-handle"));
  ASSERT_OK_AND_ASSIGN(auto async_info, Await(Stat(fs_, "same-handle")));
  EXPECT_EQ(info, async_info);
}

TEST_F(NativeS3Test, WritesConditionalMetadataAndOwnedBuffer) {
  EXPECT_FALSE(fs_->OpenOutputStream("file/").status().ok());
  auto metadata = arrow::key_value_metadata({"Content-Type", "test", "If-None-Match"}, {"text/plain", "kept", "*"});
  ASSERT_OK(Write("write #?+% 中文", arrow::Buffer::FromString("hello"), metadata).status());
  ASSERT_OK_AND_ASSIGN(auto read_metadata, Await(Metadata("write #?+% 中文")));
  ASSERT_OK_AND_ASSIGN(auto value, read_metadata->Get("test"));
  EXPECT_EQ(value, "kept");
  ASSERT_OK_AND_ASSIGN(auto type, read_metadata->Get("Content-Type"));
  EXPECT_EQ(type, "text/plain");
  EXPECT_FALSE(Write("write #?+% 中文", arrow::Buffer::FromString("bad"), metadata).status().ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("write #?+% 中文", 0, 20)));
  EXPECT_EQ(data->ToString(), "hello");
  ASSERT_OK(Write("empty-write", arrow::Buffer::FromString("")).status());
}
TEST_F(NativeS3Test, WritesAndMultipartUseNativeRequests) {
  CompletesWithoutBlockingWorker<std::shared_ptr<arrow::Buffer>>([this] {
    return Write("slow-write", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a') + "tail")).Then([] {
      return arrow::Buffer::FromString("done");
    });
  });
}
TEST_F(NativeS3Test, MultipartCompleteAndAbort) {
  ASSERT_OK(Write("multipart", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a') + "tail")).status());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("multipart", 5 * 1024 * 1024 - 2, 10)));
  EXPECT_EQ(data->ToString(), "aatail");
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("aborted"));
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(async->AbortAsync().status());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "aborted")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}
TEST_F(NativeS3Test, MultipartConditionalConflictPreservesObject) {
  ASSERT_OK(Write("multipart-conflict", arrow::Buffer::FromString("original")).status());
  auto condition = arrow::key_value_metadata({"If-None-Match"}, {"*"});
  EXPECT_FALSE(Write("multipart-conflict", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'x')), condition)
                   .status()
                   .ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("multipart-conflict", 0, 20)));
  EXPECT_EQ(data->ToString(), "original");
}
TEST_F(NativeS3Test, MultipartEmbeddedErrorIsNotSuccess) {
  EXPECT_FALSE(Write("error-complete", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))).status().ok());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "error-complete")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}
TEST_F(NativeS3Test, RejectedCompletionPreservesSuccessfulWrite) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  ASSERT_OK(Write("rejected-dispatch", arrow::Buffer::FromString("durable")).status());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("rejected-dispatch", 0, 7)));
  EXPECT_EQ(data->ToString(), "durable");
}
TEST_F(NativeS3Test, StreamBackpressureDoesNotConsumeRejectedWrite) {
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("backpressure"));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  EXPECT_TRUE(stream->Write(arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'x'))).IsCapacityError());
  ASSERT_OK_AND_ASSIGN(auto offset, stream->Tell());
  EXPECT_EQ(offset, 0);
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  ASSERT_OK(async->FlushAsync().status());
  ASSERT_OK(stream->Write(arrow::Buffer::FromString("tail")));
  ASSERT_OK(stream->CloseAsync().status());
}

TEST_F(NativeS3Test, DirectoryContentsRetainsExistingSdkBehavior) {
  auto options = options_;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);
  ASSERT_OK_AND_ASSIGN(auto fs, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
  // Directory cleanup retains the existing SDK implementation even when
  // native read/write transport does not support this configuration.
  ASSERT_OK(fs->DeleteDirContentsAsync("bucket/absent", true).status());
  EXPECT_FALSE(fs->DeleteDirContentsAsync("bucket/absent", false).status().ok());
  EXPECT_TRUE(fs->DeleteDirContentsAsync("", true).status().IsNotImplemented());
  ASSERT_OK(fs->DeleteDirContents("bucket/absent", true));
  EXPECT_FALSE(fs->DeleteDirContents("bucket/absent", false).ok());
  EXPECT_TRUE(fs->DeleteDirContents("", true).IsNotImplemented());
  ASSERT_OK(fs->CreateDir("bucket/root/sync-delete", true));
  ASSERT_OK(Write("sync-delete/file", arrow::Buffer::FromString("data")).status());
  ASSERT_OK(fs->DeleteDirContentsAsync("bucket/root/sync-delete", false).status());
  ASSERT_OK_AND_ASSIGN(auto directory, fs->GetFileInfo("bucket/root/sync-delete"));
  EXPECT_EQ(directory.type(), arrow::fs::FileType::Directory);
  ASSERT_OK_AND_ASSIGN(auto removed, fs->GetFileInfo("bucket/root/sync-delete/file"));
  EXPECT_EQ(removed.type(), arrow::fs::FileType::NotFound);
}

class NativeS3ShutdownTest : public NativeS3Test {};
TEST_F(NativeS3ShutdownTest, DrainsPendingWriteBeforeAwsShutdown) {
  auto pending = Write("slow-write", arrow::Buffer::FromString("shutdown"));
  auto shutdown = std::async(std::launch::async, [] { return FinalizeS3(); });
  ASSERT_TRUE(pending.Wait(5));
  ASSERT_OK(pending.status());
  ASSERT_EQ(shutdown.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_OK(shutdown.get());
  EXPECT_FALSE(Read("slow-write", 0, 1).status().ok());
}
#endif
}  // namespace milvus_storage
