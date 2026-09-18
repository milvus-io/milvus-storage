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
TEST(NativeS3Capability, UnsupportedProviderDoesNotFallback) {
  ASSERT_OK_AND_ASSIGN(auto executor, arrow::internal::ThreadPool::Make(1));
  auto local = std::make_shared<arrow::fs::LocalFileSystem>();
  auto fs = std::make_shared<FileSystemProxy>("", local);
  auto result = fs->GetFileInfoAsync("missing");
  EXPECT_TRUE(result.status().IsNotImplemented());
}

#ifdef WITH_CRT
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
    ASSERT_OK(executor_->Spawn([start, result] {
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
    return fs_->OpenOutputStreamAsync(path, metadata).Then([data](std::shared_ptr<arrow::io::OutputStream> stream) {
      auto status = stream->Write(data);
      if (!status.ok())
        return arrow::Future<>::MakeFinished(status);
      return stream->CloseAsync();
    });
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
  ASSERT_OK_AND_ASSIGN(auto after, Await(fs_->GetFileInfoAsync("hello #?+% 中文")));
  EXPECT_EQ(before, after);
  std::shared_ptr<arrow::fs::FileSystem> arrow_fs = fs_;
  ASSERT_OK_AND_ASSIGN(auto batch,
                       Await(arrow_fs->GetFileInfoAsync(std::vector<std::string>{"hello #?+% 中文", "missing"})));
  ASSERT_EQ(batch.size(), 2);
  EXPECT_EQ(batch[0], after);
  EXPECT_EQ(batch[1].type(), arrow::fs::FileType::NotFound);
  auto nested = std::make_shared<FileSystemProxy>("pages", fs_);
  ASSERT_OK_AND_ASSIGN(auto nested_info, Await(nested->GetFileInfoAsync("a")));
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
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs->GetFileInfoAsync("root/hello #?+% 中文")));
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
  ASSERT_OK_AND_ASSIGN(auto cached_info, Await(cached->GetFileInfoAsync(info.path())));
  EXPECT_EQ(cached_info, info);
  cache.clean();
}

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
  auto root = sync_;
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

TEST_F(NativeS3Test, AsyncWriteAndSyncReadShareOneHandle) {
  ASSERT_OK(Write("same-handle", arrow::Buffer::FromString("data")).status());
  ASSERT_OK_AND_ASSIGN(auto reader, fs_->OpenInputFile("same-handle"));
  ASSERT_OK_AND_ASSIGN(auto data, reader->Read(4));
  EXPECT_EQ(data->ToString(), "data");
  ASSERT_OK_AND_ASSIGN(auto info, fs_->GetFileInfo("same-handle"));
  ASSERT_OK_AND_ASSIGN(auto async_info, Await(fs_->GetFileInfoAsync("same-handle")));
  EXPECT_EQ(info, async_info);
}

TEST_F(NativeS3Test, WritesConditionalMetadataAndOwnedBuffer) {
  EXPECT_FALSE(fs_->OpenOutputStreamAsync("file/").status().ok());
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
  ASSERT_OK_AND_ASSIGN(auto stream, Await(fs_->OpenOutputStreamAsync("aborted")));
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(async->AbortAsync().status());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(fs_->GetFileInfoAsync("aborted")));
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
  ASSERT_OK_AND_ASSIGN(auto missing, Await(fs_->GetFileInfoAsync("error-complete")));
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
  ASSERT_OK_AND_ASSIGN(auto stream, Await(fs_->OpenOutputStreamAsync("backpressure")));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  EXPECT_TRUE(stream->Write(arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'x'))).IsCapacityError());
  ASSERT_OK_AND_ASSIGN(auto offset, stream->Tell());
  EXPECT_EQ(offset, 0);
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  ASSERT_OK(async->FlushAsync().status());
  ASSERT_OK(stream->Write(arrow::Buffer::FromString("tail")));
  ASSERT_OK(stream->CloseAsync().status());
}

TEST_F(NativeS3Test, DirectoryLifecycleAndBoundedDeletion) {
  ASSERT_OK(fs_->CreateDirAsync("dir/sub").status());
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs_->GetFileInfoAsync("dir/sub")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::Directory);
  EXPECT_FALSE(fs_->CreateDirAsync("no-parent/child", false).status().ok());
  for (int i = 0; i < 5; ++i)
    ASSERT_OK(Write("dir/sub/file" + std::to_string(i), arrow::Buffer::FromString("data")).status());
  EXPECT_FALSE(fs_->DeleteFileAsync("dir").status().ok());
  std::shared_ptr<arrow::fs::FileSystem> arrow_fs = fs_;
  ASSERT_OK(arrow_fs->DeleteDirContentsAsync("dir/sub").status());
  ASSERT_OK_AND_ASSIGN(info, Await(fs_->GetFileInfoAsync("dir/sub")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::Directory);
  arrow::fs::FileSelector selector;
  selector.base_dir = "dir/sub";
  auto listing = fs_->GetFileInfoGenerator(selector);
  ASSERT_OK_AND_ASSIGN(auto page, Await(listing()));
  EXPECT_TRUE(page.empty());
  ASSERT_OK(fs_->DeleteDirAsync("dir").status());
  ASSERT_OK_AND_ASSIGN(info, Await(fs_->GetFileInfoAsync("dir")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK(fs_->DeleteDirContentsAsync("absent-dir", true).status());
  EXPECT_FALSE(fs_->DeleteDirContentsAsync("absent-dir", false).status().ok());
}
TEST_F(NativeS3Test, CopyMoveAndDeleteSpecialKeys) {
  auto options = arrow::key_value_metadata({"test"}, {"copied"});
  ASSERT_OK(Write("source #?+% 中文", arrow::Buffer::FromString("payload"), options).status());
  ASSERT_OK(fs_->CopyFileAsync("source #?+% 中文", "copy").status());
  ASSERT_OK_AND_ASSIGN(auto metadata, Await(Metadata("copy")));
  ASSERT_OK_AND_ASSIGN(auto value, metadata->Get("test"));
  EXPECT_EQ(value, "copied");
  ASSERT_OK(fs_->MoveAsync("copy", "moved").status());
  ASSERT_OK_AND_ASSIGN(auto info, Await(fs_->GetFileInfoAsync("copy")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("moved", 0, 10)));
  EXPECT_EQ(data->ToString(), "payload");
  ASSERT_OK(fs_->DeleteFileAsync("moved").status());
  EXPECT_FALSE(fs_->DeleteFileAsync("moved").status().ok());
  ASSERT_OK(fs_->DeleteFileAsync("source #?+% 中文").status());
}
TEST_F(NativeS3Test, CopyEmbeddedErrorDoesNotDeleteSource) {
  ASSERT_OK(Write("copy-source", arrow::Buffer::FromString("data")).status());
  EXPECT_FALSE(fs_->MoveAsync("copy-source", "copy-error").status().ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("copy-source", 0, 4)));
  EXPECT_EQ(data->ToString(), "data");
}
TEST_F(NativeS3Test, BucketLifecycleHonorsPolicy) {
  if (!std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires isolated MinIO";
  ASSERT_OK_AND_ASSIGN(auto root, MakeAsyncFileSystem(sync_, arrow::io::IOContext(executor_.get())));
  EXPECT_FALSE(root->DeleteDirAsync("bucket").status().ok());
  auto options = options_;
  options.allow_bucket_deletion = true;
  ASSERT_OK_AND_ASSIGN(auto sync, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
  ASSERT_OK_AND_ASSIGN(root, MakeAsyncFileSystem(sync, arrow::io::IOContext(executor_.get())));
  ASSERT_OK(root->CreateDirAsync("native-s3-bucket-test/child").status());
  ASSERT_OK(root->WriteAsync("native-s3-bucket-test/child/file", arrow::Buffer::FromString("data")).status());
  ASSERT_OK(root->DeleteDirAsync("native-s3-bucket-test").status());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(root->GetFileInfoAsync("native-s3-bucket-test")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
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
