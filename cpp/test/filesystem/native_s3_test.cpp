// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include <memory>
#include <folly/ScopeGuard.h>
#include <vector>
#include <chrono>
#include <arrow/util/cancel.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <future>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <folly/executors/ManualExecutor.h>
#include "milvus-storage/format/parquet/folly_arrow_executor.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/async_output_stream.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/fiu_local.h"

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

TEST_F(NativeS3Test, CustomCredentialsProviderRefreshesWithoutAnIoWorker) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture credential inspection";
  class RotatingCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
 public:
    Aws::Auth::AWSCredentials GetAWSCredentials() override {
      auto suffix = std::to_string(generation.load());
      return {("fixture-" + suffix).c_str(), "fixture-secret", ("token-" + suffix).c_str()};
    }
    std::atomic<int> generation{1};
  };

  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  for (auto kind : {S3CredentialsKind::Default, S3CredentialsKind::Role, S3CredentialsKind::WebIdentity}) {
    SCOPED_TRACE(static_cast<int>(kind));
    auto provider = std::make_shared<RotatingCredentialsProvider>();
    auto options = options_;
    options.credentials_kind = kind;
    options.credentials_provider = provider;
    ASSERT_OK_AND_ASSIGN(auto s3, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
    auto fs = std::make_shared<FileSystemProxy>(prefix_, s3);
    for (int generation : {1, 2}) {
      provider->generation.store(generation);
      auto result = Stat(fs, "credential-check/" + std::to_string(generation));
      ASSERT_TRUE(result.Wait(5));
      ASSERT_OK_AND_ASSIGN(auto info, result.result());
      EXPECT_EQ(info.size(), 6);
    }
  }
}

TEST_F(NativeS3Test, ProviderFilesystemOperationsWithAndWithoutCrt) {
  // The fixture checks routing and results, not real cloud authentication.
  for (const auto* provider : {"gcp", "aliyun", "tencent", "huawei", "aws"}) {
    for (const bool crt_reads : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "provider=" << provider << " crt=" << crt_reads);
      auto options = options_;
      options.cloud_provider = provider;
      options.use_crt_async_reads = crt_reads;
      ASSERT_OK_AND_ASSIGN(auto s3, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
      auto fs = std::make_shared<FileSystemProxy>(prefix_, s3);
      ASSERT_OK_AND_ASSIGN(auto infos, Await(fs->GetFileInfoAsync({"hello #?+% 中文", "missing"})));
      ASSERT_EQ(infos.size(), 2);
      EXPECT_EQ(infos[0].size(), 6);
      EXPECT_EQ(infos[1].type(), arrow::fs::FileType::NotFound);

      arrow::fs::FileSelector selector;
      selector.base_dir = "pages";
      auto generator = fs->GetFileInfoGenerator(selector);
      size_t count = 0;
      for (;;) {
        ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
        if (page.empty())
          break;
        count += page.size();
      }
      EXPECT_EQ(count, 5);

      ASSERT_OK_AND_ASSIGN(auto input, fs->OpenInputFile("hello #?+% 中文"));
      ASSERT_OK_AND_ASSIGN(auto metadata, Await(input->ReadMetadataAsync({})));
      EXPECT_NE(metadata, nullptr);
      ASSERT_OK_AND_ASSIGN(auto data, Await(input->ReadAsync({}, 0, 6)));
      EXPECT_EQ(data->ToString(), "abcdef");
      ASSERT_OK(input->Close());
      ASSERT_OK_AND_ASSIGN(input, fs->OpenInputFile(infos[0]));
      ASSERT_OK_AND_ASSIGN(data, Await(input->ReadAsync({}, 1, 3)));
      EXPECT_EQ(data->ToString(), "bcd");
      ASSERT_OK(input->Close());
    }
  }
}

TEST_F(NativeS3Test, OtherProvidersUseNativeRequestsWithoutAnIoWorker) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  for (const auto* provider : {"aliyun", "tencent", "gcp"}) {
    SCOPED_TRACE(provider);
    auto options = options_;
    options.cloud_provider = provider;
    if (options.cloud_provider == "gcp") {
      ArrowFileSystemConfig config;
      config.access_key_id = "fixture";
      config.access_key_value = "fixture-secret";
      ASSERT_OK_AND_ASSIGN(auto credentials, BuildGcpProviderFromConfig(config));
      GcpCredentialRegistry::Instance().Register({NormalizeGcpEndpoint(options.endpoint_override, false), "bucket"},
                                                 std::move(credentials));
    }
    ASSERT_OK_AND_ASSIGN(auto s3, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
    auto fs = std::make_shared<FileSystemProxy>(prefix_, s3);
    ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs, "hello #?+% 中文")));
    EXPECT_EQ(info.size(), 6);
    arrow::fs::FileSelector selector;
    selector.base_dir = "pages";
    auto generator = fs->GetFileInfoGenerator(selector);
    size_t count = 0;
    for (;;) {
      ASSERT_OK_AND_ASSIGN(auto page, Await(generator()));
      if (page.empty())
        break;
      count += page.size();
    }
    EXPECT_EQ(count, 5);
    ASSERT_OK_AND_ASSIGN(auto input, fs->OpenInputFile("hello #?+% 中文"));
    ASSERT_OK_AND_ASSIGN(auto metadata, Await(input->ReadMetadataAsync({})));
    ASSERT_OK(input->Close());
  }
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
  ASSERT_OK_AND_ASSIGN(auto cached, cache.get(std::make_shared<const api::Properties>(properties)));
  ASSERT_OK_AND_ASSIGN(auto cached_again, cache.get(std::make_shared<const api::Properties>(properties)));
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

TEST_F(NativeS3Test, NativeCompletionDoesNotScheduleCallerExecutor) {
  folly::ManualExecutor executor;
  ASSERT_OK_AND_ASSIGN(auto adapter, parquet::MakeFollyArrowExecutor(folly::getKeepAliveToken(executor)));
  ASSERT_OK_AND_ASSIGN(auto base, S3FileSystem::Make(options_, arrow::io::IOContext(adapter.get())));
  auto proxy = std::make_shared<FileSystemProxy>(prefix_, base);
  auto stat = Stat(proxy, "slow");
  const bool completed = stat.Wait(5);
  // Drain after observing completion so a regression fails without stranding work.
  EXPECT_EQ(executor.drain(), 0);
  ASSERT_TRUE(completed);
  ASSERT_OK_AND_ASSIGN(auto info, stat.result());
  EXPECT_EQ(info.size(), 6);

  arrow::fs::FileSelector selector;
  selector.base_dir = "pages";
  auto generator = proxy->GetFileInfoGenerator(selector);
  auto page = generator();
  const bool listed = page.Wait(5);
  EXPECT_EQ(executor.drain(), 0);
  ASSERT_TRUE(listed);
  ASSERT_OK(page.status());
  EXPECT_FALSE(page.result()->empty());
}

TEST_F(NativeS3Test, InputFactoriesDoNotScheduleCallerExecutor) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  const std::string path = "missing";
  arrow::fs::FileInfo info(path, arrow::fs::FileType::File);
  info.set_size(6);
  auto by_path = fs_->OpenInputFileAsync(path);
  auto by_info = fs_->OpenInputFileAsync(info);
  auto stream_by_path = fs_->OpenInputStreamAsync(path);
  auto stream_by_info = fs_->OpenInputStreamAsync(info);
  EXPECT_TRUE(by_path.is_finished());
  EXPECT_TRUE(by_info.is_finished());
  EXPECT_TRUE(stream_by_path.is_finished());
  EXPECT_TRUE(stream_by_info.is_finished());
  ASSERT_OK(by_path.status());
  ASSERT_OK(by_info.status());
  ASSERT_OK(stream_by_path.status());
  ASSERT_OK(stream_by_info.status());
  ASSERT_OK(by_path.result().ValueOrDie()->Close());
  ASSERT_OK(by_info.result().ValueOrDie()->Close());
  ASSERT_OK(stream_by_path.result().ValueOrDie()->Close());
  ASSERT_OK(stream_by_info.result().ValueOrDie()->Close());
}

TEST_F(NativeS3Test, CompletionDoesNotRequireRunningExecutor) {
  // Native completion must not depend on the filesystem executor accepting work.
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "hello #?+% 中文")));
  EXPECT_EQ(info.size(), 6);
}

TEST_F(NativeS3Test, CompletionCanReleaseTheLastNativeTransportOwner) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  auto pending = Stat(fs_, "slow");
  fs_.reset();
  sync_.reset();
  ASSERT_TRUE(pending.Wait(5));
  ASSERT_OK_AND_ASSIGN(auto info, pending.result());
  EXPECT_EQ(info.size(), 6);
}

TEST_F(NativeS3Test, BatchCompletionDoesNotRetainTheSdkCrtHolder) {
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
  ASSERT_OK(stream->Flush());
  ASSERT_OK(stream->Abort());
  ASSERT_OK_AND_ASSIGN(auto conditional, proxy->OpenConditionalOutputStream("no-network", nullptr));
  auto conditional_async = std::dynamic_pointer_cast<AsyncOutputStream>(conditional);
  ASSERT_NE(conditional_async, nullptr);
  ASSERT_OK(conditional->Abort());
  // The sized factory uses its argument, not the default part size in options.
  ASSERT_OK_AND_ASSIGN(auto sized, s3->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 6 * 1024 * 1024));
  auto sized_async = std::dynamic_pointer_cast<AsyncOutputStream>(sized);
  ASSERT_NE(sized_async, nullptr);
  ASSERT_OK(sized->Abort());
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
TEST_F(NativeS3Test, MultipartCompleteAndAbort) {
  ASSERT_OK_AND_ASSIGN(const auto metrics, sync_->GetMetrics(kOriginMetricsSource));
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  ASSERT_OK(Write("multipart", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a') + "tail")).status());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 1);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 1);
  EXPECT_EQ(metrics->GetWriteCount(), 2);
  EXPECT_EQ(metrics->GetWriteBytes(), 5 * 1024 * 1024 + 4);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("multipart", 5 * 1024 * 1024 - 2, 10)));
  EXPECT_EQ(data->ToString(), "aatail");
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("aborted"));
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(stream->Flush());
  ASSERT_OK(stream->Abort());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 2);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 1);
  EXPECT_EQ(metrics->GetWriteCount(), 3);
  EXPECT_EQ(metrics->GetWriteBytes(), 10 * 1024 * 1024 + 4);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
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
#ifdef BUILD_WITH_FIU
TEST_F(NativeS3Test, MultipartCompletionFaultDoesNotPublishObject) {
  const auto payload = arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a') + "tail");
  for (const bool async_close : {false, true}) {
    const std::string path = async_close ? "completion-fault-async" : "completion-fault-sync";
    SCOPED_TRACE(path);
    ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream(path));
    ASSERT_OK(stream->Write(payload));
    {
      ScopedFiuFault fault(FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL, /*one_time=*/false);
      ASSERT_EQ(fault.enable_result(), 0);
      const auto status = async_close ? stream->CloseAsync().status() : stream->Close();
      EXPECT_TRUE(status.IsIOError());
      EXPECT_NE(status.message().find("Injected fault: " FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL), std::string::npos);
    }
    ASSERT_OK_AND_ASSIGN(const auto missing, Await(Stat(fs_, path)));
    EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
    ASSERT_OK(Write(path, payload).status());
    ASSERT_OK_AND_ASSIGN(const auto data, Await(Read(path, 0, payload->size())));
    EXPECT_TRUE(data->Equals(*payload));
  }
}
#endif
TEST_F(NativeS3Test, MetadataServiceErrorsPreserveDetailsAndClassification) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture error responses";
  for (const auto& [error, expected] : {std::pair{"AccessDenied", ExtendStatusCode::AwsErrorAccessDenied},
                                        std::pair{"PreconditionFailed", ExtendStatusCode::AwsErrorPreConditionFailed},
                                        std::pair{"ConditionalRequestConflict", ExtendStatusCode::AwsErrorConflict}}) {
    SCOPED_TRACE(error);
    arrow::fs::FileSelector selector;
    selector.base_dir = std::string("service-error/") + error;
    auto status = fs_->GetFileInfoGenerator(selector)().status();
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message().find("fixture service detail"), std::string::npos);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), expected);
  }
}

TEST_F(NativeS3Test, IncompleteResponsesDoNotEstablishSuccessOrAbsence) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture truncated responses";
  arrow::fs::FileSelector selector;
  selector.base_dir = "truncated-list";
  selector.allow_not_found = true;
  auto status = fs_->GetFileInfoGenerator(selector)().status();
  ASSERT_FALSE(status.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientNetwork);
}
TEST_F(NativeS3Test, InvalidAndOversizedResponsesFail) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture invalid responses";
  arrow::fs::FileSelector selector;
  selector.base_dir = "invalid-xml";
  EXPECT_FALSE(fs_->GetFileInfoGenerator(selector)().status().ok());
  selector.base_dir = "oversized-error";
  EXPECT_TRUE(fs_->GetFileInfoGenerator(selector)().status().IsCapacityError());
}
TEST_F(NativeS3Test, SinglePutDoesNotCountMultipartOperations) {
  ASSERT_OK_AND_ASSIGN(const auto metrics, sync_->GetMetrics(kOriginMetricsSource));
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  ASSERT_OK(Write("put-metrics", arrow::Buffer::FromString("data")).status());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 0);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 0);
  EXPECT_EQ(metrics->GetWriteCount(), 1);
  EXPECT_EQ(metrics->GetWriteBytes(), 4);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
}
TEST_F(NativeS3Test, FailedMultipartCreationCountsAttemptAndFailure) {
  ASSERT_OK_AND_ASSIGN(const auto metrics, sync_->GetMetrics(kOriginMetricsSource));
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  EXPECT_FALSE(Write("denied", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))).status().ok());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 1);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 0);
  EXPECT_EQ(metrics->GetWriteCount(), 0);
  EXPECT_EQ(metrics->GetWriteBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 1);
}
// Native metadata must not change output routing or impose a part admission limit.
TEST_F(NativeS3Test, OutputFactoriesRetainSdkExecutorAndWriteSemantics) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  for (const bool crt : {false, true}) {
    auto options = options_;
    options.use_crt_async_reads = crt;
    for (const bool background : {false, true}) {
      options.background_writes = background;
      ASSERT_OK_AND_ASSIGN(auto raw, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
      auto fs = std::make_shared<FileSystemProxy>(prefix_, raw);
      for (int factory = 0; factory < 3; ++factory) {
        SCOPED_TRACE(::testing::Message() << "crt=" << crt << " background=" << background << " factory=" << factory);
        const auto name =
            "sdk-output-" + std::to_string(crt) + "-" + std::to_string(background) + "-" + std::to_string(factory);
        auto opened = factory == 0   ? fs->OpenOutputStream(name)
                      : factory == 1 ? fs->OpenConditionalOutputStream(name, nullptr)
                                     : fs->OpenOutputStreamWithUploadSize(name, nullptr, 5 * 1024 * 1024);
        ASSERT_OK_AND_ASSIGN(auto stream, opened);
        // Two parts in one Write exceeds the old native limit (max_connections=1).
        auto payload = arrow::Buffer::FromString(std::string(background ? 7 : 10 * 1024 * 1024, 'x'));
        ASSERT_OK(stream->Write(payload));
        auto closed = stream->CloseAsync();
        ASSERT_TRUE(closed.Wait(5));
        if (background) {
          // The SDK path requires its executor even when native metadata is enabled.
          EXPECT_FALSE(closed.status().ok());
        } else {
          ASSERT_OK(closed.status());
          ASSERT_OK_AND_ASSIGN(auto input, fs->OpenInputFile(name));
          ASSERT_OK_AND_ASSIGN(auto data, input->Read(payload->size()));
          EXPECT_TRUE(data->Equals(*payload));
          ASSERT_OK(input->Close());
        }
      }
    }
  }
}

TEST_F(NativeS3Test, GcpConditionalMultipartIsRejectedBeforeCreatingAnUpload) {
  ASSERT_OK(Write("gcp-multipart-guard", arrow::Buffer::FromString("original")).status());
  for (const bool crt : {false, true}) {
    auto options = options_;
    options.cloud_provider = kCloudProviderGCP;
    options.use_crt_async_reads = crt;
    ASSERT_OK_AND_ASSIGN(auto raw, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
    auto fs = std::make_shared<FileSystemProxy>(prefix_, raw);
    ASSERT_OK_AND_ASSIGN(auto output, fs->OpenConditionalOutputStream("gcp-multipart-guard", nullptr));
    ASSERT_OK_AND_ASSIGN(auto metrics, raw->GetMetrics(kOriginMetricsSource));
    metrics->Reset();
    auto status = output->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'x')));
    EXPECT_TRUE(status.IsNotImplemented()) << status;
    EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 0);
    ASSERT_OK(output->Abort());
    ASSERT_OK_AND_ASSIGN(auto data, Await(Read("gcp-multipart-guard", 0, 8)));
    EXPECT_EQ(data->ToString(), "original");
  }
}

TEST_F(NativeS3Test, CrtMetadataAndRangeReadsRetryTransientFailures) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture transient failures";
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "retry-head")));
  EXPECT_EQ(info.size(), 6);
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("retry-get", 0, 6)));
  EXPECT_EQ(data->ToString(), "abcdef");
}

TEST_F(NativeS3Test, DirectoryContentsRetainsExistingSdkBehavior) {
  auto options = options_;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);
  ASSERT_OK_AND_ASSIGN(auto fs, S3FileSystem::Make(options, arrow::io::IOContext(executor_.get())));
  // Directory cleanup retains the existing SDK implementation even when
  // native metadata transport does not support this configuration.
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

class ParallelUploadS3Test : public NativeS3Test {
  protected:
  void SetUp() override {
    NativeS3Test::SetUp();
    if (HasFatalFailure() || IsSkipped())
      return;
    // Register the control client before uploads acquire the SDK finalizer lock.
    auto options = options_;
    options.use_crt_async_reads = false;
    ASSERT_OK_AND_ASSIGN(control_, S3FileSystem::Make(options));
  }
  void TearDown() override {
    control_.reset();
    NativeS3Test::TearDown();
  }
  arrow::Status ControlUpload(const std::string& part, bool release) {
    if (release) {
      return control_->DeleteFile(prefix_ + "/upload-release/" + part);
    }
    ARROW_ASSIGN_OR_RAISE(auto info, control_->GetFileInfo(prefix_ + "/upload-started/" + part));
    return info.type() == arrow::fs::FileType::File ? arrow::Status::OK()
                                                    : arrow::Status::IOError("Upload did not reach fixture: ", part);
  }
  std::shared_ptr<S3FileSystem> control_;
};

TEST_F(ParallelUploadS3Test, SmallAndEmptyUploadsUseCrtWithoutIoWorker) {
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  for (const auto& data : {std::string(), std::string("payload")}) {
    const auto path = "parallel-small-" + std::to_string(data.size());
    ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream(path, nullptr));
    ASSERT_OK(stream->Write(arrow::Buffer::FromString(data)));
    ASSERT_OK(stream->Close());
    ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read(path, 0, data.size())));
    EXPECT_EQ(uploaded->ToString(), data);
  }
}

TEST_F(ParallelUploadS3Test, CombinesWritesAcrossPartBoundaries) {
  const auto expected = std::string(4 * 1024 * 1024, 'a') + std::string(6 * 1024 * 1024, 'b') + "tail";
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("parallel-cross-write", nullptr));
  auto first = expected.substr(0, 4 * 1024 * 1024);
  ASSERT_OK(stream->Write(first.data(), first.size()));
  first.assign(first.size(), 'x');
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(expected.substr(4 * 1024 * 1024))));
  ASSERT_OK(stream->CloseAsync().status());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("parallel-cross-write", 0, expected.size())));
  EXPECT_EQ(data->ToString(), expected);
}

TEST_F(ParallelUploadS3Test, CrtUploadsPartsBeforeClose) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  const auto expected = std::string(20 * 1024 * 1024, 'a') + "tail";
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("held/crt-parallel", nullptr));
  std::future<arrow::Status> closed;
  auto release = folly::makeGuard([&] {
    for (int part = 1; part <= 4; ++part) (void)ControlUpload("crt-parallel/" + std::to_string(part), true);
  });
  auto data = arrow::Buffer::FromString(expected);
  std::weak_ptr<arrow::Buffer> retained = data;
  ASSERT_OK(stream->Write(data));
  data.reset();
  // Parts upload before EOF, while the final object remains unpublished.
  ASSERT_OK_AND_ASSIGN(auto before, Await(Stat(fs_, "held/crt-parallel")));
  EXPECT_EQ(before.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK(ControlUpload("crt-parallel/1", false));
  ASSERT_OK(ControlUpload("crt-parallel/2", false));
  closed = std::async(std::launch::async, [stream] { return stream->Close(); });
  EXPECT_EQ(closed.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  // The filesystem's max_connections is one, but CRT schedules its own parts.
  ASSERT_OK(ControlUpload("crt-parallel/2", true));
  ASSERT_OK(ControlUpload("crt-parallel/3", true));
  ASSERT_OK(ControlUpload("crt-parallel/4", true));
  ASSERT_OK(ControlUpload("crt-parallel/1", true));
  ASSERT_EQ(closed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_OK(closed.get());
  EXPECT_TRUE(retained.expired());
  ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read("held/crt-parallel", 0, expected.size())));
  EXPECT_EQ(uploaded->ToString(), expected);
}

TEST_F(ParallelUploadS3Test, CrtAbortCancelsUnpublishedUpload) {
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("crt-abort", nullptr));
  ASSERT_OK(stream->Write(data));
  ASSERT_OK(stream->Abort());
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "crt-abort")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, IdleInputDoesNotBlockAnotherUpload) {
  ASSERT_OK_AND_ASSIGN(auto idle, fs_->OpenParallelUploadOutputStream("parallel-idle", nullptr));
  ASSERT_OK(idle->Write(arrow::Buffer::FromString("prefix")));
  ASSERT_OK_AND_ASSIGN(auto next, fs_->OpenParallelUploadOutputStream("parallel-next", nullptr));
  ASSERT_OK(next->Write(arrow::Buffer::FromString("payload")));
  auto closed = next->CloseAsync();
  ASSERT_TRUE(closed.Wait(5));
  ASSERT_OK(closed.status());
  ASSERT_OK(idle->Abort());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("parallel-next", 0, 7)));
  EXPECT_EQ(data->ToString(), "payload");
}

TEST_F(ParallelUploadS3Test, AbortDuringCloseCancelsUpload) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("held/abort-closing", nullptr));
  auto release = folly::makeGuard([&] {
    for (int part = 1; part <= 3; ++part) (void)ControlUpload("abort-closing/" + std::to_string(part), true);
  });
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'))));
  auto closed = stream->CloseAsync();
  ASSERT_OK(ControlUpload("abort-closing/1", false));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  auto aborted = async->AbortAsync();
  for (int part = 1; part <= 3; ++part) ASSERT_OK(ControlUpload("abort-closing/" + std::to_string(part), true));
  ASSERT_TRUE(aborted.Wait(5));
  ASSERT_OK(aborted.status());
  ASSERT_TRUE(closed.Wait(5));
  EXPECT_TRUE(closed.status().IsCancelled());
  EXPECT_TRUE(stream->closed());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "held/abort-closing")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, DestructionCancelsUnclosedUpload) {
  {
    ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("parallel-dropped", nullptr));
    ASSERT_OK(stream->Write(arrow::Buffer::FromString("payload")));
  }
  ASSERT_OK_AND_ASSIGN(auto next, fs_->OpenParallelUploadOutputStream("parallel-after-drop", nullptr));
  ASSERT_OK(next->Write(arrow::Buffer::FromString("next")));
  ASSERT_OK(next->Close());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "parallel-dropped")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, CloseContinuationUsesCallerExecutorAndCanUpload) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("held/crt-continuation", nullptr));
  auto release = folly::makeGuard([&] {
    for (int part = 1; part <= 3; ++part) (void)ControlUpload("crt-continuation/" + std::to_string(part), true);
  });
  ASSERT_OK(stream->Write(data));
  auto closed = stream->CloseAsync();
  ASSERT_OK(ControlUpload("crt-continuation/1", false));
  auto continued = closed.Then([this, data]() -> arrow::Status {
    EXPECT_TRUE(executor_->OwnsThisThread());
    ARROW_ASSIGN_OR_RAISE(auto next, fs_->OpenParallelUploadOutputStream("crt-continuation-next", nullptr));
    ARROW_RETURN_NOT_OK(next->Write(data));
    return next->Close();
  });
  for (int part = 1; part <= 3; ++part) ASSERT_OK(ControlUpload("crt-continuation/" + std::to_string(part), true));
  ASSERT_TRUE(continued.Wait(5));
  ASSERT_OK(continued.status());
  ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read("crt-continuation-next", 0, data->size())));
  EXPECT_TRUE(uploaded->Equals(*data));
}

TEST_F(ParallelUploadS3Test, FlushContinuationCanWriteAndCloseOnCallerExecutor) {
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("parallel-flush-continuation", nullptr));
  auto chunk = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK(stream->Write(chunk));
  std::promise<void> entered;
  std::promise<void> resume;
  auto resumed = resume.get_future();
  ASSERT_OK(executor_->Spawn([&] {
    entered.set_value();
    resumed.wait();
  }));
  auto release = folly::makeGuard([&] { resume.set_value(); });
  ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  auto continued = async->FlushAsync().Then([this, stream, chunk]() -> arrow::Status {
    EXPECT_TRUE(executor_->OwnsThisThread());
    ARROW_RETURN_NOT_OK(stream->Write(chunk));
    return stream->Close();
  });
  resume.set_value();
  release.dismiss();
  ASSERT_TRUE(continued.Wait(5));
  ASSERT_OK(continued.status());
  ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read("parallel-flush-continuation", 0, 2 * chunk->size())));
  EXPECT_EQ(uploaded->ToString(), chunk->ToString() + chunk->ToString());
}

TEST_F(ParallelUploadS3Test, CrtPartFailureCleansMultipartUpload) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  std::weak_ptr<arrow::Buffer> retained = data;
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("held/failed-wait", nullptr));
  auto release = folly::makeGuard([&] {
    for (int part = 1; part <= 3; ++part) (void)ControlUpload("failed-wait/" + std::to_string(part), true);
  });
  ASSERT_OK(stream->Write(data));
  data.reset();
  auto closed = stream->CloseAsync().Then([] { return arrow::Status::OK(); },
                                          [this](const arrow::Status& status) {
                                            EXPECT_TRUE(executor_->OwnsThisThread());
                                            return status;
                                          });
  ASSERT_OK(ControlUpload("failed-wait/1", false));
  for (int part = 1; part <= 3; ++part) ASSERT_OK(ControlUpload("failed-wait/" + std::to_string(part), true));
  ASSERT_TRUE(closed.Wait(5));
  EXPECT_FALSE(closed.status().ok());
  EXPECT_TRUE(stream->closed());
  EXPECT_TRUE(retained.expired());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "held/failed-wait")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK_AND_ASSIGN(auto aborted, fs_->GetFileInfo("upload-aborted/held/failed-wait"));
  EXPECT_EQ(aborted.type(), arrow::fs::FileType::File);
}

TEST_F(ParallelUploadS3Test, CrtCancellationBeforeCloseDoesNotPublish) {
  arrow::StopSource stop;
  ASSERT_OK_AND_ASSIGN(auto fs, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get(), stop.token())));
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK_AND_ASSIGN(auto stream, fs->OpenParallelUploadOutputStream(prefix_ + "/crt-cancel", nullptr));
  ASSERT_OK(stream->Write(data));
  stop.RequestStop();
  EXPECT_TRUE(stream->CloseAsync().status().IsCancelled());
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "crt-cancel")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, CrtCancellationDuringCloseCleansMultipartUpload) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  arrow::StopSource stop;
  ASSERT_OK_AND_ASSIGN(auto s3, S3FileSystem::Make(options_, arrow::io::IOContext(executor_.get(), stop.token())));
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK_AND_ASSIGN(auto stream, s3->OpenParallelUploadOutputStream(prefix_ + "/held/crt-canceled", nullptr));
  auto release = folly::makeGuard([&] {
    stop.RequestStop();
    for (int part = 1; part <= 3; ++part) (void)ControlUpload("crt-canceled/" + std::to_string(part), true);
  });
  ASSERT_OK(stream->Write(data));
  auto closed = stream->CloseAsync();
  ASSERT_OK(ControlUpload("crt-canceled/1", false));
  stop.RequestStop();
  for (int part = 1; part <= 3; ++part) ASSERT_OK(ControlUpload("crt-canceled/" + std::to_string(part), true));
  ASSERT_TRUE(closed.Wait(5));
  EXPECT_TRUE(closed.status().IsCancelled()) << closed.status();
  EXPECT_TRUE(stream->closed());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "held/crt-canceled")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
  ASSERT_OK_AND_ASSIGN(auto aborted, fs_->GetFileInfo("upload-aborted/held/crt-canceled"));
  EXPECT_EQ(aborted.type(), arrow::fs::FileType::File);
}

TEST_F(ParallelUploadS3Test, MultipartPreservesObjectMetadata) {
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a') + "tail");
  auto metadata = arrow::key_value_metadata({"Content-Type", "test"}, {"application/test", "parallel-upload"});
  const std::string path = "parallel-metadata #?+% 中文";
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream(path, metadata));
  ASSERT_OK(stream->Write(data));
  ASSERT_OK(stream->Close());
  ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read(path, 0, data->size())));
  EXPECT_TRUE(uploaded->Equals(*data));
  ASSERT_OK_AND_ASSIGN(auto stored, Await(Metadata(path)));
  ASSERT_OK_AND_ASSIGN(auto content_type, stored->Get("Content-Type"));
  EXPECT_EQ(content_type, "application/test");
  ASSERT_OK_AND_ASSIGN(auto custom, stored->Get("test"));
  EXPECT_EQ(custom, "parallel-upload");
}

TEST_F(ParallelUploadS3Test, StoppedCompletionExecutorReportsFailure) {
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenParallelUploadOutputStream("stopped-completion", nullptr));
  ASSERT_OK(stream->Write(arrow::Buffer::FromString("payload")));
  ASSERT_OK(executor_->Shutdown());
  executor_stopped_ = true;
  auto closed = stream->CloseAsync();
  ASSERT_TRUE(closed.Wait(5));
  EXPECT_FALSE(closed.status().ok());
  EXPECT_TRUE(stream->closed());
  ASSERT_OK(stream->Close());
  ASSERT_OK_AND_ASSIGN(auto info, control_->GetFileInfo(prefix_ + "/stopped-completion"));
  EXPECT_EQ(info.size(), 7);
}
class NativeS3ShutdownTest : public NativeS3Test {};
TEST_F(NativeS3ShutdownTest, DrainsPendingMetadataBeforeAwsShutdown) {
  auto pending = Stat(fs_, "slow");
  auto shutdown = std::async(std::launch::async, [] { return FinalizeS3(); });
  ASSERT_TRUE(pending.Wait(5));
  ASSERT_OK(pending.status());
  ASSERT_EQ(shutdown.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_OK(shutdown.get());
  EXPECT_FALSE(Read("slow-write", 0, 1).status().ok());
}
#endif
}  // namespace milvus_storage
