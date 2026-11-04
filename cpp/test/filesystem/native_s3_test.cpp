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
    if (std::getenv("STORAGE_NATIVE_S3_ENDPOINT") && !IsS3Initialized()) {
      ASSERT_OK(S3FileSystemProducer(ArrowFileSystemConfig{}).InitS3());
    }
  }
  void SetUp() override {
    const char* endpoint = std::getenv("STORAGE_NATIVE_S3_ENDPOINT");
    if (!endpoint)
      GTEST_SKIP() << "Run cpp/scripts/test_native_s3.py for isolated S3 fixture";
    ASSERT_OK(CheckS3Initialized());
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
    ARROW_ASSIGN_OR_RAISE(auto file, fs_->OpenInputFile(path));
    ARROW_RETURN_NOT_OK(file->GetSize().status());
    return file->ReadAsync(arrow::io::IOContext(executor_.get()), offset, size);
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> Metadata(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto file, fs_->OpenInputFile(path));
    return file->ReadMetadata();
  }
  void TearDown() override {
    fs_.reset();
    sync_.reset();
    if (executor_ && !executor_stopped_)
      ASSERT_OK(executor_->Shutdown());
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

TEST_F(NativeS3Test, ExistingOutputFactoriesSupportAsyncFlushAndAbort) {
  auto options = options_;
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
  for (const int64_t part_size : {-1LL, 0LL, 512LL * 1024 * 1024 + 1, 5LL * 1024 * 1024 * 1024}) {
    const auto status = s3->OpenOutputStreamWithUploadSize("bucket/key", nullptr, part_size).status();
    EXPECT_TRUE(status.IsInvalid()) << status;
  }
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
  EXPECT_FALSE(fs_->OpenOutputStream("file/", nullptr).status().ok());
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
  const auto metrics = sync_->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  ASSERT_OK(Write("multipart", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a') + "tail")).status());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 1);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 1);
  EXPECT_EQ(metrics->GetWriteCount(), 2);
  EXPECT_EQ(metrics->GetWriteBytes(), 5 * 1024 * 1024 + 4);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
  ASSERT_OK_AND_ASSIGN(auto input, fs_->OpenInputFile("multipart"));
  ASSERT_OK_AND_ASSIGN(auto data, input->ReadAt(5 * 1024 * 1024 - 2, 10));
  EXPECT_EQ(data->ToString(), "aatail");
  ASSERT_OK(input->Close());
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("aborted", nullptr));
  ASSERT_OK(stream->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))));
  auto async = std::dynamic_pointer_cast<AsyncOutputStream>(stream);
  ASSERT_NE(async, nullptr);
  ASSERT_OK(stream->Flush());
  ASSERT_OK(stream->Abort());
  if (options_.use_crt_async_reads) {
    // CRT Flush only waits for input consumption; Abort may race with the part request.
    EXPECT_GE(metrics->GetWriteCount(), 2);
    EXPECT_LE(metrics->GetWriteCount(), 3);
    EXPECT_EQ(metrics->GetWriteBytes(), 5 * 1024 * 1024 + 4);
    EXPECT_EQ(metrics->GetFailedCount(), 1);
  } else {
    // SDK Flush waits for uploaded parts; aborting the multipart session is not a failed write.
    EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 2);
    EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 1);
    EXPECT_EQ(metrics->GetWriteCount(), 3);
    EXPECT_EQ(metrics->GetWriteBytes(), 10 * 1024 * 1024 + 4);
    EXPECT_EQ(metrics->GetFailedCount(), 0);
  }
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "aborted")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(NativeS3Test, MultipartConditionalConflictPreservesObject) {
  ASSERT_OK(Write("multipart-conflict", arrow::Buffer::FromString("original")).status());
  auto condition = arrow::key_value_metadata({"If-None-Match"}, {"*"});
  EXPECT_FALSE(Write("multipart-conflict", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'x')), condition)
                   .status()
                   .ok());
  ASSERT_OK_AND_ASSIGN(auto output, fs_->OpenConditionalOutputStream("multipart-conflict", nullptr));
  ASSERT_OK(output->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'x'))));
  EXPECT_FALSE(output->CloseAsync().status().ok());
  ASSERT_OK_AND_ASSIGN(auto data, Await(Read("multipart-conflict", 0, 20)));
  EXPECT_EQ(data->ToString(), "original");
}

TEST_F(NativeS3Test, SmallUploadCountsOneObjectWrite) {
  const auto metrics = sync_->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  ASSERT_OK(Write("put-metrics", arrow::Buffer::FromString("data")).status());
  // SDK uses PutObject; CRT streaming uses multipart even for one small part.
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), options_.use_crt_async_reads ? 1 : 0);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), options_.use_crt_async_reads ? 1 : 0);
  EXPECT_EQ(metrics->GetWriteCount(), 1);
  EXPECT_EQ(metrics->GetWriteBytes(), 4);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
}

TEST_F(NativeS3Test, FailedMultipartCreationCountsAttemptAndFailure) {
  const auto metrics = sync_->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  EXPECT_FALSE(Write("denied", arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a'))).status().ok());
  EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 1);
  EXPECT_EQ(metrics->GetMultiPartUploadFinished(), 0);
  EXPECT_EQ(metrics->GetWriteCount(), 0);
  EXPECT_EQ(metrics->GetWriteBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 1);
}

TEST_F(NativeS3Test, OutputFactoriesRespectCrtAndBackgroundSettings) {
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
            "output-" + std::to_string(crt) + "-" + std::to_string(background) + "-" + std::to_string(factory);
        auto opened = factory == 0   ? fs->OpenOutputStream(name, nullptr)
                      : factory == 1 ? fs->OpenConditionalOutputStream(name, nullptr)
                                     : fs->OpenOutputStreamWithUploadSize(name, nullptr, 5 * 1024 * 1024);
        ASSERT_OK_AND_ASSIGN(auto stream, opened);
        // Two parts in one Write exceeds the old native limit (max_connections=1).
        auto payload = arrow::Buffer::FromString(std::string(background ? 7 : 10 * 1024 * 1024, 'x'));
        ASSERT_OK(stream->Write(payload));
        const auto status = stream->Close();
        if (!crt && background) {
          // SDK background writes require the caller's I/O executor.
          EXPECT_FALSE(status.ok());
          continue;
        }
        // CRT uploads and synchronous SDK writes do not require that executor.
        ASSERT_OK(status);
        ASSERT_OK_AND_ASSIGN(auto input, fs->OpenInputFile(name));
        ASSERT_OK_AND_ASSIGN(auto data, input->Read(payload->size()));
        EXPECT_TRUE(data->Equals(*payload));
        ASSERT_OK(input->Close());
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
    const auto metrics = raw->GetMetrics();
    metrics->Reset();
    ASSERT_OK_AND_ASSIGN(auto out, fs->OpenConditionalOutputStream("gcp-multipart-guard", nullptr));
    const auto status = out->Write(arrow::Buffer::FromString(std::string(5 * 1024 * 1024, 'a')));
    EXPECT_TRUE(status.IsNotImplemented()) << status;
    EXPECT_EQ(metrics->GetWriteCount(), 0);
    EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 0);
    ASSERT_OK(out->Abort());
    ASSERT_OK_AND_ASSIGN(auto data, Await(Read("gcp-multipart-guard", 0, 8)));
    EXPECT_EQ(data->ToString(), "original");
  }
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
    ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream(path, nullptr));
    ASSERT_OK(stream->Write(arrow::Buffer::FromString(data)));
    ASSERT_OK(stream->Close());
    ASSERT_OK_AND_ASSIGN(auto uploaded, Await(Read(path, 0, data.size())));
    EXPECT_EQ(uploaded->ToString(), data);
  }
}

TEST_F(ParallelUploadS3Test, CombinesWritesAcrossPartBoundaries) {
  const auto expected = std::string(4 * 1024 * 1024, 'a') + std::string(6 * 1024 * 1024, 'b') + "tail";
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("parallel-cross-write", nullptr));
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
  constexpr int64_t part_size = 6 * 1024 * 1024;
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStreamWithUploadSize("held/crt-parallel", nullptr, part_size));
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
  ASSERT_OK_AND_ASSIGN(const auto first_part, control_->GetFileInfo(prefix_ + "/upload-started/crt-parallel/1"));
  ASSERT_EQ(first_part.type(), arrow::fs::FileType::File);
  EXPECT_EQ(first_part.size(), part_size);
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("crt-abort", nullptr));
  ASSERT_OK(stream->Write(data));
  ASSERT_OK(stream->Abort());
  ASSERT_OK_AND_ASSIGN(auto info, Await(Stat(fs_, "crt-abort")));
  EXPECT_EQ(info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, IdleInputDoesNotBlockAnotherUpload) {
  ASSERT_OK_AND_ASSIGN(auto idle, fs_->OpenOutputStream("parallel-idle", nullptr));
  ASSERT_OK(idle->Write(arrow::Buffer::FromString("prefix")));
  ASSERT_OK_AND_ASSIGN(auto next, fs_->OpenOutputStream("parallel-next", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("held/abort-closing", nullptr));
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
    ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("parallel-dropped", nullptr));
    ASSERT_OK(stream->Write(arrow::Buffer::FromString("payload")));
  }
  ASSERT_OK_AND_ASSIGN(auto next, fs_->OpenOutputStream("parallel-after-drop", nullptr));
  ASSERT_OK(next->Write(arrow::Buffer::FromString("next")));
  ASSERT_OK(next->Close());
  ASSERT_OK_AND_ASSIGN(auto missing, Await(Stat(fs_, "parallel-dropped")));
  EXPECT_EQ(missing.type(), arrow::fs::FileType::NotFound);
}

TEST_F(ParallelUploadS3Test, CloseContinuationUsesCallerExecutorAndCanUpload) {
  if (std::getenv("STORAGE_NATIVE_S3_REAL"))
    GTEST_SKIP() << "Requires fixture-controlled requests";
  const auto data = arrow::Buffer::FromString(std::string(10 * 1024 * 1024, 'a'));
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("held/crt-continuation", nullptr));
  auto release = folly::makeGuard([&] {
    for (int part = 1; part <= 3; ++part) (void)ControlUpload("crt-continuation/" + std::to_string(part), true);
  });
  ASSERT_OK(stream->Write(data));
  auto closed = stream->CloseAsync();
  ASSERT_OK(ControlUpload("crt-continuation/1", false));
  auto continued = closed.Then([this, data]() -> arrow::Status {
    EXPECT_TRUE(executor_->OwnsThisThread());
    ARROW_ASSIGN_OR_RAISE(auto next, fs_->OpenOutputStream("crt-continuation-next", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("parallel-flush-continuation", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("held/failed-wait", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs->OpenOutputStream(prefix_ + "/crt-cancel", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, s3->OpenOutputStream(prefix_ + "/held/crt-canceled", nullptr));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream(path, metadata));
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
  ASSERT_OK_AND_ASSIGN(auto stream, fs_->OpenOutputStream("stopped-completion", nullptr));
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
#endif
}  // namespace milvus_storage
