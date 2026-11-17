// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include <memory>
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_async_context.h"
#include <gtest/gtest.h>
#include <arrow/filesystem/mockfs.h>
#include <arrow/io/interfaces.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ManualExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <atomic>
#include <future>
#include <stdexcept>

namespace milvus_storage::api::transaction {
namespace {
struct CallerExecutors {
  folly::CPUThreadPoolExecutor work{1}, completion{1};
  static CallerExecutors& Get() {
    static CallerExecutors executors;
    return executors;
  }
};
// Persist the bytes, then report a lost response to exercise an ambiguous commit.
class FailedCloseStream : public arrow::io::OutputStream {
  public:
  FailedCloseStream(std::shared_ptr<arrow::io::OutputStream> stream, arrow::Status error)
      : stream_(std::move(stream)), error_(std::move(error)) {}
  arrow::Status Close() override {
    ARROW_RETURN_NOT_OK(stream_->Close());
    return error_;
  }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  arrow::Status Write(const void* data, int64_t size) override { return stream_->Write(data, size); }

  private:
  const std::shared_ptr<arrow::io::OutputStream> stream_;
  const arrow::Status error_;
};
class MemoryFileSystem : public arrow::fs::internal::MockFileSystem {
  public:
  MemoryFileSystem() : MockFileSystem(arrow::fs::TimePoint{}) {
    static std::atomic<int> next{0};
    path = "async-test-" + std::to_string(next++);
    EXPECT_TRUE(CreateDir(get_manifest_path(path), true).ok());
  }
  using MockFileSystem::GetFileInfo;
  std::string path;
  int info_calls = 0;
  std::thread::id io_thread;
  std::function<void()> before_info;
  arrow::Status info_error, output_error, close_error;
  using MockFileSystem::OpenOutputStream;
  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& name, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    ARROW_RETURN_NOT_OK(output_error);
    ARROW_ASSIGN_OR_RAISE(auto stream, MockFileSystem::OpenOutputStream(name, metadata));
    if (!close_error.ok())
      return std::make_shared<FailedCloseStream>(std::move(stream), close_error);
    return stream;
  }
  using MockFileSystem::OpenInputFile;
  int64_t opened_size = -1;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    opened_size = info.size();
    return MockFileSystem::OpenInputFile(info.path());
  }
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& name) override {
    ++info_calls;
    io_thread = std::this_thread::get_id();
    if (before_info)
      before_info();
    ARROW_RETURN_NOT_OK(info_error);
    return MockFileSystem::GetFileInfo(name);
  }
};
OpenResult Open(const std::shared_ptr<MemoryFileSystem>& fs,
                int64_t version = -1,
                const Resolver& resolver = FailResolver,
                uint32_t retries = 0) {
  std::shared_ptr<AsyncOperation> operation;
  auto future = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), version, -1, resolver, retries,
                                       3000, operation, fs);
  operation.reset();
  return std::move(future).via(&CallerExecutors::Get().work).get();
}
TEST(AsyncTransactionTest, UnconsumedFutureStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  {
    auto future = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), -1, -1, FailResolver, 0, 3000,
                                         operation, fs);
  }
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, SynchronousIOUsesCallerExecutor) {
  auto fs = std::make_shared<MemoryFileSystem>();
  folly::CPUThreadPoolExecutor executor(1);
  auto worker = folly::via(&executor, [] { return std::this_thread::get_id(); }).get();
  std::shared_ptr<AsyncOperation> operation;
  auto result = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), -1, -1, FailResolver, 0, 3000,
                                       operation, fs)
                    .via(&executor)
                    .get();
  ASSERT_TRUE(result.status.ok());
  EXPECT_EQ(fs->io_thread, worker);
  EXPECT_EQ(result.transaction->GetReadVersion(), 0);
  executor.join();  // Completed transactions/handles cannot pin the executor.
}
TEST(AsyncTransactionTest, ContinuationCanUseAnotherExecutor) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto& executors = CallerExecutors::Get();
  auto completion = folly::via(&executors.completion, [] { return std::this_thread::get_id(); }).get();
  std::shared_ptr<AsyncOperation> operation;
  auto result = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), 0, -1, FailResolver, 0, 3000,
                                       operation, fs)
                    .via(&executors.work)
                    .via(&executors.completion)
                    .thenValue([completion](OpenResult result) {
                      EXPECT_EQ(std::this_thread::get_id(), completion);
                      return result;
                    })
                    .get();
  EXPECT_TRUE(result.status.ok());
}
TEST(AsyncTransactionTest, CancelledQueuedOpenStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  auto future = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), -1, -1, FailResolver, 0, 3000,
                                       operation, fs);
  operation->Cancel();
  EXPECT_EQ(std::move(future).via(&CallerExecutors::Get().work).get().status.code, AsyncStatus::Cancelled);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, DeadlineIncludesQueueTime) {
  auto fs = std::make_shared<MemoryFileSystem>();
  folly::ManualExecutor executor;
  std::shared_ptr<AsyncOperation> operation;
  auto future =
      Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), -1, -1, FailResolver, 0, 1, operation, fs)
          .via(&executor);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  executor.drain();
  EXPECT_EQ(std::move(future).get().status.code, AsyncStatus::Deadline);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, InlineExecutorIsRejected) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  auto result = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), 0, -1, FailResolver, 0, 3000,
                                       operation, fs)
                    .via(&folly::InlineExecutor::instance())
                    .get();
  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, BlockingIOLeavesSubmissionFreeAndCancellationWaits) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto gate = release.get_future();
  fs->before_info = [&] {
    entered.set_value();
    gate.wait();
  };
  std::shared_ptr<AsyncOperation> operation;
  auto future = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), -1, -1, FailResolver, 0, 3000,
                                       operation, fs)
                    .via(&CallerExecutors::Get().work);
  EXPECT_EQ(entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  operation->Cancel();
  EXPECT_FALSE(future.isReady());  // In-flight synchronous calls retain ownership.
  release.set_value();
  EXPECT_TRUE(std::move(future).get().status.ok());
}
TEST(AsyncTransactionTest, ReturnedTransactionUsesExistingSyncCommitAndRead) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Open(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
  auto written = begun.transaction->Commit();
  ASSERT_TRUE(written.ok()) << written.status();
  EXPECT_EQ(*written, 1);
  auto read = Open(fs);
  ASSERT_TRUE(read.status.ok());
  EXPECT_EQ(read.transaction->GetReadVersion(), 1);
  auto manifest = read.transaction->GetManifest();
  ASSERT_TRUE(manifest.ok());
  ASSERT_EQ((*manifest)->deltaLogs().size(), 1);
  EXPECT_EQ((*manifest)->deltaLogs()[0].path, get_delta_filepath(fs->path, "delete"));
}
CommitResult Commit(Transaction* txn) {
  std::shared_ptr<AsyncOperation> operation;
  auto future = txn->CommitAsync(3000, operation);
  operation.reset();
  return std::move(future).via(&CallerExecutors::Get().work).get();
}

struct FFICommitResult {
  LoonFFIResult status;
  int64_t version;
};
FFICommitResult CommitFFI(Transaction* txn, bool async) {
  const auto handle = reinterpret_cast<LoonTransactionHandle>(txn);
  if (!async) {
    int64_t version = -1;
    auto status = loon_transaction_commit(handle, &version, nullptr);
    return {status, version};
  }
  const LoonAsyncExecutor executor{
      &CallerExecutors::Get().work, [](void* context, LoonAsyncTask task, void* data) -> int32_t {
        static_cast<folly::CPUThreadPoolExecutor*>(context)->add([task, data] { task(data); });
        return 0;
      }};
  LoonAsyncContextHandle context = nullptr;
  auto status = loon_async_context_create(&executor, &context);
  if (status.err_code != LOON_SUCCESS)
    return {status, -1};
  loon_ffi_free_result(&status);
  std::promise<FFICommitResult> completion;
  auto future = completion.get_future();
  LoonAsyncHandle operation{};
  status = loon_transaction_commit_async(
      context, handle,
      [](uintptr_t data, LoonFFIResult status, int32_t outcome, int64_t version, int64_t size) {
        EXPECT_EQ(outcome, status.err_code == LOON_SUCCESS ? LOON_COMMIT_COMMITTED : LOON_COMMIT_UNKNOWN);
        if (status.err_code != LOON_SUCCESS) {
          EXPECT_EQ(version, -1);
          EXPECT_EQ(size, -1);
        }
        reinterpret_cast<std::promise<FFICommitResult>*>(data)->set_value({status, version});
      },
      reinterpret_cast<uintptr_t>(&completion), &operation);
  FFICommitResult result{status, -1};
  if (status.err_code == LOON_SUCCESS) {
    loon_ffi_free_result(&status);
    result = future.get();
  }
  loon_async_release(&operation);
  loon_async_context_destroy(context);
  return result;
}

TEST(AsyncTransactionTest, CommitPreservesStorageErrorClassification) {
  for (const auto& error :
       {MakeExtendError(ExtendStatusCode::StorageTransientService, "manifest write failed", "request-id"),
        MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "manifest write failed", "request-id"),
        arrow::Status::Invalid("manifest write failed")}) {
    for (bool async : {false, true}) {
      auto fs = std::make_shared<MemoryFileSystem>();
      auto begun = Open(fs, 0);
      ASSERT_TRUE(begun.status.ok());
      begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
      fs->output_error = error;
      const auto status = async ? Commit(begun.transaction.get()).status.detail : begun.transaction->Commit().status();
      EXPECT_EQ(status.code(), fs->output_error.code());
      EXPECT_EQ(status.detail(), fs->output_error.detail());
      EXPECT_NE(status.message().find("manifest write failed"), std::string::npos);
      EXPECT_NE(status.message().find("read_version=0"), std::string::npos);
    }
  }
}

TEST(AsyncTransactionTest, CommitFFIPreservesStorageErrorClassification) {
  for (const auto code : {ExtendStatusCode::StorageTransientNetwork, ExtendStatusCode::StorageTransientTimeout,
                          ExtendStatusCode::StorageTransientThrottling, ExtendStatusCode::StorageTransientService,
                          ExtendStatusCode::AwsErrorNotFound, ExtendStatusCode::AwsErrorAccessDenied,
                          ExtendStatusCode::AwsErrorNonRetryable}) {
    const bool retryable =
        code == ExtendStatusCode::StorageTransientNetwork || code == ExtendStatusCode::StorageTransientTimeout ||
        code == ExtendStatusCode::StorageTransientThrottling || code == ExtendStatusCode::StorageTransientService;
    for (bool async : {false, true}) {
      for (const auto stage : {"metadata", "open-output", "close-output"}) {
        SCOPED_TRACE(std::to_string(static_cast<int>(code)) + "/" + stage + (async ? "/async" : "/sync"));
        auto fs = std::make_shared<MemoryFileSystem>();
        auto begun = Open(fs, 0);
        ASSERT_TRUE(begun.status.ok());
        begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
        auto& error = std::string(stage) == "metadata"      ? fs->info_error
                      : std::string(stage) == "open-output" ? fs->output_error
                                                            : fs->close_error;
        error = MakeExtendError(code, "storage request failed", "request-id-54386");
        auto result = CommitFFI(begun.transaction.get(), async);
        EXPECT_EQ(result.status.err_code, static_cast<int>(code));
        EXPECT_EQ(loon_ffi_is_retryable_errcode(result.status.err_code) != 0, retryable);
        EXPECT_NE(std::string(result.status.message ? result.status.message : "").find("request-id-54386"),
                  std::string::npos);
        EXPECT_EQ(result.version, -1);
        loon_ffi_free_result(&result.status);
        if (retryable && std::string(stage) == "open-output") {
          fs->output_error = arrow::Status::OK();
          auto retried = Open(fs, 0, OverwriteResolver);
          ASSERT_TRUE(retried.status.ok());
          retried.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
          auto success = CommitFFI(retried.transaction.get(), async);
          EXPECT_EQ(success.status.err_code, LOON_SUCCESS);
          EXPECT_EQ(success.version, 1);
          loon_ffi_free_result(&success.status);
        }
      }
    }
  }
}

TEST(AsyncTransactionTest, RetryFromFixedVersionAfterLostCommitResponseDoesNotDuplicateUpdates) {
  for (bool async : {false, true}) {
    auto fs = std::make_shared<MemoryFileSystem>();
    auto begun = Open(fs, 0, OverwriteResolver);
    ASSERT_TRUE(begun.status.ok());
    const auto file = fs->path + "/_data/data.parquet";
    const auto delta = get_delta_filepath(fs->path, "delete");
    auto group = std::make_shared<ColumnGroup>();
    group->columns = {"100"};
    group->format = "parquet";
    group->files = {{.path = file, .start_index = 0, .end_index = 3}};
    begun.transaction->AppendFiles({group});
    begun.transaction->AddDeltaLog({delta, DeltaLogType::PRIMARY_KEY, 3});
    const Statistics stats{{fs->path + "/_stats/bloom"}, {{"rows", "3"}}};
    begun.transaction->UpdateStat("bloom_filter.100", stats);
    fs->close_error = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "response lost", "request-id");
    auto failed = CommitFFI(begun.transaction.get(), async);
    EXPECT_EQ(failed.status.err_code, LOON_TRANSIENT_TIMEOUT);
    loon_ffi_free_result(&failed.status);

    fs->close_error = arrow::Status::OK();
    // The caller retries with its original version and updates, never LATEST.
    auto retried = Open(fs, 0, OverwriteResolver);
    ASSERT_TRUE(retried.status.ok());
    retried.transaction->AppendFiles({group});
    retried.transaction->AddDeltaLog({delta, DeltaLogType::PRIMARY_KEY, 3});
    retried.transaction->UpdateStat("bloom_filter.100", stats);
    auto success = CommitFFI(retried.transaction.get(), async);
    EXPECT_EQ(success.status.err_code, LOON_SUCCESS);
    EXPECT_EQ(success.version, 2);
    loon_ffi_free_result(&success.status);
    auto read = Open(fs);
    ASSERT_TRUE(read.status.ok());
    const auto manifest = read.transaction->GetManifest();
    ASSERT_TRUE(manifest.ok());
    ASSERT_EQ((*manifest)->columnGroups().size(), 1);
    ASSERT_EQ((*manifest)->columnGroups()[0]->files.size(), 1);
    EXPECT_EQ((*manifest)->columnGroups()[0]->files[0].path, file);
    EXPECT_EQ((*manifest)->columnGroups()[0]->files[0].end_index, 3);
    ASSERT_EQ((*manifest)->deltaLogs().size(), 1);
    EXPECT_EQ((*manifest)->deltaLogs()[0].path, delta);
    EXPECT_EQ((*manifest)->deltaLogs()[0].num_entries, 3);
    ASSERT_EQ((*manifest)->stats().size(), 1);
    EXPECT_EQ((*manifest)->stats().at("bloom_filter.100").paths, stats.paths);
    EXPECT_EQ((*manifest)->stats().at("bloom_filter.100").metadata.at("rows"), "3");
  }
}

TEST(AsyncTransactionTest, OpenPreservesExceptionMessage) {
  auto fs = std::make_shared<MemoryFileSystem>();
  fs->before_info = [] { throw std::runtime_error("metadata lookup exception"); };
  const auto result = Open(fs);
  EXPECT_EQ(result.status.code, AsyncStatus::Exception);
  EXPECT_NE(result.status.detail.message().find("metadata lookup exception"), std::string::npos);
  EXPECT_EQ(result.transaction, nullptr);
}

TEST(AsyncTransactionTest, CommitFFIPreservesExceptionMessage) {
  for (bool async : {false, true}) {
    auto fs = std::make_shared<MemoryFileSystem>();
    auto begun = Open(fs, 0);
    ASSERT_TRUE(begun.status.ok());
    begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
    fs->before_info = [] { throw std::runtime_error("commit metadata exception"); };
    auto result = CommitFFI(begun.transaction.get(), async);
    EXPECT_EQ(result.status.err_code, LOON_GOT_EXCEPTION);
    EXPECT_NE(std::string(result.status.message ? result.status.message : "").find("commit metadata exception"),
              std::string::npos);
    loon_ffi_free_result(&result.status);
  }
}
TEST(AsyncTransactionTest, DiscardedCommitReleasesReservation) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Open(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  std::shared_ptr<AsyncOperation> retained;
  {
    auto future = begun.transaction->CommitAsync(3000, retained);
    std::shared_ptr<AsyncOperation> other;
    auto busy = begun.transaction->CommitAsync(3000, other);
    EXPECT_TRUE(busy.isReady());
    EXPECT_EQ(std::move(busy).get().status.code, AsyncStatus::Busy);
    EXPECT_EQ(fs->info_calls, 0);
  }
  auto committed = Commit(begun.transaction.get());
  EXPECT_TRUE(committed.status.ok());
  EXPECT_EQ(committed.outcome, CommitOutcome::Committed);
  EXPECT_EQ(committed.version, 1);
}
TEST(AsyncTransactionTest, ManifestSizeRoundTrip) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Open(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
  const auto written = Commit(begun.transaction.get());
  ASSERT_TRUE(written.status.ok());
  ASSERT_GT(written.manifest_size, 0);
  const auto info = fs->GetFileInfo(get_manifest_filepath(fs->path, written.version));
  ASSERT_TRUE(info.ok());
  EXPECT_EQ(written.manifest_size, info->size());
  Manifest::CleanCache();
  std::shared_ptr<AsyncOperation> operation;
  auto read = Transaction::OpenAsync(fs->path, std::make_shared<const Properties>(), written.version,
                                     written.manifest_size, FailResolver, 0, 3000, operation, fs)
                  .via(&CallerExecutors::Get().work)
                  .get();
  ASSERT_TRUE(read.status.ok());
  EXPECT_EQ(fs->opened_size, written.manifest_size);
  const auto manifest = read.transaction->GetManifest();
  ASSERT_TRUE(manifest.ok());
  ASSERT_EQ((*manifest)->deltaLogs().size(), 1);
  EXPECT_EQ((*manifest)->deltaLogs()[0].num_entries, 3);
  const auto failed = Commit(read.transaction.get());
  EXPECT_FALSE(failed.status.ok());
  EXPECT_EQ(failed.manifest_size, -1);
}
TEST(AsyncTransactionTest, QueuedCommitCancellationStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Open(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  std::shared_ptr<AsyncOperation> operation;
  auto future = begun.transaction->CommitAsync(3000, operation);
  operation->Cancel();
  auto result = std::move(future).via(&CallerExecutors::Get().work).get();
  EXPECT_EQ(result.status.code, AsyncStatus::Cancelled);
  EXPECT_EQ(result.outcome, CommitOutcome::NotCommitted);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, CommitUsesAnotherExecutorAndPreservesSuccessAfterCancellation) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Open(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  folly::CPUThreadPoolExecutor executor(1);
  auto thread = folly::via(&executor, [] { return std::this_thread::get_id(); }).get();
  std::shared_ptr<AsyncOperation> operation;
  auto future = begun.transaction->CommitAsync(3000, operation);
  fs->before_info = [&] { operation->Cancel(); };
  auto result = std::move(future).via(&executor).get();
  EXPECT_TRUE(result.status.ok());
  EXPECT_EQ(result.outcome, CommitOutcome::Committed);
  EXPECT_EQ(result.version, 1);
  EXPECT_EQ(fs->io_thread, thread);
  executor.join();
}
TEST(AsyncTransactionTest, CommitErrorsConservativelyReportUnknown) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto stale = Open(fs, 0);
  auto winner = Open(fs, 0);
  ASSERT_TRUE(stale.status.ok());
  ASSERT_TRUE(winner.status.ok());
  winner.transaction->AddDeltaLog({get_delta_filepath(fs->path, "winner"), DeltaLogType::PRIMARY_KEY, 1});
  ASSERT_TRUE(Commit(winner.transaction.get()).status.ok());
  stale.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  auto result = Commit(stale.transaction.get());
  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(result.outcome, CommitOutcome::Unknown);
  EXPECT_EQ(result.version, -1);
}
TEST(AsyncTransactionTest, CustomResolverReusesSynchronousLatestManifestBehavior) {
  class MergeResolver final : public Resolver {
 public:
    bool requireLatest() const override { return true; }
    arrow::Result<std::shared_ptr<Manifest>> resolve(const std::shared_ptr<Manifest>&,
                                                     int64_t,
                                                     const std::shared_ptr<Manifest>& latest,
                                                     int64_t,
                                                     const Updates& updates) const override {
      if (!latest)
        return arrow::Status::Invalid("Expected latest manifest");
      return applyUpdates(latest, updates);
    }
  } resolver;
  auto fs = std::make_shared<MemoryFileSystem>();
  auto stale = Open(fs, 0, resolver);
  auto winner = Open(fs, 0);
  ASSERT_TRUE(stale.status.ok());
  ASSERT_TRUE(winner.status.ok());
  stale.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  winner.transaction->AddDeltaLog({get_delta_filepath(fs->path, "winner"), DeltaLogType::PRIMARY_KEY, 1});
  ASSERT_TRUE(Commit(winner.transaction.get()).status.ok());
  auto result = Commit(stale.transaction.get());
  ASSERT_TRUE(result.status.ok());
  EXPECT_EQ(result.version, 2);
  auto read = Open(fs);
  ASSERT_TRUE(read.status.ok());
  auto manifest = read.transaction->GetManifest();
  ASSERT_TRUE(manifest.ok());
  ASSERT_EQ((*manifest)->deltaLogs().size(), 2);
  EXPECT_EQ((*manifest)->deltaLogs()[0].path, get_delta_filepath(fs->path, "winner"));
  EXPECT_EQ((*manifest)->deltaLogs()[1].path, get_delta_filepath(fs->path, "own"));
}
}  // namespace
}  // namespace milvus_storage::api::transaction
