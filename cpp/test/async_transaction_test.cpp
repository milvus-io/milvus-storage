// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include "milvus-storage/transaction/async_transaction.h"
#include "milvus-storage/common/async_limits.h"
#include "milvus-storage/common/layout.h"
#include <gtest/gtest.h>
#include <arrow/filesystem/mockfs.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ManualExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <atomic>
#include <future>

namespace milvus_storage::api::transaction {
namespace {
struct CallerExecutors {
  folly::CPUThreadPoolExecutor work{1}, completion{1};
  static CallerExecutors& Get() {
    static CallerExecutors executors;
    return executors;
  }
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
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& name) override {
    ++info_calls;
    io_thread = std::this_thread::get_id();
    if (before_info)
      before_info();
    return MockFileSystem::GetFileInfo(name);
  }
};
BeginResult Begin(const std::shared_ptr<MemoryFileSystem>& fs,
                  int64_t version = -1,
                  const Resolver& resolver = FailResolver,
                  uint32_t retries = 0) {
  std::shared_ptr<AsyncOperation> operation;
  auto future = BeginAsync(fs->path, {}, version, resolver, retries, 3000, operation, fs);
  operation.reset();
  return std::move(future).via(&CallerExecutors::Get().work).get();
}
TEST(AsyncManifestLimitsTest, InvalidConfigurationReturnsStatus) {
  const char* name = "LOON_TEST_ASYNC_LIMIT_SETTING";
  ASSERT_EQ(setenv(name, "invalid", 1), 0);
  auto invalid = AsyncManifestLimits::Setting(name, 8, 16);
  ASSERT_EQ(unsetenv(name), 0);
  ASSERT_FALSE(invalid.ok());
  EXPECT_TRUE(invalid.status().IsInvalid());
}
TEST(AsyncTransactionTest, UnconsumedFutureStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  { auto future = BeginAsync(fs->path, {}, -1, FailResolver, 0, 3000, operation, fs); }
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, SynchronousIOUsesCallerExecutor) {
  auto fs = std::make_shared<MemoryFileSystem>();
  folly::CPUThreadPoolExecutor executor(1);
  auto worker = folly::via(&executor, [] { return std::this_thread::get_id(); }).get();
  std::shared_ptr<AsyncOperation> operation;
  auto result = BeginAsync(fs->path, {}, -1, FailResolver, 0, 3000, operation, fs).via(&executor).get();
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
  auto result = BeginAsync(fs->path, {}, 0, FailResolver, 0, 3000, operation, fs)
                    .via(&executors.work)
                    .via(&executors.completion)
                    .thenValue([completion](BeginResult result) {
                      EXPECT_EQ(std::this_thread::get_id(), completion);
                      return result;
                    })
                    .get();
  EXPECT_TRUE(result.status.ok());
}
TEST(AsyncTransactionTest, CancelledQueuedBeginStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  auto future = BeginAsync(fs->path, {}, -1, FailResolver, 0, 3000, operation, fs);
  operation->Cancel();
  EXPECT_EQ(std::move(future).via(&CallerExecutors::Get().work).get().status.code, AsyncStatus::Cancelled);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, DeadlineIncludesQueueTime) {
  auto fs = std::make_shared<MemoryFileSystem>();
  folly::ManualExecutor executor;
  std::shared_ptr<AsyncOperation> operation;
  auto future = BeginAsync(fs->path, {}, -1, FailResolver, 0, 1, operation, fs).via(&executor);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  executor.drain();
  EXPECT_EQ(std::move(future).get().status.code, AsyncStatus::Deadline);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, InlineExecutorIsRejected) {
  auto fs = std::make_shared<MemoryFileSystem>();
  std::shared_ptr<AsyncOperation> operation;
  auto result =
      BeginAsync(fs->path, {}, 0, FailResolver, 0, 3000, operation, fs).via(&folly::InlineExecutor::instance()).get();
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
  auto future = BeginAsync(fs->path, {}, -1, FailResolver, 0, 3000, operation, fs).via(&CallerExecutors::Get().work);
  EXPECT_EQ(entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  operation->Cancel();
  EXPECT_FALSE(future.isReady());  // In-flight synchronous calls retain ownership.
  release.set_value();
  EXPECT_TRUE(std::move(future).get().status.ok());
}
TEST(AsyncTransactionTest, ReturnedTransactionUsesExistingSyncCommitAndRead) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Begin(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "delete"), DeltaLogType::PRIMARY_KEY, 3});
  auto written = begun.transaction->Commit();
  ASSERT_TRUE(written.ok()) << written.status();
  EXPECT_EQ(*written, 1);
  auto read = Begin(fs);
  ASSERT_TRUE(read.status.ok());
  EXPECT_EQ(read.transaction->GetReadVersion(), 1);
  auto manifest = read.transaction->GetManifest();
  ASSERT_TRUE(manifest.ok());
  ASSERT_EQ((*manifest)->deltaLogs().size(), 1);
  EXPECT_EQ((*manifest)->deltaLogs()[0].path, get_delta_filepath(fs->path, "delete"));
}
CommitResult Commit(Transaction* txn) {
  std::shared_ptr<AsyncOperation> operation;
  auto future = CommitAsync(txn, 3000, operation);
  operation.reset();
  return std::move(future).via(&CallerExecutors::Get().work).get();
}
TEST(AsyncTransactionTest, DiscardedCommitReleasesReservation) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Begin(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  std::shared_ptr<AsyncOperation> retained;
  {
    auto future = CommitAsync(begun.transaction.get(), 3000, retained);
    std::shared_ptr<AsyncOperation> other;
    auto busy = CommitAsync(begun.transaction.get(), 3000, other);
    EXPECT_TRUE(busy.isReady());
    EXPECT_EQ(std::move(busy).get().status.code, AsyncStatus::Busy);
    EXPECT_EQ(fs->info_calls, 0);
  }
  auto committed = Commit(begun.transaction.get());
  EXPECT_TRUE(committed.status.ok());
  EXPECT_EQ(committed.outcome, CommitOutcome::Committed);
  EXPECT_EQ(committed.version, 1);
}
TEST(AsyncTransactionTest, QueuedCommitCancellationStartsNoIO) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Begin(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  std::shared_ptr<AsyncOperation> operation;
  auto future = CommitAsync(begun.transaction.get(), 3000, operation);
  operation->Cancel();
  auto result = std::move(future).via(&CallerExecutors::Get().work).get();
  EXPECT_EQ(result.status.code, AsyncStatus::Cancelled);
  EXPECT_EQ(result.outcome, CommitOutcome::NotCommitted);
  EXPECT_EQ(fs->info_calls, 0);
}
TEST(AsyncTransactionTest, CommitUsesAnotherExecutorAndPreservesSuccessAfterCancellation) {
  auto fs = std::make_shared<MemoryFileSystem>();
  auto begun = Begin(fs, 0);
  ASSERT_TRUE(begun.status.ok());
  begun.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  folly::CPUThreadPoolExecutor executor(1);
  auto thread = folly::via(&executor, [] { return std::this_thread::get_id(); }).get();
  std::shared_ptr<AsyncOperation> operation;
  auto future = CommitAsync(begun.transaction.get(), 3000, operation);
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
  auto stale = Begin(fs, 0);
  auto winner = Begin(fs, 0);
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
  auto stale = Begin(fs, 0, resolver);
  auto winner = Begin(fs, 0);
  ASSERT_TRUE(stale.status.ok());
  ASSERT_TRUE(winner.status.ok());
  stale.transaction->AddDeltaLog({get_delta_filepath(fs->path, "own"), DeltaLogType::PRIMARY_KEY, 1});
  winner.transaction->AddDeltaLog({get_delta_filepath(fs->path, "winner"), DeltaLogType::PRIMARY_KEY, 1});
  ASSERT_TRUE(Commit(winner.transaction.get()).status.ok());
  auto result = Commit(stale.transaction.get());
  ASSERT_TRUE(result.status.ok());
  EXPECT_EQ(result.version, 2);
  auto read = Begin(fs);
  ASSERT_TRUE(read.status.ok());
  auto manifest = read.transaction->GetManifest();
  ASSERT_TRUE(manifest.ok());
  ASSERT_EQ((*manifest)->deltaLogs().size(), 2);
  EXPECT_EQ((*manifest)->deltaLogs()[0].path, get_delta_filepath(fs->path, "winner"));
  EXPECT_EQ((*manifest)->deltaLogs()[1].path, get_delta_filepath(fs->path, "own"));
}
}  // namespace
}  // namespace milvus_storage::api::transaction
