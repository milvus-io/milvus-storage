// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include "milvus-storage/transaction/async_transaction.h"
#include <atomic>
#include <chrono>
#include <folly/executors/InlineExecutor.h>

namespace milvus_storage::api::transaction {
struct AsyncManifestOperation final : AsyncOperation {
  static bool Consume(Transaction* txn) { return !txn->async_consumed_.exchange(true); }
  static void Unconsume(Transaction* txn) { txn->async_consumed_.store(false); }
  std::atomic<bool> cancelled{false};
  std::chrono::steady_clock::time_point deadline;
  void Cancel() override { cancelled.store(true, std::memory_order_relaxed); }
  AsyncStatus BeforeStart(const folly::Executor::KeepAlive<>& executor) const {
    if (!executor || dynamic_cast<folly::InlineLikeExecutor*>(executor.get()))
      return arrow::Status::Invalid("Async transactions require a non-inline executor for blocking work");
    if (cancelled.load(std::memory_order_relaxed))
      return AsyncStatus::Cancelled;
    if (std::chrono::steady_clock::now() >= deadline)
      return AsyncStatus::Deadline;
    return {};
  }
};

folly::SemiFuture<BeginResult> BeginAsync(const std::string& path,
                                          Properties properties,
                                          int64_t version,
                                          const Resolver& resolver,
                                          uint32_t retries,
                                          uint64_t timeout_ms,
                                          std::shared_ptr<AsyncOperation>& operation,
                                          ArrowFileSystemPtr filesystem) {
  operation.reset();
  if (version < -1 || !timeout_ms || timeout_ms > 24 * 60 * 60 * 1000)
    return folly::makeSemiFuture(BeginResult{arrow::Status::Invalid("Invalid async transaction arguments"), nullptr});
  auto state = std::make_shared<AsyncManifestOperation>();
  state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  operation = state;
  return folly::makeSemiFuture().deferExValue(
      [state, path, properties = std::move(properties), version, resolver = &resolver, retries,
       filesystem = std::move(filesystem)](folly::Executor::KeepAlive<> executor, folly::Unit) mutable -> BeginResult {
        try {
          auto status = state->BeforeStart(executor);
          if (!status.ok())
            return {std::move(status), nullptr};
          if (!filesystem) {
            auto cached = FilesystemCache::getInstance().get(properties, path);
            if (!cached.ok())
              return {cached.status(), nullptr};
            filesystem = std::move(cached).ValueUnsafe();
          }
          status = state->BeforeStart(executor);
          if (!status.ok())
            return {std::move(status), nullptr};
          auto result = Transaction::Open(filesystem, path, version, *resolver, retries);
          if (!result.ok())
            return {result.status(), nullptr};
          return {{}, std::move(result).ValueUnsafe()};
        } catch (const std::bad_alloc&) {
          return {AsyncStatus::Memory, nullptr};
        } catch (...) {
          return {AsyncStatus::Exception, nullptr};
        }
      });
}
folly::SemiFuture<CommitResult> CommitAsync(Transaction* transaction,
                                            uint64_t timeout_ms,
                                            std::shared_ptr<AsyncOperation>& operation) {
  operation.reset();
  if (!transaction || !timeout_ms || timeout_ms > 24 * 60 * 60 * 1000)
    return folly::makeSemiFuture(CommitResult{arrow::Status::Invalid("Invalid async transaction arguments")});
  auto state = std::make_shared<AsyncManifestOperation>();
  state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  if (!AsyncManifestOperation::Consume(transaction))
    return folly::makeSemiFuture(CommitResult{AsyncStatus::Busy});
  std::unique_ptr<Transaction, decltype(&AsyncManifestOperation::Unconsume)> reservation(
      transaction, &AsyncManifestOperation::Unconsume);
  operation = state;
  return folly::makeSemiFuture().deferExValue(
      [state, transaction, reservation = std::move(reservation)](folly::Executor::KeepAlive<> executor,
                                                                 folly::Unit) mutable -> CommitResult {
        reservation.release();  // Execution consumes the transaction, even if cancelled.
        bool started = false;
        try {
          auto status = state->BeforeStart(executor);
          if (!status.ok())
            return {std::move(status)};
          started = true;
          auto result = transaction->Commit();
          // The existing synchronous API does not expose a write-dispatch or
          // durability outcome. Conservatively preserve uncertainty on errors.
          if (!result.ok())
            return {result.status(), CommitOutcome::Unknown};
          // Cancellation after execution began cannot erase a confirmed success.
          return {{}, CommitOutcome::Committed, *result};
        } catch (const std::bad_alloc&) {
          return {AsyncStatus::Memory, started ? CommitOutcome::Unknown : CommitOutcome::NotCommitted};
        } catch (...) {
          return {AsyncStatus::Exception, started ? CommitOutcome::Unknown : CommitOutcome::NotCommitted};
        }
      });
}
}  // namespace milvus_storage::api::transaction
