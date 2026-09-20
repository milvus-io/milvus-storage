// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#pragma once
#include <folly/futures/Future.h>
#include "milvus-storage/transaction/transaction.h"

namespace milvus_storage {
struct AsyncStatus {
  enum Code { OK, Cancelled, Deadline, Overloaded, Busy, Memory, Exception, Arrow };
  Code code = OK;
  arrow::Status detail;
  AsyncStatus() = default;
  AsyncStatus(Code code) : code(code) {}
  AsyncStatus(arrow::Status status)
      : code(status.ok()              ? OK
             : status.IsOutOfMemory() ? Memory
             : status.IsCancelled()   ? Cancelled
                                      : Arrow),
        detail(std::move(status)) {}
  bool ok() const { return code == OK; }
};
}  // namespace milvus_storage
namespace milvus_storage::api::transaction {
class AsyncOperation {
  public:
  virtual ~AsyncOperation() = default;
  virtual void Cancel() = 0;
};
struct BeginResult {
  AsyncStatus status;
  std::unique_ptr<Transaction> transaction;
};
// Lazy scheduling of the existing synchronous transaction API. Consume on a
// caller-owned blocking-work executor. No filesystem or transport is replaced.
// Cancellation/deadline checks happen before Open starts; in-flight synchronous
// I/O cannot be interrupted. The executor must accept tasks through completion.
// Keep the resolver alive for the lifetime of the returned transaction. Dropping
// an unconsumed future starts no work; releasing its handle does not cancel it.
folly::SemiFuture<BeginResult> BeginAsync(const std::string& path,
                                          Properties properties,
                                          int64_t version,
                                          const Resolver& resolver,
                                          uint32_t retries,
                                          uint64_t timeout_ms,
                                          std::shared_ptr<AsyncOperation>& operation,
                                          ArrowFileSystemPtr filesystem = nullptr);
enum class CommitOutcome { NotCommitted, Committed, Unknown };
struct CommitResult {
  AsyncStatus status;
  CommitOutcome outcome = CommitOutcome::NotCommitted;
  int64_t version = -1;
};
// Schedules Transaction::Commit on the consuming executor. A discarded lazy
// future releases its reservation; execution consumes the transaction even on
// cancellation/failure. Keep it alive and unmodified until completion. Errors
// from the synchronous commit conservatively carry Unknown; only pre-execution
// rejection/cancellation/deadline can guarantee NotCommitted. In-flight calls
// run to completion and success wins over cancellation or deadline expiry.
folly::SemiFuture<CommitResult> CommitAsync(Transaction* transaction,
                                            uint64_t timeout_ms,
                                            std::shared_ptr<AsyncOperation>& operation);
}  // namespace milvus_storage::api::transaction
