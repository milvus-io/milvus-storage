// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/transaction/async_transaction.h"
#include "milvus-storage/common/async_limits.h"
#include <folly/Executor.h>
#include <folly/executors/InlineExecutor.h>
#include <condition_variable>
#include <mutex>

using namespace milvus_storage;
struct LoonAsyncOperation {
  std::shared_ptr<api::transaction::AsyncOperation> state;
};
namespace {
// Scheduling adapter only: all threads and queues belong to the caller.
class ExternalExecutor final : public folly::Executor {
  public:
  explicit ExternalExecutor(LoonAsyncExecutor descriptor) : descriptor_(descriptor) {}
  bool Submit(folly::Func task) { return TrySubmit(task); }
  void add(folly::Func task) override {
    // Initial admission uses Submit. Once accepted, Folly tasks cannot be
    // dropped: an exceptional queue failure completes them on this thread.
    try {
      if (TrySubmit(task))
        return;
    } catch (...) {
    }
    task();
  }

  private:
  bool TrySubmit(folly::Func& task) {
    auto* raw = new folly::Func(std::move(task));
    auto rejected = descriptor_.submit(
        descriptor_.context,
        [](void* data) {
          std::unique_ptr<folly::Func> owned(static_cast<folly::Func*>(data));
          (*owned)();
        },
        raw);
    if (rejected) {
      task = std::move(*raw);
      delete raw;
    }
    return !rejected;
  }
  LoonAsyncExecutor descriptor_;
};

// C callback admission/lifecycle only. Native C++ operations inherit executors
// from their consuming SemiFuture and do not use this process-wide C bridge.
class Runtime {
  public:
  static Runtime& Instance() {
    static Runtime runtime;
    return runtime;
  }
  arrow::Status Configure(const LoonAsyncExecutor& executor) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || executor_)
      return arrow::Status::Invalid("Async executor already configured or runtime stopped");
    executor_ = std::make_shared<ExternalExecutor>(executor);
    return arrow::Status::OK();
  }
  struct Admission {
    Runtime* runtime = nullptr;
    ~Admission() {
      if (runtime) {
        std::lock_guard<std::mutex> lock(runtime->mutex_);
        --runtime->active_;
        runtime->drained_.notify_all();
      }
    }
  };
  template <typename T, typename Callback>
  AsyncStatus Submit(folly::SemiFuture<T> future, Callback callback) {
    auto admission = std::make_shared<Admission>();
    std::shared_ptr<ExternalExecutor> executor;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!executor_)
        return arrow::Status::Invalid("Caller must configure C async executor before submission");
      const auto& limits = AsyncManifestLimits::Get();
      if (!limits.ok())
        return limits.status();
      if (stopping_ || active_ >= limits->max_operations)
        return AsyncStatus::Overloaded;
      executor = executor_;
      ++active_;
      admission->runtime = this;
    }
    struct Delivery {
      Callback callback;
      std::shared_ptr<Admission> admission;
      folly::Try<T> result;
      std::atomic<bool> delivered{false};
      Delivery(Callback cb, std::shared_ptr<Admission> lease) : callback(std::move(cb)), admission(std::move(lease)) {}
      void Run() {
        if (!delivered.exchange(true))
          callback(std::move(result));
      }
    };
    auto delivery = std::make_shared<Delivery>(std::move(callback), admission);
    // Initial enqueue rejection is synchronous and starts no native work.
    bool accepted = executor->Submit([future = std::move(future), executor, delivery]() mutable {
      try {
        // Already executing inside this exact caller executor. Bind deferred
        // work inline here to avoid a second, potentially rejected initial add.
        std::move(future)
            .viaInlineUnsafe(folly::getKeepAliveToken(executor.get()))
            .via(&folly::InlineExecutor::instance())
            .thenTry([executor, delivery](folly::Try<T>&& result) mutable {
              // Observe the result without an intermediate queue: a rejected
              // completion enqueue must not erase a confirmed write outcome.
              delivery->result = std::move(result);
              try {
                executor->add([delivery] { delivery->Run(); });
              } catch (...) {
                delivery->Run();
              }
            });
      } catch (...) {
        delivery->result = folly::Try<T>(folly::exception_wrapper(std::current_exception()));
        delivery->Run();
      }
    });
    return accepted ? AsyncStatus{} : AsyncStatus{AsyncStatus::Overloaded};
  }
  void Shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    stopping_ = true;
    drained_.wait(lock, [this] { return active_ == 0; });
  }
  ~Runtime() { Shutdown(); }

  private:
  std::mutex mutex_;
  std::condition_variable drained_;
  std::shared_ptr<ExternalExecutor> executor_;
  size_t active_ = 0;
  bool stopping_ = false;
};
LoonFFIResult StatusResult(const AsyncStatus& status) noexcept {
  int code = LOON_SUCCESS;
  switch (status.code) {
    case AsyncStatus::OK:
      return {LOON_SUCCESS, nullptr};
    case AsyncStatus::Cancelled:
      code = LOON_ASYNC_CANCELLED;
      break;
    case AsyncStatus::Deadline:
      code = LOON_ASYNC_DEADLINE;
      break;
    case AsyncStatus::Overloaded:
      code = LOON_ASYNC_OVERLOADED;
      break;
    case AsyncStatus::Busy:
      code = LOON_ASYNC_BUSY;
      break;
    case AsyncStatus::Memory:
      return {LOON_MEMORY_ERROR, nullptr};
    case AsyncStatus::Exception:
      return {LOON_GOT_EXCEPTION, nullptr};
    case AsyncStatus::Arrow:
      code = FFIErrorCodeFromExtendStatus(status.detail);
      break;
  }
  try {
    return CreateFFIResult(code, status.detail.ok() ? "" : status.detail.ToString());
  } catch (...) {
    return {code, nullptr};
  }
}
LoonFFIResult ValidateOptions(const LoonAsyncOptions* options, uint64_t& timeout) {
  if (options && (options->struct_size < sizeof(LoonAsyncOptions) || options->flags != 0))
    return CreateFFIResult(LOON_INVALID_ARGS, "Invalid async options size or flags");
  timeout = options && options->timeout_ms ? options->timeout_ms : 30000;
  if (timeout > 24 * 60 * 60 * 1000)
    return CreateFFIResult(LOON_INVALID_ARGS, "Timeout exceeds one day");
  return {LOON_SUCCESS, nullptr};
}
}  // namespace

LoonFFIResult loon_async_configure_executor(const LoonAsyncExecutor* executor) {
  if (!executor || executor->struct_size < sizeof(LoonAsyncExecutor) || executor->reserved || !executor->submit)
    return {LOON_INVALID_ARGS, nullptr};
  try {
    return StatusResult(Runtime::Instance().Configure(*executor));
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_transaction_begin_async(const char* base_path,
                                           const LoonProperties* properties,
                                           int64_t read_version,
                                           int32_t resolve_id,
                                           uint32_t retry_limit,
                                           const LoonAsyncOptions* options,
                                           LoonTransactionBeginCallback callback,
                                           uintptr_t user_data,
                                           LoonAsyncHandle* out_operation) {
  using namespace milvus_storage::api;
  if (out_operation)
    *out_operation = nullptr;
  if (!out_operation || !callback || !base_path || !properties || read_version < -1 ||
      (resolve_id != LOON_TRANSACTION_RESOLVE_FAIL && resolve_id != LOON_TRANSACTION_RESOLVE_OVERWRITE))
    return {LOON_INVALID_ARGS, nullptr};
  try {
    uint64_t timeout;
    auto status = ValidateOptions(options, timeout);
    if (status.err_code)
      return status;
    if (properties->count > 1024 || (properties->count && !properties->properties) || strnlen(base_path, 16385) > 16384)
      return CreateFFIResult(LOON_INVALID_ARGS, "Invalid or oversized manifest inputs");
    for (size_t i = 0; i < properties->count; ++i) {
      auto& item = properties->properties[i];
      if (!item.key || !item.value || strnlen(item.key, 1025) > 1024 || strnlen(item.value, 16385) > 16384)
        return CreateFFIResult(LOON_INVALID_ARGS, "Invalid property");
    }
    Properties native_properties;
    auto error = ConvertFFIProperties(native_properties, properties);
    if (error)
      return CreateFFIResult(LOON_INVALID_PROPERTIES, *error);
    auto handle = std::make_unique<LoonAsyncOperation>();
    const auto& resolver =
        resolve_id == LOON_TRANSACTION_RESOLVE_OVERWRITE ? transaction::OverwriteResolver : transaction::FailResolver;
    auto uri = StorageUri::Parse(base_path);
    if (!uri.ok())
      return StatusResult(uri.status());
    auto future = transaction::BeginAsync(base_path, std::move(native_properties), read_version, resolver, retry_limit,
                                          timeout, handle->state);
    auto accepted = Runtime::Instance().Submit(
        std::move(future), [callback, user_data](folly::Try<transaction::BeginResult>&& value) {
          if (value.hasException()) {
            callback(user_data, StatusResult(AsyncStatus::Exception), 0);
            return;
          }
          auto result = std::move(value).value();
          callback(user_data, StatusResult(result.status),
                   reinterpret_cast<LoonTransactionHandle>(result.transaction.release()));
        });
    if (!accepted.ok())
      return StatusResult(accepted);
    *out_operation = handle.release();
    return {LOON_SUCCESS, nullptr};
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    return {LOON_GOT_EXCEPTION, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}
LoonFFIResult loon_transaction_commit_async(LoonTransactionHandle transaction,
                                            const LoonAsyncOptions* options,
                                            LoonTransactionCommitCallback callback,
                                            uintptr_t user_data,
                                            LoonAsyncHandle* out_operation) {
  using namespace milvus_storage::api;
  if (out_operation)
    *out_operation = nullptr;
  if (!transaction || !callback || !out_operation)
    return {LOON_INVALID_ARGS, nullptr};
  try {
    uint64_t timeout;
    auto status = ValidateOptions(options, timeout);
    if (status.err_code)
      return status;
    auto* txn = reinterpret_cast<transaction::Transaction*>(transaction);
    auto handle = std::make_unique<LoonAsyncOperation>();
    auto future = transaction::CommitAsync(txn, timeout, handle->state);
    // Ready futures here only carry preflight failures (busy/invalid).
    // Successful commits remain deferred until the caller executor consumes them.
    if (future.isReady())
      return StatusResult(std::move(future).get().status);
    auto accepted = Runtime::Instance().Submit(
        std::move(future), [callback, user_data](folly::Try<transaction::CommitResult>&& value) {
          if (value.hasException()) {
            callback(user_data, StatusResult(AsyncStatus::Exception), LOON_COMMIT_UNKNOWN, -1);
            return;
          }
          auto result = std::move(value).value();
          auto native_outcome = result.outcome == transaction::CommitOutcome::Committed ? LOON_COMMIT_COMMITTED
                                : result.outcome == transaction::CommitOutcome::Unknown ? LOON_COMMIT_UNKNOWN
                                                                                        : LOON_COMMIT_NOT_COMMITTED;
          callback(user_data, StatusResult(result.status), native_outcome, result.version);
        });
    if (!accepted.ok())
      return StatusResult(accepted);
    *out_operation = handle.release();
    return {LOON_SUCCESS, nullptr};
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    return {LOON_GOT_EXCEPTION, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}
void loon_async_cancel(LoonAsyncHandle operation) {
  if (operation)
    operation->state->Cancel();
}
void loon_async_release(LoonAsyncHandle operation) { delete operation; }
void loon_async_shutdown(void) {
  try {
    Runtime::Instance().Shutdown();
  } catch (...) { /* Initialization failure has no accepted operations to drain. */
  }
}
