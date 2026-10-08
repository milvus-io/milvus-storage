// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/ffi_internal/async_context.h"

using namespace milvus_storage;
namespace {
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
}  // namespace

LoonFFIResult loon_transaction_open_async(LoonAsyncContextHandle async_context,
                                          const char* base_path,
                                          const LoonProperties* properties,
                                          int64_t read_version,
                                          int32_t resolve_id,
                                          uint32_t retry_limit,
                                          LoonTransactionOpenCallback callback,
                                          uintptr_t user_data,
                                          LoonAsyncHandle* operation) {
  using namespace milvus_storage::api;
  if (!async_context || !operation || !callback || !base_path || !properties || read_version < -1 ||
      (resolve_id != LOON_TRANSACTION_RESOLVE_FAIL && resolve_id != LOON_TRANSACTION_RESOLVE_OVERWRITE))
    return {LOON_INVALID_ARGS, nullptr};
  try {
    uint64_t timeout;
    auto status = ValidateAsyncHandle(operation, timeout);
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
    auto future = transaction::Transaction::OpenAsync(base_path, std::move(native_properties), read_version, resolver,
                                                      retry_limit, timeout, handle->state);
    auto accepted =
        async_context->Submit(std::move(future), [callback, user_data](folly::Try<transaction::OpenResult>&& value) {
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
    operation->internal = handle.release();
    return {LOON_SUCCESS, nullptr};
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    return {LOON_GOT_EXCEPTION, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}
LoonFFIResult loon_transaction_commit_async(LoonAsyncContextHandle async_context,
                                            LoonTransactionHandle transaction,
                                            LoonTransactionCommitCallback callback,
                                            uintptr_t user_data,
                                            LoonAsyncHandle* operation) {
  using namespace milvus_storage::api;
  if (!async_context || !transaction || !callback || !operation)
    return {LOON_INVALID_ARGS, nullptr};
  try {
    uint64_t timeout;
    auto status = ValidateAsyncHandle(operation, timeout);
    if (status.err_code)
      return status;
    auto* txn = reinterpret_cast<transaction::Transaction*>(transaction);
    auto handle = std::make_unique<LoonAsyncOperation>();
    auto future = txn->CommitAsync(timeout, handle->state);
    // Ready futures here only carry preflight failures (busy/invalid).
    // Successful commits remain deferred until the caller executor consumes them.
    if (future.isReady())
      return StatusResult(std::move(future).get().status);
    auto accepted =
        async_context->Submit(std::move(future), [callback, user_data](folly::Try<transaction::CommitResult>&& value) {
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
    operation->internal = handle.release();
    return {LOON_SUCCESS, nullptr};
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    return {LOON_GOT_EXCEPTION, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}
