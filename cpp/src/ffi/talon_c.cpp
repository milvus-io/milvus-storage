// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
#include "milvus-storage/ffi_talon_c.h"

#include "milvus-storage/ffi_internal/async_context.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"
#include "talon/talon_bridge.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using milvus_storage::talon::TalonBatchLoadResult;
using milvus_storage::talon::TalonClient;
using milvus_storage::talon::TalonLoadRequest;
using milvus_storage::talon::TalonLoadResult;
using milvus_storage::talon::TalonObjectStat;

namespace {
struct TalonAsyncState final : milvus_storage::AsyncOperation {
  std::atomic<bool> cancelled{false};
  std::chrono::steady_clock::time_point deadline;
  void Cancel() override { cancelled.store(true, std::memory_order_relaxed); }
};

// Async calls queue native submission so admission, cancellation, and deadline
// checks happen before I/O. The C API can return before Rust reads the requests,
// so the queued task must own the URI/version strings instead of borrowing them.
// The vector is moved into that task. Sync calls reuse this conversion to match
// the bridge's owning request type.
static bool CopyRequests(const LoonTalonLoadRequest* requests, size_t count, std::vector<TalonLoadRequest>& copied) {
  if (count && !requests)
    return false;
  copied.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!requests[i].uri || !*requests[i].uri)
      return false;
    copied.push_back({requests[i].uri, requests[i].size, requests[i].version ? requests[i].version : ""});
  }
  return true;
}

// Resolve immutable request overrides before async admission. Rust owns the
// copied strings and reuses the original client's pools/discovery state.
static arrow::Result<std::shared_ptr<TalonClient>> ResolveRequestOptions(LoonTalonClientHandle client,
                                                                         const LoonTalonRequestOptions* options) {
  auto native = *reinterpret_cast<std::shared_ptr<TalonClient>*>(client);
  if (!options)
    return native;
  if (options->namespace_name) {
    ARROW_ASSIGN_OR_RAISE(native, native->WithNamespace(options->namespace_name));
  }
  if (options->credential_broker) {
    ARROW_ASSIGN_OR_RAISE(native, native->WithCredentialBroker(options->credential_broker));
  }
  return native;
}

template <typename T>
static LoonFFIResult ExportResult(const arrow::Result<T>& native,
                                  LoonTalonOperation operation,
                                  LoonTalonResult** output) {
  if (native.status().IsInvalid()) {
    RETURN_ERROR(LOON_INVALID_ARGS, native.status().ToString());
  }
  RETURN_ARROW_ERROR_IF(native.status(), LOON_ARROW_ERROR, native.status().ToString());

  std::unique_ptr<LoonTalonResult, decltype(&loon_talon_result_free)> payload(
      static_cast<LoonTalonResult*>(calloc(1, sizeof(LoonTalonResult))), loon_talon_result_free);
  if (!payload) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to allocate Talon result");
  }
  payload->operation = operation;
  if constexpr (std::is_same_v<T, int64_t>) {
    payload->data.bytes_read = native.ValueUnsafe();
  } else if constexpr (std::is_same_v<T, TalonObjectStat>) {
    payload->data.stat.size = native->size;
    payload->data.stat.version = strdup(native->version.c_str());
    if (!payload->data.stat.version) {
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to copy Talon object version");
    }
  } else if constexpr (std::is_same_v<T, TalonLoadResult>) {
    payload->data.load.failed = false;
    auto& load = payload->data.load;
    load.entries.results = static_cast<LoonTalonLoadResult*>(calloc(1, sizeof(LoonTalonLoadResult)));
    if (!load.entries.results) {
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to allocate Talon load result");
    }
    load.count = 1;
    load.entries.results[0] = {native->size, native->blocks};
  } else if constexpr (std::is_same_v<T, TalonBatchLoadResult>) {
    payload->data.load.failed = !native->status.ok();
    auto& load = payload->data.load;
    // Upstream does not expose successful file details on a partial failure.
    if (!load.failed) {
      load.entries.results =
          static_cast<LoonTalonLoadResult*>(calloc(native->results.size(), sizeof(LoonTalonLoadResult)));
      if (!load.entries.results && !native->results.empty()) {
        RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to allocate Talon batch load results");
      }
      load.count = native->results.size();
      for (size_t i = 0; i < native->results.size(); ++i)
        load.entries.results[i] = {native->results[i].size, native->results[i].blocks};
    } else {
      load.entries.failures =
          static_cast<LoonTalonLoadFailure*>(calloc(native->failed_files.size(), sizeof(LoonTalonLoadFailure)));
      if (!load.entries.failures && !native->failed_files.empty()) {
        RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to allocate Talon batch load failures");
      }
      load.count = native->failed_files.size();
      for (size_t i = 0; i < native->failed_files.size(); ++i) {
        load.entries.failures[i].index = native->failed_files[i].index;
        load.entries.failures[i].uncertain = native->failed_files[i].uncertain;
        load.entries.failures[i].error = strdup(native->failed_files[i].error.c_str());
        if (!load.entries.failures[i].error) {
          RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to copy Talon batch load error");
        }
      }
    }
    LoonFFIResult status{LOON_SUCCESS, nullptr};
    if (!native->status.ok()) {
      const int code = native->status.IsInvalid() ? LOON_INVALID_ARGS : FFIErrorCodeFromExtendStatus(native->status);
      status = CreateFFIResult(code, native->status.ToString());
    }
    *output = payload.release();
    return status;
  }
  *output = payload.release();
  RETURN_SUCCESS();
}

template <typename T, typename Start>
static LoonFFIResult SubmitRaw(LoonAsyncContextHandle context,
                               std::shared_ptr<TalonClient> client,
                               Start start,
                               LoonTalonOperation tag,
                               LoonTalonCallback callback,
                               uintptr_t user_data,
                               LoonAsyncHandle* operation,
                               uint64_t timeout) {
  using Outcome = std::variant<milvus_storage::AsyncStatus::Code, arrow::Future<T>>;
  auto handle = std::make_unique<LoonAsyncOperation>();
  auto state = std::make_shared<TalonAsyncState>();
  state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
  handle->state = state;
  // The deferred body only runs after the caller executor accepts the initial task.
  auto future = folly::makeSemiFuture().deferValue(
      [state, client = std::move(client), start = std::move(start)](folly::Unit) mutable -> folly::SemiFuture<Outcome> {
        if (state->cancelled.load(std::memory_order_relaxed))
          return folly::makeSemiFuture(Outcome(std::in_place_index<0>, milvus_storage::AsyncStatus::Cancelled));
        if (std::chrono::steady_clock::now() >= state->deadline)
          return folly::makeSemiFuture(Outcome(std::in_place_index<0>, milvus_storage::AsyncStatus::Deadline));
        folly::Promise<Outcome> promise;
        auto result = promise.getSemiFuture();
        // Native Talon I/O runs on its shared Tokio runtime without waiting on this caller worker.
        auto native = start(*client);
        try {
          native.AddCallback([promise = std::move(promise), client, native](const arrow::Result<T>&) mutable {
            // Retain the completed result without copying batch vectors or diagnostics on the native thread.
            promise.setValue(Outcome(std::in_place_index<1>, std::move(native)));
          });
        } catch (...) {
          // Registration failure must not release the caller's buffer while native I/O can still write it.
          // Only this exceptional path waits; the failed future preserves the original registration error.
          native.Wait();
          return folly::makeSemiFuture<Outcome>(folly::exception_wrapper(std::current_exception()));
        }
        return result;
      });
  auto admission = context->Submit(std::move(future), [callback, user_data, tag](folly::Try<Outcome>&& value) {
    LoonTalonResult* payload = nullptr;
    LoonFFIResult status{LOON_GOT_EXCEPTION, nullptr};
    try {
      Outcome outcome = std::move(value).value();
      if (outcome.index() == 0) {
        const auto code = std::get<0>(outcome);
        status = {code == milvus_storage::AsyncStatus::Cancelled ? LOON_ASYNC_CANCELLED : LOON_ASYNC_DEADLINE, nullptr};
      } else {
        // The native callback already observed completion, so result() cannot block here.
        status = ExportResult(std::get<1>(outcome).result(), tag, &payload);
      }
    } catch (const std::bad_alloc&) {
      status = {LOON_MEMORY_ERROR, nullptr};
      loon_talon_result_free(payload);
      payload = nullptr;
    } catch (const std::exception& error) {
      loon_talon_result_free(payload);
      payload = nullptr;
      // A failed diagnostic allocation must not prevent the accepted callback.
      status = {LOON_GOT_EXCEPTION, strdup(error.what())};
    } catch (...) {
      status = {LOON_GOT_EXCEPTION, nullptr};
      loon_talon_result_free(payload);
      payload = nullptr;
    }
    callback(user_data, status, payload);
  });
  if (!admission.ok())
    return {LOON_ASYNC_OVERLOADED, nullptr};
  operation->internal = handle.release();
  RETURN_SUCCESS();
}
}  // namespace

LoonFFIResult loon_talon_client_create(const LoonProperties* properties, LoonTalonClientHandle* client) {
  if (client)
    *client = 0;
  if (!client || !properties || (properties->count && !properties->properties)) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon client arguments");
  }
  try {
    for (size_t i = 0; i < properties->count; ++i) {
      const auto& item = properties->properties[i];
      if (!item.key || !item.value) {
        RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon property: key and value must not be null");
      }
    }
    milvus_storage::api::Properties native;
    if (auto error = milvus_storage::api::ConvertFFIProperties(native, properties)) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", *error, "]");
    }
    // Raw clients use Talon directly, independently of filesystem routing mode.
    native[PROPERTY_FS_TALON_MODE] = static_cast<uint32_t>(milvus_storage::TalonMode::Disabled);
    milvus_storage::ArrowFileSystemConfig config;
    const auto config_status = milvus_storage::ArrowFileSystemConfig::create_file_system_config(native, config);
    RETURN_ARROW_ERROR_IF(config_status, config_status.IsInvalid() ? LOON_INVALID_ARGS : LOON_INVALID_PROPERTIES,
                          config_status.ToString());
    auto made = TalonClient::Make(config);
    if (made.status().IsInvalid()) {
      RETURN_ERROR(LOON_INVALID_ARGS, made.status().ToString());
    }
    RETURN_ARROW_ERROR_IF(made.status(), LOON_INVALID_PROPERTIES, made.status().ToString());
    auto owned = std::make_unique<std::shared_ptr<TalonClient>>(std::move(made).ValueUnsafe());
    *client = reinterpret_cast<LoonTalonClientHandle>(owned.release());
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

void loon_talon_client_destroy(LoonTalonClientHandle client) {
  delete reinterpret_cast<std::shared_ptr<TalonClient>*>(client);
}

LoonFFIResult loon_talon_read(LoonTalonClientHandle client,
                              const char* uri,
                              uint64_t offset,
                              uint64_t length,
                              const char* version,
                              const uint64_t* object_size,
                              uint8_t* dst,
                              const LoonTalonRequestOptions* options,
                              LoonTalonResult** result) {
  if (result)
    *result = nullptr;
  if (!result || !client || !uri || !*uri || (!dst && length) || (version && *version && !object_size)) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon read arguments");
  }
  try {
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    std::optional<TalonObjectStat> stat;
    if (object_size)
      stat = TalonObjectStat{*object_size, version ? version : ""};
    return ExportResult(cpp_client->Read(uri, offset, length, dst, stat), LOON_TALON_READ, result);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_stat(LoonTalonClientHandle client,
                              const char* uri,
                              const LoonTalonRequestOptions* options,
                              LoonTalonResult** result) {
  if (result)
    *result = nullptr;
  if (!result || !client || !uri || !*uri) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon stat arguments");
  }
  try {
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    return ExportResult(cpp_client->Stat(uri), LOON_TALON_STAT, result);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_load(LoonTalonClientHandle client,
                              const LoonTalonLoadRequest* request,
                              const LoonTalonRequestOptions* options,
                              LoonTalonResult** result) {
  if (result)
    *result = nullptr;
  if (!result || !client || !request || !request->uri || !*request->uri) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon load arguments");
  }
  try {
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    const TalonLoadRequest copied{request->uri, request->size, request->version ? request->version : ""};
    return ExportResult(cpp_client->Load(copied), LOON_TALON_LOAD, result);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_batch_load(LoonTalonClientHandle client,
                                    const LoonTalonLoadRequest* requests,
                                    size_t count,
                                    const LoonTalonRequestOptions* options,
                                    LoonTalonResult** result) {
  if (result)
    *result = nullptr;
  if (!result || !client) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon batch load arguments");
  }
  try {
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    std::vector<TalonLoadRequest> copied;
    if (!CopyRequests(requests, count, copied)) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon load requests");
    }
    // The temporary Future owns the batch throughout export; no intermediate result copy is needed.
    return ExportResult(cpp_client->BatchLoadAsync(copied).result(), LOON_TALON_BATCH_LOAD, result);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_read_async(LoonAsyncContextHandle context,
                                    LoonTalonClientHandle client,
                                    const char* uri,
                                    uint64_t offset,
                                    uint64_t length,
                                    const char* version,
                                    const uint64_t* object_size,
                                    uint8_t* dst,
                                    const LoonTalonRequestOptions* options,
                                    LoonTalonCallback callback,
                                    uintptr_t user_data,
                                    LoonAsyncHandle* operation) {
  if (!context || !client || !callback || !operation) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon async arguments");
  }
  try {
    uint64_t timeout = 0;
    const auto status = milvus_storage::ValidateAsyncHandle(operation, timeout);
    if (status.err_code)
      return status;
    if (!uri || !*uri || (!dst && length) || (version && *version && !object_size)) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon read arguments");
    }
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    const std::string copied_uri(uri);
    std::optional<TalonObjectStat> stat;
    if (object_size)
      stat = TalonObjectStat{*object_size, version ? version : ""};
    return SubmitRaw<int64_t>(
        context, cpp_client,
        [copied_uri, stat = std::move(stat), offset, length, dst](const TalonClient& native) {
          return native.ReadAsync(copied_uri, offset, length, dst, stat);
        },
        LOON_TALON_READ, callback, user_data, operation, timeout);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_stat_async(LoonAsyncContextHandle context,
                                    LoonTalonClientHandle client,
                                    const char* uri,
                                    const LoonTalonRequestOptions* options,
                                    LoonTalonCallback callback,
                                    uintptr_t user_data,
                                    LoonAsyncHandle* operation) {
  if (!context || !client || !callback || !operation) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon async arguments");
  }
  try {
    uint64_t timeout = 0;
    const auto status = milvus_storage::ValidateAsyncHandle(operation, timeout);
    if (status.err_code)
      return status;
    if (!uri || !*uri) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon stat arguments");
    }
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    const std::string copied_uri(uri);
    return SubmitRaw<TalonObjectStat>(
        context, cpp_client, [copied_uri](const TalonClient& native) { return native.StatAsync(copied_uri); },
        LOON_TALON_STAT, callback, user_data, operation, timeout);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_load_async(LoonAsyncContextHandle context,
                                    LoonTalonClientHandle client,
                                    const LoonTalonLoadRequest* request,
                                    const LoonTalonRequestOptions* options,
                                    LoonTalonCallback callback,
                                    uintptr_t user_data,
                                    LoonAsyncHandle* operation) {
  if (!context || !client || !callback || !operation) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon async arguments");
  }
  try {
    uint64_t timeout = 0;
    const auto status = milvus_storage::ValidateAsyncHandle(operation, timeout);
    if (status.err_code)
      return status;
    if (!request || !request->uri || !*request->uri) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon load arguments");
    }
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    const TalonLoadRequest copied{request->uri, request->size, request->version ? request->version : ""};
    return SubmitRaw<TalonLoadResult>(
        context, cpp_client, [copied](const TalonClient& native) { return native.LoadAsync(copied); }, LOON_TALON_LOAD,
        callback, user_data, operation, timeout);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

LoonFFIResult loon_talon_batch_load_async(LoonAsyncContextHandle context,
                                          LoonTalonClientHandle client,
                                          const LoonTalonLoadRequest* requests,
                                          size_t count,
                                          const LoonTalonRequestOptions* options,
                                          LoonTalonCallback callback,
                                          uintptr_t user_data,
                                          LoonAsyncHandle* operation) {
  if (!context || !client || !callback || !operation) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon async arguments");
  }
  try {
    uint64_t timeout = 0;
    const auto status = milvus_storage::ValidateAsyncHandle(operation, timeout);
    if (status.err_code)
      return status;
    auto selected = ResolveRequestOptions(client, options);
    RETURN_ARROW_ERROR_IF(selected.status(), LOON_INVALID_ARGS, selected.status().ToString());
    auto cpp_client = std::move(selected).ValueUnsafe();
    std::vector<TalonLoadRequest> copied;
    if (!CopyRequests(requests, count, copied)) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid Talon load requests");
    }
    return SubmitRaw<TalonBatchLoadResult>(
        context, cpp_client,
        [copied = std::move(copied)](const TalonClient& native) { return native.BatchLoadAsync(copied); },
        LOON_TALON_BATCH_LOAD, callback, user_data, operation, timeout);
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

void loon_talon_result_free(LoonTalonResult* result) {
  if (!result)
    return;
  switch (result->operation) {
    case LOON_TALON_READ:
      break;
    case LOON_TALON_STAT:
      free(result->data.stat.version);
      break;
    case LOON_TALON_LOAD:
    case LOON_TALON_BATCH_LOAD:
      if (result->data.load.failed) {
        for (size_t i = 0; i < result->data.load.count; ++i) free(result->data.load.entries.failures[i].error);
        free(result->data.load.entries.failures);
      } else {
        free(result->data.load.entries.results);
      }
      break;
  }
  free(result);
}
