// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "talon/talon_bridge.h"

#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <arrow/status.h>

#include "runtime/bridge_util.h"

namespace milvus_storage::talon {
namespace {

using TalonIoCallback = void (*)(void* context, int32_t error_code, uint64_t value, const char* error_msg);

extern "C" {

void talon_object_stat_async(const void* reader, TalonIoCallback callback, void* context);

void talon_object_read_async(
    const void* reader, uint64_t offset, uint64_t length, uint8_t* dst, TalonIoCallback callback, void* context);

void talon_free_error_string(char* ptr);

}  // extern "C"

struct TalonIoCompletion {
  TalonIoCompletion(arrow::Future<int64_t> future, const char* operation)
      : future(std::move(future)), operation(operation) {}

  arrow::Future<int64_t> future;
  const char* const operation;
};

arrow::Result<int64_t> ToArrowIoResult(const char* operation,
                                       int32_t error_code,
                                       uint64_t value,
                                       const char* error_msg) {
  if (error_code == 0) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return arrow::Status::IOError(operation, ": Talon returned a value larger than INT64_MAX");
    }
    return static_cast<int64_t>(value);
  }

  const std::string message = error_msg == nullptr ? "unknown Talon error" : error_msg;
  // FIXME: Map Talon's structured error kinds to Arrow errno details and
  // ExtendStatus after Talon preserves them through the complete stat/read
  // path. Until then, do not infer retryability from incomplete error codes.
  return arrow::Status::IOError(operation, " (Talon error code ", error_code, "): ", message);
}

void MarkTalonCallbackError(arrow::Future<int64_t>& future, const char* operation, const char* message) noexcept {
  try {
    future.MarkFinished(
        arrow::Result<int64_t>(arrow::Status::IOError(operation, ": failed to complete callback: ", message)));
  } catch (...) {
    // No further error reporting is safe from a callback crossing the C ABI.
  }
}

void TalonIoCallbackImpl(void* context, int32_t error_code, uint64_t value, const char* error_msg) noexcept {
  // Rust invokes the callback exactly once, so this callback reclaims both the
  // completion context and any owned Rust error string.
  std::unique_ptr<TalonIoCompletion> completion(static_cast<TalonIoCompletion*>(context));
  std::unique_ptr<char, decltype(&talon_free_error_string)> error(const_cast<char*>(error_msg),
                                                                  &talon_free_error_string);
  try {
    completion->future.MarkFinished(ToArrowIoResult(completion->operation, error_code, value, error.get()));
  } catch (const std::exception& exception) {
    MarkTalonCallbackError(completion->future, completion->operation, exception.what());
  } catch (...) {
    MarkTalonCallbackError(completion->future, completion->operation, "unknown exception");
  }
}

}  // namespace

arrow::Result<std::shared_ptr<TalonClient>> TalonClient::Make(const std::string& coordinator,
                                                              uint32_t block_size,
                                                              uint32_t max_idle_per_addr) {
  return CatchRustResult<std::shared_ptr<TalonClient>>("Failed to create Talon client", [&]() {
    auto impl = ffi::new_talon_client(coordinator, block_size, max_idle_per_addr);
    return std::shared_ptr<TalonClient>(new TalonClient(std::move(impl)));
  });
}

arrow::Result<TalonObjectReader> TalonClient::OpenObject(const std::string& cloud_provider,
                                                         const std::string& bucket,
                                                         const std::string& key,
                                                         const std::optional<TalonObjectStat>& initial_stat) const {
  return CatchRustResult<TalonObjectReader>("Failed to open Talon object", [&]() {
    const bool has_initial_stat = initial_stat.has_value();
    const std::string empty_version;
    return TalonObjectReader(ffi::open_talon_object(*impl_, cloud_provider, bucket, key, has_initial_stat,
                                                    has_initial_stat ? initial_stat->size : 0,
                                                    has_initial_stat ? initial_stat->version : empty_version));
  });
}

int64_t TalonObjectReader::KnownSize() const { return ffi::talon_object_known_size(*impl_); }

arrow::Future<int64_t> TalonObjectReader::StatAsync() const {
  auto future = arrow::Future<int64_t>::Make();
  auto completion = std::make_unique<TalonIoCompletion>(future, "Failed to stat Talon object");
  auto* const raw_completion = completion.release();
  talon_object_stat_async(static_cast<const void*>(&*impl_), TalonIoCallbackImpl, static_cast<void*>(raw_completion));
  return future;
}

arrow::Future<int64_t> TalonObjectReader::ReadAtAsync(uint64_t offset, uint64_t length, uint8_t* dst) const {
  auto future = arrow::Future<int64_t>::Make();
  auto completion = std::make_unique<TalonIoCompletion>(future, "Failed to read Talon object");
  auto* const raw_completion = completion.release();
  talon_object_read_async(static_cast<const void*>(&*impl_), offset, length, dst, TalonIoCallbackImpl,
                          static_cast<void*>(raw_completion));
  return future;
}

}  // namespace milvus_storage::talon
