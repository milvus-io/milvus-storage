// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/talon/talon_client.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace milvus_storage::talon {

namespace {

/// RAII owner for a `talon_result`. Every callback receives ownership of the
/// result and must free it exactly once; this guarantees that even on an early
/// return path.
class TalonResultHandle {
  public:
  explicit TalonResultHandle(talon_result* result) : result_(result) {}
  ~TalonResultHandle() {
    if (result_ != nullptr) {
      talon_result_free(result_);
    }
  }
  TalonResultHandle(const TalonResultHandle&) = delete;
  TalonResultHandle& operator=(const TalonResultHandle&) = delete;

  private:
  talon_result* result_;
};

/// Heap state that outlives the ReadAsync call and is reclaimed by the
/// callback. It backs the C-string / size pointers handed to the SDK and holds
/// a shared_ptr to the client so it cannot be freed mid-flight.
struct ReadContext {
  arrow::Future<TalonReadResult> future;
  std::shared_ptr<TalonClient> client;
  std::string uri;           // backs the `uri` pointer
  std::string version;       // backs the `version` pointer (fast path only)
  uint64_t object_size = 0;  // backs the `object_size` pointer (fast path only)
};

struct StatContext {
  arrow::Future<TalonStat> future;
  std::shared_ptr<TalonClient> client;
  std::string uri;
};

void ReadCallback(talon_result* result, void* user_data) {
  std::unique_ptr<ReadContext> ctx(static_cast<ReadContext*>(user_data));
  TalonResultHandle handle(result);

  const int status = talon_result_status(result);
  if (status != TALON_STATUS_OK) {
    ctx->future.MarkFinished(
        arrow::Result<TalonReadResult>(TalonStatusToStatus(status, talon_result_error(result), "read", ctx->uri)));
    return;
  }

  TalonReadResult out;
  out.bytes_written = static_cast<int64_t>(talon_result_bytes_written(result));
  out.object_size = static_cast<int64_t>(talon_result_object_size(result));
  if (const char* version = talon_result_version(result); version != nullptr) {
    out.version = version;
  }
  ctx->future.MarkFinished(std::move(out));
}

void StatCallback(talon_result* result, void* user_data) {
  std::unique_ptr<StatContext> ctx(static_cast<StatContext*>(user_data));
  TalonResultHandle handle(result);

  const int status = talon_result_status(result);
  if (status != TALON_STATUS_OK) {
    ctx->future.MarkFinished(
        arrow::Result<TalonStat>(TalonStatusToStatus(status, talon_result_error(result), "stat", ctx->uri)));
    return;
  }

  TalonStat out;
  out.size = static_cast<int64_t>(talon_result_object_size(result));
  if (const char* version = talon_result_version(result); version != nullptr) {
    out.version = version;
  }
  ctx->future.MarkFinished(std::move(out));
}

}  // namespace

arrow::Status TalonStatusToStatus(int status, const char* error, std::string_view op, const std::string& uri) {
  std::string detail = error != nullptr && error[0] != '\0' ? error : "no error detail";
  std::string message = "Talon " + std::string(op) + " failed for '" + uri + "': " + detail +
                        " (talon_status=" + std::to_string(status) + ")";
  switch (status) {
    case TALON_STATUS_INVALID_ARGUMENT:
      return arrow::Status::Invalid(std::move(message));
    default:
      return arrow::Status::IOError(std::move(message));
  }
}

arrow::Result<std::shared_ptr<TalonClient>> TalonClient::Make(const std::string& coordinator_addr) {
  talon_client_options options;
  talon_client_options_init(&options);
  // Complete callbacks inline on the SDK runtime thread; continuations are
  // rescheduled by the caller at the future boundary, as the S3 CRT reader does.
  options.callback_executor = nullptr;

  talon_client* client = nullptr;
  const int rc = talon_client_new(coordinator_addr.c_str(), &options, &client);
  if (rc != TALON_STATUS_OK || client == nullptr) {
    return TalonStatusToStatus(rc, talon_last_error(), "client_new", coordinator_addr);
  }
  return std::shared_ptr<TalonClient>(new TalonClient(client));
}

TalonClient::~TalonClient() {
  if (client_ != nullptr) {
    talon_client_free(client_);
    client_ = nullptr;
  }
}

arrow::Future<TalonReadResult> TalonClient::ReadAsync(const std::string& uri,
                                                      int64_t offset,
                                                      uint8_t* dst,
                                                      int64_t len,
                                                      const std::optional<std::string>& version,
                                                      const std::optional<int64_t>& object_size) {
  auto ctx = std::make_unique<ReadContext>();
  ctx->future = arrow::Future<TalonReadResult>::Make();
  ctx->client = shared_from_this();
  ctx->uri = uri;

  const char* version_ptr = nullptr;
  const uint64_t* size_ptr = nullptr;
  // Fast path requires BOTH; a lone value is ignored by the SDK, so do not send
  // it (avoids implying a capability the read cannot actually use).
  if (version.has_value() && object_size.has_value()) {
    ctx->version = *version;
    ctx->object_size = static_cast<uint64_t>(*object_size);
    version_ptr = ctx->version.c_str();
    size_ptr = &ctx->object_size;
  }

  auto future = ctx->future;
  uint64_t request_id = 0;
  const int rc =
      talon_read_async(client_, ctx->uri.c_str(), static_cast<uint64_t>(offset), dst, static_cast<size_t>(len),
                       version_ptr, size_ptr, &ReadCallback, ctx.get(), &request_id);
  if (rc != TALON_STATUS_OK) {
    // Submission failed: the callback will not run, so this unique_ptr reclaims
    // the context as it goes out of scope.
    future.MarkFinished(arrow::Result<TalonReadResult>(TalonStatusToStatus(rc, talon_last_error(), "read", uri)));
    return future;
  }
  // Submission succeeded: the callback now owns the context.
  ctx.release();
  return future;
}

arrow::Future<TalonStat> TalonClient::StatAsync(const std::string& uri) {
  auto ctx = std::make_unique<StatContext>();
  ctx->future = arrow::Future<TalonStat>::Make();
  ctx->client = shared_from_this();
  ctx->uri = uri;

  auto future = ctx->future;
  uint64_t request_id = 0;
  const int rc = talon_stat_async(client_, ctx->uri.c_str(), &StatCallback, ctx.get(), &request_id);
  if (rc != TALON_STATUS_OK) {
    future.MarkFinished(arrow::Result<TalonStat>(TalonStatusToStatus(rc, talon_last_error(), "stat", uri)));
    return future;
  }
  ctx.release();
  return future;
}

arrow::Result<TalonStat> TalonClient::Stat(const std::string& uri) { return StatAsync(uri).result(); }

}  // namespace milvus_storage::talon
