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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/future.h>

// Upstream Talon C SDK ABI (milvus-io/talon, clients/c/include/talon.h). The
// header and the linked libtalon_c static library are provided by the Talon
// source tree fetched at configure time; see cpp/CMakeLists.txt (WITH_TALON).
#include <talon.h>

namespace milvus_storage::talon {

/// Object identity + length as reported by a Talon StatObject.
struct TalonStat {
  int64_t size = 0;     // total object length in bytes
  std::string version;  // opaque object generation token (may be empty)
};

/// Outcome of a single Talon read. `object_size` / `version` echo what Talon
/// resolved for the object generation it served, so a caller that did not
/// supply them up front can cache them and take the stat-skipping fast path on
/// subsequent reads.
struct TalonReadResult {
  int64_t bytes_written = 0;
  int64_t object_size = -1;  // arrow::fs::kNoSize when Talon did not report it
  std::string version;
};

/// Translate a `talon_status` code (from a result or a submit call) into an
/// arrow::Status. TALON_STATUS_INVALID_ARGUMENT maps to Invalid; every other
/// non-OK code maps to IOError. `op` and `uri` are woven into the message.
arrow::Status TalonStatusToStatus(int status, const char* error, std::string_view op, const std::string& uri);

/// Thin RAII owner of a `talon_client`. Reads and stats are submitted through
/// the C SDK's async API and completed via an arrow::Future; the client keeps
/// itself alive across in-flight operations (each submission holds a
/// shared_ptr) so callers may drop their reference without cancelling work.
///
/// Callbacks run inline on the SDK's runtime thread (no callback executor is
/// installed), so the marshalling here only marks the future finished — exactly
/// the convention the S3 CRT reader uses. Heavy or blocking continuation work
/// must be moved off that thread by the caller, at the future's continuation
/// boundary.
class TalonClient : public std::enable_shared_from_this<TalonClient> {
  public:
  /// Connect a client to `coordinator_addr`. The client uses the SDK's default
  /// block granularity, which is set to match the worker cache configuration.
  static arrow::Result<std::shared_ptr<TalonClient>> Make(const std::string& coordinator_addr);

  ~TalonClient();

  TalonClient(const TalonClient&) = delete;
  TalonClient& operator=(const TalonClient&) = delete;

  /// Submit a read of [offset, offset + len) into the caller-owned buffer
  /// [dst, dst + len). The buffer must stay valid and untouched until the
  /// returned future completes (the SDK owns it meanwhile).
  ///
  /// When BOTH `version` and `object_size` are provided the read takes Talon's
  /// fast path and skips the StatObject round trip; a lone value is dropped
  /// (passing it would not skip the stat), matching the C ABI contract.
  arrow::Future<TalonReadResult> ReadAsync(const std::string& uri,
                                           int64_t offset,
                                           uint8_t* dst,
                                           int64_t len,
                                           const std::optional<std::string>& version,
                                           const std::optional<int64_t>& object_size);

  /// Resolve object length + version via a StatObject.
  arrow::Future<TalonStat> StatAsync(const std::string& uri);

  /// Blocking convenience wrapper over StatAsync.
  arrow::Result<TalonStat> Stat(const std::string& uri);

  /// Raw handle, for tests and advanced callers. Never freed by the caller.
  talon_client* raw() const { return client_; }

  private:
  explicit TalonClient(talon_client* client) : client_(client) {}

  talon_client* client_ = nullptr;
};

}  // namespace milvus_storage::talon
