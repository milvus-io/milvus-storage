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

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>

#include <arrow/c/abi.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage::bridge {

// THE decoder for classified errors coming out of the Rust cxx bridges
// (vortex / lance / iceberg / paimon). One table, one implementation: a code
// cannot mean one thing for one format and something else for another.
//
// One channel: the universal message marker
// "__LOON_RUST_BRIDGE_ERRCODE__=<code>; message". The Rust side embeds the
// code at the error's construction (BridgeError's Display, paimon's tagged
// messages, vortex's LoonFfiError), the code rides wherever the error text
// rides, and this decoder parses a frame at byte zero or after a fixed bridge
// wrapper prefix and strips it. Marker text later in a diagnostic is ordinary
// caller-controlled text.
//
// This replaced a thread-local side channel (record beside the error, take
// after catching), which failed three independent ways: an unclassified
// construction cleared a verdict recorded earlier in the same call; a verdict
// recorded on a tokio worker thread was unreadable from the calling thread;
// and a stale verdict could attach to a later failure. The marker has none of
// those failure modes because the verdict travels INSIDE the error. The
// Rust producers escape marker literals in ordinary diagnostics before they
// enter this channel, so a caller-controlled URI/path cannot forge a code.
//
// The code is mapped as:
//   * code 12 (LOON_FILE_NOT_FOUND)      -> IOError + ENOENT detail
//   * ExtendStatusCode values (101-122, with reserved gaps)
//                                           -> IOError + ExtendStatusDetail
//   * bridge-private codes (>= 1000, never cross the C ABI):
//       1001 data-corrupt   -> DataCorrupted detail (DataFormat category)
//       1002 not-supported  -> Status::NotImplemented
//   * no / unknown marker                -> plain IOError (conservative
//     non-retriable fallback; never invent retriability)

// Bridge-private marker codes; keep in sync with rust/src/bridge_error.rs.
inline constexpr int kBridgeErrCodeDataCorrupt = 1001;
inline constexpr int kBridgeErrCodeNotSupported = 1002;

/// Status for a cxx exception: decode the marker out of `what()`.
///
/// The classification rides the universal marker the Rust side embedded in
/// the error message; a message without a marker is the conservative
/// non-retriable fallback.
arrow::Status BridgeErrorStatusFromException(const char* what);

template <typename Fn>
auto CatchBridgeError(Fn&& fn) -> arrow::Result<decltype(fn())> {
  try {
    return fn();
  } catch (const std::exception& e) {
    return BridgeErrorStatusFromException(e.what());
  } catch (...) {
    return arrow::Status::UnknownError("Rust bridge failed unexpectedly");
  }
}

/// Void-returning form.
template <typename Fn>
arrow::Status CatchBridgeStatus(Fn&& fn) {
  try {
    fn();
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    return BridgeErrorStatusFromException(e.what());
  } catch (...) {
    return arrow::Status::UnknownError("Rust bridge failed unexpectedly");
  }
}

/// Build the status for a classification an asynchronous bridge I/O callback
/// reported as an explicit code. THE decoder: every bridge -- vortex, lance,
/// iceberg, paimon -- funnels through this one table, so a code cannot mean one
/// thing for one format and something else for another.
///
/// `code` 0 means "no classification available" and yields the conservative
/// IOError fallback, same as an unrecognised code. ExtendStatus details retain
/// IOError because the callback is reporting an operation that already entered
/// the bridge I/O path.
arrow::Status MakeBridgeErrorStatus(int ffi_err_code, std::string_view message);

/// Decode a raw bridge error message (marker stripped) into a structured
/// arrow::Status per the table above. Use this only where the message is known
/// to come from a Rust bridge; for anything that may already be an ordinary
/// arrow::Status, use DecodeBridgeErrorStatus() so a non-bridge failure keeps
/// its own classification.
arrow::Status MakeBridgeErrorStatus(std::string_view message);

/// Decode `message` only when it actually carries the bridge marker.
///
/// Returns nullopt otherwise, which is the whole point: a status that did not
/// come from a Rust bridge (arrow's own Invalid/NotImplemented while importing
/// a C stream, say) must keep the classification its producer gave it instead
/// of being rewritten into a blanket IOError.
std::optional<arrow::Status> DecodeBridgeErrorStatus(std::string_view message);

/// Marker-free rendering of a raw bridge error message, for the few sites that
/// keep their own StatusCode but must not leak the internal marker into a
/// user-visible message.
std::string StripBridgeErrorMarker(std::string_view message);

/// Build the status for "these persisted bytes do not decode": DataCorrupted
/// detail, DataFormat category. Thin alias for MakeExtendError so every format
/// reports this one condition identically.
arrow::Status MakeBridgeDataFormatStatus(std::string message);

/// Prefix `context` onto `status`'s message, preserving its StatusCode and
/// detail (ExtendStatusDetail / errno). OK statuses pass through.
arrow::Status WithBridgeContext(std::string_view context, const arrow::Status& status);

/// Translate any status that may carry an (encoded or already-structured)
/// bridge error: already-classified statuses just gain context; otherwise the
/// message is checked for a marker prefix and rebuilt.
arrow::Status TranslateBridgeStatus(std::string_view context, const arrow::Status& status);

/// Wrap a RecordBatchReader whose ReadNext/Close surface raw bridge error
/// strings (arrow FFI stringification of Rust stream errors) so mid-scan
/// errors are decoded too. `context` is prefixed onto translated errors.
std::shared_ptr<arrow::RecordBatchReader> WrapBridgeRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> inner,
                                                                      std::string context);

/// Import a C stream produced by a Rust bridge and translate errors reported
/// while pulling its batches.  Arrow's generic importer preserves the error
/// text but not the bridge marker/detail, so callers that consume the whole
/// stream as a ChunkedArray must use this helper.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> ImportBridgeChunkedArray(ArrowArrayStream* stream,
                                                                             std::string_view context);

}  // namespace milvus_storage::bridge
