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
// Two channels reach it, in order of preference:
//   1. the typed side channel -- the Rust side records (code, message) beside
//      the error and the catch site takes it. Nothing is parsed, and a message
//      cannot dictate its own classification.
//   2. the message marker "__LOON_RUST_BRIDGE_ERRCODE__=<code>; message", kept
//      for producers with no C++ frame to read the slot: an error surfacing
//      through an Arrow C stream, and the vortex fork's filesystem_c.rs.
//
// The side channel is thread_local. An error constructed on a tokio worker
// thread records its verdict in that worker's slot, which the calling thread
// cannot take -- those paths still fall back to the marker. That is fail-safe
// (a missed slot degrades to the marker, never to a wrong code), so the marker
// must stay: it is not a legacy fallback waiting to be deleted.
//
// Either way the code is mapped as:
//   * code 12 (LOON_FILE_NOT_FOUND)      -> IOError + ENOENT detail
//   * ExtendStatusCode values (101-112)  -> IOError + ExtendStatusDetail
//   * bridge-private codes (>= 1000, never cross the C ABI):
//       1001 data-corrupt   -> DataCorrupted detail (DataFormat category)
//       1002 not-supported  -> Status::NotImplemented
//   * no / unknown marker                -> plain IOError (conservative
//     non-retriable fallback; never invent retriability)

// Bridge-private marker codes; keep in sync with rust/src/bridge_error.rs.
inline constexpr int kBridgeErrCodeDataCorrupt = 1001;
inline constexpr int kBridgeErrCodeNotSupported = 1002;

/// Status for a cxx exception, preferring the side channel over the message.
///
/// The classification comes from the side channel the Rust side records into
/// (see rust/src/bridge_error.rs); the error message is only a fallback for
/// producers that have not moved to it yet, and for failures with no C++ frame
/// to read the slot (an Arrow C stream error, the vortex fork). The slot is
/// cleared before the call, and every BridgeError construction overwrites it --
/// an unclassified one by clearing it -- so a code left over from an earlier
/// failure can attach itself neither to a later call nor to a later error
/// within the same call.
arrow::Status BridgeErrorStatusFromException(const char* what);

/// Clear the side channel before entering a guarded call.
void ClearBridgeErrorChannel();

namespace internal {
/// Test-only: publish a classification as if a Rust bridge had recorded it.
/// THE one test injector across all bridges -- writes the same single slot
/// production reads.
void RecordBridgeErrorForTest(int ffi_err_code, std::string_view message);
}  // namespace internal

template <typename Fn>
auto CatchBridgeError(Fn&& fn) -> arrow::Result<decltype(fn())> {
  ClearBridgeErrorChannel();
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
  ClearBridgeErrorChannel();
  try {
    fn();
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    return BridgeErrorStatusFromException(e.what());
  } catch (...) {
    return arrow::Status::UnknownError("Rust bridge failed unexpectedly");
  }
}

/// Build the status for a classification a bridge reported out of band (the
/// typed side channel). THE decoder: every bridge -- vortex, lance, iceberg,
/// paimon -- funnels through this one table, so a code cannot mean one thing
/// for one format and something else for another.
///
/// `code` 0 means "no classification available" and yields the conservative
/// IOError fallback, same as an unrecognised code.
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
/// message is scanned for the marker and rebuilt.
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
