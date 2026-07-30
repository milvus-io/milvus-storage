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
#include <string_view>

#include <arrow/record_batch.h>
#include <arrow/status.h>

namespace milvus_storage::bridge {

// Shared decoding of classified errors coming out of the Rust cxx bridges
// (vortex / lance / iceberg).
//
// The cxx boundary only carries an error as a display string; the Rust side
// (rust/src/bridge_error.rs, vortex's filesystem_c.rs) embeds the
// classification as "__LOON_RUST_BRIDGE_ERRCODE__=<code>; message". The
// helpers here parse and strip that marker and rebuild a structured
// arrow::Status:
//   * code 12 (LOON_FILE_NOT_FOUND)      -> IOError + ENOENT detail
//   * ExtendStatusCode values (101-114)  -> IOError + ExtendStatusDetail
//   * bridge-private codes (>= 1000, never cross the C ABI):
//       1000 unclassified   -> plain IOError          (conservative fallback)
//       1001 data-corrupt   -> Status::Invalid       (permanent data error)
//       1002 not-supported  -> Status::NotImplemented
//   * no / unknown marker                -> plain IOError (conservative
//     non-retriable fallback; never invent retriability)

// Bridge-private marker codes; keep in sync with rust/src/bridge_error.rs.
inline constexpr int kBridgeErrCodeUnclassified = 1000;
inline constexpr int kBridgeErrCodeDataCorrupt = 1001;
inline constexpr int kBridgeErrCodeNotSupported = 1002;

/// Decode a raw bridge error message (marker stripped) into a structured
/// arrow::Status per the table above.
arrow::Status MakeBridgeErrorStatus(std::string_view message);

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

}  // namespace milvus_storage::bridge
