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

#include "bridge_error.h"

#include <cerrno>
#include <charconv>
#include <optional>
#include <string>
#include <utility>

#include <arrow/util/io_util.h>
#include <arrow/c/bridge.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::bridge {

namespace rust_ffi = ::milvus_storage::bridge::ffi;

namespace {

// One marker, one parser: must stay byte-identical to BRIDGE_ERRCODE_MARKER in
// rust/src/bridge_error.rs. Named for the whole Rust bridge, not just vortex:
// lance, iceberg and paimon all carry it now.
constexpr std::string_view kBridgeErrCodeMarker = "__LOON_RUST_BRIDGE_ERRCODE__=";

struct ParsedBridgeError {
  std::string message;
  std::optional<int> ffi_err_code;
  // Whether the marker was present at all. A marker with an unreadable code
  // still means "a Rust bridge produced this"; no marker means the status did
  // not come from a bridge and must not be reclassified.
  bool has_marker = false;
};

std::string StripBridgeMarker(std::string_view error, size_t marker_pos, size_t code_end) {
  auto message_start = code_end;
  if (message_start < error.size() && error[message_start] == ';') {
    ++message_start;
  }
  if (message_start < error.size() && error[message_start] == ' ') {
    ++message_start;
  }

  std::string message;
  message.reserve(error.size());
  message.append(error.substr(0, marker_pos));
  message.append(error.substr(message_start));
  if (message.empty()) {
    return "Unknown bridge error";
  }
  return message;
}

ParsedBridgeError ParseBridgeError(std::string_view error) {
  auto marker_pos = error.find(kBridgeErrCodeMarker);
  if (marker_pos == std::string_view::npos) {
    return {std::string(error), std::nullopt, false};
  }

  auto code_start = marker_pos + kBridgeErrCodeMarker.size();
  auto code_end = code_start;
  while (code_end < error.size() && error[code_end] >= '0' && error[code_end] <= '9') {
    ++code_end;
  }
  if (code_end == code_start) {
    return {std::string(error), std::nullopt, true};
  }

  int ffi_err_code = 0;
  auto parse_result = std::from_chars(error.data() + code_start, error.data() + code_end, ffi_err_code);
  if (parse_result.ec != std::errc()) {
    return {StripBridgeMarker(error, marker_pos, code_end), std::nullopt, true};
  }

  return {StripBridgeMarker(error, marker_pos, code_end), ffi_err_code, true};
}

class BridgeErrorTranslatingReader final : public arrow::RecordBatchReader {
  public:
  BridgeErrorTranslatingReader(std::shared_ptr<arrow::RecordBatchReader> inner, std::string context)
      : inner_(std::move(inner)), context_(std::move(context)) {}

  [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const override { return inner_->schema(); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    return TranslateBridgeStatus(context_, inner_->ReadNext(batch));
  }

  arrow::Status Close() override { return TranslateBridgeStatus(context_, inner_->Close()); }

  private:
  std::shared_ptr<arrow::RecordBatchReader> inner_;
  std::string context_;
};

}  // namespace

void ClearBridgeErrorChannel() { rust_ffi::clear_last_bridge_error(); }

namespace internal {
void RecordBridgeErrorForTest(int ffi_err_code, std::string_view message) {
  rust_ffi::record_bridge_error(ffi_err_code, rust::String(message.data(), message.size()));
}
}  // namespace internal

arrow::Status BridgeErrorStatusFromException(const char* what) {
  // The side channel first: it carries the producer's own verdict, and unlike
  // the message it cannot be spoofed by data that happens to contain marker
  // text. Only if nothing was recorded do we fall back to reading the message.
  auto info = rust_ffi::take_last_bridge_error();
  if (info.code != 0) {
    if (info.message.size() != 0) {
      return MakeBridgeErrorStatus(info.code, std::string_view(info.message.data(), info.message.size()));
    }
    auto parsed = ParseBridgeError(what == nullptr ? "Unknown bridge error" : what);
    return MakeBridgeErrorStatus(info.code, parsed.message);
  }
  return MakeBridgeErrorStatus(what == nullptr ? "Unknown bridge error" : what);
}

arrow::Status MakeBridgeErrorStatus(int ffi_err_code, std::string_view message) {
  std::string text(message);
  switch (ffi_err_code) {
    case LOON_FILE_NOT_FOUND:
      // The errno channel. No bridge in this repository emits it any more --
      // they use StorageNotFound -- but the vortex fork does, and its errors
      // are decoded here.
      return arrow::Status::IOError(std::move(text)).WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
    case LOON_GOT_EXCEPTION:
      // Compatibility for older bridge producers. New panic containment emits
      // the typed InternalInvariantViolated status below.
      return MakeExtendError(ExtendStatusCode::InternalInvariantViolated, text, text);
    case kBridgeErrCodeDataCorrupt:
      return MakeBridgeDataFormatStatus(std::move(text));
    case kBridgeErrCodeNotSupported:
      return arrow::Status::NotImplemented(std::move(text));
    default:
      break;
  }
  if (auto code = ExtendStatusCodeFromInt(ffi_err_code); code.has_value()) {
    return MakeExtendError(*code, text, text);
  }
  // Unclassified (0) or a code from a newer producer: conservative,
  // non-retriable, never invented.
  return arrow::Status::IOError(std::move(text));
}

arrow::Status MakeBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseBridgeError(message);
  return MakeBridgeErrorStatus(parsed.ffi_err_code.value_or(0), parsed.message);
}

std::optional<arrow::Status> DecodeBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseBridgeError(message);
  if (!parsed.has_marker) {
    return std::nullopt;
  }
  return MakeBridgeErrorStatus(parsed.ffi_err_code.value_or(0), parsed.message);
}

std::string StripBridgeErrorMarker(std::string_view message) { return ParseBridgeError(message).message; }

arrow::Status MakeBridgeDataFormatStatus(std::string message) {
  // One fact, one construction. This used to build the status by hand with
  // StatusCode::Invalid, which contradicted the rule MakeExtendError enforces:
  // Invalid is reserved for failures detected BEFORE any IO, and bytes that do
  // not decode are by definition detected after it. The result was the same
  // condition reaching callers as Invalid from one format and IOError from
  // another, so anything branching on the StatusCode saw a difference that
  // does not exist. The detail -- and therefore the FFI code and the segcore
  // landing -- is unchanged.
  return MakeExtendError(ExtendStatusCode::DataCorrupted, message, message);
}

arrow::Status WithBridgeContext(std::string_view context, const arrow::Status& status) {
  if (status.ok() || context.empty()) {
    return status;
  }
  std::string message;
  message.reserve(context.size() + 2 + status.message().size());
  message.append(context);
  message.append(": ");
  message.append(status.message());
  // Same StatusCode and detail (ExtendStatusDetail / errno) — only the message
  // gains context; classification is never altered here.
  return {status.code(), std::move(message), status.detail()};
}

arrow::Status TranslateBridgeStatus(std::string_view context, const arrow::Status& status) {
  if (status.ok()) {
    return status;
  }
  if (ExtendStatusDetail::UnwrapStatus(status) || arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    // Already structured — nothing to decode.
    return WithBridgeContext(context, status);
  }
  if (auto decoded = DecodeBridgeErrorStatus(status.message()); decoded.has_value()) {
    return WithBridgeContext(context, *decoded);
  }
  // No marker: this failure did not come from a Rust bridge (arrow's own
  // Invalid/NotImplemented while importing a C stream, for instance). Keep the
  // producer's classification instead of flattening it into an IOError.
  return WithBridgeContext(context, status);
}

std::shared_ptr<arrow::RecordBatchReader> WrapBridgeRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> inner,
                                                                      std::string context) {
  return std::make_shared<BridgeErrorTranslatingReader>(std::move(inner), std::move(context));
}

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> ImportBridgeChunkedArray(ArrowArrayStream* stream,
                                                                             std::string_view context) {
  auto result = arrow::ImportChunkedArray(stream);
  if (!result.ok()) {
    return TranslateBridgeStatus(context, result.status());
  }
  return std::move(result).ValueOrDie();
}

}  // namespace milvus_storage::bridge
