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

namespace milvus_storage::bridge {

namespace {

// One marker, one parser: must stay byte-identical to BRIDGE_ERRCODE_MARKER in
// rust/src/bridge_error.rs. Named for the whole Rust bridge, not just vortex:
// lance, iceberg and paimon all carry it now.
constexpr std::string_view kBridgeErrCodeMarker = "__LOON_RUST_BRIDGE_ERRCODE__=";

constexpr std::string_view kArrowIoPrefix = "Io error: ";

struct ParsedBridgeError {
  std::string message;
  std::optional<int> ffi_err_code;
  // Whether a complete, trusted frame was present. Marker-like text with a
  // malformed code or missing delimiter remains an ordinary diagnostic.
  bool has_marker = false;
  // The producer explicitly framed this as an IO operation. This matters for
  // codes such as StorageConfigInvalid that can be detected before or after IO.
  bool io_transport = false;
};

std::string StripBridgeMarker(std::string_view error, size_t code_end) {
  auto message_start = code_end;
  if (message_start < error.size() && error[message_start] == ';') {
    ++message_start;
  }
  if (message_start < error.size() && error[message_start] == ' ') {
    ++message_start;
  }

  std::string message;
  message.reserve(error.size());
  message.append(error.substr(message_start));
  if (message.empty()) {
    return "Unknown bridge error";
  }
  return message;
}

ParsedBridgeError ParseBridgeError(std::string_view error) {
  size_t frame_start = 0;
  bool io_transport = false;
  if (error.starts_with(kBridgeErrCodeMarker)) {
    frame_start = 0;
  } else if (error.starts_with(kArrowIoPrefix) &&
             error.substr(kArrowIoPrefix.size()).starts_with(kBridgeErrCodeMarker)) {
    frame_start = kArrowIoPrefix.size();
    io_transport = true;
  } else {
    // A marker anywhere else can be part of a caller-provided URI/path in an
    // ordinary Lance, Paimon, or Vortex diagnostic and is not framing.
    return {std::string(error), std::nullopt, false, false};
  }

  auto code_start = frame_start + kBridgeErrCodeMarker.size();
  auto code_end = code_start;
  while (code_end < error.size() && error[code_end] >= '0' && error[code_end] <= '9') {
    ++code_end;
  }
  // A trusted frame is exactly "<marker><digits>; ...". Merely starting with
  // marker-like text is not enough: malformed caller text must stay an
  // ordinary diagnostic instead of entering the classification channel.
  if (code_end == code_start || code_end >= error.size() || error[code_end] != ';') {
    return {std::string(error), std::nullopt, false, false};
  }

  int ffi_err_code = 0;
  auto parse_result = std::from_chars(error.data() + code_start, error.data() + code_end, ffi_err_code);
  if (parse_result.ec != std::errc() || parse_result.ptr != error.data() + code_end) {
    return {std::string(error), std::nullopt, false, false};
  }

  return {StripBridgeMarker(error, code_end), ffi_err_code, true, io_transport};
}

arrow::Status MakeBridgeErrorStatusImpl(int ffi_err_code, std::string_view message, bool io_transport) {
  std::string text(message);
  switch (ffi_err_code) {
    case LOON_FILE_NOT_FOUND:
      return arrow::Status::IOError(std::move(text)).WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
    case LOON_GOT_EXCEPTION:
      return MakeExtendError(ExtendStatusCode::InternalInvariantViolated, text, text);
    case kBridgeErrCodeDataCorrupt:
      return MakeBridgeDataFormatStatus(std::move(text));
    case kBridgeErrCodeNotSupported:
      return arrow::Status::NotImplemented(std::move(text));
    default:
      break;
  }
  if (auto code = ExtendStatusCodeFromInt(ffi_err_code); code.has_value()) {
    if (io_transport) {
      return MakeExtendError(*code, arrow::StatusCode::IOError, text, text);
    }
    return MakeExtendError(*code, text, text);
  }
  return arrow::Status::IOError(std::move(text));
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

arrow::Status BridgeErrorStatusFromException(const char* what) {
  return MakeBridgeErrorStatus(what == nullptr ? "Unknown bridge error" : what);
}

arrow::Status MakeBridgeErrorStatus(int ffi_err_code, std::string_view message) {
  return MakeBridgeErrorStatusImpl(ffi_err_code, message, true);
}

arrow::Status MakeBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseBridgeError(message);
  return MakeBridgeErrorStatusImpl(parsed.ffi_err_code.value_or(0), parsed.message, parsed.io_transport);
}

std::optional<arrow::Status> DecodeBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseBridgeError(message);
  if (!parsed.has_marker) {
    return std::nullopt;
  }
  return MakeBridgeErrorStatusImpl(parsed.ffi_err_code.value_or(0), parsed.message, parsed.io_transport);
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
  auto parsed = ParseBridgeError(status.message());
  if (parsed.has_marker) {
    auto decoded = MakeBridgeErrorStatusImpl(parsed.ffi_err_code.value_or(0), parsed.message,
                                             parsed.io_transport || status.IsIOError());
    return WithBridgeContext(context, decoded);
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
