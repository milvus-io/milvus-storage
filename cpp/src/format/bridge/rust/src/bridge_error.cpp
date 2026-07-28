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

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"

namespace milvus_storage::bridge {
namespace {

// One marker, one parser: must stay byte-identical to the constants in
// rust/src/bridge_error.rs and rust/src/filesystem_c.rs.
constexpr std::string_view kBridgeErrCodeMarker = "__LOON_RUST_BRIDGE_ERRCODE__=";

struct ParsedBridgeError {
  std::string message;
  std::optional<int> ffi_err_code;
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
    return {std::string(error), std::nullopt};
  }

  auto code_start = marker_pos + kBridgeErrCodeMarker.size();
  auto code_end = code_start;
  while (code_end < error.size() && error[code_end] >= '0' && error[code_end] <= '9') {
    ++code_end;
  }
  if (code_end == code_start) {
    return {std::string(error), std::nullopt};
  }

  int ffi_err_code = 0;
  auto parse_result = std::from_chars(error.data() + code_start, error.data() + code_end, ffi_err_code);
  if (parse_result.ec != std::errc()) {
    return {StripBridgeMarker(error, marker_pos, code_end), std::nullopt};
  }

  return {StripBridgeMarker(error, marker_pos, code_end), ffi_err_code};
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

arrow::Status MakeBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseBridgeError(message);
  if (parsed.ffi_err_code.has_value()) {
    switch (*parsed.ffi_err_code) {
      case LOON_FILE_NOT_FOUND:
        return arrow::Status::IOError(parsed.message).WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
      case kBridgeErrCodeDataCorrupt:
        return arrow::Status::Invalid(parsed.message);
      case kBridgeErrCodeNotSupported:
        return arrow::Status::NotImplemented(parsed.message);
      default:
        break;
    }
    if (auto code = ExtendStatusCodeFromInt(*parsed.ffi_err_code); code.has_value()) {
      return MakeExtendError(*code, parsed.message, parsed.message);
    }
  }
  return arrow::Status::IOError(parsed.message);
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
    // Already structured -- nothing to decode.
    return WithBridgeContext(context, status);
  }
  // Discriminate on MARKER PRESENCE, not on StatusCode: mid-scan stream
  // errors arrive as whatever code the arrow C-stream import assigns (the
  // exporter maps Rust errors to EINVAL, so they surface as Invalid, NOT
  // IOError), still carrying the marker. Statuses without the marker (arrow's
  // own Invalid / OutOfMemory / NotImplemented) pass through untouched so
  // their StatusCode is never downgraded.
  if (status.message().find(kBridgeErrCodeMarker) == std::string::npos) {
    return WithBridgeContext(context, status);
  }
  return WithBridgeContext(context, MakeBridgeErrorStatus(status.message()));
}

std::shared_ptr<arrow::RecordBatchReader> WrapBridgeRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> inner,
                                                                      std::string context) {
  return std::make_shared<BridgeErrorTranslatingReader>(std::move(inner), std::move(context));
}

}  // namespace milvus_storage::bridge
