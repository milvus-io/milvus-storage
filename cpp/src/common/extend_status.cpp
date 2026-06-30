// Copyright 2024 Zilliz
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
#include "milvus-storage/common/extend_status.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

namespace milvus_storage {

const char* kErrorDetailTypeId = "milvus_storage::ExtendStatusDetail";

ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code) : code_{code} {}
ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code, std::string extra_info)
    : code_{code}, extra_info_(std::move(extra_info)) {}

const char* ExtendStatusDetail::type_id() const { return kErrorDetailTypeId; }

std::string ExtendStatusDetail::ToString() const { return CodeAsString() + ": " + extra_info_; }

ExtendStatusCode ExtendStatusDetail::code() const { return code_; }

std::string ExtendStatusDetail::extra_info() const { return extra_info_; }

std::string ExtendStatusDetail::CodeAsString() const {
  switch (code()) {
    case ExtendStatusCode::PackedInvalidArgs:
      return "PackedInvalidArgs";
    case ExtendStatusCode::PackedStorageIO:
      return "PackedStorageIO";
    case ExtendStatusCode::PackedMetadataCorrupted:
      return "PackedMetadataCorrupted";
    case ExtendStatusCode::PackedFileCorrupted:
      return "PackedFileCorrupted";
    case ExtendStatusCode::PackedArrowError:
      return "PackedArrowError";
    case ExtendStatusCode::PackedUnexpected:
      return "PackedUnexpected";
    case ExtendStatusCode::AwsErrorNoSuchUpload:
      return "AwsErrorNoSuchUpload";
    case ExtendStatusCode::AwsErrorConflict:
      return "AwsErrorConflict";
    case ExtendStatusCode::AwsErrorPreConditionFailed:
      return "AwsErrorPreConditionFailed";
    case ExtendStatusCode::TxnExhaustedRetry:
      return "TxnExhaustedRetry";
    case ExtendStatusCode::TxnResolutionFailed:
      return "TxnResolutionFailed";
    default:
      return "Unknown";
  }
}

void ExtendStatusDetail::set_extra_info(std::string extra_info) { extra_info_ = std::move(extra_info); }

std::shared_ptr<ExtendStatusDetail> ExtendStatusDetail::UnwrapStatus(const arrow::Status& status) {
  if (!status.detail() || status.detail()->type_id() != kErrorDetailTypeId) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<ExtendStatusDetail>(status.detail());
}

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info) {
  auto arrow_code =
      code == ExtendStatusCode::PackedInvalidArgs ? arrow::StatusCode::Invalid : arrow::StatusCode::IOError;
  return {arrow_code, std::move(message), std::make_shared<ExtendStatusDetail>(code, std::move(extra_info))};
}

arrow::Status WrapExtendError(ExtendStatusCode code, std::string message, const arrow::Status& cause) {
  auto detail = ExtendStatusDetail::UnwrapStatus(cause);
  auto wrapped_code = detail ? detail->code() : code;
  auto cause_message = cause.ToString();
  return MakeExtendError(wrapped_code, fmt::format("{}: {}", message, cause_message), cause_message);
}

// Map a producer-side ExtendStatusCode to the shared milvus ErrorCode that the
// segcore boundary (and ultimately the Go retry policy) consumes. This is the
// single place milvus-storage classifies its own codes ("producer owns
// classification").
//
// It is deliberately a switch with NO `default:` plus a post-switch fallback:
//   * a `default:` inside the switch would suppress -Wswitch, so a newly added
//     ExtendStatusCode could silently fall through to the wrong bucket;
//   * the post-switch `return` satisfies -Wreturn-type and guards out-of-range
//     values, without suppressing the exhaustiveness warning.
// The surrounding pragma turns -Wswitch into an error so adding an
// ExtendStatusCode without classifying it here breaks the build (the
// extend_status_test.cpp coverage is the runtime backstop).
//
// Invariant: the retriable verdict carried by the returned code is what matters
// most. A transient storage IO failure MUST map to the retriable
// StorageTransientError(2045); it must never collapse into the non-retriable
// StorageError(2044), which would turn a recoverable IO blip into a hard
// failure.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
milvus::ErrorCode ToSegcoreErrorCode(ExtendStatusCode code) {
  switch (code) {
    case ExtendStatusCode::PackedInvalidArgs:
      return milvus::InvalidParameter;  // 2042, caller's fault (non-retriable input)
    case ExtendStatusCode::PackedStorageIO:
      return milvus::StorageTransientError;  // 2045, retriable storage IO
    case ExtendStatusCode::PackedMetadataCorrupted:
    case ExtendStatusCode::PackedFileCorrupted:
      return milvus::DataFormatBroken;  // 2024, permanent data corruption
    case ExtendStatusCode::PackedArrowError:
    case ExtendStatusCode::PackedUnexpected:
      return milvus::StorageError;  // 2044, permanent internal storage error
    case ExtendStatusCode::AwsErrorNoSuchUpload:
    case ExtendStatusCode::AwsErrorConflict:
    case ExtendStatusCode::AwsErrorPreConditionFailed:
    case ExtendStatusCode::TxnExhaustedRetry:
    case ExtendStatusCode::TxnResolutionFailed:
      // S3 multipart / precondition / transaction failures: conservatively
      // permanent here (the retry budget is already spent or the precondition
      // genuinely failed). Promote to a more specific code if a real retriable
      // case is identified.
      return milvus::StorageError;  // 2044
  }
  return milvus::StorageError;  // out-of-range value: safe non-retriable fallback
}
#pragma GCC diagnostic pop

milvus::SegcoreError ToSegcoreError(const arrow::Status& status) {
  if (status.ok()) {
    return milvus::SegcoreError::success();
  }

  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  if (detail) {
    return {ToSegcoreErrorCode(detail->code()), status.ToString()};
  }

  // No structured ExtendStatusDetail attached: fall back to arrow's coarse
  // status code. Transient IO / OOM stay retriable, malformed data is permanent
  // corruption, and anything else is an internal storage error.
  milvus::ErrorCode code;
  if (status.IsOutOfMemory()) {
    code = milvus::MemAllocateFailed;  // 2034, retriable
  } else if (status.IsIOError()) {
    code = milvus::StorageTransientError;  // 2045, retriable
  } else if (status.IsInvalid() || status.IsTypeError() || status.IsKeyError()) {
    code = milvus::DataFormatBroken;  // 2024, permanent corruption
  } else {
    code = milvus::StorageError;  // 2044, permanent internal error
  }
  return {code, status.ToString()};
}

}  // namespace milvus_storage
