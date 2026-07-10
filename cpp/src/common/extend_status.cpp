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

#include <cerrno>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/io_util.h>
#include <fmt/format.h>

namespace milvus_storage {
namespace {

const char* kErrorDetailTypeId = "milvus_storage::ExtendStatusDetail";

struct ExtendStatusCodeMetadata {
  ExtendStatusCode code;
  std::string_view name;
  bool retryable;
};

constexpr ExtendStatusCodeMetadata kExtendStatusCodeMetadata[] = {
    {ExtendStatusCode::PackedInvalidArgs, "PackedInvalidArgs", false},
    {ExtendStatusCode::PackedStorageIO, "PackedStorageIO", false},
    {ExtendStatusCode::PackedMetadataCorrupted, "PackedMetadataCorrupted", false},
    {ExtendStatusCode::PackedFileCorrupted, "PackedFileCorrupted", false},
    {ExtendStatusCode::PackedArrowError, "PackedArrowError", false},
    {ExtendStatusCode::PackedUnexpected, "PackedUnexpected", false},
    {ExtendStatusCode::AwsErrorNoSuchUpload, "AwsErrorNoSuchUpload", true},
    {ExtendStatusCode::AwsErrorConflict, "AwsErrorConflict", false},
    {ExtendStatusCode::AwsErrorPreConditionFailed, "AwsErrorPreConditionFailed", false},
    {ExtendStatusCode::AwsErrorNotFound, "AwsErrorNotFound", false},
    {ExtendStatusCode::AwsErrorAccessDenied, "AwsErrorAccessDenied", false},
    {ExtendStatusCode::AwsErrorNonRetryable, "AwsErrorNonRetryable", false},
    {ExtendStatusCode::StorageTransientNetwork, "StorageTransientNetwork", true},
    {ExtendStatusCode::StorageTransientTimeout, "StorageTransientTimeout", true},
    {ExtendStatusCode::StorageTransientThrottling, "StorageTransientThrottling", true},
    {ExtendStatusCode::StorageTransientService, "StorageTransientService", true},
    {ExtendStatusCode::TxnExhaustedRetry, "TxnExhaustedRetry", false},
    {ExtendStatusCode::TxnResolutionFailed, "TxnResolutionFailed", false},
};

const ExtendStatusCodeMetadata* FindExtendStatusCodeMetadata(ExtendStatusCode code) {
  for (const auto& metadata : kExtendStatusCodeMetadata) {
    if (metadata.code == code) {
      return &metadata;
    }
  }
  return nullptr;
}

const ExtendStatusCodeMetadata* FindExtendStatusCodeMetadata(int code) {
  for (const auto& metadata : kExtendStatusCodeMetadata) {
    if (static_cast<int>(metadata.code) == code) {
      return &metadata;
    }
  }
  return nullptr;
}

}  // namespace

ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code)
    : code_{code}, retryable_{DefaultRetryableForExtendStatusCode(code)} {}
ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code, const char* extra_info)
    : ExtendStatusDetail(code, std::string(extra_info)) {}
ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code, std::string extra_info)
    : code_{code}, extra_info_(std::move(extra_info)), retryable_{DefaultRetryableForExtendStatusCode(code)} {}

const char* ExtendStatusDetail::type_id() const { return kErrorDetailTypeId; }

std::string ExtendStatusDetail::ToString() const { return CodeAsString() + ": " + extra_info_; }

ExtendStatusCode ExtendStatusDetail::code() const { return code_; }

std::string ExtendStatusDetail::extra_info() const { return extra_info_; }

bool ExtendStatusDetail::retryable() const { return retryable_; }

std::string ExtendStatusDetail::CodeAsString() const {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code()); metadata != nullptr) {
    return std::string(metadata->name);
  }
  return "Unknown";
}

void ExtendStatusDetail::set_extra_info(std::string extra_info) { extra_info_ = std::move(extra_info); }

std::shared_ptr<ExtendStatusDetail> ExtendStatusDetail::UnwrapStatus(const arrow::Status& status) {
  if (!status.detail() || status.detail()->type_id() != kErrorDetailTypeId) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<ExtendStatusDetail>(status.detail());
}

std::optional<ExtendStatusCode> ExtendStatusCodeFromInt(int code) {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code); metadata != nullptr) {
    return metadata->code;
  }
  return std::nullopt;
}

bool DefaultRetryableForExtendStatusCode(ExtendStatusCode code) {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code); metadata != nullptr) {
    return metadata->retryable;
  }
  return false;
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
// Retriability model (do not repeat the "v2 retries, v3 doesn't" myth):
// object-storage IO retry does NOT live in the packed / format / api::Reader
// layers. It lives once in the shared S3 ArrowFileSystem (AWS SDK
// DefaultRetryStrategy), which every read path -- v1 binlog, v2
// FileRowGroupReader, v3 api::Reader -- runs on top of. So an IO error that
// propagates up here already spent the S3 SDK retry budget, equally for v2 and
// v3; there is no per-generation retry asymmetry.
//
// Retriability is therefore decided by whether a DISTINCT upper-layer retry can
// still help: querynode can reroute a failed read to another replica/node (a
// different network path / endpoint), or the failure was a node-local transient.
// Plain IO does not assume that path and is classified conservatively as
// non-retriable StorageError/2044.
//
// Two callers reach segcore ErrorCode differently:
//   1. A status carrying an ExtendStatusDetail (Packed*/Aws*/Txn) is classified
//      by this switch. NOTE: as of this writing NO live milvus consumer routes a
//      Packed* status through here -- packed_reader_c/packed_writer_c hardcode
//      FileReadFailed/FileWriteFailed and drop the ExtendStatusCode -- so this
//      switch is a reserved, forward-looking classification, not a hot path.
//   2. A status with NO detail (plain arrow) is the LIVE segcore/storage read
//      path; its plain IO is classified as non-retriable StorageError/2044 via
//      the no-detail fallback of ToSegcoreError below, NOT this switch.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
milvus::ErrorCode ToSegcoreErrorCode(ExtendStatusCode code) {
  switch (code) {
    case ExtendStatusCode::PackedInvalidArgs:
      return milvus::InvalidParameter;  // 2042, caller's fault (non-retriable input)
    case ExtendStatusCode::PackedStorageIO:
      // Conservatively non-retriable, but this is a DORMANT branch: no live
      // consumer routes a Packed* status here (the packed C-APIs hardcode
      // FileReadFailed/FileWriteFailed and drop the code). Do NOT justify this
      // with "v2 retries internally" -- the S3 SDK retry is shared by v2 and v3
      // alike. If a real direct-link consumer ever appears, revisit: validate
      // its retry semantics before changing this non-retriable classification.
      return milvus::StorageError;  // 2044 (dormant; conservative)
    case ExtendStatusCode::PackedMetadataCorrupted:
    case ExtendStatusCode::PackedFileCorrupted:
      return milvus::DataFormatBroken;  // 2024, permanent data corruption
    case ExtendStatusCode::PackedArrowError:
    case ExtendStatusCode::PackedUnexpected:
      return milvus::StorageError;  // 2044, permanent internal storage error
    case ExtendStatusCode::AwsErrorNoSuchUpload:
      // The SDK has exhausted retries for the failed multipart upload state,
      // but an outer operation retry can create a fresh upload and succeed.
      return milvus::StorageTransientError;  // 2045
    case ExtendStatusCode::StorageTransientNetwork:
    case ExtendStatusCode::StorageTransientTimeout:
    case ExtendStatusCode::StorageTransientThrottling:
    case ExtendStatusCode::StorageTransientService:
      return milvus::StorageTransientError;  // 2045
    case ExtendStatusCode::AwsErrorConflict:
    case ExtendStatusCode::AwsErrorPreConditionFailed:
    case ExtendStatusCode::TxnExhaustedRetry:
    case ExtendStatusCode::TxnResolutionFailed:
      // S3 precondition / transaction failures: conservatively permanent here
      // (the precondition genuinely failed, or the transaction-level retry
      // budget is already spent).
      return milvus::StorageError;  // 2044
    case ExtendStatusCode::AwsErrorNotFound:
      // The object/bucket is gone: permanent, and fine-grained -- consumers can
      // distinguish "data missing" (stale loadinfo, GC'd file) from a generic
      // storage failure. Never transient/2045: a retry/reroute hits the same
      // shared object store and fails identically.
      return milvus::ObjectNotExist;  // 2017, permanent
    case ExtendStatusCode::AwsErrorAccessDenied:
    case ExtendStatusCode::AwsErrorNonRetryable:
      // Bad credentials/permissions, or the AWS SDK itself judged the error
      // non-retryable. Same rule: never transient/2045, or querynode would
      // retry-storm a request that can never succeed.
      return milvus::StorageError;  // 2044, permanent
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

  if (arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    return {milvus::ObjectNotExist, status.ToString()};
  }

  // No structured ExtendStatusDetail attached: this is the LIVE read path (plain
  // arrow from FileRowGroupReader / v3 api::Reader / ArrowFileSystem). Plain
  // filesystem not-found is handled above through ENOENT. Other propagated IO
  // errors already spent the shared S3 SDK retry budget and are classified
  // conservatively as non-retriable StorageError/2044. OOM is retriable;
  // malformed data is permanent corruption; anything else internal.
  milvus::ErrorCode code;
  if (status.IsOutOfMemory()) {
    code = milvus::MemAllocateFailed;  // 2034, retriable
  } else if (status.IsIOError()) {
    code = milvus::StorageError;  // 2044, non-retriable
  } else if (status.IsInvalid() || status.IsTypeError() || status.IsKeyError()) {
    code = milvus::DataFormatBroken;  // 2024, permanent corruption
  } else {
    code = milvus::StorageError;  // 2044, permanent internal error
  }
  return {code, status.ToString()};
}

}  // namespace milvus_storage
