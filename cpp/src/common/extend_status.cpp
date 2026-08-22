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
#include <sstream>
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
  ErrorCategory category;
  std::string_view s3_code;

  [[nodiscard]] constexpr bool retryable() const { return category == ErrorCategory::Retryable; }
};

// Generated from the single table in ffi_error_code.h, which is also the source
// for the FFI constants, error_to_string and loon_ffi_error_category. Editing
// this array by hand is not possible on purpose.
constexpr ExtendStatusCodeMetadata kExtendStatusCodeMetadata[] = {
#define MILVUS_STORAGE_EXTEND_STATUS_METADATA_ENTRY(name, code, symbol, category, s3_code) \
  {ExtendStatusCode::name, #name, static_cast<ErrorCategory>(category), s3_code},
    LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_EXTEND_STATUS_METADATA_ENTRY)
#undef MILVUS_STORAGE_EXTEND_STATUS_METADATA_ENTRY
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

ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code) : code_{code} {}
ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code, const char* extra_info)
    : ExtendStatusDetail(code, std::string(extra_info)) {}
ExtendStatusDetail::ExtendStatusDetail(ExtendStatusCode code, std::string extra_info)
    : code_{code}, extra_info_(std::move(extra_info)) {}

const char* ExtendStatusDetail::type_id() const { return kErrorDetailTypeId; }

std::string ExtendStatusDetail::ToString() const { return CodeAsString() + ": " + extra_info_; }

ExtendStatusCode ExtendStatusDetail::code() const { return code_; }

std::string ExtendStatusDetail::extra_info() const { return extra_info_; }

bool ExtendStatusDetail::retryable() const { return RetryableForExtendStatusCode(code_); }

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

ErrorCategory CategoryForExtendStatusCode(ExtendStatusCode code) {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code); metadata != nullptr) {
    return metadata->category;
  }
  // An out-of-range value is not classifiable; consumers must treat Unknown as
  // non-retryable rather than guess.
  return ErrorCategory::Unknown;
}

bool RetryableForExtendStatusCode(ExtendStatusCode code) {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code); metadata != nullptr) {
    return metadata->retryable();
  }
  return false;
}

std::string_view S3CodeForExtendStatusCode(ExtendStatusCode code) {
  if (const auto* metadata = FindExtendStatusCodeMetadata(code); metadata != nullptr) {
    return metadata->s3_code;
  }
  return {};
}

namespace {

// The conditions we detect before issuing any IO: nothing was attempted, so
// reporting them as an IO failure would be a lie. (This used to add "and the
// fallback would file them under DataFormatBroken" -- no longer true, and the
// fallback change is precisely why: an unclassified Invalid now lands on
// StorageError. The arrow-code choice still matters on its own, because callers
// branch on IsIOError.)
//
// Deliberately a small explicit set rather than a function of the category.
// Category answers the generic handling question; the arrow code answers what
// kind of operation failed. A System failure can be either -- unusable
// `extfs.*` properties never touch the network, while an S3 403 already did.
bool IsPreIoValidationFailure(ExtendStatusCode code) {
  switch (code) {
    case ExtendStatusCode::PackedInvalidArgs:
    case ExtendStatusCode::StorageConfigInvalid:
      return true;
    default:
      return false;
  }
}

}  // namespace

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info) {
  // arrow's StatusCode says what kind of operation failed; our category says
  // what generic handling is safe. They are not the same axis and must not be
  // derived from each other: an S3 403 is System but it is still, to arrow and
  // to every caller branching on `IsIOError()`, an IO failure.
  //
  // Invalid is therefore reserved for the conditions detected *before* any IO
  // is attempted -- a malformed argument, an unparseable URI, unusable
  // configuration. Everything else is IOError.
  auto arrow_code = IsPreIoValidationFailure(code) ? arrow::StatusCode::Invalid : arrow::StatusCode::IOError;
  return {arrow_code, std::move(message), std::make_shared<ExtendStatusDetail>(code, std::move(extra_info))};
}

arrow::Status WrapExtendError(ExtendStatusCode code, std::string message, const arrow::Status& cause) {
  auto cause_message = cause.ToString();
  auto wrapped_message = fmt::format("{}: {}", message, cause_message);
  if (cause.detail()) {
    return {cause.code(), std::move(wrapped_message), cause.detail()};
  }
  // Do not relabel a raw allocation failure as an unrelated domain error such
  // as malformed metadata. The Arrow status code is the classification here:
  // ToSegcoreError maps it to MemAllocateFailed.
  if (cause.IsOutOfMemory()) {
    return cause.WithMessage(std::move(wrapped_message));
  }
  return MakeExtendError(code, std::move(wrapped_message), cause_message);
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
// Direct C++ consumers reach segcore through ToSegcoreError below. FFI entry
// points use RETURN_ARROW_ERROR instead, but preserve the same
// ExtendStatusDetail so C/Python/Java consumers observe the exact code and
// category. A status without detail takes the conservative plain-Arrow fallback.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
milvus::ErrorCode ToSegcoreErrorCode(ExtendStatusCode code) {
  switch (code) {
    case ExtendStatusCode::PackedInvalidArgs:
    case ExtendStatusCode::InternalInvariantViolated:
      // Internal API misuse (an internally derived index out of range, an
      // unreachable branch, a reused closed reader) -- our bug, not an end
      // user's parameter. Public C++ argument validation stays a plain Arrow
      // Invalid status instead of attaching this detail. 2042 would make
      // milvus tell a user their query is wrong, and 2044 sent whoever was on
      // call to inspect a healthy object store.
      return milvus::UnexpectedError;  // 2001
    case ExtendStatusCode::PackedIO:
      // Direct C++ consumers receive the conservative non-retryable storage
      // code. The packed writer FFI preserves PackedIO through
      // RETURN_ARROW_ERROR, where consumers can inspect its category directly.
      return milvus::StorageError;  // 2044
    case ExtendStatusCode::PackedMetadataCorrupted:
    case ExtendStatusCode::PackedFileCorrupted:
      return milvus::DataFormatBroken;  // 2024, non-retryable data-format failure
    case ExtendStatusCode::PackedUnexpected:
      // Same reasoning: "unexpected" is a defect report, not a storage verdict.
      return milvus::UnexpectedError;  // 2001
    case ExtendStatusCode::StorageTransientNetwork:
    case ExtendStatusCode::StorageTransientTimeout:
    case ExtendStatusCode::StorageTransientThrottling:
    case ExtendStatusCode::StorageTransientService:
      return milvus::StorageTransientError;  // 2045
    case ExtendStatusCode::StorageConflict:
    case ExtendStatusCode::StoragePreConditionFailed:
    case ExtendStatusCode::TxnExhaustedRetry:
    case ExtendStatusCode::TxnResolutionFailed:
      // Conflict-aware callers inspect the ExtendStatusCode and decide whether
      // to re-read/rebase. The generic segcore retry code would instead replay
      // a generic caller, so do not map conflicts to the transient code.
      return milvus::StorageError;  // 2044
    case ExtendStatusCode::DataCorrupted:
    case ExtendStatusCode::VortexDataFormat:
      return milvus::DataFormatBroken;  // 2024
    case ExtendStatusCode::StorageBucketNotFound:
      // Not ObjectNotExist: nothing was lost. The deployment names a bucket
      // that is not there, and milvus has a code that says exactly that.
      return milvus::BucketInvalid;  // 2016
    case ExtendStatusCode::StorageConfigInvalid:
      // The storage location spec -- property map, URI, or both -- is unusable.
      // Non-retriable, and NOT reported as the caller's fault: this producer
      // cannot tell whether the strings came from the operator's config or an
      // external-source definition. The external-table boundary may present
      // this as the unified System code LOON_SOURCE_INVALID; all other
      // consumers keep this more specific System verdict.
      return milvus::ConfigInvalid;  // 2006
    case ExtendStatusCode::StorageNoSuchUpload:
      // A dead multipart upload handle is a write-path fact, not "data missing".
      // Consumers treat 2017 as "re-read the manifest / the file was GC'd"; a
      // stale upload id is neither. Non-retryable StorageError keeps it out of
      // the data-missing bucket without inviting a resend against the dead id.
      return milvus::StorageError;  // 2044
    case ExtendStatusCode::StorageNotFound:
      // The object/bucket is gone: non-retryable and fine-grained -- consumers can
      // distinguish "data missing" (stale loadinfo, GC'd file) from a generic
      // storage failure. Never transient/2045: a retry/reroute hits the same
      // shared object store and fails identically.
      return milvus::ObjectNotExist;  // 2017
    case ExtendStatusCode::StorageAccessDenied:
      // Operator credentials, not the caller's request and not a bug of ours.
      // Non-retriable either way, but it must page whoever owns the config
      // rather than be filed as a generic storage failure.
      return milvus::ConfigInvalid;  // 2006
  }
  return milvus::StorageError;  // out-of-range value: safe non-retriable fallback
}
#pragma GCC diagnostic pop

// Internal FFI codes (1-13) never carry an ExtendStatusDetail, so they need
// their own switch here. ExtendStatusCodes delegate to the enum switch above.
// Anything unrecognized -- LOON_SUCCESS, retired values, a code from a newer
// producer -- degrades to the conservative non-retryable StorageError.
milvus::ErrorCode ToSegcoreErrorCode(int ffi_err_code) {
  if (auto code = ExtendStatusCodeFromInt(ffi_err_code); code.has_value()) {
    return ToSegcoreErrorCode(*code);
  }
  switch (ffi_err_code) {
    case LOON_MEMORY_ERROR:
      // Not a storage failure; segcore has a code that points at the node that
      // could not allocate. Non-retriable either way.
      return milvus::MemAllocateFailed;  // 2034
    case LOON_USER_INVALID_ARGUMENT:
      // The one code minted with the User category: a caller-owned value
      // violates the API contract, so milvus reports it back to the caller.
      return milvus::InvalidParameter;  // 2042
    case LOON_INVALID_PROPERTIES:
      // Deployment configuration an operator can fix -- same reasoning as
      // StorageConfigInvalid.
      return milvus::ConfigInvalid;  // 2006
    case LOON_INVALID_ARGS:
    case LOON_LOGICAL_ERROR:
    case LOON_GOT_EXCEPTION:
    case LOON_UNREACHABLE_ERROR:
      // C ABI misuse, an internal invariant, an exception escaping the boundary,
      // an unreachable branch -- our bug, not a storage incident.
      return milvus::UnexpectedError;  // 2001
    case LOON_FILE_NOT_FOUND:
      return milvus::ObjectNotExist;  // 2017
    case LOON_NOT_SUPPORT:
      // Capability absence is a deployment/version fact, not a storage
      // incident: retrying never helps and the caller did not cause it.
      // Same landing as StorageAccessDenied -- page whoever owns the config.
      return milvus::ConfigInvalid;  // 2006
    case LOON_ARROW_ERROR:
    case LOON_FAULT_INJECT_ERROR:
    case LOON_SOURCE_INVALID:
    default:
      // Unclassified arrow, test-only fault injection, a missing capability,
      // an unusable external source: System "report a failure", no better
      // milvus code -- conservative non-retryable.
      return milvus::StorageError;  // 2044
  }
}

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

  // Capability absence has one meaning library-wide (FFIErrorCodeFromExtendStatus
  // maps it to LOON_NOT_SUPPORT); the direct-C++ path must reach the same
  // landing instead of collapsing into generic storage failure.
  if (status.IsNotImplemented()) {
    return {milvus::ConfigInvalid, status.ToString()};
  }

  // Running out of memory is not a storage failure, and segcore has a code that
  // says so. Filed as StorageError it sent whoever was on call to inspect an
  // object store that never saw the request; 2034 points at the node that could
  // not allocate. Non-retriable either way: this layer cannot promise that a
  // replay finds more memory, and the caller owns any backpressure decision.
  if (status.IsOutOfMemory()) {
    return {milvus::MemAllocateFailed, status.ToString()};
  }

  // No structured ExtendStatusDetail attached: this is the LIVE read path (plain
  // arrow from FileRowGroupReader / v3 api::Reader / ArrowFileSystem). Plain
  // filesystem not-found and allocation failure are handled above. Other
  // propagated IO errors already spent the shared S3 SDK retry budget, so any
  // remaining unstructured failure is the generic unexpected, non-retriable
  // StorageError. Data-format errors must carry an explicit code.
  return {milvus::StorageError, status.ToString()};
}

}  // namespace milvus_storage
