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
  ErrorCategory category;
  std::string_view s3_code;

  // Derived, never stored. Two of the seven categories are worth retrying, and
  // they need different strategies (plain backoff vs re-read-and-rebase), which
  // is why they stay separate rather than collapsing into one retriable flag.
  // Storing a second bool is how the two answers drift apart.
  [[nodiscard]] constexpr bool retryable() const {
    return category == ErrorCategory::Transient || category == ErrorCategory::Conflict;
  }
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

// Derived from the code, never stored: see ExtendStatusCodeMetadata::retryable().
bool ExtendStatusDetail::retryable() const { return DefaultRetryableForExtendStatusCode(code_); }

ErrorCategory ExtendStatusDetail::category() const { return CategoryForExtendStatusCode(code_); }

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
  // Permanent rather than guess.
  return ErrorCategory::Unknown;
}

bool DefaultRetryableForExtendStatusCode(ExtendStatusCode code) {
  auto category = CategoryForExtendStatusCode(code);
  return category == ErrorCategory::Transient || category == ErrorCategory::Conflict;
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
// Category answers "who owns this"; the arrow code answers "what failed". A
// Config failure can be either -- unusable `extfs.*` properties never touch the
// network, an S3 403 already did.
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
  // who owns the failure. They are not the same axis and must not be derived
  // from each other: an S3 403 is owned by whoever configured the credentials
  // (Config) but it is still, to arrow and to every caller branching on
  // `IsIOError()`, an IO failure.
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
      // Internal API misuse (null batch, column index out of range, path/group
      // count mismatch) -- our bug, not an end user's parameter. 2042 would
      // make milvus tell a user their query is wrong.
      return milvus::StorageError;  // 2044
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
    case ExtendStatusCode::StorageTransientNetwork:
    case ExtendStatusCode::StorageTransientTimeout:
    case ExtendStatusCode::StorageTransientThrottling:
    case ExtendStatusCode::StorageTransientService:
      return milvus::StorageTransientError;  // 2045
    case ExtendStatusCode::AwsErrorConflict:
    case ExtendStatusCode::AwsErrorPreConditionFailed:
    case ExtendStatusCode::TxnExhaustedRetry:
    case ExtendStatusCode::TxnResolutionFailed:
      // Conflict: another writer won the race. Retriable, but only by a
      // consumer that re-reads state before re-submitting -- replaying the
      // same conditional write fails identically. 2045 is the closest segcore
      // code; the rebase semantics live in the ExtendStatusCode, which the
      // in-process transaction layer reads directly.
      //
      // This supersedes the previous "conservatively permanent" treatment: a
      // spent retry budget belongs to whichever loop spent it, and says
      // nothing about an outer attempt made later, in a different contention
      // window.
      return milvus::StorageTransientError;  // 2045
    case ExtendStatusCode::ManifestCorrupted:
      return milvus::DataFormatBroken;  // 2024
    case ExtendStatusCode::AwsErrorBucketNotFound:
      // Not ObjectNotExist: nothing was lost. The deployment names a bucket
      // that is not there, and milvus has a code that says exactly that.
      return milvus::BucketInvalid;  // 2016
    case ExtendStatusCode::StorageConfigInvalid:
      // The storage location spec -- property map, URI, or both -- is unusable.
      // Non-retriable, and NOT reported as the caller's fault: this producer
      // cannot tell whether the strings came from the operator's config or from
      // a user's external-source definition. An entry point that knows they
      // came from a user re-tags to LOON_SOURCE_INVALID; everything else lands
      // here, because paging an operator for a user typo costs less than
      // telling a user to fix a broken deployment they cannot touch.
      return milvus::ConfigInvalid;  // 2006
    case ExtendStatusCode::AwsErrorNoSuchUpload:
    case ExtendStatusCode::AwsErrorNotFound:
      // The object/bucket is gone: permanent, and fine-grained -- consumers can
      // distinguish "data missing" (stale loadinfo, GC'd file) from a generic
      // storage failure. Never transient/2045: a retry/reroute hits the same
      // shared object store and fails identically.
      return milvus::ObjectNotExist;  // 2017, permanent
    case ExtendStatusCode::AwsErrorAccessDenied:
      // Operator credentials, not the caller's request and not a bug of ours.
      // Non-retriable either way, but it must page whoever owns the config
      // rather than be filed as a generic storage failure.
      return milvus::ConfigInvalid;  // 2006
    case ExtendStatusCode::AwsErrorNonRetryable:
      // The SDK's catch-all for "I judged this non-retryable": no condition
      // identified, so the conservative permanent bucket.
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
  } else {
    // Everything else, INCLUDING arrow's Invalid/TypeError/KeyError, lands on
    // the conservative permanent bucket.
    //
    // This used to map Invalid to DataFormatBroken/2024, which quietly poisoned
    // that signal: of the ~380 unclassified Status::Invalid sites in cpp/src,
    // almost none are corrupt bytes. They are null-pointer preconditions
    // ("Cannot add null column group"), missing configuration
    // ("AZURE_CLIENT_ID environment variable is not set") and caller contract
    // violations ("batch schema does not match writer schema"). Reporting those
    // as "your data is corrupt" sends people to inspect a file when the bug is
    // in code or config -- and once most corruption alerts are false, nobody
    // reads any of them.
    //
    // 2024 now has exactly one source: a producer that actually parsed the
    // bytes and found them wrong. Guessing it from a coarse arrow StatusCode is
    // not a cheaper version of that; the information was gone before this
    // function was reached.
    code = milvus::StorageError;  // 2044
  }
  return {code, status.ToString()};
}

}  // namespace milvus_storage
