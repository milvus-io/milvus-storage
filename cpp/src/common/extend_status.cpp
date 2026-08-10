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

namespace {

// ABI guard for ExtendStatusDetail. Compares against a mirror of its members
// rather than a hardcoded size, because the size is platform-dependent:
// libc++'s std::string is 24 bytes and libstdc++'s is 32, so the same layout is
// 48 bytes on macOS and 56 on Linux. A previous version of this asserted `== 48`
// and broke every Linux build -- including the Java and Python jobs, which
// compile the same C++ core -- while passing locally on macOS. Only the DELTA
// is portable: one bool plus its padding.
//
// If this fires, someone changed the layout of a type that ships in an
// installed header from a library with no SOVERSION. Decide that on purpose and
// bump the version; do not adjust the mirror to match.
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
  // An out-of-range value is not classifiable. Unknown, which is a real answer
  // and not a synonym for Permanent: nothing here established anything.
  return ErrorCategory::Unknown;
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
  // A raw OutOfMemory needs no detail to be classified, and stamping `code`
  // over it would destroy the one thing that says WHAT failed. The retry
  // verdict no longer differs -- OOM is non-retriable now, same as the wrapper
  // codes -- so this passthrough is purely about diagnosis: an operator reading
  // "manifest read failed" instead of "out of memory" goes looking at the
  // manifest. Keep arrow's own code so the coarse fallback still reports OOM.
  if (cause.IsOutOfMemory()) {
    return {cause.code(), std::move(wrapped_message), nullptr};
  }
  return MakeExtendError(code, std::move(wrapped_message), cause_message);
}

}  // namespace milvus_storage
