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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/string_builder.h>

#include "milvus-storage/ffi_internal/ffi_error_code.h"

// from milvus-common repo
#include "common/EasyAssert.h"

namespace milvus_storage {

/// \brief Coarse error fact/hint available to a generic consumer.
enum class ErrorCategory : char {
  Unknown = LOON_ERROR_CATEGORY_UNKNOWN,
  User = LOON_ERROR_CATEGORY_USER,
  /// The observed cause is transient. The operation owner decides whether and
  /// how to retry; a failed stateful writer must be recreated, never reused.
  Retryable = LOON_ERROR_CATEGORY_RETRYABLE,
  /// Business-level coordination is required. Generic retry helpers must not
  /// replay it; a conflict-aware caller may re-read/rebase and submit new work.
  Conflict = LOON_ERROR_CATEGORY_CONFLICT,
  DataFormat = LOON_ERROR_CATEGORY_DATA_FORMAT,
  /// Configuration, missing internal data, unsupported operations, bugs and
  /// other failures that are neither caller-owned nor safely replayable.
  System = LOON_ERROR_CATEGORY_SYSTEM,
};

/// \brief Error codes that can be attached to an arrow::Status and survive to
/// the FFI / segcore boundary.
///
/// Generated from LOON_EXTEND_STATUS_CODE_LIST -- see ffi_error_code.h for the
/// table, including each code's category and its AWS S3 / Aliyun OSS
/// counterpart. Values are the C ABI contract and must never be renumbered.
/// arrow::StatusCode's largest value is 45, so these all start at 50.
///
/// The underlying type is int16_t rather than char. It used to be char, whose
/// range ends at 127 on every platform we build for while the table had already
/// reached 122: five more codes and a new entry would have been narrowed by the
/// enum itself, silently, in a table whose whole point is that a code means one
/// thing everywhere. Nothing reads it as a single byte -- it travels as an int
/// across the C ABI and as an ExtendStatusDetail member in process.
enum class ExtendStatusCode : int16_t {
#define MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY(name, code, symbol, category, s3_code) name = (code),
  LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY)
#undef MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY
};

/// Every code must survive the round trip through its own underlying type. A
/// value the enum cannot represent is not a compile error by itself -- it wraps
/// -- so pin it here, where adding a code to the table is the only thing that
/// can break it.
#define MILVUS_STORAGE_EXTEND_STATUS_RANGE_ASSERT(name, code, symbol, category, s3_code) \
  static_assert(static_cast<int>(ExtendStatusCode::name) == (code),                      \
                "ExtendStatusCode::" #name                                               \
                " does not round-trip through its underlying "                           \
                "type -- widen the enum's underlying type in extend_status.h");
LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_EXTEND_STATUS_RANGE_ASSERT)
#undef MILVUS_STORAGE_EXTEND_STATUS_RANGE_ASSERT

class ExtendStatusDetail : public arrow::StatusDetail {
  public:
  explicit ExtendStatusDetail(ExtendStatusCode code);
  ExtendStatusDetail(ExtendStatusCode code, const char* extra_info);
  ExtendStatusDetail(ExtendStatusCode code, std::string extra_info);

  [[nodiscard]] const char* type_id() const override;

  [[nodiscard]] std::string ToString() const override;

  /// \brief Get the Flight status code.
  [[nodiscard]] ExtendStatusCode code() const;

  /// \brief Get the extra error info
  [[nodiscard]] std::string extra_info() const;

  /// \brief Whether this code carries the transient/retry hint.
  ///
  /// Derived from the code's category, never stored: `code -> category ->
  /// retryable` is the single source of truth, so this can never disagree with
  /// `loon_ffi_is_retryable_errcode`. It does not assert that replaying an
  /// operation is idempotent or that a failed stateful writer can be reused.
  [[nodiscard]] bool retryable() const;

  /// \brief Get the human-readable name of the status code.
  [[nodiscard]] std::string CodeAsString() const;

  /// \brief Set the extra error info
  void set_extra_info(std::string extra_info);

  /// \brief Try to extract a \a ExtendStatusDetail from any Arrow
  /// status.
  ///
  /// \return a \a ExtendStatusDetail if it could be unwrapped, \a
  /// nullptr otherwise
  static std::shared_ptr<ExtendStatusDetail> UnwrapStatus(const arrow::Status& status);

  private:
  ExtendStatusCode code_;
  std::string extra_info_;
};

std::optional<ExtendStatusCode> ExtendStatusCodeFromInt(int code);

/// \brief The generic handling category of an ExtendStatusCode.
ErrorCategory CategoryForExtendStatusCode(ExtendStatusCode code);

/// \brief Whether this code carries the transient/retry hint.
///
/// Conflict is false and replay safety is deliberately out of scope: the
/// operation-aware caller decides what new work, if any, to submit.
bool RetryableForExtendStatusCode(ExtendStatusCode code);

/// \brief The AWS S3 / Aliyun OSS error code this maps to, or "" when the
/// condition has no object-storage counterpart (packed/transaction codes).
std::string_view S3CodeForExtendStatusCode(ExtendStatusCode code);

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info = "");

/// \brief Variadic form, mirroring `arrow::Status::Invalid(a, b, c)`.
///
/// Exists so that an already-correct site that builds its message out of pieces
/// can be given a classification without rewriting how it builds that message.
/// Every unclassified `Status::Invalid(...)` we convert is one fewer failure
/// that reaches segcore as an untyped IOError, so the conversion needs to be
/// cheap enough that nobody skips it.
template <typename... Args>
arrow::Status MakeExtendErrorMsg(ExtendStatusCode code, Args&&... args) {
  return MakeExtendError(code, arrow::util::StringBuilder(std::forward<Args>(args)...));
}

arrow::Status WrapExtendError(ExtendStatusCode code, std::string message, const arrow::Status& cause);

milvus::ErrorCode ToSegcoreErrorCode(ExtendStatusCode code);

/// \brief The single int -> milvus::ErrorCode entry point, covering the whole
/// taxonomy -- internal FFI codes (1-13) as well as ExtendStatusCode values.
///
/// milvus's segcore boundary receives only a bare `int` (`result.err_code`), so
/// it must be able to classify every code through one function. ExtendStatusCodes
/// delegate to ToSegcoreErrorCode(ExtendStatusCode); the internal codes that
/// never carry an ExtendStatusDetail -- allocation failure, user input,
/// deployment config, our own bugs -- get their own landing here. Anything not
/// recognised degrades to the conservative non-retryable StorageError.
milvus::ErrorCode ToSegcoreErrorCode(int ffi_err_code);

milvus::SegcoreError ToSegcoreError(const arrow::Status& status);

}  // namespace milvus_storage
