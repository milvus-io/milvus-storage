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

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/string_builder.h>

#include "milvus-storage/ffi_internal/ffi_error_code.h"

// from milvus-common repo
#include "common/EasyAssert.h"

namespace milvus_storage {

/// \brief Whose problem an error is, and therefore whether retrying can help.
///
/// This is the single classification axis; `retryable` is derived from it
/// (`retryable == (Transient || Conflict)`). See the invariant documented in
/// ffi_error_code.h. Values match the LOON_ERROR_CATEGORY_* constants so the
/// same number crosses the C ABI.
enum class ErrorCategory : char {
  /// The code is outside the documented set -- a producer newer than this
  /// consumer. Must be handled as Permanent: never retry what you cannot
  /// classify. No producer ever emits this.
  Unknown = LOON_ERROR_CATEGORY_UNKNOWN,
  /// The caller's request is wrong. Return it to the caller; do not alert.
  User = LOON_ERROR_CATEGORY_USER,
  /// The deployment is wrong (credentials, endpoint, unusable properties).
  /// Alert an operator; do not blame the caller.
  Config = LOON_ERROR_CATEGORY_CONFIG,
  /// May clear on its own. Retry with normal backoff.
  Transient = LOON_ERROR_CATEGORY_TRANSIENT,
  /// Someone else won a race. Re-read state, rebase, re-submit. Replaying the
  /// same bytes fails identically.
  Conflict = LOON_ERROR_CATEGORY_CONFLICT,
  /// The named object is not there. Re-read the metadata, THEN decide -- this
  /// layer deliberately does not answer the retry question, because it cannot
  /// tell a GC race from real data loss and will not invent an answer.
  /// Only a producer holding a definitive not-found from the store may say
  /// this; it is never inferred from an unclassified failure.
  Missing = LOON_ERROR_CATEGORY_MISSING,
  /// The bytes are wrong. Act on the DATA: quarantine, re-fetch from a replica,
  /// rebuild. Only a producer that actually PARSED the bytes and found them
  /// malformed may say this -- see the CoarseFallbackNeverClaimsCorruption test
  /// for the machine-checked form of that rule.
  Corrupted = LOON_ERROR_CATEGORY_CORRUPTED,
  /// Our bug. Alert a developer; never retry.
  Permanent = LOON_ERROR_CATEGORY_PERMANENT,
};

/// \brief Error codes that can be attached to an arrow::Status and survive to
/// the FFI / segcore boundary.
///
/// Generated from LOON_EXTEND_STATUS_CODE_LIST -- see ffi_error_code.h for the
/// table, including each code's category and its AWS S3 / Aliyun OSS
/// counterpart. Values are the C ABI contract and must never be renumbered.
/// arrow::StatusCode's largest value is 45, so these all start at 50.
enum class ExtendStatusCode : char {
#define MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY(name, code, symbol, category, s3_code) name = (code),
  LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY)
#undef MILVUS_STORAGE_EXTEND_STATUS_ENUM_ENTRY
};

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

  /// \brief Whether retrying can help. Derived from the code, not stored.
  [[nodiscard]] bool retryable() const;

  /// \brief Who owns this failure, and what the consumer should do about it.
  /// Derived from the code, never stored.
  [[nodiscard]] ErrorCategory category() const;

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

/// \brief The category of an ExtendStatusCode: who owns the failure.
ErrorCategory CategoryForExtendStatusCode(ExtendStatusCode code);

/// \brief Whether retrying can help. Derived state, never stored per code: true
/// iff the category is Transient or Conflict. Kept as a named helper
/// because it is the question most consumers actually ask.
bool DefaultRetryableForExtendStatusCode(ExtendStatusCode code);

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

milvus::SegcoreError ToSegcoreError(const arrow::Status& status);

}  // namespace milvus_storage
