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

#include "milvus-storage/ffi_internal/ffi_error_code.h"

// from milvus-common repo
#include "common/EasyAssert.h"

namespace milvus_storage {

/// \brief Whose problem an error is, and therefore whether retrying can help.
///
/// This is the single classification axis; `retryable` is derived from it
/// (`retryable == (category == Transient)`). See the invariant documented in
/// ffi_error_code.h. Values match the LOON_ERROR_CATEGORY_* constants so the
/// same number crosses the C ABI.
enum class ErrorCategory : char {
  /// The code is outside the documented set. Consumers must treat it as
  /// Permanent (never retry something we cannot classify).
  Unknown = LOON_ERROR_CATEGORY_UNKNOWN,
  /// The caller's request or configuration is wrong. Never retriable; report
  /// it back to whoever made the request.
  User = LOON_ERROR_CATEGORY_USER,
  /// A system condition that may clear on its own. Retriable.
  Transient = LOON_ERROR_CATEGORY_TRANSIENT,
  /// A system condition that will not clear by itself (missing data, bad
  /// credentials, corruption, our own bug). Never retriable.
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

  /// \brief Who owns this failure: the caller (User) or us (Transient /
  /// Permanent). Derived from the code, not stored.
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

/// \brief Whether retrying can help. Derived state: true iff the category is
/// Transient. Kept as a named helper because it is what most consumers ask.
bool DefaultRetryableForExtendStatusCode(ExtendStatusCode code);

/// \brief The AWS S3 / Aliyun OSS error code this maps to, or "" when the
/// condition has no object-storage counterpart (packed/transaction codes).
std::string_view S3CodeForExtendStatusCode(ExtendStatusCode code);

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info = "");

arrow::Status WrapExtendError(ExtendStatusCode code, std::string message, const arrow::Status& cause);

milvus::ErrorCode ToSegcoreErrorCode(ExtendStatusCode code);

milvus::SegcoreError ToSegcoreError(const arrow::Status& status);

}  // namespace milvus_storage
