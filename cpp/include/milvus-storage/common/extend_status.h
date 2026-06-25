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
namespace milvus {
class SegcoreError;
}  // namespace milvus

namespace milvus_storage {
enum class ExtendStatusCode : char {
  // arrow::StatusCode biggest is 45.
  AwsErrorNoSuchUpload = LOON_AWS_ERROR_NO_SUCH_UPLOAD,
  AwsErrorConflict = LOON_AWS_ERROR_CONFLICT,
  AwsErrorPreConditionFailed = LOON_AWS_ERROR_PRECONDITION_FAILED,

  StorageTransientNetwork = LOON_TRANSIENT_NETWORK,
  StorageTransientTimeout = LOON_TRANSIENT_TIMEOUT,
  StorageTransientThrottling = LOON_TRANSIENT_THROTTLING,
  StorageTransientService = LOON_TRANSIENT_SERVICE,

  // Transaction-specific error codes
  TxnExhaustedRetry = LOON_TXN_EXHAUSTED_RETRY,
  TxnResolutionFailed = LOON_TXN_RESOLUTION_FAILED,
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
  bool retryable_ = false;
};

std::optional<ExtendStatusCode> ExtendStatusCodeFromInt(int code);
bool DefaultRetryableForExtendStatusCode(ExtendStatusCode code);

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info);

int ToSegcoreErrorCode(ExtendStatusCode code);

milvus::SegcoreError ToSegcoreError(const arrow::Status& status);

}  // namespace milvus_storage
