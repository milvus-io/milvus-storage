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

#include "common/EasyAssert.h"

namespace milvus_storage {
namespace {

const char* kErrorDetailTypeId = "milvus_storage::ExtendStatusDetail";

struct ExtendStatusCodeMetadata {
  ExtendStatusCode code;
  std::string_view name;
  bool retryable;
};

constexpr ExtendStatusCodeMetadata kExtendStatusCodeMetadata[] = {
    {ExtendStatusCode::AwsErrorNoSuchUpload, "AwsErrorNoSuchUpload", true},
    {ExtendStatusCode::AwsErrorConflict, "AwsErrorConflict", false},
    {ExtendStatusCode::AwsErrorPreConditionFailed, "AwsErrorPreConditionFailed", false},
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
  arrow::StatusCode arrow_code = arrow::StatusCode::IOError;
  return {arrow_code, std::move(message), std::make_shared<ExtendStatusDetail>(code, std::move(extra_info))};
}

int ToSegcoreErrorCode(ExtendStatusCode code) {
  return DefaultRetryableForExtendStatusCode(code) ? static_cast<int>(milvus::StorageTransientError)
                                                   : static_cast<int>(milvus::StorageError);
}

milvus::SegcoreError ToSegcoreError(const arrow::Status& status) {
  if (status.ok()) {
    return milvus::SegcoreError::success();
  }

  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  if (detail) {
    return {static_cast<milvus::ErrorCode>(ToSegcoreErrorCode(detail->code())), status.ToString()};
  }
  return {milvus::StorageError, status.ToString()};
}

}  // namespace milvus_storage
