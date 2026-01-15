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
    case ExtendStatusCode::AwsErrorNoSuchUpload:
      return "AwsErrorNoSuchUpload";
    case ExtendStatusCode::AwsErrorConflict:
      return "AwsErrorConflict";
    case ExtendStatusCode::AwsErrorPreConditionFailed:
      return "AwsErrorPreConditionFailed";
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
  arrow::StatusCode arrow_code = arrow::StatusCode::IOError;
  return arrow::Status(arrow_code, std::move(message),
                       std::make_shared<ExtendStatusDetail>(code, std::move(extra_info)));
}

}  // namespace milvus_storage
