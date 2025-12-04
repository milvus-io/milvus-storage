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

namespace milvus_storage {
enum class ExtendStatusCode : char {
  // arrow::StatusCode biggest is 45
  NoSuchUpload = 101,
};

class ExtendStatusDetail : public arrow::StatusDetail {
  public:
  explicit ExtendStatusDetail(ExtendStatusCode code);
  explicit ExtendStatusDetail(ExtendStatusCode code, std::string extra_info);

  [[nodiscard]] const char* type_id() const override;

  [[nodiscard]] std::string ToString() const override;

  /// \brief Get the Flight status code.
  [[nodiscard]] ExtendStatusCode code() const;

  /// \brief Get the extra error info
  [[nodiscard]] std::string extra_info() const;

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

arrow::Status MakeExtendError(ExtendStatusCode code, std::string message, std::string extra_info);

}  // namespace milvus_storage