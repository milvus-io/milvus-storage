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

#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

/// \brief Interface for filesystems that support conditional writes
///
/// Conditional writes prevent overwriting existing files by adding
/// provider-specific metadata headers (e.g., If-None-Match for AWS).
class UploadConditional {
  public:
  virtual ~UploadConditional() = default;

  /// \brief Open an output stream for conditional write (fail if file exists)
  /// \param path The file path
  /// \param metadata Optional metadata (will be modified with conditional headers)
  /// \return Output stream for writing, or error if file already exists
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) = 0;
};

}  // namespace milvus_storage
