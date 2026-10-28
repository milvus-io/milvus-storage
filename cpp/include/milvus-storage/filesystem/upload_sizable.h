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

/// \brief Interface for output streams with a specified part size or fixed total size
class UploadSizable {
  public:
  virtual ~UploadSizable() = default;

  /// \brief Open an output stream with a specified upload size for multipart uploads
  /// \param path The file path
  /// \param metadata Optional metadata
  /// \param upload_size The size threshold for triggering multipart uploads
  /// \return Output stream for writing
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, int64_t upload_size) = 0;

  /// \brief Open an output stream whose written length must match total_size
  /// \param path The file path
  /// \param metadata Optional metadata
  /// \param total_size The exact number of bytes to write
  /// \return Output stream for writing, or NotImplemented if unsupported
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithFixedSize(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, int64_t total_size) = 0;
};

}  // namespace milvus_storage
