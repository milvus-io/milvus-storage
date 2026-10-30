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
#include <string>

#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

/// \brief Interface for filesystems that support parallel streaming uploads
class UploadParallel {
  public:
  virtual ~UploadParallel() = default;

  /// \brief Open an upload stream without requiring the final object size
  /// Write supplies successive bytes; Close ends the input and publishes the object.
  /// Calls on a stream must be serialized by the caller.
  /// \param path The file path
  /// \param metadata Optional metadata
  /// \return Output stream for writing, or NotImplemented if unsupported
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenParallelUploadOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) = 0;
};

}  // namespace milvus_storage
