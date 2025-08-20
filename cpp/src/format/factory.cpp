// Copyright 2023 Zilliz
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

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/format_writer.h"

namespace milvus_storage::api {

// ==================== FormatWriterFactory Implementation ====================

std::unique_ptr<FormatWriter> FormatWriterFactory::create_writer(FileFormat format,
                                                                 std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                 const std::string& base_path,
                                                                 std::shared_ptr<arrow::Schema> schema,
                                                                 const WriteProperties& properties) {
  switch (format) {
    case FileFormat::PARQUET:
      return std::make_unique<ParquetFormatWriter>(std::move(fs), base_path, std::move(schema), properties);

    case FileFormat::BINARY:
      return std::make_unique<BinaryFormatWriter>(std::move(fs), base_path, std::move(schema), properties);

    case FileFormat::VORTEX:
    case FileFormat::LANCE:
      // TODO: Implement other format writers when needed
      throw std::runtime_error("Format not yet supported: " + std::to_string(static_cast<int>(format)));

    default:
      throw std::runtime_error("Unknown file format: " + std::to_string(static_cast<int>(format)));
  }
}

// ==================== FormatReaderFactory Implementation ====================

std::unique_ptr<FormatReader> FormatReaderFactory::create_reader(FileFormat format,
                                                                 std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                 std::shared_ptr<Manifest> manifest,
                                                                 std::shared_ptr<arrow::Schema> schema,
                                                                 const ReadProperties& properties) {
  switch (format) {
    case FileFormat::PARQUET:
      return std::make_unique<ParquetFormatReader>(std::move(fs), std::move(manifest), std::move(schema), properties);

    case FileFormat::BINARY:
      return std::make_unique<BinaryFormatReader>(std::move(fs), std::move(manifest), std::move(schema), properties);

    case FileFormat::VORTEX:
    case FileFormat::LANCE:
      // TODO: Implement other format writers when needed
      throw std::runtime_error("Format not yet supported: " + std::to_string(static_cast<int>(format)));

    default:
      throw std::runtime_error("Unknown file format: " + std::to_string(static_cast<int>(format)));
  }
}

}  // namespace milvus_storage::api