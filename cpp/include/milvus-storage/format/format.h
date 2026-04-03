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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/properties.h"

namespace arrow {
class Schema;
namespace fs {
class FileSystem;
}
}  // namespace arrow

namespace milvus_storage {

class FormatReader;

// Format is the abstract factory for a storage format.
// Every supported format (parquet, vortex, lance, iceberg) must implement all
// three operations so that nothing can be missed.
class Format {
  public:
  virtual ~Format() = default;

  // Explore a directory to discover data files of this format.
  [[nodiscard]] virtual arrow::Result<std::vector<api::ColumnGroupFile>> explore(const std::string& explore_dir,
                                                                                 const api::Properties& properties) = 0;

  // Create and open a reader for a specific file.
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<FormatReader>> create_reader(
      const std::shared_ptr<arrow::Schema>& read_schema,
      const api::ColumnGroupFile& file,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever) = 0;

  // Create a writer for the given file path.
  [[nodiscard]] virtual arrow::Result<std::unique_ptr<FormatWriter>> create_writer(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& file_path,
      const std::string& base_path,
      const api::Properties& properties) = 0;

  // Lookup the Format singleton by format name (e.g. "parquet", "vortex").
  static arrow::Result<Format*> get(const std::string& format);
};

// PlainFormat is the base for single-file formats (parquet, vortex) that
// discover files by listing a directory on a filesystem.
// Subclasses only need to implement make_reader() and make_writer().
class PlainFormat : public Format {
  public:
  [[nodiscard]] arrow::Result<std::vector<api::ColumnGroupFile>> explore(const std::string& explore_dir,
                                                                         const api::Properties& properties) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> create_reader(
      const std::shared_ptr<arrow::Schema>& read_schema,
      const api::ColumnGroupFile& file,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever) override;

  [[nodiscard]] arrow::Result<std::unique_ptr<FormatWriter>> create_writer(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& file_path,
      const std::string& base_path,
      const api::Properties& properties) override;

  protected:
  [[nodiscard]] virtual std::shared_ptr<FormatReader> make_reader(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& read_schema,
      const std::string& resolved_path,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever,
      uint64_t file_size,
      uint64_t footer_size) = 0;

  [[nodiscard]] virtual arrow::Result<std::unique_ptr<FormatWriter>> make_writer(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& file_path,
      const api::Properties& properties) = 0;
};

}  // namespace milvus_storage
