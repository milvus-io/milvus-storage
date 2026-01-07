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

#include <memory>
#include <vector>

#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::api {

/**
 * @brief Abstract base class for format writers using RAII pattern
 *
 * Format writers handle the actual writing of data to storage files
 * in specific formats (e.g., Parquet).
 */
class ColumnGroupWriter {
  public:
  virtual ~ColumnGroupWriter() = default;

  [[nodiscard]] virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) = 0;
  [[nodiscard]] virtual arrow::Status Flush() = 0;
  [[nodiscard]] virtual arrow::Result<std::vector<ColumnGroupFile>> Close() = 0;

  /**
   * @brief Create a column group writer for a column group
   *
   * @param column_group Column group containing format, path, and metadata
   * @param schema Full schema of the dataset
   * @param properties Write properties
   * @return Unique pointer to the created column group writer
   */
  [[nodiscard]] static arrow::Result<std::unique_ptr<ColumnGroupWriter>> create(
      const std::string& base_path,
      const size_t& column_group_id,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const std::shared_ptr<arrow::Schema>& schema,
      const milvus_storage::api::Properties& properties);

  protected:
  [[nodiscard]] virtual arrow::Status Open() = 0;
};

}  // namespace milvus_storage::api