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

#include <arrow/filesystem/filesystem.h>

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
  virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) = 0;
  virtual arrow::Status Flush() = 0;
  virtual arrow::Status Close() = 0;

  /**
   * @brief Create a chunk writer for a column group
   *
   * @param column_group Column group containing format, path, and metadata
   * @param schema Full schema of the dataset
   * @param fs Filesystem interface
   * @param properties Write properties
   * @return Unique pointer to the created chunk writer
   */
  [[nodiscard]] static arrow::Result<std::unique_ptr<ColumnGroupWriter>> create(
      std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
      std::shared_ptr<arrow::Schema> schema,
      const milvus_storage::api::Properties& properties);
};

}  // namespace milvus_storage::api