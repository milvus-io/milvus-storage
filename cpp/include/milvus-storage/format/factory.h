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
#include <string>
#include <map>
#include <arrow/filesystem/filesystem.h>
#include <arrow/type.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/reader.h"

namespace internal::api {

/**
 * @brief Abstract base class for format writers using RAII pattern
 *
 * Format writers handle the actual writing of data to storage files
 * in specific formats (e.g., Parquet). They use RAII pattern for
 * automatic resource management - initialization happens in constructor,
 * cleanup happens in destructor.
 */
class FormatWriter {
  public:
  virtual ~FormatWriter() = default;

  virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) = 0;
  virtual arrow::Status Flush() = 0;
  virtual int64_t count() const = 0;
  virtual int64_t bytes_written() const = 0;
  virtual int64_t num_chunks() const = 0;
  virtual arrow::Status Close() = 0;

  // Metadata management methods
  virtual arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) = 0;
  virtual arrow::Status AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) = 0;
};

/**
 * @brief Factory for creating format-specific chunk readers
 *
 * This factory creates appropriate ChunkReader instances for different
 * file formats. Each reader is responsible for reading one column group only.
 */
class ChunkReaderFactory {
  public:
  /**
   * @brief Create a chunk reader for a column group
   *
   * @param column_group Column group containing format, path, and metadata
   * @param fs Filesystem interface
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @param properties Read properties
   * @return Unique pointer to the created chunk reader
   */
  static std::unique_ptr<milvus_storage::api::ChunkReader> create_reader(
      std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const std::vector<std::string>& needed_columns,
      const milvus_storage::api::ReadProperties& properties);

  private:
  ChunkReaderFactory() = default;
};

/**
 * @brief Factory for creating format-specific chunk writers
 *
 * This factory creates appropriate ParquetFileWriter instances for column groups.
 * Each writer is responsible for writing one column group only.
 */
class ChunkWriterFactory {
  public:
  /**
   * @brief Create a chunk writer for a column group
   *
   * @param column_group Column group containing format, path, and metadata
   * @param schema Full schema of the dataset
   * @param fs Filesystem interface
   * @param storage_config Storage configuration
   * @param custom_metadata Custom metadata to include in the writer
   * @return Unique pointer to the created chunk writer
   */
  static std::unique_ptr<internal::api::FormatWriter> create_writer(
      std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const milvus_storage::StorageConfig& storage_config,
      const std::map<std::string, std::string>& custom_metadata);

  private:
  ChunkWriterFactory() = default;
};

}  // namespace internal::api