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

namespace internal::api {

class ColumnGroupReader {
  public:
  virtual ~ColumnGroupReader() = default;
  virtual arrow::Status open() = 0;
  virtual size_t total_number_of_chunks() const = 0;
  virtual size_t total_rows() const = 0;
  virtual arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) = 0;

  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) = 0;

  virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices) = 0;

  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices) = 0;

  virtual arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) = 0;
  virtual arrow::Result<int64_t> get_chunk_rows(int64_t chunk_index) = 0;
};

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
  virtual arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) = 0;
};

/**
 * @brief Factory for creating format-specific chunk readers
 *
 * This factory creates appropriate ChunkReader instances for different
 * file formats. Each reader is responsible for reading one column group only.
 */
class GroupReaderFactory {
  public:
  /**
   * @brief Create a chunk reader for a column group
   *
   * @param schema Schema of the dataset
   * @param column_group Column group containing format, path, and metadata
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @param properties Read properties
   * @return Unique pointer to the created chunk reader
   */
  static arrow::Result<std::unique_ptr<ColumnGroupReader>> create(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
      const std::vector<std::string>& needed_columns,
      const milvus_storage::api::Properties& properties,
      const std::function<std::string(const std::string&)>& key_retriever);

  private:
  GroupReaderFactory() = default;
};

/**
 * @brief Factory for creating format-specific chunk writers
 *
 * This factory creates appropriate ParquetFileWriter instances for column groups.
 * Each writer is responsible for writing one column group only.
 */
class GroupWriterFactory {
  public:
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

  private:
  GroupWriterFactory() = default;
};

}  // namespace internal::api