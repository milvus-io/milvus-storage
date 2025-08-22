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
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"

namespace milvus_storage::api {

/**
 * @brief Abstract interface for format-specific readers
 *
 * This interface abstracts the underlying reading mechanism for different
 * file formats (PARQUET, etc.). Each format reader handles a single
 * column group and only needs to manage reading data from that specific group.
 */
class FormatReader {
  public:
  virtual ~FormatReader() = default;

  /**
   * @brief Initialize the format reader with a single column group
   *
   * @param column_group The column group to read from
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @return Status indicating success or error condition
   */
  virtual arrow::Status initialize(std::shared_ptr<ColumnGroup> column_group,
                                   const std::vector<std::string>& needed_columns) = 0;

  /**
   * @brief Get a record batch reader for scanning data from this column group
   *
   * @param predicate Filter expression string for row-level filtering (empty = no filtering)
   * @param batch_size Maximum number of rows per record batch
   * @param buffer_size Target buffer size in bytes for internal I/O buffering
   * @return Result containing a RecordBatchReader for sequential data access, or error status
   */
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate = "", int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024) = 0;

  /**
   * @brief Get a chunk reader for this column group
   *
   * @return Result containing a ChunkReader for this column group, or error status
   */
  virtual arrow::Result<std::shared_ptr<ChunkReader>> get_chunk_reader() = 0;

  /**
   * @brief Extract specific rows by their indices within this column group
   *
   * @param row_indices Vector of row indices to extract (local to this column group)
   * @param parallelism Number of threads to use for parallel reading
   * @return Result containing RecordBatch with the requested rows, or error status
   */
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                                  int64_t parallelism = 1) = 0;
};

/**
 * @brief Factory for creating format-specific readers
 *
 * This factory creates appropriate FormatReader instances for a single
 * column group. Each reader is responsible for reading one column group only.
 */
class FormatReaderFactory {
  public:
  /**
   * @brief Create a format reader for a single column group
   *
   * @param format The file format to create a reader for
   * @param fs Filesystem interface
   * @param column_group The column group this reader will handle
   * @param schema Arrow schema for the columns in this group
   * @param properties Read properties
   * @return Unique pointer to the created format reader
   */
  static std::unique_ptr<FormatReader> create_reader(FileFormat format,
                                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                                     std::shared_ptr<ColumnGroup> column_group,
                                                     std::shared_ptr<arrow::Schema> schema,
                                                     const ReadProperties& properties);

  private:
  FormatReaderFactory() = default;
};

/**
 * @brief Parquet format reader implementation
 *
 * Implements the FormatReader interface for Parquet format.
 * Each instance handles reading a single column group from a single file.
 */
class ParquetFormatReader : public FormatReader {
  public:
  ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::shared_ptr<ColumnGroup> column_group,
                      std::shared_ptr<arrow::Schema> schema,
                      const ReadProperties& properties);

  ~ParquetFormatReader() override = default;

  arrow::Status initialize(std::shared_ptr<ColumnGroup> column_group,
                           const std::vector<std::string>& needed_columns) override;

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate = "", int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024) override;

  arrow::Result<std::shared_ptr<ChunkReader>> get_chunk_reader() override;

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                          int64_t parallelism = 1) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<ColumnGroup> column_group_;
  std::shared_ptr<arrow::Schema> schema_;
  ReadProperties properties_;

  std::vector<std::string> needed_columns_;
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader_;
  bool initialized_;
};

}  // namespace milvus_storage::api