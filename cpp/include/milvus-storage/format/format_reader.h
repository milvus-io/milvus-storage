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

#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/packed/column_group.h"

namespace milvus_storage {
class PackedRecordBatchReader;
}

namespace milvus_storage::api {

/**
 * @brief Abstract interface for format-specific readers
 *
 * This interface abstracts the underlying reading mechanism for different
 * file formats (PARQUET, BINARY, etc.). Each format can have its own
 * implementation with format-specific optimizations.
 */
class FormatReader {
  public:
  virtual ~FormatReader() = default;

  /**
   * @brief Initialize the format reader with column groups
   *
   * @param column_groups Vector of column groups to read from
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @return Status indicating success or error condition
   */
  virtual arrow::Status initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                   const std::vector<std::string>& needed_columns) = 0;

  /**
   * @brief Get a record batch reader for scanning data
   *
   * @param predicate Filter expression string for row-level filtering (empty = no filtering)
   * @param batch_size Maximum number of rows per record batch
   * @param buffer_size Target buffer size in bytes for internal I/O buffering
   * @return Result containing a RecordBatchReader for sequential data access, or error status
   */
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate = "", int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024) = 0;

  /**
   * @brief Get a chunk reader for a specific column group
   *
   * @param column_group_id ID of the column group to read from
   * @return Result containing a ChunkReader for the specified column group, or error status
   */
  virtual arrow::Result<std::shared_ptr<ChunkReader>> get_chunk_reader(int64_t column_group_id) = 0;

  /**
   * @brief Extract specific rows by their global indices
   *
   * @param row_indices Vector of global row indices to extract
   * @param parallelism Number of threads to use for parallel reading
   * @return Result containing RecordBatch with the requested rows, or error status
   */
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                                  int64_t parallelism = 1) = 0;
};

/**
 * @brief Factory for creating format-specific readers
 *
 * This factory creates appropriate FormatReader instances based on the
 * file format specified in the column groups. It encapsulates the
 * creation logic and allows for easy extension to new formats.
 */
class FormatReaderFactory {
  public:
  /**
   * @brief Create a format reader based on the file format
   *
   * @param format The file format to create a reader for
   * @param fs Filesystem interface
   * @param manifest Dataset manifest
   * @param schema Arrow schema
   * @param properties Read properties
   * @return Unique pointer to the created format reader
   */
  static std::unique_ptr<FormatReader> create_reader(FileFormat format,
                                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                                     std::shared_ptr<Manifest> manifest,
                                                     std::shared_ptr<arrow::Schema> schema,
                                                     const ReadProperties& properties);

  private:
  FormatReaderFactory() = default;
};

/**
 * @brief Parquet format reader implementation
 *
 * Implements the FormatReader interface for Parquet format using
 * the existing PackedRecordBatchReader.
 */
class ParquetFormatReader : public FormatReader {
  public:
  ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::shared_ptr<Manifest> manifest,
                      std::shared_ptr<arrow::Schema> schema,
                      const ReadProperties& properties);

  ~ParquetFormatReader() override = default;

  arrow::Status initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                           const std::vector<std::string>& needed_columns) override;

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate = "", int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024) override;

  arrow::Result<std::shared_ptr<ChunkReader>> get_chunk_reader(int64_t column_group_id) override;

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                          int64_t parallelism = 1) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<arrow::Schema> schema_;
  ReadProperties properties_;

  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  std::vector<std::string> needed_columns_;
  std::unique_ptr<milvus_storage::PackedRecordBatchReader> packed_reader_;
  bool initialized_;
};

/**
 * @brief Binary format reader implementation
 *
 * Implements the FormatReader interface for Binary format using
 * Arrow IPC format for efficient vector data reading.
 */
class BinaryFormatReader : public FormatReader {
  public:
  BinaryFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     std::shared_ptr<Manifest> manifest,
                     std::shared_ptr<arrow::Schema> schema,
                     const ReadProperties& properties);

  ~BinaryFormatReader() override = default;

  arrow::Status initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                           const std::vector<std::string>& needed_columns) override;

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate = "", int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024) override;

  arrow::Result<std::shared_ptr<ChunkReader>> get_chunk_reader(int64_t column_group_id) override;

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                          int64_t parallelism = 1) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<arrow::Schema> schema_;
  ReadProperties properties_;

  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  std::vector<std::string> needed_columns_;
  std::unordered_map<int64_t, std::shared_ptr<arrow::ipc::RecordBatchFileReader>> readers_;
  std::unordered_map<int64_t, std::shared_ptr<arrow::io::RandomAccessFile>> input_streams_;
  bool initialized_;
};

/**
 * @brief Custom record batch reader for binary format
 *
 * Combines data from multiple binary column groups into unified record batches.
 */
class BinaryRecordBatchReader : public arrow::RecordBatchReader {
  public:
  BinaryRecordBatchReader(
      const std::unordered_map<int64_t, std::shared_ptr<arrow::ipc::RecordBatchFileReader>>& readers,
      const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
      std::shared_ptr<arrow::Schema> schema,
      const std::vector<std::string>& needed_columns);

  ~BinaryRecordBatchReader() override = default;

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  arrow::Status Close() override;

  private:
  std::unordered_map<int64_t, std::shared_ptr<arrow::ipc::RecordBatchFileReader>> readers_;
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> needed_columns_;
  int64_t current_batch_;
  int64_t total_batches_;
};

}  // namespace milvus_storage::api