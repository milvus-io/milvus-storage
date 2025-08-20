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

#include "milvus-storage/manifest.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::api {

/**
 * @brief Abstract interface for format-specific writers
 *
 * This interface abstracts the underlying writing mechanism for different
 * file formats (PARQUET, BINARY, etc.). Each format can have its own
 * implementation with format-specific optimizations.
 */
class FormatWriter {
  public:
  virtual ~FormatWriter() = default;

  /**
   * @brief Initialize the format writer with column groups and paths
   *
   * @param column_groups Vector of column groups to write
   * @param custom_metadata Custom metadata to include
   * @return Status indicating success or error condition
   */
  virtual arrow::Status initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                   const std::map<std::string, std::string>& custom_metadata) = 0;

  /**
   * @brief Write a record batch
   *
   * @param batch Arrow RecordBatch containing the data to write
   * @return Status indicating success or error condition
   */
  virtual arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

  /**
   * @brief Flush buffered data
   *
   * @return Status indicating success or error condition
   */
  virtual arrow::Status flush() = 0;

  /**
   * @brief Close the writer and finalize files
   *
   * @return Status indicating success or error condition
   */
  virtual arrow::Status close() = 0;

  /**
   * @brief Add custom metadata
   *
   * @param key Metadata key
   * @param value Metadata value
   * @return Status indicating success or error condition
   */
  virtual arrow::Status add_metadata(const std::string& key, const std::string& value) = 0;

  /**
   * @brief Get statistics about rows and bytes written
   *
   * @return WriteStats structure with current statistics
   */
  virtual Writer::WriteStats get_stats() const = 0;
};

/**
 * @brief Factory for creating format-specific writers
 *
 * This factory creates appropriate FormatWriter instances based on the
 * file format specified in the column groups. It encapsulates the
 * creation logic and allows for easy extension to new formats.
 */
class FormatWriterFactory {
  public:
  /**
   * @brief Create a format writer based on the file format
   *
   * @param format The file format to create a writer for
   * @param fs Filesystem interface
   * @param base_path Base path for writing files
   * @param schema Arrow schema
   * @param properties Write properties
   * @return Unique pointer to the created format writer
   */
  static std::unique_ptr<FormatWriter> create_writer(FileFormat format,
                                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                                     const std::string& base_path,
                                                     std::shared_ptr<arrow::Schema> schema,
                                                     const WriteProperties& properties);

  private:
  FormatWriterFactory() = default;
};

/**
 * @brief Parquet format writer implementation
 *
 * Implements the FormatWriter interface for Parquet format using
 * the existing PackedRecordBatchWriter.
 */
class ParquetFormatWriter : public FormatWriter {
  public:
  ParquetFormatWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::string base_path,
                      std::shared_ptr<arrow::Schema> schema,
                      const WriteProperties& properties);

  ~ParquetFormatWriter() override = default;

  arrow::Status initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                           const std::map<std::string, std::string>& custom_metadata) override;

  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  arrow::Status flush() override;

  arrow::Status close() override;

  arrow::Status add_metadata(const std::string& key, const std::string& value) override;

  Writer::WriteStats get_stats() const override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  std::shared_ptr<arrow::Schema> schema_;
  WriteProperties properties_;

  std::unique_ptr<milvus_storage::PackedRecordBatchWriter> packed_writer_;
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  Writer::WriteStats stats_;
  bool initialized_;
};

}  // namespace milvus_storage::api