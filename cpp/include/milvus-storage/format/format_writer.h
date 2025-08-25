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
#include <arrow/ipc/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/writer.h"

// Forward declarations
namespace milvus_storage {
class ParquetFileWriter;
}

namespace milvus_storage::api {

/**
 * @brief Abstract interface for format-specific writers
 *
 * This interface abstracts the underlying writing mechanism for different
 * file formats (PARQUET, etc.). Each format writer handles a single
 * column group and only needs to manage writing data for that specific group.
 */
class FormatWriter {
  public:
  virtual ~FormatWriter() = default;

  /**
   * @brief Initialize the format writer with a single column group
   *
   * @param column_group The column group to write
   * @param custom_metadata Custom metadata to include
   * @return Status indicating success or error condition
   */
  virtual arrow::Status initialize(std::shared_ptr<ColumnGroup> column_group,
                                   const std::map<std::string, std::string>& custom_metadata) = 0;

  /**
   * @brief Write a record batch for this column group
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
   * @brief Close the writer and finalize the file
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
   * @brief Get statistics about rows and bytes written for this column group
   *
   * @return WriteStats structure with current statistics
   */
  virtual Writer::WriteStats get_stats() const = 0;
};

/**
 * @brief Factory for creating format-specific writers
 *
 * This factory creates appropriate FormatWriter instances for a single
 * column group. Each writer is responsible for writing one column group only.
 */
class FormatWriterFactory {
  public:
  /**
   * @brief Create a format writer for a single column group
   *
   * @param format The file format to create a writer for
   * @param fs Filesystem interface
   * @param column_group The column group this writer will handle
   * @param schema Arrow schema for the columns in this group
   * @param properties Write properties
   * @return Unique pointer to the created format writer
   */
  static std::unique_ptr<FormatWriter> create_writer(FileFormat format,
                                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                                     std::shared_ptr<ColumnGroup> column_group,
                                                     std::shared_ptr<arrow::Schema> schema,
                                                     const WriteProperties& properties);

  private:
  FormatWriterFactory() = default;
};

/**
 * @brief Parquet format writer implementation
 *
 * Implements the FormatWriter interface for Parquet format.
 * Each instance handles writing a single column group to a single file.
 */
class ParquetFormatWriter : public FormatWriter {
  public:
  ParquetFormatWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::shared_ptr<ColumnGroup> column_group,
                      std::shared_ptr<arrow::Schema> schema,
                      const WriteProperties& properties);

  ~ParquetFormatWriter() override;

  arrow::Status initialize(std::shared_ptr<ColumnGroup> column_group,
                           const std::map<std::string, std::string>& custom_metadata) override;

  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  arrow::Status flush() override;

  arrow::Status close() override;

  arrow::Status add_metadata(const std::string& key, const std::string& value) override;

  Writer::WriteStats get_stats() const override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<ColumnGroup> column_group_;
  std::shared_ptr<arrow::Schema> schema_;
  WriteProperties properties_;

  std::unique_ptr<milvus_storage::ParquetFileWriter> file_writer_;
  std::map<std::string, std::string> custom_metadata_;
  Writer::WriteStats stats_;
  bool initialized_;
  bool finished_;

  // Buffering for memory management (similar to packed implementation)
  std::vector<std::shared_ptr<arrow::RecordBatch>> buffered_batches_;
  std::vector<size_t> buffered_memory_usage_;

  // Statistics tracking (similar to packed ColumnGroupWriter)
  int flushed_batches_;
  int flushed_count_;
  int64_t flushed_rows_;
};

}  // namespace milvus_storage::api