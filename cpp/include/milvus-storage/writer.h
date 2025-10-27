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
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage::api {
/**
 * @brief Abstract base class for column grouping policies
 *
 * Column grouping policies determine how columns are organized into column groups
 * for storage. Different policies can optimize for different access patterns:
 * - All columns together (simple policy)
 * - Related columns together (semantic policy)
 * - Frequently accessed columns together (access pattern policy)
 * - Size-based grouping (balanced policy)
 */
class ColumnGroupPolicy {
  public:
  /**
   * @brief Constructs a column group policy for the given schema and default format
   *
   * @param schema Arrow schema defining the columns to be grouped
   * @param default_format Default file format for column groups
   */
  ColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, const std::string& default_format = LOON_FORMAT_PARQUET);

  /**
   * @brief Virtual destructor
   */
  virtual ~ColumnGroupPolicy() = default;

  /**
   * @brief Indicates whether this policy requires data sampling
   *
   * Some policies may need to analyze actual data to make optimal grouping decisions.
   *
   * @return true if the policy needs data samples, false otherwise
   */
  [[nodiscard]] virtual bool requires_sample() const = 0;

  /**
   * @brief Provides a data sample to the policy for analysis
   *
   * Only called if requires_sample() returns true. The policy can analyze
   * the data to make informed grouping decisions.
   *
   * @param batch Sample data batch
   * @return Status indicating success or error condition
   */
  virtual arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

  /**
   * @brief Returns the column groups determined by this policy
   *
   * @return Vector of column groups, each containing metadata about grouped columns
   */
  [[nodiscard]] virtual std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const = 0;

  /**
   * @brief Factory function to create a ColumnGroupPolicy from properties
   *
   * This function reads the "writer.policy" property to determine which
   * concrete ColumnGroupPolicy implementation to instantiate. It uses other
   * properties as needed to configure the policy.
   */
  static arrow::Result<std::unique_ptr<ColumnGroupPolicy>> create_column_group_policy(
      const Properties& properties_map, const std::shared_ptr<arrow::Schema>& schema);

  protected:
  std::shared_ptr<arrow::Schema> schema_;  ///< Schema for the columns being grouped
  std::string default_format_;             ///< Default file format for column groups
};

/**
 * @brief Simple column group policy that puts all columns in a single group
 *
 * This policy is suitable for datasets where all columns are typically accessed together,
 * or for small datasets where the overhead of multiple files isn't justified.
 */
class SingleColumnGroupPolicy : public ColumnGroupPolicy {
  public:
  explicit SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                   const std::string& default_format = LOON_FORMAT_PARQUET)
      : ColumnGroupPolicy(std::move(schema), default_format) {}

  [[nodiscard]] bool requires_sample() const override;

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const override;
};

/**
 * @brief Column group policy that creates column groups based on the schema
 *
 * This policy creates column groups based on the schema.
 */
class SchemaBasedColumnGroupPolicy : public ColumnGroupPolicy {
  public:
  explicit SchemaBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                        const std::vector<std::string>& column_name_patterns,
                                        const std::string& default_format = LOON_FORMAT_PARQUET)
      : ColumnGroupPolicy(std::move(schema), default_format), column_name_patterns_(column_name_patterns) {}

  [[nodiscard]] bool requires_sample() const override;

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const override;

  private:
  std::vector<std::string> column_name_patterns_;
};

/**
 * @brief Column group policy that creates column groups based on the size of the columns
 *
 * This policy creates column groups based on the size of the columns.
 */
class SizeBasedColumnGroupPolicy : public ColumnGroupPolicy {
  public:
  explicit SizeBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                      int64_t max_avg_column_size,
                                      int64_t max_columns_in_group,
                                      const std::string& default_format = LOON_FORMAT_PARQUET)
      : ColumnGroupPolicy(std::move(schema), default_format),
        max_avg_column_size_(max_avg_column_size),
        max_columns_in_group_(max_columns_in_group) {}

  [[nodiscard]] bool requires_sample() const override { return true; }

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const override;

  private:
  int64_t max_avg_column_size_;
  int64_t max_columns_in_group_;
  mutable std::vector<int64_t> column_sizes_;  // Cached column sizes from sampling
};

/**
 * @brief High-level writer interface for milvus storage data
 *
 * The Writer class provides a unified interface for writing data to milvus
 * storage datasets using manifest-based metadata. It supports efficient batch
 * writing, column grouping policies, compression, encryption, and automatic
 * manifest generation.
 *
 * The writer coordinates multiple column group writers to achieve optimal
 * storage layout and query performance while maintaining data consistency
 * and integrity.
 */
class Writer {
  public:
  /**
   * @brief Factory function to create a Writer instance
   *
   * Creates a concrete Writer implementation that can be used to write data to
   * milvus storage datasets. This function provides a clean interface for creating
   * writers without exposing the concrete implementation details.
   *
   * @param base_path Base directory path where column group files will be written
   * @param schema Arrow schema defining the logical structure of the data
   * @param column_group_policy Policy for organizing columns into groups
   * @param properties Write configuration properties including compression and encryption
   * @return Unique pointer to a Writer instance
   *
   * @example
   * @code
   * auto schema = arrow::schema({
   *   arrow::field("id", arrow::int64()),
   *   arrow::field("name", arrow::utf8()),
   *   arrow::field("value", arrow::float64())
   * });
   *
   * auto properties = WritePropertiesBuilder()
   *                     .with_compression(CompressionType::ZSTD)
   *                     .with_max_row_group_size(100000)
   *                     .build();
   *
   * auto policy = std::make_unique<SingleColumnGroupPolicy>(schema);
   * auto writer = Writer::create("/path/to/dataset", schema, std::move(policy), properties);
   *
   * // Use the writer
   * writer->write(batch);
   * auto manifest = writer->close().ValueOrDie();
   * @endcode
   */
  static std::unique_ptr<Writer> create(std::string base_path,
                                        std::shared_ptr<arrow::Schema> schema,
                                        std::unique_ptr<ColumnGroupPolicy> column_group_policy,
                                        const Properties& properties = {});

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of column group writers and temporary resources.
   * If close() hasn't been called, this will attempt to clean up gracefully.
   */
  virtual ~Writer() = default;

  /**
   * @brief Returns the schema used by this writer
   *
   * @return Shared pointer to the Arrow schema
   */
  virtual std::shared_ptr<arrow::Schema> schema() const = 0;

  /**
   * @brief Writes a record batch to the dataset
   *
   * Distributes the batch data across appropriate column groups based on the
   * configured column group policy. Data is buffered and written to storage
   * when buffers reach their limits or flush() is called.
   *
   * @param batch Arrow RecordBatch containing the data to write
   * @return Status indicating success or error condition
   *
   * @note The batch schema must be compatible with the writer's schema.
   *       All batches written to the same writer should have consistent schemas.
   */
  virtual arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

  /**
   * @brief Forces buffered data to be written to storage
   *
   * Flushes all pending data in column group writers to their respective
   * storage files. This ensures data durability but may impact performance
   * if called too frequently.
   *
   * @return Status indicating success or error condition
   *
   * @note This does not close the writers; additional batches can still be written
   *       after flushing.
   */
  virtual arrow::Status flush() = 0;

  /**
   * @brief Finalizes the dataset and returns the manifest
   *
   * Closes all column group writers, finalizes storage files, and constructs
   * a manifest containing metadata about the written dataset. After calling
   * close(), no additional data can be written to this writer instance.
   *
   * @return Result containing the dataset manifest, or error status
   *
   * @note This method should be called exactly once per writer instance.
   *       Subsequent calls will return an error.
   */
  virtual arrow::Result<std::shared_ptr<Manifest>> close() = 0;
};

}  // namespace milvus_storage::api