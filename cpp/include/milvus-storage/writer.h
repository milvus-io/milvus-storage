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
#include <queue>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <parquet/properties.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/common/config.h"

// Forward declarations
namespace internal::api {
class ColumnGroupWriter;
}

namespace milvus_storage::api {

/**
 * @brief Compression algorithms supported for column group storage
 */
enum class CompressionType {
  UNCOMPRESSED,  ///< No compression
  SNAPPY,        ///< Snappy compression (fast, moderate compression ratio)
  GZIP,          ///< GZIP compression (slower, good compression ratio)
  LZ4,           ///< LZ4 compression (very fast, good compression ratio)
  ZSTD,          ///< ZSTD compression (excellent compression ratio)
  BROTLI         ///< Brotli compression (excellent compression ratio)
};

/**
 * @brief Configuration properties for write operations
 *
 * This structure contains various properties that control how data is written,
 * including compression settings, encryption configurations, row group sizing,
 * and other write-time optimizations that affect performance and storage efficiency.
 */
struct WriteProperties {
  /// Maximum size of the part to upload to S3
  int64_t multi_part_upload_size = 0;

  /// Maximum number of rows per row group (affects memory usage and query granularity)
  uint64_t max_row_group_size = 64 * 1024;

  /// Write buffer size for each column group writer
  uint64_t buffer_size = 64 * 1024 * 1024;  // 64MB

  /// Compression algorithm to use for data storage
  CompressionType compression = CompressionType::ZSTD;

  /// Compression level (algorithm-specific, typically 1-9)
  int compression_level = -1;  // Use default level

  /// Enable dictionary encoding for string columns
  bool enable_dictionary = true;

  /// Encryption configuration
  struct {
    std::string cipher_type;      ///< Encryption cipher (e.g., "AES_GCM_V1")
    std::string cipher_key;       ///< Encryption key
    std::string cipher_metadata;  ///< Additional encryption metadata
  } encryption;

  /// Custom metadata to include in files
  std::map<std::string, std::string> custom_metadata;
};

/**
 * @brief Default write properties with optimized settings for typical workloads
 */
const WriteProperties default_write_properties = {.max_row_group_size = 64 * 1024,
                                                  .buffer_size = 64 * 1024 * 1024,
                                                  .compression = CompressionType::ZSTD,
                                                  .compression_level = -1,
                                                  .enable_dictionary = true,
                                                  .encryption = {},
                                                  .custom_metadata = {}};

/**
 * @brief Builder class for constructing WriteProperties objects
 *
 * Provides a fluent interface for configuring write properties with
 * method chaining and validation. This builder pattern allows for
 * easy configuration of compression, encryption, and performance settings.
 *
 * @example
 * auto properties = WritePropertiesBuilder()
 *                     .with_compression(CompressionType::ZSTD)
 *                     .with_max_row_group_size(128 * 1024)
 *                     .with_encryption("AES_GCM_V1", "secret_key")
 *                     .build();
 */
class WritePropertiesBuilder {
  public:
  /**
   * @brief Default constructor
   *
   * Initializes the builder with default write properties.
   */
  WritePropertiesBuilder();

  /**
   * @brief Sets the compression algorithm
   *
   * @param compression Compression algorithm to use
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_compression(CompressionType compression);

  /**
   * @brief Sets the compression level
   *
   * @param level Compression level (algorithm-specific, typically 1-9)
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_compression_level(int level);

  /**
   * @brief Sets the maximum row group size
   *
   * @param size Maximum number of rows per row group
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_max_row_group_size(int64_t size);

  /**
   * @brief Sets the write buffer size
   *
   * @param size Buffer size for each column group writer in bytes
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_buffer_size(int64_t size);

  /**
   * @brief Enables or disables dictionary encoding
   *
   * @param enable Whether to enable dictionary encoding for string columns
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_dictionary_encoding(bool enable);

  /**
   * @brief Sets encryption configuration
   *
   * @param cipher_type Encryption cipher type (e.g., "AES_GCM_V1")
   * @param cipher_key Encryption key
   * @param cipher_metadata Optional encryption metadata
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_encryption(const std::string& cipher_type,
                                          const std::string& cipher_key,
                                          const std::string& cipher_metadata = "");

  /**
   * @brief Adds custom metadata
   *
   * @param key Metadata key
   * @param value Metadata value
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_metadata(const std::string& key, const std::string& value);

  /**
   * @brief Builds and returns the configured WriteProperties
   *
   * @return WriteProperties object with all configured settings
   */
  [[nodiscard]] WriteProperties build() const;

  private:
  WriteProperties properties_;  ///< Internal properties being built
};

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
  explicit ColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, const std::string& default_format = "parquet")
      : schema_(std::move(schema)), default_format_(default_format) {}

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
  explicit SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, const std::string& default_format = "parquet")
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
                                        const std::string& default_format = "parquet")
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
                                      const std::string& default_format = "parquet")
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
   * @param fs Shared pointer to the filesystem interface for data access
   * @param base_path Base directory path where column group files will be written
   * @param schema Arrow schema defining the logical structure of the data
   * @param column_group_policy Policy for organizing columns into groups
   * @param properties Write configuration properties including compression and encryption
   * @return Unique pointer to a Writer instance
   *
   * @example
   * @code
   * auto fs = arrow::fs::LocalFileSystem::Make().ValueOrDie();
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
   * auto writer = Writer::create(fs, "/path/to/dataset", schema, std::move(policy), properties);
   *
   * // Use the writer
   * writer->write(batch);
   * auto manifest = writer->close().ValueOrDie();
   * @endcode
   */
  static std::unique_ptr<Writer> create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                        std::string base_path,
                                        std::shared_ptr<arrow::Schema> schema,
                                        std::unique_ptr<ColumnGroupPolicy> column_group_policy,
                                        const WriteProperties& properties = default_write_properties);

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of column group writers and temporary resources.
   * If close() hasn't been called, this will attempt to clean up gracefully.
   */
  virtual ~Writer() = default;

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