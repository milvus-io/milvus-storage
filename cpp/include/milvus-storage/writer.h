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
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <parquet/properties.h>

#include "milvus-storage/manifest.h"

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
  /// Maximum number of rows per row group (affects memory usage and query granularity)
  int64_t max_row_group_size = 64 * 1024;

  /// Target size of each column group file in bytes
  int64_t target_file_size = 128 * 1024 * 1024;  // 128MB

  /// Write buffer size for each column group writer
  int64_t buffer_size = 64 * 1024 * 1024;  // 64MB

  /// Compression algorithm to use for data storage
  CompressionType compression = CompressionType::SNAPPY;

  /// Compression level (algorithm-specific, typically 1-9)
  int compression_level = -1;  // Use default level

  /// Enable dictionary encoding for string columns
  bool enable_dictionary = true;

  /// Enable statistics collection for columns (min/max/null_count)
  bool enable_statistics = true;

  /// Write version for Parquet format compatibility
  int parquet_version = 1;  // Parquet v1.0

  /// Enable byte stream splitting for floating point columns
  bool enable_byte_stream_split = false;

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
                                                  .target_file_size = 128 * 1024 * 1024,
                                                  .buffer_size = 64 * 1024 * 1024,
                                                  .compression = CompressionType::SNAPPY,
                                                  .compression_level = -1,
                                                  .enable_dictionary = true,
                                                  .enable_statistics = true,
                                                  .parquet_version = 1,
                                                  .enable_byte_stream_split = false,
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
  WritePropertiesBuilder() : properties_(default_write_properties) {}

  /**
   * @brief Sets the compression algorithm
   *
   * @param compression Compression algorithm to use
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_compression(CompressionType compression) {
    properties_.compression = compression;
    return *this;
  }

  /**
   * @brief Sets the compression level
   *
   * @param level Compression level (algorithm-specific, typically 1-9)
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_compression_level(int level) {
    properties_.compression_level = level;
    return *this;
  }

  /**
   * @brief Sets the maximum row group size
   *
   * @param size Maximum number of rows per row group
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_max_row_group_size(int64_t size) {
    properties_.max_row_group_size = size;
    return *this;
  }

  /**
   * @brief Sets the target file size
   *
   * @param size Target size of each column group file in bytes
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_target_file_size(int64_t size) {
    properties_.target_file_size = size;
    return *this;
  }

  /**
   * @brief Sets the write buffer size
   *
   * @param size Buffer size for each column group writer in bytes
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_buffer_size(int64_t size) {
    properties_.buffer_size = size;
    return *this;
  }

  /**
   * @brief Enables or disables dictionary encoding
   *
   * @param enable Whether to enable dictionary encoding for string columns
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_dictionary_encoding(bool enable) {
    properties_.enable_dictionary = enable;
    return *this;
  }

  /**
   * @brief Enables or disables statistics collection
   *
   * @param enable Whether to collect column statistics (min/max/null_count)
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_statistics(bool enable) {
    properties_.enable_statistics = enable;
    return *this;
  }

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
                                          const std::string& cipher_metadata = "") {
    properties_.encryption.cipher_type = cipher_type;
    properties_.encryption.cipher_key = cipher_key;
    properties_.encryption.cipher_metadata = cipher_metadata;
    return *this;
  }

  /**
   * @brief Adds custom metadata
   *
   * @param key Metadata key
   * @param value Metadata value
   * @return Reference to this builder for method chaining
   */
  WritePropertiesBuilder& with_metadata(const std::string& key, const std::string& value) {
    properties_.custom_metadata[key] = value;
    return *this;
  }

  /**
   * @brief Builds and returns the configured WriteProperties
   *
   * @return WriteProperties object with all configured settings
   */
  [[nodiscard]] WriteProperties build() const { return properties_; }

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
   * @brief Constructs a column group policy for the given schema
   *
   * @param schema Arrow schema defining the columns to be grouped
   */
  explicit ColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema) : schema_(std::move(schema)) {}

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
};

/**
 * @brief Simple column group policy that puts all columns in a single group
 *
 * This policy is suitable for datasets where all columns are typically accessed together,
 * or for small datasets where the overhead of multiple files isn't justified.
 */
class SingleColumnGroupPolicy : public ColumnGroupPolicy {
  public:
  explicit SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema);

  [[nodiscard]] bool requires_sample() const override { return false; }

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override {
    return arrow::Status::OK();  // No sampling needed
  }

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
                                        std::vector<std::string> column_name_patterns);

  [[nodiscard]] bool requires_sample() const override { return false; }

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override {
    return arrow::Status::OK();  // No sampling needed
  }

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
                                      int64_t max_columns_in_group)
      : ColumnGroupPolicy(std::move(schema)),
        max_avg_column_size_(max_avg_column_size),
        max_columns_in_group_(max_columns_in_group) {}

  [[nodiscard]] bool requires_sample() const override { return true; }

  arrow::Status sample(const std::shared_ptr<arrow::RecordBatch>& batch) override;

  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const override;

  private:
  int64_t max_avg_column_size_;
  int64_t max_columns_in_group_;
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
 *
 * @example Basic usage:
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
 *
 * Writer writer(fs, "/path/to/dataset", schema, std::move(policy), properties);
 *
 * // Write data batches
 * for (auto& batch : data_batches) {
 *   writer.write(batch);
 * }
 *
 * // Finalize and get manifest
 * auto manifest = writer.close().ValueOrDie();
 * @endcode
 */
class Writer {
  public:
  /**
   * @brief Constructs a Writer instance for a milvus storage dataset
   *
   * Initializes the writer with filesystem access, target location, schema,
   * column grouping policy, and write configuration. The writer prepares
   * column group writers based on the policy and begins accepting data.
   *
   * @param fs Shared pointer to the filesystem interface for data access
   * @param base_path Base directory path where column group files will be written
   * @param schema Arrow schema defining the logical structure of the data
   * @param column_group_policy Policy for organizing columns into groups
   * @param properties Write configuration properties including compression and encryption
   */
  Writer(std::shared_ptr<arrow::fs::FileSystem> fs,
         std::string base_path,
         std::shared_ptr<arrow::Schema> schema,
         std::unique_ptr<ColumnGroupPolicy> column_group_policy,
         const WriteProperties& properties = default_write_properties);

  /**
   * @brief Destructor
   *
   * Ensures proper cleanup of column group writers and temporary resources.
   * If close() hasn't been called, this will attempt to clean up gracefully.
   */
  ~Writer();

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
  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch);

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
  arrow::Status flush();

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
  arrow::Result<std::shared_ptr<Manifest>> close();

  /**
   * @brief Adds custom metadata to be included in the manifest
   *
   * @param key Metadata key
   * @param value Metadata value
   * @return Status indicating success or error condition
   */
  arrow::Status add_metadata(const std::string& key, const std::string& value);

  /**
   * @brief Gets the current write statistics
   *
   * @return Statistics about rows written, bytes written, etc.
   */
  struct WriteStats {
    int64_t rows_written = 0;         ///< Total number of rows written
    int64_t bytes_written = 0;        ///< Total number of bytes written
    int64_t batches_written = 0;      ///< Total number of batches written
    int64_t column_groups_count = 0;  ///< Number of column groups created
  };

  [[nodiscard]] WriteStats get_stats() const;

  private:
  // Forward declarations for internal types
  class ColumnGroupWriter;

  // ==================== Internal Data Members ====================

  std::shared_ptr<arrow::fs::FileSystem> fs_;               ///< Filesystem interface for data access
  std::string base_path_;                                   ///< Base directory for column group files
  std::shared_ptr<arrow::Schema> schema_;                   ///< Logical schema of the dataset
  std::unique_ptr<ColumnGroupPolicy> column_group_policy_;  ///< Policy for organizing columns
  WriteProperties properties_;                              ///< Write configuration properties

  std::shared_ptr<Manifest> manifest_;                                    ///< Dataset manifest being built
  std::vector<std::unique_ptr<ColumnGroupWriter>> column_group_writers_;  ///< Writers for each column group
  std::map<std::string, std::string> custom_metadata_;                    ///< Custom metadata for the manifest

  WriteStats stats_;  ///< Current write statistics
  bool closed_;       ///< Whether the writer has been closed
  bool initialized_;  ///< Whether the writer has been initialized

  // ==================== Internal Helper Methods ====================

  /**
   * @brief Initializes column group writers based on the policy
   *
   * @return Status indicating success or error condition
   */
  arrow::Status initialize_column_group_writers(const std::shared_ptr<arrow::RecordBatch>& batch);

  /**
   * @brief Distributes a record batch to appropriate column group writers
   *
   * @param batch The batch to distribute
   * @return Status indicating success or error condition
   */
  arrow::Status distribute_batch(const std::shared_ptr<arrow::RecordBatch>& batch);

  /**
   * @brief Generates a unique file path for a column group
   *
   * @param column_group_id Unique identifier for the column group
   * @return Generated file path
   */
  [[nodiscard]] std::string generate_column_group_path(int64_t column_group_id) const;

  /**
   * @brief Updates the manifest with column group information
   *
   * @return Status indicating success or error condition
   */
  arrow::Status finalize_manifest();
};

}  // namespace milvus_storage::api