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

#include "milvus-storage/manifest.h"

namespace milvus_storage::api {

/**
 * @brief Configuration properties for read operations
 * 
 * This structure contains various properties that control how data is read,
 * including security-related configurations for encrypted data access.
 * These properties are used to configure encryption settings when reading
 * from encrypted storage systems.
 */
struct ReadProperties {
  /// Type of encryption cipher used (e.g., "AES256", "ChaCha20")
  std::string cipher_type;
  
  /// Encryption key for decrypting data during read operations
  std::string cipher_key;
  
  /// Additional metadata required for encryption/decryption context
  std::string cipher_metadata;
  
  // TODO: Add key retriever interface for dynamic key management
  //KeyRetriever cipher_key_retriever;
};

/**
 * @brief Default read properties with standard configuration
 * 
 * Provides a default configuration with no encryption enabled.
 * This is suitable for reading from unencrypted storage systems.
 */
const ReadProperties default_read_properties = {
  .cipher_type = "",      ///< No encryption by default
  .cipher_key = "",       ///< Empty key indicates no encryption
  .cipher_metadata = "",  ///< No encryption metadata needed
};

/**
 * @brief Builder class for constructing ReadProperties objects
 * 
 * Provides a fluent interface for configuring read properties with
 * method chaining and validation. This builder pattern allows for
 * easy configuration of encryption and other read-related settings.
 * 
 * @example
 * auto properties = ReadPropertiesBuilder()
 *                     .with_cipher_type("AES256")
 *                     .with_cipher_key("secret_key")
 *                     .build();
 */
class ReadPropertiesBuilder {
 public:
  /**
   * @brief Default constructor
   * 
   * Initializes the builder with default read properties (no encryption).
   */
  ReadPropertiesBuilder() {
    properties_ = default_read_properties;
  }

  /**
   * @brief Builds and returns the configured ReadProperties
   * 
   * @return ReadProperties object with all configured settings
   */
  ReadProperties build() {
    return properties_;
  }
  
  /**
   * @brief Destructor
   */
  ~ReadPropertiesBuilder() = default;

 private:
  ReadProperties properties_;  ///< Internal properties being built
};

/**
 * @brief Reader for individual column groups in packed storage format
 * 
 * ColumnGroupReader provides low-level access to read data from a specific
 * column group within a packed storage layout. It handles chunk-based reading
 * and supports both individual and batch chunk operations.
 * 
 * Column groups in packed storage contain related columns stored together
 * for optimal compression and query performance.
 */
class ColumnGroupReader {
 public:
  /**
   * @brief Constructs a ColumnGroupReader for a specific column group
   * 
   * @param fs Shared pointer to the filesystem interface for data access
   * @param column_group Shared pointer to the column group metadata and configuration
   */
  ColumnGroupReader(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::shared_ptr<ColumnGroup>& column_group);
  
  /**
   * @brief Destructor
   */
  ~ColumnGroupReader() = default;

  /**
   * @brief Maps row indices to their corresponding chunk indices within the column group
   * 
   * This method determines which chunks contain the specified rows, allowing for
   * efficient targeted reading of specific data ranges.
   * 
   * @param row_indices Vector of global row indices to map to chunk indices
   * @return Result containing vector of chunk indices, or error status
   */
   arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices);

   /**
    * @brief Retrieves a single chunk by its index from the column group
    * 
    * Reads and returns a complete chunk (typically corresponding to a row group
    * in the underlying Parquet file) as an Arrow RecordBatch.
    * 
    * @param chunk_index Zero-based index of the chunk to retrieve
    * @return Result containing the record batch for the specified chunk, or error status
    */
   arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int64_t chunk_index);
 
   /**
    * @brief Retrieves multiple chunks by their indices with optional parallel processing
    * 
    * This method reads multiple chunks efficiently, potentially using parallel I/O
    * operations to improve performance when accessing non-contiguous chunks.
    * 
    * @param chunk_indices Vector of chunk indices to retrieve
    * @param parallelism Number of threads to use for parallel reading (default: 1, sequential)
    * @return Result containing vector of record batches for the specified chunks, or error status
    */
   arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(const std::vector<int64_t>& chunk_indices, int64_t parallelism = 1);

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;        ///< Filesystem interface for data access
  std::shared_ptr<ColumnGroup> column_group_;       ///< Column group metadata and configuration
};

/**
 * @brief High-level reader interface for milvus storage data
 * 
 * The Reader class provides a unified interface for reading data from milvus
 * storage datasets using manifest-based metadata. It supports efficient batch
 * reading, column projection, filtering, and parallel processing of large datasets
 * stored in packed columnar format.
 * 
 * This reader leverages the manifest system to understand the dataset structure,
 * including column groups, data layout, and metadata, providing optimized access
 * patterns for analytical workloads.
 * 
 * @example Basic usage:
 * @code
 * auto fs = arrow::fs::LocalFileSystem::Make().ValueOrDie();
 * Manifest manifest = LoadManifest(fs, "/path/to/dataset");
 * auto schema = manifest.schema();
 * 
 * Reader reader(fs, manifest, *schema);
 * auto batch_reader = reader.scan().ValueOrDie();
 * 
 * std::shared_ptr<arrow::RecordBatch> batch;
 * while (batch_reader->ReadNext(&batch).ok() && batch) {
 *   // Process batch
 * }
 * @endcode
 */
class Reader {
 public:
  /**
   * @brief Constructs a Reader instance for a milvus storage dataset
   * 
   * Initializes the reader with dataset manifest and configuration. The manifest
   * provides metadata about column groups, data layout, and storage locations,
   * enabling optimized query planning and execution.
   * 
   * @param fs Shared pointer to the filesystem interface for data access
   * @param manifest Dataset manifest containing metadata and column group information
   * @param schema Arrow schema defining the logical structure of the data
   * @param needed_columns Optional vector of column names to read (nullptr reads all columns)
   * @param properties Read configuration properties including encryption settings
   */
  Reader(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::shared_ptr<Manifest>& manifest, 
    const std::shared_ptr<arrow::Schema>& schema, 
    const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr, 
    const ReadProperties& properties = default_read_properties);

  /**
   * @brief Performs a full table scan with optional filtering and buffering
   * 
   * Creates a RecordBatchReader for sequential reading of the entire dataset.
   * The reader automatically handles column group coordination and provides
   * efficient streaming access to large datasets.
   * 
   * @param predicate Filter expression string for row-level filtering 
   *                  (empty string disables filtering)
   * @param batch_size Maximum number of rows per record batch for memory management
   * @param buffer_size Target buffer size in bytes for internal I/O buffering
   * @return Result containing a RecordBatchReader for sequential data access, or error status
   * 
   * @note The predicate filtering may not be fully pushed down to storage level.
   *       Additional client-side filtering may be required for complete accuracy.
   */
  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> scan(const std::string& predicate = "", 
    int64_t batch_size = 1024, int64_t buffer_size = 32 * 1024 * 1024);

  /**
   * @brief Extracts specific rows by their global indices with parallel processing
   * 
   * Efficiently retrieves rows at the specified global indices from across all
   * column groups in the dataset. This method is optimized for random access
   * patterns and supports parallel I/O for improved performance.
   * 
   * The implementation maps row indices to appropriate column groups and chunks,
   * performs parallel reads when beneficial, and reconstructs the final result
   * maintaining the original row order.
   * 
   * @param row_indices Vector of global row indices to extract (need not be sorted)
   * @param parallelism Number of threads to use for parallel chunk reading (default: 1)
   * @return Result containing RecordBatch with the requested rows in original order,
   *         or error status if indices are out of range
   * 
   * @note For optimal performance with large index sets, consider sorting indices
   *       or using scan() with appropriate filtering for range-based access.
   */
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices, int64_t parallelism = 1);

  /**
   * @brief Destructor
   * 
   * Cleans up resources and ensures proper cleanup of column group readers
   * and cached metadata.
   */
  ~Reader();

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;     ///< Filesystem interface for data access
  std::shared_ptr<Manifest> manifest_;            ///< Dataset manifest with metadata and layout info
  std::shared_ptr<arrow::Schema> schema_;         ///< Logical Arrow schema defining data structure
  ReadProperties properties_;                     ///< Configuration properties including encryption
  std::vector<std::string> needed_columns_;       ///< Subset of columns to read (empty = all columns)
  std::vector<std::shared_ptr<ColumnGroup>> needed_column_groups_; ///< Column groups required for needed columns
};
}  // namespace milvus_storage::api