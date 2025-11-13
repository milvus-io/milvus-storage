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

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/row_offset_heap.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage::api {
/**
 * @brief Interface for reading individual column groups in packed storage format
 *
 * ChunkReader provides low-level access to read data from a specific
 * column group within a packed storage layout. It handles chunk-based reading
 * and supports both individual and batch chunk operations.
 *
 * Column groups in packed storage contain related columns stored together
 * for optimal compression and query performance.
 */
class ChunkReader {
  public:
  /**
   * @brief Virtual destructor for interface
   */
  virtual ~ChunkReader() = default;

  /**
   * @brief Returns the total number of chunks in the column group
   */
  [[nodiscard]] virtual size_t total_number_of_chunks() const = 0;

  /**
   * @brief Maps row indices to their corresponding chunk indices within the column group
   *
   * This method determines which chunks contain the specified rows, allowing for
   * efficient targeted reading of specific data ranges.
   *
   * @param row_indices Vector of global row indices to map to chunk indices
   * @return Result containing vector of chunk indices, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::vector<int64_t>> get_chunk_indices(
      const std::vector<int64_t>& row_indices) = 0;

  /**
   * @brief Retrieves a single chunk by its index from the column group
   *
   * Reads and returns a complete chunk (typically corresponding to a row group
   * in the underlying Parquet file) as an Arrow RecordBatch.
   *
   * @param chunk_index Zero-based index of the chunk to retrieve
   * @return Result containing the record batch for the specified chunk, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) = 0;

  /**
   * @brief Retrieves multiple chunks by their indices with optional parallel processing
   *
   * This method reads multiple chunks efficiently, potentially using parallel I/O
   * operations to improve performance when accessing non-contiguous chunks.
   * This has been implemented in chunk reader base class.
   * Format implementations does not need to override this method.
   *
   * @param chunk_indices Vector of chunk indices to retrieve
   * @param parallelism Number of threads to use for parallel reading
   * @return Result containing vector of record batches for the specified chunks, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, int64_t parallelism) = 0;

  /**
   * @brief Retrieves the metadata of chunks
   */
  [[nodiscard]] virtual arrow::Result<std::vector<uint64_t>> get_chunk_size() = 0;
  [[nodiscard]] virtual arrow::Result<std::vector<uint64_t>> get_chunk_rows() = 0;
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
 */
class Reader {
  public:
  /**
   * @brief Factory function to create a Reader instance
   *
   * Creates a concrete Reader implementation that can be used to read data from
   * milvus storage datasets. This function provides a clean interface for creating
   * readers without exposing the concrete implementation details.
   *
   * @param cgs Dataset column group information
   * @param schema Arrow schema defining the logical structure of the data
   * @param needed_columns Optional vector of column names to read (nullptr reads all columns)
   * @param properties Read configuration properties including encryption settings
   * @return Unique pointer to a Reader instance
   *
   * @example
   * @code
   * auto fs = arrow::fs::LocalFileSystem::Make().ValueOrDie();
   * // actully is column groups
   * Manifest manifest = LoadManifest(fs, "/path/to/dataset");
   * auto schema = manifest.schema();
   *
   * ReadProperties props;
   * props["cipher_type"] = "AES256";
   * props["buffer_size"] = "65536";
   *
   * auto reader = Reader::create(manifest, schema, nullptr, props);
   * auto batch_reader = reader->get_record_batch_reader().ValueOrDie();
   *
   * std::shared_ptr<arrow::RecordBatch> batch;
   * while (batch_reader->ReadNext(&batch).ok() && batch) {
   *   // Process batch
   * }
   * @endcode
   */
  static std::unique_ptr<Reader> create(std::shared_ptr<ColumnGroups> cgs,
                                        std::shared_ptr<arrow::Schema> schema,
                                        const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr,
                                        const Properties& properties = {});

  /**
   * @brief Virtual destructor
   *
   * Cleans up resources and ensures proper cleanup of column group readers
   * and cached metadata.
   */
  virtual ~Reader() = default;

  /**
   * @brief Convenience method for get_record_batch_reader with no predicate
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader() const {
    return get_record_batch_reader("");
  }

  /**
   * @brief Convenience method for take with default parallelism
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices) const {
    return take(row_indices, 1);
  }

  /**
   * @brief Retrieves the column groups managed by this reader
   * @return Vector of shared pointers to ColumnGroup instances
   */
  [[nodiscard]] virtual std::shared_ptr<ColumnGroups> get_column_groups() const = 0;

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
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate) const = 0;

  /**
   * @brief Get a chunk reader for a specific column group
   *
   * @param column_group_index Index of the column group to read from
   * @return Result containing a ChunkReader for the specified column group, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::unique_ptr<ChunkReader>> get_chunk_reader(
      int64_t column_group_index) const = 0;

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
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                                                int64_t parallelism) const = 0;

  /**
   * @brief Set a callback function to retrieve encryption keys based on metadata
   * @param callback Function that takes metadata string and returns the corresponding encryption key
   *        which used
   */
  virtual void set_keyretriever(const std::function<std::string(const std::string&)>& callback) = 0;
};

}  // namespace milvus_storage::api