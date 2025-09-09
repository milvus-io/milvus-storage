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
#include <memory>
#include <cstdint>

namespace milvus_storage::internal::api {

/**
 * @brief Internal interface for format-specific chunk readers
 *
 * ChunkReader is an internal implementation detail that provides low-level access
 * to read data from specific file formats (e.g., Parquet, ORC). It handles
 * format-specific optimizations and I/O operations.
 *
 * This class should not be used directly by external users. Instead, use the
 * high-level Reader API or ChunkBatchReader for coordinated chunk access.
 */
class ChunkReader {
  public:
  /**
   * @brief Constructs a ChunkReader for file-based chunk reading
   *
   * @param fs Shared pointer to the filesystem interface for data access
   * @param file_path Path to the data file
   * @param needed_columns Subset of columns to read (empty = all columns)
   *
   * @throws std::invalid_argument if fs is null or file_path is empty
   */
  explicit ChunkReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                       std::string file_path,
                       std::vector<std::string> needed_columns)
      : fs_(std::move(fs)), file_path_(std::move(file_path)), needed_columns_(std::move(needed_columns)) {
    if (!fs_ || file_path_.empty()) {
      throw std::invalid_argument("FileSystem cannot be null and file_path cannot be empty");
    }
  }

  /**
   * @brief Destructor
   */
  virtual ~ChunkReader() = default;

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
      const std::vector<int64_t>& row_indices) const = 0;

  /**
   * @brief Retrieves a single chunk by its index from the column group
   *
   * Reads and returns a complete chunk (typically corresponding to a row group
   * in the underlying Parquet file) as an Arrow RecordBatch.
   *
   * @param chunk_index Zero-based index of the chunk to retrieve
   * @return Result containing the record batch for the specified chunk, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) const = 0;

  /**
   * @brief Retrieves a range of continuous chunks in a single I/O operation
   *
   * This method is optimized for reading multiple consecutive chunks efficiently,
   * reducing I/O overhead compared to individual chunk reads.
   *
   * @param start_chunk_index Zero-based index of the first chunk to retrieve
   * @param chunk_count Number of consecutive chunks to retrieve
   * @return Result containing vector of record batches for the chunk range, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunk_range(
      int64_t start_chunk_index, int64_t chunk_count) const = 0;

  /**
   * @brief Retrieves multiple chunks by their indices with optional parallel processing
   *
   * This method reads multiple chunks efficiently, potentially using format-specific
   * optimizations (e.g., ParquetChunkReader using ReadRowGroups).
   * The default implementation reads chunks sequentially.
   *
   * @param chunk_indices Vector of chunk indices to retrieve
   * @param parallelism Number of threads to use for parallel reading (default: 1, sequential)
   * @return Result containing vector of record batches for the specified chunks, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, int64_t parallelism = 1) const;

  /**
   * @brief Gets the memory size of a specific chunk
   *
   * This method returns the memory size of a chunk by consulting the cached metadata
   * in the chunk reader, allowing for accurate memory planning.
   *
   * @param chunk_index Zero-based index of the chunk
   * @return Result containing the chunk size in bytes, or error status
   */
  [[nodiscard]] virtual arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) const = 0;

  /**
   * @brief Gets the number of rows in a specific chunk
   *
   * This method returns the actual number of rows in a chunk by consulting the cached metadata
   * in the format reader, allowing for accurate row counting and alignment
   *
   * @param chunk_index Zero-based index of the chunk
   * @return Result containing the number of rows in the chunk, or error status
   */
  [[nodiscard]] virtual arrow::Result<int64_t> get_chunk_row_num(int64_t chunk_index) const = 0;

  /**
   * @brief Check if this chunk reader supports parallel reading
   *
   * Some file formats or storage systems may not support efficient parallel reading.
   * Subclasses can override this method to indicate their parallel reading capability.
   *
   * @return True if parallel reading is supported and beneficial, false otherwise
   */
  virtual bool supports_parallel_reading() const { return true; }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;  ///< Filesystem interface for data access
  std::string file_path_;                      ///< Path to the data file
  std::vector<std::string> needed_columns_;    ///< Subset of columns to read (empty = all columns)

  /**
   * @brief Validates that the chunk index is within valid range
   *
   * Each implementation should override this method to provide format-specific validation.
   *
   * @param chunk_index Index to validate
   * @return Status indicating whether the index is valid
   */
  [[nodiscard]] virtual arrow::Status validate_chunk_index(int64_t chunk_index) const = 0;
};

}  // namespace milvus_storage::internal::api