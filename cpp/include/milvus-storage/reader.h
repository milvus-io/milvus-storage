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

#include <optional>

#include <arrow/array.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <folly/futures/Future.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/row_offset_heap.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/thread_pool.h"

namespace milvus_storage::api {

class Manifest;

struct MaskedReadOptions {
  std::optional<uint64_t> visible_until_ts;
  // Primary-key field id. milvus-storage has no inherent primary-key concept, so
  // PRIMARY_KEY delta logs require the caller to declare which schema field is
  // the primary key. Required whenever the manifest contains PRIMARY_KEY deltas;
  // unused by predicate deletes.
  std::optional<int64_t> pk_field_id;
  // Row-timestamp field id. milvus-storage has no inherent row-timestamp field
  // either, so the caller must declare which schema field carries the per-row
  // timestamp used for `row_ts <= delete_ts`. Required whenever the manifest
  // contains any delete (PRIMARY_KEY or PREDICATE).
  std::optional<int64_t> row_timestamp_field_id;
};

struct MaskedRecordBatch {
  std::shared_ptr<arrow::RecordBatch> batch;
  std::shared_ptr<arrow::BooleanArray> keep_mask;
};

class MaskedRecordBatchReader {
  public:
  virtual ~MaskedRecordBatchReader() = default;
  virtual arrow::Status ReadNext(MaskedRecordBatch* out) = 0;
};

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
   * @param parallelism The parallelism use for reading, if global threadpool have been set,
   *                    this parameter is ignored. Otherwise, reader will use this parameter
   *                    to start the threads.
   * @return Result containing vector of record batches for the specified chunks, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) = 0;

  /**
   * @brief Retrieves multiple chunks asynchronously by their indices.
   *
   * The returned future completes with the same ordered result as get_chunks().
   * Duplicate chunk indices are preserved in the output.
   *
   * @note The default implementation invokes get_chunks() before returning a
   *       ready future and may therefore block. Native implementations may defer work.
   * @note In the task-based override, parallelism is a task-granularity target,
   *       not a guaranteed concurrency limit.
   */
  [[nodiscard]] virtual folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>
  get_chunks_async(const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) {
    return folly::makeSemiFuture(get_chunks(chunk_indices, parallelism));
  }

  /**
   * @brief Returns the estimated decoded-memory size of every chunk.
   *
   * The result is indexed by chunk. When per-column metadata is available, each
   * estimate is the sum of the logical column group's fields, independently of
   * the active projection; fields present only in a physical file are excluded.
   * Formats without per-column metadata retain their aggregate estimate.
   * Estimates may be zero when a chunk has no live rows or its known
   * decoded-memory size is zero. Unavailable estimates are returned as an
   * error, typically arrow::Status::NotImplemented, rather than as zero.
   *
   * @return One estimated size in bytes per chunk, or an error status.
   */
  [[nodiscard]] virtual arrow::Result<std::vector<uint64_t>> get_chunk_estimated_size() = 0;

  /**
   * @brief Returns the estimated decoded-memory size of a top-level field in every chunk.
   *
   * The field is resolved against the logical column group, independently of
   * the active projection. A field absent from a particular physical file has
   * a known zero estimate for that chunk. Unavailable statistics are returned
   * as an error rather than as zero.
   *
   * @param field_name Name of a unique top-level field in the column group.
   * @return One estimated size in bytes per chunk, or an error status.
   */
  [[nodiscard]] virtual arrow::Result<std::vector<uint64_t>> get_chunk_column_estimated_size(
      const std::string& field_name) = 0;

  /**
   * @brief Returns estimated decoded-memory sizes for every top-level field and chunk.
   *
   * The outer dimension follows logical column-group order and the inner
   * dimension is indexed by chunk: result[column_index][chunk_index]. Results
   * are independent of the active projection. Individual estimates may be
   * zero only when the corresponding estimate is known to be zero. Unavailable
   * estimates are returned as an error rather than as zero.
   *
   * @return Estimated sizes in bytes arranged as [column][chunk], or an error status.
   */
  [[nodiscard]] virtual arrow::Result<std::vector<std::vector<uint64_t>>> get_chunk_column_estimated_size() = 0;

  /**
   * @brief Returns the exact logical row count of every chunk.
   *
   * Unlike the estimated-size APIs, failures are returned as an error status
   * and are never converted to a zero fallback. A returned zero is an exact
   * row count for an empty logical chunk.
   *
   * @return One exact row count per chunk, or an error status.
   */
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
 *
 * Memory-estimation metadata is optional and is not required for normal data
 * reads. When it is unavailable, reader creation and data access remain usable;
 * only the chunk-size estimation APIs return arrow::Status::NotImplemented.
 * The underlying statistics failure is not exposed through those APIs.
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
   * @param schema Arrow schema defining the logical structure of the data.
   *        If nullptr, the schema is derived from the underlying file metadata.
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
   * auto reader = Reader::create(manifest, schema, props);
   * auto batch_reader = reader->get_record_batch_reader().ValueOrDie();
   *
   * std::shared_ptr<arrow::RecordBatch> batch;
   * while (batch_reader->ReadNext(&batch).ok() && batch) {
   *   // Process batch
   * }
   * @endcode
   *
   * About schema:
   *  - If `schema` is provided, it defines the logical structure: field types are taken from
   *    the schema, column validation is performed against it, and missing columns are filled
   *    with NULL values.
   *  - If `schema` is nullptr, field types are derived from the underlying files (e.g. Parquet
   *    file metadata, Lance fragment schema, Vortex file schema). In this mode, all output
   *    columns must exist in the column group files; NULL filling for missing columns is not
   *    supported (since types are unknown).
   *
   *  Empty column groups behavior:
   *  - If `schema` is provided and column groups are empty, `get_record_batch_reader()` and
   *    `take()` return an empty result (empty RecordBatchReader / empty Table) with the given
   *    schema.
   *  - If `schema` is nullptr and column groups are empty, these methods return an Invalid
   *    error, since no schema can be derived to construct the result.
   *
   * About projection (set via `needed_columns` in create, or per-call on `get_chunk_reader`/`take`):
   *  Top Reader (when schema is provided):
   *  - All column names in `needed_columns` must exist as field names in the schema.
   *    Example:
   *      - schema{a,b,c}, needed_columns{a,b,c,d} or {d}
   *      - The input arguments are invalid.
   *
   *  - If `needed_columns` is nullptr, all columns will be read.
   *  - If `needed_columns` is not nullptr, only the columns in `needed_columns` will be read.
   *    - For RecordBatchReader/take:
   *      - The output schema will match `needed_columns`. Missing fields will be filled with NULL.
   *      Example 1:
   *        - Stored columns {a,b,c} (in all groups), needed_columns{b,c,d}
   *        - The output is {b,c,d} (with 'd' being a null column).
   *      Example 2:
   *        - Stored columns {a,b,c} (in all groups), needed_columns{d,e,f}
   *        - The output is {d,e,f} (all null columns).
   *
   *    - For ChunkReader:
   *      - At least one of the `needed_columns` must exist in the column group.
   *        Example:
   *           - ChunkReader(column_group_id=0), columns{a,b}, needed_columns{c,d}
   *           - The input arguments are invalid.
   *      - Only the intersection of columns will be returned.
   *        Example:
   *           - ChunkReader(column_group_id=0), columns{a,b}, needed_columns{a,c}
   *           - The output is {a}.
   *
   *  Top Reader (when schema is nullptr):
   *  - If `needed_columns` is nullptr, all columns from all column groups will be read.
   *  - If `needed_columns` is provided, only those columns will be read (no validation).
   *  - The output schema is derived from the file metadata (e.g. Parquet schema).
   *
   *  Column Group Reader: Uses the `columns` in the current column group to filter
   *    `needed_columns` and build the `out_schema`. The filtered
   *    projection must match the `out_schema`.
   *
   *  Format Reader: If the schema is empty or nullptr, it reads the columns specified in
   *    `needed_columns`. If `needed_columns` is also empty or nullptr, it reads all columns.
   */
  static std::unique_ptr<Reader> create(const std::shared_ptr<ColumnGroups>& cgs,
                                        const std::shared_ptr<arrow::Schema>& schema = nullptr,
                                        const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr,
                                        const Properties& properties = {});

  static std::unique_ptr<Reader> create(const std::shared_ptr<Manifest>& manifest,
                                        const std::shared_ptr<arrow::Schema>& schema = nullptr,
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
   * @return Result containing a RecordBatchReader for sequential data access, or error status
   *
   * @note The predicate filtering may not be fully pushed down to storage level.
   *       Additional client-side filtering may be required for complete accuracy.
   */
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate) const = 0;

  [[nodiscard]] virtual arrow::Result<std::shared_ptr<MaskedRecordBatchReader>> get_masked_record_batch_reader(
      const MaskedReadOptions& options) const = 0;

  /**
   * @brief Get a chunk reader for a specific column group
   *
   * @param column_group_index Index of the column group to read from
   * @param needed_columns Optional per-call column projection override. If non-null and non-empty,
   *        overrides the default `needed_columns` from `Reader::create`.
   * @return Result containing a ChunkReader for the specified column group, or error status
   */
  [[nodiscard]] virtual arrow::Result<std::unique_ptr<ChunkReader>> get_chunk_reader(
      int64_t column_group_index, const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) const = 0;

  /**
   * @brief Asynchronously get a chunk reader for a specific column group.
   *
   * The returned future includes the format-reader open/footer work needed to
   * initialize the chunk reader.
   *
   * @note The default implementation invokes get_chunk_reader() before returning
   *       a ready future and may therefore block.
   */
  [[nodiscard]] virtual folly::SemiFuture<arrow::Result<std::unique_ptr<ChunkReader>>> get_chunk_reader_async(
      int64_t column_group_index, const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) const {
    return folly::makeSemiFuture(get_chunk_reader(column_group_index, needed_columns));
  }

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
   * @param row_indices Vector of global row indices to extract, MUST be uniqued and sorted
   * @param parallelism The parallelism use for reading, if global threadpool have been set,
   *                    this parameter is ignored. Otherwise, reader will use this parameter
   *                    to start the threads.
   * @param needed_columns Optional per-call column projection override. If non-null and non-empty,
   *        overrides the default `needed_columns` from `Reader::create`.
   * @return Result containing RecordBatch with the requested rows in original order,
   *         or error status if indices are out of range
   *
   * @note For optimal performance with large index sets, consider sorting indices
   *       or using scan() with appropriate filtering for range-based access.
   */
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::Table>> take(
      const std::vector<int64_t>& row_indices,
      size_t parallelism = 1,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) = 0;

  /**
   * @brief Asynchronously extracts specific rows by their global indices.
   *
   * The returned future completes with the same ordered table as take().
   *
   * @note The default implementation invokes take() before returning a ready
   *       future and may therefore block.
   * @note In the task-based override, parallelism controls task splitting;
   *       actual execution depends on the format backend and caller executor.
   */
  [[nodiscard]] virtual folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(
      const std::vector<int64_t>& row_indices,
      size_t parallelism = 1,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) {
    return folly::makeSemiFuture(take(row_indices, parallelism, needed_columns));
  }

  /**
   * @brief Set a callback function to retrieve encryption keys based on metadata.
   *
   * This is a setup-only API and is not thread-safe with read operations. Call it
   * before creating record batch readers, chunk readers, or calling take().
   *
   * @param callback Function that takes metadata string and returns the corresponding encryption key.
   */
  virtual void set_keyretriever(const std::function<std::string(const std::string&)>& callback) = 0;
};

}  // namespace milvus_storage::api
