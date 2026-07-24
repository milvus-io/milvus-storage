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

#include <string>
#include <unordered_map>

#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/format_reader_cache.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/thread_pool.h"

namespace milvus_storage::api {

using ColumnMemorySizes = std::unordered_map<std::string, uint64_t>;
using ColumnMemorySizesPtr = std::shared_ptr<const ColumnMemorySizes>;

struct ChunkInfo {
  size_t file_index;               // current chunk belong which file
  size_t row_offset_in_row_group;  // the starting row offset of this row group in its file
  size_t row_offset_in_file;       // the starting row offset of file
  size_t number_of_rows;           // number of rows in this row group
  size_t row_group_index_in_file;  // the index of this row group in its file
  size_t global_row_end;           // the ending row offset of this row group in the whole chunk reader
  uint64_t avg_memory_size;        // average memory usage of this row group
  // Keyed by the current file's field names so chunks from files with different
  // physical column orders can share immutable metadata without normalization.
  ColumnMemorySizesPtr column_memory_sizes;
  // False means avg_memory_size is only a placeholder and column_memory_sizes is unavailable.
  bool memory_size_available;

  // Format all logical/file offset fields for diagnostics.
  [[nodiscard]] std::string ToString() const;
};

class ColumnGroupReader {
  public:
  virtual ~ColumnGroupReader() = default;
  virtual arrow::Status open() = 0;
  virtual size_t total_number_of_chunks() const = 0;
  virtual size_t total_rows() const = 0;
  virtual arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) = 0;

  // NOT thread-safe: concurrent calls on the same object may race on the underlying FormatReader.
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) = 0;

  // Thread-safe: each call opens an independent FormatReader from reusable metadata.
  virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) = 0;

  virtual arrow::Result<uint64_t> get_chunk_estimated_size(int64_t chunk_index) = 0;
  virtual arrow::Result<uint64_t> get_chunk_column_estimated_size(int64_t chunk_index, int col_idx) = 0;
  virtual arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) = 0;

  // Get chunk info by index (for async task planning and splitting).
  // The returned reference remains valid while the opened reader is unchanged.
  virtual const ChunkInfo& get_chunk_info(int64_t chunk_index) const = 0;

  // Async execution of a pre-planned ChunkTask.
  // The task must come from ChunkTask::Build() or be a valid split of one.
  // range_start/range_end are file-local half-open row offsets.
  // Each task uses independent mutable FormatReader state.
  virtual folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> get_chunks_async(
      const ChunkTask& task) = 0;

  // get the file schema of this column group (always derived from file metadata, not projected)
  virtual std::shared_ptr<arrow::Schema> get_schema() const = 0;

  /**
   * @brief Create a chunk reader for a column group
   *
   * @param schema Schema of the dataset
   * @param column_group Column group containing format, path, and metadata
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @param properties Read properties
   * @return Unique pointer to the created chunk reader
   */
  [[nodiscard]] static arrow::Result<std::unique_ptr<ColumnGroupReader>> create(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const std::vector<std::string>& needed_columns,
      const milvus_storage::api::Properties& properties,
      const std::function<std::string(const std::string&)>& key_retriever,
      const std::string& predicate = "",
      const milvus_storage::MetadataCache& cache = milvus_storage::MetadataCache());

  // Create and fully initialize a column-group reader through the async format
  // factory. The returned future carries initialization errors and retains the reader.
  [[nodiscard]] static folly::SemiFuture<arrow::Result<std::unique_ptr<ColumnGroupReader>>> create_async(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const std::vector<std::string>& needed_columns,
      const milvus_storage::api::Properties& properties,
      const std::function<std::string(const std::string&)>& key_retriever,
      const std::string& predicate = "",
      const milvus_storage::MetadataCache& cache = milvus_storage::MetadataCache());
};

}  // namespace milvus_storage::api
