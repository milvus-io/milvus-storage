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

#include "milvus-storage/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>

#include <future>
#include <numeric>
#include <thread>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <arrow/array/concatenate.h>

namespace milvus_storage::api {

// ==================== PackedRecordBatchReader Implementation ====================

/**
 * @brief PackedRecordBatchReader for coordinated reading across multiple column groups
 *
 * This class provides efficient streaming access to data stored across multiple column groups,
 * with proper memory management, row alignment, and I/O optimization based on the packed reader algorithm.
 */
class PackedRecordBatchReader : public arrow::RecordBatchReader {
  public:
  /**
   * @brief Constructor
   *
   * @param fs Filesystem interface
   * @param column_groups Vector of column groups to read from
   * @param schema Target schema for the output
   * @param needed_columns Columns to read (empty = all columns)
   * @param properties Read properties including encryption settings
   * @param batch_size Maximum number of rows per record batch for memory management
   * @param buffer_size Maximum memory buffer size
   */
  explicit PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                   const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const std::vector<std::string>& needed_columns,
                                   const Properties& properties,
                                   int64_t batch_size = 1024,
                                   int64_t buffer_size = 32 * 1024 * 1024);

  /**
   * @brief Destructor - explicitly clean up resources
   */
  ~PackedRecordBatchReader() override;

  /**
   * @brief Get the schema of the output data
   */
  std::shared_ptr<arrow::Schema> schema() const override;

  /**
   * @brief Read the next batch of data
   *
   * @param batch Output parameter to receive the next record batch
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  /**
   * @brief Close the reader and clean up resources
   */
  arrow::Status Close() override;

  private:
  // Column group state tracking - similar to packed reader's ColumnGroupState
  struct ColumnGroupState {
    int64_t current_chunk = -1;  // Current chunk index being read (-1 means no chunk loaded yet)
    int64_t row_offset = 0;      // Current row offset in this column group
    int64_t rows_read = 0;       // Total rows read from this column group
    int64_t memory_usage = 0;    // Current memory usage by this column group
    bool exhausted = false;      // Whether this column group has no more data

    ColumnGroupState() = default;
  };

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  std::shared_ptr<arrow::Schema> output_schema_;
  std::vector<std::string> needed_columns_;
  Properties properties_;
  int64_t memory_limit_;
  int64_t batch_size_;

  // Runtime state - adapted from packed reader
  std::vector<ColumnGroupState> cg_states_;
  std::vector<std::unique_ptr<internal::api::ColumnGroupReader>> chunk_readers_;
  std::vector<std::queue<std::shared_ptr<arrow::RecordBatch>>> batch_queues_;
  int64_t memory_used_;                                           // Current memory usage
  int64_t absolute_row_position_;                                 // Current absolute row position (like packed reader)
  int64_t row_limit_;                                             // Row limit for alignment
  bool finished_;                                                 // Whether reading is finished
  size_t current_batch_index_;                                    // Current batch index for buffered reading
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches_;  // Simplified storage for all batches

  /**
   * @brief Advance the buffer by reading more data from column groups
   */
  arrow::Status advanceBuffer();
};

PackedRecordBatchReader::PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 const std::vector<std::string>& needed_columns,
                                                 const Properties& properties,
                                                 int64_t batch_size,
                                                 int64_t buffer_size)
    : fs_(std::move(fs)),
      column_groups_(column_groups),
      output_schema_(std::move(schema)),
      needed_columns_(needed_columns),
      properties_(properties),
      batch_size_(batch_size <= 0 ? DEFAULT_READ_BATCH_SIZE : batch_size),
      memory_limit_(buffer_size <= 0 ? INT64_MAX : buffer_size),
      memory_used_(0),
      absolute_row_position_(0),
      row_limit_(0),
      finished_(false),
      current_batch_index_(0) {
  // Initialize states for column groups
  cg_states_.resize(column_groups_.size());
  chunk_readers_.resize(column_groups_.size());
  batch_queues_.resize(column_groups_.size());

  for (size_t i = 0; i < column_groups_.size(); ++i) {
    cg_states_[i] = ColumnGroupState();
  }

  // Create chunk readers for each column group
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    auto& column_group = column_groups_[i];
    auto chunk_reader =
        internal::api::GroupReaderFactory::create(schema, column_group, fs_, needed_columns_, properties_);

    if (!chunk_reader) {
      throw std::runtime_error("Failed to create chunk reader for column group " + std::to_string(i));
    }

    chunk_readers_[i] = std::move(chunk_reader);
  }

  // Load initial data buffer
  auto status = advanceBuffer();
  if (!status.ok()) {
    throw std::runtime_error("Failed to load initial data buffer: " + status.ToString());
  }
}

PackedRecordBatchReader::~PackedRecordBatchReader() { (void)Close(); }

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return output_schema_; }

arrow::Status PackedRecordBatchReader::advanceBuffer() {
  std::vector<std::vector<int64_t>> chunks_to_read(column_groups_.size());
  size_t planned_memory = 0;

  // Advance to next chunk for column groups that need data
  auto advance_chunk = [&](size_t i) -> int64_t {
    int64_t next_chunk = cg_states_[i].current_chunk + 1;

    // Get actual chunk size from metadata instead of estimation
    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(next_chunk);
    if (!chunk_size_result.ok()) {
      return -1;  // No more chunk
    }
    int64_t chunk_size = chunk_size_result.ValueOrDie();

    chunks_to_read[i].emplace_back(next_chunk);
    planned_memory += chunk_size;
    cg_states_[i].memory_usage += chunk_size;
    cg_states_[i].current_chunk = next_chunk;

    auto chunk_row_num_result = chunk_readers_[i]->get_chunk_rows(next_chunk);
    if (!chunk_row_num_result.ok()) {
      return -1;  // Error getting chunk row count
    }
    int64_t chunk_row_num = chunk_row_num_result.ValueOrDie();
    cg_states_[i].row_offset += chunk_row_num;

    return chunk_size;
  };

  // Fill in tables that have no rows available
  int drained_index = -1;
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (cg_states_[i].row_offset > row_limit_) {
      continue;
    }

    memory_used_ -= std::max(static_cast<size_t>(0), static_cast<size_t>(cg_states_[i].memory_usage));
    cg_states_[i].memory_usage = 0;

    auto next_chunk_size = advance_chunk(i);
    if (next_chunk_size < 0) {
      drained_index = i;
      break;
    }
  }

  if (drained_index >= 0) {
    if (planned_memory == 0) {
      finished_ = true;
      return arrow::Status::OK();
    } else {
      return arrow::Status::Invalid("Column group " + std::to_string(drained_index) +
                                    " exhausted while others have data");
    }
  }

  // Fill in column groups using min heap for row alignment
  RowOffsetMinHeap sorted_offsets;
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (!cg_states_[i].exhausted) {
      sorted_offsets.emplace(i, cg_states_[i].row_offset);
    }
  }

  while (!sorted_offsets.empty() && planned_memory + memory_used_ < memory_limit_) {
    size_t i = sorted_offsets.top().first;

    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(cg_states_[i].current_chunk + 1);
    if (!chunk_size_result.ok()) {
      break;  // No more chunk, skip this column group
    }
    int64_t chunk_size = chunk_size_result.ValueOrDie();

    if (planned_memory + memory_used_ + chunk_size > memory_limit_) {
      break;
    }

    advance_chunk(i);
    sorted_offsets.pop();
    sorted_offsets.emplace(i, cg_states_[i].row_offset);
  }

  // Read the planned chunks
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (chunks_to_read[i].empty()) {
      continue;
    }
    // For PackedRecordBatchReader, we use direct chunk reader access for simplicity
    // since it already manages memory and has its own coordination logic
    ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_readers_[i]->get_chunks(chunks_to_read[i]));
    for (auto& chunk : chunks) {
      if (chunk) {
        batch_queues_[i].push(chunk);
      }
    }
  }

  memory_used_ += planned_memory;

  // Set row limit based on minimum row offset
  if (!sorted_offsets.empty()) {
    row_limit_ = sorted_offsets.top().second;
  }
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  if (finished_) {
    *out = nullptr;
    return arrow::Status::OK();
  }

  if (absolute_row_position_ >= row_limit_) {
    ARROW_RETURN_NOT_OK(advanceBuffer());
    if (absolute_row_position_ >= row_limit_) {
      finished_ = true;
      *out = nullptr;
      return arrow::Status::OK();
    }
  }

  // Find the minimum number of rows we can read from all column groups
  int64_t min_rows_available = INT64_MAX;
  std::vector<std::shared_ptr<arrow::RecordBatch>> current_batches(column_groups_.size());

  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (batch_queues_[i].empty()) {
      // No more data available
      finished_ = true;
      *out = nullptr;
      return arrow::Status::OK();
    }

    current_batches[i] = batch_queues_[i].front();
    if (current_batches[i]) {
      min_rows_available = std::min(min_rows_available, current_batches[i]->num_rows());
    }
  }

  if (min_rows_available == INT64_MAX || min_rows_available <= 0) {
    finished_ = true;
    *out = nullptr;
    return arrow::Status::OK();
  }

  min_rows_available = std::min(min_rows_available, batch_size_);

  // Create combined batch with proper row alignment - preserve schema field order
  std::vector<std::shared_ptr<arrow::Array>> combined_arrays(output_schema_->num_fields());
  std::vector<std::shared_ptr<arrow::Field>> combined_fields;
  std::vector<bool> field_filled(output_schema_->num_fields(), false);

  // Process each column group and place columns in correct schema positions
  for (size_t cg_idx = 0; cg_idx < column_groups_.size(); ++cg_idx) {
    auto& batch = current_batches[cg_idx];
    if (!batch) {
      continue;
    }

    // Slice batch if necessary to ensure row alignment
    std::shared_ptr<arrow::RecordBatch> aligned_batch = batch;
    if (batch->num_rows() > min_rows_available) {
      aligned_batch = batch->Slice(0, min_rows_available);
    }

    // Add columns from this batch to correct positions in combined result
    for (int col_idx = 0; col_idx < aligned_batch->num_columns(); ++col_idx) {
      auto array = aligned_batch->column(col_idx);
      auto field = aligned_batch->schema()->field(col_idx);

      // Find the correct position for this field in the output schema
      int output_field_idx = output_schema_->GetFieldIndex(field->name());
      if (output_field_idx >= 0 && output_field_idx < output_schema_->num_fields()) {
        // Only add if this column is needed
        bool is_needed =
            std::find(needed_columns_.begin(), needed_columns_.end(), field->name()) != needed_columns_.end();
        if (is_needed) {
          combined_arrays[output_field_idx] = array;
          field_filled[output_field_idx] = true;
        }
      }
    }

    // Update batch queue - if we consumed the entire batch, remove it
    if (batch->num_rows() == min_rows_available) {
      batch_queues_[cg_idx].pop();
      // Update column group state
      cg_states_[cg_idx].rows_read += min_rows_available;
      cg_states_[cg_idx].current_chunk++;
    } else {
      // Replace with remaining slice
      auto remaining_batch = batch->Slice(min_rows_available);
      batch_queues_[cg_idx].pop();
      batch_queues_[cg_idx].push(remaining_batch);
      cg_states_[cg_idx].rows_read += min_rows_available;
    }
  }

  // Fill in null arrays for any missing fields and build final field list
  for (int i = 0; i < output_schema_->num_fields(); ++i) {
    auto field = output_schema_->field(i);
    bool is_needed = std::find(needed_columns_.begin(), needed_columns_.end(), field->name()) != needed_columns_.end();

    if (is_needed) {
      combined_fields.emplace_back(field);

      if (!field_filled[i]) {
        // Create null array for missing fields
        ARROW_ASSIGN_OR_RAISE(auto null_array, arrow::MakeArrayOfNull(field->type(), min_rows_available));
        combined_arrays[i] = null_array;
      }
    }
  }

  // Compact arrays to match the needed fields order
  std::vector<std::shared_ptr<arrow::Array>> final_arrays;
  for (int i = 0; i < output_schema_->num_fields(); ++i) {
    auto field = output_schema_->field(i);
    bool is_needed = std::find(needed_columns_.begin(), needed_columns_.end(), field->name()) != needed_columns_.end();

    if (is_needed && combined_arrays[i]) {
      final_arrays.emplace_back(combined_arrays[i]);
    }
  }

  if (final_arrays.empty()) {
    finished_ = true;
    *out = nullptr;
    return arrow::Status::OK();
  }

  // Create the output schema and batch
  auto combined_schema = arrow::schema(combined_fields);
  *out = arrow::RecordBatch::Make(combined_schema, min_rows_available, final_arrays);

  absolute_row_position_ += min_rows_available;
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::Close() {
  if (finished_) {
    return arrow::Status::OK();
  }

  // Clean up remaining data in all tables
  for (auto& queue : batch_queues_) {
    while (!queue.empty()) {
      queue.front().reset();
      queue.pop();
    }
  }

  chunk_readers_.clear();
  batch_queues_.clear();
  cg_states_.clear();
  memory_used_ = 0;
  finished_ = true;

  return arrow::Status::OK();
}

// ==================== ChunkReaderImpl Implementation ====================

/**
 * @brief Concrete implementation of ChunkReader interface
 */
class ChunkReaderImpl : public ChunkReader {
  public:
  /**
   * @brief Constructs a ChunkReaderImpl for a specific column group
   *
   * @param fs Shared pointer to the filesystem interface for data access
   * @param schema Shared pointer to the schema of the dataset
   * @param column_group Shared pointer to the column group metadata and configuration
   * @param needed_columns Subset of columns to read (empty = all columns)
   * @param properties Read properties including encryption settings
   *
   * @throws std::invalid_argument if fs or column_group is null
   */
  explicit ChunkReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
                           std::shared_ptr<arrow::Schema> schema,
                           std::shared_ptr<ColumnGroup> column_group,
                           std::vector<std::string> needed_columns,
                           Properties properties);

  /**
   * @brief Destructor
   */
  ~ChunkReaderImpl() override = default;

  // Implement ChunkReader interface
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, int64_t parallelism) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;  ///< Filesystem interface for data access
  std::shared_ptr<arrow::Schema> schema_;      ///< Schema of the dataset
  std::shared_ptr<ColumnGroup> column_group_;  ///< Column group metadata and configuration
  std::vector<std::string> needed_columns_;    ///< Subset of columns to read (empty = all columns)
  std::unique_ptr<internal::api::ColumnGroupReader> chunk_reader_;
};

// ==================== ChunkReaderImpl Method Implementations ====================

ChunkReaderImpl::ChunkReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
                                 std::shared_ptr<arrow::Schema> schema,
                                 std::shared_ptr<ColumnGroup> column_group,
                                 std::vector<std::string> needed_columns,
                                 Properties properties)
    : fs_(std::move(fs)), column_group_(std::move(column_group)), needed_columns_(std::move(needed_columns)) {
  // create schema from column group
  assert(schema != nullptr);
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& field : schema->fields()) {
    // test if the field is in the column group
    for (const auto& column : column_group_->columns) {
      if (column == field->name()) {
        fields.emplace_back(field);
        break;
      }
    }
  }
  schema_ = arrow::schema(fields);
  chunk_reader_ = internal::api::GroupReaderFactory::create(schema_, column_group_, fs_, needed_columns_, properties);
}

arrow::Result<std::vector<int64_t>> ChunkReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  return chunk_reader_->get_chunk_indices(row_indices);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkReaderImpl::get_chunk(int64_t chunk_index) {
  return chunk_reader_->get_chunk(chunk_index);
}

class ParallelDegreeChunkSplitStrategy {
  public:
  explicit ParallelDegreeChunkSplitStrategy(uint64_t parallel_degree) : parallel_degree_(parallel_degree) {}

  std::vector<std::vector<int64_t>> split(const std::vector<int64_t>& chunk_indices) {
    std::vector<std::vector<int64_t>> blocks;
    if (chunk_indices.empty()) {
      return blocks;
    }

    std::vector<int64_t> sorted_chunk_indices = chunk_indices;
    std::sort(sorted_chunk_indices.begin(), sorted_chunk_indices.end());

    uint64_t actual_parallel_degree = std::min(parallel_degree_, static_cast<uint64_t>(sorted_chunk_indices.size()));

    if (actual_parallel_degree == 0) {
      actual_parallel_degree = 1;
    }

    auto create_continuous_blocks = [&](size_t max_block_size = SIZE_MAX) {
      std::vector<std::vector<int64_t>> continuous_blocks;
      int64_t current_start = sorted_chunk_indices[0];
      int64_t current_count = 1;

      for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
        int64_t next_chunk = sorted_chunk_indices[i];

        if (next_chunk == current_start + current_count && current_count < max_block_size) {
          current_count++;
          continue;
        }
        std::vector<int64_t> block(current_count);
        std::iota(block.begin(), block.end(), current_start);
        continuous_blocks.emplace_back(block);
        current_start = next_chunk;
        current_count = 1;
      }

      if (current_count > 0) {
        std::vector<int64_t> block(current_count);
        std::iota(block.begin(), block.end(), current_start);
        continuous_blocks.emplace_back(block);
      }
      return continuous_blocks;
    };

    if (sorted_chunk_indices.size() <= actual_parallel_degree) {
      return create_continuous_blocks();
    }

    size_t avg_block_size = (sorted_chunk_indices.size() + actual_parallel_degree - 1) / actual_parallel_degree;

    return create_continuous_blocks(avg_block_size);
  }

  private:
  uint64_t parallel_degree_;
};

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism) {
  if (chunk_indices.empty()) {
    return std::vector<std::shared_ptr<arrow::RecordBatch>>();
  }

  // Single chunk case - use direct get_chunk
  if (chunk_indices.size() == 1) {
    ARROW_ASSIGN_OR_RAISE(auto chunk, get_chunk(chunk_indices[0]));
    return std::vector<std::shared_ptr<arrow::RecordBatch>>{chunk};
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> results(chunk_indices.size());

  // Create index mapping for original order restoration
  std::unordered_map<int64_t, size_t> index_to_position;
  for (size_t i = 0; i < chunk_indices.size(); ++i) {
    index_to_position[chunk_indices[i]] = i;
  }

  // Choose strategy based on memory limit and parallelism
  std::vector<std::vector<int64_t>> blocks;

  ParallelDegreeChunkSplitStrategy strategy(parallelism);
  auto parallel_blocks = strategy.split(chunk_indices);
  // Convert to compatible format
  blocks.reserve(parallel_blocks.size());
  for (const auto& block : parallel_blocks) {
    blocks.emplace_back(block);
  }

  // Create futures for parallel block processing
  std::vector<std::future<arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>>> futures;
  futures.reserve(blocks.size());

  // Launch parallel tasks for each block
  for (const auto& block : blocks) {
    futures.emplace_back(std::async(std::launch::async, [this, block]() {
      std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>> block_results;

      // Use get_chunk_range for continuous blocks to optimize I/O
      auto range_result = chunk_reader_->get_chunks(block);
      if (!range_result.ok()) {
        return arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>(
            range_result.status());
      }

      auto batches = range_result.ValueOrDie();
      for (int64_t i = 0; i < block.size(); ++i) {
        int64_t chunk_idx = block[i];
        block_results.emplace_back(chunk_idx, batches[i]);
      }

      return arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>(
          std::move(block_results));
    }));
  }

  // Collect results from all blocks and restore original order
  for (auto& future : futures) {
    auto block_result = future.get();
    if (!block_result.ok()) {
      return block_result.status();
    }

    for (const auto& [chunk_idx, batch] : block_result.ValueOrDie()) {
      auto it = index_to_position.find(chunk_idx);
      if (it != index_to_position.end()) {
        results[it->second] = batch;
      }
    }
  }

  return results;
}

// ==================== ReaderImpl Implementation ====================

/**
 * @brief Concrete implementation of the Reader interface
 *
 * This class provides the actual implementation for reading data from milvus
 * storage datasets using manifest-based metadata. It supports efficient batch
 * reading, column projection, filtering, and parallel processing of large datasets
 * stored in packed columnar format.
 */
class ReaderImpl : public Reader {
  public:
  /**
   * @brief Constructs a ReaderImpl instance for a milvus storage dataset
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
  explicit ReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::shared_ptr<Manifest> manifest,
                      std::shared_ptr<arrow::Schema> schema,
                      const std::shared_ptr<std::vector<std::string>>& needed_columns,
                      Properties properties)
      : fs_(std::move(fs)),
        manifest_(std::move(manifest)),
        schema_(std::move(schema)),
        properties_(std::move(properties)) {
    // Validate required parameters
    if (!fs_) {
      throw std::invalid_argument("FileSystem cannot be null");
    }
    if (!manifest_) {
      throw std::invalid_argument("Manifest cannot be null");
    }
    if (!schema_) {
      throw std::invalid_argument("Schema cannot be null");
    }

    // Initialize the list of columns to read from the dataset
    if (needed_columns != nullptr) {
      needed_columns_ = *needed_columns;

      // Validate that all requested columns exist in the schema
      for (const auto& column_name : needed_columns_) {
        if (!schema_->GetFieldByName(column_name)) {
          throw std::invalid_argument("Column '" + column_name + "' not found in schema");
        }
      }
    } else {
      // If no specific columns requested, read all columns from the schema
      needed_columns_.clear();
      needed_columns_.reserve(schema_->num_fields());
      for (size_t i = 0; i < schema_->num_fields(); ++i) {
        needed_columns_.emplace_back(schema_->field(i)->name());
      }
    }
  }

  /**
   * @brief Performs a full table scan with optional filtering and buffering
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& /*predicate*/, int64_t batch_size, int64_t buffer_size) const override {
    // Validate parameters
    if (batch_size <= 0) {
      return arrow::Status::Invalid("Batch size must be positive, got: " + std::to_string(batch_size));
    }
    if (buffer_size <= 0) {
      return arrow::Status::Invalid("Buffer size must be positive, got: " + std::to_string(buffer_size));
    }

    // Initialize column groups if not already done
    initialize_needed_column_groups();

    // Collect file paths from needed column groups only
    // This provides the PackedRecordBatchReader with only necessary data files
    auto paths = std::vector<std::string>(needed_column_groups_.size());

    for (const auto& column_group : needed_column_groups_) {
      if (column_group->paths.empty()) {
        return arrow::Status::Invalid("Column group has empty paths");
      }
      // Add all paths from this column group
      for (const auto& path : column_group->paths) {
        paths.emplace_back(path);
      }
    }

    if (paths.empty()) {
      return arrow::Status::Invalid("No column groups found for the requested columns");
    }

    // Create schema with only needed columns for projection
    std::vector<std::shared_ptr<arrow::Field>> needed_fields;
    for (const auto& column_name : needed_columns_) {
      auto field = schema_->GetFieldByName(column_name);
      if (field != nullptr) {
        needed_fields.emplace_back(field);
      }
    }
    auto projected_schema = arrow::schema(needed_fields);

    try {
      auto reader = std::make_shared<PackedRecordBatchReader>(fs_, needed_column_groups_, projected_schema,
                                                              needed_columns_, properties_, batch_size, buffer_size);
      return reader;
    } catch (const std::exception& e) {
      return arrow::Status::Invalid("Failed to create PackedRecordBatchReader: " + std::string(e.what()));
    }
  }

  /**
   * @brief Get a chunk reader for a specific column group
   */
  [[nodiscard]] arrow::Result<std::unique_ptr<ChunkReader>> get_chunk_reader(
      int64_t column_group_index) const override {
    auto column_groups = manifest_->get_column_groups();
    if (column_group_index < 0 || column_group_index >= column_groups.size()) {
      return arrow::Status::Invalid("Column group index out of range: " + std::to_string(column_group_index));
    }
    auto column_group = column_groups[column_group_index];

    return std::make_unique<ChunkReaderImpl>(fs_, schema_, column_group, needed_columns_, properties_);
  }

  /**
   * @brief Extracts specific rows by their global indices with parallel processing
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices,
                                                                        int64_t parallelism) const override {
    throw std::runtime_error("take is not implemented for Reader");
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;  ///< Filesystem interface for data access
  std::shared_ptr<Manifest> manifest_;         ///< Dataset manifest with metadata and layout info
  std::shared_ptr<arrow::Schema> schema_;      ///< Logical Arrow schema defining data structure
  Properties properties_;                      ///< Configuration properties including encryption
  std::vector<std::string> needed_columns_;    ///< Subset of columns to read (empty = all columns)
  mutable std::vector<std::shared_ptr<ColumnGroup>>
      needed_column_groups_;  ///< Column groups required for needed columns (cached)

  /**
   * @brief Initializes the needed column groups based on requested columns
   */
  void initialize_needed_column_groups() const {
    if (!needed_column_groups_.empty()) {
      return;  // Already initialized
    }

    // Determine which column groups are needed based on the requested columns
    // This optimization allows reading only the column groups that contain
    // the requested columns, reducing I/O and improving performance
    auto visited_column_groups = std::set<std::shared_ptr<ColumnGroup>>();
    for (const auto& column_name : needed_columns_) {
      auto column_group = manifest_->get_column_group(column_name);
      if (column_group != nullptr && visited_column_groups.find(column_group) == visited_column_groups.end()) {
        needed_column_groups_.emplace_back(column_group);
        visited_column_groups.insert(column_group);
      }
    }
  }
};

// ==================== Factory Function Implementation ====================

std::unique_ptr<Reader> Reader::create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       std::shared_ptr<Manifest> manifest,
                                       std::shared_ptr<arrow::Schema> schema,
                                       const std::shared_ptr<std::vector<std::string>>& needed_columns,
                                       const Properties& properties) {
  return std::make_unique<ReaderImpl>(std::move(fs), std::move(manifest), std::move(schema), needed_columns,
                                      properties);
}

}  // namespace milvus_storage::api
