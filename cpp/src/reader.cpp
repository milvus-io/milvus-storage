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

#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <limits>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::api {

// ==================== ChunkReader Implementation ====================

arrow::Status ChunkReader::validate_chunk_index(int64_t chunk_index) const {
  if (chunk_index < 0) {
    return arrow::Status::Invalid("Chunk index cannot be negative: " + std::to_string(chunk_index));
  }

  if (column_group_->stats.num_chunks > 0 && chunk_index >= column_group_->stats.num_chunks) {
    return arrow::Status::Invalid("Chunk index " + std::to_string(chunk_index) + " is out of range. Column group has " +
                                  std::to_string(column_group_->stats.num_chunks) + " chunks");
  }

  return arrow::Status::OK();
}

arrow::Result<std::vector<int64_t>> ChunkReader::get_chunk_indices(const std::vector<int64_t>& row_indices) const {
  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices vector cannot be empty");
  }

  // Validate row indices are non-negative
  for (const auto& row_index : row_indices) {
    if (row_index < 0) {
      return arrow::Status::Invalid("Row index cannot be negative: " + std::to_string(row_index));
    }
  }

  // Get total rows in this column group to validate row indices
  int64_t total_rows = column_group_->stats.num_rows;
  if (total_rows <= 0) {
    return arrow::Status::Invalid("Column group has no rows or invalid row count: " + std::to_string(total_rows));
  }

  // Validate all row indices are within bounds
  for (const auto& row_index : row_indices) {
    if (row_index >= total_rows) {
      return arrow::Status::Invalid("Row index " + std::to_string(row_index) + " is out of range. Column group has " +
                                    std::to_string(total_rows) + " rows");
    }
  }

  // Calculate chunk indices based on row distribution
  // Assumes uniform distribution of rows across chunks
  int64_t num_chunks = column_group_->stats.num_chunks;
  if (num_chunks <= 0) {
    return arrow::Status::Invalid("Column group has no chunks or invalid chunk count: " + std::to_string(num_chunks));
  }

  int64_t rows_per_chunk = total_rows / num_chunks;
  int64_t remaining_rows = total_rows % num_chunks;

  std::vector<int64_t> chunk_indices;
  chunk_indices.reserve(row_indices.size());

  for (const auto& row_index : row_indices) {
    int64_t chunk_index;

    // Handle the case where the last few chunks might have an extra row
    if (remaining_rows > 0 && row_index >= (num_chunks - remaining_rows) * rows_per_chunk) {
      // This row is in one of the larger chunks at the end
      int64_t adjusted_row = row_index - (num_chunks - remaining_rows) * rows_per_chunk;
      chunk_index = (num_chunks - remaining_rows) + adjusted_row / (rows_per_chunk + 1);
    } else {
      // This row is in a regular-sized chunk
      chunk_index = row_index / rows_per_chunk;
    }

    // Ensure chunk index is within bounds
    chunk_index = std::min(chunk_index, num_chunks - 1);
    chunk_indices.push_back(chunk_index);
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  // Create a format reader for this column group
  auto format_reader =
      FormatReaderFactory::create_reader(column_group_->format, fs_, column_group_, nullptr, ReadProperties{});

  if (!format_reader) {
    return arrow::Status::Invalid("Failed to create format reader for column group " +
                                  std::to_string(column_group_->id));
  }

  // Initialize the format reader
  ARROW_RETURN_NOT_OK(format_reader->initialize(column_group_, needed_columns_));

  // For chunk-based reading, we need to use the format reader's record batch reader
  // and skip to the specific chunk. This implementation assumes the format reader
  // can provide chunk-level access (e.g., row groups in Parquet).
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, format_reader->get_record_batch_reader());

  // Read through chunks until we reach the desired chunk_index
  std::shared_ptr<arrow::RecordBatch> batch;
  for (int64_t i = 0; i <= chunk_index; ++i) {
    ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
    if (!batch && i < chunk_index) {
      return arrow::Status::Invalid("Chunk index " + std::to_string(chunk_index) + " is out of range. Only " +
                                    std::to_string(i) + " chunks available");
    }
  }

  if (!batch) {
    // Return an empty batch with the correct schema if no data
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(batch_reader->schema(), 0, empty_arrays);
  }

  return batch;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism) const {
  if (chunk_indices.empty()) {
    return arrow::Status::Invalid("Chunk indices vector cannot be empty");
  }

  if (parallelism < 1) {
    return arrow::Status::Invalid("Parallelism must be at least 1, got: " + std::to_string(parallelism));
  }

  // Validate all chunk indices
  for (const auto& chunk_index : chunk_indices) {
    ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> result_batches;
  result_batches.reserve(chunk_indices.size());

  if (parallelism == 1 || chunk_indices.size() == 1) {
    // Sequential execution
    for (const auto& chunk_index : chunk_indices) {
      ARROW_ASSIGN_OR_RAISE(auto batch, get_chunk(chunk_index));
      result_batches.push_back(batch);
    }
  } else {
    // Parallel execution using std::async
    // Note: In a production system, you might want to use a thread pool
    // or Arrow's compute context for better thread management

    std::vector<std::future<arrow::Result<std::shared_ptr<arrow::RecordBatch>>>> futures;
    futures.reserve(chunk_indices.size());

    // Launch parallel tasks
    for (const auto& chunk_index : chunk_indices) {
      auto future = std::async(std::launch::async, [this, chunk_index]() { return get_chunk(chunk_index); });
      futures.push_back(std::move(future));
    }

    // Collect results in order
    for (auto& future : futures) {
      ARROW_ASSIGN_OR_RAISE(auto batch, future.get());
      result_batches.push_back(batch);
    }
  }

  return result_batches;
}

// ==================== Reader Implementation ====================

Reader::Reader(std::shared_ptr<arrow::fs::FileSystem> fs,
               std::shared_ptr<Manifest> manifest,
               std::shared_ptr<arrow::Schema> schema,
               const std::shared_ptr<std::vector<std::string>>& needed_columns,
               ReadProperties properties)
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
    for (int i = 0; i < schema_->num_fields(); ++i) {
      needed_columns_.push_back(schema_->field(i)->name());
    }
  }

  // Column groups will be initialized lazily
}

void Reader::initialize_needed_column_groups() const {
  if (!needed_column_groups_.empty()) {
    return;  // Already initialized
  }

  // Determine which column groups are needed based on the requested columns
  // This optimization allows reading only the column groups that contain
  // the requested columns, reducing I/O and improving performance
  auto visited_column_groups = std::set<int64_t>();
  for (const auto& column_name : needed_columns_) {
    auto column_group = manifest_->get_column_group(column_name);
    if (column_group != nullptr && visited_column_groups.find(column_group->id) == visited_column_groups.end()) {
      needed_column_groups_.push_back(column_group);
      visited_column_groups.insert(column_group->id);
    }
  }
}

arrow::Result<std::shared_ptr<ChunkReader>> Reader::get_chunk_reader(int64_t column_group_id) const {
  if (column_group_id < 0) {
    return arrow::Status::Invalid("Column group ID cannot be negative: " + std::to_string(column_group_id));
  }

  auto column_group = manifest_->get_column_group(column_group_id);
  if (column_group == nullptr) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(column_group_id) + " not found");
  }

  try {
    return std::make_shared<ChunkReader>(fs_, column_group, needed_columns_);
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create ChunkReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> Reader::get_record_batch_reader(const std::string& predicate,
                                                                                         int64_t batch_size,
                                                                                         int64_t buffer_size) const {
  // Validate parameters
  if (batch_size <= 0) {
    return arrow::Status::Invalid("Batch size must be positive, got: " + std::to_string(batch_size));
  }
  if (buffer_size <= 0) {
    return arrow::Status::Invalid("Buffer size must be positive, got: " + std::to_string(buffer_size));
  }

  // Initialize column groups if not already done
  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for the requested columns");
  }

  // Create schema with only needed columns for projection
  std::vector<std::shared_ptr<arrow::Field>> needed_fields;
  for (const auto& column_name : needed_columns_) {
    auto field = schema_->GetFieldByName(column_name);
    if (field != nullptr) {
      needed_fields.push_back(field);
    }
  }
  auto projected_schema = arrow::schema(needed_fields);

  // Create and return our custom ColumnGroupRecordBatchReader
  // This provides memory-controlled, row-aligned streaming access across column groups
  try {
    auto reader = std::make_shared<ColumnGroupRecordBatchReader>(fs_, needed_column_groups_, projected_schema,
                                                                 needed_columns_, properties_, buffer_size);
    return reader;
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create ColumnGroupRecordBatchReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::take(const std::vector<int64_t>& row_indices,
                                                                int64_t parallelism) const {
  // Validate parameters
  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices vector cannot be empty");
  }

  if (parallelism < 1) {
    return arrow::Status::Invalid("Parallelism must be at least 1, got: " + std::to_string(parallelism));
  }

  // Validate that all row indices are non-negative
  for (const auto& row_index : row_indices) {
    if (row_index < 0) {
      return arrow::Status::Invalid("Row index cannot be negative: " + std::to_string(row_index));
    }
  }

  // Initialize column groups if not already done
  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for the requested columns");
  }

  // For each column group, we need to create a format reader and extract the requested rows
  std::vector<std::shared_ptr<arrow::RecordBatch>> column_group_batches;
  column_group_batches.reserve(needed_column_groups_.size());

  if (parallelism == 1 || needed_column_groups_.size() == 1) {
    // Sequential execution across column groups
    for (const auto& column_group : needed_column_groups_) {
      auto format_reader =
          FormatReaderFactory::create_reader(column_group->format, fs_, column_group, schema_, properties_);

      if (!format_reader) {
        return arrow::Status::Invalid("Failed to create format reader for column group " +
                                      std::to_string(column_group->id));
      }

      // Get needed columns for this specific column group
      std::vector<std::string> cg_needed_columns;
      for (const auto& col_name : needed_columns_) {
        if (column_group->contains_column(col_name)) {
          cg_needed_columns.push_back(col_name);
        }
      }

      ARROW_RETURN_NOT_OK(format_reader->initialize(column_group, cg_needed_columns));
      ARROW_ASSIGN_OR_RAISE(auto batch, format_reader->take(row_indices, parallelism));
      column_group_batches.push_back(batch);
    }
  } else {
    // Parallel execution across column groups
    std::vector<std::future<arrow::Result<std::shared_ptr<arrow::RecordBatch>>>> futures;
    futures.reserve(needed_column_groups_.size());

    for (const auto& column_group : needed_column_groups_) {
      auto future = std::async(
          std::launch::async,
          [this, &column_group, &row_indices, parallelism]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
            auto format_reader =
                FormatReaderFactory::create_reader(column_group->format, fs_, column_group, schema_, properties_);

            if (!format_reader) {
              return arrow::Status::Invalid("Failed to create format reader for column group " +
                                            std::to_string(column_group->id));
            }

            // Get needed columns for this specific column group
            std::vector<std::string> cg_needed_columns;
            for (const auto& col_name : needed_columns_) {
              if (column_group->contains_column(col_name)) {
                cg_needed_columns.push_back(col_name);
              }
            }

            ARROW_RETURN_NOT_OK(format_reader->initialize(column_group, cg_needed_columns));
            return format_reader->take(row_indices,
                                       1);  // Use sequential within each column group to avoid over-parallelization
          });
      futures.push_back(std::move(future));
    }

    // Collect results
    for (auto& future : futures) {
      ARROW_ASSIGN_OR_RAISE(auto batch, future.get());
      column_group_batches.push_back(batch);
    }
  }

  // Now we need to merge the column group batches into a single RecordBatch
  // Each column group batch contains a subset of columns for the same rows
  if (column_group_batches.size() == 1) {
    return column_group_batches[0];
  }

  // Merge multiple column group batches
  std::vector<std::shared_ptr<arrow::Array>> merged_arrays;
  std::vector<std::shared_ptr<arrow::Field>> merged_fields;

  for (const auto& batch : column_group_batches) {
    for (int i = 0; i < batch->num_columns(); ++i) {
      merged_arrays.push_back(batch->column(i));
      merged_fields.push_back(batch->schema()->field(i));
    }
  }

  auto merged_schema = arrow::schema(merged_fields);
  return arrow::RecordBatch::Make(merged_schema, row_indices.size(), merged_arrays);
}

// ==================== ColumnGroupRecordBatchReader Implementation ====================

ColumnGroupRecordBatchReader::ColumnGroupRecordBatchReader(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
    std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::string>& needed_columns,
    const ReadProperties& properties,
    int64_t buffer_size)
    : fs_(std::move(fs)),
      column_groups_(column_groups),
      output_schema_(std::move(schema)),
      needed_columns_(needed_columns),
      properties_(properties),
      buffer_size_(buffer_size),
      memory_used_(0),
      current_row_position_(0),
      row_limit_(0),
      finished_(false) {
  // Initialize states and queues
  cg_states_.resize(column_groups_.size());
  format_readers_.resize(column_groups_.size());
  batch_readers_.resize(column_groups_.size());
  batch_queues_.resize(column_groups_.size());

  auto status = initialize();
  if (!status.ok()) {
    throw std::runtime_error("Failed to initialize ColumnGroupRecordBatchReader: " + status.ToString());
  }
}

std::shared_ptr<arrow::Schema> ColumnGroupRecordBatchReader::schema() const { return output_schema_; }

arrow::Status ColumnGroupRecordBatchReader::initialize() {
  // Create format readers for each column group
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    auto& column_group = column_groups_[i];

    // Get needed columns for this specific column group
    std::vector<std::string> cg_needed_columns;
    for (const auto& col_name : needed_columns_) {
      if (column_group->contains_column(col_name)) {
        cg_needed_columns.push_back(col_name);
      }
    }

    // Create format reader
    auto format_reader =
        FormatReaderFactory::create_reader(column_group->format, fs_, column_group, output_schema_, properties_);

    if (!format_reader) {
      return arrow::Status::Invalid("Failed to create format reader for column group " +
                                    std::to_string(column_group->id));
    }

    ARROW_RETURN_NOT_OK(format_reader->initialize(column_group, cg_needed_columns));
    format_readers_[i] = std::move(format_reader);

    // Create and cache batch reader for this column group
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, format_readers_[i]->get_record_batch_reader());
    batch_readers_[i] = batch_reader;
  }

  // Buffer all data for streaming by reading all column groups and combining them properly
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>> all_cg_batches;
  all_cg_batches.resize(batch_readers_.size());

  size_t max_batches = 0;
  for (size_t i = 0; i < batch_readers_.size(); ++i) {
    // Read all data from this column group
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(batch_readers_[i]->ReadNext(&batch));

      if (!batch) {
        break;
      }

      all_cg_batches[i].push_back(batch);
    }
    max_batches = std::max(max_batches, all_cg_batches[i].size());
  }

  // Combine batches from all column groups, handling mismatched batch counts
  std::vector<std::shared_ptr<arrow::RecordBatch>> combined_batches;
  for (size_t j = 0; j < max_batches; ++j) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_to_combine;

    for (size_t i = 0; i < all_cg_batches.size(); ++i) {
      if (j < all_cg_batches[i].size()) {
        batches_to_combine.push_back(all_cg_batches[i][j]);
      } else {
        // This column group has fewer batches - add nullptr
        batches_to_combine.push_back(nullptr);
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto combined_batch, combine_batches(batches_to_combine));

    // Split large batches to ensure streaming behavior
    int64_t max_streaming_batch_size = 2000;
    if (combined_batch->num_rows() > max_streaming_batch_size) {
      // Split into smaller chunks
      for (int64_t offset = 0; offset < combined_batch->num_rows(); offset += max_streaming_batch_size) {
        int64_t length = std::min(max_streaming_batch_size, combined_batch->num_rows() - offset);
        combined_batches.push_back(combined_batch->Slice(offset, length));
      }
    } else {
      combined_batches.push_back(combined_batch);
    }
  }

  // Store all combined batches for streaming
  buffered_batches_ = std::move(combined_batches);

  return arrow::Status::OK();
}

arrow::Status ColumnGroupRecordBatchReader::advance_buffer() {
  // This method implements memory-controlled reading across column groups
  // similar to PackedRecordBatchReader::advanceBuffer()

  if (finished_) {
    return arrow::Status::OK();
  }

  // Find column groups that need more data (have consumed current chunks)
  std::vector<bool> needs_data(column_groups_.size(), false);
  int64_t min_row_offset = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < cg_states_.size(); ++i) {
    if (!cg_states_[i].exhausted) {
      if (batch_queues_[i].empty() || cg_states_[i].row_offset < row_limit_) {
        needs_data[i] = true;
      }
      min_row_offset = std::min(min_row_offset, cg_states_[i].row_offset);
    }
  }

  // If no column group needs data and we have data available, we're good
  bool has_data = false;
  for (size_t i = 0; i < batch_queues_.size(); ++i) {
    if (!batch_queues_[i].empty()) {
      has_data = true;
      break;
    }
  }

  if (!std::any_of(needs_data.begin(), needs_data.end(), [](bool x) { return x; }) && has_data) {
    return arrow::Status::OK();
  }

  // Calculate how much memory we can use for new data
  int64_t available_memory = buffer_size_ - memory_used_;
  if (available_memory <= 0) {
    // If memory is full but we need data, we have a problem
    // Try to free some memory by processing existing batches
    return arrow::Status::OK();
  }

  // Fill in column groups if we have enough buffer size - use min heap for row alignment
  // Similar to packed reader's row offset min heap approach
  int64_t planned_memory = 0;
  RowOffsetMinHeap sorted_offsets;
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (!cg_states_[i].exhausted) {
      sorted_offsets.emplace(i, cg_states_[i].row_offset);
    }
  }

  while (!sorted_offsets.empty() && planned_memory + memory_used_ < buffer_size_) {
    size_t cg_index = sorted_offsets.top().first;

    // Check if we can add another batch from this column group
    std::shared_ptr<arrow::RecordBatch> next_batch;
    ARROW_RETURN_NOT_OK(batch_readers_[cg_index]->ReadNext(&next_batch));

    if (!next_batch) {
      // No more data from this column group
      cg_states_[cg_index].exhausted = true;
      sorted_offsets.pop();
      continue;
    }

    // Estimate memory for this batch
    int64_t batch_memory = next_batch->num_rows() * next_batch->num_columns() * 8;

    // Add the batch (we already read it, so we must keep it)
    batch_queues_[cg_index].push(next_batch);
    cg_states_[cg_index].current_chunk++;
    cg_states_[cg_index].memory_usage += batch_memory;
    cg_states_[cg_index].row_offset += next_batch->num_rows();
    planned_memory += batch_memory;

    // Update the min heap with new offset
    sorted_offsets.pop();
    sorted_offsets.emplace(cg_index, cg_states_[cg_index].row_offset);

    if (planned_memory + memory_used_ > buffer_size_) {
      // Would exceed memory limit after adding this batch - stop reading more
      break;
    }
  }

  // Update memory usage
  memory_used_ += planned_memory;

  // Set row limit based on minimum row offset to maintain alignment
  if (!sorted_offsets.empty()) {
    row_limit_ = sorted_offsets.top().second;
  } else {
    // All column groups are exhausted
    finished_ = true;
  }

  return arrow::Status::OK();
}

arrow::Status ColumnGroupRecordBatchReader::read_chunks_from_column_group(int cg_index,
                                                                          std::vector<int64_t> chunk_indices) {
  auto& state = cg_states_[cg_index];

  // Use the cached batch reader for this column group
  auto& batch_reader = batch_readers_[cg_index];

  // Read the next available batch (format reader handles chunking internally)
  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));

  if (!batch) {
    // No more data available from this column group - mark as exhausted
    cg_states_[cg_index].exhausted = true;
    return arrow::Status::OK();  // This is normal completion, not an error
  }

  // Add batch to queue
  batch_queues_[cg_index].push(batch);

  // Update state
  state.current_chunk++;
  state.row_offset += batch->num_rows();
  state.memory_usage += batch->num_rows() * batch->num_columns() * 8;  // Rough estimate
  memory_used_ += state.memory_usage;

  return arrow::Status::OK();
}

arrow::Status ColumnGroupRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  *out = nullptr;

  if (finished_) {
    return arrow::Status::OK();
  }

  // Return buffered batches one by one
  if (current_batch_index_ < buffered_batches_.size()) {
    *out = buffered_batches_[current_batch_index_];
    current_batch_index_++;
    return arrow::Status::OK();
  }

  // If we've served all buffered batches, we're done
  finished_ = true;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupRecordBatchReader::combine_batches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
  if (batches.empty()) {
    return arrow::Status::Invalid("No batches to combine");
  }

  // Find the minimum number of rows across all non-null batches
  int64_t min_rows = std::numeric_limits<int64_t>::max();
  for (const auto& batch : batches) {
    if (batch != nullptr) {
      min_rows = std::min(min_rows, batch->num_rows());
    }
  }

  if (min_rows == std::numeric_limits<int64_t>::max()) {
    min_rows = 0;
  }

  // Don't artificially limit batch sizes - use what format readers provide

  // Collect arrays from all batches
  std::vector<std::shared_ptr<arrow::Array>> combined_arrays;
  std::vector<std::shared_ptr<arrow::Field>> combined_fields;

  for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
    const auto& batch = batches[batch_idx];

    if (batch != nullptr) {
      // Add arrays from this batch
      for (int col_idx = 0; col_idx < batch->num_columns(); ++col_idx) {
        auto array = batch->column(col_idx);
        if (array->length() > min_rows) {
          // Slice array to min_rows
          array = array->Slice(0, min_rows);
        }
        combined_arrays.push_back(array);
        combined_fields.push_back(batch->schema()->field(col_idx));
      }
    } else {
      // This column group has no data - create null arrays for its columns
      auto& column_group = column_groups_[batch_idx];
      for (const auto& col_name : column_group->columns) {
        auto field = output_schema_->GetFieldByName(col_name);
        if (field != nullptr) {
          ARROW_ASSIGN_OR_RAISE(auto null_array, arrow::MakeArrayOfNull(field->type(), min_rows));
          combined_arrays.push_back(null_array);
          combined_fields.push_back(field);
        }
      }
    }
  }

  // Create the combined schema and batch
  auto combined_schema = arrow::schema(combined_fields);
  return arrow::RecordBatch::Make(combined_schema, min_rows, combined_arrays);
}

arrow::Status ColumnGroupRecordBatchReader::Close() {
  // Clean up resources
  for (auto& queue : batch_queues_) {
    while (!queue.empty()) {
      queue.pop();
    }
  }

  format_readers_.clear();
  batch_queues_.clear();
  cg_states_.clear();
  memory_used_ = 0;
  finished_ = true;

  return arrow::Status::OK();
}

int64_t ColumnGroupRecordBatchReader::get_min_row_offset() const {
  int64_t min_offset = std::numeric_limits<int64_t>::max();
  bool has_active = false;

  for (size_t i = 0; i < cg_states_.size(); ++i) {
    if (!cg_states_[i].exhausted) {
      min_offset = std::min(min_offset, cg_states_[i].row_offset);
      has_active = true;
    }
  }

  return has_active ? min_offset : 0;
}

bool ColumnGroupRecordBatchReader::all_exhausted() const {
  return std::all_of(cg_states_.begin(), cg_states_.end(),
                     [](const ColumnGroupState& state) { return state.exhausted; });
}

}  // namespace milvus_storage::api