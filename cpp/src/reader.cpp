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
#include "milvus-storage/format/factory.h"
#include "milvus-storage/common/config.h"
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
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
      needed_columns_.emplace_back(schema_->field(i)->name());
    }
  }
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
      needed_column_groups_.emplace_back(column_group);
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
    // Use factory to create concrete chunk reader implementation
    auto chunk_reader =
        internal::api::ChunkReaderFactory::create_reader(column_group, fs_, needed_columns_, properties_);
    if (!chunk_reader) {
      return arrow::Status::Invalid("Failed to create chunk reader for column group " +
                                    std::to_string(column_group_id));
    }
    return std::move(chunk_reader);
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create ChunkReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> Reader::get_record_batch_reader(
    const std::string& /*predicate*/, int64_t batch_size, int64_t buffer_size) const {
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
    if (column_group->path.empty()) {
      return arrow::Status::Invalid("Column group " + std::to_string(column_group->id) + " has empty path");
    }
    paths.emplace_back(column_group->path);
  }

  if (paths.empty()) {
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

  try {
    auto reader = std::make_shared<PackedRecordBatchReader>(fs_, needed_column_groups_, projected_schema,
                                                            needed_columns_, properties_, batch_size, buffer_size);
    return reader;
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create PackedRecordBatchReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::take(const std::vector<int64_t>& row_indices,
                                                                int64_t parallelism) const {
  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices vector cannot be empty");
  }

  initialize_needed_column_groups();
  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for the requested columns");
  }

  std::vector<std::shared_ptr<arrow::Array>> final_arrays(schema_->num_fields());
  std::vector<std::shared_ptr<arrow::Field>> final_fields(schema_->num_fields());

  for (const auto& column_group : needed_column_groups_) {
    auto chunk_reader =
        internal::api::ChunkReaderFactory::create_reader(column_group, fs_, needed_columns_, properties_);

    ARROW_ASSIGN_OR_RAISE(auto chunk_indices, chunk_reader->get_chunk_indices(row_indices));

    std::set<int64_t> unique_chunks(chunk_indices.begin(), chunk_indices.end());
    std::vector<int64_t> chunks_to_read(unique_chunks.begin(), unique_chunks.end());
    ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_reader->get_chunks(chunks_to_read, parallelism));

    std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_map;
    for (size_t i = 0; i < chunks_to_read.size(); ++i) {
      chunk_map[chunks_to_read[i]] = chunks[i];
    }

    for (size_t col = 0; col < chunks[0]->num_columns(); ++col) {
      auto field = chunks[0]->schema()->field(col);
      if (std::find(needed_columns_.begin(), needed_columns_.end(), field->name()) == needed_columns_.end()) {
        continue;  // skip unnecessary columns
      }

      // Find the schema position for this field
      int schema_pos = -1;
      for (int i = 0; i < schema_->num_fields(); ++i) {
        if (schema_->field(i)->name() == field->name()) {
          schema_pos = i;
          break;
        }
      }

      if (schema_pos == -1) {
        continue;  // Skip if field not found in schema
      }

      // extract data for each target row
      std::vector<std::shared_ptr<arrow::Array>> row_slices;
      for (size_t i = 0; i < row_indices.size(); ++i) {
        int64_t global_row = row_indices[i];
        int64_t chunk_idx = chunk_indices[i];

        auto chunk = chunk_map[chunk_idx];

        // calculate local row number in chunk
        int64_t chunk_start = 0;
        for (int64_t j = 0; j < chunk_idx; ++j) {
          chunk_start += chunk_reader->get_chunk_row_num(j).ValueOr(0);
        }
        int64_t local_row = global_row - chunk_start;

        // extract this row
        row_slices.push_back(chunk->column(col)->Slice(local_row, 1));
      }

      if (row_slices.empty()) {
        continue;  // Skip if no valid row slices
      }

      // concatenate all row slices using RecordBatch approach
      std::vector<std::shared_ptr<arrow::RecordBatch>> slice_batches;
      for (const auto& slice : row_slices) {
        auto batch = arrow::RecordBatch::Make(arrow::schema({field}), slice->length(), {slice});
        slice_batches.push_back(batch);
      }

      ARROW_ASSIGN_OR_RAISE(auto combined_table, arrow::Table::FromRecordBatches(slice_batches));
      ARROW_ASSIGN_OR_RAISE(auto combined_batch, combined_table->CombineChunksToBatch());

      // Place in correct schema position
      final_arrays[schema_pos] = combined_batch->column(0);
      final_fields[schema_pos] = schema_->field(schema_pos);
    }
  }

  auto final_schema = arrow::schema(final_fields);
  return arrow::RecordBatch::Make(final_schema, row_indices.size(), final_arrays);
}

// ==================== PackedRecordBatchReader Implementation ====================

PackedRecordBatchReader::PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 const std::vector<std::string>& needed_columns,
                                                 const ReadProperties& properties,
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
        internal::api::ChunkReaderFactory::create_reader(column_group, fs_, needed_columns_, properties_);

    if (!chunk_reader) {
      throw std::runtime_error("Failed to create chunk reader for column group " + std::to_string(column_group->id));
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
    if (next_chunk >= column_groups_[i]->stats.num_chunks) {
      return -1;  // No more chunks
    }

    // Get actual chunk size from metadata instead of estimation
    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(next_chunk);
    if (!chunk_size_result.ok()) {
      return -1;  // Error getting chunk size
    }
    int64_t chunk_size = chunk_size_result.ValueOrDie();

    chunks_to_read[i].push_back(next_chunk);
    planned_memory += chunk_size;
    cg_states_[i].memory_usage += chunk_size;
    cg_states_[i].current_chunk = next_chunk;

    auto chunk_row_num_result = chunk_readers_[i]->get_chunk_row_num(next_chunk);
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

    // Check if we can add another chunk
    if (cg_states_[i].current_chunk + 1 >= column_groups_[i]->stats.num_chunks) {
      break;
    }

    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(cg_states_[i].current_chunk + 1);
    if (!chunk_size_result.ok()) {
      break;  // Error getting chunk size, skip this column group
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
    std::vector<std::shared_ptr<arrow::RecordBatch>> chunks;
    chunks.reserve(chunks_to_read[i].size());

    for (int64_t chunk_idx : chunks_to_read[i]) {
      ARROW_ASSIGN_OR_RAISE(auto chunk, chunk_readers_[i]->get_chunk(chunk_idx));
      chunks.push_back(chunk);
    }

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
      return arrow::Status::OK();
    }

    current_batches[i] = batch_queues_[i].front();
    if (current_batches[i]) {
      min_rows_available = std::min(min_rows_available, current_batches[i]->num_rows());
    }
  }

  if (min_rows_available == INT64_MAX || min_rows_available <= 0) {
    finished_ = true;
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
      combined_fields.push_back(field);

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
      final_arrays.push_back(combined_arrays[i]);
    }
  }

  if (final_arrays.empty()) {
    finished_ = true;
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

}  // namespace milvus_storage::api
