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
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_aggregate.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

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

  if (column_group_->format != FileFormat::PARQUET) {
    return arrow::Status::NotImplemented("Only PARQUET format is supported for now");
  }

  // Open parquet file to get row group information
  ARROW_ASSIGN_OR_RAISE(auto input_stream, fs_->OpenInputFile(column_group_->path));

  auto parquet_reader = parquet::ParquetFileReader::Open(input_stream);

  auto metadata = parquet_reader->metadata();
  int num_row_groups = metadata->num_row_groups();

  // Pre-compute cumulative row counts for binary search
  std::vector<int64_t> cumulative_rows;
  cumulative_rows.reserve(num_row_groups + 1);
  cumulative_rows.push_back(0);  // Start with 0 rows

  for (int i = 0; i < num_row_groups; ++i) {
    auto row_group_metadata = metadata->RowGroup(i);
    int64_t row_group_size = row_group_metadata->num_rows();
    cumulative_rows.push_back(cumulative_rows.back() + row_group_size);
  }

  std::vector<int64_t> chunk_indices;
  std::set<int64_t> unique_chunks;

  // Helper function to find chunk index using binary search
  auto find_chunk_index = [&cumulative_rows](int64_t row_index) -> int64_t {
    // Binary search to find the chunk containing this row index
    auto it = std::upper_bound(cumulative_rows.begin(), cumulative_rows.end(), row_index);
    if (it == cumulative_rows.begin()) {
      return -1;  // Row index is before first chunk
    }
    return static_cast<int64_t>(std::distance(cumulative_rows.begin(), it) - 1);
  };

  // Map each row index to its corresponding row group (chunk) using binary search
  for (const auto& row_index : row_indices) {
    int64_t target_chunk = find_chunk_index(row_index);

    if (target_chunk == -1) {
      return arrow::Status::Invalid("Row index " + std::to_string(row_index) + " is out of range");
    }

    if (unique_chunks.find(target_chunk) == unique_chunks.end()) {
      chunk_indices.push_back(target_chunk);
      unique_chunks.insert(target_chunk);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  if (column_group_->format != FileFormat::PARQUET) {
    return arrow::Status::NotImplemented("Only PARQUET format is supported");
  }

  // Open parquet file
  ARROW_ASSIGN_OR_RAISE(auto input_stream, fs_->OpenInputFile(column_group_->path));

  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input_stream, arrow::default_memory_pool(), &arrow_reader));

  // Get the schema and validate chunk index
  auto parquet_metadata = arrow_reader->parquet_reader()->metadata();
  if (chunk_index >= parquet_metadata->num_row_groups()) {
    return arrow::Status::Invalid("Chunk index " + std::to_string(chunk_index) + " is out of range. File has " +
                                  std::to_string(parquet_metadata->num_row_groups()) + " row groups");
  }

  // Determine which columns to read
  std::vector<int> column_indices;
  std::shared_ptr<arrow::Schema> parquet_schema;
  ARROW_RETURN_NOT_OK(arrow_reader->GetSchema(&parquet_schema));

  if (needed_columns_.empty()) {
    // Read all columns
    for (const auto& column_name : column_group_->columns) {
      auto field_index = parquet_schema->GetFieldIndex(column_name);
      if (field_index != -1) {
        column_indices.push_back(field_index);
      }
    }
  } else {
    // Read only needed columns that are in this column group
    for (const auto& column_name : needed_columns_) {
      if (std::find(column_group_->columns.begin(), column_group_->columns.end(), column_name) !=
          column_group_->columns.end()) {
        auto field_index = parquet_schema->GetFieldIndex(column_name);
        if (field_index != -1) {
          column_indices.push_back(field_index);
        }
      }
    }
  }

  if (column_indices.empty()) {
    return arrow::Status::Invalid("No valid columns found to read");
  }

  // Read the specific row group (chunk)
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(arrow_reader->ReadRowGroup(chunk_index, column_indices, &table));

  // Convert table to record batch
  if (table->num_rows() == 0) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int col_idx : column_indices) {
      fields.push_back(parquet_schema->field(col_idx));
    }
    auto schema = std::make_shared<arrow::Schema>(fields);
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(schema, 0, empty_arrays);
  }

  ARROW_ASSIGN_OR_RAISE(auto combined_table, table->CombineChunks());

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (int i = 0; i < combined_table->num_columns(); ++i) {
    auto column = combined_table->column(i);
    if (column->num_chunks() > 0) {
      arrays.push_back(column->chunk(0));
    } else {
      return arrow::Status::Invalid("Column has no chunks");
    }
  }

  return arrow::RecordBatch::Make(combined_table->schema(), arrays[0]->length(), arrays);
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

  if (column_group_->format != FileFormat::PARQUET) {
    return arrow::Status::NotImplemented("Only PARQUET format is supported");
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> results;
  results.reserve(chunk_indices.size());

  // For now, implement sequential reading
  // TODO: Add parallel processing when parallelism > 1
  for (const auto& chunk_index : chunk_indices) {
    ARROW_ASSIGN_OR_RAISE(auto batch, get_chunk(chunk_index));
    results.push_back(batch);
  }

  return results;
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
      properties_(std::move(properties)),
      initialized_(false),
      memory_used_(0),
      memory_limit_(properties_.buffer_size <= 0 ? INT64_MAX : properties_.buffer_size),
      row_limit_(0),
      absolute_row_position_(0) {
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

  // Initialize packed reader components
  auto status = initializePackedReaderComponents();
  if (!status.ok()) {
    throw std::runtime_error("Failed to initialize packed reader components: " + status.ToString());
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
      needed_column_groups_.push_back(column_group);
      visited_column_groups.insert(column_group->id);
    }
  }
}

arrow::Result<std::shared_ptr<ChunkReader>> Reader::get_chunk_reader(int64_t column_group_id) const {
  if (column_group_id < 0) {
    return arrow::Status::Invalid("Column group ID cannot be negative: " + std::to_string(column_group_id));
  }

  // Initialize format readers if not already done
  ARROW_RETURN_NOT_OK(initialize_format_readers());

  // Find the column group reader for this specific column group
  auto reader_it = column_group_readers_.find(column_group_id);
  if (reader_it == column_group_readers_.end()) {
    return arrow::Status::Invalid("No reader available for column group with ID " + std::to_string(column_group_id));
  }

  // Use the format reader to get chunk reader
  return reader_it->second->get_chunk_reader();
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

  // Initialize format readers if not already done
  ARROW_RETURN_NOT_OK(initialize_format_readers());

  if (column_group_readers_.empty()) {
    return arrow::Status::Invalid("No column group readers available");
  }

  // Create a unified record batch reader that handles row alignment across multiple column groups
  // using packed reader patterns for optimal memory management and I/O
  return std::make_shared<RowAlignedRecordBatchReader>(column_group_readers_, schema_, needed_columns_, buffer_size);
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

  // Initialize format readers if not already done
  ARROW_RETURN_NOT_OK(initialize_format_readers());

  if (column_group_readers_.empty()) {
    return arrow::Status::Invalid("No column group readers available");
  }

  // Take rows from each column group and align them
  return take_aligned_rows(row_indices, parallelism);
}

arrow::Status Reader::initialize_format_readers() const {
  if (initialized_) {
    return arrow::Status::OK();  // Already initialized
  }

  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for needed columns");
  }

  // Initialize column group states for memory management
  column_group_states_.clear();
  column_group_states_.resize(needed_column_groups_.size(), milvus_storage::ColumnGroupState(0, -1, 0));

  // Initialize table buffers
  tables_.clear();
  tables_.resize(needed_column_groups_.size());

  // Create individual format readers for each column group
  column_group_readers_.clear();
  for (auto& column_group : needed_column_groups_) {
    try {
      // Create schema with only the columns for this column group
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (const auto& column_name : column_group->columns) {
        auto field = schema_->GetFieldByName(column_name);
        if (!field) {
          return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
        }
        fields.push_back(field);
      }
      auto column_group_schema = arrow::schema(fields);

      auto reader =
          FormatReaderFactory::create_reader(column_group->format, fs_, column_group, column_group_schema, properties_);

      // Initialize the format reader with this column group and its specific columns only
      std::vector<std::string> column_group_columns;
      // Only pass columns that belong to this specific column group
      for (const auto& column_name : needed_columns_) {
        if (std::find(column_group->columns.begin(), column_group->columns.end(), column_name) !=
            column_group->columns.end()) {
          column_group_columns.push_back(column_name);
        }
      }
      ARROW_RETURN_NOT_OK(reader->initialize(column_group, column_group_columns));

      column_group_readers_[column_group->id] = std::move(reader);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create format reader for column group " +
                                    std::to_string(column_group->id) + ": " + std::string(e.what()));
    }
  }

  initialized_ = true;
  return arrow::Status::OK();
}

arrow::Status Reader::initializePackedReaderComponents() const {
  // Initialize chunk manager - will be properly initialized when needed
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::take_aligned_rows(const std::vector<int64_t>& row_indices,
                                                                             int64_t parallelism) const {
  // Create a map to hold results from each column group
  std::map<int64_t, std::shared_ptr<arrow::RecordBatch>> column_group_results;

  // Read from each column group independently
  for (const auto& [column_group_id, reader] : column_group_readers_) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->take(row_indices, parallelism));
    column_group_results[column_group_id] = batch;
  }

  // Combine results from all column groups, maintaining column order from schema_
  return combine_column_group_batches(column_group_results, row_indices.size());
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::combine_column_group_batches(
    const std::map<int64_t, std::shared_ptr<arrow::RecordBatch>>& column_group_results, int64_t expected_rows) const {
  // Build arrays for final result in schema order
  std::vector<std::shared_ptr<arrow::Array>> final_arrays;
  std::vector<std::shared_ptr<arrow::Field>> final_fields;

  for (const auto& schema_field : schema_->fields()) {
    const std::string& field_name = schema_field->name();
    bool found = false;

    // Find which column group contains this field
    for (const auto& column_group : needed_column_groups_) {
      auto column_it = std::find(column_group->columns.begin(), column_group->columns.end(), field_name);
      if (column_it != column_group->columns.end()) {
        // Find the batch for this column group
        auto batch_it = column_group_results.find(column_group->id);
        if (batch_it != column_group_results.end()) {
          auto batch = batch_it->second;

          // Find the column in this batch
          int field_index = batch->schema()->GetFieldIndex(field_name);
          if (field_index >= 0) {
            final_arrays.push_back(batch->column(field_index));
            final_fields.push_back(schema_field);
            found = true;
            break;
          }
        }
      }
    }

    // If field not found in any column group, create null array
    if (!found) {
      auto null_array = arrow::MakeArrayOfNull(schema_field->type(), expected_rows);
      if (!null_array.ok()) {
        return null_array.status();
      }
      final_arrays.push_back(null_array.ValueOrDie());
      final_fields.push_back(schema_field);
    }
  }

  // Create final schema and batch
  auto final_schema = arrow::schema(final_fields);
  return arrow::RecordBatch::Make(final_schema, expected_rows, final_arrays);
}

arrow::Status Reader::advanceBuffer() const {
  // This method implements the packed reader's buffer advancement logic
  // which manages memory usage and row alignment across column groups

  if (!initialized_) {
    ARROW_RETURN_NOT_OK(initialize_format_readers());
  }

  std::vector<std::vector<int>> rgs_to_read(needed_column_groups_.size());
  size_t plan_buffer_size = 0;

  // Advance row groups for column groups that need more data
  for (size_t i = 0; i < needed_column_groups_.size(); ++i) {
    if (column_group_states_[i].row_offset > row_limit_) {
      continue;
    }

    memory_used_ -= std::max(static_cast<size_t>(0), static_cast<size_t>(column_group_states_[i].memory_size));
    column_group_states_[i].resetMemorySize();

    // Advance to next row group
    int rg = column_group_states_[i].row_group_offset + 1;
    // Note: In real implementation, we'd need to check available row groups
    // For now, this is a simplified version
    if (rg < 10) {  // Simplified condition - should check actual metadata
      rgs_to_read[i].push_back(rg);
      column_group_states_[i].setRowGroupOffset(rg);
      // Update memory size and row offset based on actual data
      plan_buffer_size += 1024 * 1024;  // Simplified - should be actual size
      column_group_states_[i].addMemorySize(1024 * 1024);
      column_group_states_[i].addRowOffset(1000);  // Simplified - should be actual row count
    }
  }

  memory_used_ += plan_buffer_size;

  // Update row limit based on current state
  if (!column_group_states_.empty()) {
    row_limit_ = column_group_states_[0].row_offset;
    for (const auto& state : column_group_states_) {
      row_limit_ = std::min(row_limit_, state.row_offset);
    }
  }

  return arrow::Status::OK();
}

// ==================== RowAlignedRecordBatchReader Implementation ====================

RowAlignedRecordBatchReader::RowAlignedRecordBatchReader(
    const std::map<int64_t, std::unique_ptr<FormatReader>>& column_group_readers,
    std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::string>& needed_columns,
    int64_t buffer_size)
    : schema_(std::move(schema)),
      needed_columns_(needed_columns),
      memory_used_(0),
      memory_limit_(buffer_size <= 0 ? INT64_MAX : buffer_size),
      row_limit_(0),
      absolute_row_position_(0),
      read_count_(0),
      closed_(false),
      initialized_(false) {
  // Create individual batch readers from format readers
  for (const auto& [column_group_id, reader] : column_group_readers) {
    auto batch_reader_result = reader->get_record_batch_reader("", 1024, buffer_size);
    if (batch_reader_result.ok()) {
      column_group_batch_readers_[column_group_id] = batch_reader_result.ValueOrDie();
    }
  }
}

std::shared_ptr<arrow::Schema> RowAlignedRecordBatchReader::schema() const { return schema_; }

arrow::Status RowAlignedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  if (closed_) {
    *batch = nullptr;
    return arrow::Status::OK();
  }

  // Simple case: single column group reader - use directly
  if (column_group_batch_readers_.size() == 1) {
    auto& single_reader = column_group_batch_readers_.begin()->second;
    return single_reader->ReadNext(batch);
  }

  // Multi-column group case with row alignment
  std::map<int64_t, std::shared_ptr<arrow::RecordBatch>> column_group_batches;
  int64_t min_rows = INT64_MAX;
  bool any_data = false;

  // Use min heap approach to prioritize reading from column groups with lower memory usage
  RowOffsetMinHeap memory_heap;
  for (const auto& [column_group_id, reader] : column_group_batch_readers_) {
    memory_heap.emplace(column_group_id, memory_used_);  // Simplified: use current memory as priority
  }

  // Read from each column group, prioritizing by min heap order for optimal I/O
  while (!memory_heap.empty() && column_group_batches.size() < column_group_batch_readers_.size()) {
    auto [column_group_id, _] = memory_heap.top();
    memory_heap.pop();

    auto reader_it = column_group_batch_readers_.find(column_group_id);
    if (reader_it != column_group_batch_readers_.end()) {
      std::shared_ptr<arrow::RecordBatch> group_batch;
      ARROW_RETURN_NOT_OK(reader_it->second->ReadNext(&group_batch));

      if (group_batch) {
        column_group_batches[column_group_id] = group_batch;
        min_rows = std::min(min_rows, group_batch->num_rows());
        any_data = true;
      }
    }
  }

  if (!any_data) {
    *batch = nullptr;
    return arrow::Status::OK();
  }

  // Row alignment: ensure all batches have the same number of rows (minimum)
  for (auto& [column_group_id, group_batch] : column_group_batches) {
    if (group_batch && group_batch->num_rows() > min_rows) {
      // Slice the batch to have consistent row count for alignment
      group_batch = group_batch->Slice(0, min_rows);
    }
  }

  // Combine batches from all column groups, maintaining schema order
  std::vector<std::shared_ptr<arrow::Array>> final_arrays;
  std::vector<std::shared_ptr<arrow::Field>> final_fields;

  for (const auto& schema_field : schema_->fields()) {
    const std::string& field_name = schema_field->name();
    bool found = false;

    // Find which column group batch contains this field
    for (const auto& [column_group_id, group_batch] : column_group_batches) {
      if (group_batch) {
        int field_index = group_batch->schema()->GetFieldIndex(field_name);
        if (field_index >= 0) {
          final_arrays.push_back(group_batch->column(field_index));
          final_fields.push_back(schema_field);
          found = true;
          break;
        }
      }
    }

    // If field not found in any column group, create null array for row alignment
    if (!found) {
      auto null_array = arrow::MakeArrayOfNull(schema_field->type(), min_rows);
      if (!null_array.ok()) {
        return null_array.status();
      }
      final_arrays.push_back(null_array.ValueOrDie());
      final_fields.push_back(schema_field);
    }
  }

  // Update memory tracking (simplified)
  memory_used_ += min_rows * final_arrays.size() * 8;  // Rough estimate
  read_count_++;

  // Create final aligned batch
  auto final_schema = arrow::schema(final_fields);
  *batch = arrow::RecordBatch::Make(final_schema, min_rows, final_arrays);

  return arrow::Status::OK();
}

arrow::Status RowAlignedRecordBatchReader::Close() {
  if (closed_) {
    return arrow::Status::OK();
  }

  // Close all column group batch readers
  for (const auto& [column_group_id, batch_reader] : column_group_batch_readers_) {
    ARROW_RETURN_NOT_OK(batch_reader->Close());
  }

  // Clean up memory tracking
  read_count_ = 0;
  memory_used_ = 0;
  closed_ = true;

  return arrow::Status::OK();
}

// Simplified placeholder methods to satisfy the interface
arrow::Status RowAlignedRecordBatchReader::initialize() const { return arrow::Status::OK(); }

arrow::Status RowAlignedRecordBatchReader::advanceBuffer() const { return arrow::Status::OK(); }

int64_t RowAlignedRecordBatchReader::getNextRowGroupSize(int64_t column_group_index) const {
  return 1024 * 1024;  // 1MB default
}

arrow::Status RowAlignedRecordBatchReader::readRowGroupsForColumnGroup(int64_t column_group_index,
                                                                       const std::vector<int>& row_groups) const {
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api