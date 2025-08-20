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

#include "milvus-storage/format_reader.h"
#include "milvus-storage/packed/reader.h"

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
      initialized_(false) {
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

  // Initialize format readers if not already done
  ARROW_RETURN_NOT_OK(initialize_format_readers());

  // Find the format reader for this column group
  auto column_group = manifest_->get_column_group(column_group_id);
  if (column_group == nullptr) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(column_group_id) + " not found");
  }

  auto format_it = format_readers_.find(column_group->format);
  if (format_it == format_readers_.end()) {
    return arrow::Status::Invalid("No format reader available for column group format: " +
                                  std::to_string(static_cast<int>(column_group->format)));
  }

  // Use the format reader to get chunk reader
  return format_it->second->get_chunk_reader(column_group_id);
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

  // For now, we only support single format reading
  // TODO: Support mixed format reading by combining readers
  if (format_readers_.size() > 1) {
    return arrow::Status::NotImplemented("Mixed format reading not yet supported");
  }

  if (format_readers_.empty()) {
    return arrow::Status::Invalid("No format readers available");
  }

  // Use the single format reader
  auto& reader = format_readers_.begin()->second;
  return reader->get_record_batch_reader(predicate, batch_size, buffer_size);
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

  // For now, we only support single format reading
  // TODO: Support mixed format reading by combining results from different readers
  if (format_readers_.size() > 1) {
    return arrow::Status::NotImplemented("Mixed format reading not yet supported");
  }

  if (format_readers_.empty()) {
    return arrow::Status::Invalid("No format readers available");
  }

  // Use the single format reader to take rows
  auto& reader = format_readers_.begin()->second;
  return reader->take(row_indices, parallelism);
}

arrow::Status Reader::initialize_format_readers() const {
  if (initialized_) {
    return arrow::Status::OK();  // Already initialized
  }

  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for needed columns");
  }

  // Group column groups by format
  format_column_groups_.clear();
  for (auto& column_group : needed_column_groups_) {
    format_column_groups_[column_group->format].push_back(column_group);
  }

  // Create format readers for each format
  format_readers_.clear();
  for (const auto& [format, format_column_groups] : format_column_groups_) {
    try {
      auto reader = FormatReaderFactory::create_reader(format, fs_, manifest_, schema_, properties_);

      // Initialize the format reader with its column groups
      ARROW_RETURN_NOT_OK(reader->initialize(format_column_groups, needed_columns_));

      format_readers_[format] = std::move(reader);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create format reader for " + std::to_string(static_cast<int>(format)) +
                                    ": " + std::string(e.what()));
    }
  }

  initialized_ = true;
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api