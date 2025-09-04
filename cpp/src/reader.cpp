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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/status.h"
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

  // TODO: Implement row-to-chunk mapping for column groups
  // This should map global row indices to local chunk indices within this column group
  // considering the column group's row boundaries and chunk organization
  return arrow::Status::NotImplemented("Row-to-chunk mapping not yet implemented for ChunkReader");
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  // TODO: Implement single chunk reading for column groups
  // This should read a specific chunk (row group) from the column group's storage files
  // and return the data as an Arrow RecordBatch
  return arrow::Status::NotImplemented("Single chunk reading not yet implemented for ChunkReader");
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

  // TODO: Implement multi-chunk reading for column groups with parallel support
  // This should efficiently read multiple chunks, potentially in parallel,
  // and return them as a vector of RecordBatches
  return arrow::Status::NotImplemented("Multi-chunk reading not yet implemented for ChunkReader");
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
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
      needed_columns_.emplace_back(schema_->field(i)->name());
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
    return std::make_shared<ChunkReader>(fs_, column_group, needed_columns_);
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

  // Create and return a PackedRecordBatchReader for sequential scanning
  // The packed reader handles coordination across multiple column group files
  // and provides efficient streaming access to the entire dataset

  // TODO: Implement predicate pushdown for server-side filtering
  // TODO: Implement batch_size parameter to control memory usage per batch
  // TODO: Implement column projection to read only needed_columns_
  // TODO: Apply encryption properties from properties_ if configured

  return std::make_shared<PackedRecordBatchReader>(fs_, paths, schema_, buffer_size);
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

  // TODO: Implement random access reading by row indices
  // This should:
  // 1. Map row indices to their corresponding column groups and chunks
  // 2. Group indices by column group for efficient batch reading
  // 3. Read the required chunks from each column group (potentially in parallel)
  // 4. Extract the specific rows from each chunk
  // 5. Reconstruct the final RecordBatch maintaining original row order
  // 6. Handle cross-column-group row assembly for complete records

  return arrow::Status::NotImplemented("Random access by row indices not yet implemented");
}

}  // namespace milvus_storage::api