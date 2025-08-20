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

#include "milvus-storage/format_reader.h"

#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/reader.h"

namespace milvus_storage::api {

// ==================== FormatReaderFactory Implementation ====================

std::unique_ptr<FormatReader> FormatReaderFactory::create_reader(FileFormat format,
                                                                 std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                 std::shared_ptr<Manifest> manifest,
                                                                 std::shared_ptr<arrow::Schema> schema,
                                                                 const ReadProperties& properties) {
  switch (format) {
    case FileFormat::PARQUET:
      return std::make_unique<ParquetFormatReader>(std::move(fs), std::move(manifest), std::move(schema), properties);

    case FileFormat::BINARY:
    case FileFormat::VORTEX:
    case FileFormat::LANCE:
      // TODO: Implement other format readers when needed
      throw std::runtime_error("Format not yet supported: " + std::to_string(static_cast<int>(format)));

    default:
      throw std::runtime_error("Unknown file format: " + std::to_string(static_cast<int>(format)));
  }
}

// ==================== ParquetFormatReader Implementation ====================

ParquetFormatReader::ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                         std::shared_ptr<Manifest> manifest,
                                         std::shared_ptr<arrow::Schema> schema,
                                         const ReadProperties& properties)
    : fs_(std::move(fs)),
      manifest_(std::move(manifest)),
      schema_(std::move(schema)),
      properties_(properties),
      packed_reader_(nullptr),
      initialized_(false) {}

arrow::Status ParquetFormatReader::initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                              const std::vector<std::string>& needed_columns) {
  if (initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader already initialized");
  }

  column_groups_ = column_groups;
  needed_columns_ = needed_columns;

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups provided");
  }

  // Prepare data for PackedRecordBatchReader
  std::vector<std::string> paths;

  paths.reserve(column_groups_.size());

  for (const auto& column_group : column_groups_) {
    paths.push_back(column_group->path);
  }

  // Create PackedRecordBatchReader
  try {
    packed_reader_ = std::make_unique<milvus_storage::PackedRecordBatchReader>(fs_, paths, schema_);
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Failed to create PackedRecordBatchReader: " + std::string(e.what()));
  }

  initialized_ = true;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ParquetFormatReader::get_record_batch_reader(
    const std::string& predicate, int64_t batch_size, int64_t buffer_size) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  if (!packed_reader_) {
    return arrow::Status::Invalid("PackedRecordBatchReader not available");
  }

  // PackedRecordBatchReader is already a RecordBatchReader
  return std::static_pointer_cast<arrow::RecordBatchReader>(
      std::shared_ptr<milvus_storage::PackedRecordBatchReader>(packed_reader_.get(), [](auto*) {}));
}

arrow::Result<std::shared_ptr<ChunkReader>> ParquetFormatReader::get_chunk_reader(int64_t column_group_id) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  // Find the column group with the specified ID
  std::shared_ptr<ColumnGroup> target_column_group = nullptr;
  for (const auto& column_group : column_groups_) {
    if (column_group->id == column_group_id) {
      target_column_group = column_group;
      break;
    }
  }

  if (!target_column_group) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(column_group_id) + " not found");
  }

  // Create ChunkReader for this column group
  return std::make_shared<ChunkReader>(fs_, target_column_group, needed_columns_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetFormatReader::take(const std::vector<int64_t>& row_indices,
                                                                             int64_t parallelism) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatReader not initialized");
  }

  if (!packed_reader_) {
    return arrow::Status::Invalid("PackedRecordBatchReader not available");
  }

  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices cannot be empty");
  }

  // Create a fresh PackedRecordBatchReader for this take operation
  std::vector<std::string> paths;
  for (const auto& column_group : column_groups_) {
    paths.push_back(column_group->path);
  }

  std::unique_ptr<milvus_storage::PackedRecordBatchReader> reader;
  try {
    reader = std::make_unique<milvus_storage::PackedRecordBatchReader>(fs_, paths, schema_);
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Failed to create PackedRecordBatchReader: " + std::string(e.what()));
  }

  // Read all data first
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
    if (!batch) {
      break;  // End of data
    }
    batches.push_back(batch);
  }

  if (batches.empty()) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(schema_, 0, empty_arrays);
  }

  // Combine all batches into a single table
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(batches));

  // Create an indices array for the take operation
  arrow::Int64Builder builder;
  for (int64_t index : row_indices) {
    ARROW_RETURN_NOT_OK(builder.Append(index));
  }
  ARROW_ASSIGN_OR_RAISE(auto indices_array, builder.Finish());

  // Use Arrow compute API to take specific rows
  ARROW_ASSIGN_OR_RAISE(auto taken_table, arrow::compute::Take(table, indices_array));

  // Convert result back to RecordBatch
  auto result_table = taken_table.table();
  ARROW_ASSIGN_OR_RAISE(auto combined_table, result_table->CombineChunks());

  if (combined_table->num_rows() == 0) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(combined_table->schema(), 0, empty_arrays);
  }

  // Convert to RecordBatch
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

}  // namespace milvus_storage::api