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

#include "milvus-storage/format/format_reader.h"

#include <set>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include "milvus-storage/reader.h"

namespace milvus_storage::api {

// ==================== BinaryFormatReader Implementation ====================

BinaryFormatReader::BinaryFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       std::shared_ptr<Manifest> manifest,
                                       std::shared_ptr<arrow::Schema> schema,
                                       const ReadProperties& properties)
    : fs_(std::move(fs)),
      manifest_(std::move(manifest)),
      schema_(std::move(schema)),
      properties_(properties),
      initialized_(false) {}

arrow::Status BinaryFormatReader::initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                             const std::vector<std::string>& needed_columns) {
  if (initialized_) {
    return arrow::Status::Invalid("BinaryFormatReader already initialized");
  }

  column_groups_ = column_groups;
  needed_columns_ = needed_columns;

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups provided");
  }

  // Create readers for each column group
  for (const auto& column_group : column_groups_) {
    // Open input stream for this column group
    auto input_stream_result = fs_->OpenInputFile(column_group->path);
    if (!input_stream_result.ok()) {
      return arrow::Status::IOError("Failed to open input stream for " + column_group->path + ": " +
                                    input_stream_result.status().ToString());
    }
    auto input_stream = std::move(input_stream_result).ValueOrDie();

    // Create Arrow IPC stream reader
    auto reader_result = arrow::ipc::RecordBatchStreamReader::Open(input_stream);
    if (!reader_result.ok()) {
      return arrow::Status::IOError("Failed to create IPC reader for " + column_group->path + ": " +
                                    reader_result.status().ToString());
    }

    readers_[column_group->id] = std::move(reader_result).ValueOrDie();
    input_streams_[column_group->id] = input_stream;
  }

  initialized_ = true;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> BinaryFormatReader::get_record_batch_reader(
    const std::string& predicate, int64_t batch_size, int64_t buffer_size) {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatReader not initialized");
  }

  // Create a custom record batch reader that combines data from all column groups
  return std::make_shared<BinaryRecordBatchReader>(readers_, column_groups_, schema_, needed_columns_);
}

arrow::Result<std::shared_ptr<ChunkReader>> BinaryFormatReader::get_chunk_reader(int64_t column_group_id) {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatReader not initialized");
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

arrow::Result<std::shared_ptr<arrow::RecordBatch>> BinaryFormatReader::take(const std::vector<int64_t>& row_indices,
                                                                            int64_t parallelism) {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatReader not initialized");
  }

  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices cannot be empty");
  }

  // Read all data from all column groups and combine
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches;

  for (const auto& column_group : column_groups_) {
    auto reader_it = readers_.find(column_group->id);
    if (reader_it == readers_.end()) {
      return arrow::Status::Invalid("Reader not found for column group " + std::to_string(column_group->id));
    }

    auto reader = reader_it->second;

    // Read all record batches from this column group (stream reader)
    std::vector<std::shared_ptr<arrow::RecordBatch>> group_batches;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      auto status = reader->ReadNext(&batch);
      if (!status.ok()) {
        return arrow::Status::IOError("Failed to read batch from column group " + std::to_string(column_group->id) +
                                      ": " + status.ToString());
      }
      if (batch == nullptr) {
        break;  // End of stream
      }
      group_batches.push_back(batch);
    }

    if (!group_batches.empty()) {
      // Combine batches from this column group
      if (group_batches.size() == 1) {
        all_batches.push_back(group_batches[0]);
      } else {
        ARROW_ASSIGN_OR_RAISE(auto combined_table, arrow::Table::FromRecordBatches(group_batches));
        ARROW_ASSIGN_OR_RAISE(auto single_chunk_table, combined_table->CombineChunks());

        std::vector<std::shared_ptr<arrow::Array>> arrays;
        for (int i = 0; i < single_chunk_table->num_columns(); ++i) {
          auto column = single_chunk_table->column(i);
          if (column->num_chunks() > 0) {
            arrays.push_back(column->chunk(0));
          }
        }

        auto combined_batch =
            arrow::RecordBatch::Make(single_chunk_table->schema(), single_chunk_table->num_rows(), arrays);
        all_batches.push_back(combined_batch);
      }
    }
  }

  if (all_batches.empty()) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(schema_, 0, empty_arrays);
  }

  // Combine all column groups horizontally
  std::shared_ptr<arrow::RecordBatch> combined_batch;
  if (all_batches.size() == 1) {
    combined_batch = all_batches[0];
  } else {
    // Merge columns from different column groups
    std::vector<std::shared_ptr<arrow::Array>> all_arrays;
    std::vector<std::shared_ptr<arrow::Field>> all_fields;

    for (const auto& batch : all_batches) {
      for (int i = 0; i < batch->num_columns(); ++i) {
        all_arrays.push_back(batch->column(i));
        all_fields.push_back(batch->schema()->field(i));
      }
    }

    auto merged_schema = arrow::schema(all_fields);
    combined_batch = arrow::RecordBatch::Make(merged_schema, all_batches[0]->num_rows(), all_arrays);
  }

  // Apply take operation
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches({combined_batch}));

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
  ARROW_ASSIGN_OR_RAISE(auto final_table, result_table->CombineChunks());

  if (final_table->num_rows() == 0) {
    // Return empty batch with correct schema
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(final_table->schema(), 0, empty_arrays);
  }

  // Convert to RecordBatch
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (int i = 0; i < final_table->num_columns(); ++i) {
    auto column = final_table->column(i);
    if (column->num_chunks() > 0) {
      arrays.push_back(column->chunk(0));
    } else {
      return arrow::Status::Invalid("Column has no chunks");
    }
  }

  return arrow::RecordBatch::Make(final_table->schema(), arrays[0]->length(), arrays);
}

// ==================== BinaryRecordBatchReader Implementation ====================

BinaryRecordBatchReader::BinaryRecordBatchReader(
    const std::unordered_map<int64_t, std::shared_ptr<arrow::ipc::RecordBatchStreamReader>>& readers,
    const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
    std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::string>& needed_columns)
    : readers_(readers),
      column_groups_(column_groups),
      schema_(std::move(schema)),
      needed_columns_(needed_columns),
      current_batch_(0),
      total_batches_(0) {
  // For stream readers, we don't know the total number of batches in advance
  // We'll read until we hit the end of the stream
  total_batches_ = -1;  // Unknown, will read until end
}

std::shared_ptr<arrow::Schema> BinaryRecordBatchReader::schema() const { return schema_; }

arrow::Status BinaryRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  // For stream readers, we don't know total_batches_ in advance
  // We'll try to read and return nullptr when we reach the end

  // Read current batch from all column groups
  std::vector<std::shared_ptr<arrow::Array>> all_arrays;
  std::vector<std::shared_ptr<arrow::Field>> all_fields;
  int64_t num_rows = 0;

  for (const auto& column_group : column_groups_) {
    auto reader_it = readers_.find(column_group->id);
    if (reader_it == readers_.end()) {
      return arrow::Status::Invalid("Reader not found for column group " + std::to_string(column_group->id));
    }

    std::shared_ptr<arrow::RecordBatch> group_batch;
    auto status = reader_it->second->ReadNext(&group_batch);
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to read batch from column group " + std::to_string(column_group->id) +
                                    ": " + status.ToString());
    }

    if (group_batch == nullptr) {
      // End of stream - return empty batch
      *batch = nullptr;
      return arrow::Status::OK();
    }
    if (num_rows == 0) {
      num_rows = group_batch->num_rows();
    } else if (num_rows != group_batch->num_rows()) {
      return arrow::Status::Invalid("Row count mismatch between column groups");
    }

    // Add columns from this group
    for (int i = 0; i < group_batch->num_columns(); ++i) {
      all_arrays.push_back(group_batch->column(i));
      all_fields.push_back(group_batch->schema()->field(i));
    }
  }

  // Create combined batch
  auto combined_schema = arrow::schema(all_fields);
  *batch = arrow::RecordBatch::Make(combined_schema, num_rows, all_arrays);

  current_batch_++;
  return arrow::Status::OK();
}

arrow::Status BinaryRecordBatchReader::Close() {
  // Nothing to close explicitly
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api
