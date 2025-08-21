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

#include "milvus-storage/format/format_writer.h"

#include <arrow/io/api.h>
#include <arrow/ipc/writer.h>

namespace milvus_storage::api {

// ==================== BinaryFormatWriter Implementation ====================

BinaryFormatWriter::BinaryFormatWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       std::string base_path,
                                       std::shared_ptr<arrow::Schema> schema,
                                       const WriteProperties& properties)
    : fs_(std::move(fs)),
      base_path_(std::move(base_path)),
      schema_(std::move(schema)),
      properties_(properties),
      stats_{},
      initialized_(false),
      closed_(false) {}

arrow::Status BinaryFormatWriter::initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                             const std::map<std::string, std::string>& custom_metadata) {
  if (initialized_) {
    return arrow::Status::Invalid("BinaryFormatWriter already initialized");
  }

  column_groups_ = column_groups;
  custom_metadata_ = custom_metadata;

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups provided");
  }

  // Store column group information for lazy writer creation
  for (const auto& column_group : column_groups_) {
    // Create schema with only the columns for this group
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<int> column_indices;

    for (const auto& column_name : column_group->columns) {
      int field_index = schema_->GetFieldIndex(column_name);
      if (field_index == -1) {
        return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
      }
      fields.push_back(schema_->field(field_index));
      column_indices.push_back(field_index);
    }

    column_group_indices_[column_group->id] = column_indices;
  }

  stats_.column_groups_count = column_groups_.size();
  initialized_ = true;

  return arrow::Status::OK();
}

arrow::Status BinaryFormatWriter::write(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatWriter not initialized");
  }

  if (closed_) {
    return arrow::Status::Invalid("BinaryFormatWriter already closed");
  }

  // Create writers if not already created (lazy initialization for metadata support)
  if (writers_.empty()) {
    for (const auto& column_group : column_groups_) {
      // Create output stream for this column group
      auto output_stream_result = fs_->OpenOutputStream(column_group->path);
      if (!output_stream_result.ok()) {
        return arrow::Status::IOError("Failed to open output stream for " + column_group->path + ": " +
                                      output_stream_result.status().ToString());
      }
      auto output_stream = std::move(output_stream_result).ValueOrDie();

      // Create schema with only the columns for this group
      std::vector<std::shared_ptr<arrow::Field>> fields;
      auto indices_it = column_group_indices_.find(column_group->id);
      if (indices_it == column_group_indices_.end()) {
        return arrow::Status::Invalid("Column indices not found for column group " + std::to_string(column_group->id));
      }

      for (int index : indices_it->second) {
        fields.push_back(schema_->field(index));
      }
      auto group_schema = arrow::schema(fields);

      // Create custom metadata from the stored metadata
      std::shared_ptr<arrow::KeyValueMetadata> file_metadata = nullptr;
      if (!custom_metadata_.empty()) {
        std::vector<std::string> keys, values;
        for (const auto& [key, value] : custom_metadata_) {
          keys.push_back(key);
          values.push_back(value);
        }
        file_metadata = std::make_shared<arrow::KeyValueMetadata>(keys, values);
      }

      // Create Arrow IPC writer (file format for random access support) with custom metadata
      auto writer_result = arrow::ipc::MakeFileWriter(output_stream, group_schema,
                                                      arrow::ipc::IpcWriteOptions::Defaults(), file_metadata);
      if (!writer_result.ok()) {
        return arrow::Status::IOError("Failed to create IPC writer for " + column_group->path + ": " +
                                      writer_result.status().ToString());
      }

      writers_[column_group->id] = std::move(writer_result).ValueOrDie();
      output_streams_[column_group->id] = output_stream;
    }
  }

  // Write to each column group
  for (const auto& column_group : column_groups_) {
    auto writer_it = writers_.find(column_group->id);
    if (writer_it == writers_.end()) {
      return arrow::Status::Invalid("Writer not found for column group " + std::to_string(column_group->id));
    }

    auto indices_it = column_group_indices_.find(column_group->id);
    if (indices_it == column_group_indices_.end()) {
      return arrow::Status::Invalid("Column indices not found for column group " + std::to_string(column_group->id));
    }

    // Extract columns for this group
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int index : indices_it->second) {
      columns.push_back(batch->column(index));
    }

    // Create schema for this group
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int index : indices_it->second) {
      fields.push_back(schema_->field(index));
    }
    auto group_schema = arrow::schema(fields);

    // Create record batch for this group
    auto group_batch = arrow::RecordBatch::Make(group_schema, batch->num_rows(), columns);

    // Write the batch
    auto status = writer_it->second->WriteRecordBatch(*group_batch);
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to write record batch to column group " + std::to_string(column_group->id) +
                                    ": " + status.ToString());
    }
  }

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status BinaryFormatWriter::flush() {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatWriter not initialized");
  }

  // Flush all output streams
  for (const auto& [id, stream] : output_streams_) {
    auto status = stream->Flush();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to flush output stream for column group " + std::to_string(id) + ": " +
                                    status.ToString());
    }
  }

  return arrow::Status::OK();
}

arrow::Status BinaryFormatWriter::close() {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatWriter not initialized");
  }

  if (closed_) {
    return arrow::Status::OK();  // Already closed
  }

  // Close all writers
  for (auto& [id, writer] : writers_) {
    auto status = writer->Close();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to close writer for column group " + std::to_string(id) + ": " +
                                    status.ToString());
    }
  }

  // Close all output streams
  for (auto& [id, stream] : output_streams_) {
    auto status = stream->Close();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to close output stream for column group " + std::to_string(id) + ": " +
                                    status.ToString());
    }
  }

  // Update statistics from Arrow writers and column group statistics
  stats_.bytes_written = 0;
  for (const auto& column_group : column_groups_) {
    auto writer_it = writers_.find(column_group->id);
    if (writer_it != writers_.end()) {
      // Use Arrow writer stats for accurate byte counting
      auto writer_stats = writer_it->second->stats();
      int64_t writer_bytes = writer_stats.total_serialized_body_size;

      // Prefer Arrow writer stats for accurate byte counting
      if (writer_bytes > 0) {
        column_group->stats.compressed_size = writer_bytes;
        column_group->stats.uncompressed_size = writer_stats.total_raw_body_size;
        stats_.bytes_written += writer_bytes;
      } else {
        // Fallback to file size if writer stats not available
        auto file_info_result = fs_->GetFileInfo(column_group->path);
        if (file_info_result.ok() && file_info_result.ValueOrDie().size() >= 0) {
          int64_t file_size = file_info_result.ValueOrDie().size();
          column_group->stats.compressed_size = file_size;
          column_group->stats.uncompressed_size = file_size;
          stats_.bytes_written += file_size;
        }
      }
    }

    column_group->stats.num_rows = stats_.rows_written;
    column_group->stats.num_chunks = 1;
  }

  closed_ = true;
  return arrow::Status::OK();
}

arrow::Status BinaryFormatWriter::add_metadata(const std::string& key, const std::string& value) {
  if (!initialized_) {
    return arrow::Status::Invalid("BinaryFormatWriter not initialized");
  }

  custom_metadata_[key] = value;
  return arrow::Status::OK();
}

Writer::WriteStats BinaryFormatWriter::get_stats() const { return stats_; }

}  // namespace milvus_storage::api