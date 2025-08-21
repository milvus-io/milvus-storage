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

#include <set>
#include <parquet/properties.h>
#include <arrow/io/api.h>
#include <arrow/ipc/writer.h>
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::api {

// ==================== Helper Functions ====================

/**
 * @brief Converts API compression type to parquet compression type
 */
static parquet::Compression::type convert_compression_type(CompressionType compression) {
  switch (compression) {
    case CompressionType::UNCOMPRESSED:
      return parquet::Compression::UNCOMPRESSED;
    case CompressionType::SNAPPY:
      return parquet::Compression::SNAPPY;
    case CompressionType::GZIP:
      return parquet::Compression::GZIP;
    case CompressionType::LZ4:
      return parquet::Compression::LZ4;
    case CompressionType::ZSTD:
      return parquet::Compression::ZSTD;
    case CompressionType::BROTLI:
      return parquet::Compression::BROTLI;
    default:
      return parquet::Compression::ZSTD;
  }
}

/**
 * @brief Converts WriteProperties to parquet::WriterProperties
 */
static std::shared_ptr<parquet::WriterProperties> convert_write_properties(const WriteProperties& properties) {
  parquet::WriterProperties::Builder builder;

  // Set compression
  builder.compression(convert_compression_type(properties.compression));

  builder.max_row_group_length(properties.max_row_group_size);

  if (properties.compression_level >= 0) {
    builder.compression_level(properties.compression_level);
  }

  if (properties.enable_dictionary) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }
  return builder.build();
}

// ==================== ParquetFormatWriter Implementation ====================

ParquetFormatWriter::ParquetFormatWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                         std::string base_path,
                                         std::shared_ptr<arrow::Schema> schema,
                                         const WriteProperties& properties)
    : fs_(std::move(fs)),
      base_path_(std::move(base_path)),
      schema_(std::move(schema)),
      properties_(properties),
      packed_writer_(nullptr),
      stats_{},
      initialized_(false) {}

arrow::Status ParquetFormatWriter::initialize(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                              const std::map<std::string, std::string>& custom_metadata) {
  if (initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter already initialized");
  }

  column_groups_ = column_groups;

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups provided");
  }

  // Prepare data for PackedRecordBatchWriter
  std::vector<std::string> paths;
  std::vector<std::vector<int>> column_group_indices;

  paths.reserve(column_groups_.size());
  column_group_indices.reserve(column_groups_.size());

  for (const auto& column_group : column_groups_) {
    paths.push_back(column_group->path);

    // Convert column names to indices
    std::vector<int> indices;
    for (const auto& column_name : column_group->columns) {
      int field_index = schema_->GetFieldIndex(column_name);
      if (field_index == -1) {
        return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
      }
      indices.push_back(field_index);
    }
    column_group_indices.push_back(indices);
  }

  // Create storage config from properties
  milvus_storage::StorageConfig storage_config;
  // TODO: Map WriteProperties to StorageConfig if needed

  // Convert WriteProperties to parquet::WriterProperties
  auto writer_props = convert_write_properties(properties_);

  // Create filtered schema with only columns from column groups
  filtered_schema_ = milvus_storage::CreateFilteredSchemaFromColumnGroups(schema_, column_groups_);

  // Create PackedRecordBatchWriter
  try {
    packed_writer_ = std::make_unique<milvus_storage::PackedRecordBatchWriter>(
        fs_, paths, filtered_schema_, storage_config, column_group_indices, properties_.buffer_size, writer_props);
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Failed to create PackedRecordBatchWriter: " + std::string(e.what()));
  }

  // Add custom metadata to packed writer
  for (const auto& [key, value] : custom_metadata) {
    auto status = packed_writer_->AddUserMetadata(key, value);
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to add metadata to packed writer: " + status.ToString());
    }
  }

  stats_.column_groups_count = column_groups_.size();
  initialized_ = true;

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::write(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  if (!packed_writer_) {
    return arrow::Status::Invalid("PackedRecordBatchWriter not available");
  }

  // Filter the batch to only include columns specified in column groups
  std::set<std::string> column_group_columns;
  for (const auto& column_group : column_groups_) {
    for (const auto& col : column_group->columns) {
      column_group_columns.insert(col);
    }
  }

  // Create filtered batch with only the columns we need
  std::vector<std::shared_ptr<arrow::Array>> filtered_arrays;

  for (int i = 0; i < batch->num_columns(); ++i) {
    const std::string& field_name = batch->schema()->field(i)->name();
    if (column_group_columns.count(field_name) > 0) {
      filtered_arrays.push_back(batch->column(i));
    }
  }

  auto filtered_batch = arrow::RecordBatch::Make(filtered_schema_, batch->num_rows(), filtered_arrays);

  // Write batch using packed writer
  auto status = packed_writer_->Write(filtered_batch);
  if (!status.ok()) {
    return arrow::Status::IOError("Failed to write batch: " + status.ToString());
  }

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::flush() {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  // PackedRecordBatchWriter doesn't have explicit flush, it manages internally
  // Just return OK for compatibility
  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::close() {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  // Close packed writer if initialized
  if (packed_writer_) {
    auto status = packed_writer_->Close();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to close packed writer: " + status.ToString());
    }
  }

  // Update column group statistics
  for (const auto& column_group : column_groups_) {
    // Get file size from filesystem
    auto file_info_result = fs_->GetFileInfo(column_group->path);
    if (file_info_result.ok() && file_info_result.ValueOrDie().size() >= 0) {
      column_group->stats.compressed_size = file_info_result.ValueOrDie().size();
      column_group->stats.uncompressed_size = file_info_result.ValueOrDie().size();
      stats_.bytes_written += file_info_result.ValueOrDie().size();
    }

    column_group->stats.num_rows = stats_.rows_written;  // All column groups have same row count
    column_group->stats.num_chunks = 1;                  // Simplified for now
  }

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::add_metadata(const std::string& key, const std::string& value) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  if (!packed_writer_) {
    return arrow::Status::Invalid("PackedRecordBatchWriter not available");
  }

  auto status = packed_writer_->AddUserMetadata(key, value);
  if (!status.ok()) {
    return arrow::Status::IOError("Failed to add metadata to packed writer: " + status.ToString());
  }

  return arrow::Status::OK();
}

Writer::WriteStats ParquetFormatWriter::get_stats() const { return stats_; }

}  // namespace milvus_storage::api
