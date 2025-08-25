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
#include <arrow/util/key_value_metadata.h>
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/parquet/file_writer.h"

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
                                         std::shared_ptr<ColumnGroup> column_group,
                                         std::shared_ptr<arrow::Schema> schema,
                                         const WriteProperties& properties)
    : fs_(std::move(fs)),
      column_group_(std::move(column_group)),
      schema_(std::move(schema)),
      properties_(properties),
      stats_{},
      initialized_(false),
      finished_(false),
      flushed_batches_(0),
      flushed_count_(0),
      flushed_rows_(0) {}

ParquetFormatWriter::~ParquetFormatWriter() = default;

arrow::Status ParquetFormatWriter::initialize(std::shared_ptr<ColumnGroup> column_group,
                                              const std::map<std::string, std::string>& custom_metadata) {
  if (initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter already initialized");
  }

  column_group_ = column_group;
  custom_metadata_ = custom_metadata;

  if (!column_group_) {
    return arrow::Status::Invalid("No column group provided");
  }

  // Convert WriteProperties to parquet writer properties
  auto writer_props = convert_write_properties(properties_);
  if (writer_props->file_encryption_properties()) {
    auto deep_copied_decryption = writer_props->file_encryption_properties()->DeepClone();
    auto builder = parquet::WriterProperties::Builder(*writer_props);
    builder.encryption(std::move(deep_copied_decryption));
    writer_props = builder.build();
  }
  if (writer_props->default_column_properties().compression() == parquet::Compression::UNCOMPRESSED) {
    auto builder = parquet::WriterProperties::Builder(*writer_props);
    builder.compression(parquet::Compression::ZSTD);
    builder.compression_level(3);
    writer_props = builder.build();
  }

  // Create storage config (simplified for now)
  milvus_storage::StorageConfig storage_config;

  // Create ParquetFileWriter using the same approach as packed ColumnGroupWriter
  file_writer_ = std::make_unique<milvus_storage::ParquetFileWriter>(schema_, fs_, column_group->path, storage_config,
                                                                     writer_props);

  // Initialize the file writer
  auto status = file_writer_->Init();
  if (!status.ok()) {
    return arrow::Status::IOError("Failed to initialize parquet file writer: " + status.ToString());
  }

  // Add custom metadata
  for (const auto& [key, value] : custom_metadata_) {
    file_writer_->AppendKVMetadata(key, value);
  }

  stats_.column_groups_count = 1;
  initialized_ = true;

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::write(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  if (finished_) {
    return arrow::Status::Invalid("Writer has been closed");
  }

  if (!batch) {
    return arrow::Status::OK();
  }

  // Buffer the batch in memory (similar to column_group_.AddRecordBatch)
  buffered_batches_.push_back(batch);
  size_t batch_memory = GetRecordBatchMemorySize(batch);
  buffered_memory_usage_.push_back(batch_memory);

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::flush() {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  if (!file_writer_) {
    return arrow::Status::Invalid("No file writer available");
  }

  if (buffered_batches_.empty()) {
    return arrow::Status::OK();
  }

  flushed_count_++;

  // Use file writer to write record batches (this will automatically split into row groups)
  auto status = file_writer_->WriteRecordBatches(buffered_batches_, buffered_memory_usage_);
  if (!status.ok()) {
    return arrow::Status::IOError("Failed to write record batches: " + status.ToString());
  }

  // Update flush statistics
  flushed_batches_ += buffered_batches_.size();
  for (const auto& batch : buffered_batches_) {
    flushed_rows_ += batch->num_rows();
  }

  // Clear buffered data
  buffered_batches_.clear();
  buffered_memory_usage_.clear();

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::close() {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  if (finished_) {
    return arrow::Status::OK();
  }

  // Flush any remaining buffered data before closing
  if (!buffered_batches_.empty()) {
    ARROW_RETURN_NOT_OK(flush());
  }

  finished_ = true;

  // Close the file writer
  if (file_writer_) {
    auto status = file_writer_->Close();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to close file writer: " + status.ToString());
    }
  }

  // Update column group statistics
  if (column_group_) {
    // Get file size from filesystem
    auto file_info_result = fs_->GetFileInfo(column_group_->path);
    if (file_info_result.ok() && file_info_result.ValueOrDie().size() >= 0) {
      column_group_->stats.compressed_size = file_info_result.ValueOrDie().size();
      column_group_->stats.uncompressed_size = file_info_result.ValueOrDie().size();
      stats_.bytes_written += file_info_result.ValueOrDie().size();
    }

    column_group_->stats.num_rows = stats_.rows_written;
    column_group_->stats.num_chunks = 1;
  }

  return arrow::Status::OK();
}

arrow::Status ParquetFormatWriter::add_metadata(const std::string& key, const std::string& value) {
  if (!initialized_) {
    return arrow::Status::Invalid("ParquetFormatWriter not initialized");
  }

  // Store metadata to be added to new files
  custom_metadata_[key] = value;

  return arrow::Status::OK();
}

Writer::WriteStats ParquetFormatWriter::get_stats() const { return stats_; }

}  // namespace milvus_storage::api
