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

#include "milvus-storage/writer.h"

#include <regex>
#include <sstream>
#include <memory>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"

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

// ==================== Column Group Policy Implementations ====================

SingleColumnGroupPolicy::SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema)
    : ColumnGroupPolicy(std::move(schema)) {}

std::vector<std::shared_ptr<ColumnGroup>> SingleColumnGroupPolicy::get_column_groups() const {
  auto column_group_builder = std::make_shared<ColumnGroupBuilder>(0);
  column_group_builder->with_format(FileFormat::PARQUET).with_columns(schema_->field_names());
  return {column_group_builder->build()};
}

SchemaBasedColumnGroupPolicy::SchemaBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                                           const std::vector<std::string>& column_name_patterns)
    : ColumnGroupPolicy(std::move(schema)), column_name_patterns_(column_name_patterns) {}

std::vector<std::shared_ptr<ColumnGroup>> SchemaBasedColumnGroupPolicy::get_column_groups() const {
  std::shared_ptr<ColumnGroupBuilder> column_groups_builders[column_name_patterns_.size() + 1];

  for (int i = 0; i < schema_->num_fields(); ++i) {
    for (int j = 0; j < column_name_patterns_.size(); ++j) {
      auto pattern = column_name_patterns_[j];
      if (std::regex_match(schema_->field(i)->name(), std::regex(pattern))) {
        if (column_groups_builders[j + 1] == nullptr) {
          // create a new column group builder
          column_groups_builders[j + 1] = std::make_shared<ColumnGroupBuilder>(j + 1);
        } else {
          column_groups_builders[j + 1]->add_column(schema_->field(i)->name());
        }
        break;
      } else {
        // if no pattern matches, add to the last group
        if (column_groups_builders[column_name_patterns_.size()] == nullptr) {
          // create a new column group builder
          column_groups_builders[column_name_patterns_.size()] =
              std::make_shared<ColumnGroupBuilder>(column_name_patterns_.size());
        }
        column_groups_builders[column_name_patterns_.size()]->add_column(schema_->field(i)->name());
      }
    }
  }

  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  column_groups.reserve(column_name_patterns_.size() + 1);
  for (int i = 0; i < column_name_patterns_.size() + 1; ++i) {
    if (column_groups_builders[i] != nullptr) {
      column_groups.push_back(column_groups_builders[i]->build());
    }
  }

  return column_groups;
}

arrow::Status SizeBasedColumnGroupPolicy::sample(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!batch || batch->num_rows() == 0) {
    return arrow::Status::Invalid("Sample batch cannot be null or empty");
  }

  // Calculate average column sizes based on the sample
  column_sizes_.clear();
  column_sizes_.reserve(schema_->num_fields());

  for (int i = 0; i < schema_->num_fields(); ++i) {
    auto column = batch->column(i);
    int64_t column_size = GetArrowArrayMemorySize(column);
    int64_t avg_size = batch->num_rows() > 0 ? column_size / batch->num_rows() : 0;
    column_sizes_.push_back(avg_size);
  }

  return arrow::Status::OK();
}

std::vector<std::shared_ptr<ColumnGroup>> SizeBasedColumnGroupPolicy::get_column_groups() const {
  if (column_sizes_.empty()) {
    // No sample data available, fallback to single group
    auto column_group_builder = std::make_shared<ColumnGroupBuilder>(0);
    column_group_builder->with_format(FileFormat::PARQUET).with_columns(schema_->field_names());
    return {column_group_builder->build()};
  }

  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  std::vector<std::string> current_group_columns;
  int current_group_id = 0;

  for (int i = 0; i < schema_->num_fields(); ++i) {
    // group all columns if the column size is less than max_avg_column_size_, else create a new group
    if (column_sizes_[i] < max_avg_column_size_ && current_group_columns.size() < max_columns_in_group_) {
      current_group_columns.push_back(schema_->field(i)->name());
    } else {
      // Create a new column group with current columns
      auto column_group_builder = std::make_shared<ColumnGroupBuilder>(current_group_id++);
      column_group_builder->with_format(FileFormat::PARQUET).with_columns(current_group_columns);
      column_groups.push_back(column_group_builder->build());
      current_group_columns.clear();
      current_group_columns.push_back(schema_->field(i)->name());
    }
  }

  // Add the last group if it has columns
  if (!current_group_columns.empty()) {
    auto column_group_builder = std::make_shared<ColumnGroupBuilder>(current_group_id);
    column_group_builder->with_format(FileFormat::PARQUET).with_columns(current_group_columns);
    column_groups.push_back(column_group_builder->build());
  }

  return column_groups;
}
// ==================== Internal ColumnGroupWriter Implementation ====================

class Writer::ColumnGroupWriter {
  public:
  ColumnGroupWriter(std::shared_ptr<ColumnGroup> column_group,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::shared_ptr<arrow::Schema> schema,
                    const WriteProperties& properties)
      : column_group_(std::move(column_group)),
        fs_(std::move(fs)),
        schema_(std::move(schema)),
        properties_(properties) {}

  ~ColumnGroupWriter() {
    if (!closed_) {
      // Attempt graceful cleanup - ignore any errors since we're in destructor
      auto status = close();
      (void)status;  // Suppress unused variable warning
    }
  }

  arrow::Status initialize() { return arrow::Status::NotImplemented("Not implemented"); }
  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) {
    return arrow::Status::NotImplemented("Not implemented");
  }

  arrow::Status flush() { return arrow::Status::NotImplemented("Not implemented"); }

  arrow::Status close() { return arrow::Status::NotImplemented("Not implemented"); }

  [[nodiscard]] std::shared_ptr<ColumnGroup> column_group() const { return column_group_; }
  [[nodiscard]] int64_t rows_written() const { return rows_written_; }
  [[nodiscard]] int64_t bytes_written() const { return bytes_written_; }

  private:
  std::shared_ptr<ColumnGroup> column_group_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string file_path_;
  std::shared_ptr<arrow::Schema> schema_;
  WriteProperties properties_;

  std::shared_ptr<arrow::io::OutputStream> output_stream_;
  std::unique_ptr<parquet::arrow::FileWriter> parquet_writer_;

  int64_t rows_written_ = 0;
  int64_t bytes_written_ = 0;
  bool closed_ = false;
};

// ==================== Writer Implementation ====================

Writer::Writer(std::shared_ptr<arrow::fs::FileSystem> fs,
               std::string base_path,
               std::shared_ptr<arrow::Schema> schema,
               std::unique_ptr<ColumnGroupPolicy> column_group_policy,
               const WriteProperties& properties)
    : fs_(std::move(fs)),
      base_path_(std::move(base_path)),
      schema_(std::move(schema)),
      column_group_policy_(std::move(column_group_policy)),
      properties_(properties),
      manifest_(std::make_shared<Manifest>()),
      stats_{},
      closed_(false),
      initialized_(false),
      packed_writer_(nullptr) {}

Writer::~Writer() {
  if (!closed_) {
    // Attempt graceful cleanup - ignore errors since we're in destructor
    auto result = close();
    (void)result;  // Suppress unused variable warning
  }
}

arrow::Status Writer::write(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (closed_) {
    return arrow::Status::Invalid("Cannot write to closed writer");
  }

  // Initialize packed writer if not already done
  if (!initialized_) {
    ARROW_RETURN_NOT_OK(initialize_packed_writer(batch));
    initialized_ = true;
  }

  // Write batch using packed writer
  auto status = packed_writer_->Write(batch);
  if (!status.ok()) {
    return arrow::Status::IOError("Failed to write batch: " + status.ToString());
  }

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status Writer::flush() {
  if (closed_) {
    return arrow::Status::Invalid("Cannot flush closed writer");
  }

  // PackedRecordBatchWriter doesn't have explicit flush, it manages internally
  // Just return OK for compatibility
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<Manifest>> Writer::close() {
  if (closed_) {
    return arrow::Status::Invalid("Writer already closed");
  }

  // Close packed writer if initialized
  if (packed_writer_) {
    auto status = packed_writer_->Close();
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to close packed writer: " + status.ToString());
    }
  }

  // Update manifest with column group statistics
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

  closed_ = true;
  return manifest_;
}

arrow::Status Writer::add_metadata(const std::string& key, const std::string& value) {
  if (closed_) {
    return arrow::Status::Invalid("Cannot add metadata to closed writer");
  }

  custom_metadata_[key] = value;

  // Also add to packed writer if initialized
  if (packed_writer_) {
    auto status = packed_writer_->AddUserMetadata(key, value);
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to add metadata to packed writer: " + status.ToString());
    }
  }

  return arrow::Status::OK();
}

Writer::WriteStats Writer::get_stats() const { return stats_; }

// ==================== Internal Helper Methods ====================

arrow::Status Writer::initialize_packed_writer(const std::shared_ptr<arrow::RecordBatch>& batch) {
  // If policy requires sampling and this is the first batch, provide sample
  if (column_group_policy_->requires_sample() && stats_.batches_written == 0) {
    ARROW_RETURN_NOT_OK(column_group_policy_->sample(batch));
  }

  // Get column groups from policy
  column_groups_ = column_group_policy_->get_column_groups();

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("Column group policy returned no column groups");
  }

  // Prepare data for PackedRecordBatchWriter
  std::vector<std::string> paths;
  std::vector<std::vector<int>> column_group_indices;

  paths.reserve(column_groups_.size());
  column_group_indices.reserve(column_groups_.size());

  for (auto& column_group : column_groups_) {
    // Generate file path for this column group
    auto file_path = generate_column_group_path(column_group->id);
    column_group->path = file_path;
    paths.push_back(file_path);

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

    // Add column group to manifest
    ARROW_RETURN_NOT_OK(manifest_->add_column_group(column_group));
  }

  // Create storage config from properties
  milvus_storage::StorageConfig storage_config;
  // TODO: Map WriteProperties to StorageConfig if needed

  // Convert WriteProperties to parquet::WriterProperties
  auto writer_props = convert_write_properties(properties_);

  // Create PackedRecordBatchWriter
  try {
    packed_writer_ = std::make_unique<milvus_storage::PackedRecordBatchWriter>(
        fs_, paths, schema_, storage_config, column_group_indices, properties_.buffer_size, writer_props);
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Failed to create PackedRecordBatchWriter: " + std::string(e.what()));
  }

  // Add existing custom metadata to packed writer
  for (const auto& [key, value] : custom_metadata_) {
    auto status = packed_writer_->AddUserMetadata(key, value);
    if (!status.ok()) {
      return arrow::Status::IOError("Failed to add metadata to packed writer: " + status.ToString());
    }
  }

  stats_.column_groups_count = column_groups_.size();

  return arrow::Status::OK();
}

std::string Writer::generate_column_group_path(int64_t column_group_id) const {
  std::ostringstream path_stream;
  path_stream << base_path_;
  if (!base_path_.empty() && base_path_.back() != '/') {
    path_stream << "/";
  }
  path_stream << "column_group_" << column_group_id << ".parquet";
  return path_stream.str();
}
}  // namespace milvus_storage::api
