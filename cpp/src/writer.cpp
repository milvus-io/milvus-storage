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
#include <parquet/properties.h>
#include "milvus-storage/format_writer.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::api {

// ==================== Column Group Policy Implementations ====================

SingleColumnGroupPolicy::SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, FileFormat format)
    : ColumnGroupPolicy(std::move(schema), format) {}

std::vector<std::shared_ptr<ColumnGroup>> SingleColumnGroupPolicy::get_column_groups() const {
  auto column_group_builder = std::make_shared<ColumnGroupBuilder>(0);
  column_group_builder->with_format(default_format_).with_columns(schema_->field_names());
  return {column_group_builder->build()};
}

SchemaBasedColumnGroupPolicy::SchemaBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                                           const std::vector<std::string>& column_name_patterns,
                                                           FileFormat format)
    : ColumnGroupPolicy(std::move(schema), format), column_name_patterns_(column_name_patterns) {}

std::vector<std::shared_ptr<ColumnGroup>> SchemaBasedColumnGroupPolicy::get_column_groups() const {
  std::shared_ptr<ColumnGroupBuilder> column_groups_builders[column_name_patterns_.size() + 1];

  for (int i = 0; i < schema_->num_fields(); ++i) {
    const std::string& field_name = schema_->field(i)->name();
    bool matched = false;

    // Try to match against each pattern
    for (int j = 0; j < column_name_patterns_.size(); ++j) {
      auto pattern = column_name_patterns_[j];
      if (std::regex_match(field_name, std::regex(pattern))) {
        if (column_groups_builders[j] == nullptr) {
          // create a new column group builder
          column_groups_builders[j] = std::make_shared<ColumnGroupBuilder>(j);
          column_groups_builders[j]->with_format(default_format_);
        }
        column_groups_builders[j]->add_column(field_name);
        matched = true;
        break;
      }
    }

    // If no pattern matched, add to the default group
    if (!matched) {
      if (column_groups_builders[column_name_patterns_.size()] == nullptr) {
        // create a new column group builder for unmatched columns
        column_groups_builders[column_name_patterns_.size()] =
            std::make_shared<ColumnGroupBuilder>(column_name_patterns_.size());
        column_groups_builders[column_name_patterns_.size()]->with_format(default_format_);
      }
      column_groups_builders[column_name_patterns_.size()]->add_column(field_name);
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
    column_group_builder->with_format(default_format_).with_columns(schema_->field_names());
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
      column_group_builder->with_format(default_format_).with_columns(current_group_columns);
      column_groups.push_back(column_group_builder->build());
      current_group_columns.clear();
      current_group_columns.push_back(schema_->field(i)->name());
    }
  }

  // Add the last group if it has columns
  if (!current_group_columns.empty()) {
    auto column_group_builder = std::make_shared<ColumnGroupBuilder>(current_group_id);
    column_group_builder->with_format(default_format_).with_columns(current_group_columns);
    column_groups.push_back(column_group_builder->build());
  }

  return column_groups;
}

MixedFormatColumnGroupPolicy::MixedFormatColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                                           std::vector<ColumnGroupConfig> configs,
                                                           FileFormat default_format)
    : ColumnGroupPolicy(std::move(schema), default_format), configs_(std::move(configs)) {}

std::vector<std::shared_ptr<ColumnGroup>> MixedFormatColumnGroupPolicy::get_column_groups() const {
  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  std::map<int, std::shared_ptr<ColumnGroupBuilder>> group_builders;  // config_index -> builder
  std::vector<std::string> unmatched_columns;

  // Process each column in the schema
  for (int i = 0; i < schema_->num_fields(); ++i) {
    const std::string& column_name = schema_->field(i)->name();
    bool matched = false;

    // Try to match against each config
    for (size_t config_idx = 0; config_idx < configs_.size(); ++config_idx) {
      const auto& config = configs_[config_idx];

      // Check if column matches any pattern in this config
      for (const auto& pattern : config.column_patterns) {
        if (std::regex_match(column_name, std::regex(pattern))) {
          // Create builder if not exists
          if (group_builders.find(config_idx) == group_builders.end()) {
            group_builders[config_idx] = std::make_shared<ColumnGroupBuilder>(config_idx);
            group_builders[config_idx]->with_format(config.format);
          }

          group_builders[config_idx]->add_column(column_name);
          matched = true;
          break;
        }
      }

      if (matched)
        break;
    }

    // If no pattern matched, add to unmatched list
    if (!matched) {
      unmatched_columns.push_back(column_name);
    }
  }

  // Create column groups from builders
  for (const auto& [config_idx, builder] : group_builders) {
    column_groups.push_back(builder->build());
  }

  // Handle unmatched columns - put them in a default format group
  if (!unmatched_columns.empty()) {
    auto default_builder = std::make_shared<ColumnGroupBuilder>(configs_.size());  // Use next available ID
    default_builder->with_format(default_format_).with_columns(unmatched_columns);
    column_groups.push_back(default_builder->build());
  }

  return column_groups;
}

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
      initialized_(false) {}

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

  // Initialize format writer if not already done
  if (!initialized_) {
    ARROW_RETURN_NOT_OK(initialize_format_writer(batch));
    initialized_ = true;
  }

  // Write batch to all format writers
  for (auto& [format, writer] : format_writers_) {
    ARROW_RETURN_NOT_OK(writer->write(batch));
  }

  // Update accumulated statistics from all format writers
  stats_ = {};
  for (const auto& [format, writer] : format_writers_) {
    auto writer_stats = writer->get_stats();
    stats_.rows_written = writer_stats.rows_written;        // All writers should have same row count
    stats_.batches_written = writer_stats.batches_written;  // All writers should have same batch count
    stats_.bytes_written += writer_stats.bytes_written;
    stats_.column_groups_count += writer_stats.column_groups_count;
  }

  return arrow::Status::OK();
}

arrow::Status Writer::flush() {
  if (closed_) {
    return arrow::Status::Invalid("Cannot flush closed writer");
  }

  for (auto& [format, writer] : format_writers_) {
    ARROW_RETURN_NOT_OK(writer->flush());
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<Manifest>> Writer::close() {
  if (closed_) {
    return arrow::Status::Invalid("Writer already closed");
  }

  // Close all format writers
  for (auto& [format, writer] : format_writers_) {
    ARROW_RETURN_NOT_OK(writer->close());
  }

  // Update final accumulated statistics
  stats_ = {};
  for (const auto& [format, writer] : format_writers_) {
    auto writer_stats = writer->get_stats();
    stats_.rows_written = writer_stats.rows_written;  // All writers should have same row count
    stats_.batches_written = writer_stats.batches_written;
    stats_.bytes_written += writer_stats.bytes_written;
    stats_.column_groups_count += writer_stats.column_groups_count;
  }

  closed_ = true;
  return manifest_;
}

arrow::Status Writer::add_metadata(const std::string& key, const std::string& value) {
  if (closed_) {
    return arrow::Status::Invalid("Cannot add metadata to closed writer");
  }

  custom_metadata_[key] = value;

  // Add to all format writers if initialized
  for (auto& [format, writer] : format_writers_) {
    ARROW_RETURN_NOT_OK(writer->add_metadata(key, value));
  }

  return arrow::Status::OK();
}

Writer::WriteStats Writer::get_stats() const { return stats_; }

// ==================== Internal Helper Methods ====================

arrow::Status Writer::initialize_format_writer(const std::shared_ptr<arrow::RecordBatch>& batch) {
  // If policy requires sampling and this is the first batch, provide sample
  if (column_group_policy_->requires_sample() && stats_.batches_written == 0) {
    ARROW_RETURN_NOT_OK(column_group_policy_->sample(batch));
  }

  // Get column groups from policy
  column_groups_ = column_group_policy_->get_column_groups();

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("Column group policy returned no column groups");
  }

  // Group column groups by format
  format_column_groups_.clear();
  for (auto& column_group : column_groups_) {
    // Generate file path for this column group
    auto file_path = generate_column_group_path(column_group->id, column_group->format);
    column_group->path = file_path;

    // Add column group to manifest
    ARROW_RETURN_NOT_OK(manifest_->add_column_group(column_group));

    // Group by format
    format_column_groups_[column_group->format].push_back(column_group);
  }

  // Create format writers for each format
  format_writers_.clear();
  for (const auto& [format, format_column_groups] : format_column_groups_) {
    try {
      auto writer = FormatWriterFactory::create_writer(format, fs_, base_path_, schema_, properties_);

      // Initialize the format writer with its column groups (make a copy to avoid reference issues)
      std::vector<std::shared_ptr<ColumnGroup>> column_groups_copy = format_column_groups;
      ARROW_RETURN_NOT_OK(writer->initialize(column_groups_copy, custom_metadata_));

      format_writers_[format] = std::move(writer);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create format writer for " + std::to_string(static_cast<int>(format)) +
                                    ": " + std::string(e.what()));
    }
  }

  stats_.column_groups_count = column_groups_.size();

  return arrow::Status::OK();
}

std::string Writer::generate_column_group_path(int64_t column_group_id, FileFormat format) const {
  std::ostringstream path_stream;
  path_stream << base_path_;
  if (!base_path_.empty() && base_path_.back() != '/') {
    path_stream << "/";
  }
  path_stream << "column_group_" << column_group_id;

  // Add appropriate file extension based on format
  switch (format) {
    case FileFormat::PARQUET:
      path_stream << ".parquet";
      break;
    case FileFormat::BINARY:
      path_stream << ".bin";
      break;
    case FileFormat::VORTEX:
      path_stream << ".vortex";
      break;
    case FileFormat::LANCE:
      path_stream << ".lance";
      break;
    default:
      path_stream << ".dat";  // fallback extension
      break;
  }

  return path_stream.str();
}

}  // namespace milvus_storage::api
