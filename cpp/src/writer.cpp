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
#include <queue>
#include <map>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/compute/api.h>
#include <parquet/properties.h>

#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::api {

// ==================== Column Group Policy Implementations ====================

SingleColumnGroupPolicy::SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, const ColumnGroupConfig& config)
    : ColumnGroupPolicy(std::move(schema), config.format) {}

std::vector<std::shared_ptr<ColumnGroup>> SingleColumnGroupPolicy::get_column_groups() const {
  auto column_group_builder = std::make_shared<ColumnGroupBuilder>(0);
  column_group_builder->with_format(default_format_).with_columns(schema_->field_names());
  return {column_group_builder->build()};
}

SchemaBasedColumnGroupPolicy::SchemaBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                                           const std::vector<ColumnGroupConfig>& configs)
    : ColumnGroupPolicy(std::move(schema), FileFormat::PARQUET), configs_(configs) {}

std::vector<std::shared_ptr<ColumnGroup>> SchemaBasedColumnGroupPolicy::get_column_groups() const {
  std::shared_ptr<ColumnGroupBuilder> column_groups_builders[configs_.size() + 1];

  for (int i = 0; i < schema_->num_fields(); ++i) {
    const std::string& field_name = schema_->field(i)->name();
    bool matched = false;

    // Try to match against each config's patterns
    for (int j = 0; j < configs_.size(); ++j) {
      const auto& config = configs_[j];
      for (const auto& pattern : config.column_patterns) {
        if (std::regex_match(field_name, std::regex(pattern))) {
          if (column_groups_builders[j] == nullptr) {
            // create a new column group builder
            column_groups_builders[j] = std::make_shared<ColumnGroupBuilder>(j);
            column_groups_builders[j]->with_format(config.format);
          }
          column_groups_builders[j]->add_column(field_name);
          matched = true;
          break;
        }
      }
      if (matched)
        break;
    }

    // If no pattern matched, add to the default group
    if (!matched) {
      if (column_groups_builders[configs_.size()] == nullptr) {
        // create a new column group builder for unmatched columns
        column_groups_builders[configs_.size()] = std::make_shared<ColumnGroupBuilder>(configs_.size());
        column_groups_builders[configs_.size()]->with_format(default_format_);
      }
      column_groups_builders[configs_.size()]->add_column(field_name);
    }
  }

  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  column_groups.reserve(configs_.size() + 1);
  for (int i = 0; i < configs_.size() + 1; ++i) {
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
    column_group_builder->with_format(config_.format).with_columns(schema_->field_names());
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
      column_group_builder->with_format(config_.format).with_columns(current_group_columns);
      column_groups.push_back(column_group_builder->build());
      current_group_columns.clear();
      current_group_columns.push_back(schema_->field(i)->name());
    }
  }

  // Add the last group if it has columns
  if (!current_group_columns.empty()) {
    auto column_group_builder = std::make_shared<ColumnGroupBuilder>(current_group_id);
    column_group_builder->with_format(config_.format).with_columns(current_group_columns);
    column_groups.push_back(column_group_builder->build());
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
      initialized_(false),
      current_memory_usage_(0),
      buffer_size_(properties.buffer_size) {}

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

  if (!batch) {
    return arrow::Status::OK();
  }

  // Initialize column group writers if not already done
  if (!initialized_) {
    ARROW_RETURN_NOT_OK(initialize_column_group_writers(batch));
    initialized_ = true;
  }

  // Use packed memory management logic
  size_t next_batch_size = GetRecordBatchMemorySize(batch);
  ARROW_RETURN_NOT_OK(distribute_batch(batch, next_batch_size));

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status Writer::flush() {
  if (closed_) {
    return arrow::Status::Invalid("Cannot flush closed writer");
  }

  // Flush all column group writers (user triggered - flush all buffers)
  for (auto& [column_group_id, writer] : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->flush());
  }

  // Clear memory tracking since all buffers are flushed
  current_memory_usage_ = 0;
  while (!memory_heap_.empty()) {
    memory_heap_.pop();
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<Manifest>> Writer::close() {
  if (closed_) {
    return arrow::Status::Invalid("Writer already closed");
  }

  // Flush all remaining buffered data before closing
  ARROW_RETURN_NOT_OK(flush());

  // Close all column group writers
  for (auto& [column_group_id, writer] : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->close());
  }

  // Update final accumulated statistics
  stats_ = {};
  for (const auto& [column_group_id, writer] : column_group_writers_) {
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

  // Add to all column group writers if initialized
  for (auto& [column_group_id, writer] : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->add_metadata(key, value));
  }

  return arrow::Status::OK();
}

Writer::WriteStats Writer::get_stats() const { return stats_; }

// ==================== Internal Helper Methods ====================

arrow::Status Writer::initialize_column_group_writers(const std::shared_ptr<arrow::RecordBatch>& batch) {
  // If policy requires sampling and this is the first batch, provide sample
  if (column_group_policy_->requires_sample() && stats_.batches_written == 0) {
    ARROW_RETURN_NOT_OK(column_group_policy_->sample(batch));
  }

  // Get column groups from policy
  column_groups_ = column_group_policy_->get_column_groups();

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("Column group policy returned no column groups");
  }

  for (auto& column_group : column_groups_) {
    // Generate file path for this column group
    auto file_path = generate_column_group_path(column_group->id, column_group->format);
    column_group->path = file_path;

    // Add column group to manifest
    ARROW_RETURN_NOT_OK(manifest_->add_column_group(column_group));
  }

  // Create individual format writers for each column group
  column_group_writers_.clear();
  for (auto& column_group : column_groups_) {
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

      // Use FormatWriterFactory to create writer based on format
      auto writer =
          FormatWriterFactory::create_writer(column_group->format, fs_, column_group, column_group_schema, properties_);

      // Initialize the writer
      ARROW_RETURN_NOT_OK(writer->initialize(column_group, custom_metadata_));

      // Custom metadata is already added during initialization

      column_group_writers_[column_group->id] = std::move(writer);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create format writer for column group " +
                                    std::to_string(column_group->id) + ": " + std::string(e.what()));
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
    default:
      path_stream << ".dat";  // fallback extension
      break;
  }

  return path_stream.str();
}

arrow::Status Writer::distribute_batch(const std::shared_ptr<arrow::RecordBatch>& batch, size_t next_batch_size) {
  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups initialized");
  }

  // Flush column groups until there's enough room for the new batch
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + next_batch_size >= buffer_size_ && !memory_heap_.empty()) {
    auto max_group = memory_heap_.top();
    memory_heap_.pop();
    current_memory_usage_ -= max_group.second;

    // Find the specific column group writer and flush it
    auto writer_it = column_group_writers_.find(max_group.first);
    if (writer_it != column_group_writers_.end()) {
      ARROW_RETURN_NOT_OK(writer_it->second->flush());
    }
  }

  // Split the batch data directly based on column groups and write to each
  for (const auto& column_group : column_groups_) {
    // Create a batch with only the columns for this group
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;

    for (const auto& column_name : column_group->columns) {
      int field_index = schema_->GetFieldIndex(column_name);
      if (field_index >= 0 && field_index < batch->num_columns()) {
        arrays.push_back(batch->column(field_index));
        fields.push_back(schema_->field(field_index));
      }
    }

    if (!arrays.empty()) {
      auto group_schema = arrow::schema(fields);
      auto group_batch = arrow::RecordBatch::Make(group_schema, batch->num_rows(), arrays);

      // Calculate memory usage for this group's data
      size_t group_memory = GetRecordBatchMemorySize(group_batch);
      current_memory_usage_ += group_memory;
      memory_heap_.emplace(column_group->id, group_memory);

      // Write data to the column group writer
      auto writer_it = column_group_writers_.find(column_group->id);
      if (writer_it != column_group_writers_.end()) {
        ARROW_RETURN_NOT_OK(writer_it->second->write(group_batch));
      }
    }
  }

  return balanceMemoryHeap();
}

arrow::Status Writer::balanceMemoryHeap() {
  std::map<size_t, size_t> group_map;
  while (!memory_heap_.empty()) {
    auto pair = memory_heap_.top();
    memory_heap_.pop();
    group_map[pair.first] += pair.second;
  }
  for (auto& pair : group_map) {
    memory_heap_.emplace(pair.first, pair.second);
  }
  group_map.clear();
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api
