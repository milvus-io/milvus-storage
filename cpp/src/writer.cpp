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

#include <algorithm>
#include <regex>
#include <sstream>
#include <memory>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>

namespace milvus_storage::api {

// ==================== Column Group Policy Implementations ====================

SingleColumnGroupPolicy::SingleColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema)
    : ColumnGroupPolicy(std::move(schema)) {}

std::vector<std::shared_ptr<ColumnGroup>> SingleColumnGroupPolicy::get_column_groups() const {
  auto column_group_builder = std::make_shared<ColumnGroupBuilder>(0);
  column_group_builder->with_format(FileFormat::PARQUET).with_columns(schema_->field_names());
  return {column_group_builder->build()};
}

SchemaBasedColumnGroupPolicy::SchemaBasedColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema,
                                                           std::vector<std::string> column_name_patterns)
    : ColumnGroupPolicy(std::move(schema)), column_name_patterns_(std::move(column_name_patterns)) {}

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
  return arrow::Status::NotImplemented("Not implemented");
}

std::vector<std::shared_ptr<ColumnGroup>> SizeBasedColumnGroupPolicy::get_column_groups() const { return {}; }

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

  // Initialize column group writers if not already done
  if (!initialized_) {
    ARROW_RETURN_NOT_OK(initialize_column_group_writers(batch));
    initialized_ = true;
  }

  // Distribute batch to column group writers
  ARROW_RETURN_NOT_OK(distribute_batch(batch));

  // Update statistics
  stats_.rows_written += batch->num_rows();
  stats_.batches_written++;

  return arrow::Status::OK();
}

arrow::Status Writer::flush() {
  if (closed_) {
    return arrow::Status::Invalid("Cannot flush closed writer");
  }

  // Flush all column group writers
  for (auto& writer : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->flush());
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<Manifest>> Writer::close() {
  if (closed_) {
    return arrow::Status::Invalid("Writer already closed");
  }

  // Close all column group writers and collect statistics
  for (auto& writer : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->close());

    // Update column group statistics in manifest
    auto column_group = writer->column_group();
    column_group->stats.num_rows = writer->rows_written();
    column_group->stats.compressed_size = writer->bytes_written();
    // TODO: Get actual file size from filesystem
    column_group->stats.uncompressed_size = writer->bytes_written();
    column_group->stats.num_chunks = 1;  // Simplified for now

    stats_.bytes_written += writer->bytes_written();
  }

  // Add custom metadata to manifest
  for (const auto& [key, value] : custom_metadata_) {
    // TODO: Add custom metadata to manifest when interface supports it
  }

  closed_ = true;

  return manifest_;
}

arrow::Status Writer::add_metadata(const std::string& key, const std::string& value) {
  if (closed_) {
    return arrow::Status::Invalid("Cannot add metadata to closed writer");
  }

  custom_metadata_[key] = value;
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
  auto column_groups = column_group_policy_->get_column_groups();

  if (column_groups.empty()) {
    return arrow::Status::Invalid("Column group policy returned no column groups");
  }

  // Create column group writers
  column_group_writers_.clear();
  column_group_writers_.reserve(column_groups.size());

  for (auto& column_group : column_groups) {
    // Generate file path for this column group
    auto file_path = generate_column_group_path(column_group->id);
    column_group->path = file_path;

    // Create schema for this column group
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(column_group->columns.size());

    for (const auto& column_name : column_group->columns) {
      auto field = schema_->GetFieldByName(column_name);
      if (!field) {
        return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
      }
      fields.push_back(field);
    }

    auto column_group_schema = std::make_shared<arrow::Schema>(fields);

    // Create column group writer
    auto writer = std::make_unique<ColumnGroupWriter>(column_group, fs_, column_group_schema, properties_);

    ARROW_RETURN_NOT_OK(writer->initialize());

    column_group_writers_.push_back(std::move(writer));

    // Add column group to manifest
    ARROW_RETURN_NOT_OK(manifest_->add_column_group(column_group));
  }

  stats_.column_groups_count = column_groups.size();

  return arrow::Status::OK();
}

arrow::Status Writer::distribute_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
  // Write the batch to all column group writers
  // Each writer will filter the batch to include only its columns
  for (auto& writer : column_group_writers_) {
    ARROW_RETURN_NOT_OK(writer->write(batch));
  }

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
