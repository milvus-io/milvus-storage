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
#include <iostream>
#include "milvus-storage/format/format.h"

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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/manifest_json.h"

namespace milvus_storage::api {

// ==================== Column Group Policy Implementations ====================

ColumnGroupPolicy::ColumnGroupPolicy(std::shared_ptr<arrow::Schema> schema, const std::string& default_format)
    : schema_(std::move(schema)), default_format_(default_format) {}

static std::vector<std::string> get_schemabase_policy_patterns(const std::shared_ptr<Manifest>& manifest) {
  std::vector<std::string> patterns;
  std::string result;
  assert(manifest);

  auto column_groups = manifest->get_column_groups();
  patterns.reserve(column_groups.size());

  for (const auto& column_group : column_groups) {
    std::string column_pattern;
    for (size_t i = 0; i < column_group->columns.size(); i++) {
      if (i > 0) {
        column_pattern += "|";
      }
      column_pattern += column_group->columns[i];
    }
    patterns.emplace_back(std::move(column_pattern));
  }

  return patterns;
}

arrow::Result<std::unique_ptr<ColumnGroupPolicy>> ColumnGroupPolicy::create_column_group_policy(
    const Properties& properties_map,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<Manifest>& existing_manifest) {
  ARROW_ASSIGN_OR_RAISE(auto policy_name, GetValue<std::string>(properties_map, PROPERTY_WRITER_POLICY));
  ARROW_ASSIGN_OR_RAISE(auto policy_format, GetValue<std::string>(properties_map, PROPERTY_FORMAT));

  if (policy_name == WRITER_POLICY_SINGLE) {
    return std::make_unique<SingleColumnGroupPolicy>(schema, policy_format);
  } else if (policy_name == WRITER_POLICY_SCHEMABASE) {
    ARROW_ASSIGN_OR_RAISE(auto patterns,
                          GetValue<std::vector<std::string>>(properties_map, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS));
    return std::make_unique<SchemaBasedColumnGroupPolicy>(schema, std::move(patterns), policy_format);
  } else if (policy_name == WRITER_POLICY_SIZEBASE) {
    // if existing manifest is provided, use its column groups as patterns
    // this is to ensure that the new data is written in the same column groups as the existing data
    if (existing_manifest) {
      auto patterns = get_schemabase_policy_patterns(existing_manifest);
      return std::make_unique<SchemaBasedColumnGroupPolicy>(schema, std::move(patterns), policy_format);
    }

    ARROW_ASSIGN_OR_RAISE(auto max_avg_column_size, GetValue<int64_t>(properties_map, PROPERTY_WRITER_SIZE_BASE_MACS));
    ARROW_ASSIGN_OR_RAISE(auto max_columns_in_group, GetValue<int64_t>(properties_map, PROPERTY_WRITER_SIZE_BASE_MCIG));
    return std::move(
        std::make_unique<SizeBasedColumnGroupPolicy>(schema, max_avg_column_size, max_columns_in_group, policy_format));
  }

  return nullptr;
}

bool SingleColumnGroupPolicy::requires_sample() const { return false; }

arrow::Status SingleColumnGroupPolicy::sample(const std::shared_ptr<arrow::RecordBatch>& batch) {
  return arrow::Status::OK();  // No sampling needed
}

std::vector<std::shared_ptr<ColumnGroup>> SingleColumnGroupPolicy::get_column_groups() const {
  auto column_group = std::make_shared<ColumnGroup>();
  column_group->columns = schema_->field_names();
  column_group->paths = {};
  column_group->format = default_format_;
  return {column_group};
}

bool SchemaBasedColumnGroupPolicy::requires_sample() const { return false; }

arrow::Status SchemaBasedColumnGroupPolicy::sample(const std::shared_ptr<arrow::RecordBatch>& batch) {
  return arrow::Status::OK();  // No sampling needed
}

std::vector<std::shared_ptr<ColumnGroup>> SchemaBasedColumnGroupPolicy::get_column_groups() const {
  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  column_groups.resize(column_name_patterns_.size() + 1);

  for (size_t i = 0; i < schema_->num_fields(); ++i) {
    const std::string& field_name = schema_->field(i)->name();
    bool matched = false;

    // Try to match against each config's patterns
    for (size_t j = 0; j < column_name_patterns_.size(); ++j) {
      const auto& pattern = column_name_patterns_[j];
      if (std::regex_match(field_name, std::regex(pattern))) {
        if (column_groups[j] == nullptr) {
          // create a new column group builder
          column_groups[j] = std::make_shared<ColumnGroup>();
          column_groups[j]->format = default_format_;
        }
        column_groups[j]->columns.push_back(field_name);
        matched = true;
        break;
      }
    }

    // If no pattern matched, add to the default group
    if (!matched) {
      if (column_groups[column_name_patterns_.size()] == nullptr) {
        // create a new column group builder for unmatched columns
        column_groups[column_name_patterns_.size()] = std::make_shared<ColumnGroup>();
        column_groups[column_name_patterns_.size()]->format = default_format_;
      }
      column_groups[column_name_patterns_.size()]->columns.push_back(field_name);
    }
  }

  // remove null column groups
  column_groups.erase(std::remove(column_groups.begin(), column_groups.end(), nullptr), column_groups.end());
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
  std::vector<std::shared_ptr<ColumnGroup>> column_groups;
  std::vector<std::string> current_group_columns;
  int current_group_id = 0;

  for (int i = 0; i < schema_->num_fields(); ++i) {
    // group all columns if the column size is less than max_avg_column_size_, else create a new group
    if (column_sizes_[i] < max_avg_column_size_ && current_group_columns.size() < max_columns_in_group_) {
      current_group_columns.push_back(schema_->field(i)->name());
    } else {
      // Create a new column group with current columns
      auto column_group = std::make_shared<ColumnGroup>();
      column_group->format = default_format_;
      column_group->columns = current_group_columns;
      column_groups.push_back(column_group);
      current_group_columns.clear();
      current_group_columns.push_back(schema_->field(i)->name());
    }
  }

  // Add the last group if it has columns
  if (!current_group_columns.empty()) {
    auto column_group = std::make_shared<ColumnGroup>();
    column_group->format = default_format_;
    column_group->columns = current_group_columns;
    column_groups.push_back(column_group);
  }

  return column_groups;
}

// ==================== WriterImpl Implementation ====================

/**
 * @brief Concrete implementation of the Writer interface
 *
 * This class provides the actual implementation for writing data to milvus
 * storage datasets using manifest-based metadata. It supports efficient batch
 * writing, column grouping policies, compression, encryption, and automatic
 * manifest generation.
 */
class WriterImpl : public Writer {
  public:
  /**
   * @brief Constructs a WriterImpl instance for a milvus storage dataset
   *
   * Initializes the writer with filesystem access, target location, schema,
   * column grouping policy, and write configuration. The writer prepares
   * column group writers based on the policy and begins accepting data.
   *
   * @param fs Shared pointer to the filesystem interface for data access
   * @param base_path Base directory path where column group files will be written
   * @param schema Arrow schema defining the logical structure of the data
   * @param column_group_policy Policy for organizing columns into groups
   * @param properties Write configuration properties including compression and encryption
   */
  WriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
             std::string base_path,
             std::shared_ptr<arrow::Schema> schema,
             std::unique_ptr<ColumnGroupPolicy> column_group_policy,
             std::shared_ptr<Manifest> existing_manifest,
             const Properties& properties)
      : fs_(std::move(fs)),
        base_path_(std::move(base_path)),
        schema_(std::move(schema)),
        column_group_policy_(std::move(column_group_policy)),
        properties_(properties),
        existing_manifest_(std::move(existing_manifest)),
        manifest_(std::make_shared<Manifest>()),
        buffer_size_(GetValueNoError<int32_t>(properties, PROPERTY_WRITER_BUFFER_SIZE)) {}

  /**
   * @brief Destructor
   *
   * Ensures proper cleanup of column group writers and temporary resources.
   * If close() hasn't been called, this will attempt to clean up gracefully.
   */
  ~WriterImpl() override {
    if (!closed_) {
      // Attempt graceful cleanup - ignore errors since we're in destructor
      auto result = close();
      (void)result;  // Suppress unused variable warning
    }
  }

  /**
   * @brief Gets the schema of the dataset being written
   *
   * @return Shared pointer to the Arrow schema
   */
  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  /**
   * @brief Writes a record batch to the dataset
   *
   * Distributes the batch data across appropriate column groups based on the
   * configured column group policy. Data is buffered and written to storage
   * when buffers reach their limits or flush() is called.
   *
   * @param batch Arrow RecordBatch containing the data to write
   * @return Status indicating success or error condition
   *
   * @note The batch schema must be compatible with the writer's schema.
   *       All batches written to the same writer should have consistent schemas.
   */
  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch) override {
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

    ARROW_RETURN_NOT_OK(distribute_batch(batch));

    return arrow::Status::OK();
  }

  /**
   * @brief Forces buffered data to be written to storage
   *
   * Flushes all pending data in column group writers to their respective
   * storage files. This ensures data durability but may impact performance
   * if called too frequently.
   *
   * @return Status indicating success or error condition
   *
   * @note This does not close the writers; additional batches can still be written
   *       after flushing.
   */
  arrow::Status flush() override {
    if (closed_) {
      return arrow::Status::Invalid("Cannot flush closed writer");
    }

    // Flush all column group writers
    for (auto& [column_group_id, writer] : column_group_writers_) {
      auto status = writer->Flush();
      if (!status.ok()) {
        return arrow::Status::IOError("Failed to flush writer: " + status.ToString());
      }
    }

    // Clear memory tracking
    current_memory_usage_ = 0;
    while (!memory_heap_.empty()) {
      memory_heap_.pop();
    }

    return arrow::Status::OK();
  }

  /**
   * @brief Finalizes the dataset and returns the manifest
   *
   * Closes all column group writers, finalizes storage files, and constructs
   * a manifest containing metadata about the written dataset. After calling
   * close(), no additional data can be written to this writer instance.
   *
   * @return Result containing the dataset manifest, or error status
   *
   * @note This method should be called exactly once per writer instance.
   *       Subsequent calls will return an error.
   */
  arrow::Result<std::shared_ptr<Manifest>> close() override {
    if (closed_) {
      return arrow::Status::Invalid("Writer already closed");
    }

    ARROW_ASSIGN_OR_RAISE(auto field_id_list_meta, field_id_list_meta());
    // Close all column group writers and collect statistics
    for (auto& [column_group_id, writer] : column_group_writers_) {
      ARROW_RETURN_NOT_OK(writer->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, field_id_list_meta));
      ARROW_RETURN_NOT_OK(writer->Close());
    }

    closed_ = true;

    // combine with existing manifest if provided
    if (existing_manifest_) {
      ARROW_RETURN_NOT_OK(Manifest::manifest_combine_paths(existing_manifest_, manifest_));
      return existing_manifest_;
    }

    return manifest_;
  }

  private:
  // ==================== Internal Data Members ====================

  std::shared_ptr<arrow::fs::FileSystem> fs_;               ///< Filesystem interface for data access
  std::string base_path_;                                   ///< Base directory for column group files
  std::shared_ptr<arrow::Schema> schema_;                   ///< Logical schema of the dataset
  std::unique_ptr<ColumnGroupPolicy> column_group_policy_;  ///< Policy for organizing columns
  Properties properties_;                                   ///< Write configuration properties

  std::shared_ptr<Manifest> existing_manifest_;              ///< Optional existing manifest to append to
  std::shared_ptr<Manifest> manifest_;                       ///< Dataset manifest being built
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;  ///< Column groups metadata
  std::map<int64_t, std::unique_ptr<internal::api::ColumnGroupWriter>>
      column_group_writers_;  ///< Writers for each column group

  bool closed_{false};       ///< Whether the writer has been closed
  bool initialized_{false};  ///< Whether the writer has been initialized

  // Memory management components (similar to packed implementation)
  size_t current_memory_usage_{0};                              ///< Current memory usage for buffered data
  size_t buffer_size_;                                          ///< Maximum buffer size before flushing
  std::priority_queue<std::pair<size_t, size_t>> memory_heap_;  ///< Memory usage tracking heap (group_id, memory_usage)

  // ==================== Internal Helper Methods ====================

  /**
   * @brief Initializes column group writers based on the policy
   *
   * @return Status indicating success or error condition
   */
  arrow::Status initialize_column_group_writers(const std::shared_ptr<arrow::RecordBatch>& batch) {
    // If policy requires sampling and this is the first batch, provide sample
    if (column_group_policy_->requires_sample()) {
      ARROW_RETURN_NOT_OK(column_group_policy_->sample(batch));
    }

    // Get column groups from policy
    column_groups_ = column_group_policy_->get_column_groups();

    if (column_groups_.empty()) {
      return arrow::Status::Invalid("Column group policy returned no column groups");
    }

    for (size_t i = 0; i < column_groups_.size(); ++i) {
      const auto& column_group = column_groups_[i];

      // Generate file path for this column group
      std::string file_path = base_path_ + "/column_group_" + std::to_string(i) + "." + column_group->format;
      column_group->paths = {file_path};

      // Create schema for this column group
      std::vector<std::shared_ptr<arrow::Field>> fields;

      for (const auto& column_name : column_group->columns) {
        auto field = schema_->GetFieldByName(column_name);
        if (!field) {
          return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
        }
        fields.emplace_back(field);
      }

      auto column_group_schema = std::make_shared<arrow::Schema>(fields);

      // Create column group writer
      auto writer = internal::api::GroupWriterFactory::create(column_group, column_group_schema, fs_, properties_);

      column_group_writers_.emplace(i, std::move(writer));

      // Add column group to manifest
      manifest_->add_column_group(column_group);
    }

    return arrow::Status::OK();
  }

  /**
   * @brief Distributes a record batch to appropriate column group writers
   *
   * @param batch The batch to distribute
   * @return Status indicating success or error condition
   */
  arrow::Status distribute_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (column_groups_.empty()) {
      return arrow::Status::Invalid("No column groups initialized");
    }

    // Flush column groups until there's enough room for the new batch
    // to ensure that memory usage stays strictly below the limit
    size_t next_batch_size = GetRecordBatchMemorySize(batch);
    while (current_memory_usage_ + next_batch_size >= buffer_size_ && !memory_heap_.empty()) {
      auto max_group = memory_heap_.top();
      memory_heap_.pop();
      current_memory_usage_ -= max_group.second;

      // Find the specific column group writer and flush it
      auto writer_it = column_group_writers_.find(max_group.first);
      if (writer_it != column_group_writers_.end()) {
        ARROW_RETURN_NOT_OK(writer_it->second->Flush());
      }
    }

    // Split the batch data directly based on column groups and write to each
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      const auto& column_group = column_groups_[i];
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
        memory_heap_.emplace(i, group_memory);

        // Write data to the column group writer
        auto writer_it = column_group_writers_.find(i);
        if (writer_it != column_group_writers_.end()) {
          auto status = writer_it->second->Write(group_batch);
          if (!status.ok()) {
            return arrow::Status::IOError("Failed to write batch: " + status.ToString());
          }
        }
      }
    }

    return balanceMemoryHeap();
  }

  /**
   * @brief Balances the memory heap to avoid duplicate entries
   *
   * @return Status indicating success or error condition
   */
  arrow::Status balanceMemoryHeap() {
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

  /**
   * @brief Creates group field id list for historical compatibility
   *
   * This function extracts the field ID list creation logic for historical
   * compatibility purposes. It creates the schema field ID list and column
   * group indices needed for the GroupFieldIDList.
   *
   * @return Result containing the GroupFieldIDList or error status
   */
  arrow::Result<std::string> field_id_list_meta() {
    auto schema_field_id_list = milvus_storage::FieldIDList::Make(schema_);
    if (!schema_field_id_list.ok()) {
      return arrow::Status::IOError("Failed to create field id list from schema: " +
                                    schema_field_id_list.status().ToString());
    }

    std::vector<std::vector<int>> column_group_indices;
    for (const auto& column_group : column_groups_) {
      std::vector<int> origin_column_indices;
      for (const auto& column_name : column_group->columns) {
        int col_index = schema_->GetFieldIndex(column_name);
        if (col_index >= 0) {
          origin_column_indices.push_back(col_index);
        }
      }
      column_group_indices.push_back(origin_column_indices);
    }
    return milvus_storage::GroupFieldIDList::Make(column_group_indices, schema_field_id_list.ValueOrDie()).Serialize();
  }
};

// ==================== Factory Function Implementation ====================

std::unique_ptr<Writer> Writer::create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       std::string base_path,
                                       std::shared_ptr<arrow::Schema> schema,
                                       std::unique_ptr<ColumnGroupPolicy> column_group_policy,
                                       std::shared_ptr<Manifest> existing_manifest,
                                       const Properties& properties) {
  return std::make_unique<WriterImpl>(std::move(fs), std::move(base_path), std::move(schema),
                                      std::move(column_group_policy), existing_manifest, properties);
}

}  // namespace milvus_storage::api
