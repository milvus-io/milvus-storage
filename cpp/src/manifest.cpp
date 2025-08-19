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

#include "milvus-storage/manifest.h"

#include <algorithm>
#include <set>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include "milvus-storage/common/result.h"

namespace milvus_storage::api {

// ==================== Manifest Implementation ====================

Manifest::Manifest() : version_(0), next_column_group_id_(1) {
  // Initialize empty manifest with default values
}

Manifest::Manifest(std::shared_ptr<arrow::Schema> schema)
    : schema_(std::move(schema)), version_(0), next_column_group_id_(1) {
  if (!schema_) {
    throw std::invalid_argument("Schema cannot be null");
  }
  // Initialize manifest with provided schema
}

// ==================== Column Group Management ====================

std::vector<std::shared_ptr<ColumnGroup>> Manifest::get_column_groups() const { return column_groups_; }

std::shared_ptr<ColumnGroup> Manifest::get_column_group(int64_t id) const {
  auto it = std::find_if(column_groups_.begin(), column_groups_.end(),
                         [id](const std::shared_ptr<ColumnGroup>& cg) { return cg->id == id; });
  return (it != column_groups_.end()) ? *it : nullptr;
}

std::shared_ptr<ColumnGroup> Manifest::get_column_group(const std::string& column_name) const {
  // Use the fast lookup map if available
  auto it = column_to_group_map_.find(column_name);
  if (it != column_to_group_map_.end()) {
    return get_column_group(it->second);
  }

  // Fallback to linear search (shouldn't happen if mapping is maintained properly)
  for (const auto& cg : column_groups_) {
    if (cg->contains_column(column_name)) {
      return cg;
    }
  }

  return nullptr;
}

std::vector<std::shared_ptr<ColumnGroup>> Manifest::get_column_groups_for_columns(
    const std::set<std::string>& column_names) const {
  std::set<int64_t> found_group_ids;
  std::vector<std::shared_ptr<ColumnGroup>> result;

  // Find all unique column groups that contain any of the requested columns
  for (const auto& column_name : column_names) {
    auto it = column_to_group_map_.find(column_name);
    if (it != column_to_group_map_.end() && found_group_ids.find(it->second) == found_group_ids.end()) {
      found_group_ids.insert(it->second);
      auto cg = get_column_group(it->second);
      if (cg != nullptr) {
        result.push_back(cg);
      }
    }
  }

  return result;
}

// ==================== Schema Management ====================

std::shared_ptr<arrow::Schema> Manifest::schema() const { return schema_; }

// ==================== Column Group Modification ====================

arrow::Status Manifest::add_column_group(std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  // Validate the column group
  auto validation_result = validate_column_group(column_group);
  if (!validation_result.ok()) {
    return arrow::Status::Invalid("Column group validation failed: " + validation_result.ToString());
  }

  // Assign ID if not already set
  if (column_group->id == 0) {
    column_group->id = generate_column_group_id();
  }

  // Check for ID conflicts
  if (get_column_group(column_group->id) != nullptr) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(column_group->id) + " already exists");
  }

  // Check for column conflicts with existing column groups
  for (const auto& column_name : column_group->columns) {
    auto existing_cg = get_column_group(column_name);
    if (existing_cg != nullptr) {
      return arrow::Status::Invalid("Column '" + column_name + "' already exists in column group " +
                                    std::to_string(existing_cg->id));
    }
  }

  // Add the column group
  column_groups_.push_back(std::move(column_group));

  // Update column mapping
  for (const auto& column_name : column_groups_.back()->columns) {
    column_to_group_map_[column_name] = column_groups_.back()->id;
  }

  return arrow::Status::OK();
}

arrow::Status Manifest::drop_column_group(int64_t id) {
  auto it = std::find_if(column_groups_.begin(), column_groups_.end(),
                         [id](const std::shared_ptr<ColumnGroup>& cg) { return cg->id == id; });

  if (it == column_groups_.end()) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(id) + " not found");
  }

  // Remove from column mapping
  for (const auto& column_name : (*it)->columns) {
    column_to_group_map_.erase(column_name);
  }

  // Remove from column groups
  column_groups_.erase(it);

  return arrow::Status::OK();
}

arrow::Status Manifest::update_column_group(int64_t id, std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  if (id <= 0) {
    return arrow::Status::Invalid("Column group ID must be positive, got: " + std::to_string(id));
  }

  auto existing_cg = get_column_group(id);
  if (!existing_cg) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(id) + " not found");
  }

  // Validate the updated column group
  auto validation_result = validate_column_group(column_group);
  if (!validation_result.ok()) {
    return arrow::Status::Invalid("Column group validation failed: " + validation_result.ToString());
  }

  // Check for column conflicts with other column groups (excluding this one)
  for (const auto& column_name : column_group->columns) {
    auto conflicting_cg = get_column_group(column_name);
    if (conflicting_cg != nullptr && conflicting_cg->id != id) {
      return arrow::Status::Invalid("Column '" + column_name + "' already exists in column group " +
                                    std::to_string(conflicting_cg->id));
    }
  }

  // Remove old column mappings
  for (const auto& column_name : existing_cg->columns) {
    column_to_group_map_.erase(column_name);
  }

  // Update the column group (preserve the original ID)
  column_group->id = id;

  // Find and replace in the vector
  auto it = std::find_if(column_groups_.begin(), column_groups_.end(),
                         [id](const std::shared_ptr<ColumnGroup>& cg) { return cg->id == id; });
  *it = std::move(column_group);

  // Add new column mappings
  for (const auto& column_name : (*it)->columns) {
    column_to_group_map_[column_name] = id;
  }

  return arrow::Status::OK();
}

// ==================== Versioning ====================

int64_t Manifest::version() const { return version_; }

void Manifest::set_version(int64_t version) { version_ = version; }

// ==================== Statistics ====================

ColumnGroupStats Manifest::get_aggregate_stats() const {
  ColumnGroupStats aggregate_stats;

  for (const auto& cg : column_groups_) {
    aggregate_stats.num_rows += cg->stats.num_rows;
    aggregate_stats.uncompressed_size += cg->stats.uncompressed_size;
    aggregate_stats.compressed_size += cg->stats.compressed_size;
    aggregate_stats.num_chunks += cg->stats.num_chunks;
  }

  return aggregate_stats;
}

Status Manifest::refresh_stats(const std::shared_ptr<arrow::fs::FileSystem>& fs) {
  if (!fs) {
    return Status::InvalidArgument("FileSystem cannot be null");
  }

  // TODO: Implement statistics refresh by reading file metadata
  // This would scan each column group file to update statistics
  // For now, return not implemented
  return Status::InvalidArgument("Statistics refresh not yet implemented");
}

// ==================== Internal Helper Methods ====================

void Manifest::rebuild_column_mapping() {
  column_to_group_map_.clear();

  for (const auto& cg : column_groups_) {
    for (const auto& column_name : cg->columns) {
      column_to_group_map_[column_name] = cg->id;
    }
  }
}

Status Manifest::validate_column_group(const std::shared_ptr<ColumnGroup>& column_group) const {
  if (!column_group) {
    return Status::InvalidArgument("Column group cannot be null");
  }

  if (column_group->columns.empty()) {
    return Status::InvalidArgument("Column group must contain at least one column");
  }

  if (column_group->path.empty()) {
    return Status::InvalidArgument("Column group must have a valid file path");
  }

  // Check for duplicate columns within the column group
  std::set<std::string> unique_columns(column_group->columns.begin(), column_group->columns.end());
  if (unique_columns.size() != column_group->columns.size()) {
    return Status::InvalidArgument("Column group contains duplicate column names");
  }

  // Validate column names are not empty
  for (const auto& column_name : column_group->columns) {
    if (column_name.empty()) {
      return Status::InvalidArgument("Column names cannot be empty");
    }
  }

  // Validate statistics are non-negative
  if (column_group->stats.num_rows < 0) {
    return Status::InvalidArgument("Number of rows cannot be negative");
  }
  if (column_group->stats.uncompressed_size < 0) {
    return Status::InvalidArgument("Uncompressed size cannot be negative");
  }
  if (column_group->stats.compressed_size < 0) {
    return Status::InvalidArgument("Compressed size cannot be negative");
  }
  if (column_group->stats.num_chunks < 0) {
    return Status::InvalidArgument("Number of chunks cannot be negative");
  }

  // Validate against schema if present
  if (schema_ != nullptr) {
    for (const auto& column_name : column_group->columns) {
      if (!schema_->GetFieldByName(column_name)) {
        return Status::InvalidArgument("Column '" + column_name + "' not found in schema");
      }
    }
  }

  return Status::OK();
}

Status Manifest::validate() const {
  // Validate that all columns in schema are covered by column groups
  if (schema_) {
    for (int i = 0; i < schema_->num_fields(); ++i) {
      const auto& field_name = schema_->field(i)->name();
      if (!get_column_group(field_name)) {
        return Status::InvalidArgument("Schema column '" + field_name + "' is not covered by any column group");
      }
    }
  }

  // Validate that column mapping is consistent
  for (const auto& cg : column_groups_) {
    for (const auto& column_name : cg->columns) {
      auto mapped_cg_id = column_to_group_map_.find(column_name);
      if (mapped_cg_id == column_to_group_map_.end()) {
        return Status::InvalidArgument("Column '" + column_name + "' not found in column mapping");
      }
      if (mapped_cg_id->second != cg->id) {
        return Status::InvalidArgument("Column '" + column_name + "' mapping is inconsistent");
      }
    }
  }

  return Status::OK();
}

int64_t Manifest::generate_column_group_id() { return next_column_group_id_++; }

}  // namespace milvus_storage::api
