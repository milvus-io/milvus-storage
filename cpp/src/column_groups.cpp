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

#include "milvus-storage/column_groups.h"

#include <unordered_set>
#include <set>
#include <string>
#include <vector>

#include <arrow/status.h>
#include <arrow/result.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/common/macro.h"

namespace milvus_storage::api {

ColumnGroups::ColumnGroups(std::vector<std::shared_ptr<ColumnGroup>> column_groups)
    : column_groups_(std::move(column_groups)) {
  rebuild_column_mapping();
}

arrow::Result<std::string> ColumnGroups::serialize() const {
  try {
    nlohmann::json j;
    j["column_groups"] = nlohmann::json::array();
    j["metadata"] = nlohmann::json::object();

    for (const auto& group : column_groups_) {
      assert(group != nullptr);
      nlohmann::json cgs_obj = nlohmann::json::object();
      cgs_obj["columns"] = group->columns;
      cgs_obj["format"] = group->format;
      cgs_obj["files"] = nlohmann::json::array();

      for (const auto& file : group->files) {
        nlohmann::json file_json = nlohmann::json::object();
        file_json["path"] = file.path;

        if (file.start_index.has_value()) {
          file_json["start_index"] = file.start_index.value();
        }
        if (file.end_index.has_value()) {
          file_json["end_index"] = file.end_index.value();
        }

        cgs_obj["files"].emplace_back(file_json);
      }

      j["column_groups"].emplace_back(cgs_obj);
    }

    for (const auto& [key, value] : metadata_) {
      j["metadata"][key] = value;
    }

    return j.dump(2);
  } catch (const nlohmann::json::exception& e) {
    return arrow::Status::Invalid("Failed to serialize ColumnGroups: " + std::string(e.what()));
  }
}

arrow::Status ColumnGroups::deserialize(const std::string_view& data) {
  try {
    std::vector<std::shared_ptr<ColumnGroup>> column_groups;
    nlohmann::json j = nlohmann::json::parse(data);

    if (UNLIKELY(!j.contains("column_groups") || !j["column_groups"].is_array())) {
      return arrow::Status::Invalid(
          "Invalid ColumnGroups JSON: missing 'column_groups' array or wrong type. [data=" + std::string(data) + "]");
    }

    for (const auto& cg_json : j["column_groups"]) {
      auto cg = std::make_shared<ColumnGroup>();
      cg_json.at("columns").get_to(cg->columns);
      cg_json.at("format").get_to(cg->format);

      if (UNLIKELY(!cg_json.contains("files") || !cg_json["files"].is_array())) {
        return arrow::Status::Invalid(
            "Invalid ColumnGroups JSON: missing 'files' array or wrong type. [data=" + std::string(data) + "]");
      }

      // parse files
      for (const auto& file_json : cg_json["files"]) {
        ColumnGroupFile file;
        file_json.at("path").get_to(file.path);

        // the start_index and end_index are optional
        if (file_json.contains("start_index")) {
          file.start_index = file_json.at("start_index").get<size_t>();
        }

        if (file_json.contains("end_index")) {
          file.end_index = file_json.at("end_index").get<size_t>();
        }

        cg->files.emplace_back(file);
      }

      column_groups.emplace_back(cg);
    }

    // The metadata is optional
    if (j.contains("metadata")) {
      if (UNLIKELY(!j["metadata"].is_object())) {
        return arrow::Status::Invalid(
            "Invalid ColumnGroups JSON: 'metadata' is not an object. [data=" + std::string(data) + "]");
      }

      for (auto it = j["metadata"].begin(); it != j["metadata"].end(); ++it) {
        metadata_.emplace_back(it.key(), it.value().get<std::string>());
      }
    }

    column_groups_ = std::move(column_groups);
    rebuild_column_mapping();
    return arrow::Status::OK();
  } catch (const nlohmann::json::exception& e) {
    return arrow::Status::Invalid("Failed to deserialize ColumnGroups: " + std::string(e.what()));
  }
}

void ColumnGroups::rebuild_column_mapping() {
  column_to_group_map_.clear();

  for (size_t i = 0; i < column_groups_.size(); i++) {
    auto& cg = column_groups_[i];
    for (const auto& column_name : cg->columns) {
      column_to_group_map_[column_name] = i;
    }
  }
}

std::vector<std::shared_ptr<ColumnGroup>> ColumnGroups::get_all() const { return column_groups_; }

arrow::Result<std::pair<std::string_view, std::string_view>> ColumnGroups::get_metadata(size_t idx) const {
  if (idx >= metadata_.size()) {
    return arrow::Status::Invalid("Metadata index out of range: " + std::to_string(idx));
  }

  return std::make_pair(std::string_view(metadata_[idx].first), std::string_view(metadata_[idx].second));
}

std::shared_ptr<ColumnGroup> ColumnGroups::get_column_group(const std::string& column_name) const {
  auto it = column_to_group_map_.find(column_name);
  if (it != column_to_group_map_.end()) {
    return column_groups_[it->second];
  }

  return nullptr;
}

std::shared_ptr<ColumnGroup> ColumnGroups::get_column_group(size_t column_group_index) const {
  if (column_group_index < column_groups_.size()) {
    return column_groups_[column_group_index];
  }

  return nullptr;
}

arrow::Status ColumnGroups::append_files(const std::shared_ptr<ColumnGroups>& new_cg) {
  auto& column_groups1 = column_groups_;
  auto& column_groups2 = new_cg->column_groups_;

  // if no existing column groups, directly assign
  // this situation happens when no data has been written yet
  if (column_groups1.empty()) {
    column_groups_ = column_groups2;
    rebuild_column_mapping();
    return arrow::Status::OK();
  }

  if (column_groups1.size() != column_groups2.size()) {
    return arrow::Status::Invalid("Column group size mismatch");
  }

  for (size_t i = 0; i < column_groups1.size(); i++) {
    auto& cg1 = column_groups1[i];
    auto& cg2 = column_groups2[i];

    // only compare columns and format, no need match the paths
    if (cg1->columns.size() != cg2->columns.size()) {
      return arrow::Status::Invalid("Column groups size mismatch at index ", std::to_string(i));
    }

    // compare format
    if (cg1->format != cg2->format) {
      return arrow::Status::Invalid("Column groups format mismatch at index ", std::to_string(i));
    }

    // check columns
    std::set<std::string_view> col_set(cg1->columns.begin(), cg1->columns.end());
    for (const auto& col : cg2->columns) {
      if (col_set.find(col) == col_set.end()) {
        return arrow::Status::Invalid("Column group columns mismatch at index ", std::to_string(i));
      }
    }

    // no need check field `files`, we allow overlap paths in this field
    // merge field `files`
    for (const auto& file : cg2->files) {
      cg1->files.emplace_back(file);
    }
  }

  return arrow::Status::OK();
}

arrow::Status ColumnGroups::add_column_group(std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return arrow::Status::Invalid("column group is empty");
  }

  // Check for column conflicts with existing column groups
  for (const auto& column_name : column_group->columns) {
    auto existing_cg = get_column_group(column_name);
    if (existing_cg != nullptr) {
      return arrow::Status::Invalid("Column '" + column_name + "' already exists in another column group");
    }
  }

  // Add the column group
  column_groups_.emplace_back(std::move(column_group));

  // Update column mapping
  for (const auto& column_name : column_groups_.back()->columns) {
    column_to_group_map_[column_name] = column_groups_.size() - 1;
  }

  return arrow::Status::OK();
}

arrow::Status ColumnGroups::add_metadatas(const std::vector<std::string_view>& keys,
                                          const std::vector<std::string_view>& values) {
  std::unordered_set<std::string_view> existing_keys;
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (existing_keys.find(keys[i]) != existing_keys.end()) {
      return arrow::Status::Invalid("Duplicate metadata key: " + std::string(keys[i]));
    }

    metadata_.emplace_back(keys[i], values[i]);
    existing_keys.insert(keys[i]);
  }
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api