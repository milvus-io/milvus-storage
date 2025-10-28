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
#include <nlohmann/json.hpp>

namespace milvus_storage::api {

ColumnGroups::ColumnGroups(std::vector<std::shared_ptr<ColumnGroup>> column_groups)
    : column_groups_(std::move(column_groups)) {
  rebuild_column_mapping();
}

arrow::Result<std::string> ColumnGroups::serialize() const {
  try {
    nlohmann::json j;
    j["column_groups"] = nlohmann::json::array();

    for (const auto& group : column_groups_) {
      if (group) {
        j["column_groups"].push_back(
            nlohmann::json{{"columns", group->columns}, {"paths", group->paths}, {"format", group->format}});
      }
    }

    return j.dump(2);
  } catch (const nlohmann::json::exception& e) {
    return arrow::Status::Invalid("Failed to serialize ColumnGroups: " + std::string(e.what()));
  }
}

arrow::Status ColumnGroups::deserialize(const std::string_view& data) {
  try {
    nlohmann::json j = nlohmann::json::parse(data);

    std::vector<std::shared_ptr<ColumnGroup>> column_groups;

    if (j.contains("column_groups") && j["column_groups"].is_array()) {
      for (const auto& cg_json : j["column_groups"]) {
        auto cg = std::make_shared<ColumnGroup>();
        cg_json.at("columns").get_to(cg->columns);
        cg_json.at("paths").get_to(cg->paths);
        cg_json.at("format").get_to(cg->format);
        column_groups.push_back(cg);
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

std::shared_ptr<ColumnGroup> ColumnGroups::get_column_group(const std::string& column_name) const {
  auto it = column_to_group_map_.find(column_name);
  if (it != column_to_group_map_.end()) {
    return column_groups_[it->second];
  }

  return nullptr;
}

bool ColumnGroups::add_column_group(std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return false;
  }

  // Check for column conflicts with existing column groups
  for (const auto& column_name : column_group->columns) {
    auto existing_cg = get_column_group(column_name);
    if (existing_cg != nullptr) {
      return false;
    }
  }

  // Add the column group
  column_groups_.push_back(std::move(column_group));

  // Update column mapping
  for (const auto& column_name : column_groups_.back()->columns) {
    column_to_group_map_[column_name] = column_groups_.size() - 1;
  }

  return true;
}

}  // namespace milvus_storage::api