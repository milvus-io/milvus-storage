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

#include <assert.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <iostream>

namespace milvus_storage::api {

void Manifest::rebuild_column_mapping() {
  column_to_group_map_.clear();

  for (size_t i = 0; i < column_groups_.size(); i++) {
    auto& cg = column_groups_[i];
    for (const auto& column_name : cg->columns) {
      column_to_group_map_[column_name] = i;
    }
  }
}

Manifest::Manifest(std::vector<std::shared_ptr<ColumnGroup>> column_groups, uint64_t version)
    : column_groups_(std::move(column_groups)), version_(version) {
  rebuild_column_mapping();
}

arrow::Status Manifest::manifest_combine_paths(const std::shared_ptr<Manifest>& manifest1,
                                               const std::shared_ptr<Manifest>& manifest2) {
  auto column_groups1 = manifest1->get_column_groups();
  auto column_groups2 = manifest2->get_column_groups();

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

#ifndef NDEBUG
    // check paths do not overlaps
    std::set<std::string> path_set(cg1->paths.begin(), cg1->paths.end());
    for (const auto& path : cg2->paths) {
      assert(path_set.count(path) == 0);  // paths should not overlap
      path_set.insert(path);
    }
#endif

    // merge paths
    for (const auto& path : cg2->paths) {
      cg1->paths.emplace_back(path);
    }
  }

  return arrow::Status::OK();
}

std::vector<std::shared_ptr<ColumnGroup>> Manifest::get_column_groups() const { return column_groups_; }

std::shared_ptr<ColumnGroup> Manifest::get_column_group(const std::string& column_name) const {
  auto it = column_to_group_map_.find(column_name);
  if (it != column_to_group_map_.end()) {
    return column_groups_[it->second];
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
      auto& cg = column_groups_[it->second];
      if (cg != nullptr) {
        result.emplace_back(cg);
      }
    }
  }

  return result;
}

std::unordered_set<std::string_view> Manifest::get_all_column_names() const {
  std::unordered_set<std::string_view> col_set;
  for (const auto& cg : column_groups_) {
    for (const auto& col : cg->columns) {
      assert(col_set.find(col) == col_set.end());  // should be unique
      col_set.insert(col);
    }
  }
  return col_set;
}

arrow::Status Manifest::add_column_group(std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return arrow::Status::Invalid("Column group is null");
  }

  // Check for column conflicts with existing column groups
  for (const auto& column_name : column_group->columns) {
    auto existing_cg = get_column_group(column_name);
    if (existing_cg != nullptr) {
      return arrow::Status::Invalid("Column ", column_name, " already exists in the manifest");
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

}  // namespace milvus_storage::api