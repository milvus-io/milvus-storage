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

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <iostream>
#include <unordered_map>

#include "milvus-storage/common/serializable.h"

namespace milvus_storage::api {

/**
 * @brief Metadata about a column group in the dataset
 *
 * A column group represents a set of related columns stored together
 * in the same physical file(s) for optimal compression and query performance.
 * This follows the principles of columnar storage with column group organization.
 */
struct ColumnGroup {
  std::vector<std::string> columns;  ///< Names of columns stored in this group
  std::vector<std::string> paths;    ///< Physical file paths where the column group is stored
  std::string format;                ///< Storage format (parquet, lance, vortex, binary)
};

/**
 * @brief Dataset Column Groups containing metadata about column groups and schema
 *
 * The ColumnGroups class serves as the central metadata repository for a milvus
 * storage dataset. It maintains information about:
 * - Column group organization and layout
 * - Physical storage locations and formats
 * - Dataset schema
 * - Transaction management for atomic updates
 *
 * This enables efficient query planning by providing metadata about data
 * distribution, statistics, and storage layout without requiring expensive
 * file system operations or data scanning.
 */
class ColumnGroups : public Serializable {
  public:
  ColumnGroups() : column_groups_() {}

  /**
   * @brief Serializes the column groups to a JSON string
   *
   * @return JSON string representation of the column groups
   */
  arrow::Result<std::string> serialize() const override;

  /**
   * @brief Deserializes the column groups from a JSON string
   *
   * @param data JSON string representation of the column groups
   * @return true if deserialization was successful, false otherwise
   */
  arrow::Status deserialize(const std::string_view& data) override;

  /**
   * @brief Constructs a new manifest with column groups
   *
   * Creates a manifest with no column groups and initializes internal
   * data structures for managing column group metadata.
   *
   * @param column_groups Vector of column groups to add to the manifest
   */
  ColumnGroups(std::vector<std::shared_ptr<ColumnGroup>> column_groups);

  // Disable copy constructor and assignment operator for performance
  ColumnGroups(const ColumnGroups&) = delete;
  ColumnGroups& operator=(const ColumnGroups&) = delete;

  // Enable move constructor and assignment operator
  ColumnGroups(ColumnGroups&&) = default;
  ColumnGroups& operator=(ColumnGroups&&) = default;

  /**
   * @brief Destructor
   */
  ~ColumnGroups() = default;

  // ==================== Column Group Management ====================

  /**
   * @brief Retrieves all column groups in the dataset
   *
   * @return Vector of shared pointers to all column group metadata
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_all() const;

  /**
   * @brief Finds the column group that contains the specified column
   *
   * This method performs a lookup to find which column group stores
   * the given column. This is essential for query planning to determine
   * which physical files need to be accessed.
   *
   * @param column_name Name of the column to locate
   * @return Shared pointer to the column group containing the column, or nullptr if not found
   */
  [[nodiscard]] std::shared_ptr<ColumnGroup> get_column_group(const std::string& column_name) const;

  // ==================== Column Group Modification ====================

  /**
   * @brief Adds a new column group to the manifest
   *
   * The column group must have a unique ID and valid metadata.
   * This operation is transactional if a transaction is active.
   *
   * @param column_group Shared pointer to the column group to add
   * @return if the column group is added successfully
   */
  bool add_column_group(std::shared_ptr<ColumnGroup> column_group);

  private:
  /**
   * @brief Rebuilds the column-to-group mapping for fast lookups
   */
  void rebuild_column_mapping();

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;  ///< All column groups in the dataset

  // temporal map for fast lookup: column name -> column group index
  std::unordered_map<std::string, int64_t> column_to_group_map_;
};

}  // namespace milvus_storage::api