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
#include <unordered_map>
#include <optional>

#include "milvus-storage/common/serializable.h"

namespace milvus_storage::api {

struct ColumnGroupFile {
  std::string path;                    ///< Physical file path where the column group is stored
  std::optional<int64_t> start_index;  ///< Optional start index of data in the file
  std::optional<int64_t> end_index;    ///< Optional end index of data in the file
};

/**
 * @brief Metadata about a column group in the dataset
 *
 * A column group represents a set of related columns stored together
 * in the same physical file(s) for optimal compression and query performance.
 * This follows the principles of columnar storage with column group organization.
 */
struct ColumnGroup {
  std::vector<std::string> columns;    ///< Names of columns stored in this group
  std::string format;                  ///< Storage format (parquet, lance, vortex, binary)
  std::vector<ColumnGroupFile> files;  ///< Physical file paths and metadata for each file
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
 *
 */
class ColumnGroups final : public Serializable {
  public:
  ColumnGroups() : column_groups_() {}

  /**
   * @brief Serializes the column groups to an Avro binary string
   *
   * @return Avro binary string representation of the column groups
   */
  arrow::Result<std::string> serialize() const override;

  /**
   * @brief Deserializes the column groups from an Avro binary string
   *
   * @param data Avro binary string representation of the column groups
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
   * @brief Retrieves the number of column groups in the dataset
   *
   * @return Number of column groups
   */
  [[nodiscard]] inline size_t size() const { return column_groups_.size(); }

  /**
   * @brief Retrieves all column groups in the dataset
   *
   * @return Vector of shared pointers to all column group metadata
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_all() const;

  /**
   * @brief Retrieves the number of metadata key-value pairs
   *
   * @return Number of metadata entries
   */
  [[nodiscard]] inline size_t meta_size() const { return metadata_.size(); }

  /**
   * @brief Retrieves a metadata key-value pair by index
   *
   * @param idx Index of the metadata entry to retrieve
   * @return Pair of string views representing the key and value
   */
  [[nodiscard]] arrow::Result<std::pair<std::string_view, std::string_view>> get_metadata(size_t idx) const;

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

  /**
   * @brief Finds the column group by its index
   *
   * @param column_group_index index of the column group to locate
   * @return Shared pointer to the column group containing the column, or nullptr if not found or out of range
   */
  [[nodiscard]] std::shared_ptr<ColumnGroup> get_column_group(size_t column_group_index) const;

  /**
   * @brief This method will append paths from new column groups to existing
   * column groups. the structure of column groups must be the same(format and
   * columns of each column group must be the same).
   *
   * @param new_cg Shared pointer to the column groups containing new file paths
   * @return Status indicating success or failure of the operation
   */
  [[nodiscard]] arrow::Status append_files(const std::shared_ptr<ColumnGroups>& new_cg);
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
  arrow::Status add_column_group(std::shared_ptr<ColumnGroup> column_group);

  /**
   * @brief Appends metadata key-value pairs to the column groups
   *        notice that the key&&value life cycle should be longer
   *        than the column groups.
   * @param keys Vector of metadata keys
   * @param values Vector of metadata values
   * @return Status indicating success or failure of the operation
   */
  arrow::Status add_metadatas(const std::vector<std::string_view>& keys, const std::vector<std::string_view>& values);

  private:
  /**
   * @brief Rebuilds the column-to-group mapping for fast lookups
   */
  void rebuild_column_mapping();

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;    ///< All column groups in the dataset
  std::vector<std::pair<std::string, std::string>> metadata_;  ///< Additional metadata key-value pairs

  // temporal map for fast lookup: column name -> column group index
  std::unordered_map<std::string, int64_t> column_to_group_map_;
};

}  // namespace milvus_storage::api