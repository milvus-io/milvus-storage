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
#include <unordered_set>
#include <unordered_map>
#include <arrow/status.h>

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
 * @brief Dataset manifest containing metadata about column groups and schema
 *
 * The Manifest class serves as the central metadata repository for a milvus
 * storage dataset. It maintains information about:
 * - Column group organization and layout
 * - Physical storage locations and formats
 * - Dataset schema and versioning
 * - Transaction management for atomic updates
 *
 * This enables efficient query planning by providing metadata about data
 * distribution, statistics, and storage layout without requiring expensive
 * file system operations or data scanning.
 */
class Manifest {
  public:
  Manifest() : column_groups_(), version_(0) {}
  /**
   * @brief Constructs a new manifest with column groups and version
   *
   * Creates a manifest with no column groups and initializes internal
   * data structures for managing column group metadata.
   *
   * @param column_groups Vector of column groups to add to the manifest
   * @param version Version number of the manifest
   */
  Manifest(std::vector<std::shared_ptr<ColumnGroup>> column_groups, uint64_t version);

  // Disable copy constructor and assignment operator for performance
  Manifest(const Manifest&) = delete;
  Manifest& operator=(const Manifest&) = delete;

  // Enable move constructor and assignment operator
  Manifest(Manifest&&) = default;
  Manifest& operator=(Manifest&&) = default;

  /**
   * @brief Destructor
   */
  ~Manifest() = default;

  /**
   * @brief Combines another manifest into this one
   *
   * Merges the column groups from another manifest into this one.
   * Both manifests must have the same number of column groups with
   * matching columns and formats. Paths are not required to match.
   * After combination, paths from the other manifest are appended
   *
   * @param other The other manifest to combine
   * @return arrow::Status
   *  - OK if the manifests were successfully combined
   *  - Invalid if the manifests are incompatible
   */
  static arrow::Status manifest_combine_paths(const std::shared_ptr<Manifest>& manifest1,
                                              const std::shared_ptr<Manifest>& manifest2);

  // ==================== Column Group Management ====================

  /**
   * @brief Retrieves all column groups in the dataset
   *
   * @return Vector of shared pointers to all column group metadata
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const;

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
   * @brief Retrieves column groups that contain any of the specified columns
   *
   * This is an optimization for queries that need multiple columns,
   * allowing efficient identification of all required column groups.
   *
   * @param column_names Set of column names to find
   * @return Vector of column groups that contain at least one of the specified columns
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups_for_columns(
      const std::set<std::string>& column_names) const;

  /**
   * @brief Retrieves all unique column names in the manifest
   */
  [[nodiscard]] std::unordered_set<std::string_view> get_all_column_names() const;

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

  // ==================== Versioning ====================

  /**
   * @brief Gets the current version of the manifest
   *
   * @return Version number (monotonically increasing)
   */
  [[nodiscard]] inline uint64_t version() const { return version_; }

  /**
   * @brief Sets the manifest version
   *
   * @param version New version number
   */
  inline void set_version(uint64_t version) { version_ = version; }

  private:
  /**
   * @brief Rebuilds the column-to-group mapping for fast lookups
   */
  void rebuild_column_mapping();

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;  ///< All column groups in the dataset
  uint64_t version_;                                         ///< Current manifest version

  // temporal map for fast lookup: column name -> column group index
  std::unordered_map<std::string, int64_t> column_to_group_map_;
};

/**
 * @brief Abstract base class for manifest serialization and deserialization
 *
 * Provides a common interface for different serialization formats.
 * Implementations can support JSON, binary, protobuf, or other formats.
 */
class ManifestSerDe {
  public:
  virtual ~ManifestSerDe() = default;

  /**
   * @brief Serializes a manifest to JSON format
   *
   * @param manifest The manifest to serialize
   * @return
   *  - bool: true if serialization was successful, false otherwise
   *  - string: On success, the second element contains the JSON string
   */
  virtual std::pair<bool, std::string> Serialize(const std::shared_ptr<api::Manifest>& manifest) = 0;

  /**
   * @brief Deserializes a manifest from the input stream
   *
   * @param input Input string to read from
   * @param manifest Output parameter for the deserialized manifest
   * @return non-nullptr if deserialization was successful, nullptr otherwise
   */
  virtual std::shared_ptr<Manifest> Deserialize(const std::string& input) = 0;

  /**
   * @brief Deserializes a manifest from the input stream
   *
   * @param input Input string to read from
   * @param manifest Output parameter for the deserialized manifest
   * @return non-nullptr if deserialization was successful, nullptr otherwise
   */
  virtual std::shared_ptr<Manifest> Deserialize(const std::string_view& input) = 0;
};

}  // namespace milvus_storage::api