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
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <arrow/filesystem/filesystem.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <arrow/io/api.h>
#include "milvus-storage/common/status.h"
#include "milvus-storage/common/result.h"

namespace milvus_storage::api {

/**
 * @brief Supported file formats for column group storage
 */
enum class FileFormat {
  PARQUET,  ///< Apache Parquet columnar format
  BINARY,   ///< Binary format
  VORTEX,   ///< Vortex format
  LANCE     ///< Lance format
};

/**
 * @brief Statistics about a column group for query optimization
 */
struct ColumnGroupStats {
  int64_t num_rows = 0;           ///< Total number of rows in the column group
  int64_t uncompressed_size = 0;  ///< Uncompressed data size in bytes
  int64_t compressed_size = 0;    ///< Compressed data size in bytes
  int64_t num_chunks = 0;         ///< Number of chunks (row groups) in the column group

  // Per-column statistics could be added here for advanced optimization
  // std::map<std::string, ColumnStats> column_stats;
};

/**
 * @brief Metadata about a column group in the dataset
 *
 * A column group represents a set of related columns stored together
 * in the same physical file(s) for optimal compression and query performance.
 * This follows the principles of columnar storage with column group organization.
 */
struct ColumnGroup {
  int64_t id;                        ///< Unique identifier for the column group
  std::vector<std::string> columns;  ///< Names of columns stored in this group
  std::string path;                  ///< Physical file path where the column group is stored
  FileFormat format;                 ///< Storage format (e.g., Parquet)
  ColumnGroupStats stats;            ///< Statistics for query optimization

  /**
   * @brief Checks if this column group contains the specified column
   *
   * @param column_name Name of the column to check
   * @return true if the column is part of this column group, false otherwise
   */
  [[nodiscard]] bool contains_column(const std::string& column_name) const {
    return std::find(columns.begin(), columns.end(), column_name) != columns.end();
  }
};

/**
 * @brief Builder class for constructing ColumnGroup objects
 *
 * Provides a fluent interface for creating ColumnGroup instances with validation.
 *
 * @example
 * auto cg = ColumnGroupBuilder(1)
 *             .with_columns({"id", "name", "age"})
 *             .with_path("/data/cg1.parquet")
 *             .with_format(FileFormat::PARQUET)
 *             .build();
 */
class ColumnGroupBuilder {
  public:
  explicit ColumnGroupBuilder(int64_t id) : id_(id) {}

  ColumnGroupBuilder& with_columns(std::vector<std::string> columns) {
    columns_ = std::move(columns);
    return *this;
  }

  ColumnGroupBuilder& add_column(const std::string& column) {
    columns_.push_back(column);
    return *this;
  }

  ColumnGroupBuilder& with_path(std::string path) {
    path_ = std::move(path);
    return *this;
  }

  ColumnGroupBuilder& with_format(FileFormat format) {
    format_ = format;
    return *this;
  }

  ColumnGroupBuilder& with_stats(const ColumnGroupStats& stats) {
    stats_ = stats;
    return *this;
  }

  [[nodiscard]] std::shared_ptr<ColumnGroup> build() const {
    auto column_group = std::make_shared<ColumnGroup>();
    column_group->id = id_;
    column_group->columns = columns_;
    column_group->path = path_;
    column_group->format = format_;
    column_group->stats = stats_;
    return column_group;
  }

  ~ColumnGroupBuilder() = default;

  // Disable copy constructor and assignment operator
  ColumnGroupBuilder(const ColumnGroupBuilder&) = delete;
  ColumnGroupBuilder& operator=(const ColumnGroupBuilder&) = delete;

  // Enable move constructor and assignment operator
  ColumnGroupBuilder(ColumnGroupBuilder&&) = default;
  ColumnGroupBuilder& operator=(ColumnGroupBuilder&&) = default;

  private:
  int64_t id_;
  std::vector<std::string> columns_;
  std::string path_;
  FileFormat format_ = FileFormat::PARQUET;
  ColumnGroupStats stats_;
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
 *
 * @example Basic usage:
 * @code
 * // Create a new manifest
 * auto manifest = std::make_shared<Manifest>();
 *
 * // Add column groups
 * auto cg1 = std::make_shared<ColumnGroup>();
 * cg1->id = 1;
 * cg1->columns = {"id", "name", "age"};
 * cg1->path = "/data/cg1.parquet";
 * cg1->format = FileFormat::PARQUET;
 * manifest->add_column_group(cg1);
 *
 * // Query column groups
 * auto age_cg = manifest->get_column_group("age");
 * auto all_cgs = manifest->get_column_groups();
 * @endcode
 */
class Manifest {
  public:
  /**
   * @brief Constructs a new empty manifest
   *
   * Creates a manifest with no column groups and initializes internal
   * data structures for managing column group metadata.
   */
  Manifest() = default;

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

  // ==================== Column Group Management ====================

  /**
   * @brief Retrieves all column groups in the dataset
   *
   * @return Vector of shared pointers to all column group metadata
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> get_column_groups() const;

  /**
   * @brief Retrieves a specific column group by its unique identifier
   *
   * @param id Unique identifier of the column group
   * @return Shared pointer to the column group, or nullptr if not found
   */
  [[nodiscard]] std::shared_ptr<ColumnGroup> get_column_group(int64_t id) const;

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

  // ==================== Schema Management ====================

  /**
   * @brief Gets the logical schema of the dataset
   *
   * @return Shared pointer to the Arrow schema, or nullptr if not set
   */
  [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const;

  // ==================== Column Group Modification ====================

  /**
   * @brief Adds a new column group to the manifest
   *
   * The column group must have a unique ID and valid metadata.
   * This operation is transactional if a transaction is active.
   *
   * @param column_group Shared pointer to the column group to add
   * @return Result containing the assigned column group ID, or error status
   */
  arrow::Status add_column_group(std::shared_ptr<ColumnGroup> column_group);

  /**
   * @brief Removes a column group from the manifest
   *
   * This operation marks the column group for deletion but does not
   * physically remove the underlying data files. Physical cleanup
   * should be handled separately.
   *
   * @param id Unique identifier of the column group to remove
   * @return Status indicating success or error condition
   */
  arrow::Status drop_column_group(int64_t id);

  // ==================== Versioning ====================

  /**
   * @brief Gets the current version of the manifest
   *
   * @return Version number (monotonically increasing)
   */
  [[nodiscard]] int64_t version() const;

  /**
   * @brief Sets the manifest version
   *
   * @param version New version number
   */
  void set_version(int64_t version);

  // ==================== Statistics ====================

  /**
   * @brief Gets aggregate statistics across all column groups
   *
   * @return Summary statistics for the entire dataset
   */
  [[nodiscard]] ColumnGroupStats get_aggregate_stats() const;

  /**
   * @brief Refreshes statistics for all column groups
   *
   * This may involve reading file metadata to update size and row count information.
   *
   * @param fs Filesystem interface for accessing files
   * @return Status indicating success or error condition
   * @throws std::invalid_argument if fs is null
   */
  Status refresh_stats(const std::shared_ptr<arrow::fs::FileSystem>& fs);

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;  ///< All column groups in the dataset
  std::map<std::string, int64_t> column_to_group_map_;       ///< Fast lookup: column name -> column group ID
  int64_t version_;                                          ///< Current manifest version
  int64_t next_column_group_id_;                             ///< Next available column group ID

  // ==================== Internal Helper Methods ====================

  /**
   * @brief Rebuilds the column-to-group mapping for fast lookups
   */
  void rebuild_column_mapping();

  /**
   * @brief Generates the next available column group ID
   *
   * @return Unique column group identifier
   */
  int64_t generate_column_group_id();
};

}  // namespace milvus_storage::api