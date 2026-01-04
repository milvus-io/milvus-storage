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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <optional>

#include <arrow/status.h>
#include <istream>
#include <ostream>

#include "milvus-storage/column_groups.h"

namespace milvus_storage::api {

constexpr int32_t MANIFEST_VERSION = 1;

/**
 * @brief Type of delta log entry
 */
enum class DeltaLogType {
  PRIMARY_KEY = 0,  // Primary key delete (default)
  POSITIONAL = 1,   // Positional delete
  EQUALITY = 2,     // Equality delete
};

/**
 * @brief Delta log entry structure
 */
struct DeltaLog {
  std::string path;     ///< Relative path to the delta log file
  DeltaLogType type;    ///< Type of delta log
  int64_t num_entries;  ///< Number of entries in the delta log
};

/**
 * @brief Manifest class containing column groups, delta logs, and stats
 *
 * The Manifest class wraps ColumnGroups and extends it with delta logs and stats.
 */
class Manifest final {
  public:
  explicit Manifest(ColumnGroups column_groups = {},
                    const std::vector<DeltaLog>& delta_logs = {},
                    const std::map<std::string, std::vector<std::string>>& stats = {},
                    uint32_t version = MANIFEST_VERSION);

  // Disable default copy constructor and assignment operator
  Manifest(const Manifest&) = delete;
  Manifest& operator=(const Manifest&) = delete;

  // Enable move constructor and assignment operator
  Manifest(Manifest&&) = default;
  Manifest& operator=(Manifest&&) = default;

  ~Manifest() = default;

  /**
   * @brief Serializes the manifest to a standard output stream
   */
  [[nodiscard]] arrow::Status serialize(std::ostream& output_stream) const;

  /**
   * @brief Deserializes the manifest from a standard input stream
   */
  arrow::Status deserialize(std::istream& input_stream);

  /**
   * @brief Get all column groups
   */
  [[nodiscard]] ColumnGroups& columnGroups() { return column_groups_; }

  /**
   * @brief Find the column group that contains the specified column
   * @param column_name Name of the column to locate
   * @return Pointer to the column group containing the column, or nullptr if not found
   */
  [[nodiscard]] std::shared_ptr<ColumnGroup> getColumnGroup(const std::string& column_name) const;

  /**
   * @brief Find the column group by its index
   * @param index Index of the column group
   * @return Shared pointer to the column group, or nullptr if out of range
   */
  [[nodiscard]] std::shared_ptr<ColumnGroup> getColumnGroup(size_t index) const;

  /**
   * @brief Get all delta log entries
   */
  [[nodiscard]] std::vector<DeltaLog>& deltaLogs() { return delta_logs_; }

  /**
   * @brief Get all stats
   */
  [[nodiscard]] std::map<std::string, std::vector<std::string>>& stats() { return stats_; }

  /**
   * @brief Resolve relative paths to absolute paths using base path
   * @param base_path Base path to prepend to relative paths
   */
  arrow::Status resolve_paths(const std::string& base_path);

  /**
   * @brief Get the manifest format version
   */
  [[nodiscard]] int32_t version() const { return version_; }

  private:
  int32_t version_;                                        ///< Manifest format version
  ColumnGroups column_groups_;                             ///< Column groups in the dataset
  std::vector<DeltaLog> delta_logs_;                       ///< Delta log entries
  std::map<std::string, std::vector<std::string>> stats_;  ///< Stats file lists keyed by stat name
};

using ManifestPtr = std::shared_ptr<Manifest>;

}  // namespace milvus_storage::api
