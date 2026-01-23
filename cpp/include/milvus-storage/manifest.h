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

constexpr int32_t MANIFEST_VERSION = 2;
constexpr int32_t MANIFEST_VERSION_MIN = 1;  // Minimum supported version for backward compatibility

/**
 * @brief Metadata for a single LOB (Large Object) file
 *
 * LOB files store large text/binary data separately from the main columnar storage.
 * This structure tracks the file location and row statistics for garbage collection
 * and query optimization.
 */
struct LobFileInfo {
  std::string path;         ///< Relative path to the LOB file
  int64_t field_id;         ///< Field ID this LOB file belongs to
  int64_t total_rows;       ///< Total number of rows in the LOB file
  int64_t valid_rows;       ///< Number of valid (non-deleted) rows
  int64_t file_size_bytes;  ///< Size of the LOB file in bytes

  LobFileInfo() : field_id(0), total_rows(0), valid_rows(0), file_size_bytes(0) {}
  LobFileInfo(std::string p, int64_t fid, int64_t total, int64_t valid, int64_t size)
      : path(std::move(p)), field_id(fid), total_rows(total), valid_rows(valid), file_size_bytes(size) {}

  bool operator==(const LobFileInfo& other) const {
    return path == other.path && field_id == other.field_id && total_rows == other.total_rows &&
           valid_rows == other.valid_rows && file_size_bytes == other.file_size_bytes;
  }
};

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
                    const std::vector<LobFileInfo>& lob_files = {},
                    uint32_t version = MANIFEST_VERSION);

  // Enable move constructor and assignment operator
  Manifest(Manifest&&) = default;
  Manifest& operator=(Manifest&&) = default;

  ~Manifest() = default;

  /**
   * @brief Serializes the manifest to a standard output stream
   */
  [[nodiscard]] arrow::Status serialize(std::ostream& output_stream,
                                        const std::optional<std::string>& base_path = std::nullopt) const;

  /**
   * @brief Deserializes the manifest from a standard input stream
   */
  arrow::Status deserialize(std::istream& input_stream, const std::optional<std::string>& base_path = std::nullopt);

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
   * @brief Get all delta log entries
   */
  [[nodiscard]] std::vector<DeltaLog>& deltaLogs() { return delta_logs_; }

  /**
   * @brief Get all stats
   */
  [[nodiscard]] std::map<std::string, std::vector<std::string>>& stats() { return stats_; }

  /**
   * @brief Get all LOB file info entries
   */
  [[nodiscard]] std::vector<LobFileInfo>& lobFiles() { return lob_files_; }

  /**
   * @brief Get all LOB file info entries (const)
   */
  [[nodiscard]] const std::vector<LobFileInfo>& lobFiles() const { return lob_files_; }

  /**
   * @brief Get LOB files for a specific field ID
   */
  [[nodiscard]] std::vector<LobFileInfo> getLobFilesForField(int64_t field_id) const {
    std::vector<LobFileInfo> result;
    for (const auto& file : lob_files_) {
      if (file.field_id == field_id) {
        result.push_back(file);
      }
    }
    return result;
  }

  /**
   * @brief Get the manifest format version
   */
  [[nodiscard]] int32_t version() const { return version_; }

  private:
  Manifest(const Manifest&);
  Manifest& operator=(const Manifest&);

  /**
   * @brief Copy the manifest and convert paths to relative
   */
  Manifest ToRelativePaths(const std::string& base_path) const;

  /**
   * @brief Convert paths to absolute
   */
  void ToAbsolutePaths(const std::string& base_path);

  private:
  int32_t version_;                                        ///< Manifest format version
  ColumnGroups column_groups_;                             ///< Column groups in the dataset
  std::vector<DeltaLog> delta_logs_;                       ///< Delta log entries
  std::map<std::string, std::vector<std::string>> stats_;  ///< Stats file lists keyed by stat name
  std::vector<LobFileInfo> lob_files_;                     ///< LOB file metadata for TEXT/BLOB columns
};

using ManifestPtr = std::shared_ptr<Manifest>;

}  // namespace milvus_storage::api
