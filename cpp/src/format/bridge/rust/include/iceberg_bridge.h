// Copyright 2024 Zilliz
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
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace milvus_storage::iceberg {

class IcebergException : public std::runtime_error {
  public:
  explicit IcebergException(const std::string& message) : std::runtime_error(message) {}
};

/// Per-file info returned from PlanFiles
struct IcebergFileInfo {
  std::string data_file_path;                 ///< Absolute data file URI
  uint64_t record_count;                      ///< Physical row count (pre-delete)
  uint64_t num_deleted_rows;                  ///< Number of rows deleted from this data file
  std::vector<uint8_t> delete_metadata_json;  ///< JSON delete file refs (empty if no deletes)
};

/// Plan files for an Iceberg table snapshot.
/// Returns one IcebergFileInfo per data file.
///
/// @param metadata_location Table metadata location (e.g., "s3://bucket/table/metadata/v1.metadata.json")
/// @param snapshot_id Which snapshot to scan
/// @param storage_options S3/cloud config as key-value pairs
/// @return Vector of file info, one per data file in the snapshot
std::vector<IcebergFileInfo> PlanFiles(const std::string& metadata_location,
                                       int64_t snapshot_id,
                                       const std::unordered_map<std::string, std::string>& storage_options);

/// Info returned after creating a test Iceberg table.
struct IcebergTestTableInfo {
  std::string metadata_location;  ///< Path to metadata.json
  int64_t snapshot_id;            ///< Snapshot ID
  std::string data_file_uri;      ///< URI of data file
};

/// Create a test Iceberg table on local filesystem or cloud storage.
///
/// Schema: id (int64), name (string), value (float64)
/// Data: id=0..N-1, name="row_0".."row_{N-1}", value=i*1.5
///
/// For cloud storage, pass iceberg-format storage options (e.g., s3.access-key-id).
/// For local filesystem, pass empty storage_options.
IcebergTestTableInfo CreateTestTable(const std::string& table_dir,
                                     uint64_t num_rows,
                                     bool with_positional_deletes,
                                     const std::vector<int64_t>& deleted_positions,
                                     const std::unordered_map<std::string, std::string>& storage_options = {});

}  // namespace milvus_storage::iceberg
