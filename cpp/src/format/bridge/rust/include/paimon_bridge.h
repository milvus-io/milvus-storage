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
#include <string_view>
#include <unordered_map>
#include <vector>

#include <arrow/result.h>

namespace milvus_storage::paimon {

using StorageOptions = std::unordered_map<std::string, std::string>;

struct PaimonFileInfo {
  std::string path;
  uint64_t file_size;
  std::string metadata_json;
};

arrow::Status MakePaimonBridgeErrorStatus(std::string_view message);

arrow::Result<std::vector<PaimonFileInfo>> PlanFiles(const std::string& table_location,
                                                     int64_t snapshot_id,
                                                     const std::string& scan_mode,
                                                     const StorageOptions& storage_options);

arrow::Result<std::vector<uint64_t>> ReadDeletionVector(const std::string& path,
                                                        uint64_t offset,
                                                        uint64_t length,
                                                        int64_t expected_cardinality,
                                                        const StorageOptions& storage_options);

arrow::Result<int64_t> CreateTestTable(const std::string& table_location,
                                       uint64_t num_rows,
                                       const std::string& mode,
                                       const std::vector<int64_t>& deleted_positions = {},
                                       const std::string& file_format = "parquet",
                                       uint32_t dimension = 0);

}  // namespace milvus_storage::paimon
