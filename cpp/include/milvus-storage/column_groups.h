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
namespace milvus_storage::api {

/**
 * @brief File metadata for a column group
 */
struct ColumnGroupFile {
  std::string path;               ///< Physical file path where the column group is stored
  int64_t start_index;            ///< Start index of data in the file
  int64_t end_index;              ///< End index of data in the file
  std::vector<uint8_t> metadata;  ///< Optional metadata for external table
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
 * @brief Type alias for a collection of column groups
 *
 * ColumnGroups is a type alias for std::vector<std::shared_ptr<ColumnGroup>>.
 * This provides backward compatibility and cleaner code.
 */
using ColumnGroups = std::vector<std::shared_ptr<ColumnGroup>>;

}  // namespace milvus_storage::api
