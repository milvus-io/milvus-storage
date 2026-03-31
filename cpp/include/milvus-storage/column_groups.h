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
#include <unordered_map>
#include <vector>

#include "milvus-storage/common/properties_convert.h"

namespace milvus_storage::api {

// Well-known property keys for ColumnGroupFile
constexpr const char* kPropertyFileSize = "file_size";
constexpr const char* kPropertyFooterSize = "footer_size";
constexpr const char* kPropertyMetadata = "metadata";

/**
 * @brief File metadata for a column group
 */
struct ColumnGroupFile {
  std::string path;                                         ///< Physical file path where the column group is stored
  int64_t start_index;                                      ///< Start index of data in the file
  int64_t end_index;                                        ///< End index of data in the file
  std::unordered_map<std::string, std::string> properties;  ///< Extensible key-value properties

  template <typename T>
  T Get(const char* key, T default_val = {}) const {
    auto it = properties.find(key);
    if (it == properties.end())
      return default_val;
    auto [ok, val] = convert::convertFunc<T>(it->second);
    return ok ? val : default_val;
  }

  template <typename T>
  void Set(const char* key, const T& value) {
    properties[key] = std::to_string(value);
  }
  void Set(const char* key, const std::string& value) { properties[key] = value; }
  void Set(const char* key, const char* value) { properties[key] = value; }
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

static ColumnGroups copy_column_groups(const ColumnGroups& source) {
  ColumnGroups dest(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    if (source[i]) {
      dest[i] = std::make_shared<ColumnGroup>(*source[i]);
    } else {
      dest[i] = nullptr;
    }
  }
  return dest;
}

}  // namespace milvus_storage::api
