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
#include <arrow/filesystem/filesystem.h>
#include <string>

namespace milvus_storage {

static const std::string GROUP_DELIMITER = ";";
static const std::string COLUMN_DELIMITER = ",";
static const std::string ROW_GROUP_SIZE_META_KEY = "row_group_size";
static const std::string COLUMN_OFFSETS_META_KEY = "column_offsets";

class PackedMetaSerde {
  public:
  // Serialize a vector of size_t to a byte array and convert it to a string
  static std::string SerializeRowGroupSizes(const std::vector<size_t>& sizes) {
    std::vector<uint8_t> byteArray(sizes.size() * sizeof(size_t));
    std::memcpy(byteArray.data(), sizes.data(), byteArray.size());
    return std::string(byteArray.begin(), byteArray.end());
  }

  // Deserialize a string back to a vector of size_t
  static std::vector<size_t> DeserializeRowGroupSizes(const std::string& input) {
    std::vector<uint8_t> byteArray(input.begin(), input.end());
    std::vector<size_t> sizes(byteArray.size() / sizeof(size_t));
    std::memcpy(sizes.data(), byteArray.data(), byteArray.size());
    return sizes;
  }

  static std::string SerializeColumnOffsets(const std::vector<std::vector<int64_t>>& column_offsets) {
    std::stringstream ss;
    for (size_t i = 0; i < column_offsets.size(); ++i) {
      if (i > 0) {
        ss << GROUP_DELIMITER;
      }

      for (size_t j = 0; j < column_offsets[i].size(); ++j) {
        if (j > 0) {
          ss << COLUMN_DELIMITER;
        }
        ss << column_offsets[i][j];
      }
    }

    auto s = ss.str();
    return s;
  }

  static std::vector<std::vector<int64_t>> DeserializeColumnOffsets(const std::string& input) {
    std::vector<std::vector<int64_t>> column_offsets;

    size_t group_start = 0;
    size_t group_end = input.find(GROUP_DELIMITER);

    while (group_start != std::string::npos) {
      std::string group = input.substr(group_start, group_end - group_start);
      std::vector<int64_t> group_indices;

      size_t column_start = 0;
      size_t column_end = group.find(COLUMN_DELIMITER);
      while (column_start != std::string::npos) {
        std::string column = group.substr(column_start, column_end - column_start);
        if (!column.empty()) {
          group_indices.push_back(std::stoll(column));
        }
        column_start = (column_end == std::string::npos) ? std::string::npos : column_end + COLUMN_DELIMITER.size();
        column_end = group.find(COLUMN_DELIMITER, column_start);
      }

      if (!group_indices.empty()) {
        column_offsets.push_back(group_indices);
      }

      group_start = (group_end == std::string::npos) ? std::string::npos : group_end + GROUP_DELIMITER.size();
      group_end = input.find(GROUP_DELIMITER, group_start);
    }

    return column_offsets;
  }
};

}  // namespace milvus_storage
