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

class PackedMetaSerde {
  public:
  // Serialize a vector of size_t to a byte array and convert it to a string
  static std::string serialize(const std::vector<size_t>& sizes) {
    std::vector<uint8_t> byteArray(sizes.size() * sizeof(size_t));
    std::memcpy(byteArray.data(), sizes.data(), byteArray.size());
    return std::string(byteArray.begin(), byteArray.end());
  }

  // Deserialize a string back to a vector of size_t
  static std::vector<size_t> deserialize(const std::string& input) {
    std::vector<uint8_t> byteArray(input.begin(), input.end());
    std::vector<size_t> sizes(byteArray.size() / sizeof(size_t));
    std::memcpy(sizes.data(), byteArray.data(), byteArray.size());
    return sizes;
  }
};

}  // namespace milvus_storage
