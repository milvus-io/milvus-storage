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

#include "milvus-storage/ffi_c.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/writer.h"

// no need namespace here
class PropertiesMapper final {
  public:
  template <typename T, typename MemberType>
  void registerField(const std::string& field, T* obj, MemberType T::*member);

  std::pair<bool, std::string> map(const std::unordered_map<std::string, std::string>& data);

  private:
  std::unordered_map<std::string, std::function<bool(const std::string&)>> mappings;
};

// Helper function to convert C Properties to C++ Properties
std::shared_ptr<std::vector<std::string>> convert_string_array(const char* const* strings, size_t count);
std::unordered_map<std::string, std::string> convert_properties(const ::Properties* properties);
std::pair<bool, std::string> create_file_system_config(
    const std::unordered_map<std::string, std::string>& properties_map, milvus_storage::ArrowFileSystemConfig& result);
arrow::Result<std::unique_ptr<milvus_storage::api::ColumnGroupPolicy>> create_column_group_policy(
    const std::unordered_map<std::string, std::string>& properties_map, const std::shared_ptr<arrow::Schema>& schema);