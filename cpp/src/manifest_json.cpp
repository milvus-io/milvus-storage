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

#include "milvus-storage/manifest_json.h"

namespace milvus_storage::api {

// ==================== ColumnGroup JSON Serialization ====================

void to_json(nlohmann::json& j, const ColumnGroup& cg) {
  j = nlohmann::json{{"columns", cg.columns}, {"paths", cg.paths}, {"format", cg.format}};
}

void from_json(const nlohmann::json& j, ColumnGroup& cg) {
  j.at("columns").get_to(cg.columns);
  j.at("paths").get_to(cg.paths);
  j.at("format").get_to(cg.format);
}

// ==================== Manifest JSON Serialization ====================

void to_json(nlohmann::json& j, const Manifest& m) {
  j["version"] = m.version();
  j["column_groups"] = nlohmann::json::array();

  const auto& groups = m.get_column_groups();
  for (const auto& group : groups) {
    if (group) {
      j["column_groups"].push_back(*group);
    }
  }
}

void from_json(const nlohmann::json& j, Manifest& manifest) {
  std::vector<std::shared_ptr<ColumnGroup>> column_groups;

  if (j.contains("column_groups") && j["column_groups"].is_array()) {
    for (const auto& cg_json : j["column_groups"]) {
      auto cg = std::make_shared<ColumnGroup>();
      *cg = cg_json.get<ColumnGroup>();
      column_groups.push_back(cg);
    }
  }

  int64_t version = j.value("version", 0);
  manifest = Manifest(std::move(column_groups), version);
}

}  // namespace milvus_storage::api

namespace milvus_storage {

// ==================== JsonManifestSerDe Implementation ====================

std::pair<bool, std::string> JsonManifestSerDe::Serialize(const std::shared_ptr<api::Manifest>& manifest) {
  try {
    nlohmann::json j;
    api::to_json(j, *manifest);
    return {true, j.dump(2)};
  } catch (const std::exception&) {
    return {false, {}};
  }
}

std::shared_ptr<api::Manifest> JsonManifestSerDe::Deserialize(const std::string& input) {
  try {
    if (input.empty()) {
      return nullptr;
    }
    nlohmann::json j;
    j = nlohmann::json::parse(input, nullptr, false);
    // Check if parsing was successful (j should not be discarded)
    if (j.is_discarded()) {
      return nullptr;
    }

    auto manifest = std::make_shared<api::Manifest>();
    // Use from_json to populate the manifest
    api::from_json(j, *manifest);
    return manifest;
  } catch (...) {
    return nullptr;
  }
}

std::shared_ptr<api::Manifest> JsonManifestSerDe::Deserialize(const std::string_view& input) {
  try {
    if (input.empty()) {
      return nullptr;
    }
    nlohmann::json j;
    j = nlohmann::json::parse(input, nullptr, false);
    // Check if parsing was successful (j should not be discarded)
    if (j.is_discarded()) {
      return nullptr;
    }

    auto manifest = std::make_shared<api::Manifest>();
    // Use from_json to populate the manifest
    api::from_json(j, *manifest);
    return manifest;
  } catch (...) {
    return nullptr;
  }
}

}  // namespace milvus_storage
