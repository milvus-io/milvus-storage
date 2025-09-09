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

#include <iostream>
#include <nlohmann/json.hpp>
#include "milvus-storage/manifest.h"

namespace milvus_storage {

// ==================== JSON Serialization Functions ====================

/**
 * @brief JSON serialization for FileFormat enum
 */
void to_json(nlohmann::json& j, const std::string& format);
void from_json(const nlohmann::json& j, std::string& format);

/**
 * @brief JSON serialization for ColumnGroup struct
 */
void to_json(nlohmann::json& j, const api::ColumnGroup& cg);
void from_json(const nlohmann::json& j, api::ColumnGroup& cg);

/**
 * @brief JSON serialization for Manifest class
 */
void to_json(nlohmann::json& j, const api::Manifest& m);
void from_json(const nlohmann::json& j, api::Manifest& manifest);

// ==================== JsonManifestSerDe Implementation ====================

/**
 * @brief JSON-based implementation of ManifestSerDe using nlohmann::json
 *
 * Provides JSON serialization and deserialization for Manifest objects.
 * Uses pretty-printed JSON format for human readability.
 */
class JsonManifestSerDe : public api::ManifestSerDe {
  public:
  /**
   * @brief Serializes a manifest to JSON format
   *
   * @param manifest The manifest to serialize
   * @param output Output stream to write JSON to
   * @return true if serialization was successful, false otherwise
   */
  bool Serialize(const std::shared_ptr<api::Manifest>& manifest, std::ostream& output) override;

  /**
   * @brief Deserializes a manifest from JSON format
   *
   * @param input Input stream containing JSON data
   * @param manifest Output parameter for the deserialized manifest
   * @return true if deserialization was successful, false otherwise
   */
  std::shared_ptr<api::Manifest> Deserialize(std::istream& input) override;
};

}  // namespace milvus_storage
