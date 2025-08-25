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

namespace milvus_storage::api {

/**
 * @brief Configuration properties for read operations
 */
struct ReadProperties {
  std::string cipher_type;
  std::string cipher_key;
  std::string cipher_metadata;
};

const ReadProperties default_read_properties = {
    .cipher_type = "",
    .cipher_key = "",
    .cipher_metadata = "",
};

class ReadPropertiesBuilder {
  public:
  ReadPropertiesBuilder() : properties_(default_read_properties) {}

  ReadPropertiesBuilder& with_cipher_type(const std::string& cipher_type) {
    properties_.cipher_type = cipher_type;
    return *this;
  }

  ReadPropertiesBuilder& with_cipher_key(const std::string& cipher_key) {
    properties_.cipher_key = cipher_key;
    return *this;
  }

  ReadPropertiesBuilder& with_cipher_metadata(const std::string& cipher_metadata) {
    properties_.cipher_metadata = cipher_metadata;
    return *this;
  }

  ReadProperties build() const { return properties_; }

  private:
  ReadProperties properties_;
};

}  // namespace milvus_storage::api