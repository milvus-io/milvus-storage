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

#include <unordered_map>
#include <string>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

namespace milvus_storage::api {
using Properties = std::unordered_map<std::string, std::string>;

// Template class to represent config keys with their expected value type
template <typename T>
class Key {
  public:
  explicit Key(std::string key_name, T default_value) : key_name_(std::move(key_name)), default_value_(default_value) {}

  [[nodiscard]] const T& default_value() const { return default_value_; }

  [[nodiscard]] const std::string& name() const { return key_name_; }

  // Type alias for the expected value type
  using value_type = T;

  private:
  std::string key_name_;
  T default_value_;
};

template <typename T>
T GetValue(const Properties& properties, const Key<T>& key);

template <typename T>
void SetValue(Properties& properties, const Key<T>& key, T value);

template <typename T>
void SetValue(Properties& properties, const Key<T>& key, const char* value);

// Extern declarations for predefined Key constants
// Common properties
extern const Key<int> BufferSizeKey;

// Encryption properties
extern const Key<std::string> EncryptionCipherTypeKey;
extern const Key<std::string> EncryptionCipherKeyKey;
extern const Key<std::string> EncryptionCipherMetadataKey;

// Reader properties
extern const Key<int> ReadBatchSizeKey;

// Writer properties
extern const Key<int> MultiPartUploadSizeKey;
extern const Key<std::string> WriteCompressionKey;
extern const Key<int> WriteCompressionLevelKey;
extern const Key<bool> WriteEnableDictionaryKey;

}  // namespace milvus_storage::api