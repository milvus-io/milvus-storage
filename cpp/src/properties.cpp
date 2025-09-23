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

#include "milvus-storage/properties.h"
#include <algorithm>

namespace milvus_storage::api {

// Definitions for Key constants declared in header
// Common properties
const Key<int> BufferSizeKey{"buffer.size", 32 * 1024 * 1024};

// Encryption properties
const Key<std::string> EncryptionCipherTypeKey{"encryption.cipher.type", "aes_gcm_v1"};
const Key<std::string> EncryptionCipherKeyKey{"encryption.cipher.key", ""};
const Key<std::string> EncryptionCipherMetadataKey{"encryption.cipher.metadata", ""};

// Reader properties
const Key<int> ReadBatchSizeKey{"read.batch.size", 1024};

// Writer properties
const Key<int> MultiPartUploadSizeKey{"write.mpu.size", 10 * 1024 * 1024};
const Key<std::string> WriteCompressionKey{"write.compression", "zstd"};
const Key<int> WriteCompressionLevelKey{"write.compression.level", 5};
const Key<bool> WriteEnableDictionaryKey{"write.enable.dictionary", true};

//=================== Template specialization implementations ===================
// Internal template function for type conversion (not exposed in header)
template <typename T>
T convert_from_string(const std::string& str, const T& default_value);

template <>
std::string convert_from_string<std::string>(const std::string& str, const std::string& default_value) {
  return str;
}

template <>
int convert_from_string<int>(const std::string& str, const int& default_value) {
  try {
    return std::stoi(str);
  } catch (const std::exception& e) {
    return default_value;
  }
}

template <>
int64_t convert_from_string<int64_t>(const std::string& str, const int64_t& default_value) {
  try {
    return std::stoll(str);
  } catch (const std::exception& e) {
    return default_value;
  }
}

template <>
double convert_from_string<double>(const std::string& str, const double& default_value) {
  try {
    return std::stod(str);
  } catch (const std::exception& e) {
    return default_value;
  }
}

template <>
bool convert_from_string<bool>(const std::string& str, const bool& default_value) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

  if (lower_str == "true" || lower_str == "1" || lower_str == "yes" || lower_str == "on") {
    return true;
  } else if (lower_str == "false" || lower_str == "0" || lower_str == "no" || lower_str == "off") {
    return false;
  } else {
    return default_value;
  }
}

//=================== Template function implementations ===================
template <typename T>
T GetValue(const Properties& properties, const Key<T>& key) {
  auto it = properties.find(key.name());
  if (it == properties.end()) {
    return key.default_value();
  }
  return convert_from_string<T>(it->second, key.default_value());
}

template <typename T>
void SetValue(Properties& properties, const Key<T>& key, T value) {
  properties[key.name()] = std::to_string(value);
}

// Specialization for string type
template <typename T>
void SetValue(Properties& properties, const Key<T>& key, const char* value) {
  properties[key.name()] = value;
}

//=================== Explicit template instantiations ===================
template std::string GetValue<std::string>(const Properties&, const Key<std::string>&);
template int GetValue<int>(const Properties&, const Key<int>&);
template int64_t GetValue<int64_t>(const Properties&, const Key<int64_t>&);
template double GetValue<double>(const Properties&, const Key<double>&);
template bool GetValue<bool>(const Properties&, const Key<bool>&);

template void SetValue<int>(Properties&, const Key<int>&, int);
template void SetValue<int64_t>(Properties&, const Key<int64_t>&, int64_t);
template void SetValue<double>(Properties&, const Key<double>&, double);
template void SetValue<bool>(Properties&, const Key<bool>&, bool);

template void SetValue<std::string>(Properties&, const Key<std::string>&, const char*);
template void SetValue<int>(Properties&, const Key<int>&, const char*);
template void SetValue<int64_t>(Properties&, const Key<int64_t>&, const char*);
template void SetValue<double>(Properties&, const Key<double>&, const char*);
template void SetValue<bool>(Properties&, const Key<bool>&, const char*);

}  // namespace milvus_storage::api