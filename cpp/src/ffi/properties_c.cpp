
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/properties.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iostream>

using namespace milvus_storage;
using namespace milvus_storage::api;

// TODO: move to the `properties.cpp` and provide the options in header file
template <typename T>
std::pair<bool, T> convertFunc(const std::string& str);

template <typename T, typename MemberType>
void PropertiesMapper::registerField(const std::string& field, T* obj, MemberType T::*member) {
  mappings[field] = [obj, member](const std::string& value) -> bool {
    auto [ok, val] = convertFunc<MemberType>(value);
    if (ok) {
      obj->*member = val;
    }

    return ok;
  };
}

std::pair<bool, std::string> PropertiesMapper::map(const std::unordered_map<std::string, std::string>& data) {
  bool ok = true;
  std::string failed_key;
  for (const auto& [key, value] : data) {
    if (auto it = mappings.find(key); it != mappings.end()) {
      if (!it->second(value)) {
        ok = false;
        failed_key = key;
        break;
      }
    }
  }
  return {ok, failed_key};
}

template <>
std::pair<bool, std::string> convertFunc<std::string>(const std::string& str) {
  return {true, str};
}

template <>
std::pair<bool, bool> convertFunc<bool>(const std::string& str) {
  std::string str_cpy = str;
  std::transform(str_cpy.begin(), str_cpy.end(), str_cpy.begin(), ::tolower);
  return {str_cpy == "true" || str_cpy == "false", str_cpy == "true"};
}

template <typename I>
std::pair<bool, I> convertIntFunc(const std::string& str) {
  I result;
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
  return {ec == std::errc{} && ptr == str.data() + str.size(), result};
}

template <>
std::pair<bool, int> convertFunc<int>(const std::string& str) {
  return convertIntFunc<int>(str);
}

template <>
std::pair<bool, int64_t> convertFunc<int64_t>(const std::string& str) {
  return convertIntFunc<int64_t>(str);
}

template <typename I>
std::pair<bool, std::vector<I>> convertVectorFunc(const std::string& str) {
  std::vector<I> result;
  if (!str.empty()) {
    size_t start = 0;
    size_t end = str.find(',');
    while (end != std::string::npos) {
      result.push_back(str.substr(start, end - start));
      start = end + 1;
      end = str.find(',', start);
    }
    result.push_back(str.substr(start));
  }
  return {true, result};
}

template <>
std::pair<bool, std::vector<std::string>> convertFunc<std::vector<std::string>>(const std::string& str) {
  return convertVectorFunc<std::string>(str);
}

template <typename T>
arrow::Result<T> properties_get(const std::unordered_map<std::string, std::string>& properties_map,
                                const std::string& key) {
  auto it = properties_map.find(key);
  if (it == properties_map.end()) {
    return arrow::Status::Invalid("Missing required property: " + key);
  }

  auto [ok, value] = convertFunc<T>(it->second);
  if (!ok) {
    return arrow::Status::Invalid("Invalid value for property: " + key);
  }

  return value;
}

std::shared_ptr<std::vector<std::string>> convert_string_array(const char* const* strings, size_t count) {
  std::vector<std::string> result;
  if (strings && count > 0) {
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (strings[i]) {
        result.emplace_back(strings[i]);
      }
    }
  }
  return std::make_shared<std::vector<std::string>>(result);
}

std::unordered_map<std::string, std::string> convert_properties(const ::Properties* properties) {
  std::unordered_map<std::string, std::string> result;
  if (properties && properties->properties && properties->count > 0) {
    for (size_t i = 0; i < properties->count; ++i) {
      const auto& prop = properties->properties[i];
      if (prop.key && prop.value) {
        result[prop.key] = prop.value;
      }
    }
  }
  return result;
}

std::pair<bool, std::string> create_file_system_config(
    const std::unordered_map<std::string, std::string>& properties_map, ArrowFileSystemConfig& result) {
  PropertiesMapper mapper;

  mapper.registerField("fs.address", &result, &ArrowFileSystemConfig::address);
  mapper.registerField("fs.bucket_name", &result, &ArrowFileSystemConfig::bucket_name);
  mapper.registerField("fs.access_key_id", &result, &ArrowFileSystemConfig::access_key_id);
  mapper.registerField("fs.access_key_value", &result, &ArrowFileSystemConfig::access_key_value);
  mapper.registerField("fs.root_path", &result, &ArrowFileSystemConfig::root_path);
  mapper.registerField("fs.storage_type", &result, &ArrowFileSystemConfig::storage_type);
  mapper.registerField("fs.cloud_provider", &result, &ArrowFileSystemConfig::cloud_provider);
  mapper.registerField("fs.iam_endpoint", &result, &ArrowFileSystemConfig::iam_endpoint);
  mapper.registerField("fs.log_level", &result, &ArrowFileSystemConfig::log_level);
  mapper.registerField("fs.region", &result, &ArrowFileSystemConfig::region);
  mapper.registerField("fs.use_ssl", &result, &ArrowFileSystemConfig::use_ssl);
  mapper.registerField("fs.ssl_ca_cert", &result, &ArrowFileSystemConfig::ssl_ca_cert);
  mapper.registerField("fs.use_iam", &result, &ArrowFileSystemConfig::use_iam);
  mapper.registerField("fs.use_virtual_host", &result, &ArrowFileSystemConfig::use_virtual_host);
  mapper.registerField("fs.request_timeout_ms", &result, &ArrowFileSystemConfig::request_timeout_ms);
  mapper.registerField("fs.gcp_native_without_auth", &result, &ArrowFileSystemConfig::gcp_native_without_auth);
  mapper.registerField("fs.gcp_credential_json", &result, &ArrowFileSystemConfig::gcp_credential_json);
  mapper.registerField("fs.use_custom_part_upload", &result, &ArrowFileSystemConfig::use_custom_part_upload);
  return mapper.map(properties_map);
}

arrow::Result<std::unique_ptr<ColumnGroupPolicy>> create_column_group_policy(
    const std::unordered_map<std::string, std::string>& properties_map, const std::shared_ptr<arrow::Schema>& schema) {
  ARROW_ASSIGN_OR_RAISE(auto policy_name, properties_get<std::string>(properties_map, "writer.policy"));

  if (policy_name == "single") {
    return std::make_unique<SingleColumnGroupPolicy>(schema);
  } else if (policy_name == "schema_based") {
    ARROW_ASSIGN_OR_RAISE(
        auto patterns, properties_get<std::vector<std::string>>(properties_map, "writer.split.schema_based.patterns"));
    return std::make_unique<SchemaBasedColumnGroupPolicy>(schema, patterns);
  } else if (policy_name == "size_based") {
    ARROW_ASSIGN_OR_RAISE(auto max_avg_column_size,
                          properties_get<int64_t>(properties_map, "writer.split.size_based.max_avg_column_size"));
    ARROW_ASSIGN_OR_RAISE(auto max_columns_in_group,
                          properties_get<int64_t>(properties_map, "writer.split.size_based.max_columns_in_group"));

    return std::move(std::make_unique<SizeBasedColumnGroupPolicy>(schema, max_avg_column_size, max_columns_in_group));
  }

  return nullptr;
}

// ==================== Properties C Implementation ====================

FFIResult properties_create(const char* const* keys,
                            const char* const* values,
                            size_t count,
                            ::Properties* properties) {
  // used to make sure no duplicate keys
  std::unordered_set<std::string_view> key_set;
  if (!properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "properties should not be empty");
  }

  properties->properties = nullptr;
  properties->count = 0;

  if (count == 0 || !keys || !values) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid keys/values");
  }

  properties->properties = static_cast<Property*>(malloc(sizeof(Property) * count));
  if (!properties->properties) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to malloc [size=", sizeof(Property) * count, "]");
  }

  for (size_t i = 0; i < count; ++i) {
    properties->properties[i].key = nullptr;
    properties->properties[i].value = nullptr;

    if (keys[i] && key_set.find(keys[i]) == key_set.end()) {
      size_t key_len = strlen(keys[i]) + 1;
      properties->properties[i].key = static_cast<char*>(malloc(key_len));
      if (properties->properties[i].key) {
        strcpy(properties->properties[i].key, keys[i]);
      }

      key_set.insert(keys[i]);
    } else {
      properties_free(properties);
      if (keys[i]) {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "Duplicate key: ", keys[i], " at index: ", i);
      } else {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "The key index: ", i, " is invalid");
      }
    }

    if (values[i]) {
      size_t value_len = strlen(values[i]) + 1;
      properties->properties[i].value = static_cast<char*>(malloc(value_len));
      if (properties->properties[i].value) {
        strcpy(properties->properties[i].value, values[i]);
      }
    } else {
      properties_free(properties);
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "The value index: ", i, " is invalid, key: ", keys[i]);
    }
  }

  properties->count = count;
  RETURN_SUCCESS();
}

const char* properties_get(const ::Properties* properties, const char* key) {
  if (!properties || !properties->properties || !key) {
    return nullptr;
  }

  for (size_t i = 0; i < properties->count; ++i) {
    if (properties->properties[i].key && strcmp(properties->properties[i].key, key) == 0) {
      return properties->properties[i].value;
    }
  }

  return nullptr;
}

void properties_free(::Properties* properties) {
  if (!properties) {
    return;
  }

  if (properties->properties) {
    for (size_t i = 0; i < properties->count; ++i) {
      free(properties->properties[i].key);
      free(properties->properties[i].value);
    }
    free(properties->properties);
  }
}