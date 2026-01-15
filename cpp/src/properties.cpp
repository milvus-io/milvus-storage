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
#include <charconv>
#include <string>
#include <vector>
#include <optional>
#include <sstream>
#include <cassert>
#include <cctype>
#include <type_traits>
#include <utility>
#include <cstdint>
#include <system_error>
#include <thread>

#include <boost/algorithm/string/trim.hpp>

#include "milvus-storage/ffi_c.h"  // for FFI Properties definition
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::api {
namespace convert {

template <typename T>
std::pair<bool, T> convertFunc(const std::string& str);

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
std::pair<bool, int32_t> convertFunc<int32_t>(const std::string& str) {
  return convertIntFunc<int32_t>(str);
}

template <>
std::pair<bool, int64_t> convertFunc<int64_t>(const std::string& str) {
  return convertIntFunc<int64_t>(str);
}

template <>
std::pair<bool, uint32_t> convertFunc<uint32_t>(const std::string& str) {
  return convertIntFunc<uint32_t>(str);
}

template <>
std::pair<bool, uint64_t> convertFunc<uint64_t>(const std::string& str) {
  return convertIntFunc<uint64_t>(str);
}

template <typename I>
std::pair<bool, std::vector<I>> convertVectorFunc(const std::string& str) {
  std::vector<I> result;
  if (!str.empty()) {
    size_t start = 0;
    size_t end = str.find(',');
    while (end != std::string::npos) {
      result.push_back(boost::trim_copy(str.substr(start, end - start)));
      start = end + 1;
      end = str.find(',', start);
    }
    result.push_back(boost::trim_copy(str.substr(start)));
  }
  return {true, result};
}

template <>
std::pair<bool, std::vector<std::string>> convertFunc<std::vector<std::string>>(const std::string& str) {
  return convertVectorFunc<std::string>(str);
}

}  // namespace convert

PropertiesValidator::PropertiesValidator() : fn(nullptr) {}
PropertiesValidator::PropertiesValidator(ValidatorFunc f) : fn(std::move(f)) {}

std::optional<std::string> PropertiesValidator::operator()(const PropertyInfo& property_info,
                                                           const std::string& v) const {
  if (!fn)
    return std::nullopt;
  return fn(property_info, v);
}

PropertiesValidator operator+(const PropertiesValidator& a, const PropertiesValidator& b) {
  return PropertiesValidator(
      [a, b](const PropertyInfo& property_info, const std::string& v) -> std::optional<std::string> {
        if (a.fn) {
          auto r = a.fn(property_info, v);
          if (r != std::nullopt) {
            return r;
          }
        }

        if (b.fn) {
          auto r = b.fn(property_info, v);
          if (r != std::nullopt) {
            return r;
          }
        }

        return std::nullopt;
      });
}

bool ValidatePropertyValueType(const PropertyInfo& property_info, const std::string& v) {
  switch (property_info.type) {
    case PropertyType::STRING:
      return true;  // any string is valid
    case PropertyType::BOOL: {
      auto [ok, _val] = convert::convertFunc<bool>(v);
      return ok;
    }
    case PropertyType::INT32: {
      auto [ok, _val] = convert::convertFunc<int32_t>(v);
      return ok;
    }
    case PropertyType::INT64: {
      auto [ok, _val] = convert::convertFunc<int64_t>(v);
      return ok;
    }
    case PropertyType::VECTOR_STR: {
      auto [ok, _val] = convert::convertFunc<std::vector<std::string>>(v);
      return ok;
    }
    case PropertyType::UINT32: {
      auto [ok, _val] = convert::convertFunc<uint32_t>(v);
      return ok;
    }
    case PropertyType::UINT64: {
      auto [ok, _val] = convert::convertFunc<uint64_t>(v);
      return ok;
    }
    default:
      return false;
  }
}

template <typename T>
T GetPropertyValue(const PropertyInfo& property_info, const std::string& v) {
  switch (property_info.type) {
    case PropertyType::STRING:
      if constexpr (std::is_same_v<T, std::string>) {
        return v;
      }
      break;
    case PropertyType::BOOL:
      if constexpr (std::is_same_v<T, bool>) {
        auto [ok, val] = convert::convertFunc<bool>(v);
        if (ok) {
          return val;
        }
      }
      break;
    case PropertyType::INT32:
      if constexpr (std::is_same_v<T, int>) {
        auto [ok, val] = convert::convertFunc<int>(v);
        if (ok) {
          return val;
        }
      }
      break;
    case PropertyType::INT64:
      if constexpr (std::is_same_v<T, int64_t>) {
        auto [ok, val] = convert::convertFunc<int64_t>(v);
        if (ok) {
          return val;
        }
      }
      break;
    case PropertyType::VECTOR_STR:
      if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        auto [ok, val] = convert::convertFunc<std::vector<std::string>>(v);
        if (ok) {
          return val;
        }
      }
      break;
    case PropertyType::UINT32:
      if constexpr (std::is_same_v<T, uint32_t>) {
        auto [ok, val] = convert::convertFunc<uint32_t>(v);
        if (ok) {
          return val;
        }
      }
      break;
    case PropertyType::UINT64:
      if constexpr (std::is_same_v<T, uint64_t>) {
        auto [ok, val] = convert::convertFunc<uint64_t>(v);
        if (ok) {
          return val;
        }
      }
    default:
      break;
  }

  assert(false && "type mismatch and no default value");
  throw std::runtime_error("Logical fault: type mismatch and no default value");
}

PropertyVariant GetPropertyValue(const PropertyInfo& property_info, const std::string& v) {
  switch (property_info.type) {
    case PropertyType::STRING:
      return GetPropertyValue<std::string>(property_info, v);
    case PropertyType::BOOL:
      return GetPropertyValue<bool>(property_info, v);
    case PropertyType::INT32:
      return GetPropertyValue<int>(property_info, v);
    case PropertyType::INT64:
      return GetPropertyValue<int64_t>(property_info, v);
    case PropertyType::VECTOR_STR:
      return GetPropertyValue<std::vector<std::string>>(property_info, v);
    case PropertyType::UINT32:
      return GetPropertyValue<uint32_t>(property_info, v);
    case PropertyType::UINT64:
      return GetPropertyValue<uint64_t>(property_info, v);
    default:
      assert(false && "unknown property type");
      return nullptr;
  }

  assert(false && "unreachable");
  return nullptr;
}

// Validator: check the type
static PropertiesValidator ValidatePropertyType() {
  return PropertiesValidator([](const PropertyInfo& property_info, const std::string& v) -> std::optional<std::string> {
    if (ValidatePropertyValueType(property_info, v)) {
      return std::nullopt;
    } else {
      return std::string("type mismatch: [key=") + property_info.name + "] [value=" + v + "]";
    }
  });
}

// Validator: check the allowed enum values
template <typename T, typename... Allowed>
static PropertiesValidator ValidatePropertyEnum(Allowed&&... allowed) {
  // capture allowed values as T
  std::vector<T> allowed_values{T(std::forward<Allowed>(allowed))...};

  return PropertiesValidator(
      [allowed_values = std::move(allowed_values)](const PropertyInfo& property_info,
                                                   const std::string& v) -> std::optional<std::string> {
        // Expect PropertyInfo to hold a PropertyVariant named `default_value`
        // to get the value, should checked by `ValidatePropertyType` before
        T val = GetPropertyValue<T>(property_info, v);

        for (const auto& a : allowed_values) {
          if (val == a)
            return std::nullopt;  // valid
        }

        std::ostringstream oss;
        for (size_t i = 0; i < allowed_values.size(); ++i) {
          if (i)
            oss << ", ";
          oss << allowed_values[i];
        }

        std::ostringstream msg;
        msg << "value '" << val << "' not in allowed set: [" << oss.str() << "]";
        return msg.str();
      });
}

// Validator: check the allowed enum values
template <typename T>
static PropertiesValidator ValidatePropertyRange(T min, T max) {
  return PropertiesValidator(
      [minval = std::move(min), maxval = std::move(max)](const PropertyInfo& property_info,
                                                         const std::string& v) -> std::optional<std::string> {
        // Expect PropertyInfo to hold a PropertyVariant named `default_value`
        // to get the value, should checked by `ValidatePropertyType` before
        T val = GetPropertyValue<T>(property_info, v);

        if (val >= minval && val <= maxval) {
          return std::nullopt;  // valid
        }

        // invalid
        std::ostringstream oss;
        oss << "value '" << val << "' not in range: [" << minval << ", " << maxval << "]";
        return oss.str();
      });
}

#define REGISTER_PROPERTY(name, type, description, default_value, validator)                  \
  {                                                                                           \
    name, PropertyInfo { name, type, description, PropertyVariant(default_value), validator } \
  }

static std::unordered_map<std::string, PropertyInfo> property_infos = {
    // --- global properties define ---
    REGISTER_PROPERTY(
        PROPERTY_FORMAT,
        PropertyType::STRING,
        "The format of the storage. Options: parquet, vortex.",
        LOON_FORMAT_PARQUET,
        ValidatePropertyType() + ValidatePropertyEnum<std::string>(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX)),
    // --- fs properties define ---
    REGISTER_PROPERTY(PROPERTY_FS_ADDRESS,
                      PropertyType::STRING,
                      "The address of the filesystem storage service.",
                      "localhost:9000",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_BUCKET_NAME,
                      PropertyType::STRING,
                      "The bucket name in the filesystem storage service.",
                      "a-bucket",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_ACCESS_KEY_ID,
                      PropertyType::STRING,
                      "The access key id for the filesystem storage service.",
                      "minioadmin",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_ACCESS_KEY_VALUE,
                      PropertyType::STRING,
                      "The access key value for the filesystem storage service.",
                      "minioadmin",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_ROOT_PATH,
                      PropertyType::STRING,
                      "The root path in the bucket for storing data.",
                      "files",
                      std::nullopt),
    REGISTER_PROPERTY(
        PROPERTY_FS_STORAGE_TYPE,
        PropertyType::STRING,
        "The type of the filesystem storage service.",
        LOON_FS_TYPE_LOCAL,
        ValidatePropertyType() + ValidatePropertyEnum<std::string>(LOON_FS_TYPE_LOCAL, LOON_FS_TYPE_REMOTE)),
    REGISTER_PROPERTY(PROPERTY_FS_CLOUD_PROVIDER,
                      PropertyType::STRING,
                      "The cloud provider of the filesystem storage service.",
                      kCloudProviderAWS,
                      ValidatePropertyType() + ValidatePropertyEnum<std::string>(kCloudProviderAWS,
                                                                                 kCloudProviderGCP,
                                                                                 kCloudProviderAliyun,
                                                                                 kCloudProviderAzure,
                                                                                 kCloudProviderTencent,
                                                                                 kCloudProviderHuawei)),
    REGISTER_PROPERTY(PROPERTY_FS_IAM_ENDPOINT,
                      PropertyType::STRING,
                      "The IAM endpoint for the filesystem storage service.",
                      "",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_LOG_LEVEL,
                      PropertyType::STRING,
                      "The log level for the filesystem storage service.",
                      "warn",
                      ValidatePropertyType() +
                          ValidatePropertyEnum<std::string>("fatal", "error", "warn", "info", "debug", "trace", "off")),
    REGISTER_PROPERTY(
        PROPERTY_FS_REGION, PropertyType::STRING, "The region for the filesystem storage service.", "", std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_USE_SSL,
                      PropertyType::BOOL,
                      "Whether to use SSL for the filesystem storage service.",
                      false,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_SSL_CA_CERT,
                      PropertyType::STRING,
                      "The CA certificate for the filesystem storage service.",
                      "",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_USE_IAM,
                      PropertyType::BOOL,
                      "Whether to use IAM for the filesystem storage service.",
                      false,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_USE_VIRTUAL_HOST,
                      PropertyType::BOOL,
                      "Whether to use virtual host style for the filesystem storage service.",
                      false,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_REQUEST_TIMEOUT_MS,
                      PropertyType::INT64,
                      "The request timeout in milliseconds for the filesystem storage service.",
                      int64_t(3000),
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_GCP_NATIVE_WITHOUT_AUTH,
                      PropertyType::BOOL,
                      "Whether to use GCP native without auth for the filesystem storage service.",
                      false,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_GCP_CREDENTIAL_JSON,
                      PropertyType::STRING,
                      "The GCP credential JSON for the filesystem storage service.",
                      "",
                      std::nullopt),
    REGISTER_PROPERTY(PROPERTY_FS_USE_CUSTOM_PART_UPLOAD,
                      PropertyType::BOOL,
                      "Whether to use custom part upload for the filesystem storage service.",
                      true,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_MAX_CONNECTIONS,
                      PropertyType::UINT32,
                      "The maximum number of connections for the filesystem storage service.",
                      uint32_t(100),
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_FS_MULTI_PART_UPLOAD_SIZE,
                      PropertyType::INT64,
                      "The multi-part upload size(Bytes) used in the writer.",
                      int64_t(DEFAULT_MULTIPART_UPLOAD_PART_SIZE),  // 10 MB
                      // According to https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
                      // the part numbers can be 5MB ~ 5GB
                      // R2 limit is 10MB ~ 5GB
                      ValidatePropertyType() + ValidatePropertyRange<int64_t>(MINIMAL_MULTIPART_UPLOAD_PART_SIZE,
                                                                              MAXIMAL_MULTIPART_UPLOAD_PART_SIZE)),
    // --- writer properties define ---
    REGISTER_PROPERTY(PROPERTY_WRITER_POLICY,
                      PropertyType::STRING,
                      "The column group policy for the writer.",
                      LOON_COLUMN_GROUP_POLICY_SINGLE,
                      ValidatePropertyType() + ValidatePropertyEnum<std::string>(LOON_COLUMN_GROUP_POLICY_SINGLE,
                                                                                 LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED,
                                                                                 LOON_COLUMN_GROUP_POLICY_SIZE_BASED)),
    REGISTER_PROPERTY(PROPERTY_WRITER_SCHEMA_BASE_PATTERNS,
                      PropertyType::VECTOR_STR,
                      "The column group patterns for the schema_based policy.",
                      std::vector<std::string>{},
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_SIZE_BASE_MACS,
                      PropertyType::INT64,
                      "The max size in bytes for each column group file when using the size_based policy.",
                      int64_t(0),  // 0 is invalid value
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_SIZE_BASE_MCIG,
                      PropertyType::INT64,
                      "The max number of columns in each column group file when using the size_based policy.",
                      int64_t(0),  // 0 is invalid value
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_BUFFER_SIZE,
                      PropertyType::INT32,
                      "The buffer size(Bytes) used in the writer.",
                      32 * 1024 * 1024,  // 32MB
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_FILE_ROLLING_SIZE,
                      PropertyType::UINT64,
                      "The max size(Bytes) for each data file rolling. Current size is uncompressed data size. Default "
                      "is 2GB. If the writer is configured with 2GB rolling size, and the writer has written 2GB data, "
                      "it will roll the file and start a new file. But the file size in disk(or object store) may be "
                      "smaller than 2GB, because of format writer may do the encoding or compression.",
                      uint64_t(2ULL * 1024 * 1024 * 1024),  // 2GB
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_COMPRESSION,
                      PropertyType::STRING,
                      "The compression type used in the writer.",
                      "zstd",
                      ValidatePropertyType() +
                          ValidatePropertyEnum<std::string>("uncompressed", "snappy", "gzip", "lz4", "zstd", "brotli")),
    REGISTER_PROPERTY(PROPERTY_WRITER_COMPRESSION_LEVEL,
                      PropertyType::INT32,
                      "The compression level used in the writer.",
                      5,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_ENABLE_DICTIONARY,
                      PropertyType::BOOL,
                      "Whether to enable dictionary encoding in the writer.",
                      true,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_ENC_ENABLE,
                      PropertyType::BOOL,
                      "Whether to enable encryption in the writer.",
                      false,
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_ENC_KEY,
                      PropertyType::STRING,
                      "The key used for encryption in the writer.",
                      "",
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_ENC_META,
                      PropertyType::STRING,
                      "The metadata generated by KMS for encryption in the writer.",
                      "",
                      ValidatePropertyType()),
    REGISTER_PROPERTY(PROPERTY_WRITER_ENC_ALGORITHM,
                      PropertyType::STRING,
                      "The encryption algorithm used in the writer.",
                      // parquet default encryption algorithm, vortex not support yet(The community already has PR)
                      ENCRYPTION_ALGORITHM_AES_GCM_V1,
                      ValidatePropertyType() + ValidatePropertyEnum<std::string>(ENCRYPTION_ALGORITHM_AES_GCM_V1,
                                                                                 ENCRYPTION_ALGORITHM_AES_GCM_CTR_V1)),
    REGISTER_PROPERTY(PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS,
                      PropertyType::BOOL,
                      "Whether to enable statistics collection in Vortex writer.",
                      false,
                      ValidatePropertyType()),

    // --- reader properties define ---
    REGISTER_PROPERTY(PROPERTY_READER_RECORD_BATCH_MAX_ROWS,
                      PropertyType::INT64,
                      "The maximum number of rows per record batch when reading.",
                      int64_t(8192),  // 8192 rows
                      ValidatePropertyType() + ValidatePropertyRange<int64_t>(1, INT64_MAX)),
    REGISTER_PROPERTY(
        PROPERTY_READER_RECORD_BATCH_MAX_SIZE,
        PropertyType::INT64,
        "The maximum size in bytes per record batch when reading.",
        int64_t(32LL * 1024 * 1024),                                                            // 32 MB
        ValidatePropertyType() + ValidatePropertyRange<int64_t>(1, 4LL * 1024 * 1024 * 1024)),  // max 4 GB

    REGISTER_PROPERTY(PROPERTY_READER_VORTEX_CHUNK_ROWS,
                      PropertyType::UINT64,
                      "The logical chunk rows for Vortex reader, notice that the actual chunk rows maybe smaller."
                      "Although vortex does not divide according to row groups, but it still have the split strategy to"
                      "split the row ranges(default strategy is base on the layout rows). So this property can help to"
                      "control the rows read per chunk. The actual chunk rows is min(logical_chunk_rows, layout_rows).",
                      uint64_t(8192),  // 8192 rows
                      ValidatePropertyType() + ValidatePropertyRange<uint64_t>(1, UINT64_MAX)),

    // --- transaction properties define ---
    REGISTER_PROPERTY(PROPERTY_TRANSACTION_COMMIT_NUM_RETRIES,
                      PropertyType::INT32,
                      "The number of retries for committing a transaction.",
                      1,                                                                // default 1 retry
                      ValidatePropertyType() + ValidatePropertyRange<int32_t>(0, 10)),  // [0, 10] retries
};

template <typename T>
arrow::Result<T> GetValue(const Properties& properties, const char* key) {
  auto it = properties.find(key);
  if (it == properties.end()) {
    auto pisit = property_infos.find(key);
    if (pisit == property_infos.end()) {
      return arrow::Status::Invalid("key not found(no predefined and no inserted): ", std::string(key));
    }

    if (!std::holds_alternative<T>(pisit->second.defval)) {
      return arrow::Status::Invalid("The key: ", std::string(key), " with invalid default type.");
    }

    return std::get<T>(pisit->second.defval);
  }

  if (!std::holds_alternative<T>(it->second)) {
    return arrow::Status::Invalid("The key: ", std::string(key), " with invalid type.");
  }

  return std::get<T>(it->second);
}

template arrow::Result<std::string> GetValue<std::string>(const Properties&, const char*);
template arrow::Result<int32_t> GetValue<int32_t>(const Properties&, const char*);
template arrow::Result<int64_t> GetValue<int64_t>(const Properties&, const char*);
template arrow::Result<uint32_t> GetValue<uint32_t>(const Properties&, const char*);
template arrow::Result<uint64_t> GetValue<uint64_t>(const Properties&, const char*);
template arrow::Result<bool> GetValue<bool>(const Properties&, const char*);
template arrow::Result<std::vector<std::string>> GetValue<std::vector<std::string>>(const Properties&, const char*);

template <typename T>
T GetValueNoError(const Properties& properties, const char* key) {
  auto it = properties.find(key);
  if (it == properties.end()) {
    auto pisit = property_infos.find(key);
    assert(pisit != property_infos.end() && "key not found(no predefined and no inserted)");
    return std::get<T>(pisit->second.defval);
  }

  assert(std::holds_alternative<T>(it->second) && "The key with invalid type.");
  return std::get<T>(it->second);
}

template std::string GetValueNoError<std::string>(const Properties&, const char*);
template int32_t GetValueNoError<int32_t>(const Properties&, const char*);
template int64_t GetValueNoError<int64_t>(const Properties&, const char*);
template uint32_t GetValueNoError<uint32_t>(const Properties&, const char*);
template uint64_t GetValueNoError<uint64_t>(const Properties&, const char*);
template bool GetValueNoError<bool>(const Properties&, const char*);
template std::vector<std::string> GetValueNoError<std::vector<std::string>>(const Properties&, const char*);

std::optional<std::string> SetValue(Properties& properties,
                                    const char* key,
                                    const char* value,
                                    bool allow_undefined_key) {
  if (property_infos.find(key) == property_infos.end()) {
    if (allow_undefined_key) {
      properties[key] = std::string(value);
      return std::nullopt;
    } else {
      {
        std::ostringstream oss;
        oss << "undefined key: '" << key << "'."
            << " should define the property first.";
        return oss.str();
      }
    }
  }
  const auto& property_info = property_infos.at(key);
  if (auto v = property_info.validator; v) {
    if (auto err = (*v)(property_info, value); err) {
      return err;
    }
  }
  properties[key] = GetPropertyValue(property_info, value);
  return std::nullopt;
}

std::optional<std::string> ConvertFFIProperties(Properties& result, const ::LoonProperties* properties) {
  if (properties && properties->properties && properties->count > 0) {
    for (size_t i = 0; i < properties->count; ++i) {
      const auto& prop = properties->properties[i];
      assert(prop.key && prop.value);
      if (auto rc = SetValue(result, prop.key, prop.value, true); rc != std::nullopt) {
        return rc;
      }
    }
  }

  return std::nullopt;
}

}  // namespace milvus_storage::api