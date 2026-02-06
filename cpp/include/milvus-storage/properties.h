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
#include <unordered_map>
#include <vector>
#include <functional>
#include <optional>
#include <variant>
#include <cstdint>
#include <arrow/result.h>

struct LoonProperties;

namespace milvus_storage::api {
struct PropertyInfo;
// Define property types and variants
enum class PropertyType { STRING, INT32, INT64, UINT32, UINT64, BOOL, VECTOR_STR };
// A variant that can hold any of the property types
using PropertyVariant =
    std::variant<std::string, int32_t, int64_t, uint32_t, uint64_t, bool, std::vector<std::string>, std::nullptr_t>;
// A map of property names to their variants
using Properties = std::unordered_map<std::string, PropertyVariant>;
// Validator function type
using ValidatorFunc = std::function<std::optional<std::string>(const PropertyInfo& property_info, const std::string&)>;

// A validator returns std::nullopt if validation passes,
// or an error message if validation fails.
class PropertiesValidator {
  public:
  PropertiesValidator();
  explicit PropertiesValidator(ValidatorFunc f);

  std::optional<std::string> operator()(const PropertyInfo& property_info, const std::string& v) const;

  // compose AND: a + b means run a then b, return first failure or success
  friend PropertiesValidator operator+(const PropertiesValidator& a, const PropertiesValidator& b);

  private:
  ValidatorFunc fn;
};

// Internal interface for properties
struct PropertyInfo {
  public:
  std::string name;
  PropertyType type;
  std::string desc;
  PropertyVariant defval;
  std::optional<PropertiesValidator> validator;
};

// --- Global property definitions ---
#define PROPERTY_FORMAT "format"

// --- Define FS property keys ---
// Standard filesystem properties (fs.*)
// These configure the default filesystem for the storage instance
#define PROPERTY_FS_PREFIX "fs."
#define PROPERTY_EXTFS_PREFIX "extfs."
#define PROPERTY_FS_ADDRESS "fs.address"
#define PROPERTY_FS_BUCKET_NAME "fs.bucket_name"
#define PROPERTY_FS_ACCESS_KEY_ID "fs.access_key_id"
#define PROPERTY_FS_ACCESS_KEY_VALUE "fs.access_key_value"
#define PROPERTY_FS_ROOT_PATH "fs.root_path"
#define PROPERTY_FS_STORAGE_TYPE "fs.storage_type"
#define PROPERTY_FS_CLOUD_PROVIDER "fs.cloud_provider"
#define PROPERTY_FS_IAM_ENDPOINT "fs.iam_endpoint"
#define PROPERTY_FS_LOG_LEVEL "fs.log_level"
#define PROPERTY_FS_REGION "fs.region"
#define PROPERTY_FS_USE_SSL "fs.use_ssl"
#define PROPERTY_FS_SSL_CA_CERT "fs.ssl_ca_cert"
#define PROPERTY_FS_USE_IAM "fs.use_iam"
#define PROPERTY_FS_USE_VIRTUAL_HOST "fs.use_virtual_host"
#define PROPERTY_FS_REQUEST_TIMEOUT_MS "fs.request_timeout_ms"
#define PROPERTY_FS_GCP_NATIVE_WITHOUT_AUTH "fs.gcp_native_without_auth"
#define PROPERTY_FS_GCP_CREDENTIAL_JSON "fs.gcp_credential_json"
#define PROPERTY_FS_USE_CUSTOM_PART_UPLOAD "fs.use_custom_part_upload"
#define PROPERTY_FS_MAX_CONNECTIONS "fs.max_connections"
#define PROPERTY_FS_MULTI_PART_UPLOAD_SIZE "fs.multi_part_upload_size"

// --- External Filesystem Properties ---
// External filesystems are configured with properties following the pattern:
//   extfs.<name>.<property>
//
// Where:
//   - <name> is a unique identifier for the external filesystem (e.g., "prod", "backup")
//   - <property> is any of the standard fs.* property names (without the "fs." prefix)
//
// Example configuration:
//   extfs.prod.address = "s3.us-west-2.amazonaws.com"
//   extfs.prod.bucket_name = "prod-data"
//   extfs.prod.access_key_id = "AKIAXXXXXXXX"
//   extfs.prod.access_key_value = "secret"
//   extfs.prod.storage_type = "remote"
//   extfs.prod.cloud_provider = kCloudProviderAWS
//
// Multiple external filesystems can be configured simultaneously:
//   extfs.backup.address = "s3.us-east-1.amazonaws.com"
//   extfs.backup.bucket_name = "backup-data"
//   ...
//
// When readers encounter absolute storage URIs (scheme://address/bucket/path or scheme://bucket/path),
// they will automatically match the address and bucket with registered external filesystems
// and use the appropriate credentials and configuration.

// --- Define Writer property keys ---
#define PROPERTY_WRITER_POLICY "writer.policy"
#define PROPERTY_WRITER_SCHEMA_BASE_PATTERNS "writer.split.schema_based.patterns"
#define PROPERTY_WRITER_SIZE_BASE_MACS "writer.split.size_based.max_avg_column_size"
#define PROPERTY_WRITER_SIZE_BASE_MCIG "writer.split.size_based.max_columns_in_group"
#define PROPERTY_WRITER_BUFFER_SIZE "writer.buffer_size"

#define PROPERTY_WRITER_FILE_ROLLING_SIZE "writer.file_rolling.size"

#define PROPERTY_WRITER_COMPRESSION "writer.compression"
#define PROPERTY_WRITER_COMPRESSION_LEVEL "writer.compression_level"
#define PROPERTY_WRITER_ENABLE_DICTIONARY "writer.enable_dictionary"
#define PROPERTY_WRITER_ENC_ENABLE "writer.enc.enable"
#define PROPERTY_WRITER_ENC_KEY "writer.enc.key"
#define PROPERTY_WRITER_ENC_META "writer.enc.meta"
#define PROPERTY_WRITER_ENC_ALGORITHM "writer.enc.algorithm"

#define PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS "writer.vortex.enable_statistics"
#define PROPERTY_WRITER_VORTEX_SEGMENT_ROW_SIZE "writer.vortex.segment_row_size"
#define PROPERTY_WRITER_VORTEX_VECTOR_SEGMENT_ROW_SIZE "writer.vortex.vector_segment_row_size"
#define PROPERTY_WRITER_VORTEX_VARLEN_SEGMENT_ROW_SIZE "writer.vortex.varlen_segment_row_size"

// --- Define Reader property keys ---
#define PROPERTY_READER_RECORD_BATCH_MAX_ROWS "reader.record_batch_max_rows"
#define PROPERTY_READER_RECORD_BATCH_MAX_SIZE "reader.record_batch_max_size"
#define PROPERTY_READER_LOGICAL_CHUNK_ROWS "reader.logical_chunk_rows"

// --- Define Transaction property keys ---
#define PROPERTY_TRANSACTION_COMMIT_NUM_RETRIES "transaction.commit.num-retries"

/**
 * Get the value of a property by key, returning an error if the key does
 * not exist.
 *
 * 1. If the key exist in the `Properties`, will direct return the value.
 * 2. If the key does not exist in the `Properties`, but is a predefined
 *    property, will return the default value.
 * 3. If the key does not exist both in the `Properties` and predefined
 *    properties, will return an error.
 */
template <typename T>
arrow::Result<T> GetValue(const Properties& properties, const char* key);

/**
 * Get the value of a property by key. Will raise assert if the key does
 * not exist.
 *
 * `GetValueNoError` can only be called inside storage.If we do have not
 * predefined the property, but still get the value of the key, it's a
 * logical error.
 */
template <typename T>
T GetValueNoError(const Properties& properties, const char* key);
std::optional<std::string> SetValue(Properties& properties,
                                    const char* key,
                                    const char* value,
                                    bool allow_undefined_key = true);
std::optional<std::string> ConvertFFIProperties(Properties& result, const ::LoonProperties* properties);

}  // namespace milvus_storage::api