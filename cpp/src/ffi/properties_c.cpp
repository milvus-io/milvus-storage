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

#include "milvus-storage/ffi_c.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iostream>

#include "milvus-storage/properties.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"

using namespace milvus_storage;
using namespace milvus_storage::api;

// --- Global property definitions ---
const char* loon_properties_format = PROPERTY_FORMAT;

// --- Define FS property keys ---
const char* loon_properties_fs_address = PROPERTY_FS_ADDRESS;
const char* loon_properties_fs_bucket_name = PROPERTY_FS_BUCKET_NAME;
const char* loon_properties_fs_access_key_id = PROPERTY_FS_ACCESS_KEY_ID;
const char* loon_properties_fs_access_key_value = PROPERTY_FS_ACCESS_KEY_VALUE;
const char* loon_properties_fs_root_path = PROPERTY_FS_ROOT_PATH;
const char* loon_properties_fs_storage_type = PROPERTY_FS_STORAGE_TYPE;
const char* loon_properties_fs_cloud_provider = PROPERTY_FS_CLOUD_PROVIDER;
const char* loon_properties_fs_iam_endpoint = PROPERTY_FS_IAM_ENDPOINT;
const char* loon_properties_fs_log_level = PROPERTY_FS_LOG_LEVEL;
const char* loon_properties_fs_region = PROPERTY_FS_REGION;
const char* loon_properties_fs_use_ssl = PROPERTY_FS_USE_SSL;
const char* loon_properties_fs_ssl_ca_cert = PROPERTY_FS_SSL_CA_CERT;
const char* loon_properties_fs_use_iam = PROPERTY_FS_USE_IAM;
const char* loon_properties_fs_use_virtual_host = PROPERTY_FS_USE_VIRTUAL_HOST;
const char* loon_properties_fs_request_timeout_ms = PROPERTY_FS_REQUEST_TIMEOUT_MS;
const char* loon_properties_fs_gcp_native_without_auth = PROPERTY_FS_GCP_NATIVE_WITHOUT_AUTH;
const char* loon_properties_fs_gcp_credential_json = PROPERTY_FS_GCP_CREDENTIAL_JSON;
const char* loon_properties_fs_use_custom_part_upload = PROPERTY_FS_USE_CUSTOM_PART_UPLOAD;
const char* loon_properties_fs_max_connections = PROPERTY_FS_MAX_CONNECTIONS;
const char* loon_properties_fs_multi_part_upload_size = PROPERTY_FS_MULTI_PART_UPLOAD_SIZE;

// --- Define Writer property keys ---
const char* loon_properties_writer_policy = PROPERTY_WRITER_POLICY;
const char* loon_properties_writer_schema_base_patterns = PROPERTY_WRITER_SCHEMA_BASE_PATTERNS;
const char* loon_properties_writer_size_base_macs = PROPERTY_WRITER_SIZE_BASE_MACS;
const char* loon_properties_writer_size_base_mcig = PROPERTY_WRITER_SIZE_BASE_MCIG;
const char* loon_properties_writer_buffer_size = PROPERTY_WRITER_BUFFER_SIZE;
const char* loon_properties_writer_file_rolling_size = PROPERTY_WRITER_FILE_ROLLING_SIZE;
const char* loon_properties_writer_compression = PROPERTY_WRITER_COMPRESSION;
const char* loon_properties_writer_compression_level = PROPERTY_WRITER_COMPRESSION_LEVEL;
const char* loon_properties_writer_enable_dictionary = PROPERTY_WRITER_ENABLE_DICTIONARY;
const char* loon_properties_writer_enc_enable = PROPERTY_WRITER_ENC_ENABLE;
const char* loon_properties_writer_enc_key = PROPERTY_WRITER_ENC_KEY;
const char* loon_properties_writer_enc_meta = PROPERTY_WRITER_ENC_META;
const char* loon_properties_writer_enc_algorithm = PROPERTY_WRITER_ENC_ALGORITHM;
const char* loon_properties_writer_vortex_enable_statistics = PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS;
const char* loon_properties_writer_vortex_segment_row_size = PROPERTY_WRITER_VORTEX_SEGMENT_ROW_SIZE;
const char* loon_properties_writer_vortex_vector_segment_row_size = PROPERTY_WRITER_VORTEX_VECTOR_SEGMENT_ROW_SIZE;
const char* loon_properties_writer_vortex_varlen_segment_row_size = PROPERTY_WRITER_VORTEX_VARLEN_SEGMENT_ROW_SIZE;

// --- Define Reader property keys ---
const char* loon_properties_reader_record_batch_max_rows = PROPERTY_READER_RECORD_BATCH_MAX_ROWS;
const char* loon_properties_reader_record_batch_max_size = PROPERTY_READER_RECORD_BATCH_MAX_SIZE;
const char* loon_properties_reader_logical_chunk_rows = PROPERTY_READER_LOGICAL_CHUNK_ROWS;

// --- Define Transaction property keys ---
const char* loon_properties_transaction_commit_num_retries = PROPERTY_TRANSACTION_COMMIT_NUM_RETRIES;

// ==================== Properties C Implementation ====================

LoonFFIResult loon_properties_create(const char* const* keys,
                                     const char* const* values,
                                     size_t count,
                                     ::LoonProperties* properties) {
  // used to make sure no duplicate keys
  std::unordered_set<std::string_view> key_set;
  if (!properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "properties should not be empty");
  }

  properties->properties = nullptr;
  properties->count = 0;

  try {
    if (count == 0 || !keys || !values) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid keys/values");
    }

    properties->properties = static_cast<LoonProperty*>(malloc(sizeof(LoonProperty) * count));
    if (!properties->properties) {
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to malloc [size=", sizeof(LoonProperty) * count, "]");
    }
    properties->count = count;

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
        loon_properties_free(properties);
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
        loon_properties_free(properties);
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "The value index: ", i, " is invalid, key: ", keys[i]);
      }
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

const char* loon_properties_get(const ::LoonProperties* properties, const char* key) {
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

void loon_properties_free(::LoonProperties* properties) {
  if (!properties) {
    return;
  }

  if (properties->properties) {
    for (size_t i = 0; i < properties->count; ++i) {
      free(properties->properties[i].key);
      free(properties->properties[i].value);
    }
    free(properties->properties);
    properties->properties = nullptr;
  }
  properties->count = 0;
}