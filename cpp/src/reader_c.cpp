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

#include "milvus-storage/reader_c.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/manifest_json.h"
#include "milvus-storage/result_c.h"
#include "milvus-storage/result_internal.h"

#include <arrow/c/helpers.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <charconv>

using namespace milvus_storage::api;
using namespace milvus_storage;
// Helper function to convert C string array to std::vector
static inline std::shared_ptr<std::vector<std::string>> convert_string_array(const char* const* strings, size_t count) {
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

// Helper function to convert C ReadProperties to C++ ReadProperties
static inline std::unordered_map<std::string, std::string> convert_read_properties(const ::ReadProperties* properties) {
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

class PropertiesMapper final {
  public:
  template <typename T, typename MemberType>
  void registerField(const std::string& field, T* obj, MemberType T::*member) {
    mappings[field] = [&](const std::string& value) -> bool {
      auto [ok, val] = convertFunc<MemberType>(value);
      if (ok) {
        obj->*member = val;
      }

      return ok;
    };
  }

  std::pair<bool, std::string> map(const std::unordered_map<std::string, std::string>& data) {
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

  private:
  template <typename T>
  std::pair<bool, T> convertFunc(const std::string& str);

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

  private:
  std::unordered_map<std::string, std::function<bool(const std::string&)>> mappings;
};

static inline std::pair<bool, std::string> create_file_system_config(
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

// ==================== ReadProperties C Implementation ====================

FFIResult read_properties_create(const char* const* keys,
                                 const char* const* values,
                                 size_t count,
                                 ::ReadProperties* properties) {
  if (!properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "properties should not be empty");
  }

  properties->properties = nullptr;
  properties->count = 0;

  if (count == 0 || !keys || !values || !*keys || !*values) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid keys/values");
  }

  properties->properties = static_cast<ReadProperty*>(malloc(sizeof(ReadProperty) * count));
  if (!properties->properties) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to malloc [size=", sizeof(ReadProperty) * count, "]");
  }

  for (size_t i = 0; i < count; ++i) {
    properties->properties[i].key = nullptr;
    properties->properties[i].value = nullptr;

    if (keys[i]) {
      size_t key_len = strlen(keys[i]) + 1;
      properties->properties[i].key = static_cast<char*>(malloc(key_len));
      if (properties->properties[i].key) {
        strcpy(properties->properties[i].key, keys[i]);
      }
    }

    if (values[i]) {
      size_t value_len = strlen(values[i]) + 1;
      properties->properties[i].value = static_cast<char*>(malloc(value_len));
      if (properties->properties[i].value) {
        strcpy(properties->properties[i].value, values[i]);
      }
    }
  }

  properties->count = count;
  RETURN_SUCCESS();
}

const char* read_properties_get(const ::ReadProperties* properties, const char* key) {
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

void read_properties_free(::ReadProperties* properties) {
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

// ==================== ChunkReader C Implementation ====================

FFIResult get_chunk_indices(ChunkReaderHandle reader,
                            const int64_t* row_indices,
                            size_t num_indices,
                            int64_t** chunk_indices,
                            size_t* num_chunk_indices) {
  if (!reader || !row_indices || num_indices == 0 || !chunk_indices || !num_chunk_indices) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  std::vector<int64_t> input_indices(row_indices, row_indices + num_indices);

  auto result = cpp_reader->get_chunk_indices(input_indices);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }

  const auto& output_indices = result.ValueOrDie();
  if (output_indices.empty()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, "Current indices(out) is empty");
  }

  *chunk_indices = static_cast<int64_t*>(malloc(sizeof(int64_t) * output_indices.size()));
  if (*chunk_indices) {
    std::copy(output_indices.begin(), output_indices.end(), *chunk_indices);
    *num_chunk_indices = output_indices.size();
  } else {
    *num_chunk_indices = 0;
  }

  RETURN_SUCCESS();
}

FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, ArrowArray* out_array) {
  if (!reader || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  auto result = cpp_reader->get_chunk(chunk_index);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }
  auto record_batch = result.ValueOrDie();
  arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_array);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
  }

  RETURN_SUCCESS();
}

FFIResult get_chunks(ChunkReaderHandle reader,
                     const int64_t* chunk_indices,
                     size_t num_indices,
                     int64_t parallelism,
                     ArrowArray** arrays,
                     size_t* num_arrays) {
  if (!reader || !chunk_indices || num_indices == 0 || !arrays || !num_arrays) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  std::vector<int64_t> indices(chunk_indices, chunk_indices + num_indices);

  auto result = cpp_reader->get_chunks(indices, parallelism);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }

  const auto& record_batches = result.ValueOrDie();
  if (record_batches.empty()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, "Empty record batch");
  }

  // Convert RecordBatches to Arrow C ABI arrays
  *arrays = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray) * record_batches.size()));
  if (*arrays) {
    *num_arrays = record_batches.size();
    // TODO: Implement RecordBatch to ArrowArray conversion
    // This requires converting Arrow C++ RecordBatch to Arrow C ABI format
    // For now, just initialize the arrays to avoid crashes
    for (size_t i = 0; i < record_batches.size(); ++i) {
      memset(&(*arrays)[i], 0, sizeof(ArrowArray));
    }
  } else {
    *num_arrays = 0;
  }

  RETURN_SUCCESS();
}

void chunk_reader_destroy(ChunkReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<ChunkReader*>(reader);
  }
}

// ==================== Reader C Implementation ====================

FFIResult reader_new(char* manifest,
                     ArrowSchema* schema,
                     const char* const* needed_columns,
                     size_t num_columns,
                     const ::ReadProperties* properties,
                     ReaderHandle* out_handle) {
  if (!manifest || !schema || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  std::unordered_map<std::string, std::string> properties_map;
  properties_map = convert_read_properties(properties);

  ArrowFileSystemConfig fs_config;
  auto [ok, failed_key] = create_file_system_config(properties_map, fs_config);
  if (!ok) {
    assert(properties_map.count(failed_key) != 0);
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", failed_key.c_str(), ", ",
                 properties_map[failed_key].c_str(), "]");
  }

  auto fs_result = CreateArrowFileSystem(fs_config);
  if (!fs_result.ok()) {
    // TODO: missing the error message in fs_result
    RETURN_ERROR(LOON_ARROW_ERROR, "Failed to create arrow file system");
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(fs_result.value());
  auto result = arrow::ImportSchema(schema);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }

  auto cpp_schema = result.ValueOrDie();
  auto cpp_properties = std::move(properties_map);
  auto cpp_needed_columns = convert_string_array(needed_columns, num_columns);
  // Parse the manifest, the manifest is a JSON string
  std::istringstream manifest_stream(manifest);
  milvus_storage::JsonManifestSerDe serde;
  auto cpp_manifest = serde.Deserialize(manifest_stream);
  auto cpp_reader = Reader::create(cpp_fs, cpp_manifest, cpp_schema, cpp_needed_columns, cpp_properties);
  auto raw_cpp_reader = reinterpret_cast<ReaderHandle>(cpp_reader.release());
  assert(raw_cpp_reader);
  *out_handle = raw_cpp_reader;

  RETURN_SUCCESS();
}

FFIResult get_record_batch_reader(ReaderHandle reader,
                                  const char* predicate,
                                  int64_t batch_size,
                                  int64_t buffer_size,
                                  ArrowArrayStream* out_array_stream) {
  if (!reader || !out_array_stream)
    RETURN_ERROR(LOON_INVALID_ARGS);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str, batch_size, buffer_size);
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
    }

    auto array_stream = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatchReader(array_stream, out_array_stream);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
    }

  } catch (...) {  // TODO: make sure which exception will be throw
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle) {
  if (!reader || !out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto result = cpp_reader->get_chunk_reader(column_group_id);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<ChunkReaderHandle>(chunk_reader);
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

FFIResult take(
    ReaderHandle reader, const int64_t* row_indices, size_t num_indices, int64_t parallelism, ArrowArray* out_arrays) {
  if (!reader || !row_indices || num_indices == 0 || !out_arrays)
    RETURN_ERROR(LOON_INVALID_ARGS);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::vector<int64_t> indices(row_indices, row_indices + num_indices);

    auto result = cpp_reader->take(indices, parallelism);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());

    // export the arrow::RecordBatch to Arrow C ABI Array
    auto record_batch = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_arrays);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
    }

  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

void reader_destroy(ReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<Reader*>(reader);
  }
}