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
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/manifest_json.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

using namespace milvus_storage::api;
using namespace milvus_storage;

extern arrow::Status properties_overwrite(::Properties* properties, const char* key, const char* value);

FFIResult writer_new(const char* base_path,
                     ArrowSchema* schema_raw,
                     const ::Properties* properties,
                     WriterHandle* out_handle) {
  if (!base_path || !schema_raw || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: base_path, schema_raw, properties, and out_handle must not be null");
  }

  milvus_storage::api::Properties properties_map;
  auto opt = FromFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  ArrowFileSystemConfig fs_config;
  auto fs_status = ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
  if (!fs_status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, fs_status.ToString());
  }

  auto fs_result = CreateArrowFileSystem(fs_config);
  if (!fs_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, fs_result.status().ToString());
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(fs_result.ValueOrDie());
  auto schema_result = arrow::ImportSchema(schema_raw);
  if (!schema_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, schema_result.status().ToString());
  }
  auto schema = schema_result.ValueOrDie();
  std::unique_ptr<ColumnGroupPolicy> policy;

  auto policy_status = ColumnGroupPolicy::create_column_group_policy(properties_map, schema).Value(&policy);
  if (!policy_status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, policy_status.ToString());
  }

  auto cpp_writer = Writer::create(std::move(cpp_fs), std::move(std::string(base_path)), schema, std::move(policy),
                                   std::move(properties_map));

  auto raw_cpp_writer = reinterpret_cast<WriterHandle>(cpp_writer.release());
  assert(raw_cpp_writer);
  *out_handle = raw_cpp_writer;

  RETURN_SUCCESS();
}

FFIResult writer_write(WriterHandle handle, struct ArrowArray* array) {
  if (!handle || !array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and array must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);

    auto rb_result = arrow::ImportRecordBatch(array, cpp_writer->schema());
    if (!rb_result.ok()) {
      array->release(array);
      RETURN_ERROR(LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto record_batch = rb_result.ValueOrDie();

    auto status = cpp_writer->write(record_batch);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult writer_flush(WriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto status = cpp_writer->flush();
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult writer_close(WriterHandle handle, char** out_manifest) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto result = cpp_writer->close();
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }
    auto manifest = result.ValueOrDie();
    auto [ok, manifest_raw] = JsonManifestSerDe().Serialize(manifest);
    if (!ok) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to serialize manifest to JSON");
    }
    *out_manifest = strdup(manifest_raw.c_str());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void writer_destroy(WriterHandle handle) {
  if (handle) {
    delete reinterpret_cast<Writer*>(handle);
  }
}

std::string get_schemabase_policy_patterns(const std::shared_ptr<Manifest>& manifest) {
  std::vector<std::vector<std::string>> patterns;
  std::string result;
  assert(manifest);

  auto column_groups = manifest->get_column_groups();
  patterns.reserve(column_groups.size());

  for (const auto& column_group : column_groups) {
    patterns.emplace_back(column_group->columns);
  }

  /*
   * Convert patterns to a single string with the following format:
   *  column_1|column_2,column_3|column_4|column_5,column_6
   * use `,` to separate different column groups
   * use `|` to separate different columns in the same column group
   */
  for (const auto& pattern : patterns) {
    if (!result.empty()) {
      result += ",";
    }
    for (size_t i = 0; i < pattern.size(); i++) {
      if (i > 0) {
        result += "|";
      }
      result += pattern[i];
    }
  }

  return result;
}
