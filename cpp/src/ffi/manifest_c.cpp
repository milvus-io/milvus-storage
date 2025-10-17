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
#include <unordered_set>
#include <vector>
#include <memory>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/manifest_json.h"

using namespace milvus_storage;
using namespace milvus_storage::api;
// ==================== Manifest C Implementation ====================

arrow::Status direct_write(std::shared_ptr<arrow::fs::FileSystem> fs,
                           const std::string& path,
                           std::shared_ptr<arrow::Buffer> buffer) {
  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(buffer));
  ARROW_RETURN_NOT_OK(output_stream->Close());

  return arrow::Status::OK();
}

arrow::Status direct_read(std::shared_ptr<arrow::fs::FileSystem> fs, const std::string& path, char** out_buffer) {
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());

  char* buffer = (char*)malloc(file_size);
  if (buffer == nullptr) {
    return arrow::Status::OutOfMemory("Failed to allocate memory for read, size =", file_size);
  }
  ARROW_ASSIGN_OR_RAISE(auto out_size, input_file->Read(file_size, (void*)buffer));
  if (out_size != file_size) {
    free(buffer);
    return arrow::Status::IOError("Failed to read the complete file, expected size =", file_size,
                                  ", actual size =", out_size);
  }

  auto statis = input_file->Close();
  if (!statis.ok()) {
    free(buffer);
    return statis;
  }

  *out_buffer = buffer;

  return arrow::Status::OK();
}

FFIResult manifest_write(const char* manifest_path, const char* manifest_json, const ::Properties* properties) {
  if (!manifest_path || !manifest_json || !properties)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: manifest_path, manifest_json and properties must not be null");
#ifndef NDEBUG
  // check the manifest JSON can be deserialized
  assert(JsonManifestSerDe().Deserialize(std::string_view(manifest_json)));
#endif

  milvus_storage::api::Properties properties_map;
  auto opt = FromFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  ArrowFileSystemConfig fs_config;
  auto status = ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  auto result = CreateArrowFileSystem(fs_config);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(result.ValueOrDie());
  status = direct_write(cpp_fs, std::string(manifest_path), arrow::Buffer::Wrap(manifest_json, strlen(manifest_json)));
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  RETURN_SUCCESS();
}

FFIResult manifest_read(const char* manifest_path, char** out_manifest, const ::Properties* properties) {
  if (!manifest_path || !properties || !out_manifest)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: manifest_path, properties and out_manifest must not be null");

  milvus_storage::api::Properties properties_map;
  auto opt = FromFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  ArrowFileSystemConfig fs_config;
  auto status = ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  auto result = CreateArrowFileSystem(fs_config);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(result.ValueOrDie());
  // direct read from file
  status = direct_read(cpp_fs, std::string(manifest_path), out_manifest);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

#ifndef NDEBUG
  // check the manifest JSON can be deserialized
  assert(JsonManifestSerDe().Deserialize(std::string_view(*out_manifest)));
#endif

  RETURN_SUCCESS();
}

static arrow::Status manifest_add_columns(const std::shared_ptr<Manifest>& manifest1,
                                          const std::shared_ptr<Manifest>& manifest2) {
  auto column_groups1 = manifest1->get_column_groups();
  auto column_groups2 = manifest2->get_column_groups();
  auto unque_colnames1 = manifest1->get_all_column_names();

  for (const auto& cg2 : column_groups2) {
    // check if the column already exists
    for (const auto& col : cg2->columns) {
      if (unque_colnames1.count(col) != 0) {
        return arrow::Status::Invalid("Column ", col, " already exists in the last manifest");
      }
    }

    // add the new column group
    ARROW_RETURN_NOT_OK(manifest1->add_column_group(cg2));
  }

  return arrow::Status::OK();
}

FFIResult manifest_combine(const char* manifest1, const char* manifest2, char** out_manifest) {
  if (!manifest1 || !manifest2 || !out_manifest)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: manifest1, manifest2 and out_manifest must not be null");

  // parse two manifest
  auto cpp_manifest1 = JsonManifestSerDe().Deserialize(std::string_view(manifest1));
  if (!cpp_manifest1) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize manifest JSON: ", std::string(manifest1));
  }

  auto cpp_manifest2 = JsonManifestSerDe().Deserialize(std::string_view(manifest2));
  if (!cpp_manifest2) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize manifest JSON: ", std::string(manifest2));
  }

  auto status = Manifest::manifest_combine_paths(cpp_manifest1, cpp_manifest2);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  auto [ok, manifest_raw] = JsonManifestSerDe().Serialize(cpp_manifest1);
  if (!ok) {
    RETURN_ERROR(LOON_ARROW_ERROR, "Failed to serialize manifest to JSON");
  }
  *out_manifest = strdup(manifest_raw.c_str());
  RETURN_SUCCESS();
}

FFIResult manifest_add_columns(const char* manifest1, const char* manifest2, char** out_manifest) {
  if (!manifest1 || !manifest2 || !out_manifest)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: manifest1, manifest2 and out_manifest must not be null");

  // parse two manifest
  auto cpp_manifest1 = JsonManifestSerDe().Deserialize(std::string_view(manifest1));
  if (!cpp_manifest1) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize manifest JSON: ", std::string(manifest1));
  }

  auto cpp_manifest2 = JsonManifestSerDe().Deserialize(std::string_view(manifest2));
  if (!cpp_manifest2) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize manifest JSON: ", std::string(manifest2));
  }

  // add columns from manifest2 to manifest1
  auto status = manifest_add_columns(cpp_manifest1, cpp_manifest2);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  auto [ok, manifest_raw] = JsonManifestSerDe().Serialize(cpp_manifest1);
  if (!ok) {
    RETURN_ERROR(LOON_ARROW_ERROR, "Failed to serialize manifest to JSON");
  }
  *out_manifest = strdup(manifest_raw.c_str());

  RETURN_SUCCESS();
}

void manifest_destory(char* manifest) {
  if (manifest)
    free(manifest);
}
