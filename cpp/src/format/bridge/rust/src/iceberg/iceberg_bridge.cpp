// Copyright 2024 Zilliz
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

#include "iceberg/iceberg_bridge.h"
#include "runtime/bridge_util.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

#include "rust/cxx.h"
#include "rust-bridge/lib.h"
#include "iceberg/source_uri_bridge.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/ffi_internal/result.h"

namespace milvus_storage::iceberg::ffi {
SourceUriResult resolve_source_uri(const FileSystemWrapper& filesystem, rust::Str uri) noexcept {
  SourceUriResult result{};
  try {
    auto source = std::dynamic_pointer_cast<SourceFileSystemProxy>(filesystem.get());
    if (!source)
      return result;
    result.is_source = true;
    auto path = source->PathFromUri(std::string(uri.data(), uri.size()));
    if (!path.ok()) {
      result.error_code = FFIErrorCodeFromExtendStatus(path.status());
      if (!result.error_code)
        result.error_code = LOON_LOGICAL_ERROR;
      return result;
    }
    result.path = rust::String(*path);
  } catch (const std::bad_alloc&) {
    result.error_code = LOON_MEMORY_ERROR;
  } catch (...) {
    result.error_code = LOON_GOT_EXCEPTION;
  }
  return result;
}
}  // namespace milvus_storage::iceberg::ffi

namespace milvus_storage::iceberg {

using milvus_storage::ConvertStorageOptions;

arrow::Result<std::vector<IcebergFileInfo>> PlanFiles(
    const std::string& metadata_location,
    int64_t snapshot_id,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const std::unordered_map<std::string, std::string>& read_options) {
  if (!filesystem) {
    return arrow::Status::Invalid("PlanFiles requires a non-null filesystem");
  }
  return CatchRustResult<std::vector<IcebergFileInfo>>("Failed to plan Iceberg files", [&]() {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(read_options, keys, values);

    auto lease = std::make_shared<FileSystemWrapper>(filesystem);
    auto rust_results =
        ffi::iceberg_plan_files(std::move(lease), rust::Str(metadata_location.data(), metadata_location.length()),
                                snapshot_id, std::move(keys), std::move(values));

    std::vector<IcebergFileInfo> result;
    result.reserve(rust_results.size());
    for (const auto& item : rust_results) {
      result.push_back(IcebergFileInfo{
          std::string(item.data_file_path.data(), item.data_file_path.size()),
          item.record_count,
          item.num_deleted_rows,
          std::vector<uint8_t>(item.delete_metadata_json.begin(), item.delete_metadata_json.end()),
      });
    }
    return result;
  });
}

arrow::Result<IcebergTestTableInfo> CreateTestTable(const std::string& table_dir,
                                                    uint64_t num_rows,
                                                    bool with_positional_deletes,
                                                    const std::vector<int64_t>& deleted_positions,
                                                    const std::unordered_map<std::string, std::string>& storage_options,
                                                    const std::string& record_scheme_override) {
  return CatchRustResult<IcebergTestTableInfo>("Failed to create Iceberg test table", [&]() {
    rust::Vec<int64_t> rust_positions;
    for (auto pos : deleted_positions) {
      rust_positions.push_back(pos);
    }

    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);

    auto result = ffi::iceberg_create_test_table(
        rust::Str(table_dir.data(), table_dir.length()), num_rows, with_positional_deletes, std::move(rust_positions),
        std::move(keys), std::move(values), rust::Str(record_scheme_override.data(), record_scheme_override.length()));

    return IcebergTestTableInfo{
        std::string(result.metadata_location.data(), result.metadata_location.size()),
        result.snapshot_id,
        std::string(result.data_file_uri.data(), result.data_file_uri.size()),
    };
  });
}

}  // namespace milvus_storage::iceberg
