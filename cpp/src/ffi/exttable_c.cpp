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
#include "milvus-storage/ffi_exttable_c.h"

#include <cstring>
#include <memory>
#include <optional>

#include "milvus-storage/common/layout.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/transaction/transaction.h"

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

LoonFFIResult loon_exttable_explore(const char** columns,
                                    size_t col_lens,
                                    const char* format,
                                    const char* base_path,
                                    const char* explore_dir,
                                    const ::LoonProperties* properties,
                                    uint64_t* out_num_of_files,
                                    char** out_column_groups_file_path) {
  if (!columns || !format || !base_path || !explore_dir || !properties || !out_num_of_files ||
      !out_column_groups_file_path) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments, columns, format, base_path, explore_dir, properties, out_num_of_files, "
                 "out_column_groups_file_path must not be null");
  }

  if (col_lens == 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments, col_lens should GT 0");
  }

  try {
    milvus_storage::api::Properties properties_map;
    std::string format_str(format);

    auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    auto fmt_res = milvus_storage::Format::get(format_str);
    if (!fmt_res.ok()) {
      RETURN_ERROR(LOON_INVALID_ARGS, fmt_res.status().ToString());
    }

    auto files_result = fmt_res.ValueOrDie()->explore(explore_dir, properties_map);
    if (!files_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, files_result.status().ToString());
    }
    auto files = files_result.ValueOrDie();

    std::vector<std::string> columns_cpp;
    for (size_t i = 0; i < col_lens; i++) {
      columns_cpp.emplace_back(columns[i]);
    }

    // construct the column groups
    ColumnGroups cgs;
    cgs.push_back(
        std::make_shared<ColumnGroup>(ColumnGroup{.columns = columns_cpp, .format = format_str, .files = files}));

    // commit the column groups
    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(properties_map);
    if (!fs_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_result.status().ToString());
    }
    auto transaction_result = Transaction::Open(fs_result.ValueOrDie(), base_path);
    if (!transaction_result.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, transaction_result.status().ToString());
    }
    auto transaction = std::move(transaction_result.ValueOrDie());

    // Append column groups directly
    transaction->AppendFiles(cgs);

    auto commit_result = transaction->Commit();
    if (!commit_result.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, commit_result.status().ToString());
    }

    auto committed_version = commit_result.ValueOrDie();

    *out_num_of_files = files.size();
    *out_column_groups_file_path = strdup(milvus_storage::get_manifest_filepath(base_path, committed_version).c_str());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_exttable_get_file_info(const char* format,
                                          const char* file_path,
                                          const ::LoonProperties* properties,
                                          uint64_t* out_num_of_rows) {
  if (!format || !file_path || !properties || !out_num_of_rows) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: format, file_path, properties, and out_num_of_rows must not be null");
  }

  try {
    milvus_storage::api::Properties properties_map;
    std::string format_str(format);

    auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    auto fmt_res = milvus_storage::Format::get(format_str);
    if (!fmt_res.ok()) {
      RETURN_ERROR(LOON_INVALID_ARGS, fmt_res.status().ToString());
    }

    // Create a ColumnGroupFile to pass to the reader factory
    ColumnGroupFile cg_file{std::string(file_path), 0, 0, {}};
    auto reader_res = fmt_res.ValueOrDie()->create_reader(nullptr, cg_file, properties_map, {}, nullptr);
    if (!reader_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, reader_res.status().ToString());
    }

    auto rg_infos_res = reader_res.ValueOrDie()->get_row_group_infos();
    if (!rg_infos_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, rg_infos_res.status().ToString());
    }
    auto& rg_infos = rg_infos_res.ValueOrDie();
    *out_num_of_rows = rg_infos.empty() ? 0 : rg_infos.back().end_offset;

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

static arrow::Result<std::shared_ptr<milvus_storage::api::Manifest>> read_manifest(const char* path,
                                                                                   const ::LoonProperties* properties) {
  milvus_storage::api::Properties properties_map;

  auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    return arrow::Status::Invalid("Failed to parse properties [", opt->c_str(), "]");
  }

  ARROW_ASSIGN_OR_RAISE(auto fs, milvus_storage::FilesystemCache::getInstance().get(properties_map, path));
  return milvus_storage::api::Manifest::ReadFrom(fs, path);
}

LoonFFIResult loon_exttable_read_manifest(const char* manifest_file_path,
                                          const ::LoonProperties* properties,
                                          LoonManifest** out_manifest) {
  if (!manifest_file_path || !properties || !out_manifest) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: manifest_file_path, properties, and out_manifest must not be null");
  }

  try {
    auto manifest_res = read_manifest(manifest_file_path, properties);
    if (!manifest_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, manifest_res.status().ToString());
    }
    auto manifest = manifest_res.ValueOrDie();

    // Export full manifest including column groups, delta logs, and stats
    auto st = milvus_storage::manifest_export(manifest, out_manifest);
    if (!st.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}
