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
    RETURN_USER_SOURCE_ERROR_IF(fmt_res.status(), LOON_SOURCE_INVALID, fmt_res.status().ToString());

    // explore_dir and the extfs.* credentials come straight from the user's
    // external-source definition, so a missing bucket or a rejected key here
    // is a user error, not a system failure. See
    // UserSourceErrorCodeFromStatus().
    // Same provenance question as get_file_info: only an absolute explore_dir
    // is a location the caller chose.
    const bool location_is_user_supplied = milvus_storage::HasUriScheme(explore_dir);
    auto files_result = fmt_res.ValueOrDie()->explore(explore_dir, properties_map);
    RETURN_USER_SOURCE_ERROR_IF_AT(files_result.status(), LOON_ARROW_ERROR, location_is_user_supplied,
                                   files_result.status().ToString());
    auto files = files_result.ValueOrDie();

    std::vector<std::string> columns_cpp;
    for (size_t i = 0; i < col_lens; i++) {
      columns_cpp.emplace_back(columns[i]);
    }

    // construct the column groups
    ColumnGroups cgs;
    cgs.push_back(
        std::make_shared<ColumnGroup>(ColumnGroup{.columns = columns_cpp, .format = format_str, .files = files}));

    // commit the column groups. This filesystem is the storage the manifest
    // gets committed to (base_path), not the user's explore location -- keep
    // the producer's classification rather than re-tagging a commit-side
    // bucket/credential/missing failure as the user's error.
    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(properties_map);
    RETURN_ARROW_ERROR_IF(fs_result.status(), LOON_ARROW_ERROR, fs_result.status().ToString());
    auto transaction_result = Transaction::Open(fs_result.ValueOrDie(), base_path);
    RETURN_ARROW_ERROR_IF(transaction_result.status(), LOON_LOGICAL_ERROR, transaction_result.status().ToString());
    auto transaction = std::move(transaction_result.ValueOrDie());

    // Append column groups directly
    transaction->AppendFiles(cgs);

    auto commit_result = transaction->Commit();
    RETURN_ARROW_ERROR_IF(commit_result.status(), LOON_LOGICAL_ERROR, commit_result.status().ToString());

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
    RETURN_USER_SOURCE_ERROR_IF(fmt_res.status(), LOON_SOURCE_INVALID, fmt_res.status().ToString());

    // Resolve filesystem and validate file existence before creating reader.
    //
    // Whether a config failure here is the caller's depends on whether they
    // named a concrete location. An absolute URI is theirs; a relative path is
    // resolved against the deployment's own fs.* settings, and blaming the
    // caller for those sent them to fix something only an operator can reach.
    const bool location_is_user_supplied = milvus_storage::HasUriScheme(file_path);
    auto fs_res = milvus_storage::FilesystemCache::getInstance().get(properties_map, file_path);
    RETURN_USER_SOURCE_ERROR_IF_AT(fs_res.status(), LOON_ARROW_ERROR, location_is_user_supplied,
                                   fs_res.status().ToString());
    auto fs = fs_res.ValueOrDie();

    auto uri_res = milvus_storage::StorageUri::Parse(file_path);
    RETURN_USER_SOURCE_ERROR_IF(uri_res.status(), LOON_SOURCE_INVALID, "Failed to parse file_path URI '", file_path,
                                "': ", uri_res.status().ToString());
    std::string resolved_path = uri_res->scheme.empty() ? file_path : uri_res->key;

    // file_path is user-supplied: not-found / access-denied are user errors.
    auto file_info_res = fs->GetFileInfo(resolved_path);
    // Authentication usually fails HERE rather than while resolving the
    // filesystem, so this carries the same provenance -- otherwise a rejected
    // deployment credential still came back as the caller's fault.
    RETURN_USER_SOURCE_ERROR_IF_AT(file_info_res.status(), LOON_ARROW_ERROR, location_is_user_supplied,
                                   file_info_res.status().ToString());
    auto file_info = file_info_res.ValueOrDie();

    if (file_info.type() == arrow::fs::FileType::NotFound) {
      RETURN_ERROR(LOON_SOURCE_INVALID, "File not found: ", file_path);
    }
    if (file_info.type() != arrow::fs::FileType::File) {
      RETURN_ERROR(LOON_SOURCE_INVALID, "Path is not a file: ", file_path);
    }

    // Create a ColumnGroupFile to pass to the reader factory
    ColumnGroupFile cg_file{std::string(file_path), 0, 0, {}};
    auto reader_res = fmt_res.ValueOrDie()->create_reader(nullptr, cg_file, properties_map, {}, nullptr);
    RETURN_USER_SOURCE_ERROR_IF_AT(reader_res.status(), LOON_ARROW_ERROR, location_is_user_supplied,
                                   reader_res.status().ToString());

    auto rg_infos_res = reader_res.ValueOrDie()->get_row_group_infos();
    RETURN_USER_SOURCE_ERROR_IF_AT(rg_infos_res.status(), LOON_ARROW_ERROR, location_is_user_supplied,
                                   rg_infos_res.status().ToString());
    auto& rg_infos = rg_infos_res.ValueOrDie();
    *out_num_of_rows = rg_infos.empty() ? 0 : rg_infos.back().end_offset;

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_exttable_read_manifest(const char* manifest_file_path,
                                          const ::LoonProperties* properties,
                                          LoonManifest** out_manifest) {
  if (!manifest_file_path || !properties || !out_manifest) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: manifest_file_path, properties, and out_manifest must not be null");
  }

  try {
    // Unlike explore/get_file_info, this entry point is NOT handed a
    // user-supplied location: manifest_file_path is the path
    // loon_exttable_explore generated and milvus stored. A missing manifest is
    // a GC race or lost data (Missing); a rejected credential or absent bucket
    // is deployment configuration (Config). The producer's classification
    // stands -- nothing here may re-tag to LOON_SOURCE_INVALID.
    milvus_storage::api::Properties properties_map;
    auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    auto fs_res = milvus_storage::FilesystemCache::getInstance().get(properties_map, manifest_file_path);
    RETURN_ARROW_ERROR_IF(fs_res.status(), LOON_ARROW_ERROR, fs_res.status().ToString());

    auto manifest_res = milvus_storage::api::Manifest::ReadFrom(fs_res.ValueOrDie(), manifest_file_path);
    RETURN_ARROW_ERROR_IF(manifest_res.status(), LOON_ARROW_ERROR, manifest_res.status().ToString());
    auto manifest = manifest_res.ValueOrDie();

    // Export full manifest including column groups, delta logs, and stats
    auto st = milvus_storage::manifest_export(manifest, out_manifest);
    RETURN_ARROW_ERROR_IF(st, LOON_LOGICAL_ERROR, st.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}
