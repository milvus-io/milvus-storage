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
#include <optional>
#include <sstream>

#include <parquet/arrow/reader.h>
#include <arrow/c/bridge.h>
#include <arrow/type_fwd.h>
#include <avro/Stream.hh>
#include <avro/Decoder.hh>

#include "milvus-storage/common/path_util.h"  // for kSep
#include "milvus-storage/common/layout.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/format/lance/lance_common.h"

using namespace milvus_storage::api;
using namespace milvus_storage::vortex;
using namespace milvus_storage::lance;
using namespace milvus_storage::api::transaction;

struct ColumnGroupsExporter;
struct ColumnGroupsImporter;

static inline arrow::Result<std::vector<milvus_storage::api::ColumnGroupFile>> get_cg_files(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::string& base_dir,
    const milvus_storage::api::Properties& properties,
    const milvus_storage::StorageUri* explore_uri = nullptr) {
  arrow::fs::FileSelector selector;
  selector.base_dir = base_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

  // Build URI base from explore_uri (if provided) or fall back to properties
  milvus_storage::StorageUri uri_base;
  if (explore_uri != nullptr && !explore_uri->scheme.empty()) {
    uri_base.scheme = explore_uri->scheme;
    uri_base.address = explore_uri->address;
    uri_base.bucket_name = explore_uri->bucket_name;
  } else if (milvus_storage::IsLocalFileSystem(fs)) {
    uri_base.scheme = fs->type_name();
    uri_base.bucket_name = "local";
  } else {
    uri_base.scheme = fs->type_name();
    ARROW_ASSIGN_OR_RAISE(uri_base.address, GetValue<std::string>(properties, PROPERTY_FS_ADDRESS));
    ARROW_ASSIGN_OR_RAISE(uri_base.bucket_name, GetValue<std::string>(properties, PROPERTY_FS_BUCKET_NAME));
  }

  std::vector<ColumnGroupFile> files;
  for (const auto& file_info : file_infos) {
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    uri_base.key = file_info.path();
    ARROW_ASSIGN_OR_RAISE(auto file_uri, milvus_storage::StorageUri::Make(uri_base));
    files.emplace_back(milvus_storage::api::ColumnGroupFile{
        std::move(file_uri), -1, /*start_index */
        -1,                      /*end_index */
        std::vector<uint8_t>(),  /*metadata */
    });
  }

  return files;
}

static inline arrow::Result<std::vector<ColumnGroupFile>> get_lance_cg_files(const char* explore_dir,
                                                                             const Properties& properties) {
  // Resolve config: if explore_dir is a URI, use matching extfs config; otherwise default fs config
  ARROW_ASSIGN_OR_RAISE(auto fs_config, milvus_storage::FilesystemCache::resolve_config(properties, explore_dir));

  // Extract key from URI for BuildLanceBaseUri (needs relative path, not full URI)
  std::string resolved_dir(explore_dir);
  auto uri_res = milvus_storage::StorageUri::Parse(explore_dir);
  if (uri_res.ok() && !uri_res->scheme.empty()) {
    resolved_dir = uri_res->key;
  }

  ARROW_ASSIGN_OR_RAISE(auto lance_base_uri, BuildLanceBaseUri(fs_config, resolved_dir));
  auto storage_options = ToLanceStorageOptions(fs_config);

  auto dataset = BlockingDataset::Open(lance_base_uri, storage_options);
  auto fragment_ids = dataset->GetAllFragmentIds();

  std::vector<ColumnGroupFile> files;
  for (auto frag_id : fragment_ids) {
    auto row_count = dataset->GetFragmentRowCount(frag_id);
    files.emplace_back(ColumnGroupFile{
        MakeLanceUri(lance_base_uri, frag_id), 0, /*start_index */
        static_cast<int64_t>(row_count),          /*end_index */
        std::vector<uint8_t>(),                   /*metadata */
    });
  }

  return files;
}

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

    std::vector<ColumnGroupFile> files;

    if (format_str == LOON_FORMAT_LANCE_TABLE) {
      auto lance_files_result = get_lance_cg_files(explore_dir, properties_map);
      if (!lance_files_result.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, lance_files_result.status().ToString());
      }
      files = lance_files_result.ValueOrDie();
    } else {
      // Resolve explore_dir: if URI, extract key for filesystem ops and use URI for file URI construction
      std::string resolved_explore_dir(explore_dir);
      milvus_storage::StorageUri explore_uri;
      auto uri_res = milvus_storage::StorageUri::Parse(explore_dir);
      if (uri_res.ok() && !uri_res->scheme.empty()) {
        explore_uri = uri_res.ValueOrDie();
        resolved_explore_dir = explore_uri.key;
      }

      // create external filesystem for parquet/vortex
      auto ext_fs_res = milvus_storage::FilesystemCache::getInstance().get(properties_map, explore_dir);
      if (!ext_fs_res.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, ext_fs_res.status().ToString());
      }
      auto ext_fs = ext_fs_res.ValueOrDie();

      // list path and get files using the external filesystem
      auto cg_files_result = get_cg_files(ext_fs, resolved_explore_dir, properties_map,
                                          explore_uri.scheme.empty() ? nullptr : &explore_uri);
      if (!cg_files_result.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, cg_files_result.status().ToString());
      }
      files = cg_files_result.ValueOrDie();
    }

    // create origin filesystem (always needed for Transaction commit)
    auto fs_res = milvus_storage::FilesystemCache::getInstance().get(properties_map);
    if (!fs_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_res.status().ToString());
    }
    auto fs = fs_res.ValueOrDie();

    std::vector<std::string> columns_cpp;
    for (size_t i = 0; i < col_lens; i++) {
      columns_cpp.emplace_back(columns[i]);
    }

    // construct the column groups
    ColumnGroups cgs;
    cgs.push_back(std::make_shared<ColumnGroup>(
        ColumnGroup{.columns = columns_cpp, .format = std::string(format), .files = files}));

    // commit the column groups with origin filesystem
    auto transaction_result = Transaction::Open(fs, base_path);
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

    // Create or resolve filesystem based on the file path
    // This handles both external filesystems (for absolute URIs) and default filesystem
    auto fs_res = milvus_storage::FilesystemCache::getInstance().get(properties_map, file_path);
    if (!fs_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_res.status().ToString());
    }
    milvus_storage::ArrowFileSystemPtr fs = fs_res.ValueOrDie();

    // Resolve URI to get the key path for filesystem operations
    auto uri_res = milvus_storage::StorageUri::Parse(file_path);
    if (!uri_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to parse file_path URI '", file_path, "': ", uri_res.status().ToString());
    }
    auto uri = uri_res.ValueOrDie();
    std::string resolved_path = uri.scheme.empty() ? file_path : uri.key;

    // Check file_path is a file
    auto file_info_res = fs->GetFileInfo(resolved_path);
    if (!file_info_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, file_info_res.status().ToString());
    }
    auto file_info = file_info_res.ValueOrDie();

    if (file_info.type() == arrow::fs::FileType::NotFound) {
      RETURN_ERROR(LOON_INVALID_ARGS, "File not found: ", file_path);
    }

    if (file_info.type() != arrow::fs::FileType::File) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Path is not a file: ", file_path);
    }

    if (format_str == LOON_FORMAT_PARQUET) {
      // Open and read file metadata
      auto file_reader_res = fs->OpenInputFile(file_info.path());
      if (!file_reader_res.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, file_reader_res.status().ToString());
      }
      auto file_reader = file_reader_res.ValueOrDie();

      auto parquet_reader = parquet::ParquetFileReader::Open(file_reader);
      *out_num_of_rows = parquet_reader->metadata()->num_rows();

      auto close_status = file_reader->Close();
      if (!close_status.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, close_status.ToString());
      }
    } else if (format_str == LOON_FORMAT_VORTEX) {
      VortexFormatReader reader(fs, nullptr /* schema */, resolved_path, properties_map,
                                std::vector<std::string>{} /* projection */);
      auto open_result = reader.open();
      if (!open_result.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, "Open failed. " + open_result.ToString());
      }
      *out_num_of_rows = reader.rows();
    } else {
      RETURN_ERROR(LOON_INVALID_ARGS, "Unsupported format: " + format_str + ", file_path: " + file_path);
    }

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

  // Open input file and get size
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());
  ARROW_ASSIGN_OR_RAISE(auto column_groups_buffer, input_file->Read(file_size));

  // Ensure we read the expected size
  if (column_groups_buffer->size() != file_size) {
    return arrow::Status::IOError(fmt::format("Failed to read the complete file, expected size ={}, actual size ={}",
                                              file_size,  // NOLINT
                                              static_cast<int64_t>(column_groups_buffer->size())));
  }
  ARROW_RETURN_NOT_OK(input_file->Close());

  // Read as Manifest
  auto manifest = std::make_shared<milvus_storage::api::Manifest>();
  std::string manifest_data(reinterpret_cast<const char*>(column_groups_buffer->data()), column_groups_buffer->size());
  std::istringstream in(manifest_data);
  ARROW_RETURN_NOT_OK(manifest->deserialize(in));
  return manifest;
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
