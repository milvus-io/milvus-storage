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

#include <cstring>
#include <optional>
#include <sstream>

#include <parquet/arrow/reader.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type_fwd.h>
#include <avro/Stream.hh>
#include <avro/Decoder.hh>

#include "milvus-storage/ffi_c.h"
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

using namespace milvus_storage::api;

#ifdef BUILD_VORTEX_BRIDGE
using namespace milvus_storage::vortex;
#endif
using namespace milvus_storage::api::transaction;

struct ColumnGroupsExporter;
struct ColumnGroupsImporter;

static inline arrow::Result<std::vector<milvus_storage::api::ColumnGroupFile>> get_cg_files(
    const std::shared_ptr<arrow::fs::FileSystem>& fs, const char* base_dir) {
  arrow::fs::FileSelector selector;
  selector.base_dir = std::string(base_dir);
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

  std::vector<ColumnGroupFile> files;
  for (const auto& file_info : file_infos) {
    const std::string& file_name = file_info.base_name();
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    files.emplace_back(milvus_storage::api::ColumnGroupFile{
        file_info.path() + kSep + file_name, -1, /*start_index */
        -1,                                      /*end_index */
        std::vector<uint8_t>(),                  /*metadata */
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
    milvus_storage::ArrowFileSystemConfig fs_config;
    milvus_storage::api::Properties properties_map;
    std::string format_str(format);

    auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    // Configure filesystem from properties
    auto status = milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    auto fs_res = milvus_storage::CreateArrowFileSystem(fs_config);
    if (!fs_res.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, fs_res.status().ToString());
    }
    auto fs = fs_res.ValueOrDie();

    // list path and get files
    auto cg_files_result = get_cg_files(fs, explore_dir);
    if (!cg_files_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, cg_files_result.status().ToString());
    }
    auto files = cg_files_result.ValueOrDie();

    std::vector<std::string> columns_cpp;
    for (size_t i = 0; i < col_lens; i++) {
      columns_cpp.emplace_back(columns[i]);
    }

    // construct the column groups
    ColumnGroups cgs;
    cgs.push_back(std::make_shared<ColumnGroup>(
        ColumnGroup{.columns = columns_cpp, .format = std::string(format), .files = files}));

    // commit the column groups
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
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
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
    milvus_storage::ArrowFileSystemConfig fs_config;
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

    // Check file_path is a file
    auto file_info_res = fs->GetFileInfo(file_path);
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
    }
#ifdef BUILD_VORTEX_BRIDGE
    else if (format_str == LOON_FORMAT_VORTEX) {
      VortexFormatReader reader(fs, nullptr /* schema */, file_path, properties_map,
                                std::vector<std::string>{} /* projection */);
      auto open_result = reader.open();
      if (!open_result.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, "Open failed. " + open_result.ToString());
      }
      *out_num_of_rows = reader.rows();
    }
#endif
    else {
      RETURN_ERROR(LOON_INVALID_ARGS, "Unsupported format: " + format_str + ", file_path: " + file_path);
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

static arrow::Result<std::shared_ptr<milvus_storage::api::Manifest>> read_manifest(const char* path,
                                                                                   const ::LoonProperties* properties) {
  milvus_storage::ArrowFileSystemConfig fs_config;
  milvus_storage::api::Properties properties_map;

  auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    return arrow::Status::Invalid("Failed to parse properties [", opt->c_str(), "]");
  }

  // Configure filesystem from properties
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config));
  ARROW_ASSIGN_OR_RAISE(auto fs, milvus_storage::CreateArrowFileSystem(fs_config));

  // Open input file and get size
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());
  ARROW_ASSIGN_OR_RAISE(auto column_groups_buffer, input_file->Read(file_size));

  // Ensure we read the expected size
  if (column_groups_buffer->size() != file_size) {
    return arrow::Status::IOError("Failed to read the complete file, expected size =", file_size,
                                  ", actual size =", static_cast<int64_t>(column_groups_buffer->size()));
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
    auto st = milvus_storage::export_manifest(manifest, out_manifest);
    if (!st.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}
