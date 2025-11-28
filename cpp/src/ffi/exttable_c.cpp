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

#include <parquet/arrow/reader.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type_fwd.h>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

using namespace milvus_storage::api;

#ifdef BUILD_VORTEX_BRIDGE
using namespace milvus_storage::vortex;
#endif

FFIResult exttable_get_file_info(const char* format,
                                 const char* file_path,
                                 const ::Properties* properties,
                                 uint64_t* out_num_of_rows,
                                 [[maybe_unused]] struct ArrowSchema* out_schema) {
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
      VortexFormatReader reader(std::make_shared<FileSystemWrapper>(fs), nullptr /* schema */, file_path,
                                {} /* projection */);
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

FFIResult exttable_generate_column_groups(char** columns,
                                          size_t col_lens,
                                          char* format,
                                          char** paths,
                                          int64_t* start_indices,  // optional
                                          int64_t* end_indices,    // optional
                                          size_t file_lens,
                                          ColumnGroupsHandle* out_column_groups) {
  if (!columns || !col_lens || !paths || !format || !file_lens || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments");
  }

  try {
    // The external table will generate a `single` column group
    std::shared_ptr<ColumnGroups> cgs = std::make_shared<ColumnGroups>();
    std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
    cg->columns.reserve(col_lens);
    for (size_t col_idx = 0; col_idx < col_lens; col_idx++) {
      cg->columns.emplace_back(columns[col_idx]);
    }

    cg->files.reserve(file_lens);
    for (size_t file_idx = 0; file_idx < file_lens; file_idx++) {
      if (!paths[file_idx]) {
        RETURN_ERROR(LOON_INVALID_ARGS, "Path is null [index=" + std::to_string(file_idx) + "]");
      }

      if (start_indices && end_indices) {
        cg->files.emplace_back(ColumnGroupFile{paths[file_idx], start_indices[file_idx], end_indices[file_idx]});
      } else {
        cg->files.emplace_back(ColumnGroupFile{
            paths[file_idx],
            std::nullopt,
            std::nullopt,
        });
      }
    }
    cg->format = format;
    auto status = cgs->add_column_group(std::move(cg));
    if (!status.ok()) {
      RETURN_ERROR(LOON_INVALID_ARGS, status.ToString());
    }

    *out_column_groups = reinterpret_cast<ColumnGroupsHandle>(new std::shared_ptr<ColumnGroups>(cgs));

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}
