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

#include <memory>
#include <optional>
#include <vector>
#include <assert.h>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/transaction/manifest.h"

using namespace milvus_storage::api;

FFIResult column_groups_export(ColumnGroupsHandle handle, CColumnGroups* out_ccgs) {
  if (!handle || !out_ccgs) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_ccgs must not be null");
  }

  std::shared_ptr<ColumnGroups> cgs = *reinterpret_cast<std::shared_ptr<ColumnGroups>*>(handle);
  auto st = export_column_groups(cgs.get(), out_ccgs);
  if (!st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
  }

  RETURN_SUCCESS();
}

FFIResult column_groups_import(CColumnGroups* in_cgs, ColumnGroupsHandle* handle) {
  if (!in_cgs || !handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: in_cgs and handle must not be null");
  }

  std::shared_ptr<ColumnGroups> cgs = std::make_shared<ColumnGroups>();
  auto st = import_column_groups(in_cgs, cgs.get());
  if (!st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
  }

  *handle = reinterpret_cast<ColumnGroupsHandle>(new std::shared_ptr<ColumnGroups>(cgs));
  RETURN_SUCCESS();
}

FFIResult column_groups_create(const char** columns,
                               size_t col_lens,
                               char* format,
                               char** paths,
                               int64_t* start_indices,
                               int64_t* end_indices,
                               size_t file_lens,
                               ColumnGroupsHandle* out_column_groups) {
  if (!columns || !col_lens || !paths || !format || !file_lens || !out_column_groups || !start_indices ||
      !end_indices) {
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

      cg->files.emplace_back(ColumnGroupFile{paths[file_idx], start_indices[file_idx], end_indices[file_idx]});
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

void column_groups_ptr_destroy(ColumnGroupsHandle handle) {
  if (handle) {
    delete reinterpret_cast<std::shared_ptr<ColumnGroups>*>(handle);
  }
}
