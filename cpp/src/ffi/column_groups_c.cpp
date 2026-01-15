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

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/manifest.h"

using namespace milvus_storage::api;

LoonFFIResult loon_column_groups_create(const char** columns,
                                        size_t col_lens,
                                        char* format,
                                        char** paths,
                                        int64_t* start_indices,
                                        int64_t* end_indices,
                                        size_t file_lens,
                                        LoonColumnGroups** out_column_groups) {
  if (!columns || !col_lens || !paths || !format || !file_lens || !out_column_groups || !start_indices ||
      !end_indices) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments");
  }

  try {
    // The external table will generate a `single` column group
    ColumnGroups cgs;
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
    cgs.push_back(cg);

    // Export to LoonColumnGroups structure
    auto st = milvus_storage::export_column_groups(cgs, out_column_groups);
    if (!st.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

static void destroy_column_group_file(LoonColumnGroupFile* ccgf) {
  if (!ccgf)
    return;

  // Free path
  if (ccgf->path) {
    delete[] const_cast<char*>(ccgf->path);
    ccgf->path = nullptr;
  }

  // Free metadata
  if (ccgf->metadata) {
    delete[] ccgf->metadata;
    ccgf->metadata = nullptr;
    ccgf->metadata_size = 0;
  }
}

void destroy_column_group(LoonColumnGroup* ccg) {
  if (!ccg)
    return;

  // Free column names
  if (ccg->columns) {
    for (uint32_t i = 0; i < ccg->num_of_columns; i++) {
      if (ccg->columns[i]) {
        delete[] const_cast<char*>(ccg->columns[i]);
      }
    }
    delete[] ccg->columns;
    ccg->columns = nullptr;
    ccg->num_of_columns = 0;
  }

  // Free format
  if (ccg->format) {
    delete[] const_cast<char*>(ccg->format);
    ccg->format = nullptr;
  }

  // Free files
  if (ccg->files) {
    for (uint32_t i = 0; i < ccg->num_of_files; i++) {
      destroy_column_group_file(&ccg->files[i]);
    }
    delete[] ccg->files;
    ccg->files = nullptr;
    ccg->num_of_files = 0;
  }
}

// Helper function to destroy embedded column groups (used by loon_manifest_destroy)
void destroy_column_groups_contents(LoonColumnGroups* cgroups) {
  if (!cgroups)
    return;

  // Destroy column groups
  if (cgroups->column_group_array) {
    for (uint32_t i = 0; i < cgroups->num_of_column_groups; i++) {
      destroy_column_group(&cgroups->column_group_array[i]);
    }
    delete[] cgroups->column_group_array;
    cgroups->column_group_array = nullptr;
    cgroups->num_of_column_groups = 0;
  }
}

void loon_column_groups_destroy(LoonColumnGroups* cgroups) {
  if (!cgroups)
    return;

  // Destroy contents
  destroy_column_groups_contents(cgroups);

  // Free the structure itself
  delete cgroups;
}
