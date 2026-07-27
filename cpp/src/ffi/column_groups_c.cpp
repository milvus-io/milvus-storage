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

#include <cstring>
#include <memory>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/manifest.h"

using namespace milvus_storage::api;

namespace {

char* dup_cstr(const char* value) {
  const auto size = std::strlen(value);
  auto* copy = new char[size + 1];
  std::memcpy(copy, value, size + 1);
  return copy;
}

}  // namespace

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

      cg->files.emplace_back(ColumnGroupFile{
          .path = paths[file_idx],
          .start_index = start_indices[file_idx],
          .end_index = end_indices[file_idx],
      });
    }
    cg->format = format;
    cgs.push_back(cg);

    // Export to LoonColumnGroups structure
    auto st = milvus_storage::column_groups_export(cgs, out_column_groups);
    RETURN_ARROW_ERROR_IF(st, LOON_LOGICAL_ERROR, st.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_column_group_file_set_property(LoonColumnGroups* column_groups,
                                                  uint32_t column_group_index,
                                                  uint32_t file_index,
                                                  const char* key,
                                                  const char* value) {
  if (column_groups == nullptr || key == nullptr || value == nullptr) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Column groups, property key, and property value must be non-null");
  }
  if (column_group_index >= column_groups->num_of_column_groups || column_groups->column_group_array == nullptr) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Column group index is out of range");
  }
  auto& column_group = column_groups->column_group_array[column_group_index];
  if (file_index >= column_group.num_of_files || column_group.files == nullptr) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Column group file index is out of range");
  }

  try {
    auto& file = column_group.files[file_index];
    for (uint32_t index = 0; index < file.num_properties; ++index) {
      if (std::strcmp(file.property_keys[index], key) == 0) {
        std::unique_ptr<char[]> replacement(dup_cstr(value));
        delete[] const_cast<char*>(file.property_values[index]);
        file.property_values[index] = replacement.release();
        RETURN_SUCCESS();
      }
    }

    const auto old_size = file.num_properties;
    auto keys = std::make_unique<const char*[]>(old_size + 1);
    auto values = std::make_unique<const char*[]>(old_size + 1);
    std::unique_ptr<char[]> new_key(dup_cstr(key));
    std::unique_ptr<char[]> new_value(dup_cstr(value));
    for (uint32_t index = 0; index < old_size; ++index) {
      keys[index] = file.property_keys[index];
      values[index] = file.property_values[index];
    }
    keys[old_size] = new_key.release();
    values[old_size] = new_value.release();
    delete[] file.property_keys;
    delete[] file.property_values;
    file.property_keys = keys.release();
    file.property_values = values.release();
    file.num_properties = old_size + 1;
    RETURN_SUCCESS();
  } catch (const std::exception& error) {
    RETURN_EXCEPTION(error.what());
  }

  RETURN_UNREACHABLE();
}

static void destroy_column_group_file(LoonColumnGroupFile* ccgf) {
  if (!ccgf) {
    return;
  }

  // Free path
  if (ccgf->path) {
    delete[] const_cast<char*>(ccgf->path);
    ccgf->path = nullptr;
  }

  // Free properties
  if (ccgf->property_keys) {
    for (uint32_t i = 0; i < ccgf->num_properties; i++) {
      delete[] const_cast<char*>(ccgf->property_keys[i]);
      delete[] const_cast<char*>(ccgf->property_values[i]);
    }
    delete[] ccgf->property_keys;
    delete[] ccgf->property_values;
    ccgf->property_keys = nullptr;
    ccgf->property_values = nullptr;
    ccgf->num_properties = 0;
  }
}

void destroy_column_group(LoonColumnGroup* ccg) {
  if (!ccg) {
    return;
  }

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
  if (!cgroups) {
    return;
  }

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
  if (!cgroups) {
    return;
  }

  // Destroy contents
  destroy_column_groups_contents(cgroups);

  // Free the structure itself
  delete cgroups;
}

char* loon_column_groups_debug_string(const LoonColumnGroups* cgroups) {
  std::string result = milvus_storage::column_groups_debug_string(cgroups);
  return strdup(result.c_str());
}
