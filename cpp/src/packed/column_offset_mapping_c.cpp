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

#include "packed/column_offset_mapping_c.h"
#include "packed/column_group.h"

int NewColumnOffsetMapping(CColumnOffsetMapping* c_column_offset_mapping) {
  try {
    auto column_offset_mapping = std::make_unique<milvus_storage::ColumnOffsetMapping>();
    *c_column_offset_mapping = column_offset_mapping.release();
    return 0;
  } catch (const std::exception& ex) {
    return -1;
  }
}

void DeleteColumnOffsetMapping(CColumnOffsetMapping c_column_offset_mapping) {
  auto column_offset_mapping = (milvus_storage::ColumnOffsetMapping*)c_column_offset_mapping;
  delete column_offset_mapping;
}

int AddColumnOffset(CColumnOffsetMapping c_column_offset_mapping,
                    const char* field_name,
                    int64_t path_index,
                    int64_t col_index) {
  try {
    auto column_offset_mapping = (milvus_storage::ColumnOffsetMapping*)c_column_offset_mapping;
    std::string field_name_str(field_name);
    column_offset_mapping->AddColumnOffset(field_name_str, path_index, col_index);
    return 0;
  } catch (const std::exception& ex) {
    return -1;
  }
}

void GetColumnOffsetMappingKeys(CColumnOffsetMapping c_column_offset_mapping, void* keys) {
  auto column_offset_mapping = (milvus_storage::ColumnOffsetMapping*)c_column_offset_mapping;
  const char** keys_ = (const char**)keys;
  auto map_ = column_offset_mapping->GetMapping();
  std::size_t i = 0;
  for (auto it = map_.begin(); it != map_.end(); ++it, i++) {
    keys_[i] = it->first.c_str();
  }
}

int GetColumnOffsetMappingSize(CColumnOffsetMapping c_column_offset_mapping) {
  auto column_offset_mapping = (milvus_storage::ColumnOffsetMapping*)c_column_offset_mapping;
  return column_offset_mapping->Size();
}

int GetColumnOffset(CColumnOffsetMapping c_column_offset_mapping,
                    const char* field_name,
                    int* path_index,
                    int* col_index) {
  auto column_offset_mapping = (milvus_storage::ColumnOffsetMapping*)c_column_offset_mapping;
  std::string field_name_str(field_name);
  auto column_offset = column_offset_mapping->GetByFieldName(field_name_str);
  if (column_offset == nullptr) {
    return 0;
  }
  *path_index = column_offset->path_index;
  *col_index = column_offset->col_index;
  return 0;
}
