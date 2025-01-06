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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef void* CColumnOffsetMapping;

int NewColumnOffsetMapping(CColumnOffsetMapping* c_column_offset_mapping);

void DeleteColumnOffsetMapping(CColumnOffsetMapping c_column_offset_mapping);

int AddColumnOffset(CColumnOffsetMapping c_column_offset_mapping,
                    const char* field_name,
                    int64_t path_index,
                    int64_t col_index);

void GetColumnOffsetMappingKeys(CColumnOffsetMapping c_column_offset_mapping, void* keys);

int GetColumnOffsetMappingSize(CColumnOffsetMapping c_column_offset_mapping);

int GetColumnOffset(CColumnOffsetMapping c_column_offset_mapping,
                    const char* field_name,
                    int* path_index,
                    int* col_index);

#ifdef __cplusplus
}
#endif
