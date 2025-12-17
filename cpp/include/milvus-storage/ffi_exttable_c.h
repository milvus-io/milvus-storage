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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LOON_FFI_EXTERNAL_TABLE_C
#define LOON_FFI_EXTERNAL_TABLE_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <arrow/c/abi.h>

struct ffi_result;
struct Properties;
struct CColumnGroups;

typedef struct ffi_result FFIResult;
typedef struct Properties Properties;
typedef struct CColumnGroups CColumnGroups;

typedef uintptr_t ColumnGroupsHandle;

/**
 * @brief Import external files into a dataset
 *
 * @param columns Array of column names
 * @param col_lens Array of column names sizes
 * @param format The file format type (currently only "parquet" is supported)
 * @param base_dir base directory path
 * @param dir_path directory path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_num_of_files output number of files
 * @param out_column_groups_file_path output column groups file path, need call `free_cstr` to free memory
 * @return FFIResult
 */
FFIResult exttable_explore(const char** columns,
                           size_t col_lens,
                           const char* format,
                           const char* base_dir,
                           const char* explore_dir,
                           const Properties* properties,
                           uint64_t* out_num_of_files,
                           char** out_column_groups_file_path);

/**
 * @brief Get file info
 *
 * @param format The file format type (currently only "parquet" is supported)
 * @param file_path file path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_num_of_rows output number of rows
 * @param out_schema output schema
 * @return FFIResult
 */
FFIResult exttable_get_file_info(const char* format,
                                 const char* file_path,
                                 const Properties* properties,
                                 uint64_t* out_num_of_rows);

/**
 * @brief Read column groups from file
 *
 * @param out_column_groups_file_path output column groups file path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_column_groups output column groups
 * @return FFIResult
 */
FFIResult exttable_read_column_groups(const char* out_column_groups_file_path,
                                      const Properties* properties,
                                      CColumnGroups* out_column_groups);

#endif  // LOON_FFI_EXTERNAL_TABLE_C

#ifdef __cplusplus
}
#endif
