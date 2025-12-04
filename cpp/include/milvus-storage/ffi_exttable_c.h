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

typedef struct ffi_result FFIResult;
typedef struct Properties Properties;

typedef struct ExploreFile {
  char* file_path;
  int64_t start_indices;
  int64_t end_indices;

  // private data, used by internal
  char* private_data;
  size_t pdsize;
} ExploreFile;

typedef struct ExploreFiles {
  ExploreFile *files;
  size_t counts;
  char *format;
} ExploreFiles;

/**
 * @brief Import external files into a dataset
 *
 * @param columns Array of column names
 * @param col_lens Array of column names sizes
 * @param format The file format type (currently only "parquet" is supported)
 * @param base_dir base directory path
 * @param dir_path directory path
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_manifest_path output manifest path
 * @return FFIResult
 */
FFIResult exttable_explore(char** columns,
                          size_t col_lens,
                          const char* format,
                          const char* base_dir,
                          const char* dir_path,
                          const Properties* properties,
                          char* out_manifest_path);

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
                                 uint64_t* out_num_of_rows,
                                 struct ArrowSchema* out_schema);

/**
 * @brief Generate column groups from external files
 *
 * @param columns Array of column names
 * @param col_lens Array of column names sizes
 * @param in_files Input files
 * @param out_column_groups Output parameter for the generated column groups
 * @return FFIResult
 */
FFIResult exttable_generate_column_groups(char** columns,
                                          size_t col_lens,
                                          const ExploreFiles* in_files,
                                          ColumnGroupsHandle* out_column_groups);

/**
 * @brief Destroy an ExploreFiles object
 *
 * @param files The ExtTableFiles object to destroy
 */
void exttable_files_destroy(ExtTableFiles* files);

#endif  // LOON_FFI_EXTERNAL_TABLE_C

#ifdef __cplusplus
}
#endif
