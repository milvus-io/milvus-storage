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

/**
 * @brief Extracts metadata from a single external Parquet file
 *
 * This function reads metadata from a single Parquet file, extracting the row count
 * and Arrow schema structure.
 *
 * @param format The file format type (currently only "parquet" is supported)
 * @param file_path Path to a single Parquet file (must be a file, not a directory)
 * @param properties Configuration properties for filesystem access (e.g., S3 credentials, Azure config)
 * @param out_num_of_rows Output parameter containing the number of rows in the file
 * @param out_schema Output Arrow schema structure extracted from the Parquet file.
 *                   The schema is exported using Arrow C Data Interface.
 *                   Caller must call out_schema->release() to free resources.
 *
 * @return FFIResult with:
 *         - LOON_SUCCESS: Operation completed successfully
 *         - LOON_INVALID_PROPERTIES: Failed to parse properties
 *         - LOON_INVALID_ARGS: Invalid format, file not found, path is not a file, or file extension mismatch
 *         - LOON_ARROW_ERROR: Arrow/Parquet library error during file operations
 *
 * @note This function supports both local and cloud storage (S3, GCS, Azure, etc.) through
 *       Arrow filesystem abstraction configured via properties parameter.
 */
FFIResult exttable_get_file_info(const char* format,
                                 const char* file_path,
                                 const Properties* properties,
                                 uint64_t* out_num_of_rows,
                                 struct ArrowSchema* out_schema);

/**
 * @brief Generates column groups from external file paths
 *
 * This function creates a ColumnGroups structure containing a single column group
 * with the specified columns and file paths. It's used for external table support.
 *
 * @param columns Array of column names to include in the column group
 * @param col_lens Number of columns in the columns array
 * @param format File format (e.g., "parquet", "vortex")
 * @param paths Array of file paths
 * @param start_indices Optional array of start row indices for each file
 * @param end_indices Optional array of end row indices for each file
 * @param file_lens Number of files in the paths array
 * @param out_column_groups Output parameter for the generated ColumnGroups handle
 *
 * @return FFIResult
 *
 * @note The caller is responsible for calling column_groups_destroy() on the output handle
 *       The start_indices and end_indices are optional, It describes the logical range of the file.
 *       If not provided, the full range of the file will be used.
 */
FFIResult exttable_generate_column_groups(char** columns,
                                          size_t col_lens,
                                          char* format,
                                          char** paths,
                                          int64_t* start_indices,
                                          int64_t* end_indices,
                                          size_t file_lens,
                                          ColumnGroupsHandle* out_column_groups);

#endif  // LOON_FFI_EXTERNAL_TABLE_C

#ifdef __cplusplus
}
#endif
