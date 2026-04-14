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

#include "test_runner.h"
#include "ffi_test_env.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <arrow/c/abi.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_exttable_c.h"

#define TEST_ROOT_PATH FFI_TEST_ROOT_PATH
#define TEST_BASE_PATH "external-test-dir"

// Forward declarations from arrow_c_wrapper.c and ffi_writer_test.c
void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable);
struct ArrowSchema* create_test_struct_schema();
struct ArrowArray* create_int64_array(const int64_t* data,
                                      int64_t length,
                                      const uint8_t* null_bitmap,
                                      int64_t null_count);
struct ArrowArray* create_int32_array(const int32_t* data,
                                      int64_t length,
                                      const uint8_t* null_bitmap,
                                      int64_t null_count);
struct ArrowArray* create_string_array(const char** data,
                                       int64_t length,
                                       const uint8_t* null_bitmap,
                                       int64_t null_count);
struct ArrowArray* create_struct_array(struct ArrowArray** children, int64_t n_children, int64_t length);
struct ArrowArray* create_test_struct_arrow_array(int64_t* int64_data,
                                                  int32_t* int32_data,
                                                  const char** str_data,
                                                  int length);

static FileSystemHandle get_fs(LoonProperties* pp) {
  FileSystemHandle fs = 0;
  LoonFFIResult rc = loon_filesystem_get(pp, TEST_ROOT_PATH, strlen(TEST_ROOT_PATH), &fs);
  assert(loon_ffi_is_success(&rc));
  return fs;
}

/// Find the first file under `dir_path` whose name ends with `suffix`.
/// Returns true and writes the path into `out_path` (relative to fs root).
/// Returns false if no matching file found.
static bool find_first_file(
    FileSystemHandle fs, const char* dir_path, const char* suffix, char* out_path, size_t out_path_size) {
  LoonFileInfoList list = {0};
  LoonFFIResult rc = loon_filesystem_list_dir(fs, dir_path, (uint32_t)strlen(dir_path), true, &list);
  if (!loon_ffi_is_success(&rc)) {
    loon_ffi_free_result(&rc);
    return false;
  }
  size_t suffix_len = strlen(suffix);
  for (uint32_t i = 0; i < list.count; i++) {
    if (list.entries[i].is_dir)
      continue;
    if (list.entries[i].path_len >= suffix_len &&
        strcmp(list.entries[i].path + list.entries[i].path_len - suffix_len, suffix) == 0) {
      snprintf(out_path, out_path_size, "%s", list.entries[i].path);
      loon_filesystem_free_file_info_list(&list);
      return true;
    }
  }
  loon_filesystem_free_file_info_list(&list);
  return false;
}

// Helper function to create test properties
LoonFFIResult create_test_external_pp(LoonProperties* rp, const char* format) {
  const char* keys[500] = {"format"};
  const char* vals[500] = {format ? format : "parquet"};
  size_t count = init_test_props(keys, vals, 1, 500, TEST_ROOT_PATH);

  return loon_properties_create((const char* const*)keys, (const char* const*)vals, count, rp);
}

// Helper function to create a simple parquet file using writer FFI
LoonFFIResult create_testfile(const char* base_path, int64_t num_rows, LoonProperties* props) {
  struct ArrowSchema* schema;
  LoonWriterHandle writer;
  LoonFFIResult rc;
  LoonColumnGroups* column_groups = NULL;

  schema = create_test_struct_schema();
  rc = loon_writer_new(base_path, schema, props, &writer);
  if (!loon_ffi_is_success(&rc)) {
    if (schema->release) {
      schema->release(schema);
    }
    free(schema);
    return rc;
  }

  // Create test data for all 3 fields
  int64_t* int64_data = (int64_t*)malloc(num_rows * sizeof(int64_t));
  int32_t* int32_data = (int32_t*)malloc(num_rows * sizeof(int32_t));
  const char** str_data = (const char**)malloc(num_rows * sizeof(char*));

  for (int64_t i = 0; i < num_rows; i++) {
    int64_data[i] = i;
    int32_data[i] = (int32_t)(i * 10);
    str_data[i] = "test";
  }

  struct ArrowArray* struct_array = create_test_struct_arrow_array(int64_data, int32_data, str_data, num_rows);

  rc = loon_writer_write(writer, struct_array);

  // Clean up data arrays
  free(int64_data);
  free(int32_data);
  free(str_data);

  if (!loon_ffi_is_success(&rc)) {
    loon_writer_destroy(writer);
    if (schema->release) {
      schema->release(schema);
    }
    free(schema);
    // Ensure struct_array is freed if write failed
    if (struct_array->release) {
      struct_array->release(struct_array);
    }
    free(struct_array);
    return rc;
  }

  // Close writer
  rc = loon_writer_close(writer, NULL, NULL, 0, &column_groups);
  if (loon_ffi_is_success(&rc)) {
    loon_column_groups_destroy(column_groups);
  }
  loon_writer_destroy(writer);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);

  return rc;
}

static void test_exttable_get_file_info_single_file(const char* format) {
  LoonFFIResult rc;
  LoonProperties rp;
  uint64_t num_rows = 0;
  char file_path[512];

  rc = create_test_external_pp(&rp, format);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  FileSystemHandle fs = get_fs(&rp);

  // Create a test file (creates directory with file inside)
  rc = create_testfile(TEST_BASE_PATH, 100, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Find the actual file created by the writer (has UUID in name)
  char suffix[32];
  snprintf(suffix, sizeof(suffix), ".%s", format);
  char data_dir[512];
  snprintf(data_dir, sizeof(data_dir), "%s/_data", TEST_BASE_PATH);
  ck_assert(find_first_file(fs, data_dir, suffix, file_path, sizeof(file_path)));
  printf("Found %s file: %s\n", format, file_path);

  // file_path is already relative to fs root
  char* relative_path = file_path;

  // Get file info for the specific file
  rc = loon_exttable_get_file_info(format, relative_path, &rp, &num_rows);

  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(num_rows, 100);

  printf("num_rows=%" PRIu64 "\n", num_rows);

  // Clean up
  loon_filesystem_destroy(fs);
  loon_properties_free(&rp);
}

static void test_exttable_explore_and_read(void) {
  LoonFFIResult rc;
  LoonProperties rp;
  char data_path[512], base_dir[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  FileSystemHandle fs = get_fs(&rp);

  snprintf(base_dir, sizeof(base_dir), "%s-base-dir", TEST_BASE_PATH);
  snprintf(data_path, sizeof(data_path), "%s-data-dir", TEST_BASE_PATH);
  clean_test_dir(fs, base_dir);
  clean_test_dir(fs, data_path);

  // Create some test parquet file
  for (int i = 0; i < 10; i++) {
    rc = create_testfile(data_path, 50, &rp);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  }
  char* columns_cstrs[3] = {"int64_field", "int32_field", "string_field"};

  uint64_t num_of_files = 0;
  char* out_column_groups_file_path = NULL;
  char data_path_with_prefix[1024];
  snprintf(data_path_with_prefix, sizeof(data_path_with_prefix), "%s/_data/", data_path);

  rc = loon_exttable_explore((const char**)(columns_cstrs), 3, "parquet", base_dir, data_path_with_prefix, &rp,
                             &num_of_files, &out_column_groups_file_path);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(num_of_files, 10);

  LoonManifest* out_cmanifest = NULL;
  rc = loon_exttable_read_manifest(out_column_groups_file_path, &rp, &out_cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Check column groups (embedded in manifest)
  ck_assert(out_cmanifest->column_groups.column_group_array != NULL);
  ck_assert_int_eq(out_cmanifest->column_groups.num_of_column_groups, 1);

  LoonColumnGroup* ccg0 = &(out_cmanifest->column_groups.column_group_array[0]);

  ck_assert(ccg0->columns != NULL);
  ck_assert_int_eq(ccg0->num_of_columns, 3);
  ck_assert_str_eq(ccg0->columns[0], columns_cstrs[0]);
  ck_assert_str_eq(ccg0->columns[1], columns_cstrs[1]);
  ck_assert_str_eq(ccg0->columns[2], columns_cstrs[2]);
  ck_assert_str_eq(ccg0->format, "parquet");

  ck_assert(ccg0->files != NULL);
  ck_assert_int_eq(ccg0->num_of_files, 10);
  for (int i = 0; i < 10; i++) {
    ck_assert(ccg0->files[i].path != NULL);
    ck_assert_int_eq(ccg0->files[i].start_index, -1);
    ck_assert_int_eq(ccg0->files[i].end_index, -1);
    ck_assert_int_eq(ccg0->files[i].num_properties, 0);
  }

  loon_free_cstr(out_column_groups_file_path);
  loon_manifest_destroy(out_cmanifest);
  clean_test_dir(fs, base_dir);
  clean_test_dir(fs, data_path);
  loon_filesystem_destroy(fs);
  loon_properties_free(&rp);
}

static void test_exttable_get_file_info_single_file_parquet(void) {
  test_exttable_get_file_info_single_file("parquet");
}

static void test_exttable_get_file_info_single_file_vortex(void) { test_exttable_get_file_info_single_file("vortex"); }

static void test_exttable_get_file_info_directory_error(const char* format) {
  LoonFFIResult rc;
  LoonProperties rp;
  uint64_t num_rows = 0;
  char full_path[512];
  char relative_path[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Create absolute path to a directory
  snprintf(full_path, sizeof(full_path), "/tmp/%s-dir", TEST_BASE_PATH);
  snprintf(relative_path, sizeof(relative_path), "%s-dir", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_testfile(relative_path, 50, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Try to get file info for directory (should fail - not a file)
  rc = loon_exttable_get_file_info(format, relative_path, &rp, &num_rows);

  ck_assert(!loon_ffi_is_success(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error for directory: %s\n", loon_ffi_get_errmsg(&rc));

  // Clean up
  loon_ffi_free_result(&rc);
  loon_properties_free(&rp);
}

static void test_exttable_get_file_info_directory_error_parquet(void) {
  test_exttable_get_file_info_directory_error("parquet");
}

static void test_exttable_get_file_info_directory_error_vortex(void) {
  test_exttable_get_file_info_directory_error("vortex");
}

static void test_exttable_get_file_info_invalid_format(void) {
  LoonFFIResult rc;
  LoonProperties rp;
  uint64_t num_rows = 0;
  char file_path[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  FileSystemHandle fs = get_fs(&rp);

  char relative_path[512];
  snprintf(relative_path, sizeof(relative_path), "%s-invalid", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_testfile(relative_path, 100, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Find the actual parquet file
  char data_dir[512];
  snprintf(data_dir, sizeof(data_dir), "%s/_data", relative_path);
  ck_assert(find_first_file(fs, data_dir, ".parquet", file_path, sizeof(file_path)));

  // Try to get info with invalid format
  rc = loon_exttable_get_file_info("invalid_format", file_path, &rp, &num_rows);

  ck_assert(!loon_ffi_is_success(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", loon_ffi_get_errmsg(&rc));

  // Clean up
  loon_ffi_free_result(&rc);
  loon_filesystem_destroy(fs);
  loon_properties_free(&rp);
}

static void test_exttable_get_file_info_file_not_found(void) {
  LoonFFIResult rc;
  LoonProperties rp;
  uint64_t num_rows = 0;

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Try to get info for nonexistent file
  rc = loon_exttable_get_file_info("parquet", "/tmp/nonexistent-path-12345.parquet", &rp, &num_rows);

  ck_assert(!loon_ffi_is_success(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", loon_ffi_get_errmsg(&rc));

  // Clean up
  loon_ffi_free_result(&rc);
  loon_properties_free(&rp);
}

// will create two parquet files with file1_row_count rows and file2_row_count rows
static void create_two_parquet_test_files(const char* base_path,
                                          char file_path1[512],
                                          char file_path2[512],
                                          uint64_t file1_row_count,
                                          uint64_t file2_row_count) {
  LoonFFIResult rc;
  LoonProperties rp;
  char relative_path[512];
  char data_dir[512];

  memset(file_path1, 0, 512);
  memset(file_path2, 0, 512);

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  FileSystemHandle fs = get_fs(&rp);

  // Create first test parquet file
  snprintf(relative_path, sizeof(relative_path), "%s/cg-test", base_path);
  rc = create_testfile(relative_path, file1_row_count, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  snprintf(data_dir, sizeof(data_dir), "%s/_data", relative_path);
  ck_assert(find_first_file(fs, data_dir, ".parquet", file_path1, 512));

  // Create second test parquet file
  snprintf(relative_path, sizeof(relative_path), "%s/cg-test2", base_path);
  rc = create_testfile(relative_path, file2_row_count, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  snprintf(data_dir, sizeof(data_dir), "%s/_data", relative_path);
  ck_assert(find_first_file(fs, data_dir, ".parquet", file_path2, 512));

  printf("Test file 1: %s\n", file_path1);
  printf("Test file 2: %s\n", file_path2);

  loon_filesystem_destroy(fs);
  loon_properties_free(&rp);
}

static void test_column_groups_create(void) {
  LoonFFIResult rc;
  LoonColumnGroups* column_groups = NULL;
  char abs_base_dir[512];
  char file_path1[512];
  char file_path2[512];
  uint64_t file_start = 0;
  uint64_t file1_row_count = 100;
  uint64_t file2_row_count = 50;

  {
    LoonProperties _pp;
    create_test_external_pp(&_pp, "parquet");
    FileSystemHandle _fs = get_fs(&_pp);
    clean_test_dir(_fs, TEST_BASE_PATH);
    loon_filesystem_destroy(_fs);
    loon_properties_free(&_pp);
  }
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2, file1_row_count, file2_row_count);

  // Test 1: Basic test with single file, no start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = loon_column_groups_create((const char**)columns, 3, "parquet", paths, start_indices, end_indices, 1,
                                   &column_groups);

    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    // Clean up
    loon_column_groups_destroy(column_groups);
  }

  // Test 2: Multiple files without start/end indices
  // Should i verify the columns with schema?
  {
    char* columns[] = {"int64_field", "int32_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {file1_row_count, file2_row_count};

    rc = loon_column_groups_create((const char**)columns, 2, "parquet", paths, start_indices, end_indices, 2,
                                   &column_groups);

    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    // Clean up
    loon_column_groups_destroy(column_groups);
  }

  // Test 3: Multiple files with start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {50, 25};

    rc = loon_column_groups_create((const char**)columns, 3, "parquet", paths, start_indices, end_indices, 2,
                                   &column_groups);

    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    // Clean up
    loon_column_groups_destroy(column_groups);
  }

  // Test: Error case - NULL columns
  {
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = loon_column_groups_create(NULL, 1, "parquet", paths, start_indices, end_indices, 1, &column_groups);

    ck_assert(!loon_ffi_is_success(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL columns: %s\n", loon_ffi_get_errmsg(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test: Error case - NULL paths
  {
    char* columns[] = {"int64_field"};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = loon_column_groups_create((const char**)columns, 1, "parquet", NULL, start_indices, end_indices, 1,
                                   &column_groups);

    ck_assert(!loon_ffi_is_success(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL paths: %s\n", loon_ffi_get_errmsg(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test: Error case - NULL format
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc =
        loon_column_groups_create((const char**)columns, 1, NULL, paths, start_indices, end_indices, 1, &column_groups);

    ck_assert(!loon_ffi_is_success(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL format: %s\n", loon_ffi_get_errmsg(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test: Error case - zero columns
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = loon_column_groups_create((const char**)columns, 0, "parquet", paths, start_indices, end_indices, 1,
                                   &column_groups);

    ck_assert(!loon_ffi_is_success(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for zero columns: %s\n", loon_ffi_get_errmsg(&rc));
    loon_ffi_free_result(&rc);
  }
}

static void test_column_groups_create_then_read(void) {
  LoonFFIResult rc;
  LoonColumnGroups* column_groups = NULL;
  LoonReaderHandle reader = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArrayStream arraystream;
  LoonProperties rp;
  char abs_base_dir[512];
  char file_path1[512];
  char file_path2[512];
  uint64_t file_start = 0;
  uint64_t file1_row_count = 100;
  uint64_t file2_row_count = 50;

  memset(&arraystream, 0, sizeof(arraystream));

  {
    LoonProperties _pp;
    create_test_external_pp(&_pp, "parquet");
    FileSystemHandle _fs = get_fs(&_pp);
    clean_test_dir(_fs, TEST_BASE_PATH);
    loon_filesystem_destroy(_fs);
    loon_properties_free(&_pp);
  }
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2, file1_row_count, file2_row_count);

  // Create properties for reader
  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {file1_row_count, file2_row_count};

    size_t length_of_columns = sizeof(columns) / sizeof(columns[0]);
    size_t length_of_paths = sizeof(paths) / sizeof(paths[0]);
    ck_assert_int_eq(length_of_paths, sizeof(start_indices) / sizeof(start_indices[0]));
    ck_assert_int_eq(length_of_paths, sizeof(end_indices) / sizeof(end_indices[0]));

    rc = loon_column_groups_create((const char**)columns, length_of_columns, "parquet", paths, start_indices,
                                   end_indices, length_of_paths, &column_groups);

    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    // Create schema for reader
    schema = create_test_struct_schema();

    // Create reader with the column groups
    rc = loon_reader_new(column_groups, schema, NULL, 0, &rp, &reader);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(reader != 0);

    // Get record batch reader
    rc = loon_get_record_batch_reader(reader, NULL, &arraystream);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    // Verify we can read data
    {
      struct ArrowArray array_result;
      int arrow_rc;
      int64_t total_rows = 0;

      memset(&array_result, 0, sizeof(array_result));

      while (true) {
        arrow_rc = arraystream.get_next(&arraystream, &array_result);
        ck_assert_int_eq(0, arrow_rc);

        if (array_result.release == NULL) {
          // End of stream
          break;
        }

        ck_assert_int_eq(array_result.n_children, 3);  // 3 columns
        total_rows += array_result.length;

        // Release array
        if (array_result.release) {
          array_result.release(&array_result);
          array_result.release = NULL;
        }
      }

      ck_assert_int_eq(total_rows, file1_row_count + file2_row_count);
    }

    // Clean up
    if (arraystream.release) {
      arraystream.release(&arraystream);
    }
    loon_reader_destroy(reader);
    loon_column_groups_destroy(column_groups);
    if (schema && schema->release) {
      schema->release(schema);
    }
    free(schema);
  }

  loon_properties_free(&rp);
}

// Test that file paths returned by explore are valid and not duplicated
static void test_exttable_explore_file_paths_valid(void) {
  LoonFFIResult rc;
  LoonProperties rp;
  char data_path[512], base_dir[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  FileSystemHandle fs = get_fs(&rp);

  snprintf(base_dir, sizeof(base_dir), "%s-path-test-base", TEST_BASE_PATH);
  snprintf(data_path, sizeof(data_path), "%s-path-test-data", TEST_BASE_PATH);
  clean_test_dir(fs, base_dir);
  clean_test_dir(fs, data_path);

  // Create test files
  for (int i = 0; i < 3; i++) {
    rc = create_testfile(data_path, 10, &rp);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  }

  char* columns_cstrs[3] = {"int64_field", "int32_field", "string_field"};
  uint64_t num_of_files = 0;
  char* out_column_groups_file_path = NULL;
  char data_path_with_prefix[1024];
  snprintf(data_path_with_prefix, sizeof(data_path_with_prefix), "%s/_data/", data_path);

  rc = loon_exttable_explore((const char**)(columns_cstrs), 3, "parquet", base_dir, data_path_with_prefix, &rp,
                             &num_of_files, &out_column_groups_file_path);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(num_of_files, 3);

  LoonManifest* out_cmanifest = NULL;
  rc = loon_exttable_read_manifest(out_column_groups_file_path, &rp, &out_cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  LoonColumnGroup* ccg0 = &(out_cmanifest->column_groups.column_group_array[0]);
  ck_assert_int_eq(ccg0->num_of_files, 3);

  for (int i = 0; i < 3; i++) {
    const char* file_path = ccg0->files[i].path;
    ck_assert(file_path != NULL);

    // Verify path ends with .parquet (not .parquet/something.parquet)
    size_t path_len = strlen(file_path);
    ck_assert_msg(path_len > 8, "Path too short: %s", file_path);
    ck_assert_msg(strcmp(file_path + path_len - 8, ".parquet") == 0, "Path should end with .parquet: %s", file_path);

    // Verify no duplicate .parquet in path (would indicate the bug we fixed)
    const char* first_parquet = strstr(file_path, ".parquet");
    const char* second_parquet = first_parquet ? strstr(first_parquet + 1, ".parquet") : NULL;
    ck_assert_msg(second_parquet == NULL, "Path contains duplicate .parquet segment (bug!): %s", file_path);

    // File readability is verified by test_exttable_explore_and_read.
    // This test only validates path format.
    printf("Verified file path[%d]: %s\n", i, file_path);
  }

  loon_free_cstr(out_column_groups_file_path);
  loon_manifest_destroy(out_cmanifest);
  clean_test_dir(fs, base_dir);
  clean_test_dir(fs, data_path);
  loon_filesystem_destroy(fs);
  loon_properties_free(&rp);
}

void run_external_suite(void) {
  RUN_TEST(test_exttable_explore_and_read);
  RUN_TEST(test_exttable_explore_file_paths_valid);
  RUN_TEST(test_exttable_get_file_info_single_file_parquet);
  RUN_TEST(test_exttable_get_file_info_single_file_vortex);
  RUN_TEST(test_exttable_get_file_info_directory_error_parquet);
  RUN_TEST(test_exttable_get_file_info_directory_error_vortex);
  RUN_TEST(test_exttable_get_file_info_invalid_format);
  RUN_TEST(test_exttable_get_file_info_file_not_found);
  RUN_TEST(test_column_groups_create);
  RUN_TEST(test_column_groups_create_then_read);
}
