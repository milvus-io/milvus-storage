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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <inttypes.h>

#include <arrow/c/abi.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_exttable_c.h"

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

int remove_directory(const char* path);

// Helper function to create test properties
FFIResult create_test_external_pp(Properties* rp, const char* format) {
  const char* test_key[] = {
      "fs.address",
      "fs.root_path",
      "format",
  };

  const char* test_val[] = {
      "local",
      "/tmp/",
      format ? format : "parquet",
  };

  size_t test_count = sizeof(test_key) / sizeof(test_key[0]);
  return properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, rp);
}

// Helper function to create a simple parquet file using writer FFI
FFIResult create_testfile(const char* base_path, int64_t num_rows, Properties* props) {
  struct ArrowSchema* schema;
  WriterHandle writer;
  FFIResult rc;
  ColumnGroupsHandle column_groups = 0;

  schema = create_test_struct_schema();
  rc = writer_new(base_path, schema, props, &writer);
  if (!IsSuccess(&rc)) {
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

  rc = writer_write(writer, struct_array);

  // Clean up data arrays
  free(int64_data);
  free(int32_data);
  free(str_data);

  if (!IsSuccess(&rc)) {
    writer_destroy(writer);
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
  rc = writer_close(writer, NULL, NULL, 0, &column_groups);
  if (IsSuccess(&rc) && column_groups) {
    column_groups_ptr_destroy(column_groups);
  }
  writer_destroy(writer);
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
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  rc = create_test_external_pp(&rp, format);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path
  snprintf(full_path, sizeof(full_path), "/tmp/%s", TEST_BASE_PATH);

  // Create a test parquet file (creates directory with parquet file inside)
  rc = create_testfile(full_path, 100, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the actual parquet file created by the writer (has UUID in name)
  snprintf(cmd, sizeof(cmd), "find %s -name '*.%s' -type f | head -1", full_path, format);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path, sizeof(file_path), fp) != NULL);
  pclose(fp);

  // Remove trailing newline
  file_path[strcspn(file_path, "\n")] = 0;

  printf("Found %s file: %s\n", format, file_path);

  // Get file info for the specific file
  rc = exttable_get_file_info(format, file_path, &rp, &num_rows);

  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_int_eq(num_rows, 100);

  printf("num_rows=%" PRIu64 "\n", num_rows);

  // Clean up
  properties_free(&rp);
}

static void test_exttable_explore_and_read(void) {
  FFIResult rc;
  Properties rp;
  char data_path[512], base_dir[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path to a directory
  snprintf(base_dir, sizeof(base_dir), "/tmp/%s-base-dir", TEST_BASE_PATH);
  snprintf(data_path, sizeof(data_path), "/tmp/%s-data-dir", TEST_BASE_PATH);
  remove_directory(base_dir);
  remove_directory(data_path);

  // Create some test parquet file
  for (int i = 0; i < 10; i++) {
    rc = create_testfile(data_path, 50, &rp);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  }
  char* columns_cstrs[3] = {"int64_field", "int32_field", "string_field"};

  uint64_t num_of_files = 0;
  char* out_column_groups_file_path = NULL;
  char data_path_with_prefix[1024];
  snprintf(data_path_with_prefix, sizeof(data_path_with_prefix), "%s/_data/", data_path);

  rc = exttable_explore((const char**)(columns_cstrs), 3, "parquet", base_dir, data_path_with_prefix, &rp,
                        &num_of_files, &out_column_groups_file_path);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_int_eq(num_of_files, 10);

  CColumnGroups out_ccgs;
  rc = exttable_read_column_groups(out_column_groups_file_path, &rp, &out_ccgs);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  ck_assert(out_ccgs.column_group_array != NULL);
  ck_assert_int_eq(out_ccgs.num_of_column_groups, 1);
  ck_assert(out_ccgs.meta_keys == NULL);
  ck_assert(out_ccgs.meta_values == NULL);
  ck_assert_int_eq(out_ccgs.meta_len, 0);

  CColumnGroup* ccg0 = &(out_ccgs.column_group_array[0]);

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
    ck_assert(ccg0->files[i].private_data == NULL);
    ck_assert_int_eq(ccg0->files[i].private_data_size, 0);
  }

  out_ccgs.release(&out_ccgs);
  free_cstr(out_column_groups_file_path);
  properties_free(&rp);
}

static void test_exttable_get_file_info_single_file_parquet(void) {
  test_exttable_get_file_info_single_file("parquet");
}

static void test_exttable_get_file_info_single_file_vortex(void) {
#ifdef BUILD_VORTEX_BRIDGE
  test_exttable_get_file_info_single_file("vortex");
#endif
}

static void test_exttable_get_file_info_directory_error(const char* format) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  char full_path[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path to a directory
  snprintf(full_path, sizeof(full_path), "/tmp/%s-dir", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_testfile(full_path, 50, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get file info for directory (should fail - not a file)
  rc = exttable_get_file_info("parquet", full_path, &rp, &num_rows);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error for directory: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}

static void test_exttable_get_file_info_directory_error_parquet(void) {
  test_exttable_get_file_info_directory_error("parquet");
}

static void test_exttable_get_file_info_directory_error_vortex(void) {
#ifdef BUILD_VORTEX_BRIDGE
  test_exttable_get_file_info_directory_error("vortex");
#endif
}

static void test_exttable_get_file_info_invalid_format(void) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path
  snprintf(full_path, sizeof(full_path), "/tmp/%s-invalid", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_testfile(full_path, 100, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the actual parquet file
  snprintf(cmd, sizeof(cmd), "find %s -name '*.parquet' -type f | head -1", full_path);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path, sizeof(file_path), fp) != NULL);
  pclose(fp);
  file_path[strcspn(file_path, "\n")] = 0;

  // Try to get info with invalid format
  rc = exttable_get_file_info("invalid_format", file_path, &rp, &num_rows);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}

static void test_exttable_get_file_info_file_not_found(void) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get info for nonexistent file
  rc = exttable_get_file_info("parquet", "/tmp/nonexistent-path-12345.parquet", &rp, &num_rows);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}

// will create two parquet files with 100 rows and 50 rows
static void create_two_parquet_test_files(const char* base_path,
                                          char file_path1[512],
                                          char file_path2[512],
                                          uint64_t file1_row_count,
                                          uint64_t file2_row_count) {
  FFIResult rc;
  Properties rp;
  char full_path[512];
  char cmd[1024];
  FILE* fp;

  memset(file_path1, 0, 512);
  memset(file_path2, 0, 512);

  // Create test properties
  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path
  snprintf(full_path, sizeof(full_path), "%s/cg-test", base_path);

  // Create two test parquet files
  rc = create_testfile(full_path, file1_row_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the first parquet file
  snprintf(cmd, sizeof(cmd), "find %s -name '*.parquet' -type f | head -1", full_path);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path1, 512, fp) != NULL);
  pclose(fp);
  file_path1[strcspn(file_path1, "\n")] = 0;

  // Create a second test file in a different directory
  snprintf(full_path, sizeof(full_path), "%s/cg-test2", base_path);
  rc = create_testfile(full_path, file2_row_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the second parquet file
  snprintf(cmd, sizeof(cmd), "find %s -name '*.parquet' -type f | head -1", full_path);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path2, 512, fp) != NULL);
  pclose(fp);
  file_path2[strcspn(file_path2, "\n")] = 0;

  printf("Test file 1: %s\n", file_path1);
  printf("Test file 2: %s\n", file_path2);

  properties_free(&rp);
}

static void test_column_groups_create(void) {
  FFIResult rc;
  ColumnGroupsHandle column_groups = 0;
  char file_path1[512];
  char file_path2[512];
  uint64_t file_start = 0;
  uint64_t file1_row_count = 100;
  uint64_t file2_row_count = 50;

  remove_directory(TEST_BASE_PATH);
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2, file1_row_count, file2_row_count);

  // Test 1: Basic test with single file, no start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc =
        column_groups_create((const char**)columns, 3, "parquet", paths, start_indices, end_indices, 1, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_ptr_destroy(column_groups);
    column_groups = 0;
  }

  // Test 2: Multiple files without start/end indices
  // Should i verify the columns with schema?
  {
    char* columns[] = {"int64_field", "int32_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {file1_row_count, file2_row_count};

    rc =
        column_groups_create((const char**)columns, 2, "parquet", paths, start_indices, end_indices, 2, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_ptr_destroy(column_groups);
    column_groups = 0;
  }

  // Test 3: Multiple files with start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {50, 25};

    rc =
        column_groups_create((const char**)columns, 3, "parquet", paths, start_indices, end_indices, 2, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_ptr_destroy(column_groups);
    column_groups = 0;
  }

  // Test: Error case - NULL columns
  {
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = column_groups_create(NULL, 1, "parquet", paths, start_indices, end_indices, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL columns: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test: Error case - NULL paths
  {
    char* columns[] = {"int64_field"};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = column_groups_create((const char**)columns, 1, "parquet", NULL, start_indices, end_indices, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL paths: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test: Error case - NULL format
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc = column_groups_create((const char**)columns, 1, NULL, paths, start_indices, end_indices, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL format: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test: Error case - zero columns
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};
    int64_t start_indices[] = {0};
    int64_t end_indices[] = {file1_row_count};

    rc =
        column_groups_create((const char**)columns, 0, "parquet", paths, start_indices, end_indices, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for zero columns: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }
}

static void test_column_groups_create_then_read(void) {
  FFIResult rc;
  ColumnGroupsHandle column_groups = 0;
  ReaderHandle reader = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArrayStream arraystream;
  Properties rp;
  char file_path1[512];
  char file_path2[512];
  uint64_t file_start = 0;
  uint64_t file1_row_count = 100;
  uint64_t file2_row_count = 50;

  memset(&arraystream, 0, sizeof(arraystream));

  remove_directory(TEST_BASE_PATH);
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2, file1_row_count, file2_row_count);

  // Create properties for reader
  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {file1_row_count, file2_row_count};

    size_t length_of_columns = sizeof(columns) / sizeof(columns[0]);
    size_t length_of_paths = sizeof(paths) / sizeof(paths[0]);
    ck_assert_int_eq(length_of_paths, sizeof(start_indices) / sizeof(start_indices[0]));
    ck_assert_int_eq(length_of_paths, sizeof(end_indices) / sizeof(end_indices[0]));

    rc = column_groups_create((const char**)columns, length_of_columns, "parquet", paths, start_indices, end_indices,
                              length_of_paths, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Create schema for reader
    schema = create_test_struct_schema();

    // Create reader with the column groups
    rc = reader_new(column_groups, schema, NULL, 0, &rp, &reader);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(reader != 0);

    // Get record batch reader
    rc = get_record_batch_reader(reader, NULL, &arraystream);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

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
    reader_destroy(reader);
    column_groups_ptr_destroy(column_groups);
    if (schema && schema->release) {
      schema->release(schema);
    }
    free(schema);
  }

  properties_free(&rp);
}

void run_external_suite(void) {
  RUN_TEST(test_exttable_explore_and_read);
  RUN_TEST(test_exttable_get_file_info_single_file_parquet);
  RUN_TEST(test_exttable_get_file_info_single_file_vortex);
  RUN_TEST(test_exttable_get_file_info_directory_error_parquet);
  RUN_TEST(test_exttable_get_file_info_directory_error_vortex);
  RUN_TEST(test_exttable_get_file_info_invalid_format);
  RUN_TEST(test_exttable_get_file_info_file_not_found);
  RUN_TEST(test_column_groups_create);
  RUN_TEST(test_column_groups_create_then_read);
}
