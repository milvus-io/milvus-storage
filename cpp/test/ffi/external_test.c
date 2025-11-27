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
#include "milvus-storage/ffi_exttable_c.h"

#include <check.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <arrow/c/abi.h>

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
      "/",
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
  ColumnGroupsHandle column_groups;

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
    return rc;
  }

  // Close writer
  rc = writer_close(writer, NULL, NULL, 0, &column_groups);
  if (IsSuccess(&rc) && column_groups) {
    column_groups_destroy(column_groups);
  }
  writer_destroy(writer);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);

  return rc;
}

static void test_exttable_get_file_info_single_file(const char* format) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

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
  rc = exttable_get_file_info(format, file_path, &rp, &num_rows, &out_schema);

  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_int_eq(num_rows, 100);

  printf("num_rows=%llu\n", num_rows);

  // Clean up
  if (out_schema.release) {
    out_schema.release(&out_schema);
  }
  properties_free(&rp);
}

START_TEST(test_exttable_get_file_info_single_file_parquet) { test_exttable_get_file_info_single_file("parquet"); }
END_TEST

#ifdef BUILD_VORTEX_BRIDGE
START_TEST(test_exttable_get_file_info_single_file_vortex) { test_exttable_get_file_info_single_file("vortex"); }
END_TEST
#endif  // BUILD_VORTEX_BRIDGE

START_TEST(test_exttable_get_file_info_directory_error) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path to a directory
  snprintf(full_path, sizeof(full_path), "/tmp/%s-dir", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_testfile(full_path, 50, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get file info for directory (should fail - not a file)
  rc = exttable_get_file_info("parquet", full_path, &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error for directory: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

START_TEST(test_exttable_get_file_info_invalid_format) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

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
  rc = exttable_get_file_info("invalid_format", file_path, &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

START_TEST(test_exttable_get_file_info_file_not_found) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get info for nonexistent file
  rc = exttable_get_file_info("parquet", "/tmp/nonexistent-path-12345.parquet", &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

// will create two parquet files with 100 rows and 50 rows
static void create_two_parquet_test_files(const char* base_path, char file_path1[512], char file_path2[512]) {
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
  rc = create_testfile(full_path, 100, &rp);
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
  rc = create_testfile(full_path, 50, &rp);
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
}

START_TEST(test_exttable_generate_column_groups) {
  FFIResult rc;
  ColumnGroupsHandle column_groups = 0;
  char file_path1[512];
  char file_path2[512];

  remove_directory(TEST_BASE_PATH);
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2);

  // Test 1: Basic test with single file, no start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1};

    rc = exttable_generate_column_groups(columns, 3, "parquet", paths, NULL, NULL, 1, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_destroy(column_groups);
    column_groups = 0;
  }

  // Test 2: Multiple files without start/end indices
  // Should i verify the columns with schema?
  {
    char* columns[] = {"int64_field", "int32_field"};
    char* paths[] = {file_path1, file_path2};

    rc = exttable_generate_column_groups(columns, 2, "parquet", paths, NULL, NULL, 2, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_destroy(column_groups);
    column_groups = 0;
  }

  // Test 3: Multiple files with start/end indices
  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {50, 25};

    rc = exttable_generate_column_groups(columns, 3, "parquet", paths, start_indices, end_indices, 2, &column_groups);

    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(column_groups != 0);

    // Clean up
    column_groups_destroy(column_groups);
    column_groups = 0;
  }

  // Test 4: Error case - NULL columns
  {
    char* paths[] = {file_path1};

    rc = exttable_generate_column_groups(NULL, 1, "parquet", paths, NULL, NULL, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL columns: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test 5: Error case - NULL paths
  {
    char* columns[] = {"int64_field"};

    rc = exttable_generate_column_groups(columns, 1, "parquet", NULL, NULL, NULL, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL paths: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test 6: Error case - NULL format
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};

    rc = exttable_generate_column_groups(columns, 1, NULL, paths, NULL, NULL, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for NULL format: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }

  // Test 7: Error case - zero columns
  {
    char* columns[] = {"int64_field"};
    char* paths[] = {file_path1};

    rc = exttable_generate_column_groups(columns, 0, "parquet", paths, NULL, NULL, 1, &column_groups);

    ck_assert(!IsSuccess(&rc));
    ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
    printf("Expected error for zero columns: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);
  }
}
END_TEST

START_TEST(test_exttable_generate_column_groups_then_read) {
  FFIResult rc;
  ColumnGroupsHandle column_groups = 0;
  ReaderHandle reader = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArrayStream arraystream;
  Properties rp;
  char file_path1[512];
  char file_path2[512];

  memset(&arraystream, 0, sizeof(arraystream));

  remove_directory(TEST_BASE_PATH);
  create_two_parquet_test_files(TEST_BASE_PATH, file_path1, file_path2);

  // Create properties for reader
  rc = create_test_external_pp(&rp, "parquet");
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  {
    char* columns[] = {"int64_field", "int32_field", "string_field"};
    char* paths[] = {file_path1, file_path2};
    int64_t start_indices[] = {0, 0};
    int64_t end_indices[] = {100, 100};

    size_t length_of_columns = sizeof(columns) / sizeof(columns[0]);
    size_t length_of_paths = sizeof(paths) / sizeof(paths[0]);
    assert(length_of_paths == sizeof(start_indices) / sizeof(start_indices[0]));
    assert(length_of_paths == sizeof(end_indices) / sizeof(end_indices[0]));

    rc = exttable_generate_column_groups(columns, length_of_columns, "parquet", paths, start_indices, end_indices,
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

      ck_assert_int_eq(total_rows, 150);
    }

    // Clean up
    if (arraystream.release) {
      arraystream.release(&arraystream);
    }
    reader_destroy(reader);
    column_groups_destroy(column_groups);
    if (schema && schema->release) {
      schema->release(schema);
    }
    free(schema);
  }

  properties_free(&rp);
}
END_TEST

Suite* make_external_suite(void) {
  Suite* external_s;

  external_s = suite_create("FFI external interface");

  {
    TCase* external_tc;
    external_tc = tcase_create("External");
    tcase_add_test(external_tc, test_exttable_get_file_info_single_file_parquet);
#ifdef BUILD_VORTEX_BRIDGE
    tcase_add_test(external_tc, test_exttable_get_file_info_single_file_vortex);
#endif
    tcase_add_test(external_tc, test_exttable_get_file_info_directory_error);
    tcase_add_test(external_tc, test_exttable_get_file_info_invalid_format);
    tcase_add_test(external_tc, test_exttable_get_file_info_file_not_found);
    tcase_add_test(external_tc, test_exttable_generate_column_groups);
    tcase_add_test(external_tc, test_exttable_generate_column_groups_then_read);

    suite_add_tcase(external_s, external_tc);
  }

  return external_s;
}
