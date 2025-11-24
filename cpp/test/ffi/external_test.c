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
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <arrow/c/abi.h>
#include <unistd.h>

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

// Helper function to create test properties
FFIResult create_test_external_pp(Properties* rp) {
  const char* test_key[] = {
      "fs.address",
      "fs.root_path",
  };

  const char* test_val[] = {
      "/",
      "/tmp/",
  };

  size_t test_count = sizeof(test_key) / sizeof(test_key[0]);
  return properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, rp);
}

// Helper function to create a simple parquet file using writer FFI
FFIResult create_test_parquet_file(const char* base_path, int64_t num_rows, Properties* props) {
  struct ArrowSchema* schema;
  WriterHandle writer;
  FFIResult rc;
  char* column_groups = NULL;

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
    free_cstr(column_groups);
  }
  writer_destroy(writer);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);

  return rc;
}

START_TEST(test_external_get_file_info_single_file) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path
  snprintf(full_path, sizeof(full_path), "/tmp/%s", TEST_BASE_PATH);

  // Create a test parquet file (creates directory with parquet file inside)
  rc = create_test_parquet_file(full_path, 100, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the actual parquet file created by the writer (has UUID in name)
  snprintf(cmd, sizeof(cmd), "find %s -name '*.parquet' -type f | head -1", full_path);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path, sizeof(file_path), fp) != NULL);
  pclose(fp);

  // Remove trailing newline
  file_path[strcspn(file_path, "\n")] = 0;

  printf("Found parquet file: %s\n", file_path);

  // Get file info for the specific file
  rc = external_get_file_info("parquet", file_path, &rp, &num_rows, &out_schema);

  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_int_eq(num_rows, 100);

  printf("test_external_get_file_info_single_file: num_rows=%lu\n", num_rows);

  // Clean up
  if (out_schema.release) {
    out_schema.release(&out_schema);
  }
  properties_free(&rp);
}
END_TEST

START_TEST(test_external_get_file_info_directory_error) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path to a directory
  snprintf(full_path, sizeof(full_path), "/tmp/%s-dir", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_test_parquet_file(full_path, 50, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get file info for directory (should fail - not a file)
  rc = external_get_file_info("parquet", full_path, &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error for directory: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

START_TEST(test_external_get_file_info_invalid_format) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;
  char full_path[512];
  char cmd[1024];
  FILE* fp;
  char file_path[512];

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Create absolute path
  snprintf(full_path, sizeof(full_path), "/tmp/%s-invalid", TEST_BASE_PATH);

  // Create a test parquet file
  rc = create_test_parquet_file(full_path, 100, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Find the actual parquet file
  snprintf(cmd, sizeof(cmd), "find %s -name '*.parquet' -type f | head -1", full_path);
  fp = popen(cmd, "r");
  ck_assert(fp != NULL);
  ck_assert(fgets(file_path, sizeof(file_path), fp) != NULL);
  pclose(fp);
  file_path[strcspn(file_path, "\n")] = 0;

  // Try to get info with invalid format
  rc = external_get_file_info("invalid_format", file_path, &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

START_TEST(test_external_get_file_info_file_not_found) {
  FFIResult rc;
  Properties rp;
  uint64_t num_rows = 0;
  struct ArrowSchema out_schema;

  memset(&out_schema, 0, sizeof(out_schema));

  rc = create_test_external_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Try to get info for nonexistent file
  rc = external_get_file_info("parquet", "/tmp/nonexistent-path-12345.parquet", &rp, &num_rows, &out_schema);

  ck_assert(!IsSuccess(&rc));
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  ck_assert(rc.message != NULL);
  printf("Expected error: %s\n", GetErrorMessage(&rc));

  // Clean up
  FreeFFIResult(&rc);
  properties_free(&rp);
}
END_TEST

Suite* make_external_suite(void) {
  Suite* external_s;

  external_s = suite_create("FFI external interface");

  {
    TCase* external_tc;
    external_tc = tcase_create("External");
    tcase_add_test(external_tc, test_external_get_file_info_single_file);
    tcase_add_test(external_tc, test_external_get_file_info_directory_error);
    tcase_add_test(external_tc, test_external_get_file_info_invalid_format);
    tcase_add_test(external_tc, test_external_get_file_info_file_not_found);

    suite_add_tcase(external_s, external_tc);
  }

  return external_s;
}
