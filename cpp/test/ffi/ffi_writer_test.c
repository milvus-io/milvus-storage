#include "milvus-storage/ffi_c.h"
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <arrow/c/abi.h>
#include <time.h>

#define TEST_BASE_PATH "./writer_test_dir"

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

FFIResult create_test_writer_pp(Properties* rp) {
  FFIResult rc;
  size_t test_count;
  const char* test_key[] = {
      "writer.policy",
      "fs.storage_type",
      "fs.root_path",
  };

  const char* test_val[] = {
      "single",
      "local",
      "/tmp/",
  };

  test_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(test_count == sizeof(test_val) / sizeof(test_val[0]));

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, rp);
  return rc;
}

struct ArrowArray* create_test_struct_arrow_array(int64_t* int64_data,
                                                  int32_t* int32_data,
                                                  const char** str_data,
                                                  int length) {
  struct ArrowArray* children[] = {create_int64_array(int64_data, length, NULL, 0),
                                   create_int32_array(int32_data, length, NULL, 0),
                                   create_string_array(str_data, length, NULL, 0)};
  struct ArrowArray* struct_array = create_struct_array(children, 3, length);

  return struct_array;
}

START_TEST(test_basic) {
  WriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  FFIResult rc;
  Properties rp;
  int64_t length = 5;
  int64_t int64_data[] = {1, 2, 3, 4, 5};
  int32_t int32_data[] = {25, 30, 35, 40, 45};
  const char* str_data[] = {"ABC", "BCD", "DDDD", "EEEEEa", "CCCC23123"};

  // perpare the struct array and schema
  struct_array = create_test_struct_arrow_array(int64_data, int32_data, str_data, length);
  schema = create_test_struct_schema();

  // perpare the properties
  rc = create_test_writer_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // do the writer test
  rc = writer_new(TEST_BASE_PATH, schema, &rp, &writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = writer_write(writer_handle, struct_array);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = writer_flush(writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  char* out_manifest;
  size_t out_manifest_size;

  rc = writer_close(writer_handle, &out_manifest, &out_manifest_size);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_msg(out_manifest != NULL, "out_manifest should not be NULL");
  ck_assert_msg(out_manifest_size > 0, "out_manifest_size should be greater than 0");

  printf("out_manifest_size: %zu\n", out_manifest_size);
  printf("out_manifest: %s\n", out_manifest);

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);

  free_manifest(out_manifest);
  writer_destroy(writer_handle);

  // still need release the schema(struct)
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

int64_t* create_random_int64_array(size_t length) {
  assert(length > 0);
  int64_t* arr = (int64_t*)malloc(length * sizeof(int64_t));
  assert(arr != NULL);
  srand((unsigned int)time(NULL));

  for (size_t i = 0; i < length; i++) {
    int64_t high = (int64_t)rand() << 32;
    int64_t low = (int64_t)rand();
    arr[i] = high | low;
  }

  return arr;
}

int32_t* create_random_int32_array(size_t length) {
  assert(length > 0);
  int32_t* arr = (int32_t*)malloc(length * sizeof(int32_t));
  assert(arr != NULL);
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < length; i++) {
    arr[i] = (int32_t)rand();
  }
  return arr;
}

const char** create_random_str_array(size_t length, int str_max_len) {
  assert(length > 0);
  char** arr = (char**)malloc(length * sizeof(char*));
  assert(arr != NULL);
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < length; i++) {
    int str_len = 1 + rand() % str_max_len;  // [1, str_max_len]
    arr[i] = (char*)malloc((str_len + 1) * sizeof(char));
    assert(arr[i] != NULL);
    for (int j = 0; j < str_len; j++) {
      arr[i][j] = 32 + rand() % 95;
    }
    arr[i][str_len] = '\0';
  }

  return (const char**)arr;
}

void create_writer_test_file_with_pp(char* write_path,
                                     char** out_manifest,
                                     Properties* rp,
                                     size_t* out_manifest_size,
                                     int16_t loop_times,
                                     int64_t str_max_len,
                                     bool with_flush) {
  WriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  FFIResult rc;

  schema = create_test_struct_schema();

  // do the writer test
  rc = writer_new(write_path, schema, rp, &writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  for (int16_t len = 1; len < (loop_times + 1); len++) {
    int64_t* int64_data = create_random_int64_array(len);
    int32_t* int32_data = create_random_int32_array(len);
    const char** str_data = create_random_str_array(len, str_max_len);

    struct_array = create_test_struct_arrow_array(int64_data, int32_data, str_data, len);
    rc = writer_write(writer_handle, struct_array);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    if (with_flush) {
      rc = writer_flush(writer_handle);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    }

    free(int64_data);
    free(int32_data);
    for (int j = 0; j < len; j++) {
      free((void*)str_data[j]);
    }
    free(str_data);
    if (struct_array->release) {
      struct_array->release(struct_array);
    }
    free(struct_array);
  }

  rc = writer_flush(writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = writer_close(writer_handle, out_manifest, out_manifest_size);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  writer_destroy(writer_handle);

  // still need release the schema(struct)
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
}

// Also used on reader test
void create_writer_test_file(char* write_path,
                             char** out_manifest,
                             size_t* out_manifest_size,
                             int16_t loop_times,
                             int64_t str_max_len,
                             bool with_flush) {
  FFIResult rc;
  Properties rp;

  // perpare the properties
  rc = create_test_writer_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  create_writer_test_file_with_pp(write_path, out_manifest, &rp, out_manifest_size, loop_times, str_max_len,
                                  with_flush);

  properties_free(&rp);
}

void create_writer_size_based_test_file(char* write_path, char** out_manifest, size_t* out_manifest_size) {
  FFIResult rc;
  Properties rp;

  const char* test_key[] = {
      "writer.policy",
      "writer.split.size_based.max_avg_column_size",
      "writer.split.size_based.max_columns_in_group",
      "fs.storage_type",
      "fs.root_path",
  };

  const char* test_val[] = {
      "size_based", "10", "10", "local", "/tmp/",
  };

  // perpare the properties
  size_t test_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(test_count == sizeof(test_val) / sizeof(test_val[0]));

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  create_writer_test_file_with_pp(write_path, out_manifest, &rp, out_manifest_size, 10, 20, false);
  properties_free(&rp);
}

START_TEST(test_multi_write) {
  char* out_manifest;
  size_t out_manifest_size;

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10, 20, false);

  ck_assert_msg(out_manifest != NULL, "out_manifest should not be NULL");
  ck_assert_msg(out_manifest_size > 0, "out_manifest_size should be greater than 0");

  printf("out_manifest_size: %zu\n", out_manifest_size);
  printf("out_manifest: %s\n", out_manifest);

  free_manifest(out_manifest);
}
END_TEST

START_TEST(test_multi_no_close) {
  WriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  FFIResult rc;
  Properties rp;
  int64_t length = 5;
  int64_t int64_data[] = {1, 2, 3, 4, 5};
  int32_t int32_data[] = {25, 30, 35, 40, 45};
  const char* str_data[] = {"ABC", "BCD", "DDDD", "EEEEEa", "CCCC23123"};

  // perpare the struct array and schema
  struct_array = create_test_struct_arrow_array(int64_data, int32_data, str_data, length);
  schema = create_test_struct_schema();

  // perpare the properties
  rc = create_test_writer_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // do the writer test
  rc = writer_new(TEST_BASE_PATH, schema, &rp, &writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = writer_write(writer_handle, struct_array);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = writer_flush(writer_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);
  // will close the writer if caller have not call the `close`
  writer_destroy(writer_handle);

  // still need release the schema(struct)
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

START_TEST(test_multi_write_size_based) {
  char* out_manifest;
  size_t out_manifest_size;

  create_writer_size_based_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size);

  ck_assert_msg(out_manifest != NULL, "out_manifest should not be NULL");
  ck_assert_msg(out_manifest_size > 0, "out_manifest_size should be greater than 0");

  printf("out_manifest_size: %zu\n", out_manifest_size);
  printf("out_manifest: %s\n", out_manifest);

  free_manifest(out_manifest);
}
END_TEST

Suite* make_writer_suite(void) {
  Suite* writer_s;

  writer_s = suite_create("FFI writer interface");

  {
    TCase* writer_tc;
    writer_tc = tcase_create("writer");
    tcase_add_test(writer_tc, test_basic);
    tcase_add_test(writer_tc, test_multi_write);
    tcase_add_test(writer_tc, test_multi_no_close);
    tcase_add_test(writer_tc, test_multi_write_size_based);
    suite_add_tcase(writer_s, writer_tc);
  }

  return writer_s;
}