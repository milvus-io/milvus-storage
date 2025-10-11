#include "milvus-storage/ffi_c.h"
#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <arrow/c/abi.h>

#define TEST_BASE_PATH "reader-test-dir"

// will writer 10 recordbacth
// each of recordbacth rows [1...10]
void create_writer_test_file(char* write_path,
                             char** out_manifest,
                             size_t* out_manifest_size,
                             int16_t loop_times,
                             int64_t str_max_len,
                             bool with_flush);

void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable);
struct ArrowSchema* create_test_struct_schema();

FFIResult create_test_reader_pp(Properties* rp) {
  FFIResult rc;
  size_t test_count;

#if 0
  // minio config
  const char* test_key[] = {
      "fs.storage_type",
      "fs.access_key_id",
      "fs.access_key_value",
      "fs.bucket_name",
      "fs.use_ssl",
      "fs.address",
      "fs.region"
  };

  const char* test_val[] = {
      "remote",
      "minioadmin",
      "minioadmin",
      "testbucket",
      "false",
      "localhost:9000",
      "us-west-2"
  };
#else
  // local config
  const char* test_key[] = {
      "fs.storage_type",
      "fs.root_path",
  };

  const char* test_val[] = {
      "local",
      "/tmp/",
  };
#endif

  test_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(test_count == sizeof(test_val) / sizeof(test_val[0]));

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, rp);
  return rc;
}

START_TEST(test_basic) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10 /*loop_times*/, 20 /*str_max_len*/,
                          false /*with_flush*/);
  printf("out_manifest: %s\n", out_manifest);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  {
    rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, 1024 /*batch_size*/,
                                 8 * 1024 * 1024 /*buffer_size*/, &arraystream);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    arraystream.release(&arraystream);
  }

  // test create chunkreader
  {
    ChunkReaderHandle chunk_reader_handle;
    rc = get_chunk_reader(reader_handle, 0, &chunk_reader_handle);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    chunk_reader_destroy(chunk_reader_handle);
  }

  // test take interface
  {
    struct ArrowArray arrow_array;
    int64_t rowidx = 0;
    rc = take(reader_handle, &rowidx, 1, 0 /* parallelism */, &arrow_array);

    // NOT IMPLEMENT YET
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);
  }

  free_manifest(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

char* replace_substring(const char* original, const char* old_str, const char* new_str) {
  char* result;
  int i, count = 0;
  int new_len = strlen(new_str);
  int old_len = strlen(old_str);

  const char* tmp = original;
  while ((tmp = strstr(tmp, old_str)) != NULL) {
    count++;
    tmp += old_len;
  }

  result = (char*)malloc(strlen(original) + (new_len - old_len) * count + 1);
  assert(result != NULL);
  i = 0;
  while (*original) {
    if (strstr(original, old_str) == original) {
      strcpy(&result[i], new_str);
      i += new_len;
      original += old_len;
    } else {
      result[i++] = *original++;
    }
  }

  result[i] = '\0';
  return result;
}

START_TEST(test_reader_with_invalid_manifest) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10 /*loop_times*/, 20 /*str_max_len*/,
                          false /*with_flush*/);
  printf("out_manifest: %s\n", out_manifest);

  const char* str_in_manifest_need_replace[] = {
      TEST_BASE_PATH,
      "column_group_",
  };

  const char* str_replace_to[] = {"non-exist-path", "fake_column_group_"};

  int size_to_replace = sizeof(str_in_manifest_need_replace) / sizeof(str_in_manifest_need_replace[0]);
  assert(size_to_replace == sizeof(str_replace_to) / sizeof(str_replace_to[0]));

  // invalid manifest, paths is wrong
  for (int i = 0; i < size_to_replace; i++) {
    schema = create_test_struct_schema();
    char* new_manifest = replace_substring(out_manifest, str_in_manifest_need_replace[i], str_replace_to[i]);

    rc = create_test_reader_pp(&rp);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    rc = reader_new(new_manifest, schema, needed_columns, 3, &rp, &reader_handle);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    // test create arrowarraysteam
    rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, 1024 /*batch_size*/,
                                 8 * 1024 * 1024 /*buffer_size*/, &arraystream);
    ck_assert(!IsSuccess(&rc));
    printf("Expected error: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);

    free_manifest(new_manifest);
    reader_destroy(reader_handle);
    if (schema->release) {
      schema->release(schema);
    }
    free(schema);
    properties_free(&rp);
  }

  free_manifest(out_manifest);
}
END_TEST

START_TEST(test_record_batch_reader_verify_schema) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* writer_schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10 /*loop_times*/, 20 /*str_max_len*/,
                          false /*with_flush*/);

  writer_schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, writer_schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, 1024 /*batch_size*/, 8 * 1024 * 1024 /*buffer_size*/,
                               &arraystream);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  ck_assert(arraystream.get_schema != NULL);
  ck_assert(arraystream.get_next != NULL);
  ck_assert(arraystream.get_last_error != NULL);
  ck_assert(arraystream.release != NULL);

  // recreate a schema to verify the result
  // old one have been released in reader_new
  free(writer_schema);
  writer_schema = create_test_struct_schema();

  // verify schema
  struct ArrowSchema schema_result;
  int arrow_rc;
  arrow_rc = arraystream.get_schema(&arraystream, &schema_result);
  ck_assert_int_eq(0, arrow_rc);
  ck_assert(schema_result.release != NULL);
  ck_assert_str_eq(writer_schema->format, schema_result.format);
  ck_assert_int_eq(writer_schema->n_children, schema_result.n_children);
  ck_assert(schema_result.children != NULL);
  ck_assert(schema_result.dictionary == NULL);

  for (int64_t i = 0; i < schema_result.n_children; i++) {
    ck_assert(schema_result.children[i] != NULL && writer_schema->children[i] != NULL);
    ck_assert(schema_result.children[i]->format != NULL);
    ck_assert(schema_result.children[i]->name != NULL);

    ck_assert_str_eq(writer_schema->children[i]->format, schema_result.children[i]->format);
    ck_assert_str_eq(writer_schema->children[i]->name, schema_result.children[i]->name);
  }

  schema_result.release(&schema_result);

  free_manifest(out_manifest);
  reader_destroy(reader_handle);

  // recreated one need call the `release`
  writer_schema->release(writer_schema);
  free(writer_schema);

  properties_free(&rp);
}
END_TEST

void verify_arrow_array(struct ArrowArray* arrowarray) {
  ck_assert_int_eq(55, arrowarray->length);
  ck_assert_int_eq(3, arrowarray->n_children);
  ck_assert_int_eq(1, arrowarray->n_buffers);
  ck_assert(arrowarray->children != NULL);
  ck_assert(arrowarray->dictionary == NULL);
  ck_assert(arrowarray->release != NULL);
  ck_assert(arrowarray->children[0] != NULL);
  ck_assert(arrowarray->children[1] != NULL);
  ck_assert(arrowarray->children[2] != NULL);

  // verify int64 child
  {
    struct ArrowArray* int64_array = arrowarray->children[0];
    ck_assert_int_eq(55, int64_array->length);
    ck_assert_int_eq(0, int64_array->null_count);
    ck_assert_int_eq(2, int64_array->n_buffers);
    ck_assert_int_eq(0, int64_array->n_children);
    ck_assert(int64_array->buffers != NULL);
    ck_assert(int64_array->buffers[0] == NULL);
    ck_assert(int64_array->buffers[1] != NULL);
  }

  // verify int32 child
  {
    struct ArrowArray* int32_array = arrowarray->children[1];
    ck_assert_int_eq(55, int32_array->length);
    ck_assert_int_eq(0, int32_array->null_count);
    ck_assert_int_eq(2, int32_array->n_buffers);
    ck_assert_int_eq(0, int32_array->n_children);
    ck_assert(int32_array->buffers != NULL);

    ck_assert(int32_array->buffers[1] != NULL);
  }

  // veryify string child
  {
    struct ArrowArray* str_array = arrowarray->children[2];
    ck_assert_int_eq(55, str_array->length);
    ck_assert_int_eq(0, str_array->null_count);
    ck_assert_int_eq(3, str_array->n_buffers);
    ck_assert_int_eq(0, str_array->n_children);
    ck_assert(str_array->buffers != NULL);
    ck_assert(str_array->buffers[0] == NULL);
    ck_assert(str_array->buffers[1] != NULL);
    ck_assert(str_array->buffers[2] != NULL);
  }
}
END_TEST

START_TEST(test_record_batch_reader_verify_arrowarray) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10 /*loop_times*/, 20 /*str_max_len*/,
                          false /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, 1024 /*batch_size*/, 8 * 1024 * 1024 /*buffer_size*/,
                               &arraystream);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  ck_assert(arraystream.get_schema != NULL);
  ck_assert(arraystream.get_next != NULL);
  ck_assert(arraystream.get_last_error != NULL);
  ck_assert(arraystream.release != NULL);

  // verify arrowarray
  struct ArrowArray arrowarray;
  int arrow_rc = arraystream.get_next(&arraystream, &arrowarray);
  ck_assert_int_eq(0, arrow_rc);
  verify_arrow_array(&arrowarray);
  arrowarray.release(&arrowarray);

  // reach end of stream
  arrow_rc = arraystream.get_next(&arraystream, &arrowarray);
  ck_assert_int_eq(0, arrow_rc);
  ck_assert(arrowarray.release == NULL);

  free_manifest(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

START_TEST(test_chunk_reader) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 10 /*loop_times*/, 20 /*str_max_len*/,
                          false /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create chunkreader

  ChunkReaderHandle chunk_reader_handle;
  rc = get_chunk_reader(reader_handle, 0, &chunk_reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  int64_t rowidx = 10;
  int64_t* chunk_indices = NULL;
  size_t num_chunk_indices = 0;
  struct ArrowArray arrowarray;

  // test get_chunk_indices
  {
    rc = get_chunk_indices(chunk_reader_handle, &rowidx, 1, &chunk_indices, &num_chunk_indices);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(num_chunk_indices == 1);
    ck_assert(chunk_indices != NULL);
  }
  // test get_chunk
  {
    rc = get_chunk(chunk_reader_handle, chunk_indices[0], &arrowarray);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    verify_arrow_array(&arrowarray);
    arrowarray.release(&arrowarray);
  }

  free_chunk_indices(chunk_indices);
  chunk_reader_destroy(chunk_reader_handle);

  free_manifest(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

START_TEST(test_chunk_reader_get_chunks) {
  char* out_manifest;
  size_t out_manifest_size;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, &out_manifest_size, 100 /*loop_times*/, 2000 /*str_max_len*/,
                          true /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create chunkreader
  ChunkReaderHandle chunk_reader_handle;
  rc = get_chunk_reader(reader_handle, 0, &chunk_reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  int64_t rowidx[] = {1, 11, 21, 5000, 5049};
  int64_t* chunk_indices = NULL;
  size_t num_chunk_indices = 0;
  struct ArrowArray arrowarray;

  // chunk index should be 3
  {
    rc = get_chunk_indices(chunk_reader_handle, rowidx, 4, &chunk_indices, &num_chunk_indices);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    printf("num_chunk_indices: %zu\n", num_chunk_indices);
    ck_assert(num_chunk_indices == 2);
    ck_assert(chunk_indices != NULL);
  }

  // test get_chunks
  {
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    rc = get_chunks(chunk_reader_handle, chunk_indices, num_chunk_indices, 0 /*parallelism*/, &arrays, &num_arrays);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert(num_arrays == 2);
    ck_assert(arrays != NULL);

    for (size_t i = 0; i < num_arrays; i++) {
      ck_assert_int_gt(arrays[i].length, 0);
      arrays[i].release(&arrays[i]);
    }

    free_chunk_indices(chunk_indices);
    free(arrays);
  }

  chunk_reader_destroy(chunk_reader_handle);

  free_manifest(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}
END_TEST

Suite* make_reader_suite(void) {
  Suite* reader_s;

  reader_s = suite_create("FFI reader interface");
  {
    TCase* reader_tc;
    reader_tc = tcase_create("reader");
    tcase_add_test(reader_tc, test_basic);
    tcase_add_test(reader_tc, test_reader_with_invalid_manifest);
    tcase_add_test(reader_tc, test_record_batch_reader_verify_schema);
    tcase_add_test(reader_tc, test_record_batch_reader_verify_arrowarray);
    tcase_add_test(reader_tc, test_chunk_reader);
    tcase_add_test(reader_tc, test_chunk_reader_get_chunks);

    suite_add_tcase(reader_s, reader_tc);
  }

  return reader_s;
}