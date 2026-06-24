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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>

#include <arrow/c/abi.h>

#include "milvus-storage/ffi_c.h"

#define TEST_BASE_PATH "reader-test-dir"

// will writer 10 recordbacth
// each of recordbacth rows [1...10]
void create_writer_test_file2(char* write_path,
                              char** meta_keys,
                              char** meta_values,
                              uint16_t meta_len,
                              LoonColumnGroups** out_manifest,
                              int16_t loop_times,
                              int64_t str_max_len,
                              bool with_flush);

void create_writer_test_file(
    char* write_path, LoonColumnGroups** out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);

void create_writer_test_file_with_pp(char* write_path,
                                     char** meta_keys,
                                     char** meta_values,
                                     uint16_t meta_len,
                                     LoonColumnGroups** out_manifest,
                                     LoonProperties* rp,
                                     int16_t loop_times,
                                     int64_t str_max_len,
                                     bool with_flush);
void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable, int field_offset);
struct ArrowSchema* create_test_struct_schema();
LoonFFIResult create_test_writer_pp(LoonProperties* rp);

// Array creation functions from ffi_writer_test.c
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

LoonFFIResult create_test_reader_pp(LoonProperties* rp) {
  const char* keys[500];
  const char* vals[500];
  size_t count = init_test_props(keys, vals, 0, 500, FFI_TEST_ROOT_PATH);

  LoonFFIResult rc = loon_properties_create((const char* const*)keys, (const char* const*)vals, count, rp);
  return rc;
}

static void test_basic(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_cgs, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test create arrowarraysteam
  {
    rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    arraystream.release(&arraystream);
  }

  // test create chunkreader
  {
    LoonChunkReaderHandle chunk_reader_handle;
    rc = loon_get_chunk_reader(reader_handle, 0, NULL, 0, &chunk_reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    loon_chunk_reader_destroy(chunk_reader_handle);
  }

  // test take interface
  {
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    uint64_t rowidx[] = {0, 3, 5};
    rc = loon_take(reader_handle, (const int64_t*)rowidx, 3, 1 /* parallelism */, NULL, 0, &arrays, &num_arrays, NULL);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    ck_assert(arrays != NULL);
    size_t number_of_rows = 0;
    for (int i = 0; i < num_arrays; i++) {
      number_of_rows += arrays[i].length;
    }
    ck_assert_int_eq(number_of_rows, 3);
    loon_free_chunk_arrays(arrays, num_arrays);
  }

  loon_column_groups_destroy(out_cgs);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_empty_projection(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // full projection with needed_columns all null
  rc = loon_reader_new(out_cgs, schema, NULL, 0, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  // verify arraystream number of columns and number of rows
  {
    struct ArrowSchema schema_result;
    struct ArrowArray array_result;
    int arrow_rc;

    arrow_rc = arraystream.get_schema(&arraystream, &schema_result);
    ck_assert_int_eq(0, arrow_rc);
    ck_assert(schema_result.release != NULL);
    ck_assert_int_eq(schema_result.n_children, 3);  // all columns

    // release schema_result
    if (schema_result.release) {
      schema_result.release(&schema_result);
      schema_result.release = NULL;
    }

    arrow_rc = arraystream.get_next(&arraystream, &array_result);
    ck_assert_int_eq(0, arrow_rc);
    ck_assert(array_result.release != NULL);
    ck_assert_int_eq(array_result.n_children, 3);  // all columns
    ck_assert_int_gt(array_result.length, 0);      // total 10 rows

    // release array_result
    if (array_result.release) {
      array_result.release(&array_result);
      array_result.release = NULL;
    }
  }

  if (arraystream.release) {
    arraystream.release(&arraystream);
    arraystream.release = NULL;
  }

  loon_column_groups_destroy(out_cgs);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

// Removed replace_substring and test_reader_with_invalid_manifest as they are incompatible with opaque
// ColumnGroupsHandle

static void test_record_batch_reader_verify_schema(void) {
  LoonColumnGroups* out_manifest = NULL;
  struct ArrowSchema* writer_schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

  writer_schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_manifest, writer_schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test create arrowarraysteam
  rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

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
  arraystream.release(&arraystream);

  loon_column_groups_destroy(out_manifest);
  loon_reader_destroy(reader_handle);

  // recreated one need call the `release`
  writer_schema->release(writer_schema);
  free(writer_schema);

  loon_properties_free(&rp);
}

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

static void test_record_batch_reader_verify_arrowarray(void) {
  LoonColumnGroups* out_manifest = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test create arrowarraysteam
  rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

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

  arraystream.release(&arraystream);

  loon_column_groups_destroy(out_manifest);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_chunk_reader(void) {
  LoonColumnGroups* out_manifest = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test create chunkreader

  LoonChunkReaderHandle chunk_reader_handle;
  rc = loon_get_chunk_reader(reader_handle, 0, NULL, 0, &chunk_reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t rowidx = 10;
  int64_t* chunk_indices = NULL;
  size_t num_chunk_indices = 0;
  struct ArrowArray arrowarray;

  // test get_chunk_indices
  {
    rc = loon_get_chunk_indices(chunk_reader_handle, &rowidx, 1, &chunk_indices, &num_chunk_indices);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(num_chunk_indices == 1);
    ck_assert(chunk_indices != NULL);
  }
  // test get_chunk
  {
    rc = loon_get_chunk(chunk_reader_handle, chunk_indices[0], &arrowarray, NULL);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    verify_arrow_array(&arrowarray);
    arrowarray.release(&arrowarray);
  }

  loon_free_chunk_indices(chunk_indices);
  loon_chunk_reader_destroy(chunk_reader_handle);

  loon_column_groups_destroy(out_manifest);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_chunk_reader_get_chunks(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 100 /*loop_times*/, 2000 /*str_max_len*/, true /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_cgs, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test create chunkreader
  LoonChunkReaderHandle chunk_reader_handle;
  rc = loon_get_chunk_reader(reader_handle, 0, NULL, 0, &chunk_reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t rowidx[] = {1, 11, 21, 5000, 5049};
  int64_t* chunk_indices = NULL;
  size_t num_chunk_indices = 0;
  struct ArrowArray arrowarray;

  // chunk index should be 3
  {
    rc = loon_get_chunk_indices(chunk_reader_handle, rowidx, 4, &chunk_indices, &num_chunk_indices);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    printf("num_chunk_indices: %zu\n", num_chunk_indices);
    ck_assert(num_chunk_indices == 2);
    ck_assert(chunk_indices != NULL);
  }

  // test get_chunks
  {
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    rc = loon_get_chunks(chunk_reader_handle, chunk_indices, num_chunk_indices, 1 /* parallelism */, &arrays,
                         &num_arrays, NULL);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(num_arrays == 2);
    ck_assert(arrays != NULL);

    for (size_t i = 0; i < num_arrays; i++) {
      ck_assert_int_gt(arrays[i].length, 0);
      arrays[i].release(&arrays[i]);
    }

    loon_free_chunk_indices(chunk_indices);
    free(arrays);
  }

  loon_chunk_reader_destroy(chunk_reader_handle);

  loon_column_groups_destroy(out_cgs);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_chunk_metadatas(void) {
  LoonColumnGroups* out_cgs = NULL;
  LoonFFIResult rc;
  LoonProperties pp;
  size_t pp_count;
  struct ArrowSchema* schema;
  LoonReaderHandle reader_handle;

  const char* test_key[500] = {
      "writer.policy",
      "writer.split.schema_based.patterns",
  };
  const char* test_val[500] = {
      "schema_based",
      "int64_field,int32_field,string_field",
  };
  pp_count = init_test_props(test_key, test_val, 2, 500, FFI_TEST_ROOT_PATH);

  char* meta_keys[] = {"key1", "key2", "key3"};
  char* meta_vals[] = {"value101", "value2", "value3value3"};
  uint16_t meta_len = 3;

  rc = loon_properties_create((const char* const*)test_key, (const char* const*)test_val, pp_count, &pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // 500 * 501 / 2 = 125250 rows
  create_writer_test_file_with_pp(TEST_BASE_PATH, (char**)meta_keys, (char**)meta_vals, meta_len, &out_cgs, &pp,
                                  500 /*loop_times*/, 128 /*str_max_len*/, false /*with_flush*/);
  ck_assert_msg(out_cgs->num_of_column_groups > 0, "expected column group num > 0, got %u",
                out_cgs->num_of_column_groups);

  schema = create_test_struct_schema();
  rc = loon_reader_new(out_cgs, schema, NULL, 0, &pp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // test get_chunk_metadatas and get_number_of_chunks
  {
    for (int i = 0; i < out_cgs->num_of_column_groups; i++) {
      LoonChunkReaderHandle chunk_reader;
      uint64_t num_chunks = 0;
      rc = loon_get_chunk_reader(reader_handle, i, NULL, 0, &chunk_reader);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

      rc = loon_get_number_of_chunks(chunk_reader, &num_chunks);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
      ck_assert_msg(num_chunks > 0, "expected num_chunks > 0, got %" PRIu64, num_chunks);
      printf("column_group_id: %d, num_chunks: %" PRIu64 "\n", i, num_chunks);

      LoonChunkMetadatas chunk_mds1, chunk_mds2, chunk_mds3;

      rc = loon_get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_ESTIMATED_MEMORY, &chunk_mds1);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

      rc = loon_get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_NUMOFROWS, &chunk_mds2);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

      rc = loon_get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_ALL, &chunk_mds3);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

      ck_assert_msg(chunk_mds1.metadatas_size == 1, "expected ESTIMATE_MEMORY metadatas_size %hhu == 1",
                    chunk_mds1.metadatas_size);
      ck_assert_msg(chunk_mds2.metadatas_size == 1, "expected NUMOFROWS metadatas_size %hhu == 1",
                    chunk_mds2.metadatas_size);
      ck_assert_msg(chunk_mds3.metadatas_size == 2, "expected ALL metadatas_size %hhu == 2", chunk_mds3.metadatas_size);

      ck_assert(chunk_mds1.metadatas[0].number_of_chunks == num_chunks &&
                chunk_mds2.metadatas[0].number_of_chunks == num_chunks &&
                chunk_mds3.metadatas[0].number_of_chunks == num_chunks &&
                chunk_mds3.metadatas[1].number_of_chunks == num_chunks);

      for (uint64_t k = 0; k < chunk_mds1.metadatas[0].number_of_chunks; k++) {
        ck_assert_msg(chunk_mds1.metadatas[0].data[k].estimated_memsz > 0,
                      "expected chunk estimated_memsz > 0, got %llu",
                      (unsigned long long)chunk_mds1.metadatas[0].data[k].estimated_memsz);
        ck_assert_msg(chunk_mds2.metadatas[0].data[k].number_of_rows > 0, "expected chunk number_of_rows > 0, got %llu",
                      (unsigned long long)chunk_mds2.metadatas[0].data[k].number_of_rows);

        ck_assert_int_eq(chunk_mds1.metadatas[0].data[k].estimated_memsz,
                         chunk_mds3.metadatas[0].data[k].estimated_memsz);
        ck_assert_int_eq(chunk_mds2.metadatas[0].data[k].estimated_memsz,
                         chunk_mds3.metadatas[1].data[k].estimated_memsz);

        printf("  chunk %" PRIu64 ": estimated_memsz=%" PRIu64 " number_of_rows=%" PRIu64 "\n", k,
               chunk_mds1.metadatas[0].data[k].estimated_memsz, chunk_mds2.metadatas[0].data[k].number_of_rows);
      }

      loon_free_chunk_metadatas(&chunk_mds1);
      loon_free_chunk_metadatas(&chunk_mds2);
      loon_free_chunk_metadatas(&chunk_mds3);

      loon_chunk_reader_destroy(chunk_reader);
    }
  }

  // free resources in proper order
  loon_reader_destroy(reader_handle);
  loon_column_groups_destroy(out_cgs);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&pp);
}

// Test error handling for reader functions
static void test_reader_error_handling(void) {
  LoonFFIResult rc;
  LoonReaderHandle reader_handle = 0;
  LoonChunkReaderHandle chunk_reader_handle = 0;
  struct ArrowArrayStream arraystream;
  struct ArrowArray arrowarray;
  struct ArrowArray* arrays = NULL;
  size_t num_arrays = 0;
  int64_t* chunk_indices = NULL;
  size_t num_chunk_indices = 0;
  uint64_t num_chunks = 0;
  LoonChunkMetadatas chunk_mds;

  // Test loon_reader_new with null arguments
  rc = loon_reader_new(NULL, NULL, NULL, 0, NULL, &reader_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_record_batch_reader with null arguments
  rc = loon_get_record_batch_reader(0, NULL, &arraystream);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_record_batch_reader((LoonReaderHandle)1, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunk_reader with null arguments
  rc = loon_get_chunk_reader(0, 0, NULL, 0, &chunk_reader_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunk_reader((LoonReaderHandle)1, 0, NULL, 0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_take with null arguments
  int64_t row_indices[] = {0, 1, 2};
  rc = loon_take(0, row_indices, 3, 1, NULL, 0, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_take((LoonReaderHandle)1, NULL, 3, 1, NULL, 0, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_take((LoonReaderHandle)1, row_indices, 0, 1, NULL, 0, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_take((LoonReaderHandle)1, row_indices, 3, 0, NULL, 0, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunk_indices with null arguments
  rc = loon_get_chunk_indices(0, row_indices, 3, &chunk_indices, &num_chunk_indices);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunk_indices((LoonChunkReaderHandle)1, NULL, 3, &chunk_indices, &num_chunk_indices);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunk_indices((LoonChunkReaderHandle)1, row_indices, 0, &chunk_indices, &num_chunk_indices);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_number_of_chunks with null arguments
  rc = loon_get_number_of_chunks(0, &num_chunks);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_number_of_chunks((LoonChunkReaderHandle)1, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunk with null arguments
  rc = loon_get_chunk(0, 0, &arrowarray, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunk((LoonChunkReaderHandle)1, 0, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunk_metadatas with null arguments
  rc = loon_get_chunk_metadatas(0, LOON_CHUNK_METADATA_ESTIMATED_MEMORY, &chunk_mds);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunk_metadatas((LoonChunkReaderHandle)1, LOON_CHUNK_METADATA_ESTIMATED_MEMORY, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunk_metadatas with invalid metadata_type (0)
  rc = loon_get_chunk_metadatas((LoonChunkReaderHandle)1, 0, &chunk_mds);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_get_chunks with null arguments
  int64_t indices[] = {0, 1};
  rc = loon_get_chunks(0, indices, 2, 1, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunks((LoonChunkReaderHandle)1, NULL, 2, 1, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunks((LoonChunkReaderHandle)1, indices, 0, 1, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_get_chunks((LoonChunkReaderHandle)1, indices, 2, 0, &arrays, &num_arrays, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test destroy functions with 0 (should not crash)
  loon_reader_destroy(0);
  loon_chunk_reader_destroy(0);

  // Test free functions with NULL (should not crash)
  loon_free_chunk_indices(NULL);
  loon_free_chunk_arrays(NULL, 0);

  LoonChunkMetadatas null_mds = {0};
  loon_free_chunk_metadatas(&null_mds);
  loon_free_chunk_metadatas(NULL);
}

// Test key retriever callback - global variables to track callback invocation
static int g_keyretriever_called = 0;
static char g_keyretriever_metadata[256] = {0};

static const char* test_key_retriever_callback(const char* metadata) {
  g_keyretriever_called++;
  if (metadata != NULL) {
    strncpy(g_keyretriever_metadata, metadata, sizeof(g_keyretriever_metadata) - 1);
    g_keyretriever_metadata[sizeof(g_keyretriever_metadata) - 1] = '\0';
  }
  // Return the same key that was used for encryption
  return "footer_key_16B__";
}

// Helper function to create encrypted test file
static void create_encrypted_writer_test_file(char* write_path, LoonColumnGroups** out_cgs) {
  LoonWriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  LoonFFIResult rc;
  LoonProperties rp;
  int64_t length = 5;
  int64_t int64_data[] = {1, 2, 3, 4, 5};
  int32_t int32_data[] = {25, 30, 35, 40, 45};
  const char* str_data[] = {"ABC", "BCD", "DDDD", "EEEEEa", "CCCC23123"};

  // Create properties with encryption enabled
  const char* test_key[500] = {
      "writer.policy", "writer.enc.enable", "writer.enc.key", "writer.enc.meta", "writer.enc.algorithm",
  };
  const char* test_val[500] = {
      "single",
      "true",
      "footer_key_16B__",      // 16-byte key for AES-128
      "encryption_meta_data",  // metadata passed to key retriever
      "AES_GCM_V1",
  };
  size_t test_count = init_test_props(test_key, test_val, 5, 500, FFI_TEST_ROOT_PATH);

  rc = loon_properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Prepare the struct array and schema
  struct ArrowArray* children[] = {create_int64_array(int64_data, length, NULL, 0),
                                   create_int32_array(int32_data, length, NULL, 0),
                                   create_string_array(str_data, length, NULL, 0)};
  struct_array = create_struct_array(children, 3, length);
  schema = create_test_struct_schema();

  // Create writer and write data
  rc = loon_writer_new(write_path, schema, &rp, &writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_write(writer_handle, struct_array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_flush(writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_close(writer_handle, NULL, NULL, 0, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Cleanup
  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);
  loon_writer_destroy(writer_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_file_encryption(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  // Reset global tracking variables
  g_keyretriever_called = 0;
  memset(g_keyretriever_metadata, 0, sizeof(g_keyretriever_metadata));

  // Create encrypted test file
  create_encrypted_writer_test_file("keyretriever-test-dir", &out_cgs);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_cgs, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Set key retriever callback before reading
  loon_reader_set_keyretriever(reader_handle, test_key_retriever_callback);

  // Read data - this should trigger the key retriever callback
  rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Verify the key retriever was called
  ck_assert_msg(g_keyretriever_called >= 1, "Key retriever should have been called at least once, called %d times",
                g_keyretriever_called);
  ck_assert_str_eq(g_keyretriever_metadata, "encryption_meta_data");

  // Read some data to verify decryption works
  struct ArrowArray array_result;
  int arrow_rc = arraystream.get_next(&arraystream, &array_result);
  ck_assert_int_eq(0, arrow_rc);
  ck_assert(array_result.release != NULL);
  ck_assert_int_eq(array_result.length, 5);  // We wrote 5 rows

  if (array_result.release) {
    array_result.release(&array_result);
  }

  // Clean up
  if (arraystream.release) {
    arraystream.release(&arraystream);
  }
  loon_column_groups_destroy(out_cgs);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void verify_out_schema(struct ArrowSchema* out_schema,
                              const char** expected_names,
                              const char** expected_formats,
                              int expected_n_children) {
  ck_assert(out_schema->release != NULL);
  ck_assert_str_eq(out_schema->format, "+s");  // struct format
  ck_assert_int_eq(out_schema->n_children, expected_n_children);
  ck_assert(out_schema->children != NULL);

  for (int i = 0; i < expected_n_children; i++) {
    ck_assert(out_schema->children[i] != NULL);
    ck_assert_str_eq(out_schema->children[i]->name, expected_names[i]);
    ck_assert_str_eq(out_schema->children[i]->format, expected_formats[i]);
  }
}

static int alive_bit_is_set(const LoonAliveBitset* bitset, int64_t index) {
  int64_t bit_index = bitset->bit_offset + index;
  return (bitset->data[bit_index / 8] & (uint8_t)(1U << (bit_index % 8))) != 0;
}

static struct ArrowSchema* create_alive_pk_schema(void) {
  struct ArrowSchema* schema = malloc(sizeof(struct ArrowSchema));
  if (schema == NULL) {
    return NULL;
  }
  schema->format = "+s";
  schema->name = strdup("alive_pk_table");
  schema->metadata = NULL;
  schema->flags = 0;
  schema->n_children = 3;
  schema->children = malloc(sizeof(struct ArrowSchema*) * 3);
  if (schema->children == NULL) {
    free((void*)schema->name);
    free(schema);
    return NULL;
  }
  schema->children[0] = create_test_field_schema("l", "Timestamp", 0, 0);
  schema->children[1] = create_test_field_schema("l", "pk", 0, 1);
  schema->children[2] = create_test_field_schema("u", "payload", 1, 2);
  schema->dictionary = NULL;
  schema->release = struct_schema_release;
  schema->private_data = NULL;
  return schema;
}

static void create_alive_pk_base_file(const char* write_path, LoonColumnGroups** out_cgs) {
  LoonWriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  LoonFFIResult rc;
  LoonProperties rp;
  int64_t length = 5;
  int64_t timestamp_data[] = {10, 20, 30, 40, 50};
  int64_t pk_data[] = {1, 2, 3, 4, 5};
  const char* str_data[] = {"one", "two", "three", "four", "five"};

  struct ArrowArray* children[] = {create_int64_array(timestamp_data, length, NULL, 0),
                                   create_int64_array(pk_data, length, NULL, 0),
                                   create_string_array(str_data, length, NULL, 0)};
  struct_array = create_struct_array(children, 3, length);
  schema = create_alive_pk_schema();
  ck_assert(schema != NULL);

  rc = create_test_writer_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_new(write_path, schema, &rp, &writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_write(writer_handle, struct_array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_close(writer_handle, NULL, NULL, 0, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);
  loon_writer_destroy(writer_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static struct ArrowSchema* create_pk_delta_schema(void) {
  struct ArrowSchema* schema = malloc(sizeof(struct ArrowSchema));
  if (schema == NULL) {
    return NULL;
  }
  schema->format = "+s";
  schema->name = strdup("pk_delta");
  schema->metadata = NULL;
  schema->flags = 0;
  schema->n_children = 2;
  schema->children = malloc(sizeof(struct ArrowSchema*) * 2);
  if (schema->children == NULL) {
    free((void*)schema->name);
    free(schema);
    return NULL;
  }
  schema->children[0] = create_test_field_schema("l", "pk", 0, 0);
  schema->children[1] = create_test_field_schema("l", "ts", 0, 1);
  schema->dictionary = NULL;
  schema->release = struct_schema_release;
  schema->private_data = NULL;
  return schema;
}

static struct ArrowSchema* create_predicate_delta_schema(void) {
  struct ArrowSchema* schema = malloc(sizeof(struct ArrowSchema));
  if (schema == NULL) {
    return NULL;
  }
  schema->format = "+s";
  schema->name = strdup("predicate_delta");
  schema->metadata = NULL;
  schema->flags = 0;
  schema->n_children = 2;
  schema->children = malloc(sizeof(struct ArrowSchema*) * 2);
  if (schema->children == NULL) {
    free((void*)schema->name);
    free(schema);
    return NULL;
  }
  schema->children[0] = create_test_field_schema("u", "predicate", 0, 0);
  schema->children[1] = create_test_field_schema("l", "delete_timestamp", 0, 1);
  schema->dictionary = NULL;
  schema->release = struct_schema_release;
  schema->private_data = NULL;
  return schema;
}

static void create_alive_pk_delta_file(const char* write_path, LoonColumnGroups** out_cgs) {
  LoonWriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  LoonFFIResult rc;
  LoonProperties rp;
  int64_t length = 2;
  int64_t pk_data[] = {2, 3};
  int64_t ts_data[] = {25, 35};

  struct ArrowArray* children[] = {create_int64_array(pk_data, length, NULL, 0),
                                   create_int64_array(ts_data, length, NULL, 0)};
  struct_array = create_struct_array(children, 2, length);
  schema = create_pk_delta_schema();
  ck_assert(schema != NULL);

  rc = create_test_writer_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_new(write_path, schema, &rp, &writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_write(writer_handle, struct_array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_close(writer_handle, NULL, NULL, 0, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);
  loon_writer_destroy(writer_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void create_alive_predicate_delta_file(const char* write_path, LoonColumnGroups** out_cgs) {
  LoonWriterHandle writer_handle;
  struct ArrowSchema* schema;
  struct ArrowArray* struct_array;
  LoonFFIResult rc;
  LoonProperties rp;
  int64_t length = 1;
  int64_t delete_ts_data[] = {35};
  const char* predicate_sql_data[] = {"pk = 2 or payload = 'three'"};

  struct ArrowArray* children[] = {create_string_array(predicate_sql_data, length, NULL, 0),
                                   create_int64_array(delete_ts_data, length, NULL, 0)};
  struct_array = create_struct_array(children, 2, length);
  schema = create_predicate_delta_schema();
  ck_assert(schema != NULL);

  rc = create_test_writer_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_new(write_path, schema, &rp, &writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_write(writer_handle, struct_array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_writer_close(writer_handle, NULL, NULL, 0, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  if (struct_array->release) {
    struct_array->release(struct_array);
  }
  free(struct_array);
  loon_writer_destroy(writer_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void verify_alive_pk_record_batch(struct ArrowArray* out_array, struct ArrowSchema* out_schema) {
  ck_assert(out_array != NULL);
  ck_assert(out_schema != NULL);
  ck_assert_int_eq(out_array->length, 5);
  ck_assert_int_eq(out_array->n_children, 3);
  verify_out_schema(out_schema, (const char*[]){"Timestamp", "pk", "payload"}, (const char*[]){"l", "l", "u"}, 3);

  struct ArrowArray* ts_array = out_array->children[0];
  ck_assert(ts_array != NULL);
  ck_assert(ts_array->buffers != NULL);
  ck_assert(ts_array->buffers[1] != NULL);
  const int64_t* ts_values = (const int64_t*)ts_array->buffers[1];
  for (int64_t i = 0; i < out_array->length; ++i) {
    ck_assert_int_eq(ts_values[ts_array->offset + i], (i + 1) * 10);
  }

  struct ArrowArray* pk_array = out_array->children[1];
  ck_assert(pk_array != NULL);
  ck_assert(pk_array->buffers != NULL);
  ck_assert(pk_array->buffers[1] != NULL);
  const int64_t* pk_values = (const int64_t*)pk_array->buffers[1];
  for (int64_t i = 0; i < out_array->length; ++i) {
    ck_assert_int_eq(pk_values[pk_array->offset + i], i + 1);
  }
}

static void test_alive_reader_primary_key_delete_mask(void) {
  LoonColumnGroups* data_cgs = NULL;
  LoonColumnGroups* delta_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonAliveReaderHandle alive_reader;
  LoonManifest manifest;
  LoonAliveReadOptions options;
  const char* needed_columns[] = {"Timestamp", "pk", "payload"};

  create_alive_pk_base_file("reader-alive-pk-base", &data_cgs);
  create_alive_pk_delta_file("reader-alive-pk-delta", &delta_cgs);
  schema = create_alive_pk_schema();

  const char* delta_paths[] = {delta_cgs->column_group_array[0].files[0].path};
  uint32_t delta_num_entries[] = {2};
  uint32_t delta_types[] = {LOON_DELTA_LOG_TYPE_PRIMARY_KEY};
  const char* stat_keys[] = {"bloom_filter.2"};
  const char** stat_files[] = {NULL};
  uint32_t stat_file_counts[] = {0};
  const char** stat_metadata_keys[] = {NULL};
  const char** stat_metadata_values[] = {NULL};
  uint32_t stat_metadata_counts[] = {0};

  memset(&manifest, 0, sizeof(manifest));
  manifest.column_groups = *data_cgs;
  manifest.delta_logs.delta_log_paths = delta_paths;
  manifest.delta_logs.delta_log_num_entries = delta_num_entries;
  manifest.delta_logs.delta_log_types = delta_types;
  manifest.delta_logs.num_delta_logs = 1;
  manifest.stats.stat_keys = stat_keys;
  manifest.stats.stat_files = stat_files;
  manifest.stats.stat_file_counts = stat_file_counts;
  manifest.stats.stat_metadata_keys = stat_metadata_keys;
  manifest.stats.stat_metadata_values = stat_metadata_values;
  manifest.stats.stat_metadata_counts = stat_metadata_counts;
  // No bloom_filter stat is added (num_stats = 0); the primary-key field id is
  // supplied explicitly via LoonAliveReadOptions.pk_field_id below.
  manifest.stats.num_stats = 0;

  memset(&options, 0, sizeof(options));
  options.parallelism = 1;
  options.visible_until_ts = 25;
  options.pk_field_id = 2;

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_alive_reader_new(&manifest, schema, needed_columns, 3, &rp, &alive_reader, &options);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  struct ArrowArray* out_array = NULL;
  struct ArrowSchema* out_schema = NULL;
  LoonAliveBitset alive;
  rc = loon_alive_reader_next(alive_reader, &out_array, &out_schema, &alive);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  verify_alive_pk_record_batch(out_array, out_schema);
  ck_assert(alive.data != NULL);
  ck_assert_int_eq(alive.num_bits, out_array->length);
  ck_assert_int_eq(alive_bit_is_set(&alive, 0), 1);
  ck_assert_int_eq(alive_bit_is_set(&alive, 1), 0);
  ck_assert_int_eq(alive_bit_is_set(&alive, 2), 1);
  ck_assert_int_eq(alive_bit_is_set(&alive, 3), 1);
  ck_assert_int_eq(alive_bit_is_set(&alive, 4), 1);

  loon_alive_bitset_free(&alive);
  out_array->release(out_array);
  free(out_array);
  out_schema->release(out_schema);
  free(out_schema);

  out_array = NULL;
  out_schema = NULL;
  rc = loon_alive_reader_next(alive_reader, &out_array, &out_schema, &alive);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(out_array == NULL);
  ck_assert(out_schema == NULL);
  ck_assert(alive.data == NULL);
  ck_assert_int_eq(alive.num_bits, 0);

  loon_alive_reader_destroy(alive_reader);
  loon_column_groups_destroy(delta_cgs);
  loon_column_groups_destroy(data_cgs);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_alive_reader_predicate_delete_mask(void) {
  LoonColumnGroups* data_cgs = NULL;
  LoonColumnGroups* delta_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonAliveReaderHandle alive_reader;
  LoonManifest manifest;
  LoonAliveReadOptions options;
  const char* needed_columns[] = {"pk"};

  create_alive_pk_base_file("reader-alive-predicate-base", &data_cgs);
  create_alive_predicate_delta_file("reader-alive-predicate-delta", &delta_cgs);
  schema = create_alive_pk_schema();

  const char* delta_paths[] = {delta_cgs->column_group_array[0].files[0].path};
  uint32_t delta_num_entries[] = {1};
  uint32_t delta_types[] = {LOON_DELTA_LOG_TYPE_PREDICATE};

  memset(&manifest, 0, sizeof(manifest));
  manifest.column_groups = *data_cgs;
  manifest.delta_logs.delta_log_paths = delta_paths;
  manifest.delta_logs.delta_log_num_entries = delta_num_entries;
  manifest.delta_logs.delta_log_types = delta_types;
  manifest.delta_logs.num_delta_logs = 1;
  manifest.stats.stat_keys = NULL;
  manifest.stats.stat_files = NULL;
  manifest.stats.stat_file_counts = NULL;
  manifest.stats.stat_metadata_keys = NULL;
  manifest.stats.stat_metadata_values = NULL;
  manifest.stats.stat_metadata_counts = NULL;
  manifest.stats.num_stats = 0;

  memset(&options, 0, sizeof(options));
  options.parallelism = 1;
  options.visible_until_ts = 35;

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_alive_reader_new(&manifest, schema, needed_columns, 1, &rp, &alive_reader, &options);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  struct ArrowArray* out_array = NULL;
  struct ArrowSchema* out_schema = NULL;
  LoonAliveBitset alive;
  rc = loon_alive_reader_next(alive_reader, &out_array, &out_schema, &alive);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(out_array != NULL);
  ck_assert(out_schema != NULL);
  ck_assert_int_eq(out_array->length, 5);
  ck_assert_int_eq(out_array->n_children, 1);
  verify_out_schema(out_schema, (const char*[]){"pk"}, (const char*[]){"l"}, 1);
  ck_assert(alive.data != NULL);
  ck_assert_int_eq(alive.num_bits, out_array->length);
  ck_assert_int_eq(alive_bit_is_set(&alive, 0), 1);
  ck_assert_int_eq(alive_bit_is_set(&alive, 1), 0);
  ck_assert_int_eq(alive_bit_is_set(&alive, 2), 0);
  ck_assert_int_eq(alive_bit_is_set(&alive, 3), 1);
  ck_assert_int_eq(alive_bit_is_set(&alive, 4), 1);

  loon_alive_bitset_free(&alive);
  out_array->release(out_array);
  free(out_array);
  out_schema->release(out_schema);
  free(out_schema);

  out_array = NULL;
  out_schema = NULL;
  rc = loon_alive_reader_next(alive_reader, &out_array, &out_schema, &alive);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(out_array == NULL);
  ck_assert(out_schema == NULL);
  ck_assert(alive.data == NULL);
  ck_assert_int_eq(alive.num_bits, 0);

  loon_alive_reader_destroy(alive_reader);
  loon_column_groups_destroy(delta_cgs);
  loon_column_groups_destroy(data_cgs);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_alive_reader_noop_mask(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonAliveReaderHandle alive_reader;
  LoonManifest manifest;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  schema = create_test_struct_schema();
  manifest.column_groups = *out_cgs;
  manifest.delta_logs.delta_log_paths = NULL;
  manifest.delta_logs.delta_log_num_entries = NULL;
  manifest.delta_logs.delta_log_types = NULL;
  manifest.delta_logs.num_delta_logs = 0;
  manifest.stats.stat_keys = NULL;
  manifest.stats.stat_files = NULL;
  manifest.stats.stat_file_counts = NULL;
  manifest.stats.stat_metadata_keys = NULL;
  manifest.stats.stat_metadata_values = NULL;
  manifest.stats.stat_metadata_counts = NULL;
  manifest.stats.num_stats = 0;
  manifest.lob_files.files = NULL;
  manifest.lob_files.num_files = 0;

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_alive_reader_new(&manifest, schema, needed_columns, 3, &rp, &alive_reader, NULL);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t total_rows = 0;
  while (true) {
    struct ArrowArray* out_array = NULL;
    struct ArrowSchema* out_schema = NULL;
    LoonAliveBitset alive;
    rc = loon_alive_reader_next(alive_reader, &out_array, &out_schema, &alive);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    if (out_array == NULL) {
      ck_assert(out_schema == NULL);
      ck_assert(alive.data == NULL);
      ck_assert_int_eq(alive.num_bits, 0);
      break;
    }

    ck_assert(out_schema != NULL);
    ck_assert(alive.data != NULL);
    ck_assert_int_eq(alive.num_bits, out_array->length);
    ck_assert_int_eq(alive.bit_offset, 0);
    for (int64_t i = 0; i < alive.num_bits; ++i) {
      ck_assert_msg(alive_bit_is_set(&alive, i), "alive bit at row %" PRId64 " should be true", i);
    }
    total_rows += out_array->length;

    loon_alive_bitset_free(&alive);
    out_array->release(out_array);
    free(out_array);
    out_schema->release(out_schema);
    free(out_schema);
  }
  ck_assert_int_gt(total_rows, 0);

  loon_column_groups_destroy(out_cgs);
  loon_alive_reader_destroy(alive_reader);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

static void test_out_schema(void) {
  LoonColumnGroups* out_cgs = NULL;
  struct ArrowSchema* schema;
  LoonFFIResult rc;
  LoonProperties rp;
  LoonReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};
  const char* expected_names[] = {"int64_field", "int32_field", "string_field"};
  const char* expected_formats[] = {"l", "i", "u"};

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_reader_new(out_cgs, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Test loon_get_chunk with out_schema
  {
    LoonChunkReaderHandle chunk_reader_handle;
    rc = loon_get_chunk_reader(reader_handle, 0, NULL, 0, &chunk_reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    struct ArrowArray arrowarray;
    struct ArrowSchema out_schema;
    rc = loon_get_chunk(chunk_reader_handle, 0, &arrowarray, &out_schema);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    verify_out_schema(&out_schema, expected_names, expected_formats, 3);

    arrowarray.release(&arrowarray);
    out_schema.release(&out_schema);
    loon_chunk_reader_destroy(chunk_reader_handle);
  }

  // Test loon_get_chunks with out_schema
  {
    LoonChunkReaderHandle chunk_reader_handle;
    rc = loon_get_chunk_reader(reader_handle, 0, NULL, 0, &chunk_reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    int64_t chunk_idx[] = {0};
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    struct ArrowSchema out_schema;
    rc = loon_get_chunks(chunk_reader_handle, chunk_idx, 1, 1 /* parallelism */, &arrays, &num_arrays, &out_schema);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(num_arrays > 0);

    verify_out_schema(&out_schema, expected_names, expected_formats, 3);

    out_schema.release(&out_schema);
    loon_free_chunk_arrays(arrays, num_arrays);
    loon_chunk_reader_destroy(chunk_reader_handle);
  }

  // Test loon_take with out_schema
  {
    uint64_t rowidx[] = {0, 3, 5};
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    struct ArrowSchema out_schema;
    rc = loon_take(reader_handle, (const int64_t*)rowidx, 3, 1 /* parallelism */, NULL, 0, &arrays, &num_arrays,
                   &out_schema);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(num_arrays > 0);

    verify_out_schema(&out_schema, expected_names, expected_formats, 3);

    out_schema.release(&out_schema);
    loon_free_chunk_arrays(arrays, num_arrays);
  }

  // Test loon_take with partial projection via per-call needed_columns
  {
    const char* partial_columns[] = {"int64_field", "string_field"};
    const char* partial_names[] = {"int64_field", "string_field"};
    const char* partial_formats[] = {"l", "u"};

    uint64_t rowidx[] = {0, 1};
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    struct ArrowSchema out_schema;
    rc = loon_take(reader_handle, (const int64_t*)rowidx, 2, 1 /* parallelism */, partial_columns, 2, &arrays,
                   &num_arrays, &out_schema);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(num_arrays > 0);

    verify_out_schema(&out_schema, partial_names, partial_formats, 2);

    out_schema.release(&out_schema);
    loon_free_chunk_arrays(arrays, num_arrays);
  }

  loon_column_groups_destroy(out_cgs);
  loon_reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&rp);
}

void run_reader_suite(void) {
  RUN_TEST(test_basic);
  RUN_TEST(test_empty_projection);
  // RUN_TEST(test_reader_with_invalid_manifest); // Incompatible with opaque handle
  RUN_TEST(test_record_batch_reader_verify_schema);
  RUN_TEST(test_record_batch_reader_verify_arrowarray);
  RUN_TEST(test_chunk_metadatas);
  RUN_TEST(test_chunk_reader);
  RUN_TEST(test_chunk_reader_get_chunks);
  RUN_TEST(test_out_schema);
  RUN_TEST(test_alive_reader_noop_mask);
  RUN_TEST(test_alive_reader_primary_key_delete_mask);
  RUN_TEST(test_alive_reader_predicate_delete_mask);
  RUN_TEST(test_file_encryption);
  RUN_TEST(test_reader_error_handling);
}
