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
#include "test_runner.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <arrow/c/abi.h>

#define TEST_BASE_PATH "reader-test-dir"

// will writer 10 recordbacth
// each of recordbacth rows [1...10]
void create_writer_test_file2(char* write_path,
                              char** meta_keys,
                              char** meta_values,
                              uint16_t meta_len,
                              ColumnGroupsHandle* out_manifest,
                              int16_t loop_times,
                              int64_t str_max_len,
                              bool with_flush);

void create_writer_test_file(
    char* write_path, ColumnGroupsHandle* out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);

void create_writer_test_file_with_pp(char* write_path,
                                     char** meta_keys,
                                     char** meta_values,
                                     uint16_t meta_len,
                                     ColumnGroupsHandle* out_manifest,
                                     Properties* rp,
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

static void test_basic(void) {
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  // printf("out_manifest: %s\n", out_manifest); // out_manifest is handle, cannot print
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  {
    rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
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
    struct ArrowArray* arrays = NULL;
    size_t num_arrays = 0;
    uint64_t rowidx[] = {0, 3, 5};
    rc = take(reader_handle, (const int64_t*)rowidx, 3, 1 /* parallelism */, &arrays, &num_arrays);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    ck_assert(arrays != NULL);
    size_t number_of_rows = 0;
    for (int i = 0; i < num_arrays; i++) {
      number_of_rows += arrays[i].length;
    }
    ck_assert_int_eq(number_of_rows, 3);
    free_chunk_arrays(arrays, num_arrays);
  }

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}

static void test_empty_projection(void) {
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);
  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // full projection with needed_columns all null
  rc = reader_new(out_manifest, schema, NULL, 0, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
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

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}

// Removed replace_substring and test_reader_with_invalid_manifest as they are incompatible with opaque
// ColumnGroupsHandle

static void test_record_batch_reader_verify_schema(void) {
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* writer_schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

  writer_schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, writer_schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
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
  arraystream.release(&arraystream);

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);

  // recreated one need call the `release`
  writer_schema->release(writer_schema);
  free(writer_schema);

  properties_free(&rp);
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
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

  schema = create_test_struct_schema();

  rc = create_test_reader_pp(&rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  rc = reader_new(out_manifest, schema, needed_columns, 3, &rp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test create arrowarraysteam
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
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

  arraystream.release(&arraystream);

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}

static void test_chunk_reader(void) {
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 10 /*loop_times*/, 20 /*str_max_len*/, false /*with_flush*/);

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

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}

static void test_chunk_reader_get_chunks(void) {
  ColumnGroupsHandle out_manifest = 0;
  struct ArrowSchema* schema;
  FFIResult rc;
  Properties rp;
  ReaderHandle reader_handle;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 100 /*loop_times*/, 2000 /*str_max_len*/, true /*with_flush*/);

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

  column_groups_destroy(out_manifest);
  reader_destroy(reader_handle);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&rp);
}

static void test_chunk_metadatas(void) {
  ColumnGroupsHandle out_manifest = 0;
  FFIResult rc;
  Properties pp;
  size_t pp_count;
  struct ArrowSchema* schema;
  ReaderHandle reader_handle;

#if 0
  // minio config
  const char* test_key[] = {
      "writer.policy",
      "fs.storage_type",
      "fs.access_key_id",
      "fs.access_key_value",
      "fs.bucket_name",
      "fs.use_ssl",
      "fs.address",
      "fs.region"
  };

  const char* test_val[] = {
      "single",
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
      "writer.policy",
      "writer.split.schema_based.patterns",
  };

  const char* test_val[] = {
      "local",
      "/tmp/",
      "schema_based",
      "int64_field,int32_field,string_field",
  };
#endif

  pp_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(pp_count == sizeof(test_val) / sizeof(test_val[0]));

  char* meta_keys[] = {"key1", "key2", "key3"};
  char* meta_vals[] = {"value101", "value2", "value3value3"};
  uint16_t meta_len = 3;

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, pp_count, &pp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // 500 * 501 / 2 = 125250 rows
  create_writer_test_file_with_pp(TEST_BASE_PATH, (char**)meta_keys, (char**)meta_vals, meta_len, &out_manifest, &pp,
                                  500 /*loop_times*/, 128 /*str_max_len*/, false /*with_flush*/);

  schema = create_test_struct_schema();
  rc = reader_new(out_manifest, schema, NULL, 0, &pp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // test get_column_group_infos
  ColumnGroupInfos column_group_infos;
  {
    rc = get_column_group_infos(reader_handle, &column_group_infos, false /*with_meta*/);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    ck_assert_msg(column_group_infos.cginfos_size == 3, "expected column group num 3, got %zu",
                  column_group_infos.cginfos_size);
    ck_assert_msg(column_group_infos.cg_infos[0].columns_size == 1, "expected column size 1 in column group 0, got %zu",
                  column_group_infos.cg_infos[0].columns_size);
    ck_assert_str_eq(column_group_infos.cg_infos[0].columns[0], "int64_field");
    ck_assert_msg(column_group_infos.cg_infos[1].columns_size == 1, "expected column size 1 in column group 1, got %zu",
                  column_group_infos.cg_infos[1].columns_size);
    ck_assert_str_eq(column_group_infos.cg_infos[1].columns[0], "int32_field");
    ck_assert_msg(column_group_infos.cg_infos[2].columns_size == 1, "expected column size 1 in column group 2, got %zu",
                  column_group_infos.cg_infos[2].columns_size);
    ck_assert_str_eq(column_group_infos.cg_infos[2].columns[0], "string_field");

    ck_assert_msg(column_group_infos.meta_size == 0, "expected meta keys size 0, got %zu",
                  column_group_infos.meta_size);
  }

  // test get_column_group_infos with meta
  {
    ColumnGroupInfos column_group_infos_with_meta;
    rc = get_column_group_infos(reader_handle, &column_group_infos_with_meta, true /*with_meta*/);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    ck_assert_msg(column_group_infos_with_meta.cginfos_size == meta_len, "expected column group num %hu, got %zu",
                  meta_len, column_group_infos_with_meta.cginfos_size);
    for (uint16_t i = 0; i < meta_len; i++) {
      ck_assert_str_eq(column_group_infos_with_meta.meta_keys[i], meta_keys[i]);
      ck_assert_str_eq(column_group_infos_with_meta.meta_values[i], meta_vals[i]);
    }

    free_column_group_infos(&column_group_infos_with_meta);
  }

  // test get_chunk_metadatas and get_number_of_chunks
  {
    for (int i = 0; i < column_group_infos.cginfos_size; i++) {
      ChunkReaderHandle chunk_reader;
      uint64_t num_chunks = 0;
      rc = get_chunk_reader(reader_handle, column_group_infos.cg_infos[i].column_group_id, &chunk_reader);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

      rc = get_number_of_chunks(chunk_reader, &num_chunks);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
      ck_assert_msg(num_chunks > 0, "expected num_chunks > 0, got %lld", num_chunks);
      printf("column_group_id: %lld, num_chunks: %lld\n", column_group_infos.cg_infos[i].column_group_id, num_chunks);

      ChunkMetadatas chunk_mds1, chunk_mds2, chunk_mds3;

      rc = get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_ESTIMATED_MEMORY, &chunk_mds1);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

      rc = get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_NUMOFROWS, &chunk_mds2);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

      rc = get_chunk_metadatas(chunk_reader, LOON_CHUNK_METADATA_ALL, &chunk_mds3);
      ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

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
                      "expected chunk estimated_memsz > 0, got %llu", chunk_mds1.metadatas[0].data[k].estimated_memsz);
        ck_assert_msg(chunk_mds2.metadatas[0].data[k].number_of_rows > 0, "expected chunk number_of_rows > 0, got %llu",
                      chunk_mds2.metadatas[0].data[k].number_of_rows);

        ck_assert_int_eq(chunk_mds1.metadatas[0].data[k].estimated_memsz,
                         chunk_mds3.metadatas[0].data[k].estimated_memsz);
        ck_assert_int_eq(chunk_mds2.metadatas[0].data[k].estimated_memsz,
                         chunk_mds3.metadatas[1].data[k].estimated_memsz);

        printf("  chunk %llu: estimated_memsz=%llu number_of_rows=%llu \n", k,
               chunk_mds1.metadatas[0].data[k].estimated_memsz, chunk_mds2.metadatas[0].data[k].number_of_rows);
      }

      free_chunk_metadatas(&chunk_mds1);
      free_chunk_metadatas(&chunk_mds2);
      free_chunk_metadatas(&chunk_mds3);

      chunk_reader_destroy(chunk_reader);
    }
  }

  // free resources
  column_groups_destroy(out_manifest);
  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&pp);
  free_column_group_infos(&column_group_infos);
  reader_destroy(reader_handle);
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
}
