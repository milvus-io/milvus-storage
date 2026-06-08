// Copyright 2026 Zilliz
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
#include "milvus-storage/ffi_filesystem_c.h"
#include "milvus-storage/ffi_internal/v2_packed_writer_c.h"
#include "test_runner.h"
#include "ffi_test_env.h"

#include <arrow/c/abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEGMENT_TEST_ROOT_PATH "test_filesystem_ffi"
#define SEGMENT_TEST_BASE_PATH "segment-ffi-test-dir"
#define PACKED_TEST_BASE_PATH "packed-ffi-test-dir"
#define PACKED_TEST_PATH_0 "packed-ffi-test-dir/group-0.parquet"
#define PACKED_TEST_PATH_1 "packed-ffi-test-dir/group-1.parquet"

void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_struct_schema();

struct ArrowArray* create_test_struct_arrow_array(int64_t* int64_data,
                                                  int32_t* int32_data,
                                                  const char** str_data,
                                                  int length);

static void release_schema(struct ArrowSchema* schema) {
  if (schema) {
    if (schema->release) {
      schema->release(schema);
    }
    free(schema);
  }
}

static void release_array(struct ArrowArray* array) {
  if (array) {
    if (array->release) {
      array->release(array);
    }
    free(array);
  }
}

static LoonFFIResult create_test_segment_pp(LoonProperties* pp) {
  const char* keys[500] = {"writer.policy", "writer.format"};
  const char* vals[500] = {"single", "parquet"};
  size_t count = init_test_props(keys, vals, 2, 500, SEGMENT_TEST_ROOT_PATH);

  return loon_properties_create((const char* const*)keys, (const char* const*)vals, count, pp);
}

static LoonFFIResult create_test_packed_pp(LoonProperties* pp) {
  const char* keys[500];
  const char* vals[500];
  size_t count = init_test_props(keys, vals, 0, 500, SEGMENT_TEST_ROOT_PATH);

  return loon_properties_create((const char* const*)keys, (const char* const*)vals, count, pp);
}

static void clean_segment_test_path(LoonProperties* pp) {
  LoonFFIResult rc;
  FileSystemHandle fs = 0;

  rc = loon_filesystem_get(pp, NULL, 0, &fs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  clean_test_dir(fs, SEGMENT_TEST_BASE_PATH);
  clean_test_dir(fs, PACKED_TEST_BASE_PATH);

  loon_filesystem_destroy(fs);
  loon_reset_context();
}

static struct ArrowArray* create_segment_test_array(void) {
  int64_t int64_data[] = {1, 2, 3, 4, 5};
  int32_t int32_data[] = {25, 30, 35, 40, 45};
  const char* str_data[] = {"ABC", "BCD", "DDDD", "EEEEEa", "CCCC23123"};

  return create_test_struct_arrow_array(int64_data, int32_data, str_data, 5);
}

static void assert_stream_first_batch(struct ArrowArrayStream* stream, int64_t expected_rows) {
  struct ArrowSchema out_schema;
  struct ArrowArray out_array;
  int arrow_rc;

  arrow_rc = stream->get_schema(stream, &out_schema);
  ck_assert_int_eq(0, arrow_rc);
  ck_assert(out_schema.release != NULL);
  ck_assert_int_eq(out_schema.n_children, 3);
  out_schema.release(&out_schema);

  arrow_rc = stream->get_next(stream, &out_array);
  ck_assert_int_eq(0, arrow_rc);
  ck_assert(out_array.release != NULL);
  ck_assert_int_eq(out_array.length, expected_rows);
  ck_assert_int_eq(out_array.n_children, 3);
  out_array.release(&out_array);
}

static void assert_ffi_error_code(LoonFFIResult* rc, int expected_code) {
  ck_assert_msg(!loon_ffi_is_success(rc), "expected FFI failure");
  ck_assert_msg(rc->err_code == expected_code, "expected error code %d, got %d: %s", expected_code, rc->err_code,
                loon_ffi_get_errmsg(rc));
  loon_ffi_free_result(rc);
}

static void create_committed_segment(LoonProperties* pp) {
  LoonFFIResult rc;
  LoonSegmentWriterHandle writer = 0;
  LoonTransactionHandle txn = 0;
  LoonSegmentWriteOutput output;
  int64_t committed_version = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArray* array = NULL;
  LoonSegmentWriterConfig config;

  memset(&output, 0, sizeof(output));
  memset(&config, 0, sizeof(config));
  config.segment_path = SEGMENT_TEST_BASE_PATH;

  schema = create_test_struct_schema();
  rc = loon_segment_writer_new(schema, &config, pp, &writer);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_schema(schema);

  array = create_segment_test_array();
  rc = loon_segment_writer_write(writer, array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_array(array);

  rc = loon_segment_writer_flush(writer);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_segment_writer_close(writer, &output);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(output.rows_written, 5);
  ck_assert(output.column_groups != NULL);

  rc = loon_transaction_begin(SEGMENT_TEST_BASE_PATH, pp, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &txn);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_append_files(txn, output.column_groups);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  for (size_t i = 0; i < output.num_lob_files; i++) {
    rc = loon_transaction_add_lob_file(txn, &output.lob_files[i]);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  }

  rc = loon_transaction_commit(txn, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_gt(committed_version, 0);

  loon_transaction_destroy(txn);
  loon_segment_writer_destroy(writer);
  loon_segment_write_output_free(&output);
  loon_column_groups_destroy(output.column_groups);

  (void)committed_version;
}

static void test_segment_writer_reader_basic(void) {
  LoonFFIResult rc;
  LoonProperties pp;
  LoonSegmentReaderHandle reader = 0;
  LoonChunkReaderHandle chunk_reader = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArrayStream stream;
  struct ArrowArrayStream take_stream;
  struct ArrowArrayStream filtered_stream;
  int64_t row_indices[] = {0, 2, 4};
  uint64_t number_of_chunks = 0;
  const char* needed_columns[] = {"int64_field", "int32_field", "string_field"};
  LoonSegmentReaderConfig reader_config;

  memset(&reader_config, 0, sizeof(reader_config));
  reader_config.read_buffer_size = 1024;

  rc = create_test_segment_pp(&pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  clean_segment_test_path(&pp);
  create_committed_segment(&pp);

  schema = create_test_struct_schema();
  rc = loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, needed_columns, 3, &reader_config, &pp, &reader);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_schema(schema);

  rc = loon_segment_reader_get_chunk_reader(reader, 0, NULL, 0, &chunk_reader);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_get_number_of_chunks(chunk_reader, &number_of_chunks);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_gt(number_of_chunks, 0);
  loon_chunk_reader_destroy(chunk_reader);

  memset(&stream, 0, sizeof(stream));
  rc = loon_segment_reader_get_stream(reader, &stream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  assert_stream_first_batch(&stream, 5);
  stream.release(&stream);

  memset(&take_stream, 0, sizeof(take_stream));
  rc = loon_segment_reader_take(reader, row_indices, 3, 0, &take_stream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  assert_stream_first_batch(&take_stream, 3);
  take_stream.release(&take_stream);

  memset(&filtered_stream, 0, sizeof(filtered_stream));
  rc = loon_segment_reader_get_filtered_stream(reader, NULL, &filtered_stream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  assert_stream_first_batch(&filtered_stream, 5);
  filtered_stream.release(&filtered_stream);

  loon_segment_reader_destroy(reader);
  clean_segment_test_path(&pp);
  loon_properties_free(&pp);
}

static void test_segment_error_handling(void) {
  LoonFFIResult rc;
  struct ArrowArrayStream stream;
  int64_t row_indices[] = {0};
  LoonChunkReaderHandle chunk_reader = 0;
  LoonSegmentWriteOutput output;

  memset(&stream, 0, sizeof(stream));
  memset(&output, 0, sizeof(output));

  rc = loon_segment_writer_new(NULL, NULL, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_writer_write(0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_writer_flush(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_writer_close(0, &output);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  loon_segment_writer_destroy(0);
  loon_segment_write_output_free(NULL);

  rc = loon_segment_reader_open(NULL, -1, NULL, NULL, 0, NULL, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_reader_get_stream(0, &stream);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_reader_take(0, row_indices, 1, 1, &stream);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_reader_take((LoonSegmentReaderHandle)1, NULL, 0, 1, &stream);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_reader_get_filtered_stream(0, NULL, &stream);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_segment_reader_get_chunk_reader(0, 0, NULL, 0, &chunk_reader);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  loon_segment_reader_destroy(0);
}

static void test_segment_write_output_free_ownership(void) {
  LoonFFIResult rc;
  LoonColumnGroups* column_groups = NULL;
  LoonSegmentWriteOutput output;
  const char* columns[] = {"int64_field"};
  char format[] = "parquet";
  char path0[] = "data.parquet";
  char* paths[] = {path0};
  int64_t start_indices[] = {0};
  int64_t end_indices[] = {0};

  rc = loon_column_groups_create(columns, 1, format, paths, start_indices, end_indices, 1, &column_groups);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(column_groups != NULL);

  memset(&output, 0, sizeof(output));
  output.column_groups = column_groups;
  output.lob_files = (LoonLobFileInfo*)calloc(1, sizeof(LoonLobFileInfo));
  ck_assert(output.lob_files != NULL);
  output.lob_files[0].path = strdup("lob-file.bin");
  ck_assert(output.lob_files[0].path != NULL);
  output.lob_files[0].field_id = 10;
  output.lob_files[0].total_rows = 5;
  output.lob_files[0].valid_rows = 5;
  output.lob_files[0].file_size_bytes = 128;
  output.num_lob_files = 1;
  output.rows_written = 5;

  loon_segment_write_output_free(&output);
  ck_assert(output.column_groups == column_groups);
  ck_assert(output.lob_files == NULL);
  ck_assert_int_eq(output.num_lob_files, 0);
  ck_assert_int_eq(output.rows_written, 5);

  loon_column_groups_destroy(column_groups);
}

static void test_segment_writer_reader_invalid_configs(void) {
  LoonFFIResult rc;
  LoonProperties pp;
  LoonSegmentWriterHandle writer = 0;
  LoonSegmentReaderHandle reader = 0;
  LoonChunkReaderHandle chunk_reader = 0;
  struct ArrowSchema* schema = NULL;
  LoonSegmentWriterConfig writer_config;
  LoonSegmentReaderConfig reader_config;
  struct ArrowArrayStream stream;
  int64_t row_indices[] = {0};
  const char* needed_columns[] = {"int64_field"};
  const char* bad_needed_columns[] = {"int64_field", NULL};

  rc = create_test_segment_pp(&pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  memset(&writer_config, 0, sizeof(writer_config));
  writer_config.segment_path = SEGMENT_TEST_BASE_PATH;
  writer_config.num_lob_columns = 1;
  schema = create_test_struct_schema();
  rc = loon_segment_writer_new(schema, &writer_config, &pp, &writer);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(writer, 0);
  release_schema(schema);

  memset(&writer_config, 0, sizeof(writer_config));
  schema = create_test_struct_schema();
  rc = loon_segment_writer_new(schema, &writer_config, &pp, &writer);
  assert_ffi_error_code(&rc, LOON_ARROW_ERROR);
  ck_assert_int_eq(writer, 0);
  release_schema(schema);

  memset(&reader_config, 0, sizeof(reader_config));
  reader_config.num_lob_columns = 1;
  schema = create_test_struct_schema();
  rc = loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, NULL, 0, &reader_config, &pp, &reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(reader, 0);
  release_schema(schema);

  memset(&reader_config, 0, sizeof(reader_config));
  schema = create_test_struct_schema();
  rc = loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, NULL, -1, &reader_config, &pp, &reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(reader, 0);
  release_schema(schema);

  schema = create_test_struct_schema();
  rc = loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, NULL, 1, &reader_config, &pp, &reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(reader, 0);
  release_schema(schema);

  schema = create_test_struct_schema();
  rc =
      loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, bad_needed_columns, 2, &reader_config, &pp, &reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(reader, 0);
  release_schema(schema);

  memset(&stream, 0, sizeof(stream));
  rc = loon_segment_reader_get_stream((LoonSegmentReaderHandle)1, NULL);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);

  rc = loon_segment_reader_take((LoonSegmentReaderHandle)1, row_indices, 1, 1, NULL);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);

  rc = loon_segment_reader_get_filtered_stream((LoonSegmentReaderHandle)1, NULL, NULL);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);

  rc = loon_segment_reader_get_chunk_reader((LoonSegmentReaderHandle)1, 0, NULL, 1, &chunk_reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(chunk_reader, 0);

  clean_segment_test_path(&pp);
  create_committed_segment(&pp);

  schema = create_test_struct_schema();
  rc = loon_segment_reader_open(SEGMENT_TEST_BASE_PATH, -1, schema, needed_columns, 1, &reader_config, &pp, &reader);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_schema(schema);

  rc = loon_segment_reader_get_chunk_reader(reader, 0, bad_needed_columns, 2, &chunk_reader);
  assert_ffi_error_code(&rc, LOON_INVALID_ARGS);
  ck_assert_int_eq(chunk_reader, 0);

  loon_segment_reader_destroy(reader);
  clean_segment_test_path(&pp);
  loon_properties_free(&pp);
}

static void test_packed_writer_basic(void) {
  LoonFFIResult rc;
  LoonProperties pp;
  LoonPackedWriterHandle writer = 0;
  struct ArrowSchema* schema = NULL;
  struct ArrowArray* array = NULL;
  const char* paths[] = {PACKED_TEST_PATH_0, PACKED_TEST_PATH_1};
  int32_t group_offsets[] = {0, 2, 3};
  int32_t group_indices[] = {0, 1, 2};

  rc = create_test_packed_pp(&pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  clean_segment_test_path(&pp);

  schema = create_test_struct_schema();
  rc = loon_packed_writer_new(paths, 2, group_offsets, group_indices, 3, schema, &pp, 1024 * 1024, &writer);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_schema(schema);

  array = create_segment_test_array();
  rc = loon_packed_writer_write(writer, array);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  release_array(array);

  rc = loon_packed_writer_close(writer);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  loon_packed_writer_destroy(writer);

  clean_segment_test_path(&pp);
  loon_properties_free(&pp);
}

static void test_packed_writer_error_handling(void) {
  LoonFFIResult rc;
  LoonProperties pp;
  LoonPackedWriterHandle writer = 0;
  struct ArrowSchema* schema = NULL;
  const char* paths[] = {PACKED_TEST_PATH_0};
  int32_t bad_offsets_start[] = {1, 1};
  int32_t offsets[] = {0, 1};
  int32_t bad_indices[] = {100};

  rc = create_test_packed_pp(&pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_packed_writer_new(NULL, 0, NULL, NULL, 0, NULL, NULL, 0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  schema = create_test_struct_schema();
  rc = loon_packed_writer_new(paths, 1, bad_offsets_start, offsets, 1, schema, &pp, 0, &writer);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
  release_schema(schema);

  schema = create_test_struct_schema();
  rc = loon_packed_writer_new(paths, 1, offsets, bad_indices, 1, schema, &pp, 0, &writer);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
  release_schema(schema);

  rc = loon_packed_writer_write(0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_packed_writer_close(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  loon_packed_writer_destroy(0);
  loon_properties_free(&pp);
}

void run_segment_suite(void) {
  RUN_TEST(test_segment_writer_reader_basic);
  RUN_TEST(test_segment_error_handling);
  RUN_TEST(test_segment_write_output_free_ownership);
  RUN_TEST(test_segment_writer_reader_invalid_configs);
  RUN_TEST(test_packed_writer_basic);
  RUN_TEST(test_packed_writer_error_handling);
}
