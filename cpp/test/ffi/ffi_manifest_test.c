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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>

#define TEST_BASE_PATH "manifest-test-dir"

void create_writer_test_file(
    char* write_path, char** out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);
void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable);
struct ArrowSchema* create_test_struct_schema();

int remove_directory(const char* path) {
  DIR* dir = opendir(path);
  if (dir == NULL) {
    return remove(path);
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    char full_path[1024];
    snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);

    struct stat statbuf;
    if (lstat(full_path, &statbuf) == -1) {
      return -1;
    }

    if (S_ISDIR(statbuf.st_mode)) {
      remove_directory(full_path);
    } else {
      if (remove(full_path) != 0) {
        return -1;
      }
    }
  }

  closedir(dir);
  return rmdir(path);
}

void create_test_pp(Properties* pp) {
  FFIResult rc;
  size_t test_pp_count;
  const char* test_key[] = {
      "fs.storage_type",
      "fs.root_path",
  };

  const char* test_val[] = {
      "local",
      "/tmp/",
  };

  test_pp_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(test_pp_count == sizeof(test_val) / sizeof(test_val[0]));

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_pp_count, pp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
}

START_TEST(test_empty_manifests) {
  Properties pp;
  FFIResult rc;

  struct ArrowSchema* schema;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  struct ArrowArray arrowarray;
  char* out_manifest;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_BASE_PATH);
  int mrc = mkdir(TEST_BASE_PATH, 0755);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // empty
  int64_t read_version = -1;
  rc = get_latest_column_groups(TEST_BASE_PATH, &pp, &out_manifest, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 for empty manifests");

  schema = create_test_struct_schema();

  // read the empty column group
  rc = reader_new(out_manifest, schema, NULL, 0, &pp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // get record batch reader
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // read nothing
  int arrow_rc = arraystream.get_next(&arraystream, &arrowarray);
  ck_assert(arrowarray.release == NULL);

  if (arraystream.release) {
    arraystream.release(&arraystream);
  }

  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  reader_destroy(reader_handle);
  free_cstr(out_manifest);

  properties_free(&pp);
  remove_directory(TEST_BASE_PATH);
}
END_TEST

START_TEST(test_manifests_write_read) {
  TransactionHandle tranhandle;
  Properties pp;
  FFIResult rc;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_BASE_PATH);
  int mrc = mkdir(TEST_BASE_PATH, 0755);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  char *out_manifest, *last_manifest;
  bool commit_result = false;

  create_writer_test_file(TEST_BASE_PATH, &out_manifest, 1, 20, false);

  rc = transaction_begin(TEST_BASE_PATH, &pp, &tranhandle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(tranhandle != 0);

  rc = transaction_commit(tranhandle, LOON_TRANSACTION_UPDATE_ADDFILES, LOON_TRANSACTION_RESOLVE_FAIL, out_manifest,
                          &commit_result);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(commit_result == true);

  transaction_destroy(tranhandle);

  int64_t read_version = -1;
  rc = get_latest_column_groups(TEST_BASE_PATH, &pp, &last_manifest, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(last_manifest != NULL);
  ck_assert_str_eq(last_manifest, out_manifest);
  ck_assert_msg(read_version == 1, "read_version should be 1 after write manifest 1 time");

  free_cstr(out_manifest);
  free_cstr(last_manifest);

  properties_free(&pp);
  remove_directory(TEST_BASE_PATH);
}
END_TEST

START_TEST(test_abort) {
  TransactionHandle tranhandle;
  char *last_manifest1, *last_manifest2;
  Properties pp;
  FFIResult rc;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_BASE_PATH);
  int mrc = mkdir(TEST_BASE_PATH, 0755);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  int64_t read_version = -1;
  rc = get_latest_column_groups(TEST_BASE_PATH, &pp, &last_manifest1, &read_version);
  ck_assert(last_manifest1 != NULL);
  ck_assert_msg(read_version == 0, "read_version should be 0 for empty manifests");

  rc = transaction_begin(TEST_BASE_PATH, &pp, &tranhandle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(tranhandle != 0);

  rc = transaction_abort(tranhandle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  transaction_destroy(tranhandle);
  rc = get_latest_column_groups(TEST_BASE_PATH, &pp, &last_manifest2, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(last_manifest2 != NULL);
  ck_assert_str_eq(last_manifest1, last_manifest2);
  ck_assert_msg(read_version == 0, "read_version should be 0 after abort");

  free_cstr(last_manifest1);
  free_cstr(last_manifest2);

  properties_free(&pp);
  remove_directory(TEST_BASE_PATH);
}
END_TEST

Suite* make_manifest_suite(void) {
  Suite* manifest_s;

  manifest_s = suite_create("FFI manifest interface");
  TCase* manifest_create_tc;
  manifest_create_tc = tcase_create("Create");
  tcase_add_test(manifest_create_tc, test_empty_manifests);
  tcase_add_test(manifest_create_tc, test_manifests_write_read);
  tcase_add_test(manifest_create_tc, test_abort);
  suite_add_tcase(manifest_s, manifest_create_tc);
  return manifest_s;
}