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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>

#define TEST_ROOT_PATH "/tmp"
#define TEST_BASE_PATH "manifest-test-dir"

void create_writer_test_file(
    char* write_path, CColumnGroups** out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);
void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable);
struct ArrowSchema* create_test_struct_schema();

int remove_dir(const char* path) {
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
      remove_dir(full_path);
    } else {
      if (remove(full_path) != 0) {
        return -1;
      }
    }
  }

  closedir(dir);
  return rmdir(path);
}

int remove_directory(const char* root_path, const char* sub_dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", root_path, sub_dir);
  return remove_dir(path);
}

int make_directory(const char* root_path, const char* sub_dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", root_path, sub_dir);
  return mkdir(path, 0755);
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
      TEST_ROOT_PATH,
  };

  test_pp_count = sizeof(test_key) / sizeof(test_key[0]);
  assert(test_pp_count == sizeof(test_val) / sizeof(test_val[0]));

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_pp_count, pp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
}

static void test_empty_manifests(void) {
  Properties pp;
  FFIResult rc;
  TransactionHandle transaction = 0;
  CManifest* cmanifest = NULL;

  struct ArrowSchema* schema;
  ReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  struct ArrowArray arrowarray;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open transaction to get latest manifest
  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, &transaction);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // Get read version
  int64_t read_version = -1;
  rc = transaction_get_read_version(transaction, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  // read_version should be 0 for empty manifests
  ck_assert_int_eq(read_version, 0);

  // Get manifest
  rc = transaction_get_manifest(transaction, &cmanifest);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  schema = create_test_struct_schema();

  // read the empty column group - use cmanifest.column_groups directly
  rc = reader_new(&cmanifest->column_groups, schema, NULL, 0, &pp, &reader_handle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // get record batch reader
  rc = get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  // read nothing
  int arrow_rc = arraystream.get_next(&arraystream, &arrowarray);
  ck_assert(arrowarray.release == NULL);

  // Clean up resources in proper order
  if (arraystream.release) {
    arraystream.release(&arraystream);
  }
  reader_destroy(reader_handle);

  // Clean up CManifest (must be after reader_destroy since reader uses manifest's column_groups)
  manifest_destroy(cmanifest);
  transaction_destroy(transaction);

  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

static void test_manifests_write_read(void) {
  TransactionHandle tranhandle;
  TransactionHandle read_transaction = 0;
  Properties pp;
  FFIResult rc;
  CManifest* cmanifest = NULL;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  CColumnGroups* out_cgs = NULL;
  int64_t committed_version = 0;

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);

  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, &tranhandle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(tranhandle != 0);

  rc = transaction_append_files(tranhandle, out_cgs);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(committed_version > 0);

  transaction_destroy(tranhandle);

  // Open a new transaction to read the committed manifest
  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, &read_transaction);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  int64_t read_version = -1;
  rc = transaction_get_read_version(read_transaction, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_msg(read_version == 1, "read_version should be 1 after write manifest 1 time");

  rc = transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(cmanifest->column_groups.num_of_column_groups > 0 || read_version == 0);

  // Clean up
  transaction_destroy(read_transaction);
  column_groups_destroy(out_cgs);
  manifest_destroy(cmanifest);

  properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

static void test_abort(void) {
  TransactionHandle tranhandle;
  TransactionHandle read_transaction1 = 0, read_transaction2 = 0;
  CManifest *cmanifest1 = NULL, *cmanifest2 = NULL;
  Properties pp;
  FFIResult rc;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open first transaction to read initial state
  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, &read_transaction1);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  int64_t read_version = -1;
  rc = transaction_get_read_version(read_transaction1, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 for empty manifests");

  rc = transaction_get_manifest(read_transaction1, &cmanifest1);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* read_version */, &tranhandle);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert(tranhandle != 0);

  transaction_destroy(tranhandle);

  // Open second transaction to read state after abort
  rc = transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, &read_transaction2);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = transaction_get_read_version(read_transaction2, &read_version);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 after abort");

  rc = transaction_get_manifest(read_transaction2, &cmanifest2);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  // Cannot compare handles like strings
  // ck_assert_str_eq(last_manifest1, last_manifest2);

  // Clean up
  manifest_destroy(cmanifest1);
  manifest_destroy(cmanifest2);
  transaction_destroy(read_transaction1);
  transaction_destroy(read_transaction2);

  properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

void run_manifest_suite(void) {
  RUN_TEST(test_empty_manifests);
  RUN_TEST(test_manifests_write_read);
  RUN_TEST(test_abort);
}
