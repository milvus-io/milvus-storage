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
    char* write_path, LoonColumnGroups** out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);
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

void create_test_pp(LoonProperties* pp) {
  LoonFFIResult rc;
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

  rc = loon_properties_create((const char* const*)test_key, (const char* const*)test_val, test_pp_count, pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

static void test_empty_manifests(void) {
  LoonProperties pp;
  LoonFFIResult rc;
  LoonTransactionHandle transaction = 0;
  LoonManifest* cmanifest = NULL;

  struct ArrowSchema* schema;
  LoonReaderHandle reader_handle;
  struct ArrowArrayStream arraystream;
  struct ArrowArray arrowarray;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open transaction to get latest manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Get read version
  int64_t read_version = -1;
  rc = loon_transaction_get_read_version(transaction, &read_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  // read_version should be 0 for empty manifests
  ck_assert_int_eq(read_version, 0);

  // Get manifest
  rc = loon_transaction_get_manifest(transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  schema = create_test_struct_schema();

  // read the empty column group - use cmanifest.column_groups directly
  rc = loon_reader_new(&cmanifest->column_groups, schema, NULL, 0, &pp, &reader_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // get record batch reader
  rc = loon_get_record_batch_reader(reader_handle, NULL /*predicate*/, &arraystream);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // read nothing
  int arrow_rc = arraystream.get_next(&arraystream, &arrowarray);
  ck_assert(arrowarray.release == NULL);

  // Clean up resources in proper order
  if (arraystream.release) {
    arraystream.release(&arraystream);
  }
  loon_reader_destroy(reader_handle);

  // Clean up LoonManifest (must be after reader_destroy since reader uses manifest's column_groups)
  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(transaction);

  if (schema->release) {
    schema->release(schema);
  }
  free(schema);
  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

static void test_manifests_write_read(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  LoonColumnGroups* out_cgs = NULL;
  int64_t committed_version = 0;

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(tranhandle != 0);

  rc = loon_transaction_append_files(tranhandle, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Open a new transaction to read the committed manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t read_version = -1;
  rc = loon_transaction_get_read_version(read_transaction, &read_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_msg(read_version == 1, "read_version should be 1 after write manifest 1 time");

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(cmanifest->column_groups.num_of_column_groups > 0 || read_version == 0);

  // Clean up
  loon_transaction_destroy(read_transaction);
  loon_column_groups_destroy(out_cgs);
  loon_manifest_destroy(cmanifest);

  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

static void test_abort(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction1 = 0, read_transaction2 = 0;
  LoonManifest *cmanifest1 = NULL, *cmanifest2 = NULL;
  LoonProperties pp;
  LoonFFIResult rc;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open first transaction to read initial state
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t read_version = -1;
  rc = loon_transaction_get_read_version(read_transaction1, &read_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 for empty manifests");

  rc = loon_transaction_get_manifest(read_transaction1, &cmanifest1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* read_version */, 1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(tranhandle != 0);

  loon_transaction_destroy(tranhandle);

  // Open second transaction to read state after abort
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction2);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_read_version(read_transaction2, &read_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 after abort");

  rc = loon_transaction_get_manifest(read_transaction2, &cmanifest2);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  // Cannot compare handles like strings
  // ck_assert_str_eq(last_manifest1, last_manifest2);

  // Clean up
  loon_manifest_destroy(cmanifest1);
  loon_manifest_destroy(cmanifest2);
  loon_transaction_destroy(read_transaction1);
  loon_transaction_destroy(read_transaction2);

  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

// Test loon_transaction_add_column_group
static void test_add_column_group(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;
  int64_t committed_version = 0;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // First create some files using writer
  LoonColumnGroups* out_cgs = NULL;
  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);
  ck_assert(out_cgs != NULL);
  ck_assert(out_cgs->num_of_column_groups > 0);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add column group (use the first one from writer output)
  rc = loon_transaction_add_column_group(tranhandle, &out_cgs->column_group_array[0]);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Commit
  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Verify by reading the manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(cmanifest->column_groups.num_of_column_groups > 0);

  // Clean up
  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(read_transaction);
  loon_column_groups_destroy(out_cgs);
  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

// Test loon_transaction_add_delta_log
static void test_add_delta_log(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;
  int64_t committed_version = 0;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add delta log
  rc = loon_transaction_add_delta_log(tranhandle, "delta_log_path_1.log", 100);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add another delta log
  rc = loon_transaction_add_delta_log(tranhandle, "delta_log_path_2.log", 200);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Commit
  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Verify by reading the manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Verify delta logs are in the manifest
  ck_assert_int_eq(cmanifest->delta_logs.num_delta_logs, 2);
  ck_assert(cmanifest->delta_logs.delta_log_paths != NULL);
  ck_assert(cmanifest->delta_logs.delta_log_num_entries != NULL);
  ck_assert_msg(strstr(cmanifest->delta_logs.delta_log_paths[0], "delta_log_path_1.log") != NULL,
                "Expected path %s to contain delta_log_path_1.log", cmanifest->delta_logs.delta_log_paths[0]);
  ck_assert_msg(strstr(cmanifest->delta_logs.delta_log_paths[1], "delta_log_path_2.log") != NULL,
                "Expected path %s to contain delta_log_path_2.log", cmanifest->delta_logs.delta_log_paths[1]);
  ck_assert_int_eq(cmanifest->delta_logs.delta_log_num_entries[0], 100);
  ck_assert_int_eq(cmanifest->delta_logs.delta_log_num_entries[1], 200);

  // Clean up - this will also test the delta_logs cleanup path in loon_manifest_destroy
  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(read_transaction);
  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

// Test loon_transaction_update_stat
static void test_update_stat(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;
  int64_t committed_version = 0;

  create_test_pp(&pp);

  // recreate the test base path
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  int mrc = make_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
  ck_assert_msg(mrc == 0, "can't mkdir test base path errno: %d", mrc);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add stat with multiple files
  const char* stat_files1[] = {"file1.parquet", "file2.parquet", "file3.parquet"};
  rc = loon_transaction_update_stat(tranhandle, "stat_key_1", stat_files1, 3);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add another stat
  const char* stat_files2[] = {"other_file1.parquet"};
  rc = loon_transaction_update_stat(tranhandle, "stat_key_2", stat_files2, 1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Commit
  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Verify by reading the manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, 1 /* retry_limit */, &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Verify stats are in the manifest
  ck_assert_int_eq(cmanifest->stats.num_stats, 2);
  ck_assert(cmanifest->stats.stat_keys != NULL);
  ck_assert(cmanifest->stats.stat_files != NULL);
  ck_assert(cmanifest->stats.stat_file_counts != NULL);

  // Note: stats are stored in a map so order might vary
  // Find and verify each stat
  int found_stat1 = 0, found_stat2 = 0;
  for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
    if (strcmp(cmanifest->stats.stat_keys[i], "stat_key_1") == 0) {
      found_stat1 = 1;
      ck_assert_int_eq(cmanifest->stats.stat_file_counts[i], 3);
      ck_assert_msg(strstr(cmanifest->stats.stat_files[i][0], "file1.parquet") != NULL,
                    "Expected path %s to contain file1.parquet", cmanifest->stats.stat_files[i][0]);
      ck_assert_msg(strstr(cmanifest->stats.stat_files[i][1], "file2.parquet") != NULL,
                    "Expected path %s to contain file2.parquet", cmanifest->stats.stat_files[i][1]);
      ck_assert_msg(strstr(cmanifest->stats.stat_files[i][2], "file3.parquet") != NULL,
                    "Expected path %s to contain file3.parquet", cmanifest->stats.stat_files[i][2]);
    } else if (strcmp(cmanifest->stats.stat_keys[i], "stat_key_2") == 0) {
      found_stat2 = 1;
      ck_assert_int_eq(cmanifest->stats.stat_file_counts[i], 1);
      ck_assert_msg(strstr(cmanifest->stats.stat_files[i][0], "other_file1.parquet") != NULL,
                    "Expected path %s to contain other_file1.parquet", cmanifest->stats.stat_files[i][0]);
    }
  }
  ck_assert_msg(found_stat1, "stat_key_1 not found in manifest");
  ck_assert_msg(found_stat2, "stat_key_2 not found in manifest");

  // Clean up - this will also test the stats cleanup path in loon_manifest_destroy
  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(read_transaction);
  loon_properties_free(&pp);
  remove_directory(TEST_ROOT_PATH, TEST_BASE_PATH);
}

// Test error handling for transaction functions
static void test_transaction_error_handling(void) {
  LoonFFIResult rc;
  LoonTransactionHandle handle = 0;
  LoonManifest* manifest = NULL;
  int64_t version = 0;

  // Test null arguments for loon_transaction_begin
  rc = loon_transaction_begin(NULL, NULL, -1, 1, &handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_commit
  rc = loon_transaction_commit(0, &version);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_commit((LoonTransactionHandle)1, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_get_manifest
  rc = loon_transaction_get_manifest(0, &manifest);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_get_manifest((LoonTransactionHandle)1, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_get_read_version
  rc = loon_transaction_get_read_version(0, &version);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_get_read_version((LoonTransactionHandle)1, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_add_column_group
  rc = loon_transaction_add_column_group(0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_append_files
  rc = loon_transaction_append_files(0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_add_delta_log
  rc = loon_transaction_add_delta_log(0, "path", 100);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_add_delta_log((LoonTransactionHandle)1, NULL, 100);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test null arguments for loon_transaction_update_stat
  rc = loon_transaction_update_stat(0, "key", NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_update_stat((LoonTransactionHandle)1, NULL, NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  const char* files[] = {"file1"};
  rc = loon_transaction_update_stat((LoonTransactionHandle)1, "key", NULL, 1);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_transaction_destroy with 0 (should not crash)
  loon_transaction_destroy(0);

  // Test loon_manifest_destroy with NULL (should not crash)
  loon_manifest_destroy(NULL);
}

void run_manifest_suite(void) {
  RUN_TEST(test_empty_manifests);
  RUN_TEST(test_manifests_write_read);
  RUN_TEST(test_abort);
  RUN_TEST(test_add_column_group);
  RUN_TEST(test_add_delta_log);
  RUN_TEST(test_update_stat);
  RUN_TEST(test_transaction_error_handling);
}
