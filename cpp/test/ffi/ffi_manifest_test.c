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
#include "milvus-storage/ffi_fiu_c.h"
#include "test_runner.h"
#include "ffi_test_env.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>
#include <pthread.h>

#define TEST_ROOT_PATH FFI_TEST_ROOT_PATH
#define TEST_BASE_PATH "manifest-test-dir"

void create_writer_test_file(
    char* write_path, LoonColumnGroups** out_manifest, int16_t loop_times, int64_t str_max_len, bool with_flush);
void field_schema_release(struct ArrowSchema* schema);
void struct_schema_release(struct ArrowSchema* schema);
struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable);
struct ArrowSchema* create_test_struct_schema();

void create_test_pp(LoonProperties* pp) {
  const char* keys[500];
  const char* vals[500];
  size_t count = init_test_props(keys, vals, 0, 500, TEST_ROOT_PATH);

  LoonFFIResult rc = loon_properties_create((const char* const*)keys, (const char* const*)vals, count, pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

static FileSystemHandle get_fs(LoonProperties* pp) {
  FileSystemHandle fs = 0;
  LoonFFIResult rc = loon_filesystem_get(pp, TEST_ROOT_PATH, strlen(TEST_ROOT_PATH), &fs);
  assert(loon_ffi_is_success(&rc));
  return fs;
}

static void recreate_dir(FileSystemHandle fs, const char* path) {
  clean_test_dir(fs, path);
  LoonFFIResult rc = loon_filesystem_create_dir(fs, path, (uint32_t)strlen(path), true);
  ck_assert_msg(loon_ffi_is_success(&rc), "create_dir %s: %s", path, loon_ffi_get_errmsg(&rc));
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
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  // Open transaction to get latest manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &transaction);
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
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

static void test_manifests_write_read(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;

  create_test_pp(&pp);
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  LoonColumnGroups* out_cgs = NULL;
  int64_t committed_version = 0;

  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(tranhandle != 0);

  rc = loon_transaction_append_files(tranhandle, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Open a new transaction to read the committed manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction);
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

  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

static void test_abort(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction1 = 0, read_transaction2 = 0;
  LoonManifest *cmanifest1 = NULL, *cmanifest2 = NULL;
  LoonProperties pp;
  LoonFFIResult rc;

  create_test_pp(&pp);
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  // Open first transaction to read initial state
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  int64_t read_version = -1;
  rc = loon_transaction_get_read_version(read_transaction1, &read_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_msg(read_version == 0, "read_version should be 0 for empty manifests");

  rc = loon_transaction_get_manifest(read_transaction1, &cmanifest1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* read_version */, LOON_TRANSACTION_RESOLVE_FAIL,
                              1 /* retry_limit */, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(tranhandle != 0);

  loon_transaction_destroy(tranhandle);

  // Open second transaction to read state after abort
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction2);
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

  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
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
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  // First create some files using writer
  LoonColumnGroups* out_cgs = NULL;
  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);
  ck_assert(out_cgs != NULL);
  ck_assert(out_cgs->num_of_column_groups > 0);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &tranhandle);
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
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(cmanifest->column_groups.num_of_column_groups > 0);

  // Clean up
  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(read_transaction);
  loon_column_groups_destroy(out_cgs);
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
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
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &tranhandle);
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
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction);
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
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
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
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  // Open transaction
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add stat with multiple files
  const char* stat_files1[] = {"file1.parquet", "file2.parquet", "file3.parquet"};
  rc = loon_transaction_update_stat(tranhandle, "stat_key_1", stat_files1, 3, NULL, NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add another stat
  const char* stat_files2[] = {"other_file1.parquet"};
  rc = loon_transaction_update_stat(tranhandle, "stat_key_2", stat_files2, 1, NULL, NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Commit
  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(committed_version > 0);

  loon_transaction_destroy(tranhandle);

  // Verify by reading the manifest
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1 /* LATEST */, LOON_TRANSACTION_RESOLVE_FAIL, 1 /* retry_limit */,
                              &read_transaction);
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
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

// Test loon_transaction_update_stat with metadata
static void test_update_stat_with_metadata(void) {
  LoonTransactionHandle tranhandle;
  LoonTransactionHandle read_transaction = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  LoonManifest* cmanifest = NULL;
  int64_t committed_version = 0;

  create_test_pp(&pp);
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &tranhandle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add stat with files and metadata
  const char* files[] = {"bloom1.parquet", "bloom2.parquet"};
  const char* meta_keys[] = {"version", "build_id", "memory_size"};
  const char* meta_vals[] = {"3", "42", "1048576"};
  rc = loon_transaction_update_stat(tranhandle, "bloom_filter.100", files, 2, meta_keys, meta_vals, 3);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Add stat with files only (no metadata)
  const char* bm25_files[] = {"bm25_stats.bin"};
  rc = loon_transaction_update_stat(tranhandle, "bm25.101", bm25_files, 1, NULL, NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Commit and read back
  rc = loon_transaction_commit(tranhandle, &committed_version);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  loon_transaction_destroy(tranhandle);

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &read_transaction);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_get_manifest(read_transaction, &cmanifest);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  ck_assert_int_eq(cmanifest->stats.num_stats, 2);

  // Find and verify bloom_filter.100 stat
  for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
    if (strcmp(cmanifest->stats.stat_keys[i], "bloom_filter.100") == 0) {
      // Verify files
      ck_assert_int_eq(cmanifest->stats.stat_file_counts[i], 2);
      ck_assert_msg(strstr(cmanifest->stats.stat_files[i][0], "bloom1.parquet") != NULL,
                    "Expected bloom1.parquet in %s", cmanifest->stats.stat_files[i][0]);

      // Verify metadata
      ck_assert_int_eq(cmanifest->stats.stat_metadata_counts[i], 3);
      ck_assert(cmanifest->stats.stat_metadata_keys[i] != NULL);
      ck_assert(cmanifest->stats.stat_metadata_values[i] != NULL);

      // Check metadata key-value pairs exist (order may vary from map)
      int found_version = 0, found_build_id = 0, found_memory_size = 0;
      for (uint32_t j = 0; j < cmanifest->stats.stat_metadata_counts[i]; j++) {
        if (strcmp(cmanifest->stats.stat_metadata_keys[i][j], "version") == 0) {
          ck_assert_str_eq(cmanifest->stats.stat_metadata_values[i][j], "3");
          found_version = 1;
        } else if (strcmp(cmanifest->stats.stat_metadata_keys[i][j], "build_id") == 0) {
          ck_assert_str_eq(cmanifest->stats.stat_metadata_values[i][j], "42");
          found_build_id = 1;
        } else if (strcmp(cmanifest->stats.stat_metadata_keys[i][j], "memory_size") == 0) {
          ck_assert_str_eq(cmanifest->stats.stat_metadata_values[i][j], "1048576");
          found_memory_size = 1;
        }
      }
      ck_assert_msg(found_version, "version metadata not found");
      ck_assert_msg(found_build_id, "build_id metadata not found");
      ck_assert_msg(found_memory_size, "memory_size metadata not found");
    } else if (strcmp(cmanifest->stats.stat_keys[i], "bm25.101") == 0) {
      // Verify files-only stat has no metadata
      ck_assert_int_eq(cmanifest->stats.stat_file_counts[i], 1);
      ck_assert_int_eq(cmanifest->stats.stat_metadata_counts[i], 0);
    }
  }

  loon_manifest_destroy(cmanifest);
  loon_transaction_destroy(read_transaction);
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

// Test error handling for transaction functions
static void test_transaction_error_handling(void) {
  LoonFFIResult rc;
  LoonTransactionHandle handle = 0;
  LoonManifest* manifest = NULL;
  int64_t version = 0;

  // Test null arguments for loon_transaction_begin
  rc = loon_transaction_begin(NULL, NULL, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &handle);
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
  rc = loon_transaction_update_stat(0, "key", NULL, 0, NULL, NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_transaction_update_stat((LoonTransactionHandle)1, NULL, NULL, 0, NULL, NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  const char* files[] = {"file1"};
  rc = loon_transaction_update_stat((LoonTransactionHandle)1, "key", NULL, 1, NULL, NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_transaction_destroy with 0 (should not crash)
  loon_transaction_destroy(0);

  // Test loon_manifest_destroy with NULL (should not crash)
  loon_manifest_destroy(NULL);
}

// Helper payload for the A-committer thread in test_txn_exhausted_retry
typedef struct {
  LoonTransactionHandle txn;
  int64_t committed_version;
  LoonFFIResult rc;
} commit_thread_arg_t;

static void* commit_thread_fn(void* arg) {
  commit_thread_arg_t* ctx = (commit_thread_arg_t*)arg;
  // Give B a small head start so it enters write_manifest and hits the sleep FIU first
  usleep(500 * 1000);  // 500ms
  ctx->rc = loon_transaction_commit(ctx->txn, &ctx->committed_version);
  return NULL;
}

// Test LOON_TXN_EXHAUSTED_RETRY: real concurrent conflict constructed via FIU sleep.
//
// Timeline:
//   t=0    : B enters write_manifest(version=1), hits FIU sleep for 5s
//   t=500ms: A starts committing, reloads latest=0, writes version=1 successfully (FIU was one-time)
//   t=5s   : B wakes up, tries to write version=1 -> file exists -> AlreadyExists
//            -> retry_limit=0 -> LOON_TXN_EXHAUSTED_RETRY
static void test_txn_exhausted_retry(void) {
  LoonTransactionHandle txn_a = 0, txn_b = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  int64_t committed_version = 0;

  // Skip if FIU is not compiled in — this test relies on injected sleep
  if (!loon_fiu_is_enabled()) {
    fprintf(stdout, "[  SKIPPED ] test_txn_exhausted_retry (FIU not enabled)\n");
    return;
  }

  create_test_pp(&pp);
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  LoonColumnGroups* out_cgs = NULL;
  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);

  // Both transactions read version 0 (empty), use MergeResolver
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_MERGE, 0 /* retry_limit=0 */, &txn_a);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_transaction_append_files(txn_a, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_MERGE, 0 /* retry_limit=0 */, &txn_b);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_transaction_append_files(txn_b, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Enable sleep FIU ONETIME: only B (the first to hit write_manifest) will sleep
  const char* sleep_key = loon_fiukey_sleep_before_commit_manifest;
  rc = loon_fiu_enable(sleep_key, (uint32_t)strlen(sleep_key), 1 /* one_time */);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Launch A in a background thread; it will wait 500ms then commit, beating B to version 1
  commit_thread_arg_t a_ctx = {.txn = txn_a, .committed_version = 0};
  pthread_t a_thread;
  int pt_rc = pthread_create(&a_thread, NULL, commit_thread_fn, &a_ctx);
  ck_assert_msg(pt_rc == 0, "pthread_create failed: %d", pt_rc);

  // B commits on the main thread and blocks 10s inside write_manifest due to FIU
  LoonFFIResult b_rc;
  int64_t b_version = 0;
  b_rc = loon_transaction_commit(txn_b, &b_version);

  pthread_join(a_thread, NULL);

  // Print both results up front so on failure we can see exactly what each commit returned
  fprintf(stdout, "[ INFO ] A commit result: success=%d, err_code=%d, committed_version=%lld, msg=%s\n",
          loon_ffi_is_success(&a_ctx.rc), a_ctx.rc.err_code, (long long)a_ctx.committed_version,
          loon_ffi_is_success(&a_ctx.rc) ? "(null)" : loon_ffi_get_errmsg(&a_ctx.rc));
  fprintf(stdout, "[ INFO ] B commit result: success=%d, err_code=%d, committed_version=%lld, msg=%s\n",
          loon_ffi_is_success(&b_rc), b_rc.err_code, (long long)b_version,
          loon_ffi_is_success(&b_rc) ? "(null)" : loon_ffi_get_errmsg(&b_rc));

  // A should have committed version 1 successfully
  ck_assert_msg(loon_ffi_is_success(&a_ctx.rc), "A commit failed: %s", loon_ffi_get_errmsg(&a_ctx.rc));
  ck_assert_int_eq(a_ctx.committed_version, 1);
  loon_ffi_free_result(&a_ctx.rc);

  // B should have failed with LOON_TXN_EXHAUSTED_RETRY
  ck_assert_msg(!loon_ffi_is_success(&b_rc), "B commit unexpectedly succeeded, committed_version=%lld",
                (long long)b_version);
  ck_assert_msg(b_rc.err_code == LOON_TXN_EXHAUSTED_RETRY, "expected LOON_TXN_EXHAUSTED_RETRY(%d), got %d: %s",
                LOON_TXN_EXHAUSTED_RETRY, b_rc.err_code, loon_ffi_get_errmsg(&b_rc));
  loon_ffi_free_result(&b_rc);

  loon_fiu_disable_all();
  loon_transaction_destroy(txn_a);
  loon_transaction_destroy(txn_b);
  loon_column_groups_destroy(out_cgs);
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

// Test LOON_TXN_RESOLUTION_FAILED: FailResolver detects version drift
static void test_txn_resolution_failed(void) {
  LoonTransactionHandle txn_a = 0, txn_b = 0;
  LoonProperties pp;
  LoonFFIResult rc;
  int64_t committed_version = 0;

  create_test_pp(&pp);
  FileSystemHandle fs = get_fs(&pp);
  recreate_dir(fs, TEST_BASE_PATH);

  LoonColumnGroups* out_cgs = NULL;
  create_writer_test_file(TEST_BASE_PATH, &out_cgs, 1, 20, false);

  // Transaction A: read version 0
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &txn_a);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_append_files(txn_a, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Transaction B: also read version 0, uses FailResolver
  rc = loon_transaction_begin(TEST_BASE_PATH, &pp, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1, &txn_b);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_transaction_append_files(txn_b, out_cgs);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // A commits first → version 1
  LoonFFIResult a_rc;
  int64_t a_version = 0;
  a_rc = loon_transaction_commit(txn_a, &a_version);
  fprintf(stdout, "[ INFO ] A commit result: success=%d, err_code=%d, committed_version=%lld, msg=%s\n",
          loon_ffi_is_success(&a_rc), a_rc.err_code, (long long)a_version,
          loon_ffi_is_success(&a_rc) ? "(null)" : loon_ffi_get_errmsg(&a_rc));
  ck_assert_msg(loon_ffi_is_success(&a_rc), "%s", loon_ffi_get_errmsg(&a_rc));
  ck_assert_int_eq(a_version, 1);
  loon_ffi_free_result(&a_rc);

  // B tries to commit → FailResolver sees read_version(0) != latest_version(1) → resolution failed
  LoonFFIResult b_rc;
  int64_t b_version = 0;
  b_rc = loon_transaction_commit(txn_b, &b_version);
  fprintf(stdout, "[ INFO ] B commit result: success=%d, err_code=%d, committed_version=%lld, msg=%s\n",
          loon_ffi_is_success(&b_rc), b_rc.err_code, (long long)b_version,
          loon_ffi_is_success(&b_rc) ? "(null)" : loon_ffi_get_errmsg(&b_rc));
  ck_assert_msg(!loon_ffi_is_success(&b_rc), "B commit unexpectedly succeeded, committed_version=%lld",
                (long long)b_version);
  ck_assert_msg(b_rc.err_code == LOON_TXN_RESOLUTION_FAILED, "expected LOON_TXN_RESOLUTION_FAILED(%d), got %d: %s",
                LOON_TXN_RESOLUTION_FAILED, b_rc.err_code, loon_ffi_get_errmsg(&b_rc));
  loon_ffi_free_result(&b_rc);

  loon_transaction_destroy(txn_a);
  loon_transaction_destroy(txn_b);
  loon_column_groups_destroy(out_cgs);
  clean_test_dir(fs, TEST_BASE_PATH);
  loon_filesystem_destroy(fs);
  loon_properties_free(&pp);
}

void run_manifest_suite(void) {
  RUN_TEST(test_empty_manifests);
  loon_reset_context();
  RUN_TEST(test_manifests_write_read);
  loon_reset_context();
  RUN_TEST(test_abort);
  loon_reset_context();
  RUN_TEST(test_add_column_group);
  loon_reset_context();
  RUN_TEST(test_add_delta_log);
  loon_reset_context();
  RUN_TEST(test_update_stat);
  loon_reset_context();
  RUN_TEST(test_update_stat_with_metadata);
  loon_reset_context();
  RUN_TEST(test_transaction_error_handling);
  loon_reset_context();
  RUN_TEST(test_txn_exhausted_retry);
  loon_reset_context();
  RUN_TEST(test_txn_resolution_failed);
}
