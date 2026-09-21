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
#include <sys/stat.h>

#include "milvus-storage/ffi_filesystem_metrics_c.h"

#define TEST_ROOT_PATH "test_filesystem_ffi"
#define TEST_FILE_NAME "test_filesystem_file"
enum { TEST_BUFFER_SIZE = 4096 };

static void count_async_callback(void* user_data, LoonFFIResult result, uint64_t bytes_read) {
  (void)result;
  (void)bytes_read;
  int* callback_count = (int*)user_data;
  (*callback_count)++;
}

static void create_filesystem_pp(LoonProperties* pp, const char* root_path) {
  const char* keys[500];
  const char* vals[500];
  size_t count = init_test_props(keys, vals, 0, 500, root_path);

  LoonFFIResult rc = loon_properties_create((const char* const*)keys, (const char* const*)vals, count, pp);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

static void get_test_filesystem(FileSystemHandle* fs, const char* root_path) {
  LoonProperties pp;
  LoonFFIResult rc;
  create_filesystem_pp(&pp, root_path);

  FileSystemHandle fs_handle;
  rc = loon_filesystem_get(&pp, root_path, strlen(root_path), &fs_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(fs_handle != 0);

  loon_properties_free(&pp);

  *fs = fs_handle;
}

static void write_single_file(FileSystemHandle* fs_handle) {
  LoonFFIResult rc;
  uint8_t test_buffer[TEST_BUFFER_SIZE];

  for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
    test_buffer[i] = i;
  }

  rc = loon_filesystem_write_file(*fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), test_buffer, TEST_BUFFER_SIZE,
                                  NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

// writer and read file without open writer/reader
static void test_filesystem_direct_write_and_read(void) {
  FileSystemHandle fs_handle;
  LoonFFIResult rc;
  uint8_t test_buffer[TEST_BUFFER_SIZE];

  for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
    test_buffer[i] = i;
  }
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // will overwrite the file
  rc = loon_filesystem_write_file(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), test_buffer, TEST_BUFFER_SIZE,
                                  NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  uint8_t read_buffer[TEST_BUFFER_SIZE];
  rc = loon_filesystem_read_file(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), 0, TEST_BUFFER_SIZE, read_buffer);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(memcmp(read_buffer, test_buffer, TEST_BUFFER_SIZE), 0);

  uint8_t* read_all_result = NULL;
  uint64_t read_all_result_len = 0;

  rc = loon_filesystem_read_file_all(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &read_all_result,
                                     &read_all_result_len);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(read_all_result_len, TEST_BUFFER_SIZE);
  ck_assert_int_eq(memcmp(read_all_result, test_buffer, TEST_BUFFER_SIZE), 0);
  loon_free_cstr((char*)read_all_result);

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_write_and_read(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;

  uint8_t test_buffer[TEST_BUFFER_SIZE];
  for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
    test_buffer[i] = i;
  }

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // test write
  {
    FileSystemWriterHandle write_handle;
    rc = loon_filesystem_open_writer(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), NULL, 0, false, &write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(write_handle != 0);

    rc = loon_filesystem_writer_write(write_handle, test_buffer, TEST_BUFFER_SIZE);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_flush(write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_write(write_handle, test_buffer, TEST_BUFFER_SIZE);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_write(write_handle, test_buffer, TEST_BUFFER_SIZE);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_flush(write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_close(write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    loon_filesystem_writer_destroy(write_handle);
  }

  // test read
  {
    FileSystemReaderHandle reader_handle;
    uint8_t read_buffer[TEST_BUFFER_SIZE];
    size_t read_len;

    rc = loon_filesystem_open_reader(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), 0, &reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(reader_handle != 0);

    bool supports_async = true;
    rc = loon_filesystem_reader_supports_async(reader_handle, &supports_async);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    if (!supports_async) {
      int callback_count = 0;
      rc = loon_filesystem_reader_readat_async(reader_handle, 0, TEST_BUFFER_SIZE, read_buffer, count_async_callback,
                                               &callback_count);
      ck_assert_int_eq(rc.err_code, loon_errcode_not_support);
      loon_ffi_free_result(&rc);
      ck_assert_int_eq(callback_count, 0);
    }

    rc = loon_filesystem_reader_readat(reader_handle, 0, TEST_BUFFER_SIZE, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, TEST_BUFFER_SIZE), 0);

    rc = loon_filesystem_reader_readat(reader_handle, TEST_BUFFER_SIZE, TEST_BUFFER_SIZE, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, TEST_BUFFER_SIZE), 0);

    rc = loon_filesystem_reader_readat(reader_handle, TEST_BUFFER_SIZE * 2, TEST_BUFFER_SIZE, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, TEST_BUFFER_SIZE), 0);

    rc = loon_filesystem_reader_close(reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    loon_filesystem_reader_destroy(reader_handle);
  }

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_get_file_info(void) {
  uint64_t out_size = 0;
  LoonFFIResult rc;
  FileSystemHandle fs_handle;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);
  // create file `TEST_FILE_NAME` and write
  write_single_file(&fs_handle);

  rc = loon_filesystem_get_file_info(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &out_size);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_gt(out_size, 0);

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_get_object_info(void) {
  FileSystemHandle fs_handle = 0;
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);
  write_single_file(&fs_handle);

  uint64_t size = 0;
  int64_t mtime_ns = 0;
  bool is_dir = true;
  LoonFFIResult rc =
      loon_filesystem_get_object_info(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &size, &mtime_ns, &is_dir);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(size > 0);
  ck_assert(!is_dir);
  if (!is_cloud_env()) {
    ck_assert_int_gt(mtime_ns, 0);
  }

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_get_object_info_rejects_unknown_size(void) {
  if (is_cloud_env()) {
    return;
  }

  const char* fifo_path = TEST_ROOT_PATH "/unknown_size_object";
  const char* relative_path = "unknown_size_object";
  (void)remove(fifo_path);

  FileSystemHandle fs_handle = 0;
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);
  ck_assert_int_eq(mkfifo(fifo_path, 0600), 0);

  uint64_t size = 0;
  int64_t mtime_ns = 0;
  bool is_dir = false;
  LoonFFIResult rc =
      loon_filesystem_get_object_info(fs_handle, relative_path, strlen(relative_path), &size, &mtime_ns, &is_dir);
  ck_assert_msg(!loon_ffi_is_success(&rc), "get_object_info unexpectedly accepted an entry with unknown size");
  ck_assert_int_eq(rc.err_code, loon_errcode_not_support);
  loon_ffi_free_result(&rc);

  ck_assert_int_eq(remove(fifo_path), 0);
  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_list_dir_rejects_unknown_size(void) {
  if (is_cloud_env()) {
    return;
  }

  const char* fifo_path = TEST_ROOT_PATH "/unknown_size_list_entry";
  (void)remove(fifo_path);

  FileSystemHandle fs_handle = 0;
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);
  ck_assert_int_eq(mkfifo(fifo_path, 0600), 0);

  LoonFileInfoList file_list = {0};
  LoonFFIResult rc = loon_filesystem_list_dir(fs_handle, "", 0, true, &file_list);
  ck_assert_msg(!loon_ffi_is_success(&rc), "list_dir unexpectedly accepted an entry with unknown size");
  ck_assert_int_eq(rc.err_code, loon_errcode_not_support);
  loon_ffi_free_result(&rc);

  ck_assert_int_eq(remove(fifo_path), 0);
  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_delete_file(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);
  // create file `TEST_FILE_NAME` and write
  write_single_file(&fs_handle);

  rc = loon_filesystem_delete_file(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME));
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_dir_operator(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  const char* valid_path[] = {
      "dir1",                  // NOLINT
      "dir2",                  // NOLINT
      "dir1/subdir1",          // NOLINT
      "dir3/subdir1",          // NOLINT
      "dir4/subdir1/lastdir1"  // NOLINT
  };
  size_t len_of_valid_path = sizeof(valid_path) / sizeof(valid_path[0]);

  // test create dir
  {
    // duplicate path won't return error
    const char* duplicate_path[] = {
        "dir1",          // NOLINT
        "dir4/subdir1/"  // NOLINT
    };
    size_t len_of_duplicate_path = sizeof(duplicate_path) / sizeof(duplicate_path[0]);

    for (size_t i = 0; i < len_of_valid_path; i++) {
      rc = loon_filesystem_create_dir(fs_handle, valid_path[i], strlen(valid_path[i]), true);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    }

    for (size_t i = 0; i < len_of_duplicate_path; i++) {
      rc = loon_filesystem_create_dir(fs_handle, duplicate_path[i], strlen(duplicate_path[i]), true);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    }
  }

  // test get path info
  {
    bool out_exists = false;
    bool out_is_dir = false;
    int64_t out_mtime_ns;

    // exist dir
    for (size_t i = 0; i < len_of_valid_path; i++) {
      rc = loon_filesystem_get_path_info(fs_handle, valid_path[i], strlen(valid_path[i]), &out_exists, &out_is_dir,
                                         &out_mtime_ns);
      ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
      ck_assert(out_exists);
      ck_assert(out_is_dir);
      if (!is_cloud_env()) {
        ck_assert_int_gt(out_mtime_ns, 0);
      }
    }

    // no exist dir/file
    rc = loon_filesystem_get_path_info(fs_handle, "no-exist", strlen("no-exist"), &out_exists, &out_is_dir,
                                       &out_mtime_ns);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);

    // file not a dir
    // create file `TEST_FILE_NAME` and write
    write_single_file(&fs_handle);
    rc = loon_filesystem_get_path_info(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &out_exists, &out_is_dir,
                                       &out_mtime_ns);
    ck_assert(out_exists);
    ck_assert(!out_is_dir);
    ck_assert_int_gt(out_mtime_ns, 0);
  }

  // test list dir
  {
    LoonFileInfoList file_list;

    const char* root = "";
    rc = loon_filesystem_list_dir(fs_handle, root, 0, true, &file_list);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_gt(file_list.count, len_of_valid_path);

    // Free the result
    loon_filesystem_free_file_info_list(&file_list);
  }

  // test delete nonexistent file (should fail)
  {
    rc = loon_filesystem_delete_file(fs_handle, "nonexistent_file_12345", strlen("nonexistent_file_12345"));
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  // test delete directory using delete_file (should fail - only files can be deleted)
  {
    rc = loon_filesystem_delete_file(fs_handle, "dir1", strlen("dir1"));
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  loon_filesystem_destroy(fs_handle);
}

static void verify_filesystem_metrics_all_zero(LoonFilesystemMetricsSnapshot* metrics_snapshot) {
  ck_assert_int_eq(metrics_snapshot->read_count, 0);
  ck_assert_int_eq(metrics_snapshot->write_count, 0);
  ck_assert_int_eq(metrics_snapshot->read_bytes, 0);
  ck_assert_int_eq(metrics_snapshot->write_bytes, 0);
  ck_assert_int_eq(metrics_snapshot->get_file_info_count, 0);
  ck_assert_int_eq(metrics_snapshot->create_dir_count, 0);
  ck_assert_int_eq(metrics_snapshot->delete_dir_count, 0);
  ck_assert_int_eq(metrics_snapshot->delete_file_count, 0);
  ck_assert_int_eq(metrics_snapshot->move_count, 0);
  ck_assert_int_eq(metrics_snapshot->copy_file_count, 0);
  ck_assert_int_eq(metrics_snapshot->failed_count, 0);
  ck_assert_int_eq(metrics_snapshot->multi_part_upload_created, 0);
  ck_assert_int_eq(metrics_snapshot->multi_part_upload_finished, 0);
}

static void test_filesystem_metrics(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  LoonFilesystemMetricsSnapshot metrics_snapshot;

  // init filesystem and verify metrics are all zero
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // reset and verify all zero
  rc = loon_filesystem_reset_metrics(fs_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_filesystem_get_metrics(fs_handle, &metrics_snapshot);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  verify_filesystem_metrics_all_zero(&metrics_snapshot);

  // do some write and read and verify metrics
  write_single_file(&fs_handle);
  rc = loon_filesystem_get_metrics(fs_handle, &metrics_snapshot);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  ck_assert_int_gt(metrics_snapshot.write_count, 0);
  ck_assert_int_gt(metrics_snapshot.write_bytes, 0);
  ck_assert_int_eq(metrics_snapshot.read_count, 0);
  ck_assert_int_eq(metrics_snapshot.read_bytes, 0);

  loon_filesystem_destroy(fs_handle);
}

static void test_filesystem_metrics_list(void) {
  if (is_cloud_env()) {
    return;
  }

  loon_close_filesystems();

  LoonFilesystemMetricsList metrics_list = {0};
  LoonFFIResult rc = loon_filesystem_list_metrics(&metrics_list);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(metrics_list.count, 0);
  ck_assert(metrics_list.entries == NULL);

  FileSystemHandle fs_a;
  FileSystemHandle fs_b;
  get_test_filesystem(&fs_a, "test_filesystem_metrics_list_a");
  get_test_filesystem(&fs_b, "test_filesystem_metrics_list_b");
  write_single_file(&fs_a);
  write_single_file(&fs_b);

  rc = loon_filesystem_list_metrics(&metrics_list);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(metrics_list.count, 2);
  ck_assert(metrics_list.entries != NULL);
  ck_assert(metrics_list.entries[0].display_key != NULL);
  ck_assert(metrics_list.entries[1].display_key != NULL);
  ck_assert(strcmp(metrics_list.entries[0].display_key, metrics_list.entries[1].display_key) != 0);

  bool found_a = false;
  bool found_b = false;
  const char* prefix_a = "file://test_filesystem_metrics_list_a#fs:";
  const char* prefix_b = "file://test_filesystem_metrics_list_b#fs:";
  for (uint32_t i = 0; i < metrics_list.count; ++i) {
    found_a = found_a || strncmp(metrics_list.entries[i].display_key, prefix_a, strlen(prefix_a)) == 0;
    found_b = found_b || strncmp(metrics_list.entries[i].display_key, prefix_b, strlen(prefix_b)) == 0;
  }
  ck_assert(found_a);
  ck_assert(found_b);

  ck_assert_int_gt(metrics_list.entries[0].metrics.write_count, 0);
  ck_assert_int_gt(metrics_list.entries[1].metrics.write_count, 0);

  loon_filesystem_free_metrics_list(&metrics_list);
  ck_assert_int_eq(metrics_list.count, 0);
  ck_assert(metrics_list.entries == NULL);

  rc = loon_filesystem_list_metrics(NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_delete_file(fs_a, TEST_FILE_NAME, strlen(TEST_FILE_NAME));
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  rc = loon_filesystem_delete_file(fs_b, TEST_FILE_NAME, strlen(TEST_FILE_NAME));
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  loon_filesystem_destroy(fs_a);
  loon_filesystem_destroy(fs_b);
  loon_close_filesystems();
  ck_assert_int_eq(remove("test_filesystem_metrics_list_a"), 0);
  ck_assert_int_eq(remove("test_filesystem_metrics_list_b"), 0);
}

// Test filesystem get_file_stats function
static void test_filesystem_get_file_stats(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  uint64_t out_size = 0;
  LoonFileSystemMeta* out_meta_array = NULL;
  uint32_t out_meta_count = 0;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Write a test file first
  write_single_file(&fs_handle);

  // Test get_file_stats without metadata request (out_meta_array and out_meta_count are NULL)
  rc = loon_filesystem_get_file_stats(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &out_size, NULL, NULL);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(out_size, TEST_BUFFER_SIZE);

  // Test get_file_stats with metadata request
  out_size = 0;
  rc = loon_filesystem_get_file_stats(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &out_size, &out_meta_array,
                                      &out_meta_count);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(out_size, TEST_BUFFER_SIZE);
  if (!is_cloud_env()) {
    // Local filesystem doesn't support metadata, so count should be 0
    ck_assert_int_eq(out_meta_count, 0);
    ck_assert(out_meta_array == NULL);
  }
  // Cloud metadata varies by provider: S3/MinIO returns HTTP headers (Content-Type,
  // ETag, etc.) while Azure/Azurite returns none, so we only assert size here.

  // Clean up if there was metadata
  if (out_meta_array) {
    loon_filesystem_free_meta_array(out_meta_array, out_meta_count);
  }

  loon_filesystem_destroy(fs_handle);
}

static void check_file_not_found(LoonFFIResult* result, const char* operation) {
  // Filesystem metadata paths report the generic code; S3 reads retain the
  // provider-specific NotFound code produced by ErrorToStatus.
  if (result->err_code != loon_errcode_file_not_found && result->err_code != loon_errcode_aws_not_found) {
    fprintf(stderr, "FAIL: %s: expected NotFound (%d or %d), got %d: %s\n", operation, loon_errcode_file_not_found,
            loon_errcode_aws_not_found, result->err_code,
            loon_ffi_get_errmsg(result) ? loon_ffi_get_errmsg(result) : "success");
    global_tests_failed++;
  }
  // Keep running so an unexpected error cannot bypass filesystem cleanup.
  loon_ffi_free_result(result);
}

// Test missing files preserve their generic or provider-specific NotFound error.
static void test_filesystem_file_not_found(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  uint64_t out_size = 0;
  const char* missing = "this_file_definitely_does_not_exist";
  const uint32_t missing_len = (uint32_t)strlen(missing);

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Make sure the path really isn't there (in case a previous run left it)
  rc = loon_filesystem_delete_file(fs_handle, missing, missing_len);
  // ignore rc: may fail with NotFound, which is fine
  loon_ffi_free_result(&rc);

  // --- loon_filesystem_get_file_stats (no metadata) ---
  rc = loon_filesystem_get_file_stats(fs_handle, missing, missing_len, &out_size, NULL, NULL);
  check_file_not_found(&rc, "get_file_stats");

  // --- loon_filesystem_get_file_stats (with metadata) ---
  LoonFileSystemMeta* out_meta_array = NULL;
  uint32_t out_meta_count = 0;
  out_size = 0;
  rc = loon_filesystem_get_file_stats(fs_handle, missing, missing_len, &out_size, &out_meta_array, &out_meta_count);
  check_file_not_found(&rc, "get_file_stats(meta)");
  const bool metadata_empty = out_meta_array == NULL && out_meta_count == 0;
  loon_filesystem_free_meta_array(out_meta_array, out_meta_count);

  // --- loon_filesystem_get_file_info ---
  out_size = 0;
  rc = loon_filesystem_get_file_info(fs_handle, missing, missing_len, &out_size);
  check_file_not_found(&rc, "get_file_info");

  // --- loon_filesystem_get_object_info ---
  int64_t out_mtime_ns = 1;
  bool out_is_dir = true;
  out_size = 1;
  rc = loon_filesystem_get_object_info(fs_handle, missing, missing_len, &out_size, &out_mtime_ns, &out_is_dir);
  check_file_not_found(&rc, "get_object_info");
  const bool object_info_empty = out_size == 0 && out_mtime_ns == 0 && !out_is_dir;

  // --- loon_filesystem_read_file ---
  uint8_t read_buf[16];
  rc = loon_filesystem_read_file(fs_handle, missing, missing_len, 0, sizeof(read_buf), read_buf);
  check_file_not_found(&rc, "read_file");

  // --- loon_filesystem_read_file_all ---
  uint8_t* read_all_data = NULL;
  uint64_t read_all_size = 0;
  rc = loon_filesystem_read_file_all(fs_handle, missing, missing_len, &read_all_data, &read_all_size);
  check_file_not_found(&rc, "read_file_all");
  const bool read_all_empty = read_all_data == NULL;
  free(read_all_data);

  // Remote readers open lazily: check the error on read if opening succeeds.
  FileSystemReaderHandle reader_handle = 0;
  rc = loon_filesystem_open_reader(fs_handle, missing, missing_len, 0, &reader_handle);
  if (loon_ffi_is_success(&rc)) {
    loon_ffi_free_result(&rc);
    rc = loon_filesystem_reader_readat(reader_handle, 0, sizeof(read_buf), read_buf);
    check_file_not_found(&rc, "reader_readat");
    rc = loon_filesystem_reader_close(reader_handle);
    loon_ffi_free_result(&rc);
    loon_filesystem_reader_destroy(reader_handle);
  } else {
    check_file_not_found(&rc, "open_reader");
  }

  // --- loon_filesystem_delete_file ---
  rc = loon_filesystem_delete_file(fs_handle, missing, missing_len);
  check_file_not_found(&rc, "delete_file");

  // --- loon_filesystem_get_path_info ---
  bool exists = true;
  rc = loon_filesystem_get_path_info(fs_handle, missing, missing_len, &exists, NULL, NULL);
  check_file_not_found(&rc, "get_path_info");

  // --- loon_filesystem_list_dir on a missing directory ---
  const char* missing_dir = "this_dir_definitely_does_not_exist";
  LoonFileInfoList list = {0};
  rc = loon_filesystem_list_dir(fs_handle, missing_dir, (uint32_t)strlen(missing_dir), false, &list);
  check_file_not_found(&rc, "list_dir");
  loon_filesystem_free_file_info_list(&list);

  loon_filesystem_destroy(fs_handle);
  ck_assert(metadata_empty);
  ck_assert(object_info_empty);
  ck_assert(read_all_empty);
}

// Test filesystem open_writer with metadata
static void test_filesystem_write_with_metadata(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  FileSystemWriterHandle writer_handle;
  uint8_t test_buffer[128];

  for (int i = 0; i < 128; i++) {
    test_buffer[i] = i;
  }

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Create metadata
  LoonFileSystemMeta meta_array[2];
  meta_array[0].key = "contenttype";
  meta_array[0].value = "application/octet-stream";
  meta_array[1].key = "customheader";
  meta_array[1].value = "test-value";

  // Open writer with metadata
  rc = loon_filesystem_open_writer(fs_handle, "file_with_meta", strlen("file_with_meta"), meta_array, 2, false,
                                   &writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(writer_handle != 0);

  rc = loon_filesystem_writer_write(writer_handle, test_buffer, 128);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  rc = loon_filesystem_writer_close(writer_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  loon_filesystem_writer_destroy(writer_handle);

  // Clean up
  rc = loon_filesystem_delete_file(fs_handle, "file_with_meta", strlen("file_with_meta"));
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  loon_filesystem_destroy(fs_handle);
}

// Test filesystem error handling
static void test_filesystem_error_handling(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  FileSystemWriterHandle writer_handle;
  FileSystemReaderHandle reader_handle;
  uint64_t out_size;
  uint8_t buffer[128];

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Test loon_filesystem_get with null arguments
  rc = loon_filesystem_get(NULL, NULL, 0, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_open_writer with null arguments
  rc = loon_filesystem_open_writer(0, "path", 4, NULL, 0, false, &writer_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_open_writer(fs_handle, NULL, 0, NULL, 0, false, &writer_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_open_writer(fs_handle, "path", 4, NULL, 0, false, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_open_writer with invalid metadata (num_of_meta > 0 but meta_array is NULL)
  rc = loon_filesystem_open_writer(fs_handle, "path", 4, NULL, 1, false, &writer_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_writer_write with null arguments
  rc = loon_filesystem_writer_write(0, buffer, 128);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_writer_flush with null handle
  rc = loon_filesystem_writer_flush(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_writer_close with null handle
  rc = loon_filesystem_writer_close(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_get_file_info with null arguments
  rc = loon_filesystem_get_file_info(0, "path", 4, &out_size);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_get_file_info(fs_handle, NULL, 0, &out_size);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_get_object_info with null arguments
  int64_t out_mtime_ns;
  bool out_is_dir;
  rc = loon_filesystem_get_object_info(0, "path", 4, &out_size, &out_mtime_ns, &out_is_dir);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_get_object_info(fs_handle, NULL, 0, &out_size, &out_mtime_ns, &out_is_dir);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_get_object_info(fs_handle, "path", 4, NULL, &out_mtime_ns, &out_is_dir);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_get_object_info(fs_handle, "path", 4, &out_size, NULL, &out_is_dir);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_get_object_info(fs_handle, "path", 4, &out_size, &out_mtime_ns, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_read_file with null arguments
  rc = loon_filesystem_read_file(0, "path", 4, 0, 128, buffer);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_read_file(fs_handle, NULL, 0, 0, 128, buffer);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_open_reader with null arguments
  rc = loon_filesystem_open_reader(0, "path", 4, 0, &reader_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_open_reader(fs_handle, NULL, 0, 0, &reader_handle);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_reader_readat with null arguments
  rc = loon_filesystem_reader_readat(0, 0, 128, buffer);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_reader_close with null handle
  rc = loon_filesystem_reader_close(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_delete_file with null arguments
  rc = loon_filesystem_delete_file(0, "path", 4);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  rc = loon_filesystem_delete_file(fs_handle, NULL, 0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_get_path_info with null arguments
  bool exists;
  rc = loon_filesystem_get_path_info(0, "path", 4, &exists, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_create_dir with null arguments
  rc = loon_filesystem_create_dir(0, "path", 4, true);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_list_dir with null arguments
  LoonFileInfoList file_list;
  rc = loon_filesystem_list_dir(0, "path", 4, false, &file_list);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_write_file with invalid arguments
  rc = loon_filesystem_write_file(fs_handle, "path", 4, buffer, 128, NULL, 1);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_read_file_all with null arguments
  uint8_t* out_data;
  uint64_t out_data_size;
  rc = loon_filesystem_read_file_all(0, "path", 4, &out_data, &out_data_size);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test loon_filesystem_get_file_stats with null arguments
  rc = loon_filesystem_get_file_stats(0, "path", 4, &out_size, NULL, NULL);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);

  // Test destroy with 0 (should not crash)
  loon_filesystem_destroy(0);
  loon_filesystem_writer_destroy(0);
  loon_filesystem_reader_destroy(0);

  // Test free_meta_array with NULL (should not crash)
  loon_filesystem_free_meta_array(NULL, 0);

  // Test free_file_info_list with NULL (should not crash)
  loon_filesystem_free_file_info_list(NULL);

  loon_filesystem_destroy(fs_handle);
}

// Test filesystem metadata error handling (null key/value in open_writer and write_file)
static void test_filesystem_metadata(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  FileSystemWriterHandle writer_handle;
  uint8_t test_buffer[128];

  for (int i = 0; i < 128; i++) {
    test_buffer[i] = i;
  }

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  LoonFileSystemMeta meta_array[1];

  // Test open_writer with null metadata key
  {
    meta_array[0].key = NULL;
    meta_array[0].value = "test-value";
    rc = loon_filesystem_open_writer(fs_handle, "test_null_meta", strlen("test_null_meta"), meta_array, 1, false,
                                     &writer_handle);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test open_writer with null metadata value
  {
    meta_array[0].key = "test-key";
    meta_array[0].value = NULL;
    rc = loon_filesystem_open_writer(fs_handle, "test_null_meta", strlen("test_null_meta"), meta_array, 1, false,
                                     &writer_handle);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test write_file with null metadata key
  {
    meta_array[0].key = NULL;
    meta_array[0].value = "test-value";
    rc = loon_filesystem_write_file(fs_handle, "test_null_meta_write", strlen("test_null_meta_write"), test_buffer, 128,
                                    meta_array, 1);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test write_file with null metadata value
  {
    meta_array[0].key = "test-key";
    meta_array[0].value = NULL;
    rc = loon_filesystem_write_file(fs_handle, "test_null_meta_write", strlen("test_null_meta_write"), test_buffer, 128,
                                    meta_array, 1);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  // Test write_file with data_size > 0 but data is NULL
  {
    rc = loon_filesystem_write_file(fs_handle, "test_null_data", strlen("test_null_data"), NULL, 100, NULL, 0);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
  }

  loon_filesystem_destroy(fs_handle);
}

// Test filesystem list_dir on empty directory
static void test_filesystem_list_empty_dir(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  LoonFileInfoList file_list;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Create an empty directory
  rc = loon_filesystem_create_dir(fs_handle, "empty_test_dir", strlen("empty_test_dir"), false);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // List the empty directory
  rc = loon_filesystem_list_dir(fs_handle, "empty_test_dir", strlen("empty_test_dir"), false, &file_list);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(file_list.count, 0);
  ck_assert(file_list.entries == NULL);

  loon_filesystem_destroy(fs_handle);
}

// Test filesystem write_file with zero data_size (empty file)
static void test_filesystem_write_empty_file(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  uint64_t out_size = 0;

  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Write an empty file (data_size = 0)
  rc = loon_filesystem_write_file(fs_handle, "empty_file", strlen("empty_file"), NULL, 0, NULL, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Verify the file exists and is empty
  rc = loon_filesystem_get_file_info(fs_handle, "empty_file", strlen("empty_file"), &out_size);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert_int_eq(out_size, 0);

  // Clean up
  rc = loon_filesystem_delete_file(fs_handle, "empty_file", strlen("empty_file"));
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  loon_filesystem_destroy(fs_handle);
}

// Test loon_close_filesystems
static void test_filesystem_close_all(void) {
  LoonFFIResult rc;
  FileSystemHandle fs_handle;

  // Get a filesystem first
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Write a file to ensure the filesystem is working
  write_single_file(&fs_handle);

  // Close all filesystems
  loon_close_filesystems();

  // Destroy handle (even though internal FS is closed)
  loon_filesystem_destroy(fs_handle);

  // Get a new filesystem (should work after close_filesystems)
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Verify filesystem is working
  uint64_t out_size = 0;
  rc = loon_filesystem_get_file_info(fs_handle, TEST_FILE_NAME, strlen(TEST_FILE_NAME), &out_size);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  loon_filesystem_destroy(fs_handle);
}

#define CONDITIONAL_WRITE_FILE "test_conditional_write_file"

static void test_filesystem_conditional_write(void) {
  if (!is_cloud_env()) {
    fprintf(stdout, "  [SKIPPED] conditional write test requires cloud environment\n");
    return;
  }

  LoonFFIResult rc;
  FileSystemHandle fs_handle;
  get_test_filesystem(&fs_handle, TEST_ROOT_PATH);

  // Clean up target file first (ignore error if not exists)
  rc = loon_filesystem_delete_file(fs_handle, CONDITIONAL_WRITE_FILE, strlen(CONDITIONAL_WRITE_FILE));
  loon_ffi_free_result(&rc);

  uint8_t buffer[] = "conditional write test data";
  size_t buffer_len = sizeof(buffer) - 1;

  // First conditional write should succeed
  {
    FileSystemWriterHandle writer;
    rc = loon_filesystem_open_writer(fs_handle, CONDITIONAL_WRITE_FILE, strlen(CONDITIONAL_WRITE_FILE), NULL, 0, true,
                                     &writer);
    ck_assert_msg(loon_ffi_is_success(&rc), "first conditional open failed: %s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_write(writer, buffer, buffer_len);
    ck_assert_msg(loon_ffi_is_success(&rc), "first conditional write failed: %s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_close(writer);
    ck_assert_msg(loon_ffi_is_success(&rc), "first conditional close failed: %s", loon_ffi_get_errmsg(&rc));

    loon_filesystem_writer_destroy(writer);
  }

  // Second conditional write to same path should fail (at open for Azure, at close for S3/GCP)
  {
    FileSystemWriterHandle writer;
    rc = loon_filesystem_open_writer(fs_handle, CONDITIONAL_WRITE_FILE, strlen(CONDITIONAL_WRITE_FILE), NULL, 0, true,
                                     &writer);
    if (!loon_ffi_is_success(&rc)) {
      // Azure fails at open time - this is expected
      loon_ffi_free_result(&rc);
    } else {
      // S3/GCP: open succeeds but close should fail
      rc = loon_filesystem_writer_write(writer, buffer, buffer_len);
      ck_assert_msg(loon_ffi_is_success(&rc), "second conditional write failed unexpectedly: %s",
                    loon_ffi_get_errmsg(&rc));

      rc = loon_filesystem_writer_close(writer);
      ck_assert(!loon_ffi_is_success(&rc));
      loon_ffi_free_result(&rc);

      loon_filesystem_writer_destroy(writer);
    }
  }

  // Clean up
  rc = loon_filesystem_delete_file(fs_handle, CONDITIONAL_WRITE_FILE, strlen(CONDITIONAL_WRITE_FILE));
  loon_ffi_free_result(&rc);
  loon_filesystem_destroy(fs_handle);
}

// Test threadpool with invalid args (num_of_thread = 0)
static void test_threadpool_invalid_args(void) {
  LoonFFIResult rc;

  // Test with num_of_thread = 0 (should fail)
  rc = loon_thread_pool_singleton(0);
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
}

static void test_retryable_errcode_helper(void) {
  ck_assert(loon_ffi_is_retryable_errcode(loon_errcode_transient_network));
  ck_assert(loon_ffi_is_retryable_errcode(loon_errcode_transient_timeout));
  ck_assert(loon_ffi_is_retryable_errcode(loon_errcode_transient_throttling));
  ck_assert(loon_ffi_is_retryable_errcode(loon_errcode_transient_service));
  ck_assert(loon_ffi_is_retryable_errcode(loon_errcode_aws_no_such_upload));

  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_success));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_arrow));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_file_not_found));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_aws_conflict));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_aws_precondition_failed));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_aws_not_found));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_aws_access_denied));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_aws_non_retryable));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_txn_exhausted_retry));
  ck_assert(!loon_ffi_is_retryable_errcode(loon_errcode_txn_resolution_failed));
  ck_assert(!loon_ffi_is_retryable_errcode(99999));
}

void run_filesystem_suite(void) {
  RUN_TEST(test_retryable_errcode_helper);
  RUN_TEST(test_threadpool_invalid_args);
  RUN_TEST(test_filesystem_write_and_read);
  RUN_TEST(test_filesystem_direct_write_and_read);
  RUN_TEST(test_filesystem_get_file_info);
  RUN_TEST(test_filesystem_get_object_info);
  RUN_TEST(test_filesystem_get_object_info_rejects_unknown_size);
  RUN_TEST(test_filesystem_list_dir_rejects_unknown_size);
  RUN_TEST(test_filesystem_delete_file);
  RUN_TEST(test_filesystem_dir_operator);
  RUN_TEST(test_filesystem_metrics);
  RUN_TEST(test_filesystem_metrics_list);
  RUN_TEST(test_filesystem_get_file_stats);
  RUN_TEST(test_filesystem_file_not_found);
  RUN_TEST(test_filesystem_write_with_metadata);
  RUN_TEST(test_filesystem_error_handling);
  RUN_TEST(test_filesystem_close_all);
  RUN_TEST(test_filesystem_metadata);
  RUN_TEST(test_filesystem_list_empty_dir);
  RUN_TEST(test_filesystem_write_empty_file);
  RUN_TEST(test_filesystem_conditional_write);
}
