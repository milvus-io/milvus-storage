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

#include "milvus-storage/ffi_fiu_c.h"
#include "test_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test that loon_fiu_is_enabled returns correct value
static void test_fiu_is_enabled(void) {
  int enabled = loon_fiu_is_enabled();
  // Should return 1 if built with FIU, 0 otherwise
  // We just verify it returns a valid boolean value
  ck_assert(enabled == 0 || enabled == 1);
}

// Test that fault point name constants are accessible and non-null
static void test_fiu_key_constants(void) {
  // Writer fault points
  ck_assert(loon_fiukey_writer_write_fail != NULL);
  ck_assert(strlen(loon_fiukey_writer_write_fail) > 0);

  ck_assert(loon_fiukey_writer_flush_fail != NULL);
  ck_assert(strlen(loon_fiukey_writer_flush_fail) > 0);

  ck_assert(loon_fiukey_writer_close_fail != NULL);
  ck_assert(strlen(loon_fiukey_writer_close_fail) > 0);

  // Reader fault points
  ck_assert(loon_fiukey_column_group_read_fail != NULL);
  ck_assert(strlen(loon_fiukey_column_group_read_fail) > 0);

  ck_assert(loon_fiukey_take_rows_fail != NULL);
  ck_assert(strlen(loon_fiukey_take_rows_fail) > 0);

  ck_assert(loon_fiukey_chunk_reader_read_fail != NULL);
  ck_assert(strlen(loon_fiukey_chunk_reader_read_fail) > 0);

  ck_assert(loon_fiukey_reader_open_fail != NULL);
  ck_assert(strlen(loon_fiukey_reader_open_fail) > 0);

  // Transaction/Manifest fault points
  ck_assert(loon_fiukey_manifest_commit_fail != NULL);
  ck_assert(strlen(loon_fiukey_manifest_commit_fail) > 0);

  ck_assert(loon_fiukey_manifest_read_fail != NULL);
  ck_assert(strlen(loon_fiukey_manifest_read_fail) > 0);

  ck_assert(loon_fiukey_manifest_write_fail != NULL);
  ck_assert(strlen(loon_fiukey_manifest_write_fail) > 0);

  // Filesystem fault points
  ck_assert(loon_fiukey_fs_open_output_fail != NULL);
  ck_assert(strlen(loon_fiukey_fs_open_output_fail) > 0);

  ck_assert(loon_fiukey_fs_open_input_fail != NULL);
  ck_assert(strlen(loon_fiukey_fs_open_input_fail) > 0);

  // S3 Filesystem fault points
  ck_assert(loon_fiukey_s3fs_create_upload_fail != NULL);
  ck_assert(strlen(loon_fiukey_s3fs_create_upload_fail) > 0);

  ck_assert(loon_fiukey_s3fs_part_upload_fail != NULL);
  ck_assert(strlen(loon_fiukey_s3fs_part_upload_fail) > 0);

  ck_assert(loon_fiukey_s3fs_complete_upload_fail != NULL);
  ck_assert(strlen(loon_fiukey_s3fs_complete_upload_fail) > 0);

  ck_assert(loon_fiukey_s3fs_read_fail != NULL);
  ck_assert(strlen(loon_fiukey_s3fs_read_fail) > 0);

  ck_assert(loon_fiukey_s3fs_readat_fail != NULL);
  ck_assert(strlen(loon_fiukey_s3fs_readat_fail) > 0);

  // ColumnGroup fault points
  ck_assert(loon_fiukey_column_group_write_fail != NULL);
  ck_assert(strlen(loon_fiukey_column_group_write_fail) > 0);
}

// Test loon_fiu_enable with null name
static void test_fiu_enable_null_name(void) {
  LoonFFIResult rc = loon_fiu_enable(NULL, 0, 0);
  // Should fail with invalid args
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
}

// Test loon_fiu_enable with empty name
static void test_fiu_enable_empty_name(void) {
  LoonFFIResult rc = loon_fiu_enable("", 0, 0);
  // Should fail with invalid args
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
}

// Test loon_fiu_disable with null name
static void test_fiu_disable_null_name(void) {
  LoonFFIResult rc = loon_fiu_disable(NULL, 0);
  // Should fail with invalid args
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
}

// Test loon_fiu_disable with empty name
static void test_fiu_disable_empty_name(void) {
  LoonFFIResult rc = loon_fiu_disable("", 0);
  // Should fail with invalid args
  ck_assert(!loon_ffi_is_success(&rc));
  loon_ffi_free_result(&rc);
}

// Test loon_fiu_enable and disable with valid fault point
static void test_fiu_enable_disable_valid(void) {
  if (!loon_fiu_is_enabled()) {
    // If FIU is not enabled, enable should return error
    const char* name = loon_fiukey_writer_write_fail;
    LoonFFIResult rc = loon_fiu_enable(name, (uint32_t)strlen(name), 1);
    ck_assert(!loon_ffi_is_success(&rc));
    loon_ffi_free_result(&rc);
    return;
  }

  // FIU is enabled, test enable/disable cycle
  const char* name = loon_fiukey_writer_write_fail;
  uint32_t name_len = (uint32_t)strlen(name);

  // Enable fault point (one_time)
  LoonFFIResult rc = loon_fiu_enable(name, name_len, 1);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Disable fault point
  rc = loon_fiu_disable(name, name_len);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

// Test loon_fiu_enable with one_time=0 (always)
static void test_fiu_enable_always(void) {
  if (!loon_fiu_is_enabled()) {
    return;  // Skip if FIU not enabled
  }

  const char* name = loon_fiukey_writer_flush_fail;
  uint32_t name_len = (uint32_t)strlen(name);

  // Enable fault point (always)
  LoonFFIResult rc = loon_fiu_enable(name, name_len, 0);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

  // Disable fault point
  rc = loon_fiu_disable(name, name_len);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

// Test loon_fiu_disable_all
static void test_fiu_disable_all(void) {
  if (!loon_fiu_is_enabled()) {
    // Should be a no-op when FIU is not enabled
    loon_fiu_disable_all();
    return;
  }

  // Enable multiple fault points
  const char* names[] = {
      loon_fiukey_writer_write_fail,
      loon_fiukey_writer_flush_fail,
      loon_fiukey_writer_close_fail,
  };

  for (int i = 0; i < 3; i++) {
    LoonFFIResult rc = loon_fiu_enable(names[i], (uint32_t)strlen(names[i]), 0);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  }

  // Disable all
  loon_fiu_disable_all();

  // Re-enabling should work (proves they were disabled)
  for (int i = 0; i < 3; i++) {
    LoonFFIResult rc = loon_fiu_enable(names[i], (uint32_t)strlen(names[i]), 1);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  }

  // Clean up
  loon_fiu_disable_all();
}

// Test that disabling a non-enabled fault point doesn't error
static void test_fiu_disable_not_enabled(void) {
  if (!loon_fiu_is_enabled()) {
    return;  // Skip if FIU not enabled
  }

  // Disable a fault point that was never enabled
  const char* name = loon_fiukey_manifest_commit_fail;
  LoonFFIResult rc = loon_fiu_disable(name, (uint32_t)strlen(name));
  // Should succeed (fiu_disable returns -1 for not enabled, but we ignore it)
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
}

void run_fiu_suite(void) {
  RUN_TEST(test_fiu_is_enabled);
  RUN_TEST(test_fiu_key_constants);
  RUN_TEST(test_fiu_enable_null_name);
  RUN_TEST(test_fiu_enable_empty_name);
  RUN_TEST(test_fiu_disable_null_name);
  RUN_TEST(test_fiu_disable_empty_name);
  RUN_TEST(test_fiu_enable_disable_valid);
  RUN_TEST(test_fiu_enable_always);
  RUN_TEST(test_fiu_disable_all);
  RUN_TEST(test_fiu_disable_not_enabled);
}
