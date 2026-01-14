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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_filesystem_c.h"

#define TEST_ROOT_PATH "./"

void create_filesystem_pp(LoonProperties* pp) {
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

static void test_filesystem_write_and_read(void) {
  LoonProperties pp;
  LoonFFIResult rc;

  const char* file_path = "test_filesystem_write";
  const size_t test_buffer_len = 4096;
  uint8_t test_buffer[test_buffer_len];

  for (int i = 0; i < test_buffer_len; i++) {
    test_buffer[i] = i;
  }

  create_filesystem_pp(&pp);

  FileSystemHandle fs_handle;
  rc = loon_filesystem_get(&pp, TEST_ROOT_PATH, strlen(TEST_ROOT_PATH), &fs_handle);
  ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
  ck_assert(fs_handle != 0);

  // test write
  {
    FileSystemWriterHandle write_handle;
    rc = loon_filesystem_open_writer(fs_handle, file_path, strlen(file_path), NULL, 0, &write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(write_handle != 0);

    rc = loon_filesystem_writer_write(write_handle, test_buffer, test_buffer_len);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_flush(write_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_write(write_handle, test_buffer, test_buffer_len);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    rc = loon_filesystem_writer_write(write_handle, test_buffer, test_buffer_len);
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
    uint8_t read_buffer[test_buffer_len];
    size_t read_len;

    rc = loon_filesystem_open_reader(fs_handle, file_path, strlen(file_path), &reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert(reader_handle != 0);

    rc = loon_filesystem_reader_readat(reader_handle, 0, test_buffer_len, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, test_buffer_len), 0);

    rc = loon_filesystem_reader_readat(reader_handle, test_buffer_len, test_buffer_len, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, test_buffer_len), 0);

    rc = loon_filesystem_reader_readat(reader_handle, test_buffer_len * 2, test_buffer_len, read_buffer);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));
    ck_assert_int_eq(memcmp(read_buffer, test_buffer, test_buffer_len), 0);

    rc = loon_filesystem_reader_close(reader_handle);
    ck_assert_msg(loon_ffi_is_success(&rc), "%s", loon_ffi_get_errmsg(&rc));

    loon_filesystem_reader_destroy(reader_handle);
  }

  loon_properties_free(&pp);
}

void run_filesystem_suite(void) { RUN_TEST(test_filesystem_write_and_read); }
