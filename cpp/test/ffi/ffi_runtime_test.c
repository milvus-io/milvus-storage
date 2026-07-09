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

static void test_configure_storage_runtime_rejects_zero_threads(void) {
  LoonFFIResult rc = loon_configure_storage_runtime(0, 1);
  ck_assert_msg(!loon_ffi_is_success(&rc), "expected zero CPU thread count to fail");
  ck_assert_int_eq(rc.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&rc);
}

static void test_configure_storage_runtime_success(void) {
  LoonFFIResult rc = loon_configure_storage_runtime(2, 3);
  const char* msg = loon_ffi_get_errmsg(&rc);
  ck_assert_msg(loon_ffi_is_success(&rc), "expected storage runtime configuration to succeed, got code=%d: %s",
                rc.err_code, msg ? msg : "<null>");
  loon_ffi_free_result(&rc);
}

static void test_configure_storage_runtime_duplicate_fails(void) {
  LoonFFIResult rc = loon_configure_storage_runtime(2, 3);
  ck_assert_msg(!loon_ffi_is_success(&rc), "expected duplicate storage runtime configuration to fail");
  ck_assert_int_eq(rc.err_code, LOON_ARROW_ERROR);
  loon_ffi_free_result(&rc);
}

void run_runtime_suite(void) {
  RUN_TEST(test_configure_storage_runtime_rejects_zero_threads);
  RUN_TEST(test_configure_storage_runtime_success);
  RUN_TEST(test_configure_storage_runtime_duplicate_fails);
}
