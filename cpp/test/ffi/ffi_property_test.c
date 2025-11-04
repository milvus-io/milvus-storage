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

#define PROPERTIES_TEST_COUNT 10
#define PROPERTIES_TEST_KVSIZE 10

void create_properties_test_kvs_internal(char*** out_test_key,
                                         char*** out_test_val,
                                         size_t test_count,
                                         size_t test_kvsize) {
  assert(test_count > 0);

  // test_kvsize should not smaller than 5(3 + number + '\0')
  assert(sizeof("key0") < test_kvsize);
  char** test_key = (char**)malloc(sizeof(char*) * test_count);
  char** test_val = (char**)malloc(sizeof(char*) * test_count);

  for (int i = 0; i < test_count; i++) {
    test_key[i] = (char*)malloc(test_kvsize);
    snprintf(test_key[i], test_kvsize, "key%d", i);
    test_val[i] = (char*)malloc(test_kvsize);
    snprintf(test_val[i], test_kvsize, "val%d", i);
  }
  *out_test_key = test_key;
  *out_test_val = test_val;
}

void create_properties_test_kvs(char*** out_test_key, char*** out_test_val) {
  create_properties_test_kvs_internal(out_test_key, out_test_val, PROPERTIES_TEST_COUNT, PROPERTIES_TEST_KVSIZE);
}

void free_properties_test_kvs_internal(char** test_key, char** test_val, size_t test_count) {
  for (int i = 0; i < test_count; i++) {
    free(test_key[i]);
    free(test_val[i]);
  }
  free(test_key);
  free(test_val);
}

void free_properties_test_kvs(char** test_key, char** test_val) {
  free_properties_test_kvs_internal(test_key, test_val, PROPERTIES_TEST_COUNT);
}

START_TEST(test_basic) {
  const char* test_key = "key";
  const char* test_val = "val";
  Properties rp;
  FFIResult rc;

  rc = properties_create(&test_key, &test_val, 1, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  const char* test_val_got = properties_get(&rp, (const char*)test_key);
  ck_assert(test_val_got != NULL && strcmp(test_val_got, test_val) == 0);
  properties_free(&rp);
}
END_TEST

START_TEST(test_properties_create_multi_kvs) {
  FFIResult rc;
  Properties rp;
  char** test_key = NULL;
  char** test_val = NULL;
  size_t test_count = PROPERTIES_TEST_COUNT;

  create_properties_test_kvs(&test_key, &test_val);
  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  for (int i = 0; i < test_count; i++) {
    const char* test_val_got = properties_get(&rp, (const char*)test_key[i]);
    ck_assert(test_val_got != NULL && strcmp(test_val_got, test_val[i]) == 0);
  }

  properties_free(&rp);
  free_properties_test_kvs(test_key, test_val);
}
END_TEST

START_TEST(test_properties_create_null_kvs) {
  FFIResult rc;
  Properties rp;
  char** test_key = NULL;
  char** test_val = NULL;
  size_t test_count = PROPERTIES_TEST_COUNT;

  create_properties_test_kvs(&test_key, &test_val);

  rc = properties_create(NULL, (const char* const*)test_val, test_count, &rp);
  ck_assert(!IsSuccess(&rc));
  printf("rc message: %s\n", GetErrorMessage(&rc));
  FreeFFIResult(&rc);

  rc = properties_create((const char* const*)test_key, NULL, test_count, &rp);
  ck_assert(!IsSuccess(&rc));
  printf("rc message: %s\n", GetErrorMessage(&rc));
  FreeFFIResult(&rc);
  free_properties_test_kvs(test_key, test_val);
}
END_TEST

START_TEST(test_properties_create_null_kv) {
  FFIResult rc;
  Properties rp;
  char** test_key = NULL;
  char** test_val = NULL;
  size_t test_count = PROPERTIES_TEST_COUNT;

  create_properties_test_kvs(&test_key, &test_val);

  // save the key
  char* temp_key = test_key[test_count - 1];
  test_key[test_count - 1] = NULL;

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert(!IsSuccess(&rc));
  printf("rc message: %s\n", GetErrorMessage(&rc));
  FreeFFIResult(&rc);

  // restore the key
  test_key[test_count - 1] = temp_key;

  // save the value
  char* temp_val = test_val[test_count - 1];
  test_val[test_count - 1] = NULL;

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert(!IsSuccess(&rc));
  printf("rc message: %s\n", GetErrorMessage(&rc));
  FreeFFIResult(&rc);

  // restore the value
  test_val[test_count - 1] = temp_val;
  free_properties_test_kvs(test_key, test_val);
}
END_TEST

START_TEST(test_properties_get) {
  FFIResult rc;
  Properties rp;
  char** test_key = NULL;
  char** test_val = NULL;
  size_t test_count = PROPERTIES_TEST_COUNT;

  create_properties_test_kvs(&test_key, &test_val);

  rc = properties_create((const char* const*)test_key, (const char* const*)test_val, test_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  const char* test_val_got = properties_get(&rp, (const char*)test_key[test_count - 1]);
  ck_assert(test_val_got != NULL && strcmp(test_val_got, test_val[test_count - 1]) == 0);
  // memory ptr should not be the same
  ck_assert(test_val_got != test_val[test_count - 1]);

  test_val_got = properties_get(&rp, "Invalid.Key");
  ck_assert(test_val_got == NULL);
}
END_TEST

START_TEST(test_properties_create_dup_kv) {
  const char** test_key;
  const char** test_val;
  Properties rp;
  FFIResult rc;

  test_key = (const char**)malloc(sizeof(char*) * 2);
  test_val = (const char**)malloc(sizeof(char*) * 2);
  test_key[0] = "key";
  test_key[1] = "key";  // duplicate key

  test_val[0] = "value1";
  test_val[1] = "value2";

  rc = properties_create(test_key, test_val, 2, &rp);
  ck_assert(!IsSuccess(&rc));
  printf("rc message: %s\n", GetErrorMessage(&rc));
  FreeFFIResult(&rc);
}
END_TEST

Suite* make_properties_suite(void) {
  Suite* properties_s;

  properties_s = suite_create("FFI properties interface");

  {
    TCase* properties_create_tc;
    properties_create_tc = tcase_create("Create");
    tcase_add_test(properties_create_tc, test_basic);
    tcase_add_test(properties_create_tc, test_properties_create_multi_kvs);
    tcase_add_test(properties_create_tc, test_properties_create_null_kvs);
    tcase_add_test(properties_create_tc, test_properties_create_null_kv);
    tcase_add_test(properties_create_tc, test_properties_create_dup_kv);
    suite_add_tcase(properties_s, properties_create_tc);
  }

  {
    TCase* properties_get_tc;
    properties_get_tc = tcase_create("Get");
    tcase_add_test(properties_get_tc, test_properties_get);
    suite_add_tcase(properties_s, properties_get_tc);
  }

  return properties_s;
}