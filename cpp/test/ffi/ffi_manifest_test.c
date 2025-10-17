#include "milvus-storage/ffi_c.h"
#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <arrow/c/abi.h>

const char* manifest1 =
    "{\n"
    "  \"column_groups\": [\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"id\",\n"
    "        \"value\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file1.parquet\"\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"name\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file2.parquet\"\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"vector\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file3.parquet\"\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    "  \"version\": 0\n"
    "}";

const char* manifest2 =
    "{\n"
    "  \"column_groups\": [\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"id\",\n"
    "        \"value\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file101.parquet\",\n"
    "        \"file102.parquet\"\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"name\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file103.parquet\",\n"
    "        \"file104.parquet\"\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"columns\": [\n"
    "        \"vector\"\n"
    "      ],\n"
    "      \"format\": \"parquet\",\n"
    "      \"paths\": [\n"
    "        \"file105.parquet\"\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    "  \"version\": 0\n"
    "}";

const char* manifest3 =
    "{"
    "  \"column_groups\": ["
    "    {"
    "      \"columns\": ["
    "        \"new_column1\""
    "      ],"
    "      \"format\": \"parquet\","
    "      \"paths\": ["
    "        \"new_column1_file101.parquet\""
    "      ]"
    "    }"
    "  ],"
    "  \"version\": 0"
    "}";

const char* manifest4 =
    "{"
    "  \"column_groups\": ["
    "    {"
    "      \"columns\": ["
    "        \"new_column1\","
    "        \"new_column2\""
    "      ],"
    "      \"format\": \"parquet\","
    "      \"paths\": ["
    "        \"new_column_file101.parquet\","
    "        \"new_column_file102.parquet\""
    "      ]"
    "    }"
    "  ],"
    "  \"version\": 0"
    "}";

const char* manifest5 =
    "{"
    "  \"column_groups\": ["
    "    {"
    "      \"columns\": ["
    "        \"new_column1\","
    "        \"new_column2\""
    "      ],"
    "      \"format\": \"parquet\","
    "      \"paths\": ["
    "        \"new_column_file101.parquet\","
    "        \"new_column_file102.parquet\""
    "      ]"
    "    },"
    "    {"
    "      \"columns\": ["
    "        \"new_column3\""
    "      ],"
    "      \"format\": \"parquet\","
    "      \"paths\": ["
    "        \"new_column_file103.parquet\""
    "      ]"
    "    }"
    "  ],"
    "  \"version\": 0"
    "}";

START_TEST(test_manifest_write_read) {
  char* test_manifest;
  FFIResult rc;
  Properties rp;

  const char* pp_key[] = {
      "fs.storage_type",
      "fs.root_path",
  };

  const char* pp_val[] = {
      "local",
      "/tmp/",
  };

  // perpare the properties
  size_t test_count = sizeof(pp_key) / sizeof(pp_key[0]);
  assert(test_count == sizeof(pp_val) / sizeof(pp_val[0]));

  rc = properties_create((const char* const*)pp_key, (const char* const*)pp_val, test_count, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = manifest_write("manifest1.json", manifest1, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

  rc = manifest_read("manifest1.json", &test_manifest, &rp);
  ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
  ck_assert_str_eq(test_manifest, manifest1);

  properties_free(&rp);
  manifest_destory(test_manifest);
}
END_TEST

START_TEST(test_manifest_combine) {
  FFIResult rc;
  char* test_manifest;

  {
    rc = manifest_combine(manifest1, manifest2, &test_manifest);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    printf("Combined manifest: %s\n", test_manifest);
    manifest_destory(test_manifest);
  }

  {
    rc = manifest_combine(manifest2, manifest1, &test_manifest);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));

    printf("Combined manifest: %s\n", test_manifest);
    manifest_destory(test_manifest);
  }

  // no match column groups
  {
    rc = manifest_combine(manifest4, manifest5, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);

    rc = manifest_combine(manifest5, manifest4, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);
  }

  // invalid
  {
    rc = manifest_combine(manifest1, manifest3, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);

    rc = manifest_combine(manifest2, manifest3, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);

    rc = manifest_combine(manifest3, manifest1, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);

    rc = manifest_combine(manifest3, manifest2, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);
  }
}
END_TEST

START_TEST(test_manifest_add_column) {
  FFIResult rc;
  char* test_manifest;

  {
    rc = manifest_add_columns(manifest1, manifest3, &test_manifest);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    printf("Updated manifest: %s\n", test_manifest);
    manifest_destory(test_manifest);

    rc = manifest_add_columns(manifest1, manifest4, &test_manifest);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    printf("Updated manifest: %s\n", test_manifest);
    manifest_destory(test_manifest);

    rc = manifest_add_columns(manifest1, manifest5, &test_manifest);
    ck_assert_msg(IsSuccess(&rc), "%s", GetErrorMessage(&rc));
    printf("Updated manifest: %s\n", test_manifest);
    manifest_destory(test_manifest);
  }

  // invalid
  {
    rc = manifest_add_columns(manifest1, manifest2, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    printf("Expected error: %s\n", GetErrorMessage(&rc));
    FreeFFIResult(&rc);

    rc = manifest_add_columns(manifest2, manifest1, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);

    rc = manifest_add_columns(manifest4, manifest5, &test_manifest);
    ck_assert(!IsSuccess(&rc));
    FreeFFIResult(&rc);
  }
}
END_TEST

Suite* make_manifest_suite(void) {
  Suite* manifest_s;

  manifest_s = suite_create("FFI manifest interface");
  {
    TCase* manifest_tc;
    manifest_tc = tcase_create("manifest");
    tcase_add_test(manifest_tc, test_manifest_write_read);
    tcase_add_test(manifest_tc, test_manifest_combine);
    tcase_add_test(manifest_tc, test_manifest_add_column);
    suite_add_tcase(manifest_s, manifest_tc);
  }

  return manifest_s;
}