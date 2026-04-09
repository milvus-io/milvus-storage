// Copyright 2024 Zilliz
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

#include "ffi_test_env.h"
#include "test_runner.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

bool is_cloud_env(void) {
  const char* storage_type = getenv("TEST_ENV_STORAGE_TYPE");
  return storage_type != NULL && strcmp(storage_type, "remote") == 0;
}

size_t init_test_props(const char** keys, const char** vals, size_t count, size_t capacity, const char* root_path) {
  assert(keys != NULL);
  assert(vals != NULL);
  assert(count < capacity);

#define APPEND_PROP(k, v)     \
  do {                        \
    assert(count < capacity); \
    keys[count] = (k);        \
    vals[count] = (v);        \
    count++;                  \
  } while (0)

  if (is_cloud_env()) {
    const char* cloud_provider = getenv("TEST_ENV_CLOUD_PROVIDER");
    const char* address = getenv("TEST_ENV_ADDRESS");
    const char* bucket_name = getenv("TEST_ENV_BUCKET_NAME");
    const char* region = getenv("TEST_ENV_REGION");
    const char* use_ssl = getenv("TEST_ENV_USE_SSL");
    const char* use_iam = getenv("TEST_ENV_USE_IAM");
    const char* access_key = getenv("TEST_ENV_ACCESS_KEY");
    const char* secret_key = getenv("TEST_ENV_SECRET_KEY");

    APPEND_PROP(loon_properties_fs_storage_type, "remote");

    if (cloud_provider) {
      APPEND_PROP(loon_properties_fs_cloud_provider, cloud_provider);
    }
    if (address) {
      APPEND_PROP(loon_properties_fs_address, address);
    }
    if (bucket_name) {
      APPEND_PROP(loon_properties_fs_bucket_name, bucket_name);
    }
    if (region) {
      APPEND_PROP(loon_properties_fs_region, region);
    }
    if (use_ssl && (strcmp(use_ssl, "true") == 0 || strcmp(use_ssl, "1") == 0)) {
      APPEND_PROP(loon_properties_fs_use_ssl, "true");
    }
    if (use_iam && (strcmp(use_iam, "true") == 0 || strcmp(use_iam, "1") == 0)) {
      APPEND_PROP(loon_properties_fs_use_iam, "true");
    } else {
      if (access_key) {
        APPEND_PROP(loon_properties_fs_access_key_id, access_key);
      }
      if (secret_key) {
        APPEND_PROP(loon_properties_fs_access_key_value, secret_key);
      }
    }
  } else {
    APPEND_PROP(loon_properties_fs_storage_type, "local");
    APPEND_PROP(loon_properties_fs_root_path, root_path);
  }

#undef APPEND_PROP

  return count;
}

void clean_test_dir(FileSystemHandle fs, const char* path) {
  LoonFFIResult rc;
  LoonFileInfoList list = {0};
  uint32_t path_len = (uint32_t)strlen(path);

  rc = loon_filesystem_list_dir(fs, path, path_len, true, &list);
  if (!loon_ffi_is_success(&rc)) {
    // Directory may not exist, that's fine
    loon_ffi_free_result(&rc);
    return;
  }

  // Delete files in reverse order (deepest first)
  for (int i = (int)list.count - 1; i >= 0; i--) {
    if (!list.entries[i].is_dir) {
      rc = loon_filesystem_delete_file(fs, list.entries[i].path, list.entries[i].path_len);
      if (!loon_ffi_is_success(&rc)) {
        loon_ffi_free_result(&rc);
      }
    }
  }

  loon_filesystem_free_file_info_list(&list);
}

void ensure_test_dir(FileSystemHandle fs, const char* path) {
  clean_test_dir(fs, path);

  LoonFFIResult rc = loon_filesystem_create_dir(fs, path, (uint32_t)strlen(path), true);
  ck_assert_msg(loon_ffi_is_success(&rc), "failed to create dir %s: %s", path, loon_ffi_get_errmsg(&rc));
}
