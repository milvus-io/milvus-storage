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

#ifndef FFI_TEST_ENV_H
#define FFI_TEST_ENV_H

#include <stdbool.h>
#include <stddef.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_filesystem_c.h"

/// Default root path for local filesystem tests.
#define FFI_TEST_ROOT_PATH "/tmp"

/// Returns true if STORAGE_TYPE env var is "remote".
bool is_cloud_env(void);

/// Append filesystem-related properties into the given key/value arrays.
/// In cloud mode, reads env vars (CLOUD_PROVIDER, ADDRESS, BUCKET_NAME, etc.).
/// In local mode, uses root_path with fs.storage_type=local.
///
/// Usage:
///   const char* keys[500] = { "writer.policy" };
///   const char* vals[500] = { "single" };
///   size_t n = init_test_props(keys, vals, 1, 500, FFI_TEST_ROOT_PATH);
///   loon_properties_create(keys, vals, n, &pp);
///
/// @param keys      Array with existing property keys (fs props will be appended)
/// @param vals      Array with existing property values
/// @param offset    Number of entries already in keys/vals
/// @param capacity  Total capacity of keys/vals arrays
/// @param root_path Root path for local mode
/// @return Total count of entries (offset + number of fs props appended)
size_t init_test_props(const char** keys, const char** vals, size_t offset, size_t capacity, const char* root_path);

/// Delete all files under the given path (recursively), then remove the directory itself.
/// Ignores errors if the path does not exist.
void clean_test_dir(FileSystemHandle fs, const char* path);

/// Clean and re-create a directory. Asserts on failure.
void ensure_test_dir(FileSystemHandle fs, const char* path);

#endif  // FFI_TEST_ENV_H
