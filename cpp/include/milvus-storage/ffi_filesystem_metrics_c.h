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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "milvus-storage/ffi_filesystem_c.h"

#ifndef LOON_FILESYSTEM_METRICS_C
#define LOON_FILESYSTEM_METRICS_C

/**
 * C structure representing filesystem metrics snapshot.
 */
typedef struct LoonFilesystemMetricsSnapshot {  // NOLINT
  int64_t read_count;
  int64_t write_count;
  int64_t read_bytes;
  int64_t write_bytes;
  int64_t get_file_info_count;
  int64_t create_dir_count;
  int64_t delete_dir_count;
  int64_t delete_file_count;
  int64_t move_count;
  int64_t copy_file_count;
  int64_t failed_count;
  // S3-specific metrics
  int64_t multi_part_upload_created;
  int64_t multi_part_upload_finished;
} LoonFilesystemMetricsSnapshot;  // NOLINT

/**
 * Get metrics from a filesystem handle.
 * Returns metrics if the filesystem is observable, otherwise returns error.
 *
 * @param handle The filesystem instance handle.
 * @param out_metrics The output metrics snapshot structure (caller allocates).
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_metrics(FileSystemHandle handle,
                                                     LoonFilesystemMetricsSnapshot* out_metrics);

/**
 * Reset all metrics for a filesystem.
 *
 * @param handle The filesystem instance handle.
 * @return result of FFI
 */
FFI_EXPORT LoonFFIResult loon_filesystem_reset_metrics(FileSystemHandle handle);

#endif  // LOON_FILESYSTEM_METRICS_C

#ifdef __cplusplus
}
#endif
