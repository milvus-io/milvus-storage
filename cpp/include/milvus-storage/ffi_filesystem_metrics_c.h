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
 * One named metrics snapshot. Strings are owned by the returned result.
 * display_key is NULL for handle queries and populated for cache enumeration.
 */
typedef struct LoonFilesystemMetricsSourceEntry {  // NOLINT
  char* display_key;
  char* source;
  LoonFilesystemMetricsSnapshot metrics;
} LoonFilesystemMetricsSourceEntry;  // NOLINT

typedef struct LoonFilesystemMetricsSources {  // NOLINT
  LoonFilesystemMetricsSourceEntry* entries;
  uint32_t count;
} LoonFilesystemMetricsSources;  // NOLINT

/**
 * Get all metrics sources from a filesystem handle, including uncached handles.
 * Each display_key is NULL. An empty source collection is a successful result.
 * Snapshot fields are sampled independently, not as a transaction.
 * out_sources must not own an existing result; release returned results using
 * loon_filesystem_free_metrics_sources.
 */
FFI_EXPORT LoonFFIResult loon_filesystem_get_metrics_sources(FileSystemHandle handle,
                                                             LoonFilesystemMetricsSources* out_sources);

/**
 * List all metrics sources from cached filesystems. Each entry includes a
 * storage-generated display_key; multiple sources may share that key.
 * Results own their strings and snapshots independently of the filesystem cache.
 * Snapshot fields are sampled independently, not as a transaction.
 * out_sources must not own an existing result; release returned results using
 * loon_filesystem_free_metrics_sources.
 */
FFI_EXPORT LoonFFIResult loon_filesystem_list_metrics_sources(LoonFilesystemMetricsSources* out_sources);

/**
 * Release a metrics sources result and clear its entries/count.
 * NULL and repeated calls on the cleared result are safe.
 */
FFI_EXPORT void loon_filesystem_free_metrics_sources(LoonFilesystemMetricsSources* sources);

#endif  // LOON_FILESYSTEM_METRICS_C

#ifdef __cplusplus
}
#endif
