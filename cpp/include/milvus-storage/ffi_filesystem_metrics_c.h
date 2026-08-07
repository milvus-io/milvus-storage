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

// Cardinalities of the labeled metrics registry. Mirror the C++ enums in
// milvus-storage/filesystem/metrics/op_type.h and buckets.h.
#define LOON_OP_TYPE_COUNT 15
#define LOON_STATUS_COUNT 9
#define LOON_LATENCY_BUCKETS 16
#define LOON_SIZE_BUCKETS 14
#define LOON_TRANSFER_COUNT 3  // Read=0, Write=1, MultipartUploadPart=2

// Indices into LoonFilesystemMetricsSnapshot.ops[] (mirror milvus_storage::OpType).
enum {
  LOON_OP_READ = 0,
  LOON_OP_WRITE,
  LOON_OP_LIST,
  LOON_OP_HEAD,
  LOON_OP_OPEN_INPUT,
  LOON_OP_OPEN_OUTPUT,
  LOON_OP_CREATE_DIR,
  LOON_OP_DELETE_DIR,
  LOON_OP_DELETE_FILE,
  LOON_OP_MOVE,
  LOON_OP_COPY,
  LOON_OP_MULTIPART_CREATE,
  LOON_OP_MULTIPART_UPLOAD_PART,
  LOON_OP_MULTIPART_COMPLETE,
  LOON_OP_MULTIPART_ABORT
};

// Indices into LoonOpStats.count_by_status[] (mirror milvus_storage::OpStatus).
enum {
  LOON_STATUS_OK = 0,
  LOON_STATUS_NOT_FOUND,
  LOON_STATUS_THROTTLED,
  LOON_STATUS_AUTH,
  LOON_STATUS_TIMEOUT,
  LOON_STATUS_NETWORK,
  LOON_STATUS_SERVER_ERROR,
  LOON_STATUS_CLIENT_ERROR,
  LOON_STATUS_UNKNOWN
};

// Indices into LoonFilesystemMetricsSnapshot.transfers[].
enum { LOON_XFER_READ = 0, LOON_XFER_WRITE, LOON_XFER_MULTIPART_UPLOAD_PART };

/**
 * Per-operation statistics (indexed by OpType).
 */
typedef struct LoonOpStats {  // NOLINT
  int64_t count_by_status[LOON_STATUS_COUNT];
  int64_t retry_count;
  int64_t latency_sum_us;
  int64_t latency_count;
  int64_t latency_buckets[LOON_LATENCY_BUCKETS];
} LoonOpStats;  // NOLINT

/**
 * Per-transfer-op payload statistics (Read, Write, MultipartUploadPart).
 */
typedef struct LoonTransferStats {  // NOLINT
  int64_t bytes_total;
  int64_t size_sum_bytes;
  int64_t size_count;
  int64_t size_buckets[LOON_SIZE_BUCKETS];
} LoonTransferStats;  // NOLINT

/**
 * Fixed-size labeled filesystem metrics snapshot (caller allocates).
 */
typedef struct LoonFilesystemMetricsSnapshot {  // NOLINT
  LoonOpStats ops[LOON_OP_TYPE_COUNT];
  LoonTransferStats transfers[LOON_TRANSFER_COUNT];
  int64_t in_flight;
  int64_t open_connections;
  int64_t idle_connections;
  int64_t pending_multipart_created;
  int64_t pending_multipart_finished;
} LoonFilesystemMetricsSnapshot;  // NOLINT

/**
 * Static latency bucket upper bounds, microseconds. The returned pointer is
 * owned by the library; do not free. Writes the number of bounds to out_len.
 */
FFI_EXPORT const int64_t* loon_fs_latency_bucket_bounds_us(int32_t* out_len);

/**
 * Static size bucket upper bounds, bytes. The returned pointer is owned by the
 * library; do not free. Writes the number of bounds to out_len.
 */
FFI_EXPORT const int64_t* loon_fs_size_bucket_bounds_bytes(int32_t* out_len);

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
