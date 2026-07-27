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

#include "milvus-storage/ffi_filesystem_metrics_c.h"

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/metrics/buckets.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

using ::FileSystemWrapper;
using milvus_storage::Observable;

const int64_t* loon_fs_latency_bucket_bounds_us(int32_t* out_len) {
  if (out_len) {
    *out_len = static_cast<int32_t>(milvus_storage::metrics::kLatencyBoundsUs.size());
  }
  return milvus_storage::metrics::kLatencyBoundsUs.data();
}

const int64_t* loon_fs_size_bucket_bounds_bytes(int32_t* out_len) {
  if (out_len) {
    *out_len = static_cast<int32_t>(milvus_storage::metrics::kSizeBoundsBytes.size());
  }
  return milvus_storage::metrics::kSizeBoundsBytes.data();
}

LoonFFIResult loon_filesystem_get_metrics(FileSystemHandle handle, LoonFilesystemMetricsSnapshot* out_metrics) {
  try {
    if (!handle || !out_metrics) {
      RETURN_ERROR(LOON_INVALID_ARGS, "handle and out_metrics must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();

    auto observable = std::dynamic_pointer_cast<Observable>(fs);
    if (!observable) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Filesystem does not implement Observable interface");
    }

    auto metrics = observable->GetMetrics();
    if (!metrics) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Filesystem metrics are not enabled");
    }

    static_assert(LOON_OP_TYPE_COUNT == milvus_storage::kOpTypeCount, "op type count mismatch");
    static_assert(LOON_STATUS_COUNT == milvus_storage::kStatusCount, "status count mismatch");
    static_assert(LOON_TRANSFER_COUNT == milvus_storage::kTransferCount, "transfer count mismatch");

    auto snapshot = metrics->GetSnapshot();
    for (int i = 0; i < LOON_OP_TYPE_COUNT; ++i) {
      const auto& src = snapshot.ops[i];
      auto& dst = out_metrics->ops[i];
      for (int j = 0; j < LOON_STATUS_COUNT; ++j) {
        dst.count_by_status[j] = src.count_by_status[j];
      }
      dst.retry_count = src.retry_count;
      dst.latency_sum_us = src.latency_sum_us;
      dst.latency_count = src.latency_count;
      for (int j = 0; j < LOON_LATENCY_BUCKETS; ++j) {
        dst.latency_buckets[j] = src.latency_buckets[j];
      }
    }
    for (int i = 0; i < LOON_TRANSFER_COUNT; ++i) {
      out_metrics->transfers[i].bytes_total = snapshot.transfers[i].bytes_total;
      out_metrics->transfers[i].size_sum_bytes = snapshot.transfers[i].size_sum_bytes;
      out_metrics->transfers[i].size_count = snapshot.transfers[i].size_count;
      for (int j = 0; j < LOON_SIZE_BUCKETS; ++j) {
        out_metrics->transfers[i].size_buckets[j] = snapshot.transfers[i].size_buckets[j];
      }
    }
    out_metrics->in_flight = snapshot.in_flight;
    out_metrics->open_connections = snapshot.open_connections;
    out_metrics->idle_connections = snapshot.idle_connections;
    out_metrics->pending_multipart_created = snapshot.pending_multipart_created;
    out_metrics->pending_multipart_finished = snapshot.pending_multipart_finished;

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_reset_metrics(FileSystemHandle handle) {
  try {
    if (!handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "handle must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();

    auto observable = std::dynamic_pointer_cast<Observable>(fs);
    if (!observable) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Filesystem does not implement Observable interface");
    }

    auto metrics = observable->GetMetrics();
    if (!metrics) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Filesystem metrics are not enabled");
    }

    metrics->Reset();

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}
