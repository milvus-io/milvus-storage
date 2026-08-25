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

#include <cstdlib>
#include <cstring>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

using ::FileSystemWrapper;
using milvus_storage::FilesystemCache;
using milvus_storage::FilesystemMetrics;
using milvus_storage::Observable;

static void FillMetricsSnapshot(const FilesystemMetrics::MetricsSnapshot& snapshot,
                                LoonFilesystemMetricsSnapshot* out_metrics) {
  out_metrics->read_count = snapshot.read_count;
  out_metrics->write_count = snapshot.write_count;
  out_metrics->read_bytes = snapshot.read_bytes;
  out_metrics->write_bytes = snapshot.write_bytes;
  out_metrics->get_file_info_count = snapshot.get_file_info_count;
  out_metrics->create_dir_count = snapshot.create_dir_count;
  out_metrics->delete_dir_count = snapshot.delete_dir_count;
  out_metrics->delete_file_count = snapshot.delete_file_count;
  out_metrics->move_count = snapshot.move_count;
  out_metrics->copy_file_count = snapshot.copy_file_count;
  out_metrics->failed_count = snapshot.failed_count;
  out_metrics->multi_part_upload_created = snapshot.multi_part_upload_created;
  out_metrics->multi_part_upload_finished = snapshot.multi_part_upload_finished;
}

LoonFFIResult loon_filesystem_get_metrics(FileSystemHandle handle, LoonFilesystemMetricsSnapshot* out_metrics) {
  try {
    if (!handle || !out_metrics) {
      RETURN_ERROR(LOON_INVALID_ARGS, "handle and out_metrics must not be null");
    }

    auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();

    auto observable = std::dynamic_pointer_cast<Observable>(fs);
    if (!observable) {
      RETURN_ERROR(LOON_NOT_SUPPORT, "Filesystem does not implement Observable interface");
    }

    auto metrics = observable->GetMetrics();
    if (!metrics) {
      RETURN_ERROR(LOON_NOT_SUPPORT, "Filesystem metrics are not enabled");
    }

    FillMetricsSnapshot(metrics->GetSnapshot(), out_metrics);

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

void loon_filesystem_free_metrics_list(LoonFilesystemMetricsList* list) {
  if (!list) {
    return;
  }
  if (list->entries) {
    for (uint32_t i = 0; i < list->count; ++i) {
      free(list->entries[i].display_key);
    }
    free(list->entries);
    list->entries = nullptr;
  }
  list->count = 0;
}

LoonFFIResult loon_filesystem_list_metrics(LoonFilesystemMetricsList* out_list) {
  try {
    if (!out_list) {
      RETURN_ERROR(LOON_INVALID_ARGS, "out_list must not be null");
    }

    out_list->entries = nullptr;
    out_list->count = 0;

    auto filesystems = FilesystemCache::getInstance().list();
    if (filesystems.empty()) {
      RETURN_SUCCESS();
    }

    auto count = static_cast<uint32_t>(filesystems.size());
    out_list->entries = static_cast<LoonFilesystemMetricsEntry*>(calloc(count, sizeof(LoonFilesystemMetricsEntry)));
    if (!out_list->entries) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate memory for filesystem metrics list");
    }
    out_list->count = count;

    for (uint32_t i = 0; i < count; ++i) {
      const auto& [display_key, fs] = filesystems[i];
      auto observable = std::dynamic_pointer_cast<Observable>(fs);
      if (!observable) {
        loon_filesystem_free_metrics_list(out_list);
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Cached filesystem does not implement Observable interface");
      }

      auto metrics = observable->GetMetrics();
      if (!metrics) {
        loon_filesystem_free_metrics_list(out_list);
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Cached filesystem metrics are not enabled");
      }

      auto* entry = &out_list->entries[i];
      entry->display_key = strdup(display_key.c_str());
      if (!entry->display_key) {
        loon_filesystem_free_metrics_list(out_list);
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to duplicate filesystem display key");
      }
      FillMetricsSnapshot(metrics->GetSnapshot(), &entry->metrics);
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    if (out_list) {
      loon_filesystem_free_metrics_list(out_list);
    }
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
      RETURN_ERROR(LOON_NOT_SUPPORT, "Filesystem does not implement Observable interface");
    }

    auto metrics = observable->GetMetrics();
    if (!metrics) {
      RETURN_ERROR(LOON_NOT_SUPPORT, "Filesystem metrics are not enabled");
    }

    metrics->Reset();

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}
