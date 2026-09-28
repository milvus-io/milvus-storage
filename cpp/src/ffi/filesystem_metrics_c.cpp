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
#include <limits>

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

// The zero-initialized result owns each allocation immediately, including partial entries.
static bool FillMetricsSource(const std::string& source,
                              const FilesystemMetrics& metrics,
                              const char* display_key,
                              LoonFilesystemMetricsSourceEntry* entry) {
  entry->source = strdup(source.c_str());
  if (!entry->source) {
    return false;
  }
  if (display_key) {
    entry->display_key = strdup(display_key);
    if (!entry->display_key) {
      return false;
    }
  }
  FillMetricsSnapshot(metrics.GetSnapshot(), &entry->metrics);
  return true;
}

void loon_filesystem_free_metrics_sources(LoonFilesystemMetricsSources* sources) {
  if (!sources) {
    return;
  }
  if (sources->entries) {
    for (uint32_t i = 0; i < sources->count; ++i) {
      free(sources->entries[i].display_key);
      free(sources->entries[i].source);
    }
    free(sources->entries);
  }
  sources->entries = nullptr;
  sources->count = 0;
}

LoonFFIResult loon_filesystem_get_metrics_sources(FileSystemHandle handle, LoonFilesystemMetricsSources* out_sources) {
  if (!out_sources) {
    RETURN_ERROR(LOON_INVALID_ARGS, "out_sources must not be null");
  }
  *out_sources = {};
  try {
    if (!handle) {
      RETURN_ERROR(LOON_INVALID_ARGS, "handle must not be null");
    }
    const auto fs = reinterpret_cast<FileSystemWrapper*>(handle)->get();
    const auto observable = std::dynamic_pointer_cast<Observable>(fs);
    if (!observable) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Filesystem does not implement Observable interface");
    }
    const auto sources = observable->GetMetricsSources();
    if (sources.empty()) {
      RETURN_SUCCESS();
    }
    if (sources.size() > std::numeric_limits<uint32_t>::max() ||
        sources.size() > std::numeric_limits<size_t>::max() / sizeof(LoonFilesystemMetricsSourceEntry)) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Too many filesystem metrics sources");
    }
    out_sources->entries = static_cast<LoonFilesystemMetricsSourceEntry*>(
        calloc(sources.size(), sizeof(LoonFilesystemMetricsSourceEntry)));
    if (!out_sources->entries) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate filesystem metrics sources");
    }
    out_sources->count = static_cast<uint32_t>(sources.size());
    size_t entry_index = 0;
    for (const auto& [source, metrics] : sources) {
      if (!FillMetricsSource(source, *metrics, nullptr, &out_sources->entries[entry_index++])) {
        loon_filesystem_free_metrics_sources(out_sources);
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to duplicate filesystem metrics source");
      }
    }
    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    loon_filesystem_free_metrics_sources(out_sources);
    RETURN_EXCEPTION(e.what());
  }
  RETURN_UNREACHABLE();
}

LoonFFIResult loon_filesystem_list_metrics_sources(LoonFilesystemMetricsSources* out_sources) {
  if (!out_sources) {
    RETURN_ERROR(LOON_INVALID_ARGS, "out_sources must not be null");
  }
  *out_sources = {};
  try {
    const auto filesystems = FilesystemCache::getInstance().list();
    std::vector<std::unordered_map<std::string, std::shared_ptr<FilesystemMetrics>>> all_sources;
    all_sources.reserve(filesystems.size());
    size_t count = 0;
    for (const auto& [display_key, fs] : filesystems) {
      const auto observable = std::dynamic_pointer_cast<Observable>(fs);
      if (!observable) {
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Cached filesystem does not implement Observable interface");
      }
      auto sources = observable->GetMetricsSources();
      if (sources.size() > std::numeric_limits<uint32_t>::max() - count) {
        RETURN_ERROR(LOON_LOGICAL_ERROR, "Too many filesystem metrics sources");
      }
      count += sources.size();
      all_sources.push_back(std::move(sources));
    }
    if (count == 0) {
      RETURN_SUCCESS();
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(LoonFilesystemMetricsSourceEntry)) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Filesystem metrics sources allocation size overflow");
    }
    out_sources->entries =
        static_cast<LoonFilesystemMetricsSourceEntry*>(calloc(count, sizeof(LoonFilesystemMetricsSourceEntry)));
    if (!out_sources->entries) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to allocate filesystem metrics sources");
    }
    out_sources->count = static_cast<uint32_t>(count);
    size_t entry_index = 0;
    for (size_t i = 0; i < filesystems.size(); ++i) {
      for (const auto& [source, metrics] : all_sources[i]) {
        if (!FillMetricsSource(source, *metrics, filesystems[i].first.c_str(), &out_sources->entries[entry_index++])) {
          loon_filesystem_free_metrics_sources(out_sources);
          RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to duplicate filesystem metrics source or display key");
        }
      }
    }
    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    loon_filesystem_free_metrics_sources(out_sources);
    RETURN_EXCEPTION(e.what());
  }
  RETURN_UNREACHABLE();
}
