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
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

using ::FileSystemWrapper;
using milvus_storage::Observable;

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

    auto snapshot = metrics->GetSnapshot();
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
