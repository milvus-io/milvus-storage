// Copyright 2023 Zilliz
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

#include "milvus-storage/ffi_c.h"

#include <memory>
#include <optional>
#include <vector>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/filesystem/fs.h"

// Forward declaration
extern void destroy_column_groups_contents(LoonColumnGroups* cgroups);

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

LoonFFIResult loon_transaction_begin(const char* base_path,
                                     const ::LoonProperties* properties,
                                     int64_t read_version,
                                     uint32_t retry_limit,
                                     LoonTransactionHandle* out_handle) {
  if (!base_path || !properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties must not be null");
  }
  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  // Get filesystem from properties
  auto fs_result = milvus_storage::FilesystemCache::getInstance().get(properties_map);
  if (!fs_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, fs_result.status().ToString());
  }
  auto fs = fs_result.ValueOrDie();

  // Open transaction (automatically begun)
  auto transaction_result = Transaction::Open(fs, base_path, read_version, FailResolver, retry_limit);
  if (!transaction_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, transaction_result.status().ToString());
  }
  auto transaction = std::move(transaction_result.ValueOrDie());

  auto raw_transaction = reinterpret_cast<LoonTransactionHandle>(transaction.release());
  assert(raw_transaction);
  *out_handle = raw_transaction;

  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_commit(LoonTransactionHandle handle, int64_t* out_committed_version) {
  if (!handle || !out_committed_version) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_committed_version must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
  // Commit
  auto commit_result = cpp_transaction->Commit();
  if (!commit_result.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, commit_result.status().ToString());
  }

  *out_committed_version = commit_result.ValueOrDie();
  RETURN_SUCCESS();
}

void loon_transaction_destroy(LoonTransactionHandle handle) {
  if (handle) {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    delete cpp_transaction;
  }
}

LoonFFIResult loon_transaction_get_manifest(LoonTransactionHandle handle, LoonManifest** out_manifest) {
  if (!handle || !out_manifest) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_manifest must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
  auto manifest_result = cpp_transaction->GetManifest();
  if (!manifest_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, manifest_result.status().ToString());
  }
  auto manifest = manifest_result.ValueOrDie();
  // Export manifest to LoonManifest structure
  auto st = milvus_storage::export_manifest(manifest, out_manifest);
  if (!st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, st.ToString());
  }

  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_get_read_version(LoonTransactionHandle handle, int64_t* out_read_version) {
  if (!handle || !out_read_version) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_read_version must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
  *out_read_version = cpp_transaction->GetReadVersion();

  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_add_column_group(LoonTransactionHandle handle, const LoonColumnGroup* column_group) {
  if (!handle || !column_group) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and column_group must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

  // Import LoonColumnGroup to ColumnGroup
  // Create a temporary LoonColumnGroups with one element
  LoonColumnGroups temp_ccgs;
  temp_ccgs.column_group_array = const_cast<LoonColumnGroup*>(column_group);
  temp_ccgs.num_of_column_groups = 1;

  ColumnGroups cgs;
  auto import_st = milvus_storage::import_column_groups(&temp_ccgs, &cgs);
  if (!import_st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, import_st.ToString());
  }

  if (cgs.empty()) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to import column group");
  }

  cpp_transaction->AddColumnGroup(cgs[0]);
  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_append_files(LoonTransactionHandle handle, const LoonColumnGroups* column_groups) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

  // Import LoonColumnGroups to ColumnGroups
  ColumnGroups cgs;
  auto import_st = milvus_storage::import_column_groups(column_groups, &cgs);
  if (!import_st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, import_st.ToString());
  }

  cpp_transaction->AppendFiles(cgs);
  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_add_delta_log(LoonTransactionHandle handle, const char* path, int64_t num_entries) {
  if (!handle || !path) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and path must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

  // Create DeltaLog with hardcoded PRIMARY_KEY type
  DeltaLog delta_log;
  delta_log.path = path;
  delta_log.type = DeltaLogType::PRIMARY_KEY;
  delta_log.num_entries = num_entries;

  cpp_transaction->AddDeltaLog(delta_log);
  RETURN_SUCCESS();
}

LoonFFIResult loon_transaction_update_stat(LoonTransactionHandle handle,
                                           const char* key,
                                           const char* const* files,
                                           size_t files_len) {
  if (!handle || !key || !files) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, key, and files must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

  // Convert C array to vector
  std::vector<std::string> file_vec;
  file_vec.reserve(files_len);
  for (size_t i = 0; i < files_len; ++i) {
    if (files[i]) {
      file_vec.emplace_back(files[i]);
    }
  }

  cpp_transaction->UpdateStat(key, file_vec);
  RETURN_SUCCESS();
}

void loon_manifest_destroy(LoonManifest* cmanifest) {
  if (!cmanifest)
    return;

  // Destroy column groups (embedded structure, not a pointer)
  destroy_column_groups_contents(&cmanifest->column_groups);

  // Destroy delta logs
  if (cmanifest->delta_logs.delta_log_paths) {
    for (uint32_t i = 0; i < cmanifest->delta_logs.num_delta_logs; i++) {
      delete[] const_cast<char*>(cmanifest->delta_logs.delta_log_paths[i]);
    }
    delete[] cmanifest->delta_logs.delta_log_paths;
    cmanifest->delta_logs.delta_log_paths = nullptr;
  }
  if (cmanifest->delta_logs.delta_log_num_entries) {
    delete[] cmanifest->delta_logs.delta_log_num_entries;
    cmanifest->delta_logs.delta_log_num_entries = nullptr;
  }
  cmanifest->delta_logs.num_delta_logs = 0;

  // Destroy stats
  if (cmanifest->stats.stat_keys) {
    for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
      delete[] const_cast<char*>(cmanifest->stats.stat_keys[i]);
    }
    delete[] cmanifest->stats.stat_keys;
    cmanifest->stats.stat_keys = nullptr;
  }
  if (cmanifest->stats.stat_files) {
    for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
      if (cmanifest->stats.stat_files[i]) {
        for (uint32_t j = 0; j < cmanifest->stats.stat_file_counts[i]; j++) {
          delete[] const_cast<char*>(cmanifest->stats.stat_files[i][j]);
        }
        delete[] cmanifest->stats.stat_files[i];
      }
    }
    delete[] cmanifest->stats.stat_files;
    cmanifest->stats.stat_files = nullptr;
  }
  if (cmanifest->stats.stat_file_counts) {
    delete[] cmanifest->stats.stat_file_counts;
    cmanifest->stats.stat_file_counts = nullptr;
  }
  cmanifest->stats.num_stats = 0;

  // Free the structure itself
  delete cmanifest;
}