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

#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/extend_status.h"

// Forward declaration
extern void destroy_column_groups_contents(LoonColumnGroups* cgroups);

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

LoonFFIResult loon_transaction_begin(const char* base_path,
                                     const ::LoonProperties* properties,
                                     int64_t read_version,
                                     int32_t resolve_id,
                                     uint32_t retry_limit,
                                     LoonTransactionHandle* out_handle) {
  RETURN_ERROR_IF(!base_path || !properties || !out_handle, LOON_INVALID_ARGS,
                  "Invalid arguments: base_path, properties, and out_handle must not be null");
  try {
    milvus_storage::api::Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    RETURN_ERROR_IF(opt != std::nullopt, LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");

    // Select resolver based on resolve_id
    const Resolver* resolver = nullptr;
    switch (resolve_id) {
      case LOON_TRANSACTION_RESOLVE_OVERWRITE:
        resolver = &OverwriteResolver;
        break;
      case LOON_TRANSACTION_RESOLVE_FAIL:
      default:
        resolver = &FailResolver;
        break;
    }

    // Open transaction
    auto fs_result = milvus_storage::FilesystemCache::getInstance().get(properties_map);
    RETURN_ERROR_IF(!fs_result.ok(), LOON_ARROW_ERROR, fs_result.status().ToString());
    auto transaction_result =
        Transaction::Open(fs_result.ValueOrDie(), base_path, read_version, *resolver, retry_limit);
    RETURN_ERROR_IF(!transaction_result.ok(), LOON_ARROW_ERROR, transaction_result.status().ToString());
    auto transaction = std::move(transaction_result.ValueOrDie());

    auto raw_transaction = reinterpret_cast<LoonTransactionHandle>(transaction.release());
    assert(raw_transaction);
    *out_handle = raw_transaction;

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_commit(LoonTransactionHandle handle, int64_t* out_committed_version) {
  RETURN_ERROR_IF(!handle || !out_committed_version, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and out_committed_version must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    // Commit
    auto commit_result = cpp_transaction->Commit();
    if (!commit_result.ok()) {
      auto detail = milvus_storage::ExtendStatusDetail::UnwrapStatus(commit_result.status());
      if (detail) {
        switch (detail->code()) {
          case milvus_storage::ExtendStatusCode::TxnExhaustedRetry:
            RETURN_ERROR(LOON_TXN_EXHAUSTED_RETRY, commit_result.status().ToString());
          case milvus_storage::ExtendStatusCode::TxnResolutionFailed:
            RETURN_ERROR(LOON_TXN_RESOLUTION_FAILED, commit_result.status().ToString());
          default:
            break;
        }
      }
      RETURN_ERROR(LOON_LOGICAL_ERROR, commit_result.status().ToString());
    }

    *out_committed_version = commit_result.ValueOrDie();
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_transaction_destroy(LoonTransactionHandle handle) {
  if (handle) {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    delete cpp_transaction;
  }
}

LoonFFIResult loon_transaction_get_manifest(LoonTransactionHandle handle, LoonManifest** out_manifest) {
  RETURN_ERROR_IF(!handle || !out_manifest, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and out_manifest must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    auto manifest_result = cpp_transaction->GetManifest();
    RETURN_ERROR_IF(!manifest_result.ok(), LOON_ARROW_ERROR, manifest_result.status().ToString());
    auto manifest = manifest_result.ValueOrDie();
    // Export manifest to LoonManifest structure
    auto st = milvus_storage::manifest_export(manifest, out_manifest);
    RETURN_ERROR_IF(!st.ok(), LOON_LOGICAL_ERROR, st.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_get_read_version(LoonTransactionHandle handle, int64_t* out_read_version) {
  RETURN_ERROR_IF(!handle || !out_read_version, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and out_read_version must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    RETURN_ERROR_IF(!cpp_transaction, LOON_INVALID_ARGS, "Invalid arguments: transaction handle must not be null");
    *out_read_version = cpp_transaction->GetReadVersion();

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_drop_column(LoonTransactionHandle handle, const char* column_name) {
  RETURN_ERROR_IF(!handle || !column_name, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and column_name must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);
    cpp_transaction->DropColumn(column_name);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_add_column_group(LoonTransactionHandle handle, const LoonColumnGroup* column_group) {
  RETURN_ERROR_IF(!handle || !column_group, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and column_group must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

    // Import LoonColumnGroup to ColumnGroup
    // Create a temporary LoonColumnGroups with one element
    LoonColumnGroups temp_ccgs;
    temp_ccgs.column_group_array = const_cast<LoonColumnGroup*>(column_group);
    temp_ccgs.num_of_column_groups = 1;

    ColumnGroups cgs;
    auto import_st = milvus_storage::column_groups_import(&temp_ccgs, &cgs);
    RETURN_ERROR_IF(!import_st.ok(), import_st.IsInvalid() ? LOON_INVALID_ARGS : LOON_LOGICAL_ERROR,
                    import_st.ToString());

    RETURN_ERROR_IF(cgs.empty(), LOON_INVALID_ARGS, "Failed to import column group");

    cpp_transaction->AddColumnGroup(cgs[0]);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_append_files(LoonTransactionHandle handle, const LoonColumnGroups* column_groups) {
  RETURN_ERROR_IF(!handle || !column_groups, LOON_INVALID_ARGS,
                  "Invalid arguments: handle and column_groups must not be null");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

    // Import LoonColumnGroups to ColumnGroups
    ColumnGroups cgs;
    auto import_st = milvus_storage::column_groups_import(column_groups, &cgs);
    RETURN_ERROR_IF(!import_st.ok(), import_st.IsInvalid() ? LOON_INVALID_ARGS : LOON_LOGICAL_ERROR,
                    import_st.ToString());

    cpp_transaction->AppendFiles(cgs);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_add_delta_log(LoonTransactionHandle handle, const char* path, int64_t num_entries) {
  RETURN_ERROR_IF(!handle || !path, LOON_INVALID_ARGS, "Invalid arguments: handle and path must not be null");
  RETURN_ERROR_IF(num_entries <= 0, LOON_INVALID_ARGS, "Invalid arguments: num_entries must be positive");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

    // Create DeltaLog with hardcoded PRIMARY_KEY type
    DeltaLog delta_log;
    delta_log.path = path;
    delta_log.type = DeltaLogType::PRIMARY_KEY;
    delta_log.num_entries = num_entries;

    cpp_transaction->AddDeltaLog(delta_log);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_update_stat(LoonTransactionHandle handle,
                                           const char* key,
                                           const char* const* files,
                                           size_t files_len,
                                           const char* const* metadata_keys,
                                           const char* const* metadata_values,
                                           size_t metadata_len) {
  RETURN_ERROR_IF(!handle || !key || !files, LOON_INVALID_ARGS,
                  "Invalid arguments: handle, key, and files must not be null");
  RETURN_ERROR_IF(metadata_len > 0 && (!metadata_keys || !metadata_values), LOON_INVALID_ARGS,
                  "Invalid arguments: metadata_keys and metadata_values must not be null when metadata_len > 0");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

    Statistics stat;
    stat.paths.reserve(files_len);
    for (size_t i = 0; i < files_len; ++i) {
      RETURN_ERROR_IF(!files[i], LOON_INVALID_ARGS, "Invalid arguments: files entries must not be null [index=", i,
                      "]");
      stat.paths.emplace_back(files[i]);
    }
    for (size_t i = 0; i < metadata_len; ++i) {
      RETURN_ERROR_IF(!metadata_keys[i] || !metadata_values[i], LOON_INVALID_ARGS,
                      "Invalid arguments: metadata entries must not be null [index=", i, "]");
      stat.metadata[metadata_keys[i]] = metadata_values[i];
    }

    cpp_transaction->UpdateStat(key, stat);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_transaction_add_lob_file(LoonTransactionHandle handle, const LoonLobFileInfo* lob_file) {
  RETURN_ERROR_IF(!handle || !lob_file, LOON_INVALID_ARGS, "Invalid arguments: handle and lob_file must not be null");
  RETURN_ERROR_IF(!lob_file->path, LOON_INVALID_ARGS, "Invalid arguments: lob_file.path must not be null");
  RETURN_ERROR_IF(lob_file->total_rows < 0 || lob_file->valid_rows < 0 || lob_file->file_size_bytes < 0,
                  LOON_INVALID_ARGS, "Invalid arguments: LOB row counts and file size must be non-negative");
  RETURN_ERROR_IF(lob_file->valid_rows > lob_file->total_rows, LOON_INVALID_ARGS,
                  "Invalid arguments: LOB valid_rows must not exceed total_rows");
  try {
    auto* cpp_transaction = reinterpret_cast<Transaction*>(handle);

    // Convert C struct to C++ struct
    LobFileInfo cpp_lob_file;
    cpp_lob_file.path = lob_file->path;
    cpp_lob_file.field_id = lob_file->field_id;
    cpp_lob_file.total_rows = lob_file->total_rows;
    cpp_lob_file.valid_rows = lob_file->valid_rows;
    cpp_lob_file.file_size_bytes = lob_file->file_size_bytes;

    cpp_transaction->AddLobFile(cpp_lob_file);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_manifest_destroy(LoonManifest* cmanifest) {
  if (!cmanifest) {
    return;
  }

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
  // Destroy stats metadata
  if (cmanifest->stats.stat_metadata_keys) {
    for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
      if (cmanifest->stats.stat_metadata_keys[i]) {
        for (uint32_t j = 0; j < cmanifest->stats.stat_metadata_counts[i]; j++) {
          delete[] const_cast<char*>(cmanifest->stats.stat_metadata_keys[i][j]);
        }
        delete[] cmanifest->stats.stat_metadata_keys[i];
      }
    }
    delete[] cmanifest->stats.stat_metadata_keys;
    cmanifest->stats.stat_metadata_keys = nullptr;
  }
  if (cmanifest->stats.stat_metadata_values) {
    for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
      if (cmanifest->stats.stat_metadata_values[i]) {
        for (uint32_t j = 0; j < cmanifest->stats.stat_metadata_counts[i]; j++) {
          delete[] const_cast<char*>(cmanifest->stats.stat_metadata_values[i][j]);
        }
        delete[] cmanifest->stats.stat_metadata_values[i];
      }
    }
    delete[] cmanifest->stats.stat_metadata_values;
    cmanifest->stats.stat_metadata_values = nullptr;
  }
  if (cmanifest->stats.stat_metadata_counts) {
    delete[] cmanifest->stats.stat_metadata_counts;
    cmanifest->stats.stat_metadata_counts = nullptr;
  }
  cmanifest->stats.num_stats = 0;

  // Destroy LOB files
  if (cmanifest->lob_files.files) {
    for (uint32_t i = 0; i < cmanifest->lob_files.num_files; i++) {
      delete[] const_cast<char*>(cmanifest->lob_files.files[i].path);
    }
    delete[] cmanifest->lob_files.files;
    cmanifest->lob_files.files = nullptr;
  }
  cmanifest->lob_files.num_files = 0;

  // Free the structure itself
  delete cmanifest;
}

char* loon_manifest_debug_string(const LoonManifest* manifest) {
  std::string result = milvus_storage::manifest_debug_string(manifest);
  return strdup(result.c_str());
}

#ifdef BUILD_GTEST
void loon_reset_context(void) {
  milvus_storage::api::Manifest::CleanCache();
  milvus_storage::FilesystemCache::getInstance().clean();
}
#endif
