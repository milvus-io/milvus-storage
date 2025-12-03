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

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/transaction/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/filesystem/fs.h"

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

FFIResult get_latest_column_groups(const char* base_path,
                                   const ::Properties* properties,
                                   ColumnGroupsHandle* out_column_groups,
                                   int64_t* read_version) {
  if (!base_path || !properties || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties, out_column_groups must not be null");
  }

  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  auto transaction = std::make_unique<TransactionImpl<Manifest>>(properties_map, base_path);
  auto latest_manifest_result = transaction->get_latest_manifest();
  if (!latest_manifest_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, latest_manifest_result.status().ToString());
  }
  auto latest_manifest = latest_manifest_result.ValueOrDie();
  *out_column_groups = reinterpret_cast<ColumnGroupsHandle>(new std::shared_ptr<Manifest>(latest_manifest));

  // fill read_version if the pointer is provided
  if (read_version) {
    *read_version = transaction->read_version();
    assert(*read_version >= 0);
  }

  RETURN_SUCCESS();
}

FFIResult get_column_groups_by_version(const char* base_path,
                                       const ::Properties* properties,
                                       int64_t read_version,
                                       ColumnGroupsHandle* out_column_groups) {
  if (!base_path || !properties || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties, out_column_groups must not be null");
  }
  // no need verify read_version here, let transaction handle it

  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  auto transaction = std::make_unique<TransactionImpl<Manifest>>(properties_map, base_path);
  auto begin_status = transaction->begin(read_version);
  if (!begin_status.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, begin_status.ToString());
  }
  auto manifest_result = transaction->get_current_manifest();
  if (!manifest_result.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, manifest_result.status().ToString());
  }

  auto cur_manifest = manifest_result.ValueOrDie();
  *out_column_groups = reinterpret_cast<ColumnGroupsHandle>(new std::shared_ptr<Manifest>(cur_manifest));

  // abort the transaction after get manifest
  auto abort_result = transaction->abort();
  if (!abort_result.ok()) {
    free(out_column_groups);
    *out_column_groups = NULL;
    RETURN_ERROR(LOON_LOGICAL_ERROR, abort_result.ToString());
  }

  RETURN_SUCCESS();
}

FFIResult transaction_begin(const char* base_path,
                            const ::Properties* properties,
                            TransactionHandle* out_handle,
                            int64_t read_version) {
  if (!base_path || !properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties must not be null");
  }
  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }
  auto transaction = std::make_unique<TransactionImpl<Manifest>>(properties_map, base_path);
  auto status = transaction->begin(read_version);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }
  auto raw_transaction = reinterpret_cast<TransactionHandle>(transaction.release());
  assert(raw_transaction);
  *out_handle = raw_transaction;

  RETURN_SUCCESS();
}

FFIResult transaction_get_column_groups(TransactionHandle handle, ColumnGroupsHandle* out_column_groups) {
  if (!handle || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_column_groups must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
  auto current_manifest_result = cpp_transaction->get_current_manifest();
  if (!current_manifest_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, current_manifest_result.status().ToString());
  }
  auto current_manifest = current_manifest_result.ValueOrDie();
  *out_column_groups = reinterpret_cast<ColumnGroupsHandle>(new std::shared_ptr<Manifest>(current_manifest));

  RETURN_SUCCESS();
}

FFIResult transaction_commit(TransactionHandle handle,
                             int16_t update_id,
                             int16_t resolve_id,
                             ColumnGroupsHandle in_column_groups,
                             TransactionCommitResult* out_commit_result) {
  if (!handle || !out_commit_result) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_commit_result must not be null");
  }

  if (update_id < 0 || update_id >= LOON_TRANSACTION_UPDATE_MAX) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: update_id is invalid [id=", update_id, "]");
  }

  if (resolve_id < 0 || resolve_id >= LOON_TRANSACTION_RESOLVE_MAX) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: resolve_id is invalid [id=", resolve_id, "]");
  }

  auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
  auto* cg_ptr = reinterpret_cast<std::shared_ptr<Manifest>*>(in_column_groups);
  std::shared_ptr<Manifest> new_manifest = *cg_ptr;

  auto commit_result = cpp_transaction->commit(new_manifest, static_cast<UpdateType>(update_id),
                                               static_cast<TransResolveStrategy>(resolve_id));
  if (!commit_result.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, commit_result.status().ToString());
  }

  auto commit_result_cpp = commit_result.ValueOrDie();
  out_commit_result->success = commit_result_cpp.success;
  out_commit_result->committed_version = commit_result_cpp.committed_version;
  out_commit_result->read_version = commit_result_cpp.read_version;
  out_commit_result->failed_message =
      !commit_result_cpp.failed_message.empty() ? strdup(commit_result_cpp.failed_message.c_str()) : nullptr;

  RETURN_SUCCESS();
}

FFIResult transaction_abort(TransactionHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
  auto status = cpp_transaction->abort();
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  RETURN_SUCCESS();
}

void transaction_destroy(TransactionHandle handle) {
  if (handle) {
    auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
    delete cpp_transaction;
  }
}

void close_filesystems() {
  auto& fs_cache = milvus_storage::LRUCache<milvus_storage::ArrowFileSystemConfig,
                                            milvus_storage::ArrowFileSystemPtr>::getInstance();
  fs_cache.clean();
}