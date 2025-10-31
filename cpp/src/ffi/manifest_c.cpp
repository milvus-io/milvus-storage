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

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

FFIResult get_latest_column_groups(const char* base_path, const ::Properties* properties, char** out_column_groups) {
  if (!base_path || !properties || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties, and out_column_groups must not be null");
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
  auto serialize_result = latest_manifest->serialize();
  if (!serialize_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, serialize_result.status().ToString());
  }
  auto serialized_str = serialize_result.ValueOrDie();
  *out_column_groups = strdup(serialized_str.c_str());

  RETURN_SUCCESS();
}

FFIResult transaction_begin(const char* base_path, const ::Properties* properties, TransactionHandle* out_handle) {
  if (!base_path || !properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: base_path, properties must not be null");
  }
  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }
  auto transaction = std::make_unique<TransactionImpl<Manifest>>(properties_map, base_path);
  auto status = transaction->begin();
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }
  auto raw_transaction = reinterpret_cast<TransactionHandle>(transaction.release());
  assert(raw_transaction);
  *out_handle = raw_transaction;

  RETURN_SUCCESS();
}

FFIResult transaction_get_column_groups(TransactionHandle* handle, char** out_column_groups) {
  if (!handle || !out_column_groups) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_column_groups must not be null");
  }

  auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
  auto current_manifest_result = cpp_transaction->get_current_manifest();
  if (!current_manifest_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, current_manifest_result.status().ToString());
  }
  auto current_manifest = current_manifest_result.ValueOrDie();
  auto serialize_result = current_manifest->serialize();
  if (!serialize_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, serialize_result.status().ToString());
  }
  auto serialized_str = serialize_result.ValueOrDie();
  *out_column_groups = strdup(serialized_str.c_str());

  RETURN_SUCCESS();
}

FFIResult transaction_commit(
    TransactionHandle* handle, int16_t update_id, int16_t reslove_id, char* in_column_groups, bool* out_commit_result) {
  if (!handle || !out_commit_result) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_commit_result must not be null");
  }

  if (update_id < 0 || update_id >= LOON_TRANSACTION_UPDATE_MAX) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: update_id is invalid [id=", update_id, "]");
  }

  if (reslove_id < 0 || reslove_id >= LOON_TRANSACTION_RESLOVE_MAX) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reslove_id is invalid [id=", reslove_id, "]");
  }

  auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
  if (cpp_transaction->status() != TransStatus::STATUS_COMMITTED) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, "Transaction not committed. [status=", cpp_transaction->status(), "]");
  }

  std::shared_ptr<Manifest> new_manifest = std::make_shared<Manifest>();
  auto desresult = new_manifest->deserialize(in_column_groups);
  if (!desresult.ok()) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize input column groups JSON: ", desresult.ToString());
  }

  auto commit_result = cpp_transaction->commit(new_manifest, static_cast<UpdateType>(update_id),
                                               static_cast<TransResolveStrategy>(reslove_id));
  if (!commit_result.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, commit_result.status().ToString());
  }
  *out_commit_result = commit_result.ValueOrDie();

  RETURN_SUCCESS();
}

FFIResult transaction_abort(TransactionHandle* handle) {
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

void transaction_destroy(TransactionHandle* handle) {
  if (handle) {
    auto* cpp_transaction = reinterpret_cast<TransactionImpl<Manifest>*>(handle);
    delete cpp_transaction;
  }
}