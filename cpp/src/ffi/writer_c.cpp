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
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/ffi_internal/result.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

using namespace milvus_storage::api;
using namespace milvus_storage;

FFIResult writer_new(const char* base_path,
                     ArrowSchema* schema_raw,
                     const ::Properties* properties,
                     WriterHandle* out_handle) {
  if (!base_path || !schema_raw || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: base_path, schema_raw, properties, and out_handle must not be null");
  }

  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  auto schema_result = arrow::ImportSchema(schema_raw);
  if (!schema_result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, schema_result.status().ToString());
  }
  auto schema = schema_result.ValueOrDie();
  std::unique_ptr<ColumnGroupPolicy> policy;

  auto policy_status = ColumnGroupPolicy::create_column_group_policy(properties_map, schema).Value(&policy);
  if (!policy_status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, policy_status.ToString());
  }

  auto cpp_writer =
      Writer::create(std::move(std::string(base_path)), schema, std::move(policy), std::move(properties_map));

  auto raw_cpp_writer = reinterpret_cast<WriterHandle>(cpp_writer.release());
  assert(raw_cpp_writer);
  *out_handle = raw_cpp_writer;

  RETURN_SUCCESS();
}

FFIResult writer_write(WriterHandle handle, struct ArrowArray* array) {
  if (!handle || !array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and array must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);

    auto rb_result = arrow::ImportRecordBatch(array, cpp_writer->schema());
    if (!rb_result.ok()) {
      array->release(array);
      RETURN_ERROR(LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto record_batch = rb_result.ValueOrDie();

    auto status = cpp_writer->write(record_batch);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult writer_flush(WriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto status = cpp_writer->flush();
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult writer_close(
    WriterHandle handle, char** meta_keys, char** meta_vals, uint16_t meta_len, char** out_columngroups) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  if (meta_len > 0 && (!meta_keys || !meta_vals)) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: meta_keys and meta_vals should not be null when meta_len > 0");
  }

  try {
    std::vector<std::string_view> meta_keys_vec;
    std::vector<std::string_view> meta_vals_vec;

    for (uint16_t i = 0; i < meta_len; ++i) {
      // actually, it's a logical error.
      assert(meta_keys[i] && meta_vals[i]);
      if (!meta_keys[i] || !meta_vals[i]) {
        RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: meta_keys and meta_vals should not be null [index=", i,
                     "]");
      }

      meta_keys_vec.emplace_back(std::string_view(meta_keys[i]));
      meta_vals_vec.emplace_back(std::string_view(meta_vals[i]));
    }

    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto result = cpp_writer->close(meta_keys_vec, meta_vals_vec);
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }
    auto cgs = result.ValueOrDie();

    auto cgsraw_result = cgs->serialize();
    if (!cgsraw_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to serialize column groups to JSON:", cgsraw_result.status().ToString());
    }
    *out_columngroups = strdup(cgsraw_result.ValueOrDie().c_str());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void free_cstr(char* cstr) {
  if (cstr)
    free(cstr);
}

void writer_destroy(WriterHandle handle) {
  if (handle) {
    delete reinterpret_cast<Writer*>(handle);
  }
}
