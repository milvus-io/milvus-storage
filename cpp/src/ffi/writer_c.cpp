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
#include "milvus-storage/ffi_internal/bridge.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/util/base64.h>

using namespace milvus_storage::api;
using namespace milvus_storage;

LoonFFIResult loon_writer_new(const char* base_path,
                              ArrowSchema* schema_raw,
                              const ::LoonProperties* properties,
                              LoonWriterHandle* out_handle) {
  if (!base_path || !schema_raw || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: base_path, schema_raw, properties, and out_handle must not be null");
  }
  try {
    milvus_storage::api::Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    if (GetValueNoError<bool>(properties_map, PROPERTY_WRITER_ENC_ENABLE)) {
      // The C properties API carries NUL-terminated text. Decode only after
      // conversion so the native writer receives the complete binary DEK.
      const auto encoded_key = GetValueNoError<std::string>(properties_map, PROPERTY_WRITER_ENC_KEY);
      if (encoded_key.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") !=
          std::string::npos) {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "writer.enc.key must be standard padded Base64");
      }
      auto key = arrow::util::base64_decode(encoded_key);
      // Arrow's decoder accepts prefixes and ignores trailing input. Require a
      // canonical round trip as well as an AES key length; never echo key data.
      if (arrow::util::base64_encode(key) != encoded_key ||
          (key.size() != 16 && key.size() != 24 && key.size() != 32)) {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "writer.enc.key must be Base64 of a 16, 24 or 32 byte AES key");
      }
      properties_map[PROPERTY_WRITER_ENC_KEY] = std::move(key);
    }

    auto schema_result = arrow::ImportSchema(schema_raw);
    RETURN_ARROW_ERROR_IF(schema_result.status(), LOON_ARROW_ERROR, schema_result.status().ToString());
    auto schema = schema_result.ValueOrDie();
    std::unique_ptr<ColumnGroupPolicy> policy;

    auto policy_status = ColumnGroupPolicy::create_column_group_policy(properties_map, schema).Value(&policy);
    RETURN_ARROW_ERROR_IF(policy_status, LOON_ARROW_ERROR, policy_status.ToString());

    auto cpp_writer = Writer::create(std::move(std::string(base_path)), schema, std::move(policy), properties_map);

    auto raw_cpp_writer = reinterpret_cast<LoonWriterHandle>(cpp_writer.release());
    assert(raw_cpp_writer);
    *out_handle = raw_cpp_writer;

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_writer_write(LoonWriterHandle handle, struct ArrowArray* array) {
  if (!handle || !array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and array must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);

    auto rb_result = arrow::ImportRecordBatch(array, cpp_writer->schema());
    if (!rb_result.ok()) {
      array->release(array);
      RETURN_ARROW_ERROR(rb_result.status(), LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto record_batch = rb_result.ValueOrDie();

    auto status = cpp_writer->write(record_batch);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_writer_flush(LoonWriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }
  try {
    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto status = cpp_writer->flush();
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_writer_close(LoonWriterHandle handle,
                                char** meta_keys,
                                char** meta_vals,
                                uint16_t meta_len,
                                LoonColumnGroups** out_column_groups) {
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

      meta_keys_vec.emplace_back(meta_keys[i]);
      meta_vals_vec.emplace_back(meta_vals[i]);
    }

    auto* cpp_writer = reinterpret_cast<Writer*>(handle);
    auto result = cpp_writer->close(meta_keys_vec, meta_vals_vec);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());
    auto cgs = result.ValueOrDie();

    // Export to LoonColumnGroups structure
    auto st = milvus_storage::column_groups_export(*cgs, out_column_groups);
    RETURN_ARROW_ERROR_IF(st, LOON_LOGICAL_ERROR, st.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_free_cstr(char* cstr) {
  if (cstr) {
    free(cstr);
  }
}

void loon_writer_destroy(LoonWriterHandle handle) {
  if (handle) {
    delete reinterpret_cast<Writer*>(handle);
  }
}
