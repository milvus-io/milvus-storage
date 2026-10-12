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
#include <arrow/c/helpers.h>
#include <arrow/array.h>
#include <arrow/util/bitmap_ops.h>

using namespace milvus_storage::api;
using namespace milvus_storage;

namespace {
struct OwnedArrowArray {
  ArrowArray array{};
  OwnedArrowArray() = default;
  OwnedArrowArray(const OwnedArrowArray&) = delete;
  OwnedArrowArray& operator=(const OwnedArrowArray&) = delete;
  ~OwnedArrowArray() { ArrowArrayRelease(&array); }
};

// Import columns as separate owners: a cached slice of one column must not keep
// the producer's entire exported struct alive through Arrow's parent import.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> ImportWriterBatch(ArrowArray* array,
                                                                     const std::shared_ptr<arrow::Schema>& schema) {
  if (ArrowArrayIsReleased(array)) {
    return arrow::Status::Invalid("Cannot import released ArrowArray");
  }
  OwnedArrowArray root;
  ArrowArrayMove(array, &root.array);
  const auto& batch = root.array;
  if (batch.length < 0 || batch.offset != 0 || batch.n_buffers != 1 || !batch.buffers || batch.dictionary ||
      batch.n_children != schema->num_fields() || (batch.n_children > 0 && !batch.children)) {
    return arrow::Status::Invalid("Invalid ArrowArray record batch layout");
  }
  auto null_count = batch.null_count;
  if (null_count == -1) {
    null_count = batch.buffers[0] ? batch.length - arrow::internal::CountSetBits(
                                                       static_cast<const uint8_t*>(batch.buffers[0]), 0, batch.length)
                                  : 0;
  }
  if (null_count != 0) {
    return arrow::Status::Invalid("ArrowArray record batch must not contain null rows");
  }
  const auto length = batch.length;
  for (int i = 0; i < schema->num_fields(); ++i) {
    if (!batch.children[i] || ArrowArrayIsReleased(batch.children[i]) || batch.children[i]->length < length) {
      return arrow::Status::Invalid("Invalid ArrowArray record batch child at index ", i);
    }
  }

  std::vector<OwnedArrowArray> children(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    ArrowArrayMove(batch.children[i], &children[i].array);
  }
  // The C Data Interface requires releasing the parent immediately after moving
  // its children. Each moved child now owns its own release callback.
  ArrowArrayRelease(&root.array);

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto column, arrow::ImportArray(&children[i].array, schema->field(i)->type()));
    columns.emplace_back(column->Slice(0, length));
  }
  auto record = arrow::RecordBatch::Make(schema, length, std::move(columns));
  ARROW_RETURN_NOT_OK(record->Validate());
  return record;
}
}  // namespace

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

    auto rb_result = ImportWriterBatch(array, cpp_writer->schema());
    if (!rb_result.ok()) {
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
