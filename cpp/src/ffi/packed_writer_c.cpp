// Copyright 2025 Zilliz
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

#include "milvus-storage/packed_writer_c.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/packed/writer.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <parquet/properties.h>

#include <memory>
#include <string>
#include <vector>

using milvus_storage::FilesystemCache;
using milvus_storage::PackedRecordBatchWriter;
using milvus_storage::StorageConfig;
using milvus_storage::api::ConvertFFIProperties;
using milvus_storage::api::GetValue;
using milvus_storage::api::Properties;

namespace {

// PackedRecordBatchWriter does not expose schema(), but ImportRecordBatch
// needs one — cache it alongside the writer so the FFI handle is
// self-contained.
struct PackedWriterHolder {
  std::shared_ptr<PackedRecordBatchWriter> writer;
  std::shared_ptr<arrow::Schema> schema;
};

}  // namespace

LoonFFIResult loon_packed_writer_new(const char* const* paths,
                                     int32_t num_groups,
                                     const int32_t* group_offsets,
                                     const int32_t* group_indices,
                                     int32_t total_indices,
                                     ArrowSchema* schema_raw,
                                     const LoonProperties* properties,
                                     int64_t buffer_size,
                                     LoonPackedWriterHandle* out_handle) {
  if (!paths || !group_offsets || !group_indices || !schema_raw || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: paths/group_offsets/group_indices/schema/properties/out_handle "
                 "must not be null");
  }
  if (num_groups <= 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "num_groups must be > 0, got ", num_groups);
  }
  if (total_indices < 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "total_indices must be >= 0, got ", total_indices);
  }

  try {
    Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    auto schema_result = arrow::ImportSchema(schema_raw);
    if (!schema_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, schema_result.status().ToString());
    }
    auto schema = schema_result.ValueOrDie();

    // CSR → vector<vector<int>>; validate offsets monotonic + bounded.
    std::vector<std::vector<int>> column_groups(num_groups);
    if (group_offsets[0] != 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "group_offsets[0] must be 0, got ", group_offsets[0]);
    }
    for (int32_t i = 0; i < num_groups; ++i) {
      int32_t lo = group_offsets[i];
      int32_t hi = group_offsets[i + 1];
      if (hi < lo || hi > total_indices) {
        RETURN_ERROR(LOON_INVALID_ARGS, "invalid group_offsets at i=", i, " [lo=", lo, ", hi=", hi,
                     ", total_indices=", total_indices, "]");
      }
      column_groups[i].reserve(hi - lo);
      for (int32_t k = lo; k < hi; ++k) {
        int idx = group_indices[k];
        if (idx < 0 || idx >= schema->num_fields()) {
          RETURN_ERROR(LOON_INVALID_ARGS, "column index out of range: ", idx, " (schema has ", schema->num_fields(),
                       " fields)");
        }
        column_groups[i].push_back(idx);
      }
    }

    std::vector<std::string> path_vec(num_groups);
    for (int32_t i = 0; i < num_groups; ++i) {
      if (!paths[i]) {
        RETURN_ERROR(LOON_INVALID_ARGS, "paths[", i, "] is null");
      }
      path_vec[i] = paths[i];
    }

    // Filesystem comes from the same cache loon_writer_new uses (via
    // ColumnGroupWriter). Key by path[0] — the cache uses address+bucket, not
    // exact path.
    auto fs_result = FilesystemCache::getInstance().get(properties_map, path_vec[0]);
    if (!fs_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to obtain filesystem: ", fs_result.status().ToString());
    }
    auto fs = fs_result.ValueOrDie();

    auto part_size_result = GetValue<int64_t>(properties_map, PROPERTY_FS_MULTI_PART_UPLOAD_SIZE);
    if (!part_size_result.ok()) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES,
                   "Failed to read fs.multi_part_upload_size: ", part_size_result.status().ToString());
    }
    StorageConfig storage_config{part_size_result.ValueOrDie()};

    size_t effective_buffer = buffer_size > 0 ? static_cast<size_t>(buffer_size)
                                              : static_cast<size_t>(milvus_storage::DEFAULT_WRITE_BUFFER_SIZE);

    auto writer_result = PackedRecordBatchWriter::Make(fs, path_vec, schema, storage_config, column_groups,
                                                       effective_buffer, ::parquet::default_writer_properties());
    if (!writer_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, "Failed to create PackedRecordBatchWriter: ", writer_result.status().ToString());
    }

    auto* holder = new PackedWriterHolder{writer_result.ValueOrDie(), std::move(schema)};
    *out_handle = reinterpret_cast<LoonPackedWriterHandle>(holder);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_packed_writer_write(LoonPackedWriterHandle handle, ArrowArray* array) {
  if (!handle || !array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "handle and array must not be null");
  }
  try {
    auto* holder = reinterpret_cast<PackedWriterHolder*>(handle);
    auto rb_result = arrow::ImportRecordBatch(array, holder->schema);
    if (!rb_result.ok()) {
      if (array->release) {
        array->release(array);
      }
      RETURN_ERROR(LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto status = holder->writer->Write(rb_result.ValueOrDie());
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }
  RETURN_UNREACHABLE();
}

LoonFFIResult loon_packed_writer_close(LoonPackedWriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "handle must not be null");
  }
  try {
    auto* holder = reinterpret_cast<PackedWriterHolder*>(handle);
    auto status = holder->writer->Close();
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }
  RETURN_UNREACHABLE();
}

void loon_packed_writer_destroy(LoonPackedWriterHandle handle) {
  if (handle) {
    auto* holder = reinterpret_cast<PackedWriterHolder*>(handle);
    delete holder;
  }
}
