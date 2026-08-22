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
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/c/abi.h>
#include <arrow/c/helpers.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ThreadPoolExecutor.h>

#include <fmt/format.h>

#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/record_batch_reader.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/thread_pool.h"

using namespace milvus_storage::api;
using namespace milvus_storage;

namespace milvus_storage::ffi_internal {

static void ReleaseArrowArray(ArrowArray* array) noexcept {
  if (array != nullptr && array->release != nullptr) {
    try {
      array->release(array);
    } catch (...) {
      // A foreign release callback must not terminate a C ABI cleanup path.
    }
    array->release = nullptr;
  }
}

static void ReleaseArrowSchema(ArrowSchema* schema) noexcept {
  if (schema != nullptr && schema->release != nullptr) {
    try {
      schema->release(schema);
    } catch (...) {
      // A foreign release callback must not terminate a C ABI cleanup path.
    }
    schema->release = nullptr;
  }
}

LoonFFIResult RecordBatchReaderReadNext(LoonRecordBatchReaderHandle handle,
                                        ArrowArray* out_array,
                                        bool materialize_offsets) {
  if (!handle || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_array must not be null");
  }

  // A failed call must never leave a stale release callback supplied by the
  // caller. The C Data Interface requires an uninitialised output struct; make
  // the failure/EOF contract deterministic at this ABI boundary as well.
  *out_array = ArrowArray{};

  try {
    auto* holder = reinterpret_cast<RecordBatchReaderHolder*>(handle);
    if (holder->exhausted) {
      // Normal end-of-stream is idempotent: keep answering EOF, per the Arrow
      // C stream convention. The underlying reader was already released.
      out_array->release = nullptr;
      RETURN_SUCCESS();
    }
    if (!holder->reader) {
      // Terminal after a FAILED read: reusing the handle is a caller contract
      // violation, not a library bug.
      RETURN_ERROR(LOON_INVALID_ARGS, "RecordBatchReader failed earlier; create a new reader");
    }
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = holder->reader->ReadNext(&batch);
    if (!status.ok()) {
      // A RecordBatchReader is stateful: after ReadNext fails its internal
      // cursor/buffers may already have advanced. Make the handle terminal so
      // callers cannot accidentally retry the same reader instance.
      holder->reader.reset();
      auto ffi_code = FFIErrorCodeFromExtendStatus(status, LOON_ARROW_ERROR);
      RETURN_ERROR(ffi_code, status.ToString());
    }

    if (batch == nullptr) {
      // Release the reader's resources now, but remember this was a normal
      // exhaustion so later calls keep reporting EOF instead of erroring.
      holder->reader.reset();
      holder->exhausted = true;
      out_array->release = nullptr;
      RETURN_SUCCESS();
    }

    // ArrowArray.offset is part of the C Data Interface contract. Standard C,
    // Python and Rust importers honour it and therefore keep the zero-copy
    // export below. Arrow Java currently does not; the JNI wrapper opts into
    // materialization for that consumer only.
    if (materialize_offsets) {
      bool has_sliced_column = false;
      for (int i = 0; i < batch->num_columns(); ++i) {
        if (batch->column(i)->offset() != 0) {
          has_sliced_column = true;
          break;
        }
      }
      if (has_sliced_column) {
        std::vector<std::shared_ptr<arrow::Array>> fresh_cols;
        fresh_cols.reserve(batch->num_columns());
        for (int i = 0; i < batch->num_columns(); ++i) {
          auto col = batch->column(i);
          if (col->offset() == 0) {
            fresh_cols.push_back(col);
          } else {
            auto concat_result = arrow::Concatenate({col}, arrow::default_memory_pool());
            if (!concat_result.ok()) {
              auto failure = concat_result.status();
              holder->reader.reset();
              RETURN_ARROW_ERROR(failure, LOON_ARROW_ERROR, failure.ToString());
            }
            fresh_cols.push_back(concat_result.ValueOrDie());
          }
        }
        batch = arrow::RecordBatch::Make(batch->schema(), batch->num_rows(), fresh_cols);
      }
    }

    auto export_status = arrow::ExportRecordBatch(*batch, out_array);
    if (!export_status.ok()) {
      ReleaseArrowArray(out_array);
      holder->reader.reset();
      RETURN_ARROW_ERROR(export_status, LOON_ARROW_ERROR, export_status.ToString());
    }
    RETURN_SUCCESS();
  } catch (...) {
    auto* holder = reinterpret_cast<RecordBatchReaderHolder*>(handle);
    ReleaseArrowArray(out_array);
    holder->reader.reset();
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult RecordBatchReaderReadNextForJava(LoonRecordBatchReaderHandle handle,
                                               ArrowArray* out_array,
                                               ArrowSchema* out_schema) {
  if (out_schema == nullptr) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: out_schema must not be null");
  }
  *out_schema = ArrowSchema{};
  auto result = RecordBatchReaderReadNext(handle, out_array, /*materialize_offsets=*/true);
  if (!loon_ffi_is_success(&result) || out_array->release == nullptr) {
    return result;
  }

  auto* holder = reinterpret_cast<RecordBatchReaderHolder*>(handle);
  auto export_status = arrow::ExportSchema(*holder->reader->schema(), out_schema);
  if (!export_status.ok()) {
    ReleaseArrowArray(out_array);
    ReleaseArrowSchema(out_schema);
    holder->reader.reset();
    return CreateFFIResult(FFIErrorCodeFromExtendStatus(export_status, LOON_ARROW_ERROR), export_status.ToString());
  }
  return result;
}

}  // namespace milvus_storage::ffi_internal

// ==================== ChunkReader C Implementation ====================

LoonFFIResult loon_get_chunk_indices(LoonChunkReaderHandle reader,
                                     const int64_t* row_indices,
                                     size_t num_indices,
                                     int64_t** chunk_indices,
                                     size_t* num_chunk_indices) {
  if (!reader || !row_indices || !chunk_indices || !num_chunk_indices) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: reader, row_indices, chunk_indices, and num_chunk_indices must not be null");
  }
  if (num_indices == 0) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: num_indices must be > 0");
  }

  try {
    auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
    std::vector<int64_t> input_indices(row_indices, row_indices + num_indices);

    auto result = cpp_reader->get_chunk_indices(input_indices);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    const auto& output_indices = result.ValueOrDie();
    if (output_indices.empty()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Current indices(out) is empty");
    }

    *chunk_indices = static_cast<int64_t*>(malloc(sizeof(int64_t) * output_indices.size()));
    if (*chunk_indices) {
      std::copy(output_indices.begin(), output_indices.end(), *chunk_indices);
      *num_chunk_indices = output_indices.size();
    } else {
      *chunk_indices = nullptr;
      *num_chunk_indices = 0;
      RETURN_ERROR(LOON_INTERNAL_INVARIANT,
                   "Unexpected allocation failure for chunk indices [size=", output_indices.size(), "]");
    }

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

void loon_free_chunk_indices(int64_t* chunk_indices) { free(chunk_indices); }

LoonFFIResult loon_get_number_of_chunks(LoonChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks) {
  if (!chunk_reader || !out_number_of_chunks) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: chunk_reader and out_number_of_chunks must not be null");
  }

  try {
    auto* cpp_reader = reinterpret_cast<ChunkReader*>(chunk_reader);
    *out_number_of_chunks = cpp_reader->total_number_of_chunks();
    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_get_chunk(LoonChunkReaderHandle reader,
                             int64_t chunk_index,
                             ArrowArray* out_array,
                             ArrowSchema* out_schema) {
  if (!reader || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_array must not be null");
  }

  try {
    // Fault injection point for testing
    FIU_DO_ON(FIUKEY_CHUNK_READER_READ_FAIL,
              RETURN_ERROR(LOON_FAULT_INJECT_ERROR, fmt::format("Injected fault: {}", FIUKEY_CHUNK_READER_READ_FAIL)));
    auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
    auto result = cpp_reader->get_chunk(chunk_index);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());
    auto record_batch = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_array);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    if (out_schema) {
      status = arrow::ExportSchema(*record_batch->schema(), out_schema);
      if (!status.ok()) {
        if (out_array->release) {
          out_array->release(out_array);
        }
        RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, status.ToString());
      }
    }

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_get_chunk_metadatas(LoonChunkReaderHandle reader,
                                       uint32_t metadata_type,
                                       LoonChunkMetadatas* out_chunk_metadata) {
  // no need check chunk_index here, will check in ChunkReader implementation
  if (!reader || !out_chunk_metadata) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_chunk_metadata must not be null");
  }

  try {
    uint32_t masked_values = metadata_type & LOON_CHUNK_METADATA_ALL;
    int meta_count = 0;
    while (masked_values) {
      meta_count += masked_values & 1;
      masked_values >>= 1;
    }
    if (meta_count == 0) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: metadata_type has no valid metadata type bits set",
                   " [metadata_type=", metadata_type, "]");
    }

    out_chunk_metadata->metadatas = static_cast<LoonChunkMetadata*>(calloc(1, sizeof(LoonChunkMetadata) * meta_count));
    if (!out_chunk_metadata->metadatas) {
      out_chunk_metadata->metadatas = nullptr;
      out_chunk_metadata->metadatas_size = 0;
      RETURN_ERROR(LOON_INTERNAL_INVARIANT, "Unexpected allocation failure for chunk metadata");
    }

    auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);

    out_chunk_metadata->metadatas_size = 0;
    if (metadata_type & LOON_CHUNK_METADATA_ESTIMATED_MEMORY) {
      auto estimated_mem_result = cpp_reader->get_chunk_estimated_size();
      if (!estimated_mem_result.ok()) {
        // must be 0 because calloc and `number_of_chunks` will be updated at last.
        loon_free_chunk_metadatas(out_chunk_metadata);
        RETURN_ARROW_ERROR(estimated_mem_result.status(), LOON_ARROW_ERROR, estimated_mem_result.status().ToString());
      }
      const auto& estimated_memsz = estimated_mem_result.ValueOrDie();
      assert(estimated_memsz.size() == cpp_reader->total_number_of_chunks());

      assert(out_chunk_metadata->metadatas_size < meta_count);
      auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

      chunk_meta->metadata_type = LOON_CHUNK_METADATA_ESTIMATED_MEMORY;
      chunk_meta->data = static_cast<LoonChunkMetadata::result_u*>(
          malloc(sizeof(LoonChunkMetadata::result_u) * estimated_memsz.size()));
      if (!chunk_meta->data) {
        assert(chunk_meta->number_of_chunks == 0);
        loon_free_chunk_metadatas(out_chunk_metadata);
        RETURN_ERROR(LOON_INTERNAL_INVARIANT, "Unexpected allocation failure for chunk metadata");
      }
      static_assert(sizeof(uint64_t) == sizeof(LoonChunkMetadata::result_u));
      std::memcpy(chunk_meta->data, estimated_memsz.data(),
                  sizeof(LoonChunkMetadata::result_u) * estimated_memsz.size());

      chunk_meta->number_of_chunks = estimated_memsz.size();
    }

    if (metadata_type & LOON_CHUNK_METADATA_NUMOFROWS) {
      auto chunk_rows = cpp_reader->get_chunk_rows();
      if (!chunk_rows.ok()) {
        loon_free_chunk_metadatas(out_chunk_metadata);
        RETURN_ARROW_ERROR(chunk_rows.status(), LOON_ARROW_ERROR, chunk_rows.status().ToString());
      }
      const auto& rows_per_chunk = chunk_rows.ValueOrDie();
      assert(rows_per_chunk.size() == cpp_reader->total_number_of_chunks());

      assert(out_chunk_metadata->metadatas_size < meta_count);
      auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

      chunk_meta->metadata_type = LOON_CHUNK_METADATA_NUMOFROWS;
      chunk_meta->data = static_cast<LoonChunkMetadata::result_u*>(
          malloc(sizeof(LoonChunkMetadata::result_u) * rows_per_chunk.size()));
      if (!chunk_meta->data) {
        assert(chunk_meta->number_of_chunks == 0);
        loon_free_chunk_metadatas(out_chunk_metadata);
        RETURN_ERROR(LOON_INTERNAL_INVARIANT, "Unexpected allocation failure for chunk metadata");
      }

      /* rows_per_chunk is a vector<uint64_t> and LoonChunkMetadata::result_u
         is a union containing a uint64_t member. It's safe to copy the
         underlying uint64_t array in one shot. */
      static_assert(sizeof(uint64_t) == sizeof(LoonChunkMetadata::result_u));
      std::memcpy(chunk_meta->data, rows_per_chunk.data(), sizeof(LoonChunkMetadata::result_u) * rows_per_chunk.size());

      chunk_meta->number_of_chunks = rows_per_chunk.size();
    }

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

void loon_free_chunk_metadatas(LoonChunkMetadatas* chunk_metadata) {
  if (chunk_metadata) {
    assert_if(chunk_metadata->metadatas_size > 0, chunk_metadata->metadatas != nullptr);
    for (size_t i = 0; i < chunk_metadata->metadatas_size; ++i) {
      if (chunk_metadata->metadatas[i].data) {
        free(chunk_metadata->metadatas[i].data);
      }
    }
    free(chunk_metadata->metadatas);
    chunk_metadata->metadatas = nullptr;
    chunk_metadata->metadatas_size = 0;
  }
}

LoonFFIResult loon_get_chunks(LoonChunkReaderHandle reader,
                              const int64_t* chunk_indices,
                              size_t num_indices,
                              size_t parallelism,
                              ArrowArray** arrays,
                              size_t* num_arrays,
                              ArrowSchema* out_schema) {
  if (!reader || !chunk_indices || !arrays || !num_arrays) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: reader, chunk_indices, arrays, and num_arrays must not be null");
  }
  if (num_indices == 0) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: num_indices must be > 0");
  }
  if (parallelism == 0) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: parallelism must be > 0");
  }

  try {
    auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
    std::vector<int64_t> indices(chunk_indices, chunk_indices + num_indices);

    auto result = cpp_reader->get_chunks(indices, parallelism);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    const auto& record_batches = result.ValueOrDie();
    if (record_batches.empty()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Empty record batch");
    }

    // Convert RecordBatches to Arrow C ABI arrays
    *arrays = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray) * record_batches.size()));
    if (*arrays) {
      *num_arrays = record_batches.size();
      for (size_t i = 0; i < *num_arrays; ++i) {
        arrow::Status status = arrow::ExportRecordBatch(*(record_batches[i]), &(*arrays)[i]);
        if (!status.ok()) {
          // Free previously allocated arrays
          loon_free_chunk_arrays(*arrays, i);
          *num_arrays = 0;
          *arrays = nullptr;
          RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, status.ToString());
        }
      }
    } else {
      *num_arrays = 0;
      *arrays = nullptr;
      RETURN_ERROR(LOON_INTERNAL_INVARIANT,
                   "Unexpected allocation failure for chunk arrays [rb size=", record_batches.size(), "]");
    }

    if (out_schema && !record_batches.empty()) {
      arrow::Status status = arrow::ExportSchema(*record_batches[0]->schema(), out_schema);
      if (!status.ok()) {
        loon_free_chunk_arrays(*arrays, *num_arrays);
        *num_arrays = 0;
        *arrays = nullptr;
        RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, status.ToString());
      }
    }

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

void loon_free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays) {
  if (arrays) {
    for (size_t i = 0; i < num_arrays; ++i) {
      milvus_storage::ffi_internal::ReleaseArrowArray(&arrays[i]);
    }
    free(arrays);
  }
}

void loon_chunk_reader_destroy(LoonChunkReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<ChunkReader*>(reader);
  }
}

// ==================== Reader C Implementation ====================
static inline std::shared_ptr<std::vector<std::string>> convert_needed_columns(const char* const* strings,
                                                                               size_t count) {
  // empty projections
  if (count == 0) {
    return nullptr;
  }

  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(strings[i]);
  }

  return std::make_shared<std::vector<std::string>>(std::move(result));
}

LoonFFIResult loon_reader_new(const LoonColumnGroups* column_groups,
                              ArrowSchema* schema,
                              const char* const* needed_columns,
                              size_t num_columns,
                              const ::LoonProperties* properties,
                              LoonReaderHandle* out_handle) {
  if (!column_groups || !schema || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: columngroups, schema, properties, and out_handle must not be null");
  }
  try {
    // Fault injection point for testing
    FIU_DO_ON(FIUKEY_READER_OPEN_FAIL,
              RETURN_ERROR(LOON_FAULT_INJECT_ERROR, fmt::format("Injected fault: {}", FIUKEY_READER_OPEN_FAIL)));
    milvus_storage::api::Properties properties_map;
    auto opt = ConvertFFIProperties(properties_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
    }

    auto result = arrow::ImportSchema(schema);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto cpp_schema = result.ValueOrDie();
    auto field_id_status = ValidateFieldIds(cpp_schema);
    RETURN_ARROW_ERROR_IF(field_id_status, LOON_USER_INVALID_ARGUMENT, field_id_status.ToString());
    auto cpp_properties = std::move(properties_map);
    auto cpp_needed_columns = convert_needed_columns(needed_columns, num_columns);

    // Import LoonColumnGroups to ColumnGroups
    ColumnGroups cpp_column_groups;
    auto import_st = milvus_storage::column_groups_import(column_groups, &cpp_column_groups);
    RETURN_ARROW_ERROR_IF(import_st, LOON_LOGICAL_ERROR, import_st.ToString());

    // Wrap in shared_ptr for Reader::create
    auto cpp_column_groups_ptr = std::make_shared<ColumnGroups>(std::move(cpp_column_groups));
    auto cpp_reader = Reader::create(cpp_column_groups_ptr, cpp_schema, cpp_needed_columns, cpp_properties);
    auto raw_cpp_reader = reinterpret_cast<LoonReaderHandle>(cpp_reader.release());
    assert(raw_cpp_reader);
    *out_handle = raw_cpp_reader;

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

void loon_reader_set_keyretriever(LoonReaderHandle reader, const char* (*key_retriever)(const char* metadata)) {
  assert(reader && key_retriever);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    cpp_reader->set_keyretriever([key_retriever](const std::string& metadata) -> std::string {
      const char* result = key_retriever(metadata.c_str());
      return result ? std::string(result) : std::string();
    });
  } catch (...) {
    // This legacy void setter cannot report failure across the C ABI.
  }
}

LoonFFIResult loon_get_record_batch_reader(LoonReaderHandle reader,
                                           const char* predicate,
                                           ArrowArrayStream* out_array_stream) {
  if (!reader || !out_array_stream) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_array_stream must not be null");
  }

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto array_stream = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatchReader(array_stream, out_array_stream);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_record_batch_reader_new(LoonReaderHandle reader,
                                           const char* predicate,
                                           LoonRecordBatchReaderHandle* out_handle,
                                           ArrowSchema* out_schema) {
  if (!reader || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_handle must not be null");
  }
  *out_handle = 0;
  if (out_schema != nullptr) {
    *out_schema = ArrowSchema{};
  }

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto batch_reader = result.ValueOrDie();
    auto holder = std::make_unique<milvus_storage::ffi_internal::RecordBatchReaderHolder>(
        milvus_storage::ffi_internal::RecordBatchReaderHolder{std::move(batch_reader)});
    if (out_schema != nullptr) {
      auto export_status = arrow::ExportSchema(*holder->reader->schema(), out_schema);
      if (!export_status.ok()) {
        milvus_storage::ffi_internal::ReleaseArrowSchema(out_schema);
        RETURN_ARROW_ERROR(export_status, LOON_ARROW_ERROR, export_status.ToString());
      }
    }

    auto* raw_holder = holder.release();
    *out_handle = reinterpret_cast<LoonRecordBatchReaderHandle>(raw_holder);
    RETURN_SUCCESS();
  } catch (...) {
    milvus_storage::ffi_internal::ReleaseArrowSchema(out_schema);
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_record_batch_reader_read_next(LoonRecordBatchReaderHandle handle, ArrowArray* out_array) {
  try {
    return milvus_storage::ffi_internal::RecordBatchReaderReadNext(handle, out_array, /*materialize_offsets=*/false);
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }
}

void loon_record_batch_reader_destroy(LoonRecordBatchReaderHandle handle) {
  if (handle) {
    delete reinterpret_cast<milvus_storage::ffi_internal::RecordBatchReaderHolder*>(handle);
  }
}

LoonFFIResult loon_get_chunk_reader(LoonReaderHandle reader,
                                    int64_t column_group_id,
                                    const char* const* needed_columns,
                                    size_t num_columns,
                                    LoonChunkReaderHandle* out_handle) {
  if (!reader || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_handle must not be null");
  }
  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto cpp_needed_columns = convert_needed_columns(needed_columns, num_columns);
    auto result = cpp_reader->get_chunk_reader(column_group_id, cpp_needed_columns);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<LoonChunkReaderHandle>(chunk_reader);
    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_take(LoonReaderHandle reader,
                        const int64_t* row_indices,
                        size_t num_indices,
                        size_t parallelism,
                        const char* const* needed_columns,
                        size_t num_columns,
                        ArrowArray** arrays,
                        size_t* num_arrays,
                        ArrowSchema* out_schema) {
  try {
    if (!reader || !row_indices || !arrays || !num_arrays) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader, row_indices, and out_arrays must not be null");
    }
    if (num_indices == 0) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: num_indices must be > 0");
    }
    if (parallelism == 0) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: parallelism must be > 0");
    }
    for (size_t i = 0; i < num_indices; ++i) {
      if (row_indices[i] < 0) {
        RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid row index at position ", i, ": ", row_indices[i],
                     " (must be non-negative)");
      }
      if (i > 0 && row_indices[i] <= row_indices[i - 1]) {
        RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "row_indices must be strictly increasing; position ", i, " has value ",
                     row_indices[i], " after ", row_indices[i - 1]);
      }
    }

    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto column_groups = cpp_reader->get_column_groups();
    if (column_groups && !column_groups->empty() && (*column_groups)[0]) {
      int64_t total_rows = 0;
      for (const auto& file : (*column_groups)[0]->files) {
        if (file.start_index < 0 || file.end_index < file.start_index) {
          total_rows = -1;
          break;
        }
        const auto file_rows = file.end_index - file.start_index;
        if (file_rows > std::numeric_limits<int64_t>::max() - total_rows) {
          total_rows = -1;
          break;
        }
        total_rows += file_rows;
      }
      if (total_rows >= 0 && row_indices[num_indices - 1] >= total_rows) {
        RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Row index out of range: ", row_indices[num_indices - 1],
                     " (row count: ", total_rows, ")");
      }
    }

    std::vector<int64_t> indices(row_indices, row_indices + num_indices);
    auto cpp_needed_columns = convert_needed_columns(needed_columns, num_columns);

    auto result = cpp_reader->take(indices, parallelism, cpp_needed_columns);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto table = result.ValueOrDie();
    auto rbs_result = ConvertTableToRecordBatchs(table);
    RETURN_ARROW_ERROR_IF(rbs_result.status(), LOON_ARROW_ERROR, rbs_result.status().ToString());
    auto record_batches = rbs_result.ValueOrDie();

    // Convert RecordBatches to Arrow C ABI arrays
    *arrays = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray) * record_batches.size()));
    if (*arrays) {
      *num_arrays = record_batches.size();
      for (size_t i = 0; i < *num_arrays; ++i) {
        arrow::Status status = arrow::ExportRecordBatch(*(record_batches[i]), &(*arrays)[i]);
        if (!status.ok()) {
          // Free previously allocated arrays
          loon_free_chunk_arrays(*arrays, i);
          *num_arrays = 0;
          *arrays = nullptr;
          RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, status.ToString());
        }
      }

      // Export schema if requested
      if (out_schema && !record_batches.empty()) {
        auto status = arrow::ExportSchema(*record_batches[0]->schema(), out_schema);
        if (!status.ok()) {
          loon_free_chunk_arrays(*arrays, *num_arrays);
          *num_arrays = 0;
          *arrays = nullptr;
          RETURN_ARROW_ERROR(status, LOON_ARROW_ERROR, status.ToString());
        }
      }
    } else {
      *num_arrays = 0;
      *arrays = nullptr;
      RETURN_ERROR(LOON_INTERNAL_INVARIANT,
                   "Unexpected allocation failure for chunk arrays [rb size=", record_batches.size(), "]");
    }

    RETURN_SUCCESS();
  } catch (...) {
    RETURN_EXCEPTION("Native reader operation failed");
  }

  RETURN_UNREACHABLE();
}

void loon_reader_destroy(LoonReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<Reader*>(reader);
  }
}
