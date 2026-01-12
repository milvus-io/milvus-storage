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
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include <arrow/c/abi.h>
#include <arrow/c/helpers.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ThreadPoolExecutor.h>

#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/thread_pool.h"

using namespace milvus_storage::api;
using namespace milvus_storage;

// ==================== ChunkReader C Implementation ====================

LoonFFIResult loon_get_chunk_indices(LoonChunkReaderHandle reader,
                                     const int64_t* row_indices,
                                     size_t num_indices,
                                     int64_t** chunk_indices,
                                     size_t* num_chunk_indices) {
  if (!reader || !row_indices || num_indices == 0 || !chunk_indices || !num_chunk_indices) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: reader, row_indices, chunk_indices, and num_chunk_indices must not be null, and "
                 "num_indices must be > 0");
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  std::vector<int64_t> input_indices(row_indices, row_indices + num_indices);

  auto result = cpp_reader->get_chunk_indices(std::move(input_indices));
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

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
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk indices [size=", output_indices.size(), "]");
  }

  RETURN_SUCCESS();
}

void loon_free_chunk_indices(int64_t* chunk_indices) { free(chunk_indices); }

LoonFFIResult loon_get_number_of_chunks(LoonChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks) {
  if (!chunk_reader || !out_number_of_chunks) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: chunk_reader and out_number_of_chunks must not be null");
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(chunk_reader);
  *out_number_of_chunks = cpp_reader->total_number_of_chunks();
  RETURN_SUCCESS();
}

LoonFFIResult loon_get_chunk(LoonChunkReaderHandle reader, int64_t chunk_index, ArrowArray* out_array) {
  if (!reader || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_array must not be null");
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  auto result = cpp_reader->get_chunk(chunk_index);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }
  auto record_batch = result.ValueOrDie();
  arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_array);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  RETURN_SUCCESS();
}

LoonFFIResult loon_get_chunk_metadatas(LoonChunkReaderHandle reader,
                                       uint32_t metadata_type,
                                       LoonChunkMetadatas* out_chunk_metadata) {
  // no need check chunk_index here, will check in ChunkReader implementation
  if (!reader || !out_chunk_metadata) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_chunk_metadata must not be null");
  }

  uint32_t masked_values = metadata_type & LOON_CHUNK_METADATA_ALL;
  int meta_count = 0;
  while (masked_values) {
    meta_count += masked_values & 1;
    masked_values >>= 1;
  }
  if (meta_count == 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: metadata_type has no valid metadata type bits set",
                 " [metadata_type=", metadata_type, "]");
  }

  out_chunk_metadata->metadatas = static_cast<LoonChunkMetadata*>(calloc(1, sizeof(LoonChunkMetadata) * meta_count));
  if (!out_chunk_metadata->metadatas) {
    out_chunk_metadata->metadatas = nullptr;
    out_chunk_metadata->metadatas_size = 0;
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk metadata");
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);

  out_chunk_metadata->metadatas_size = 0;
  if (metadata_type & LOON_CHUNK_METADATA_ESTIMATED_MEMORY) {
    auto estimated_mem_result = cpp_reader->get_chunk_size();
    if (!estimated_mem_result.ok()) {
      // must be 0 because calloc and `number_of_chunks` will be updated at last.
      loon_free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_ARROW_ERROR, estimated_mem_result.status().ToString());
    }
    const auto& estimated_memsz = estimated_mem_result.ValueOrDie();
    assert(estimated_memsz.size() == cpp_reader->total_number_of_chunks());

    assert(out_chunk_metadata->metadatas_size < meta_count);
    auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

    chunk_meta->metadata_type = LOON_CHUNK_METADATA_ESTIMATED_MEMORY;
    chunk_meta->data =
        static_cast<LoonChunkMetadata::result_u*>(malloc(sizeof(LoonChunkMetadata::result_u) * estimated_memsz.size()));
    if (!chunk_meta->data) {
      assert(chunk_meta->number_of_chunks == 0);
      loon_free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk metadata");
    }
    static_assert(sizeof(uint64_t) == sizeof(LoonChunkMetadata::result_u));
    std::memcpy(chunk_meta->data, estimated_memsz.data(), sizeof(LoonChunkMetadata::result_u) * estimated_memsz.size());

    chunk_meta->number_of_chunks = estimated_memsz.size();
  }

  if (metadata_type & LOON_CHUNK_METADATA_NUMOFROWS) {
    auto chunk_rows = cpp_reader->get_chunk_rows();
    if (!chunk_rows.ok()) {
      loon_free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_ARROW_ERROR, chunk_rows.status().ToString());
    }
    const auto& rows_per_chunk = chunk_rows.ValueOrDie();
    assert(rows_per_chunk.size() == cpp_reader->total_number_of_chunks());

    assert(out_chunk_metadata->metadatas_size < meta_count);
    auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

    chunk_meta->metadata_type = LOON_CHUNK_METADATA_NUMOFROWS;
    chunk_meta->data =
        static_cast<LoonChunkMetadata::result_u*>(malloc(sizeof(LoonChunkMetadata::result_u) * rows_per_chunk.size()));
    if (!chunk_meta->data) {
      assert(chunk_meta->number_of_chunks == 0);
      loon_free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk metadata");
    }

    /* rows_per_chunk is a vector<uint64_t> and LoonChunkMetadata::result_u
       is a union containing a uint64_t member. It's safe to copy the
       underlying uint64_t array in one shot. */
    static_assert(sizeof(uint64_t) == sizeof(LoonChunkMetadata::result_u));
    std::memcpy(chunk_meta->data, rows_per_chunk.data(), sizeof(LoonChunkMetadata::result_u) * rows_per_chunk.size());

    chunk_meta->number_of_chunks = rows_per_chunk.size();
  }

  RETURN_SUCCESS();
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
                              size_t* num_arrays) {
  if (!reader || !chunk_indices || num_indices == 0 || !arrays || !num_arrays) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: reader, chunk_indices, arrays, and num_arrays must not be null, and num_indices "
                 "must be > 0");
  }

  if (parallelism == 0)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: parallelism must be > 0");

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  std::vector<int64_t> indices(chunk_indices, chunk_indices + num_indices);

  auto result = cpp_reader->get_chunks(indices, parallelism);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

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
        *arrays = NULL;
        RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
      }
    }
  } else {
    *num_arrays = 0;
    *arrays = NULL;
    RETURN_ERROR(LOON_MEMORY_ERROR, "Fail to alloc for chunk arrays [rb size=", record_batches.size(), "]");
  }

  RETURN_SUCCESS();
}

void loon_free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays) {
  if (arrays) {
    for (size_t i = 0; i < num_arrays; ++i) {
      if (arrays[i].release) {
        arrays[i].release(&arrays[i]);
      }
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
  std::vector<std::string> result;

  // empty projections
  if (count == 0 || strings == nullptr) {
    return nullptr;
  }

  if (strings && count > 0) {
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (strings[i]) {
        result.emplace_back(strings[i]);
      }
    }
  }

  // projections result is empty
  if (result.empty()) {
    return nullptr;
  }

  return std::make_shared<std::vector<std::string>>(result);
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

  milvus_storage::api::Properties properties_map;
  auto opt = ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  auto result = arrow::ImportSchema(schema);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

  auto cpp_schema = result.ValueOrDie();
  auto cpp_properties = std::move(properties_map);
  auto cpp_needed_columns = convert_needed_columns(needed_columns, num_columns);

  // Import LoonColumnGroups to ColumnGroups
  ColumnGroups cpp_column_groups;
  auto import_st = milvus_storage::import_column_groups(column_groups, &cpp_column_groups);
  if (!import_st.ok()) {
    RETURN_ERROR(LOON_LOGICAL_ERROR, import_st.ToString());
  }

  // Wrap in shared_ptr for Reader::create
  auto cpp_column_groups_ptr = std::make_shared<ColumnGroups>(std::move(cpp_column_groups));
  auto cpp_reader = Reader::create(cpp_column_groups_ptr, cpp_schema, cpp_needed_columns, cpp_properties);
  auto raw_cpp_reader = reinterpret_cast<LoonReaderHandle>(cpp_reader.release());
  assert(raw_cpp_reader);
  *out_handle = raw_cpp_reader;

  RETURN_SUCCESS();
}

void loon_reader_set_keyretriever(LoonReaderHandle reader, const char* (*key_retriever)(const char* metadata)) {
  assert(reader && key_retriever);

  auto* cpp_reader = reinterpret_cast<Reader*>(reader);
  cpp_reader->set_keyretriever([key_retriever](const std::string& metadata) -> std::string {
    const char* result = key_retriever(metadata.c_str());
    return result ? std::string(result) : std::string();
  });
}

LoonFFIResult loon_get_record_batch_reader(LoonReaderHandle reader,
                                           const char* predicate,
                                           ArrowArrayStream* out_array_stream) {
  if (!reader || !out_array_stream)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_array_stream must not be null");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str);
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }

    auto array_stream = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatchReader(array_stream, out_array_stream);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {  // TODO: make sure which exception will be throw
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_get_chunk_reader(LoonReaderHandle reader,
                                    int64_t column_group_id,
                                    LoonChunkReaderHandle* out_handle) {
  if (!reader || !out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_handle must not be null");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto result = cpp_reader->get_chunk_reader(column_group_id);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<LoonChunkReaderHandle>(chunk_reader);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_take(LoonReaderHandle reader,
                        const int64_t* row_indices,
                        size_t num_indices,
                        size_t parallelism,
                        ArrowArray** arrays,
                        size_t* num_arrays) {
  if (!reader || !row_indices || num_indices == 0 || !arrays || !num_arrays)
    RETURN_ERROR(
        LOON_INVALID_ARGS,
        "Invalid arguments: reader, row_indices, and out_arrays must not be null, and num_indices must be > 0");

  if (parallelism == 0)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: parallelism must be > 0");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::vector<int64_t> indices(row_indices, row_indices + num_indices);

    auto result = cpp_reader->take(indices, parallelism);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());

    auto table = result.ValueOrDie();
    auto rbs_result = ConvertTableToRecordBatchs(table);
    if (!rbs_result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, rbs_result.status().ToString());
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
          *arrays = NULL;
          RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
        }
      }
    } else {
      *num_arrays = 0;
      *arrays = NULL;
      RETURN_ERROR(LOON_MEMORY_ERROR, "Fail to alloc for chunk arrays [rb size=", record_batches.size(), "]");
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_reader_destroy(LoonReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<Reader*>(reader);
  }
}