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

#include "milvus-storage/common/macro.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/reader.h"

using namespace milvus_storage::api;
using namespace milvus_storage;

// ==================== ChunkReader C Implementation ====================

FFIResult get_chunk_indices(ChunkReaderHandle reader,
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

void free_chunk_indices(int64_t* chunk_indices) { free(chunk_indices); }

FFIResult get_number_of_chunks(ChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks) {
  if (!chunk_reader || !out_number_of_chunks) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: chunk_reader and out_number_of_chunks must not be null");
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(chunk_reader);
  *out_number_of_chunks = cpp_reader->total_number_of_chunks();
  RETURN_SUCCESS();
}

FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, ArrowArray* out_array) {
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

FFIResult get_chunk_metadatas(ChunkReaderHandle reader, uint32_t metadata_type, ChunkMetadatas* out_chunk_metadata) {
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

  out_chunk_metadata->metadatas = static_cast<ChunkMetadata*>(calloc(1, sizeof(ChunkMetadata) * meta_count));
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
      free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_ARROW_ERROR, estimated_mem_result.status().ToString());
    }
    const auto& estimated_memsz = estimated_mem_result.ValueOrDie();
    assert(estimated_memsz.size() == cpp_reader->total_number_of_chunks());

    assert(out_chunk_metadata->metadatas_size < meta_count);
    auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

    chunk_meta->metadata_type = LOON_CHUNK_METADATA_ESTIMATED_MEMORY;
    chunk_meta->data =
        static_cast<ChunkMetadata::result_u*>(malloc(sizeof(ChunkMetadata::result_u) * estimated_memsz.size()));
    if (!chunk_meta->data) {
      assert(chunk_meta->number_of_chunks == 0);
      free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk metadata");
    }
    static_assert(sizeof(uint64_t) == sizeof(ChunkMetadata::result_u));
    std::memcpy(chunk_meta->data, estimated_memsz.data(), sizeof(ChunkMetadata::result_u) * estimated_memsz.size());

    chunk_meta->number_of_chunks = estimated_memsz.size();
  }

  if (metadata_type & LOON_CHUNK_METADATA_NUMOFROWS) {
    auto chunk_rows = cpp_reader->get_chunk_rows();
    if (!chunk_rows.ok()) {
      free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_ARROW_ERROR, chunk_rows.status().ToString());
    }
    const auto& rows_per_chunk = chunk_rows.ValueOrDie();
    assert(rows_per_chunk.size() == cpp_reader->total_number_of_chunks());

    assert(out_chunk_metadata->metadatas_size < meta_count);
    auto* chunk_meta = &out_chunk_metadata->metadatas[out_chunk_metadata->metadatas_size++];

    chunk_meta->metadata_type = LOON_CHUNK_METADATA_NUMOFROWS;
    chunk_meta->data =
        static_cast<ChunkMetadata::result_u*>(malloc(sizeof(ChunkMetadata::result_u) * rows_per_chunk.size()));
    if (!chunk_meta->data) {
      assert(chunk_meta->number_of_chunks == 0);
      free_chunk_metadatas(out_chunk_metadata);
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for chunk metadata");
    }

    /* rows_per_chunk is a vector<uint64_t> and ChunkMetadata::result_u
       is a union containing a uint64_t member. It's safe to copy the
       underlying uint64_t array in one shot. */
    static_assert(sizeof(uint64_t) == sizeof(ChunkMetadata::result_u));
    std::memcpy(chunk_meta->data, rows_per_chunk.data(), sizeof(ChunkMetadata::result_u) * rows_per_chunk.size());

    chunk_meta->number_of_chunks = rows_per_chunk.size();
  }

  RETURN_SUCCESS();
}

void free_chunk_metadatas(ChunkMetadatas* chunk_metadata) {
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

FFIResult get_chunks(ChunkReaderHandle reader,
                     const int64_t* chunk_indices,
                     size_t num_indices,
                     int64_t parallelism,
                     ArrowArray** arrays,
                     size_t* num_arrays) {
  if (!reader || !chunk_indices || num_indices == 0 || !arrays || !num_arrays) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: reader, chunk_indices, arrays, and num_arrays must not be null, and num_indices "
                 "must be > 0");
  }

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
        free_chunk_arrays(*arrays, i);
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

void free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays) {
  if (arrays) {
    for (size_t i = 0; i < num_arrays; ++i) {
      if (arrays[i].release) {
        arrays[i].release(&arrays[i]);
      }
    }
    free(arrays);
  }
}

void chunk_reader_destroy(ChunkReaderHandle reader) {
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

FFIResult reader_new(char* cloumngroups,
                     ArrowSchema* schema,
                     const char* const* needed_columns,
                     size_t num_columns,
                     const ::Properties* properties,
                     ReaderHandle* out_handle) {
  if (!cloumngroups || !schema || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: cloumngroups, schema, properties, and out_handle must not be null");
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
  // Parse the column groups, the column groups is a JSON string
  auto cpp_column_groups = std::make_shared<ColumnGroups>();
  auto des_result = cpp_column_groups->deserialize(std::string_view(cloumngroups));
  if (!des_result.ok()) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Failed to deserialize column groups JSON: ", des_result.ToString());
  }
  auto cpp_reader = Reader::create(cpp_column_groups, cpp_schema, cpp_needed_columns, cpp_properties);
  auto raw_cpp_reader = reinterpret_cast<ReaderHandle>(cpp_reader.release());
  assert(raw_cpp_reader);
  *out_handle = raw_cpp_reader;

  RETURN_SUCCESS();
}

void reader_set_keyretriever(ReaderHandle reader, const char* (*key_retriever)(const char* metadata)) {
  assert(reader && key_retriever);

  auto* cpp_reader = reinterpret_cast<Reader*>(reader);
  cpp_reader->set_keyretriever([key_retriever](const std::string& metadata) -> std::string {
    const char* result = key_retriever(metadata.c_str());
    return result ? std::string(result) : std::string();
  });
}

FFIResult get_record_batch_reader(ReaderHandle reader, const char* predicate, ArrowArrayStream* out_array_stream) {
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

FFIResult get_column_group_infos(ReaderHandle reader, ColumnGroupInfos* column_group_infos, bool with_meta) {
  if (!reader || !column_group_infos)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and column_group_infos must not be null");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);

    auto cgsptr = cpp_reader->get_column_groups();
    assert(cgsptr);

    auto all_cgs = cgsptr->get_all();
    assert(!all_cgs.empty());

    // Initialize output structure
    memset(column_group_infos, 0, sizeof(ColumnGroupInfos));

    // Populate the column_group_infos structure
    column_group_infos->cg_infos = static_cast<ColumnGroupInfo*>(calloc(1, sizeof(ColumnGroupInfo) * all_cgs.size()));
    if (!column_group_infos->cg_infos) {
      RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for column group infos");
    }
    column_group_infos->cginfos_size = all_cgs.size();

    for (size_t i = 0; i < all_cgs.size(); ++i) {
      assert(all_cgs[i]);

      column_group_infos->cg_infos[i].column_group_id = i;
      const std::vector<std::string>& columns = all_cgs[i]->columns;

      column_group_infos->cg_infos[i].columns = static_cast<char**>(malloc(sizeof(char*) * columns.size()));
      if (!column_group_infos->cg_infos[i].columns) {
        // columns_size will be set after this allocation success
        // must be 0 if allocation failed
        assert(column_group_infos->cg_infos[i].columns_size == 0);
        free_column_group_infos(column_group_infos);
        RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for column group columns");
      }
      column_group_infos->cg_infos[i].columns_size = columns.size();
      for (size_t cn_idx = 0; cn_idx < columns.size(); ++cn_idx) {
        const std::string& col = columns[cn_idx];
        column_group_infos->cg_infos[i].columns[cn_idx] = strdup(col.c_str());
      }
    }

    // Poplulate metadata if requested
    auto metasz = cgsptr->meta_size();
    if (with_meta && metasz > 0) {
      column_group_infos->meta_keys = static_cast<char**>(calloc(1, sizeof(char*) * metasz));
      column_group_infos->meta_values = static_cast<char**>(calloc(1, sizeof(char*) * metasz));
      if (!column_group_infos->meta_keys || !column_group_infos->meta_values) {
        free(column_group_infos->meta_keys);
        free(column_group_infos->meta_values);

        free_column_group_infos(column_group_infos);
        RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to alloc for column group metadata");
      }

      for (size_t mi = 0; mi < metasz; ++mi) {
        const auto& metadata_result = cgsptr->get_metadata(mi);
        if (!metadata_result.ok()) {
          // free previously allocated metadata keys and values
          for (size_t mj = 0; mj < mi; ++mj) {
            free(column_group_infos->meta_keys[mj]);
            free(column_group_infos->meta_values[mj]);
          }
          free(column_group_infos->meta_keys);
          free(column_group_infos->meta_values);

          free_column_group_infos(column_group_infos);
          RETURN_ERROR(LOON_ARROW_ERROR, metadata_result.status().ToString());
        }

        auto metadata = metadata_result.ValueOrDie();

        column_group_infos->meta_keys[mi] = strdup(metadata.first.data());
        column_group_infos->meta_values[mi] = strdup(metadata.second.data());
      }

      column_group_infos->meta_size = metasz;
    }  // else do nothing, because memset already set them to nullptr/0

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }
  RETURN_UNREACHABLE();
}

void free_column_group_infos(ColumnGroupInfos* column_group_infos) {
  if (!column_group_infos) {
    return;
  }

  // free the column group infos
  {
    assert_if(column_group_infos->cginfos_size != 0, column_group_infos->cg_infos != nullptr);

    for (size_t i = 0; i < column_group_infos->cginfos_size; ++i) {
      ColumnGroupInfo& cg_info = column_group_infos->cg_infos[i];
      if (cg_info.columns) {
        for (size_t j = 0; j < cg_info.columns_size; ++j) {
          free(cg_info.columns[j]);
        }
        free(cg_info.columns);
      }
    }

    free(column_group_infos->cg_infos);
    column_group_infos->cg_infos = nullptr;
    column_group_infos->cginfos_size = 0;
  }

  // free the metadata if exists
  if (column_group_infos->meta_size > 0) {
    for (size_t i = 0; i < column_group_infos->meta_size; ++i) {
      free(column_group_infos->meta_keys[i]);
      free(column_group_infos->meta_values[i]);
    }

    free(column_group_infos->meta_keys);
    free(column_group_infos->meta_values);

    column_group_infos->meta_keys = nullptr;
    column_group_infos->meta_values = nullptr;
    column_group_infos->meta_size = 0;
  }
}

FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle) {
  if (!reader || !out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_handle must not be null");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto result = cpp_reader->get_chunk_reader(column_group_id);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<ChunkReaderHandle>(chunk_reader);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

FFIResult take(
    ReaderHandle reader, const int64_t* row_indices, size_t num_indices, int64_t parallelism, ArrowArray* out_arrays) {
  if (!reader || !row_indices || num_indices == 0 || !out_arrays)
    RETURN_ERROR(
        LOON_INVALID_ARGS,
        "Invalid arguments: reader, row_indices, and out_arrays must not be null, and num_indices must be > 0");

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::vector<int64_t> indices(row_indices, row_indices + num_indices);

    auto result = cpp_reader->take(indices, parallelism);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());

    // export the arrow::RecordBatch to Arrow C ABI Array
    auto record_batch = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_arrays);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void reader_destroy(ReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<Reader*>(reader);
  }
}