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
#include "milvus-storage/reader.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/manifest_json.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_internal/properties.h"

#include <arrow/c/helpers.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include <arrow/c/abi.h>

using namespace milvus_storage::api;
using namespace milvus_storage;

// ==================== ChunkReader C Implementation ====================

FFIResult get_chunk_indices(ChunkReaderHandle reader,
                            const int64_t* row_indices,
                            size_t num_indices,
                            int64_t** chunk_indices,
                            size_t* num_chunk_indices) {
  if (!reader || !row_indices || num_indices == 0 || !chunk_indices || !num_chunk_indices) {
    RETURN_ERROR(LOON_INVALID_ARGS);
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
    *num_chunk_indices = 0;
  }

  RETURN_SUCCESS();
}

void free_chunk_indices(int64_t* chunk_indices) { free(chunk_indices); }

FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, ArrowArray* out_array) {
  if (!reader || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS);
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

FFIResult get_chunks(ChunkReaderHandle reader,
                     const int64_t* chunk_indices,
                     size_t num_indices,
                     int64_t parallelism,
                     ArrowArray** arrays,
                     size_t* num_arrays) {
  if (!reader || !chunk_indices || num_indices == 0 || !arrays || !num_arrays) {
    RETURN_ERROR(LOON_INVALID_ARGS);
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
    // TODO: Implement RecordBatch to ArrowArray conversion
    // This requires converting Arrow C++ RecordBatch to Arrow C ABI format
    // For now, just initialize the arrays to avoid crashes
    for (size_t i = 0; i < record_batches.size(); ++i) {
      memset(&(*arrays)[i], 0, sizeof(ArrowArray));
    }
  } else {
    *num_arrays = 0;
  }

  RETURN_SUCCESS();
}

void chunk_reader_destroy(ChunkReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<ChunkReader*>(reader);
  }
}

// ==================== Reader C Implementation ====================

FFIResult reader_new(char* manifest,
                     ArrowSchema* schema,
                     const char* const* needed_columns,
                     size_t num_columns,
                     const ::Properties* properties,
                     ReaderHandle* out_handle) {
  if (!manifest || !schema || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  std::unordered_map<std::string, std::string> properties_map;
  properties_map = convert_properties(properties);

  ArrowFileSystemConfig fs_config;
  auto [ok, failed_key] = create_file_system_config(properties_map, fs_config);
  if (!ok) {
    assert(properties_map.count(failed_key) != 0);
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", failed_key.c_str(), ", ",
                 properties_map[failed_key].c_str(), "]");
  }

  auto fs_result = CreateArrowFileSystem(fs_config);
  if (!fs_result.ok()) {
    // TODO: missing the error message in fs_result
    RETURN_ERROR(LOON_ARROW_ERROR, "Failed to create arrow file system");
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(fs_result.value());
  auto result = arrow::ImportSchema(schema);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
  }

  auto cpp_schema = result.ValueOrDie();
  auto cpp_properties = std::move(properties_map);
  auto cpp_needed_columns = convert_string_array(needed_columns, num_columns);
  // Parse the manifest, the manifest is a JSON string
  auto cpp_manifest = JsonManifestSerDe().Deserialize(std::string(manifest));
  auto cpp_reader = Reader::create(cpp_fs, cpp_manifest, cpp_schema, cpp_needed_columns, cpp_properties);
  auto raw_cpp_reader = reinterpret_cast<ReaderHandle>(cpp_reader.release());
  assert(raw_cpp_reader);
  *out_handle = raw_cpp_reader;

  RETURN_SUCCESS();
}

FFIResult get_record_batch_reader(ReaderHandle reader,
                                  const char* predicate,
                                  int64_t batch_size,
                                  int64_t buffer_size,
                                  ArrowArrayStream* out_array_stream) {
  if (!reader || !out_array_stream)
    RETURN_ERROR(LOON_INVALID_ARGS);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str, batch_size, buffer_size);
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }

    auto array_stream = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatchReader(array_stream, out_array_stream);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (...) {  // TODO: make sure which exception will be throw
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle) {
  if (!reader || !out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS);

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    auto result = cpp_reader->get_chunk_reader(column_group_id);
    if (!result.ok())
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<ChunkReaderHandle>(chunk_reader);
    RETURN_SUCCESS();
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

FFIResult take(
    ReaderHandle reader, const int64_t* row_indices, size_t num_indices, int64_t parallelism, ArrowArray* out_arrays) {
  if (!reader || !row_indices || num_indices == 0 || !out_arrays)
    RETURN_ERROR(LOON_INVALID_ARGS);

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
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION);
  }

  RETURN_UNREACHABLE();
}

void reader_destroy(ReaderHandle reader) {
  if (reader) {
    delete reinterpret_cast<Reader*>(reader);
  }
}