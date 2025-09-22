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

#include "milvus-storage/reader_c.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/manifest_json.h"
#include "milvus-storage/result_c.h"
#include "milvus-storage/result_internal.h"

#include <arrow/c/helpers.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

using namespace milvus_storage::api;

// Helper function to convert C string array to std::vector
static inline std::shared_ptr<std::vector<std::string>> convert_string_array(const char* const* strings, size_t count) {
  std::vector<std::string> result;
  if (strings && count > 0) {
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (strings[i]) {
        result.emplace_back(strings[i]);
      }
    }
  }
  return std::make_shared<std::vector<std::string>>(result);
}

// Helper function to convert C ReadProperties to C++ ReadProperties
static inline std::unordered_map<std::string, std::string> convert_read_properties(const ::ReadProperties* properties) {
  std::unordered_map<std::string, std::string> result;
  if (properties && properties->properties && properties->count > 0) {
    for (size_t i = 0; i < properties->count; ++i) {
      const auto& prop = properties->properties[i];
      if (prop.key && prop.value) {
        result[prop.key] = prop.value;
      }
    }
  }
  return result;
}

// ==================== ReadProperties C Implementation ====================

FFIResult read_properties_create(const char* const* keys,
                                 const char* const* values,
                                 size_t count,
                                 ::ReadProperties* properties) {
  if (!properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "properties should not be empty");
  }

  properties->properties = nullptr;
  properties->count = 0;

  if (count == 0 || !keys || !values || !*keys || !*values) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid keys/values");
  }

  properties->properties = static_cast<ReadProperty*>(malloc(sizeof(ReadProperty) * count));
  if (!properties->properties) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to malloc [size=", sizeof(ReadProperty) * count, "]");
  }

  for (size_t i = 0; i < count; ++i) {
    properties->properties[i].key = nullptr;
    properties->properties[i].value = nullptr;

    if (keys[i]) {
      size_t key_len = strlen(keys[i]) + 1;
      properties->properties[i].key = static_cast<char*>(malloc(key_len));
      if (properties->properties[i].key) {
        strcpy(properties->properties[i].key, keys[i]);
      }
    }

    if (values[i]) {
      size_t value_len = strlen(values[i]) + 1;
      properties->properties[i].value = static_cast<char*>(malloc(value_len));
      if (properties->properties[i].value) {
        strcpy(properties->properties[i].value, values[i]);
      }
    }
  }

  properties->count = count;
  RETURN_SUCCESS();
}

const char* read_properties_get(const ::ReadProperties* properties, const char* key) {
  if (!properties || !properties->properties || !key) {
    return nullptr;
  }

  for (size_t i = 0; i < properties->count; ++i) {
    if (properties->properties[i].key && strcmp(properties->properties[i].key, key) == 0) {
      return properties->properties[i].value;
    }
  }

  return nullptr;
}

void read_properties_free(::ReadProperties* properties) {
  if (!properties) {
    return;
  }

  if (properties->properties) {
    for (size_t i = 0; i < properties->count; ++i) {
      free(properties->properties[i].key);
      free(properties->properties[i].value);
    }
    free(properties->properties);
  }
}

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

  auto result = cpp_reader->get_chunk_indices(input_indices);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
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

FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, ArrowArray* out_array) {
  if (!reader || !out_array) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  auto* cpp_reader = reinterpret_cast<ChunkReader*>(reader);
  auto result = cpp_reader->get_chunk(chunk_index);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }
  auto record_batch = result.ValueOrDie();
  arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_array);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
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
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
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

FFIResult reader_new(FileSystemHandle fs,
                     char* manifest,
                     ArrowSchema* schema,
                     const char* const* needed_columns,
                     size_t num_columns,
                     const ::ReadProperties* properties,
                     ReaderHandle* out_handle) {
  if (!fs || !manifest || !schema || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS);
  }

  auto cpp_fs = std::shared_ptr<arrow::fs::FileSystem>(reinterpret_cast<arrow::fs::FileSystem*>(fs));
  auto result = arrow::ImportSchema(schema);
  if (!result.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
  }

  auto cpp_schema = result.ValueOrDie();
  auto cpp_properties = convert_read_properties(properties);
  auto cpp_needed_columns = convert_string_array(needed_columns, num_columns);
  // Parse the manifest, the manifest is a JSON string
  std::istringstream manifest_stream(manifest);
  milvus_storage::JsonManifestSerDe serde;
  auto cpp_manifest = serde.Deserialize(manifest_stream);
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
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());
    }

    auto array_stream = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatchReader(array_stream, out_array_stream);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
    }

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
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());

    // Transfer ownership to a raw pointer for C interface
    auto* chunk_reader = result.ValueOrDie().release();

    *out_handle = reinterpret_cast<ChunkReaderHandle>(chunk_reader);
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
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString().c_str());

    // export the arrow::RecordBatch to Arrow C ABI Array
    auto record_batch = result.ValueOrDie();
    arrow::Status status = arrow::ExportRecordBatch(*record_batch, out_arrays);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString().c_str());
    }

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