// Copyright 2024 Zilliz
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
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/segment/segment_writer.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <cstring>

using namespace milvus_storage;
using namespace milvus_storage::api;
using namespace milvus_storage::segment;

// helper to convert FFI config to C++ config
static arrow::Result<SegmentWriterConfig> ConvertWriterConfig(const LoonSegmentWriterConfig* config,
                                                              const api::Properties& properties) {
  SegmentWriterConfig cpp_config;

  if (config->lob_base_path) {
    cpp_config.lob_base_path = config->lob_base_path;
  }

  if (!config->segment_path) {
    return arrow::Status::Invalid("segment_path is required");
  }
  cpp_config.segment_path = config->segment_path;

  cpp_config.read_version = config->read_version;
  cpp_config.retry_limit = config->retry_limit > 0 ? config->retry_limit : 1;
  cpp_config.properties = properties;

  // convert TEXT column configs
  for (size_t i = 0; i < config->num_text_columns; i++) {
    const auto& tc = config->text_columns[i];
    text_column::TextColumnConfig text_config;
    text_config.field_id = tc.field_id;
    if (tc.lob_base_path) {
      text_config.lob_base_path = tc.lob_base_path;
    }
    text_config.inline_threshold = tc.inline_threshold > 0 ? tc.inline_threshold : 256;
    text_config.max_lob_file_bytes = tc.max_lob_file_bytes > 0 ? tc.max_lob_file_bytes : 64 * 1024 * 1024;
    text_config.flush_threshold_bytes = tc.flush_threshold_bytes > 0 ? tc.flush_threshold_bytes : 16 * 1024 * 1024;
    text_config.properties = properties;

    cpp_config.text_columns[tc.field_id] = text_config;
  }

  return cpp_config;
}

LoonFFIResult loon_segment_writer_new(ArrowSchema* schema_raw,
                                      const LoonSegmentWriterConfig* config,
                                      const LoonProperties* properties,
                                      LoonSegmentWriterHandle* out_handle) {
  if (!schema_raw || !config || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: schema_raw, config, properties, and out_handle must not be null");
  }

  try {
    // convert properties
    api::Properties props_map;
    auto opt = ConvertFFIProperties(props_map, properties);
    if (opt != std::nullopt) {
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties: ", opt->c_str());
    }

    // import arrow schema
    auto schema_result = arrow::ImportSchema(schema_raw);
    if (!schema_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, schema_result.status().ToString());
    }
    auto schema = schema_result.ValueOrDie();

    // convert config
    auto config_result = ConvertWriterConfig(config, props_map);
    if (!config_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, config_result.status().ToString());
    }
    auto cpp_config = config_result.ValueOrDie();

    // get filesystem from singleton
    auto fs = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
    if (!fs) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Filesystem not initialized");
    }

    // create segment writer
    auto writer_result = SegmentWriter::Create(fs, schema, cpp_config);
    if (!writer_result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, writer_result.status().ToString());
    }

    auto writer = std::move(writer_result).ValueOrDie();
    *out_handle = reinterpret_cast<LoonSegmentWriterHandle>(writer.release());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_writer_write(LoonSegmentWriterHandle handle, ArrowArray* array) {
  if (!handle || !array) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and array must not be null");
  }

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);

    auto rb_result = arrow::ImportRecordBatch(array, writer->GetOriginalSchema());
    if (!rb_result.ok()) {
      array->release(array);
      RETURN_ERROR(LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto batch = rb_result.ValueOrDie();

    auto status = writer->Write(batch);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_writer_flush(LoonSegmentWriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    auto status = writer->Flush();
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_writer_close(LoonSegmentWriterHandle handle, LoonSegmentWriterResult* out_result) {
  if (!handle || !out_result) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_result must not be null");
  }

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    auto result = writer->Close();
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }

    auto cpp_result = result.ValueOrDie();

    // copy manifest path
    out_result->manifest_path = strdup(cpp_result.manifest_path.c_str());
    out_result->committed_version = cpp_result.committed_version;
    out_result->rows_written = cpp_result.rows_written;

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_writer_abort(LoonSegmentWriterHandle handle) {
  if (!handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle must not be null");
  }

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    auto status = writer->Abort();
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_segment_writer_destroy(LoonSegmentWriterHandle handle) {
  if (handle) {
    delete reinterpret_cast<SegmentWriter*>(handle);
  }
}

void loon_segment_writer_result_free(LoonSegmentWriterResult* result) {
  if (result) {
    if (result->manifest_path) {
      free(result->manifest_path);
      result->manifest_path = nullptr;
    }
  }
}
