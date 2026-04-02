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

  if (!config->segment_path) {
    return arrow::Status::Invalid("segment_path is required");
  }
  cpp_config.segment_path = config->segment_path;
  cpp_config.properties = properties;

  // convert TEXT column configs
  for (size_t i = 0; i < config->num_lob_columns; i++) {
    const auto& tc = config->lob_columns[i];
    lob_column::LobColumnConfig text_config;
    text_config.field_id = tc.field_id;
    if (tc.lob_base_path) {
      text_config.lob_base_path = tc.lob_base_path;
    }
    text_config.inline_threshold = tc.inline_threshold > 0 ? tc.inline_threshold : 256;
    text_config.max_lob_file_bytes = tc.max_lob_file_bytes > 0 ? tc.max_lob_file_bytes : 64 * 1024 * 1024;
    text_config.flush_threshold_bytes = tc.flush_threshold_bytes > 0 ? tc.flush_threshold_bytes : 16 * 1024 * 1024;
    text_config.rewrite_mode = tc.rewrite_mode;
    text_config.properties = properties;

    cpp_config.lob_columns[tc.field_id] = text_config;
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

LoonFFIResult loon_segment_writer_close(LoonSegmentWriterHandle handle, LoonSegmentWriteOutput* out_output) {
  if (!handle || !out_output) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_output must not be null");
  }

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    auto result = writer->Close();
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }

    auto output = std::move(result).ValueOrDie();

    auto cgs = output.column_groups;
    auto export_st = milvus_storage::column_groups_export(*cgs, &out_output->column_groups);
    if (!export_st.ok()) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, export_st.ToString());
    }
    out_output->rows_written = output.rows_written;

    out_output->num_lob_files = output.lob_files.size();
    if (!output.lob_files.empty()) {
      out_output->lob_files = static_cast<LoonLobFileInfo*>(malloc(sizeof(LoonLobFileInfo) * output.lob_files.size()));
      for (size_t i = 0; i < output.lob_files.size(); i++) {
        out_output->lob_files[i].path = strdup(output.lob_files[i].path.c_str());
        out_output->lob_files[i].field_id = output.lob_files[i].field_id;
        out_output->lob_files[i].total_rows = output.lob_files[i].total_rows;
        out_output->lob_files[i].valid_rows = output.lob_files[i].valid_rows;
        out_output->lob_files[i].file_size_bytes = output.lob_files[i].file_size_bytes;
      }
    } else {
      out_output->lob_files = nullptr;
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_segment_write_output_free(LoonSegmentWriteOutput* output) {
  if (output) {
    if (output->lob_files) {
      for (size_t i = 0; i < output->num_lob_files; i++) {
        free(const_cast<char*>(output->lob_files[i].path));
      }
      free(output->lob_files);
      output->lob_files = nullptr;
    }
  }
}

void loon_segment_writer_destroy(LoonSegmentWriterHandle handle) {
  if (handle) {
    delete reinterpret_cast<SegmentWriter*>(handle);
  }
}
