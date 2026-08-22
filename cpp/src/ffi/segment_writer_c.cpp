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
#include "milvus-storage/common/arrow_util.h"

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
  if (!config->segment_path) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: config.segment_path must not be null");
  }
  if (config->segment_path[0] == '\0') {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: config.segment_path must not be empty");
  }
  if (config->num_lob_columns > 0 && config->lob_columns == nullptr) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: config.lob_columns must not be null when num_lob_columns is greater than 0");
  }
  for (size_t i = 0; i < config->num_lob_columns; ++i) {
    if (config->lob_columns[i].lob_base_path == nullptr) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: config.lob_columns contains a null lob_base_path [index=", i,
                   "]");
    }
    if (config->lob_columns[i].lob_base_path[0] == '\0') {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT,
                   "Invalid arguments: config.lob_columns contains an empty lob_base_path [index=", i, "]");
    }
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
    RETURN_ARROW_ERROR_IF(schema_result.status(), LOON_ARROW_ERROR, schema_result.status().ToString());
    auto schema = schema_result.ValueOrDie();
    auto field_id_status = ValidateFieldIds(schema);
    RETURN_ARROW_ERROR_IF(field_id_status, LOON_USER_INVALID_ARGUMENT, field_id_status.ToString());

    // convert config
    auto config_result = ConvertWriterConfig(config, props_map);
    RETURN_ARROW_ERROR_IF(config_result.status(), LOON_ARROW_ERROR, config_result.status().ToString());
    auto cpp_config = config_result.ValueOrDie();

    // resolve filesystem from properties via the fs cache
    auto fs_result = FilesystemCache::getInstance().get(props_map, cpp_config.segment_path);
    RETURN_ARROW_ERROR_IF(fs_result.status(), LOON_ARROW_ERROR, fs_result.status().ToString());
    auto fs = std::move(fs_result).ValueOrDie();

    // create segment writer
    auto writer_result = SegmentWriter::Create(fs, schema, cpp_config);
    RETURN_ARROW_ERROR_IF(writer_result.status(), LOON_ARROW_ERROR, writer_result.status().ToString());

    auto writer = std::move(writer_result).ValueOrDie();
    *out_handle = reinterpret_cast<LoonSegmentWriterHandle>(writer.release());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
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
      RETURN_ARROW_ERROR(rb_result.status(), LOON_ARROW_ERROR, rb_result.status().ToString());
    }
    auto batch = rb_result.ValueOrDie();

    auto status = writer->Write(batch);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
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
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_writer_close(LoonSegmentWriterHandle handle, LoonSegmentWriteOutput* out_output) {
  if (!handle || !out_output) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_output must not be null");
  }

  out_output->column_groups = nullptr;
  out_output->lob_files = nullptr;
  out_output->num_lob_files = 0;
  out_output->rows_written = 0;

  try {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    auto result = writer->Close();
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto output = std::move(result).ValueOrDie();

    auto cgs = output.column_groups;
    auto export_st = milvus_storage::column_groups_export(*cgs, &out_output->column_groups);
    RETURN_ARROW_ERROR_IF(export_st, LOON_ARROW_ERROR, export_st.ToString());
    out_output->rows_written = output.rows_written;

    if (!output.lob_files.empty()) {
      out_output->lob_files = static_cast<LoonLobFileInfo*>(calloc(output.lob_files.size(), sizeof(LoonLobFileInfo)));
      if (out_output->lob_files == nullptr) {
        loon_column_groups_destroy(out_output->column_groups);
        out_output->column_groups = nullptr;
        out_output->rows_written = 0;
        RETURN_ERROR(LOON_INTERNAL_INVARIANT, "Unexpected allocation failure for LOB file metadata");
      }
      out_output->num_lob_files = output.lob_files.size();
      for (size_t i = 0; i < output.lob_files.size(); i++) {
        out_output->lob_files[i].path = strdup(output.lob_files[i].path.c_str());
        if (out_output->lob_files[i].path == nullptr) {
          loon_segment_write_output_free(out_output);
          loon_column_groups_destroy(out_output->column_groups);
          out_output->column_groups = nullptr;
          out_output->rows_written = 0;
          RETURN_ERROR(LOON_INTERNAL_INVARIANT, "Unexpected allocation failure while copying LOB file path");
        }
        out_output->lob_files[i].field_id = output.lob_files[i].field_id;
        out_output->lob_files[i].total_rows = output.lob_files[i].total_rows;
        out_output->lob_files[i].valid_rows = output.lob_files[i].valid_rows;
        out_output->lob_files[i].file_size_bytes = output.lob_files[i].file_size_bytes;
      }
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
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
    output->num_lob_files = 0;
  }
}

// A handle destroyed without a successful close is the C caller saying it
// gives up on this writer. That is an explicit abandonment, not a destructor
// side effect, so it is where the abort belongs: R2.7 keeps storage I/O out of
// destruction, and this is the last frame that still knows the writer existed.
// Whatever the writer holds in the store -- above all an S3 multipart upload,
// whose parts no bucket listing can even show -- is released here or never.
void loon_segment_writer_destroy(LoonSegmentWriterHandle handle) {
  if (handle) {
    auto* writer = reinterpret_cast<SegmentWriter*>(handle);
    try {
      writer->Abort();
    } catch (...) {
      // Destruction is best effort and must not throw across the C ABI.
    }
    delete writer;
  }
}
