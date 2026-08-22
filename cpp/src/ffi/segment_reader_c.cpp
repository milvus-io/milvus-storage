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
#include "milvus-storage/segment/segment_reader.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/common/arrow_util.h"

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include <limits>

using namespace milvus_storage;
using namespace milvus_storage::api;
using namespace milvus_storage::segment;

static arrow::Result<SegmentReaderConfig> ConvertReaderConfig(const LoonSegmentReaderConfig* config,
                                                              const api::Properties& properties) {
  SegmentReaderConfig cpp_config;
  cpp_config.properties = properties;

  if (config->read_buffer_size > 0) {
    cpp_config.read_buffer_size = static_cast<size_t>(config->read_buffer_size);
  }

  for (size_t i = 0; i < config->num_lob_columns; i++) {
    const auto& tc = config->lob_columns[i];
    lob_column::LobColumnConfig lob_config;
    lob_config.field_id = tc.field_id;
    if (tc.lob_base_path) {
      lob_config.lob_base_path = tc.lob_base_path;
    }
    lob_config.properties = properties;
    cpp_config.lob_columns[tc.field_id] = lob_config;
  }

  return cpp_config;
}

static std::optional<LoonFFIResult> ValidateReaderConfig(const LoonSegmentReaderConfig* config) {
  if (config->num_lob_columns > 0 && config->lob_columns == nullptr) {
    return CreateFFIResult(LOON_INVALID_ARGS,
                           "config.lob_columns must not be null when num_lob_columns is greater than 0");
  }
  if (config->read_buffer_size < 0) {
    return CreateFFIResult(LOON_USER_INVALID_ARGUMENT, "config.read_buffer_size must be greater than or equal to 0");
  }
  for (size_t i = 0; i < config->num_lob_columns; ++i) {
    if (config->lob_columns[i].lob_base_path == nullptr) {
      return CreateFFIResult(LOON_INVALID_ARGS, "config.lob_columns contains a null lob_base_path [index=", i, "]");
    }
    if (config->lob_columns[i].lob_base_path[0] == '\0') {
      return CreateFFIResult(LOON_USER_INVALID_ARGUMENT,
                             "config.lob_columns contains an empty lob_base_path [index=", i, "]");
    }
  }
  return std::nullopt;
}

LoonFFIResult loon_segment_reader_open(const char* segment_path,
                                       int64_t version,
                                       ArrowSchema* schema_raw,
                                       const char** needed_columns,
                                       int64_t num_columns,
                                       const LoonSegmentReaderConfig* config,
                                       const LoonProperties* properties,
                                       LoonSegmentReaderHandle* out_handle) {
  if (!segment_path || !schema_raw || !config || !properties || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: segment_path, schema_raw, config, properties, and out_handle must not be null");
  }
  if (segment_path[0] == '\0') {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: segment_path must not be empty");
  }
  if (version < api::transaction::LATEST) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: version must be -1 or greater");
  }
  if (num_columns < 0) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: num_columns must be greater than or equal to 0");
  }
  if (auto invalid_config = ValidateReaderConfig(config); invalid_config.has_value()) {
    return std::move(invalid_config).value();
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
    auto config_result = ConvertReaderConfig(config, props_map);
    RETURN_ARROW_ERROR_IF(config_result.status(), LOON_ARROW_ERROR, config_result.status().ToString());
    auto cpp_config = config_result.ValueOrDie();

    // build needed columns
    std::vector<std::string> columns;
    if (needed_columns && num_columns > 0) {
      for (int64_t i = 0; i < num_columns; i++) {
        columns.push_back(needed_columns[i]);
      }
    }

    // resolve filesystem from properties via the fs cache
    auto fs_result = FilesystemCache::getInstance().get(props_map, segment_path);
    RETURN_ARROW_ERROR_IF(fs_result.status(), LOON_ARROW_ERROR, fs_result.status().ToString());
    auto fs = std::move(fs_result).ValueOrDie();

    // open transaction to read manifest
    auto txn_result = api::transaction::Transaction::Open(fs, segment_path, version, api::transaction::FailResolver, 1);
    RETURN_ARROW_ERROR_IF(txn_result.status(), LOON_ARROW_ERROR, txn_result.status().ToString());
    auto txn = std::move(txn_result).ValueOrDie();

    auto manifest_result = txn->GetManifest();
    RETURN_ARROW_ERROR_IF(manifest_result.status(), LOON_ARROW_ERROR, manifest_result.status().ToString());
    auto manifest = std::move(manifest_result).ValueOrDie();

    // open reader from manifest
    auto reader_result = SegmentReader::Open(fs, manifest, schema, columns, cpp_config);
    RETURN_ARROW_ERROR_IF(reader_result.status(), LOON_ARROW_ERROR, reader_result.status().ToString());

    auto reader = std::move(reader_result).ValueOrDie();
    *out_handle = reinterpret_cast<LoonSegmentReaderHandle>(reader.release());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_reader_get_stream(LoonSegmentReaderHandle handle, ArrowArrayStream* out_stream) {
  if (!handle || !out_stream) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_stream must not be null");
  }

  try {
    auto* reader = reinterpret_cast<SegmentReader*>(handle);

    // SegmentReader implements arrow::RecordBatchReader, export as stream.
    // Use a no-op destructor since the handle is owned externally (destroyed via loon_segment_reader_destroy).
    auto status = arrow::ExportRecordBatchReader(
        std::shared_ptr<arrow::RecordBatchReader>(reader, [](arrow::RecordBatchReader*) {}), out_stream);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_reader_take(LoonSegmentReaderHandle handle,
                                       const int64_t* row_indices,
                                       int64_t num_indices,
                                       int64_t parallelism,
                                       ArrowArrayStream* out_stream) {
  try {
    if (!handle || !out_stream) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_stream must not be null");
    }

    if (!row_indices) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: row_indices must not be null");
    }
    if (num_indices <= 0) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: num_indices must be > 0");
    }
    if (parallelism < 0) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid arguments: parallelism must be >= 0");
    }
    for (int64_t i = 0; i < num_indices; ++i) {
      if (row_indices[i] < 0) {
        RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "Invalid row index at position ", i, ": ", row_indices[i],
                     " (must be non-negative)");
      }
      if (i > 0 && row_indices[i] <= row_indices[i - 1]) {
        RETURN_ERROR(LOON_USER_INVALID_ARGUMENT, "row_indices must be strictly increasing; position ", i, " has value ",
                     row_indices[i], " after ", row_indices[i - 1]);
      }
    }

    auto* reader = reinterpret_cast<SegmentReader*>(handle);
    auto column_groups = reader->GetColumnGroups();
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
    size_t par = parallelism == 0 ? 1 : static_cast<size_t>(parallelism);

    auto result = reader->Take(indices, par);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());
    auto table = std::move(result).ValueOrDie();

    // Convert Table to RecordBatchReader, then export as ArrowArrayStream
    auto batch_reader = std::make_shared<arrow::TableBatchReader>(std::move(table));
    auto status = arrow::ExportRecordBatchReader(batch_reader, out_stream);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_reader_get_filtered_stream(LoonSegmentReaderHandle handle,
                                                      const char* predicate,
                                                      ArrowArrayStream* out_stream) {
  if (!handle || !out_stream) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_stream must not be null");
  }

  try {
    auto* reader = reinterpret_cast<SegmentReader*>(handle);
    std::string pred = predicate ? predicate : "";

    auto result = reader->GetStream(pred);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());
    auto batch_reader = std::move(result).ValueOrDie();

    auto status = arrow::ExportRecordBatchReader(batch_reader, out_stream);
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, status.ToString());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

LoonFFIResult loon_segment_reader_get_chunk_reader(LoonSegmentReaderHandle handle,
                                                   int64_t column_group_index,
                                                   const char* const* needed_columns,
                                                   size_t num_columns,
                                                   LoonChunkReaderHandle* out_handle) {
  if (!handle || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle and out_handle must not be null");
  }
  if (column_group_index < 0) {
    RETURN_ERROR(LOON_USER_INVALID_ARGUMENT,
                 "Invalid arguments: column_group_index must be greater than or equal to 0");
  }

  try {
    auto* reader = reinterpret_cast<SegmentReader*>(handle);
    const auto& column_groups = reader->GetColumnGroups();
    if (column_groups == nullptr || static_cast<size_t>(column_group_index) >= column_groups->size()) {
      RETURN_ERROR(LOON_USER_INVALID_ARGUMENT,
                   "Invalid arguments: column_group_index is out of range [index=", column_group_index,
                   ", size=", column_groups == nullptr ? 0 : column_groups->size(), "]");
    }

    std::shared_ptr<std::vector<std::string>> columns;
    if (needed_columns && num_columns > 0) {
      columns = std::make_shared<std::vector<std::string>>();
      for (size_t i = 0; i < num_columns; i++) {
        columns->push_back(needed_columns[i]);
      }
    }

    auto result = reader->GetChunkReader(column_group_index, columns);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, result.status().ToString());

    auto chunk_reader = std::move(result).ValueOrDie();
    *out_handle = reinterpret_cast<LoonChunkReaderHandle>(chunk_reader.release());

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  } catch (...) {
    RETURN_EXCEPTION("unknown exception");
  }

  RETURN_UNREACHABLE();
}

void loon_segment_reader_destroy(LoonSegmentReaderHandle handle) {
  if (handle) {
    delete reinterpret_cast<SegmentReader*>(handle);
  }
}
