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

#include "milvus-storage/ffi_internal/bridge.h"

#include <memory>
#include <optional>
#include <vector>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <fmt/format.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/ffi_c.h"

namespace milvus_storage {
using namespace milvus_storage::api;

static void export_column_group_file(const ColumnGroupFile* cgf, LoonColumnGroupFile* ccgf) {
  // Copy path
  size_t path_len = cgf->path.length();
  char* path = new char[path_len + 1];
  std::memcpy(path, cgf->path.c_str(), path_len);
  path[path_len] = '\0';
  ccgf->path = path;

  ccgf->start_index = cgf->start_index;
  ccgf->end_index = cgf->end_index;

  // Copy properties
  size_t num_props = cgf->properties.size();
  ccgf->num_properties = num_props;
  if (num_props > 0) {
    ccgf->property_keys = new const char*[num_props];
    ccgf->property_values = new const char*[num_props];
    size_t idx = 0;
    for (const auto& [k, v] : cgf->properties) {
      auto* key = new char[k.size() + 1];
      std::memcpy(key, k.c_str(), k.size() + 1);
      ccgf->property_keys[idx] = key;
      auto* val = new char[v.size() + 1];
      std::memcpy(val, v.c_str(), v.size() + 1);
      ccgf->property_values[idx] = val;
      ++idx;
    }
  } else {
    ccgf->property_keys = nullptr;
    ccgf->property_values = nullptr;
  }
}

static void export_column_group(const ColumnGroup* cg, LoonColumnGroup* ccg) {
  assert(cg != nullptr && ccg != nullptr);

  // export columns - allocate memory for column names
  size_t num_of_columns = cg->columns.size();
  const char** columns = new const char*[num_of_columns];
  for (size_t i = 0; i < num_of_columns; i++) {
    size_t len = cg->columns[i].length();
    char* col_str = new char[len + 1];
    std::memcpy(col_str, cg->columns[i].c_str(), len);
    col_str[len] = '\0';
    columns[i] = col_str;
  }
  ccg->columns = columns;
  ccg->num_of_columns = num_of_columns;

  // export format - allocate memory for format string
  size_t format_len = cg->format.length();
  char* format = new char[format_len + 1];
  std::memcpy(format, cg->format.c_str(), format_len);
  format[format_len] = '\0';
  ccg->format = format;

  // export files
  size_t num_of_files = cg->files.size();
  auto* files = new LoonColumnGroupFile[num_of_files];
  for (size_t i = 0; i < num_of_files; i++) {
    export_column_group_file(&cg->files[i], files + i);
  }
  ccg->files = files;
  ccg->num_of_files = num_of_files;
}

static void import_column_group_file(const LoonColumnGroupFile* in_ccgf, ColumnGroupFile* cgf) {
  assert(in_ccgf != nullptr && cgf != nullptr);
  cgf->path = std::string(in_ccgf->path);
  cgf->start_index = in_ccgf->start_index;
  cgf->end_index = in_ccgf->end_index;

  for (uint32_t i = 0; i < in_ccgf->num_properties; ++i) {
    cgf->properties[in_ccgf->property_keys[i]] = in_ccgf->property_values[i];
  }
}

static void import_column_group(const LoonColumnGroup* in_ccg, ColumnGroup* cg) {
  assert(in_ccg != nullptr && cg != nullptr);
  for (size_t i = 0; i < in_ccg->num_of_columns; i++) {
    cg->columns.emplace_back(in_ccg->columns[i]);
  }
  cg->format = std::string(in_ccg->format);

  for (size_t i = 0; i < in_ccg->num_of_files; i++) {
    ColumnGroupFile cgf;
    import_column_group_file(&in_ccg->files[i], &cgf);
    cg->files.emplace_back(std::move(cgf));
  }
}

// Core logic to populate an already-allocated LoonColumnGroups structure
static arrow::Status column_groups_export_internal(const ColumnGroups& cgs, LoonColumnGroups* out_ccgs) {
  assert(out_ccgs != nullptr);

  out_ccgs->column_group_array = nullptr;
  out_ccgs->num_of_column_groups = 0;

  out_ccgs->column_group_array = new LoonColumnGroup[cgs.size()]{};
  // Assign array immediately so destroy functions can clean up on exception
  out_ccgs->num_of_column_groups = cgs.size();

  for (size_t i = 0; i < cgs.size(); i++) {
    export_column_group(cgs[i].get(), out_ccgs->column_group_array + i);
  }
  return arrow::Status::OK();
}

arrow::Status column_groups_export(const ColumnGroups& cgs, LoonColumnGroups** out_ccgs) {
  assert(out_ccgs != nullptr);

  try {
    *out_ccgs = new LoonColumnGroups();
    ARROW_RETURN_NOT_OK(column_groups_export_internal(cgs, *out_ccgs));
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    if (*out_ccgs) {
      loon_column_groups_destroy(*out_ccgs);
      *out_ccgs = nullptr;
    }
    return arrow::Status::UnknownError("Exception in column_groups_export: ", e.what());
  } catch (...) {
    if (*out_ccgs) {
      loon_column_groups_destroy(*out_ccgs);
      *out_ccgs = nullptr;
    }
    return arrow::Status::UnknownError("Unknown exception in column_groups_export");
  }
}

arrow::Status column_groups_import(const LoonColumnGroups* ccgs, ColumnGroups* out_cgs) {
  assert(ccgs != nullptr && out_cgs != nullptr);
  out_cgs->clear();
  if (ccgs->num_of_column_groups == 0) {
    return arrow::Status::OK();
  }
  if (!ccgs->column_group_array) {
    return arrow::Status::Invalid("column_group_array is null");
  }
  out_cgs->reserve(ccgs->num_of_column_groups);
  for (size_t i = 0; i < ccgs->num_of_column_groups; i++) {
    std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
    import_column_group(&ccgs->column_group_array[i], cg.get());
    out_cgs->push_back(cg);
  }
  return arrow::Status::OK();
}

arrow::Status manifest_export(const std::shared_ptr<milvus_storage::api::Manifest>& manifest,
                              LoonManifest** out_cmanifest) {
  assert(manifest != nullptr && out_cmanifest != nullptr);

  try {
    const auto copy_string = [](const std::string& value) {
      auto* result = new char[value.size() + 1];
      std::memcpy(result, value.c_str(), value.size() + 1);
      return result;
    };

    // Value-initialize to ensure all pointers are nullptr
    *out_cmanifest = new LoonManifest{};
    (*out_cmanifest)->column_groups.column_group_array = nullptr;
    (*out_cmanifest)->column_groups.num_of_column_groups = 0;
    (*out_cmanifest)->delta_logs.delta_log_paths = nullptr;
    (*out_cmanifest)->delta_logs.delta_log_num_entries = nullptr;
    (*out_cmanifest)->delta_logs.num_delta_logs = 0;
    (*out_cmanifest)->stats.stat_keys = nullptr;
    (*out_cmanifest)->stats.stat_files = nullptr;
    (*out_cmanifest)->stats.stat_file_counts = nullptr;
    (*out_cmanifest)->stats.stat_metadata_keys = nullptr;
    (*out_cmanifest)->stats.stat_metadata_values = nullptr;
    (*out_cmanifest)->stats.stat_metadata_counts = nullptr;
    (*out_cmanifest)->stats.num_stats = 0;
    (*out_cmanifest)->lob_files.files = nullptr;
    (*out_cmanifest)->lob_files.num_files = 0;
    (*out_cmanifest)->indexes.indexes = nullptr;
    (*out_cmanifest)->indexes.num_indexes = 0;

    // Export column groups directly into embedded structure
    const auto& cgs = manifest->columnGroups();
    ARROW_RETURN_NOT_OK(column_groups_export_internal(cgs, &(*out_cmanifest)->column_groups));

    // Export delta logs (only PRIMARY_KEY type for FFI)
    const auto& delta_logs = manifest->deltaLogs();
    std::vector<std::string> delta_log_paths;
    std::vector<uint32_t> delta_log_num_entries;
    for (const auto& delta_log : delta_logs) {
      if (delta_log.type == DeltaLogType::PRIMARY_KEY) {
        delta_log_paths.push_back(delta_log.path);
        delta_log_num_entries.push_back(static_cast<uint32_t>(delta_log.num_entries));
      }
    }
    if (!delta_log_paths.empty()) {
      // Assign arrays immediately so destroy functions can clean up on exception
      (*out_cmanifest)->delta_logs.delta_log_paths = new const char* [delta_log_paths.size()] {};
      (*out_cmanifest)->delta_logs.delta_log_num_entries = new uint32_t[delta_log_paths.size()];
      (*out_cmanifest)->delta_logs.num_delta_logs = static_cast<uint32_t>(delta_log_paths.size());

      for (size_t i = 0; i < delta_log_paths.size(); i++) {
        (*out_cmanifest)->delta_logs.delta_log_paths[i] = copy_string(delta_log_paths[i]);
        (*out_cmanifest)->delta_logs.delta_log_num_entries[i] = delta_log_num_entries[i];
      }
    }

    // Export stats
    const auto& stats = manifest->stats();
    if (!stats.empty()) {
      size_t num_stats = stats.size();
      (*out_cmanifest)->stats.stat_keys = new const char* [num_stats] {};
      (*out_cmanifest)->stats.stat_files = new const char** [num_stats] {};
      (*out_cmanifest)->stats.stat_file_counts = new uint32_t[num_stats];
      (*out_cmanifest)->stats.stat_metadata_keys = new const char** [num_stats] {};
      (*out_cmanifest)->stats.stat_metadata_values = new const char** [num_stats] {};
      (*out_cmanifest)->stats.stat_metadata_counts = new uint32_t[num_stats];
      (*out_cmanifest)->stats.num_stats = num_stats;

      size_t idx = 0;
      for (const auto& [key, stat] : stats) {
        (*out_cmanifest)->stats.stat_keys[idx] = copy_string(key);

        // Copy file paths
        size_t num_files = stat.paths.size();
        (*out_cmanifest)->stats.stat_files[idx] = new const char* [num_files] {};
        for (size_t j = 0; j < num_files; j++) {
          (*out_cmanifest)->stats.stat_files[idx][j] = copy_string(stat.paths[j]);
        }
        (*out_cmanifest)->stats.stat_file_counts[idx] = num_files;

        // Copy metadata
        size_t num_metadata = stat.metadata.size();
        if (num_metadata > 0) {
          (*out_cmanifest)->stats.stat_metadata_keys[idx] = new const char* [num_metadata] {};
          (*out_cmanifest)->stats.stat_metadata_values[idx] = new const char* [num_metadata] {};
          size_t m_idx = 0;
          for (const auto& [meta_key, meta_val] : stat.metadata) {
            (*out_cmanifest)->stats.stat_metadata_keys[idx][m_idx] = copy_string(meta_key);
            (*out_cmanifest)->stats.stat_metadata_values[idx][m_idx] = copy_string(meta_val);
            m_idx++;
          }
        }
        (*out_cmanifest)->stats.stat_metadata_counts[idx] = num_metadata;
        idx++;
      }
    }

    // Export LOB files
    const auto& lob_files = manifest->lobFiles();
    if (!lob_files.empty()) {
      size_t num_lob_files = lob_files.size();
      (*out_cmanifest)->lob_files.files = new LoonLobFileInfo[num_lob_files]{};
      (*out_cmanifest)->lob_files.num_files = static_cast<uint32_t>(num_lob_files);

      for (size_t i = 0; i < num_lob_files; i++) {
        const auto& lob_file = lob_files[i];
        auto& out_lob = (*out_cmanifest)->lob_files.files[i];

        out_lob.path = copy_string(lob_file.path);

        out_lob.field_id = lob_file.field_id;
        out_lob.total_rows = lob_file.total_rows;
        out_lob.valid_rows = lob_file.valid_rows;
        out_lob.file_size_bytes = lob_file.file_size_bytes;
      }
    }

    // Export indexes.
    const auto& indexes = manifest->indexes();
    if (!indexes.empty()) {
      // Publish each allocation in the output structure immediately. If a
      // later allocation throws, the catch block below delegates cleanup of
      // this partially constructed object to loon_manifest_destroy().
      (*out_cmanifest)->indexes.indexes = new LoonIndexInfo[indexes.size()]{};
      (*out_cmanifest)->indexes.num_indexes = static_cast<uint32_t>(indexes.size());

      for (size_t i = 0; i < indexes.size(); ++i) {
        const auto& index = indexes[i];
        auto& out_index = (*out_cmanifest)->indexes.indexes[i];

        out_index.column_name = copy_string(index.column_name);
        out_index.index_name = copy_string(index.index_name);
        out_index.index_type = copy_string(index.index_type);
        out_index.path = copy_string(index.path);
        out_index.field_id = index.field_id;
        out_index.index_id = index.index_id;
        out_index.build_id = index.build_id;
        out_index.index_version = index.index_version;
        out_index.num_rows = index.num_rows;
        out_index.serialized_size = index.serialized_size;
        out_index.mem_size = index.mem_size;
        out_index.current_index_version = index.current_index_version;
        out_index.current_scalar_index_version = index.current_scalar_index_version;
        out_index.index_store_path_version = index.index_store_path_version;
        out_index.num_index_file_keys = static_cast<uint32_t>(index.index_file_keys.size());
        if (!index.index_file_keys.empty()) {
          out_index.index_file_keys = new const char* [index.index_file_keys.size()] {};
          for (size_t file_index = 0; file_index < index.index_file_keys.size(); ++file_index) {
            out_index.index_file_keys[file_index] = copy_string(index.index_file_keys[file_index]);
          }
        }
        out_index.num_properties = static_cast<uint32_t>(index.properties.size());
        if (!index.properties.empty()) {
          out_index.property_keys = new const char* [index.properties.size()] {};
          out_index.property_values = new const char* [index.properties.size()] {};
          size_t property_index = 0;
          for (const auto& [key, value] : index.properties) {
            out_index.property_keys[property_index] = copy_string(key);
            out_index.property_values[property_index] = copy_string(value);
            ++property_index;
          }
        }
      }
    }

    return arrow::Status::OK();
  } catch (const std::exception& e) {
    if (*out_cmanifest) {
      loon_manifest_destroy(*out_cmanifest);
      *out_cmanifest = nullptr;
    }
    return arrow::Status::UnknownError("Exception in manifest_export: ", e.what());
  } catch (...) {
    if (*out_cmanifest) {
      loon_manifest_destroy(*out_cmanifest);
      *out_cmanifest = nullptr;
    }
    return arrow::Status::UnknownError("Unknown exception in manifest_export");
  }
}

arrow::Status manifest_import(const LoonManifest* cmanifest,
                              std::shared_ptr<milvus_storage::api::Manifest>* out_manifest) {
  assert(cmanifest != nullptr && out_manifest != nullptr);

  // Import column groups
  ColumnGroups cgs;
  cgs.reserve(cmanifest->column_groups.num_of_column_groups);
  for (size_t i = 0; i < cmanifest->column_groups.num_of_column_groups; i++) {
    std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
    import_column_group(&cmanifest->column_groups.column_group_array[i], cg.get());
    cgs.push_back(cg);
  }

  // Import delta logs (only PRIMARY_KEY type supported in FFI)
  std::vector<DeltaLog> delta_logs;
  delta_logs.reserve(cmanifest->delta_logs.num_delta_logs);
  for (uint32_t i = 0; i < cmanifest->delta_logs.num_delta_logs; i++) {
    DeltaLog delta_log;
    delta_log.path = std::string(cmanifest->delta_logs.delta_log_paths[i]);
    delta_log.type = DeltaLogType::PRIMARY_KEY;
    delta_log.num_entries = cmanifest->delta_logs.delta_log_num_entries[i];
    delta_logs.push_back(delta_log);
  }

  // Import stats
  std::map<std::string, Statistics> stats;
  for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
    std::string key(cmanifest->stats.stat_keys[i]);
    Statistics stat;
    stat.paths.reserve(cmanifest->stats.stat_file_counts[i]);
    for (uint32_t j = 0; j < cmanifest->stats.stat_file_counts[i]; j++) {
      stat.paths.emplace_back(cmanifest->stats.stat_files[i][j]);
    }
    if (cmanifest->stats.stat_metadata_keys && cmanifest->stats.stat_metadata_keys[i]) {
      for (uint32_t j = 0; j < cmanifest->stats.stat_metadata_counts[i]; j++) {
        stat.metadata[cmanifest->stats.stat_metadata_keys[i][j]] = cmanifest->stats.stat_metadata_values[i][j];
      }
    }
    stats[key] = std::move(stat);
  }

  // Import index metadata.
  std::vector<Index> indexes;
  indexes.reserve(cmanifest->indexes.num_indexes);
  for (uint32_t i = 0; i < cmanifest->indexes.num_indexes; ++i) {
    const auto& in_index = cmanifest->indexes.indexes[i];
    if (!in_index.column_name || !in_index.index_type || !in_index.path) {
      return arrow::Status::Invalid("Index metadata requires column_name, index_type, and path");
    }
    if (in_index.num_properties > 0 && (!in_index.property_keys || !in_index.property_values)) {
      return arrow::Status::Invalid("Index metadata properties are missing");
    }

    Index index;
    index.column_name = in_index.column_name;
    if (in_index.index_name) {
      index.index_name = in_index.index_name;
    }
    index.index_type = in_index.index_type;
    index.path = in_index.path;
    index.field_id = in_index.field_id;
    index.index_id = in_index.index_id;
    index.build_id = in_index.build_id;
    index.index_version = in_index.index_version;
    index.num_rows = in_index.num_rows;
    index.serialized_size = in_index.serialized_size;
    index.mem_size = in_index.mem_size;
    index.current_index_version = in_index.current_index_version;
    index.current_scalar_index_version = in_index.current_scalar_index_version;
    index.index_store_path_version = in_index.index_store_path_version;
    if (in_index.num_index_file_keys > 0 && !in_index.index_file_keys) {
      return arrow::Status::Invalid("Index metadata file keys are missing");
    }
    index.index_file_keys.reserve(in_index.num_index_file_keys);
    for (uint32_t j = 0; j < in_index.num_index_file_keys; ++j) {
      if (!in_index.index_file_keys[j]) {
        return arrow::Status::Invalid("Index metadata file key is null");
      }
      index.index_file_keys.emplace_back(in_index.index_file_keys[j]);
    }
    for (uint32_t j = 0; j < in_index.num_properties; ++j) {
      if (!in_index.property_keys[j] || !in_index.property_values[j]) {
        return arrow::Status::Invalid("Index metadata property key or value is null");
      }
      index.properties[in_index.property_keys[j]] = in_index.property_values[j];
    }
    indexes.push_back(std::move(index));
  }

  // Import LOB files too so an FFI manifest round trip preserves every field.
  std::vector<LobFileInfo> lob_files;
  lob_files.reserve(cmanifest->lob_files.num_files);
  for (uint32_t i = 0; i < cmanifest->lob_files.num_files; ++i) {
    const auto& in_lob = cmanifest->lob_files.files[i];
    if (!in_lob.path) {
      return arrow::Status::Invalid("LOB file path is null");
    }
    lob_files.emplace_back(in_lob.path, in_lob.field_id, in_lob.total_rows, in_lob.valid_rows, in_lob.file_size_bytes);
  }

  // Create Manifest
  *out_manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats, indexes, lob_files);

  return arrow::Status::OK();
}

std::string column_groups_debug_string(const LoonColumnGroups* ccgs) {
  if (ccgs == nullptr) {
    return "LoonColumnGroups(null)";
  }

  std::string result = fmt::format("LoonColumnGroups(num_of_column_groups={})\n", ccgs->num_of_column_groups);

  for (uint32_t i = 0; i < ccgs->num_of_column_groups; i++) {
    const auto& cg = ccgs->column_group_array[i];
    result += fmt::format("  ColumnGroup[{}]:\n", i);
    result += fmt::format("    format: {}\n", cg.format ? cg.format : "(null)");
    result += fmt::format("    num_of_columns: {}\n", cg.num_of_columns);
    result += "    columns: [";
    for (uint32_t j = 0; j < cg.num_of_columns; j++) {
      if (j > 0) {
        result += ", ";
      }
      result += cg.columns[j] ? cg.columns[j] : "(null)";
    }
    result += "]\n";
    result += fmt::format("    num_of_files: {}\n", cg.num_of_files);
    for (uint32_t j = 0; j < cg.num_of_files; j++) {
      const auto& f = cg.files[j];
      result += fmt::format("      File[{}]: path={}, start_index={}, end_index={}, num_properties={}\n", j,
                            f.path ? f.path : "(null)", f.start_index, f.end_index, f.num_properties);
      for (uint32_t k = 0; k < f.num_properties; k++) {
        result += fmt::format("        {}={}\n", f.property_keys[k], f.property_values[k]);
      }
    }
  }

  return result;
}

std::string manifest_debug_string(const LoonManifest* cmanifest) {
  if (cmanifest == nullptr) {
    return "LoonManifest(null)";
  }

  std::string result = "LoonManifest:\n";

  // Column groups
  result += "  " + column_groups_debug_string(&cmanifest->column_groups);

  // Delta logs
  result += fmt::format("  DeltaLogs(num_delta_logs={}):\n", cmanifest->delta_logs.num_delta_logs);
  for (uint32_t i = 0; i < cmanifest->delta_logs.num_delta_logs; i++) {
    result +=
        fmt::format("    DeltaLog[{}]: path={}, num_entries={}\n", i,
                    cmanifest->delta_logs.delta_log_paths[i] ? cmanifest->delta_logs.delta_log_paths[i] : "(null)",
                    cmanifest->delta_logs.delta_log_num_entries[i]);
  }

  // Stats
  result += fmt::format("  Stats(num_stats={}):\n", cmanifest->stats.num_stats);
  for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
    uint32_t num_metadata = cmanifest->stats.stat_metadata_counts ? cmanifest->stats.stat_metadata_counts[i] : 0;
    result += fmt::format("    Stat[{}]: key={}, num_files={}, num_metadata={}\n", i,
                          cmanifest->stats.stat_keys[i] ? cmanifest->stats.stat_keys[i] : "(null)",
                          cmanifest->stats.stat_file_counts[i], num_metadata);
    for (uint32_t j = 0; j < cmanifest->stats.stat_file_counts[i]; j++) {
      result += fmt::format("      file[{}]: {}\n", j,
                            cmanifest->stats.stat_files[i][j] ? cmanifest->stats.stat_files[i][j] : "(null)");
    }
    if (cmanifest->stats.stat_metadata_keys && cmanifest->stats.stat_metadata_keys[i]) {
      for (uint32_t j = 0; j < num_metadata; j++) {
        result += fmt::format("      metadata[{}]: {}={}\n", j, cmanifest->stats.stat_metadata_keys[i][j],
                              cmanifest->stats.stat_metadata_values[i][j]);
      }
    }
  }

  result += fmt::format("  LobFiles(num_files={}):\n", cmanifest->lob_files.num_files);
  for (uint32_t i = 0; i < cmanifest->lob_files.num_files; ++i) {
    const auto& lob_file = cmanifest->lob_files.files[i];
    result += fmt::format("    LobFile[{}]: path={}, field_id={}, total_rows={}, valid_rows={}, file_size_bytes={}\n",
                          i, lob_file.path ? lob_file.path : "(null)", lob_file.field_id, lob_file.total_rows,
                          lob_file.valid_rows, lob_file.file_size_bytes);
  }

  result += fmt::format("  Indexes(num_indexes={}):\n", cmanifest->indexes.num_indexes);
  for (uint32_t i = 0; i < cmanifest->indexes.num_indexes; ++i) {
    const auto& index = cmanifest->indexes.indexes[i];
    result += fmt::format(
        "    Index[{}]: column_name={}, field_id={}, index_name={}, index_type={}, index_id={}, build_id={}, "
        "index_version={}, "
        "path={}, num_files={}, serialized_size={}, mem_size={}, num_properties={}\n",
        i, index.column_name ? index.column_name : "(null)", index.field_id,
        index.index_name ? index.index_name : "(null)", index.index_type ? index.index_type : "(null)", index.index_id,
        index.build_id, index.index_version, index.path ? index.path : "(null)", index.num_index_file_keys,
        index.serialized_size, index.mem_size, index.num_properties);
  }

  return result;
}

}  // namespace milvus_storage
