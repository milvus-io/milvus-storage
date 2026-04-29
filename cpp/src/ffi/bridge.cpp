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

static arrow::Status import_column_group_file(const LoonColumnGroupFile* in_ccgf, ColumnGroupFile* cgf) {
  if (!in_ccgf || !cgf) {
    return arrow::Status::Invalid("column group file and output must not be null");
  }
  if (!in_ccgf->path) {
    return arrow::Status::Invalid("column group file path is null");
  }
  if (in_ccgf->start_index < 0 || in_ccgf->end_index < 0 || in_ccgf->start_index > in_ccgf->end_index) {
    return arrow::Status::Invalid("column group file row range is invalid");
  }
  if (in_ccgf->num_properties > 0 && (!in_ccgf->property_keys || !in_ccgf->property_values)) {
    return arrow::Status::Invalid("column group file property arrays are null with nonzero count");
  }
  cgf->path = std::string(in_ccgf->path);
  cgf->start_index = in_ccgf->start_index;
  cgf->end_index = in_ccgf->end_index;

  for (uint32_t i = 0; i < in_ccgf->num_properties; ++i) {
    if (!in_ccgf->property_keys[i] || !in_ccgf->property_values[i]) {
      return arrow::Status::Invalid("column group file property key/value is null");
    }
    cgf->properties[in_ccgf->property_keys[i]] = in_ccgf->property_values[i];
  }
  return arrow::Status::OK();
}

static arrow::Status import_column_group(const LoonColumnGroup* in_ccg, ColumnGroup* cg) {
  if (!in_ccg || !cg) {
    return arrow::Status::Invalid("column group and output must not be null");
  }
  if (!in_ccg->format) {
    return arrow::Status::Invalid("column group format is null");
  }
  if (in_ccg->num_of_columns > 0 && !in_ccg->columns) {
    return arrow::Status::Invalid("column group columns array is null with nonzero count");
  }
  if (in_ccg->num_of_files > 0 && !in_ccg->files) {
    return arrow::Status::Invalid("column group files array is null with nonzero count");
  }
  for (size_t i = 0; i < in_ccg->num_of_columns; i++) {
    if (!in_ccg->columns[i]) {
      return arrow::Status::Invalid("column group column entry is null");
    }
    cg->columns.emplace_back(in_ccg->columns[i]);
  }
  cg->format = std::string(in_ccg->format);

  for (size_t i = 0; i < in_ccg->num_of_files; i++) {
    ColumnGroupFile cgf;
    ARROW_RETURN_NOT_OK(import_column_group_file(&in_ccg->files[i], &cgf));
    cg->files.emplace_back(std::move(cgf));
  }
  return arrow::Status::OK();
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
  if (!ccgs || !out_cgs) {
    return arrow::Status::Invalid("column groups and output must not be null");
  }
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
    ARROW_RETURN_NOT_OK(import_column_group(&ccgs->column_group_array[i], cg.get()));
    out_cgs->push_back(cg);
  }
  return arrow::Status::OK();
}

arrow::Status manifest_export(const std::shared_ptr<milvus_storage::api::Manifest>& manifest,
                              LoonManifest** out_cmanifest) {
  assert(manifest != nullptr && out_cmanifest != nullptr);

  for (const auto& delta_log : manifest->deltaLogs()) {
    if (delta_log.type == DeltaLogType::PRIMARY_KEY && delta_log.num_entries <= 0) {
      return arrow::Status::Invalid("delta log num_entries must be positive");
    }
  }

  try {
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

    // Export column groups directly into embedded structure
    const auto& cgs = manifest->columnGroups();
    ARROW_RETURN_NOT_OK(column_groups_export_internal(cgs, &(*out_cmanifest)->column_groups));

    // Export delta logs (only PRIMARY_KEY type for FFI)
    const auto& delta_logs = manifest->deltaLogs();
    std::vector<std::string> delta_log_paths;
    std::vector<int64_t> delta_log_num_entries;
    for (const auto& delta_log : delta_logs) {
      if (delta_log.type == DeltaLogType::PRIMARY_KEY) {
        delta_log_paths.push_back(delta_log.path);
        delta_log_num_entries.push_back(delta_log.num_entries);
      }
    }
    if (!delta_log_paths.empty()) {
      // Assign arrays immediately so destroy functions can clean up on exception
      (*out_cmanifest)->delta_logs.delta_log_paths = new const char* [delta_log_paths.size()] {};
      (*out_cmanifest)->delta_logs.delta_log_num_entries = new int64_t[delta_log_paths.size()];
      (*out_cmanifest)->delta_logs.num_delta_logs = static_cast<uint32_t>(delta_log_paths.size());

      for (size_t i = 0; i < delta_log_paths.size(); i++) {
        size_t len = delta_log_paths[i].length();
        char* path_str = new char[len + 1];
        std::memcpy(path_str, delta_log_paths[i].c_str(), len);
        path_str[len] = '\0';
        (*out_cmanifest)->delta_logs.delta_log_paths[i] = path_str;
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
        // Copy key
        size_t key_len = key.length();
        char* key_str = new char[key_len + 1];
        std::memcpy(key_str, key.c_str(), key_len);
        key_str[key_len] = '\0';
        (*out_cmanifest)->stats.stat_keys[idx] = key_str;

        // Copy file paths
        size_t num_files = stat.paths.size();
        (*out_cmanifest)->stats.stat_files[idx] = new const char* [num_files] {};
        for (size_t j = 0; j < num_files; j++) {
          size_t file_len = stat.paths[j].length();
          char* file_str = new char[file_len + 1];
          std::memcpy(file_str, stat.paths[j].c_str(), file_len);
          file_str[file_len] = '\0';
          (*out_cmanifest)->stats.stat_files[idx][j] = file_str;
        }
        (*out_cmanifest)->stats.stat_file_counts[idx] = num_files;

        // Copy metadata
        size_t num_metadata = stat.metadata.size();
        if (num_metadata > 0) {
          (*out_cmanifest)->stats.stat_metadata_keys[idx] = new const char* [num_metadata] {};
          (*out_cmanifest)->stats.stat_metadata_values[idx] = new const char* [num_metadata] {};
          size_t m_idx = 0;
          for (const auto& [meta_key, meta_val] : stat.metadata) {
            size_t mk_len = meta_key.length();
            char* mk_str = new char[mk_len + 1];
            std::memcpy(mk_str, meta_key.c_str(), mk_len);
            mk_str[mk_len] = '\0';
            (*out_cmanifest)->stats.stat_metadata_keys[idx][m_idx] = mk_str;

            size_t mv_len = meta_val.length();
            char* mv_str = new char[mv_len + 1];
            std::memcpy(mv_str, meta_val.c_str(), mv_len);
            mv_str[mv_len] = '\0';
            (*out_cmanifest)->stats.stat_metadata_values[idx][m_idx] = mv_str;
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

        // Copy path
        size_t path_len = lob_file.path.length();
        char* path_str = new char[path_len + 1];
        std::memcpy(path_str, lob_file.path.c_str(), path_len);
        path_str[path_len] = '\0';
        out_lob.path = path_str;

        out_lob.field_id = lob_file.field_id;
        out_lob.total_rows = lob_file.total_rows;
        out_lob.valid_rows = lob_file.valid_rows;
        out_lob.file_size_bytes = lob_file.file_size_bytes;
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
  if (!cmanifest || !out_manifest) {
    return arrow::Status::Invalid("manifest and output must not be null");
  }
  *out_manifest = nullptr;

  // Import column groups
  ColumnGroups cgs;
  ARROW_RETURN_NOT_OK(column_groups_import(&cmanifest->column_groups, &cgs));

  // Import delta logs (only PRIMARY_KEY type supported in FFI)
  std::vector<DeltaLog> delta_logs;
  delta_logs.reserve(cmanifest->delta_logs.num_delta_logs);
  if (cmanifest->delta_logs.num_delta_logs > 0) {
    if (!cmanifest->delta_logs.delta_log_paths || !cmanifest->delta_logs.delta_log_num_entries) {
      return arrow::Status::Invalid("delta log arrays must not be null with nonzero count");
    }
  }
  for (uint32_t i = 0; i < cmanifest->delta_logs.num_delta_logs; i++) {
    if (!cmanifest->delta_logs.delta_log_paths[i]) {
      return arrow::Status::Invalid("delta log path is null");
    }
    if (cmanifest->delta_logs.delta_log_num_entries[i] <= 0) {
      return arrow::Status::Invalid("delta log num_entries must be positive");
    }
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

  // Create Manifest
  *out_manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats);

  return arrow::Status::OK();
}

std::string column_groups_debug_string(const LoonColumnGroups* ccgs) {
  if (ccgs == nullptr) {
    return "LoonColumnGroups(null)";
  }

  std::string result = fmt::format("LoonColumnGroups(num_of_column_groups={})\n", ccgs->num_of_column_groups);
  if (ccgs->num_of_column_groups > 0 && !ccgs->column_group_array) {
    result += "  column_group_array: (null)\n";
    return result;
  }

  for (uint32_t i = 0; i < ccgs->num_of_column_groups; i++) {
    const auto& cg = ccgs->column_group_array[i];
    result += fmt::format("  ColumnGroup[{}]:\n", i);
    result += fmt::format("    format: {}\n", cg.format ? cg.format : "(null)");
    result += fmt::format("    num_of_columns: {}\n", cg.num_of_columns);
    result += "    columns: [";
    if (cg.num_of_columns > 0 && !cg.columns) {
      result += "(null array)";
    } else {
      for (uint32_t j = 0; j < cg.num_of_columns; j++) {
        if (j > 0) {
          result += ", ";
        }
        result += cg.columns[j] ? cg.columns[j] : "(null)";
      }
    }
    result += "]\n";
    result += fmt::format("    num_of_files: {}\n", cg.num_of_files);
    if (cg.num_of_files > 0 && !cg.files) {
      result += "    files: (null array)\n";
      continue;
    }
    for (uint32_t j = 0; j < cg.num_of_files; j++) {
      const auto& f = cg.files[j];
      result += fmt::format("      File[{}]: path={}, start_index={}, end_index={}, num_properties={}\n", j,
                            f.path ? f.path : "(null)", f.start_index, f.end_index, f.num_properties);
      if (f.num_properties > 0 && (!f.property_keys || !f.property_values)) {
        result += "        properties: (null array)\n";
        continue;
      }
      for (uint32_t k = 0; k < f.num_properties; k++) {
        result += fmt::format("        {}={}\n", f.property_keys[k] ? f.property_keys[k] : "(null)",
                              f.property_values[k] ? f.property_values[k] : "(null)");
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
    const char* path = (cmanifest->delta_logs.delta_log_paths && cmanifest->delta_logs.delta_log_paths[i])
                           ? cmanifest->delta_logs.delta_log_paths[i]
                           : "(null)";
    std::string num_entries = cmanifest->delta_logs.delta_log_num_entries
                                  ? fmt::format("{}", cmanifest->delta_logs.delta_log_num_entries[i])
                                  : "(null)";
    result += fmt::format("    DeltaLog[{}]: path={}, num_entries={}\n", i, path, num_entries);
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

  return result;
}

}  // namespace milvus_storage
