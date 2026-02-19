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

  // Copy metadata
  if (!cgf->metadata.empty()) {
    ccgf->metadata = new uint8_t[cgf->metadata.size()];
    std::copy(cgf->metadata.begin(), cgf->metadata.end(), ccgf->metadata);
    ccgf->metadata_size = cgf->metadata.size();
  } else {
    ccgf->metadata = nullptr;
    ccgf->metadata_size = 0;
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

  if (in_ccgf->metadata != nullptr) {
    cgf->metadata = std::vector<uint8_t>(in_ccgf->metadata, in_ccgf->metadata + in_ccgf->metadata_size);
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
    (*out_cmanifest)->stats.num_stats = 0;

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
      (*out_cmanifest)->delta_logs.delta_log_paths = new const char*[delta_log_paths.size()]{};
      (*out_cmanifest)->delta_logs.delta_log_num_entries = new uint32_t[delta_log_paths.size()];
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
      (*out_cmanifest)->stats.stat_keys = new const char*[num_stats]{};
      (*out_cmanifest)->stats.stat_files = new const char** [num_stats] {};
      (*out_cmanifest)->stats.stat_file_counts = new uint32_t[num_stats];
      (*out_cmanifest)->stats.num_stats = num_stats;

      size_t idx = 0;
      for (const auto& [key, files] : stats) {
        // Copy key
        size_t key_len = key.length();
        char* key_str = new char[key_len + 1];
        std::memcpy(key_str, key.c_str(), key_len);
        key_str[key_len] = '\0';
        (*out_cmanifest)->stats.stat_keys[idx] = key_str;

        // Copy files
        size_t num_files = files.size();
        (*out_cmanifest)->stats.stat_files[idx] = new const char*[num_files]{};
        for (size_t j = 0; j < num_files; j++) {
          size_t file_len = files[j].length();
          char* file_str = new char[file_len + 1];
          std::memcpy(file_str, files[j].c_str(), file_len);
          file_str[file_len] = '\0';
          (*out_cmanifest)->stats.stat_files[idx][j] = file_str;
        }
        (*out_cmanifest)->stats.stat_file_counts[idx] = num_files;
        idx++;
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
  std::map<std::string, std::vector<std::string>> stats;
  for (uint32_t i = 0; i < cmanifest->stats.num_stats; i++) {
    std::string key(cmanifest->stats.stat_keys[i]);
    std::vector<std::string> files;
    files.reserve(cmanifest->stats.stat_file_counts[i]);
    for (uint32_t j = 0; j < cmanifest->stats.stat_file_counts[i]; j++) {
      files.emplace_back(cmanifest->stats.stat_files[i][j]);
    }
    stats[key] = std::move(files);
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

  for (uint32_t i = 0; i < ccgs->num_of_column_groups; i++) {
    const auto& cg = ccgs->column_group_array[i];
    result += fmt::format("  ColumnGroup[{}]:\n", i);
    result += fmt::format("    format: {}\n", cg.format ? cg.format : "(null)");
    result += fmt::format("    num_of_columns: {}\n", cg.num_of_columns);
    result += "    columns: [";
    for (uint32_t j = 0; j < cg.num_of_columns; j++) {
      if (j > 0)
        result += ", ";
      result += cg.columns[j] ? cg.columns[j] : "(null)";
    }
    result += "]\n";
    result += fmt::format("    num_of_files: {}\n", cg.num_of_files);
    for (uint32_t j = 0; j < cg.num_of_files; j++) {
      const auto& f = cg.files[j];
      result += fmt::format("      File[{}]: path={}, start_index={}, end_index={}, metadata_size={}\n", j,
                            f.path ? f.path : "(null)", f.start_index, f.end_index, f.metadata_size);
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
    result += fmt::format("    Stat[{}]: key={}, num_files={}\n", i,
                          cmanifest->stats.stat_keys[i] ? cmanifest->stats.stat_keys[i] : "(null)",
                          cmanifest->stats.stat_file_counts[i]);
    for (uint32_t j = 0; j < cmanifest->stats.stat_file_counts[i]; j++) {
      result += fmt::format("      file[{}]: {}\n", j,
                            cmanifest->stats.stat_files[i][j] ? cmanifest->stats.stat_files[i][j] : "(null)");
    }
  }

  return result;
}

}  // namespace milvus_storage
