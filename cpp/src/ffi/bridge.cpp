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
#include <assert.h>

#include "milvus-storage/manifest.h"

namespace milvus_storage {
using namespace milvus_storage::api;

struct ColumnGroupFileExporter {
  public:
  void Export(const ColumnGroupFile* cgf, CColumnGroupFile* ccgf) {
    path_ = cgf->path;
    ccgf->path = path_.c_str();

    ccgf->start_index = cgf->start_index;
    ccgf->end_index = cgf->end_index;
    if (!cgf->metadata.empty()) {
      metadata_ = cgf->metadata;  // copy
      ccgf->metadata = metadata_.data();
      ccgf->metadata_size = metadata_.size();
    } else {
      ccgf->metadata = nullptr;
      ccgf->metadata_size = 0;
    }
  }

  private:
  std::string path_;
  std::vector<uint8_t> metadata_;
};

struct ColumnGroupExporter {
  public:
  void Export(const ColumnGroup* cg, CColumnGroup* ccg) {
    assert(cg != nullptr && ccg != nullptr);

    // export columns
    size_t num_of_columns = cg->columns.size();
    columns_holder_ = std::make_unique<const char*[]>(num_of_columns);
    colnames_.resize(num_of_columns);
    for (size_t i = 0; i < num_of_columns; i++) {
      colnames_[i] = cg->columns[i];
      columns_holder_[i] = colnames_[i].c_str();
    }
    ccg->columns = columns_holder_.get();
    ccg->num_of_columns = num_of_columns;

    // export format
    format_ = cg->format;
    ccg->format = format_.c_str();

    // export files
    size_t num_of_files = cg->files.size();
    files_holder_ = std::make_unique<CColumnGroupFile[]>(num_of_files);
    cgf_exporters_.resize(num_of_files);
    for (size_t i = 0; i < num_of_files; i++) {
      cgf_exporters_[i] = std::make_unique<ColumnGroupFileExporter>();
      cgf_exporters_[i]->Export(&cg->files[i], files_holder_.get() + i);
    }
    ccg->files = files_holder_.get();
    ccg->num_of_files = num_of_files;
  }

  private:
  std::vector<std::string> colnames_;
  std::unique_ptr<const char*[]> columns_holder_;

  std::string format_;

  std::unique_ptr<CColumnGroupFile[]> files_holder_;
  std::vector<std::unique_ptr<ColumnGroupFileExporter>> cgf_exporters_;
};

struct ColumnGroupsExporter {
  public:
  static arrow::Status Export(const ColumnGroups& cgs, CColumnGroups* out_ccgs) {
    ColumnGroupsExporter* exporter = new ColumnGroupsExporter();
    return exporter->ExportInternal(cgs, out_ccgs);
  }

  private:
  static void Release(CColumnGroups* ccg) { ccg->release = nullptr; }

  arrow::Status ExportInternal(const ColumnGroups& cgs, CColumnGroups* ccgs) {
    assert(ccgs != nullptr);
    ccgp_holder_ = std::make_unique<CColumnGroup[]>(cgs.size());
    cg_exporters_.resize(cgs.size());
    for (size_t i = 0; i < cgs.size(); i++) {
      cg_exporters_[i] = std::make_unique<ColumnGroupExporter>();
      cg_exporters_[i]->Export(cgs[i].get(), ccgp_holder_.get() + i);
    }
    ccgs->column_group_array = ccgp_holder_.get();
    ccgs->num_of_column_groups = cgs.size();

    ccgs->release = ColumnGroupsExporter::Release;

    return arrow::Status::OK();
  }

  private:
  std::unique_ptr<CColumnGroup[]> ccgp_holder_;
  std::vector<std::unique_ptr<ColumnGroupExporter>> cg_exporters_;
};

struct ColumnGroupsImporter {
  public:
  static void ImportColumnGroupFile(const CColumnGroupFile* in_ccgf, ColumnGroupFile* cgf) {
    assert(in_ccgf != nullptr && cgf != nullptr);
    cgf->path = std::string(in_ccgf->path);
    cgf->start_index = in_ccgf->start_index;
    cgf->end_index = in_ccgf->end_index;

    if (in_ccgf->metadata != nullptr) {
      cgf->metadata = std::vector<uint8_t>(in_ccgf->metadata, in_ccgf->metadata + in_ccgf->metadata_size);
    }
  }

  static void ImportColumnGroup(const CColumnGroup* in_ccg, ColumnGroup* cg) {
    assert(in_ccg != nullptr && cg != nullptr);
    for (size_t i = 0; i < in_ccg->num_of_columns; i++) {
      cg->columns.emplace_back(std::string(in_ccg->columns[i]));
    }
    cg->format = std::string(in_ccg->format);

    for (size_t i = 0; i < in_ccg->num_of_files; i++) {
      ColumnGroupFile cgf;
      ImportColumnGroupFile(&in_ccg->files[i], &cgf);
      cg->files.emplace_back(std::move(cgf));
    }
  }

  static arrow::Status Import(const CColumnGroups* in_ccgs, ColumnGroups* cgs) {
    assert(in_ccgs != nullptr && cgs != nullptr);
    cgs->clear();
    cgs->reserve(in_ccgs->num_of_column_groups);
    for (size_t i = 0; i < in_ccgs->num_of_column_groups; i++) {
      std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
      ImportColumnGroup(&in_ccgs->column_group_array[i], cg.get());
      cgs->push_back(cg);
    }

    // Metadata removed from ColumnGroups - no longer imported
    return arrow::Status::OK();
  }
};

arrow::Status export_column_groups(const ColumnGroups& cgs, CColumnGroups* out_ccgs) {
  return ColumnGroupsExporter::Export(cgs, out_ccgs);
}

arrow::Status import_column_groups(const CColumnGroups* ccgs, ColumnGroups* out_cgs) {
  return ColumnGroupsImporter::Import(ccgs, out_cgs);
}

struct ManifestExporter {
  public:
  static arrow::Status Export(const std::shared_ptr<Manifest>& manifest, CManifest* out_cmanifest) {
    ManifestExporter* exporter = new ManifestExporter();
    auto status = exporter->ExportInternal(manifest, out_cmanifest);
    if (status.ok()) {
      // Store exporter instance in private_data for cleanup
      out_cmanifest->private_data = exporter;
    } else {
      delete exporter;
    }
    return status;
  }

  private:
  static void Release(CManifest* cmanifest) {
    if (cmanifest && cmanifest->private_data) {
      ManifestExporter* exporter = static_cast<ManifestExporter*>(cmanifest->private_data);
      delete exporter;
      cmanifest->private_data = nullptr;
    }
    if (cmanifest) {
      cmanifest->release = nullptr;
    }
  }

  arrow::Status ExportInternal(const std::shared_ptr<Manifest>& manifest, CManifest* cmanifest) {
    assert(manifest != nullptr && cmanifest != nullptr);

    // Export column groups
    const auto& cgs = manifest->columnGroups();
    ccgp_holder_ = std::make_unique<CColumnGroup[]>(cgs.size());
    cg_exporters_.resize(cgs.size());
    for (size_t i = 0; i < cgs.size(); i++) {
      cg_exporters_[i] = std::make_unique<ColumnGroupExporter>();
      cg_exporters_[i]->Export(cgs[i].get(), ccgp_holder_.get() + i);
    }
    cmanifest->column_groups.column_group_array = ccgp_holder_.get();
    cmanifest->column_groups.num_of_column_groups = cgs.size();
    cmanifest->column_groups.release = nullptr;  // Will be handled by Manifest release

    // Export delta logs (only PRIMARY_KEY type for FFI)
    const auto& delta_logs = manifest->deltaLogs();
    delta_log_paths_.clear();
    delta_log_num_entries_.clear();
    for (const auto& delta_log : delta_logs) {
      if (delta_log.type == DeltaLogType::PRIMARY_KEY) {
        delta_log_paths_.push_back(delta_log.path);
        delta_log_num_entries_.push_back(delta_log.num_entries);
      }
    }
    if (!delta_log_paths_.empty()) {
      delta_log_paths_holder_ = std::make_unique<const char*[]>(delta_log_paths_.size());
      for (size_t i = 0; i < delta_log_paths_.size(); i++) {
        delta_log_paths_holder_[i] = delta_log_paths_[i].c_str();
      }
      cmanifest->delta_log_paths = delta_log_paths_holder_.get();
      cmanifest->delta_log_num_entries = delta_log_num_entries_.data();
      cmanifest->num_delta_logs = delta_log_paths_.size();
    } else {
      cmanifest->delta_log_paths = nullptr;
      cmanifest->delta_log_num_entries = nullptr;
      cmanifest->num_delta_logs = 0;
    }

    // Export stats
    const auto& stats = manifest->stats();
    stat_keys_.clear();
    stat_files_holders_.clear();
    stat_file_counts_.clear();
    for (const auto& [key, files] : stats) {
      stat_keys_.push_back(key);
      auto files_holder = std::make_unique<const char*[]>(files.size());
      for (size_t i = 0; i < files.size(); i++) {
        files_holder[i] = files[i].c_str();
      }
      stat_files_holders_.push_back(std::move(files_holder));
      stat_file_counts_.push_back(files.size());
    }
    if (!stat_keys_.empty()) {
      stat_keys_holder_ = std::make_unique<const char*[]>(stat_keys_.size());
      stat_files_array_holder_ = std::make_unique<const char**[]>(stat_keys_.size());
      for (size_t i = 0; i < stat_keys_.size(); i++) {
        stat_keys_holder_[i] = stat_keys_[i].c_str();
        stat_files_array_holder_[i] = stat_files_holders_[i].get();
      }
      cmanifest->stat_keys = stat_keys_holder_.get();
      cmanifest->stat_files = stat_files_array_holder_.get();
      cmanifest->stat_file_counts = stat_file_counts_.data();
      cmanifest->num_stats = stat_keys_.size();
    } else {
      cmanifest->stat_keys = nullptr;
      cmanifest->stat_files = nullptr;
      cmanifest->stat_file_counts = nullptr;
      cmanifest->num_stats = 0;
    }

    cmanifest->release = ManifestExporter::Release;

    return arrow::Status::OK();
  }

  private:
  std::unique_ptr<CColumnGroup[]> ccgp_holder_;
  std::vector<std::unique_ptr<ColumnGroupExporter>> cg_exporters_;

  std::vector<std::pair<std::string, std::string>> metadata_holder_;
  std::unique_ptr<const char*[]> meta_keys_holder_;
  std::unique_ptr<const char*[]> meta_values_holder_;

  std::vector<std::string> delta_log_paths_;
  std::unique_ptr<const char*[]> delta_log_paths_holder_;
  std::vector<int64_t> delta_log_num_entries_;

  std::vector<std::string> stat_keys_;
  std::unique_ptr<const char*[]> stat_keys_holder_;
  std::vector<std::unique_ptr<const char*[]>> stat_files_holders_;
  std::unique_ptr<const char**[]> stat_files_array_holder_;
  std::vector<uint32_t> stat_file_counts_;
};

struct ManifestImporter {
  public:
  static arrow::Status Import(const CManifest* cmanifest, std::shared_ptr<Manifest>* out_manifest) {
    assert(cmanifest != nullptr && out_manifest != nullptr);

    // Import column groups
    ColumnGroups cgs;
    cgs.reserve(cmanifest->column_groups.num_of_column_groups);
    for (size_t i = 0; i < cmanifest->column_groups.num_of_column_groups; i++) {
      std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
      ColumnGroupsImporter::ImportColumnGroup(&cmanifest->column_groups.column_group_array[i], cg.get());
      cgs.push_back(cg);
    }

    // Import delta logs (only PRIMARY_KEY type supported in FFI)
    std::vector<DeltaLog> delta_logs;
    delta_logs.reserve(cmanifest->num_delta_logs);
    for (uint32_t i = 0; i < cmanifest->num_delta_logs; i++) {
      DeltaLog delta_log;
      delta_log.path = std::string(cmanifest->delta_log_paths[i]);
      delta_log.type = DeltaLogType::PRIMARY_KEY;
      delta_log.num_entries = cmanifest->delta_log_num_entries[i];
      delta_logs.push_back(delta_log);
    }

    // Import stats
    std::map<std::string, std::vector<std::string>> stats;
    for (uint32_t i = 0; i < cmanifest->num_stats; i++) {
      std::string key(cmanifest->stat_keys[i]);
      std::vector<std::string> files;
      files.reserve(cmanifest->stat_file_counts[i]);
      for (uint32_t j = 0; j < cmanifest->stat_file_counts[i]; j++) {
        files.push_back(std::string(cmanifest->stat_files[i][j]));
      }
      stats[key] = std::move(files);
    }

    // Create Manifest
    *out_manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats);

    return arrow::Status::OK();
  }
};

arrow::Status export_manifest(const std::shared_ptr<milvus_storage::api::Manifest>& manifest,
                              CManifest* out_cmanifest) {
  return ManifestExporter::Export(manifest, out_cmanifest);
}

arrow::Status import_manifest(const CManifest* cmanifest,
                              std::shared_ptr<milvus_storage::api::Manifest>* out_manifest) {
  return ManifestImporter::Import(cmanifest, out_manifest);
}

}  // namespace milvus_storage
