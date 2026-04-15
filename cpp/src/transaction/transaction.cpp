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

#include "milvus-storage/transaction/transaction.h"

#include <cassert>
#include <charconv>
#include <set>
#include <fmt/format.h>

#include <arrow/status.h>
#include <arrow/result.h>
#include "milvus-storage/common/log.h"

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage::api::transaction {

// ==================== Updates Class Implementation ====================

Updates::Updates() = default;
Updates::~Updates() = default;

bool Updates::hasChanges() const {
  return !dropped_columns_.empty() || !added_column_groups_.empty() || !appended_files_.empty() ||
         !added_delta_logs_.empty() || !added_stats_.empty() || !added_indexes_.empty() || !dropped_indexes_.empty() ||
         !added_lob_files_.empty();
}

void Updates::DropColumn(const std::string& column_name) { dropped_columns_.push_back(column_name); }

void Updates::AddColumnGroup(const std::shared_ptr<ColumnGroup>& cg) { added_column_groups_.push_back(cg); }

void Updates::AppendFiles(const ColumnGroups& cgs) { appended_files_.push_back(cgs); }

void Updates::AddDeltaLog(const DeltaLog& delta_log) { added_delta_logs_.push_back(delta_log); }

void Updates::UpdateStat(const std::string& key, const Statistics& stat) { added_stats_[key] = stat; }

void Updates::AddLobFile(const LobFileInfo& lob_file) { added_lob_files_.push_back(lob_file); }

const std::vector<std::string>& Updates::GetDroppedColumns() const { return dropped_columns_; }

const ColumnGroups& Updates::GetAddedColumnGroups() const { return added_column_groups_; }

const std::vector<ColumnGroups>& Updates::GetAppendedFiles() const { return appended_files_; }

const std::vector<DeltaLog>& Updates::GetAddedDeltaLogs() const { return added_delta_logs_; }

const std::map<std::string, Statistics>& Updates::GetAddedStats() const { return added_stats_; }

void Updates::AddIndex(const Index& index) { added_indexes_.push_back(index); }

void Updates::DropIndex(const std::string& column_name, const std::string& index_type) {
  dropped_indexes_.emplace_back(column_name, index_type);
}

const std::vector<Index>& Updates::GetAddedIndexes() const { return added_indexes_; }

const std::vector<std::pair<std::string, std::string>>& Updates::GetDroppedIndexes() const { return dropped_indexes_; }

const std::vector<LobFileInfo>& Updates::GetAddedLobFiles() const { return added_lob_files_; }

// ==================== Helper Functions ====================

arrow::Result<std::shared_ptr<Manifest>> applyUpdates(const std::shared_ptr<Manifest>& manifest,
                                                      const Updates& updates) {
  // Deep copy the entire manifest to avoid mutating the (potentially cached) original
  auto base = std::make_shared<Manifest>(*manifest);

  auto& cgs = base->columnGroups();
  auto& delta_logs = base->deltaLogs();
  auto& stats = base->stats();
  auto& indexes = base->indexes();
  auto& lob_files = base->lobFiles();

  // Apply dropped columns (MUST execute before AddColumnGroup validation)
  // Noop if column doesn't exist — consistent with DropIndex, supports idempotency
  for (const auto& col_name : updates.GetDroppedColumns()) {
    // Phase 1: remove the column name from every column group that contains it
    for (const auto& cg : cgs) {
      if (!cg) {
        return arrow::Status::Invalid("Unexpected null column group in manifest");
      }
      auto& cols = cg->columns;
      cols.erase(std::remove(cols.begin(), cols.end(), col_name), cols.end());
    }

    // Phase 2: drop any column group that became empty after removal
    cgs.erase(std::remove_if(cgs.begin(), cgs.end(),
                             [](const std::shared_ptr<ColumnGroup>& cg) { return cg->columns.empty(); }),
              cgs.end());

    // Phase 3: auto-drop all indexes attached to this column
    indexes.erase(
        std::remove_if(indexes.begin(), indexes.end(), [&](const Index& idx) { return idx.column_name == col_name; }),
        indexes.end());
  }

  // Validate: Check if adding column groups has existing column names
  for (const auto& new_cg : updates.GetAddedColumnGroups()) {
    if (!new_cg) {
      return arrow::Status::Invalid("Cannot add null column group");
    }
    for (const auto& column_name : new_cg->columns) {
      if (base->getColumnGroup(column_name) != nullptr) {
        return arrow::Status::Invalid(fmt::format("Column '{}' already exists in existing column groups", column_name));
      }
    }

    // Check column group number of rows aligned
    size_t origin_group_rows = 0;
    if (!cgs.empty()) {
      for (const auto& f : cgs[0]->files) {
        origin_group_rows += f.end_index - f.start_index;
      }
    }

    size_t new_group_rows = 0;
    for (const auto& f : new_cg->files) {
      new_group_rows += f.end_index - f.start_index;
    }

    if (!cgs.empty() && origin_group_rows != new_group_rows) {
      return arrow::Status::Invalid(fmt::format(
          "Column group size mismatch: existing(column group 0) has {} rows, but appended column group has {} rows",
          origin_group_rows, new_group_rows));
    }
  }

  // Validate: Check if appending files are aligned with existing column groups
  for (const auto& new_cgs : updates.GetAppendedFiles()) {
    if (!cgs.empty() && cgs.size() != new_cgs.size()) {
      return arrow::Status::Invalid(
          fmt::format("Column group size mismatch: existing has {} groups, but appended has {} groups", cgs.size(),
                      new_cgs.size()));
    }

    for (size_t i = 0; i < cgs.size() && i < new_cgs.size(); ++i) {
      if (!cgs[i] || !new_cgs[i]) {
        return arrow::Status::Invalid(fmt::format("Null column group at index {}", i));
      }
      if (cgs[i]->columns.size() != new_cgs[i]->columns.size()) {
        return arrow::Status::Invalid(
            fmt::format("Column count mismatch at index {}: existing has {} columns, but appended has {} columns", i,
                        cgs[i]->columns.size(), new_cgs[i]->columns.size()));
      }
      if (cgs[i]->format != new_cgs[i]->format) {
        return arrow::Status::Invalid(
            fmt::format("Format mismatch at index {}: existing format is '{}', but appended format is '{}'", i,
                        cgs[i]->format, new_cgs[i]->format));
      }
      std::set<std::string> base_cols(cgs[i]->columns.begin(), cgs[i]->columns.end());
      std::set<std::string> new_cols(new_cgs[i]->columns.begin(), new_cgs[i]->columns.end());
      if (base_cols != new_cols) {
        return arrow::Status::Invalid(fmt::format("Column mismatch at index {}: columns do not match", i));
      }
    }
  }

  // Collect columns affected by AppendFiles (for index deprecation)
  std::set<std::string> affected_columns;
  for (const auto& new_cgs : updates.GetAppendedFiles()) {
    for (const auto& cg : new_cgs) {
      if (cg && !cg->files.empty()) {
        for (const auto& col : cg->columns) {
          affected_columns.insert(col);
        }
      }
    }
  }

  // Resolve indexes: drop affected, apply DropIndex, apply AddIndex
  indexes.erase(std::remove_if(
                    indexes.begin(), indexes.end(),
                    [&](const Index& idx) { return affected_columns.find(idx.column_name) != affected_columns.end(); }),
                indexes.end());

  for (const auto& dropped : updates.GetDroppedIndexes()) {
    const auto& col = dropped.first;
    const auto& type = dropped.second;
    indexes.erase(std::remove_if(indexes.begin(), indexes.end(),
                                 [&](const Index& idx) { return idx.column_name == col && idx.index_type == type; }),
                  indexes.end());
  }

  for (const auto& new_idx : updates.GetAddedIndexes()) {
    indexes.erase(std::remove_if(indexes.begin(), indexes.end(),
                                 [&](const Index& idx) {
                                   return idx.column_name == new_idx.column_name &&
                                          idx.index_type == new_idx.index_type;
                                 }),
                  indexes.end());
    indexes.push_back(new_idx);
  }

  // Apply delta logs
  for (const auto& dl : updates.GetAddedDeltaLogs()) {
    delta_logs.push_back(dl);
  }

  // Apply stats (new values override)
  for (const auto& [key, stat] : updates.GetAddedStats()) {
    stats[key] = stat;
  }

  // Apply LOB files
  for (const auto& lob_file : updates.GetAddedLobFiles()) {
    lob_files.push_back(lob_file);
  }

  // Apply append files
  for (const auto& new_cgs : updates.GetAppendedFiles()) {
    if (cgs.empty()) {
      cgs = new_cgs;
    } else {
      for (size_t i = 0; i < cgs.size() && i < new_cgs.size(); ++i) {
        if (cgs[i] && new_cgs[i]) {
          for (const auto& file : new_cgs[i]->files) {
            cgs[i]->files.push_back(file);
          }
        }
      }
    }
  }

  // Apply add column groups
  for (const auto& cg : updates.GetAddedColumnGroups()) {
    cgs.push_back(cg);
  }

  return base;
}

// ==================== Helper Resolver Functions ====================

Resolver MergeResolver = [](const std::shared_ptr<Manifest>& /*read_manifest*/,
                            int64_t /*read_version*/,
                            const std::shared_ptr<Manifest>& latest_manifest,
                            int64_t /*latest_version*/,
                            const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  return applyUpdates(latest_manifest, updates);
};

Resolver OverwriteResolver = [](const std::shared_ptr<Manifest>& read_manifest,
                                int64_t /*read_version*/,
                                const std::shared_ptr<Manifest>& /*latest_manifest*/,
                                int64_t /*latest_version*/,
                                const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  return applyUpdates(read_manifest, updates);
};

Resolver FailResolver = [](const std::shared_ptr<Manifest>& /*read_manifest*/,
                           int64_t read_version,
                           const std::shared_ptr<Manifest>& latest_manifest,
                           int64_t latest_version,
                           const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  // Check if read_version equals latest_version (no concurrent changes)
  if (read_version == latest_version) {
    return applyUpdates(latest_manifest, updates);
  }

  return arrow::Status::Invalid(
      fmt::format("FailResolver: concurrent transaction detected, [read_version={}][latest_version={}]", read_version,
                  latest_version));
};

// ==================== Transaction Implementation ====================

arrow::Result<std::unique_ptr<Transaction>> Transaction::Open(const milvus_storage::ArrowFileSystemPtr& fs,
                                                              const std::string& base_path,
                                                              int64_t read_version,
                                                              const Resolver& resolver,
                                                              uint32_t retry_limit) {
  auto txn = std::unique_ptr<Transaction>(new Transaction(fs, base_path, read_version, resolver, retry_limit));

  // Determine actual read version
  int64_t actual_read_version = read_version;
  if (read_version == LATEST) {
    ARROW_ASSIGN_OR_RAISE(actual_read_version, txn->get_latest_version());
    // If no manifests exist, actual_read_version will be 0
  }

  // Record read_version_
  txn->read_version_ = actual_read_version;

  // Eagerly load manifest if version exists (version > 0)
  if (actual_read_version > 0) {
    ARROW_ASSIGN_OR_RAISE(txn->read_manifest_, txn->read_manifest(actual_read_version));
  } else {
    // Version 0: no manifest exists yet, create empty manifest
    txn->read_manifest_ = std::make_shared<Manifest>();
  }

  // Reset updates tracking
  txn->updates_ = Updates{};

  return txn;
}

Transaction::Transaction(const milvus_storage::ArrowFileSystemPtr& fs,
                         const std::string& base_path,
                         int64_t read_version,
                         const Resolver& resolver,
                         uint32_t retry_limit)
    : read_version_(read_version),
      base_path_(base_path),
      read_manifest_(nullptr),
      updates_(),
      resolver_(resolver),
      fs_(fs),
      retry_limit_(retry_limit) {}

arrow::Result<int64_t> Transaction::Commit() {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_MANIFEST_COMMIT_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_MANIFEST_COMMIT_FAIL)));

  assert(resolver_ != nullptr);

  // Fail if there are no updates
  if (!updates_.hasChanges()) {
    return arrow::Status::Invalid("Cannot commit: no updates recorded");
  }

  // Lambda to reload latest manifest and return version and manifest
  auto reload_latest_manifest = [this]() -> arrow::Result<std::pair<int64_t, std::shared_ptr<Manifest>>> {
    ARROW_ASSIGN_OR_RAISE(auto latest_version, get_latest_version());

    std::shared_ptr<Manifest> latest_manifest;
    if (latest_version == read_version_) {
      // Latest version is the same as read version, use read_manifest as latest_manifest
      latest_manifest = read_manifest_;
    } else {
      // Latest version differs, load the latest manifest
      LOG_STORAGE_DEBUG_ << fmt::format("Manifest version drift detected: [read_version={}][latest_version={}]",
                                        read_version_, latest_version);
      ARROW_ASSIGN_OR_RAISE(latest_manifest, read_manifest(latest_version));
    }

    return std::make_pair(latest_version, latest_manifest);
  };

  // Retry loop for handling commit conflicts
  uint32_t retry_count = 0;
  while (retry_count <= retry_limit_) {
    // Reload latest manifest
    ARROW_ASSIGN_OR_RAISE(auto latest_result, reload_latest_manifest());
    int64_t latest_version = latest_result.first;
    std::shared_ptr<Manifest> latest_manifest = latest_result.second;

    // Always call resolver to get merged manifest
    auto resolved_manifest_result = resolver_(read_manifest_, read_version_, latest_manifest, latest_version, updates_);
    if (!resolved_manifest_result.ok()) {
      return arrow::Status::Invalid(fmt::format("Resolution failed: {}", resolved_manifest_result.status().ToString()));
    }
    auto resolved_manifest = resolved_manifest_result.ValueOrDie();

    // Determine committed version based on latest_version
    int64_t committed_version = latest_version + 1;

    // Try to commit the resolved manifest
    auto status = write_manifest(resolved_manifest, latest_version, committed_version);

    // If commit succeeded, return the committed version
    if (status.ok()) {
      LOG_STORAGE_DEBUG_ << fmt::format(
          "Manifest committed successfully: [committed_version={}][read_version={}][retries={}]", committed_version,
          read_version_, retry_count);
      return committed_version;
    }

    // If commit failed due to conflict (file already exists), retry if within limit
    if (status.code() == arrow::StatusCode::AlreadyExists) {
      LOG_STORAGE_DEBUG_ << fmt::format(
          "Commit conflict: manifest version {} already exists, "
          "[read_version={}][latest_version={}][retry={}/{}]",
          committed_version, read_version_, latest_version, retry_count, retry_limit_);
      retry_count++;
      if (retry_count > retry_limit_) {
        return arrow::Status::Invalid(
            fmt::format("Commit failed: exceeded retry limit of {} attempts due to concurrent transactions, "
                        "[read_version={}][latest_version={}]",
                        retry_limit_, read_version_, latest_version));
      }
      // Continue loop to retry with updated manifest
      continue;
    }

    // Other errors (not conflict-related) should be returned immediately
    return arrow::Status::IOError(
        fmt::format("Commit failed: write manifest error, "
                    "[read_version={}][latest_version={}][committed_version={}]: {}",
                    read_version_, latest_version, committed_version, status.ToString()));
  }

  // This should never be reached, but included for safety
  return arrow::Status::Invalid("Commit failed: unexpected retry loop exit");
}

arrow::Result<std::shared_ptr<Manifest>> Transaction::GetManifest() {
  // Manifest should have been loaded eagerly during Open
  if (!read_manifest_) {
    return arrow::Status::Invalid("Manifest should have been loaded during Open");
  }

  return read_manifest_;
}

int64_t Transaction::GetReadVersion() const { return read_version_; }

arrow::Result<std::shared_ptr<Manifest>> Transaction::read_manifest(int64_t version) {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_MANIFEST_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_MANIFEST_READ_FAIL)));

  // If version is 0 or less, return empty manifest (no manifests exist yet)
  if (version <= 0) {
    return std::make_shared<Manifest>();
  }

  auto path = get_manifest_filepath(base_path_, version);
  return Manifest::ReadFrom(fs_, path);
}

Transaction& Transaction::DropColumn(const std::string& column_name) {
  updates_.DropColumn(column_name);
  return *this;
}

Transaction& Transaction::AddColumnGroup(const std::shared_ptr<ColumnGroup>& cg) {
  updates_.AddColumnGroup(cg);
  return *this;
}

Transaction& Transaction::AppendFiles(const ColumnGroups& cgs) {
  updates_.AppendFiles(cgs);
  return *this;
}

Transaction& Transaction::AddDeltaLog(const DeltaLog& delta_log) {
  updates_.AddDeltaLog(delta_log);
  return *this;
}

Transaction& Transaction::UpdateStat(const std::string& key, const Statistics& stat) {
  updates_.UpdateStat(key, stat);
  return *this;
}

Transaction& Transaction::AddIndex(const Index& index) {
  updates_.AddIndex(index);
  return *this;
}

Transaction& Transaction::DropIndex(const std::string& column_name, const std::string& index_type) {
  updates_.DropIndex(column_name, index_type);
  return *this;
}

Transaction& Transaction::AddLobFile(const LobFileInfo& lob_file) {
  updates_.AddLobFile(lob_file);
  return *this;
}

arrow::Result<int64_t> Transaction::get_latest_version() {
  std::string metadata_dir = get_manifest_path(base_path_);

  // Check if metadata directory exists
  ARROW_ASSIGN_OR_RAISE(auto dir_info, fs_->GetFileInfo(metadata_dir));
  if (dir_info.type() == arrow::fs::FileType::NotFound) {
    return 0;  // No manifests yet, return 0 (next commit will be version 1)
  }

  // List files in metadata directory
  arrow::fs::FileSelector selector;
  selector.base_dir = metadata_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;
  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs_->GetFileInfo(selector));

  int64_t latest_version = 0;
  for (const auto& file_info : file_infos) {
    const std::string file_name = file_info.base_name();
    // filter manifest files with prefix and suffix
    if (file_name.find(kManifestFileNamePrefix) != 0) {
      continue;  // must start with prefix
    }
    if (file_name.size() <= kManifestFileNamePrefix.length() + kManifestFileNameSuffix.length()) {
      continue;  // too short to contain version number
    }
    // extract version number (between prefix and suffix)
    std::string version_str =
        file_name.substr(kManifestFileNamePrefix.length(),
                         file_name.length() - kManifestFileNamePrefix.length() - kManifestFileNameSuffix.length());
    int64_t version = 0;
    auto [ptr, ec] = std::from_chars(version_str.data(), version_str.data() + version_str.size(), version);
    if (ec != std::errc() || ptr != version_str.data() + version_str.size()) {
      continue;  // not a valid version number
    }
    latest_version = std::max<int64_t>(latest_version, version);
  }

  return latest_version;
}

arrow::Status Transaction::write_manifest(const std::shared_ptr<Manifest>& manifest,
                                          int64_t old_version,
                                          int64_t new_version) {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_MANIFEST_WRITE_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_MANIFEST_WRITE_FAIL)));

  return Manifest::WriteTo(fs_, get_manifest_filepath(base_path_, new_version), *manifest);
}

}  // namespace milvus_storage::api::transaction
