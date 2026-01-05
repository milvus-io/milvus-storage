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
#include <string_view>
#include <sstream>
#include <mutex>
#include <set>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include "milvus-storage/filesystem/filesystem_extend.h"
#include <avro/Encoder.hh>
#include <avro/Decoder.hh>
#include <avro/Stream.hh>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage::api::transaction {

// ==================== Updates Class Implementation ====================

Updates::Updates() = default;
Updates::~Updates() = default;

bool Updates::hasChanges() const {
  return !added_column_groups_.empty() || !appended_files_.empty() || !added_delta_logs_.empty() ||
         !added_stats_.empty();
}

void Updates::AddColumnGroup(const std::shared_ptr<ColumnGroup>& cg) { added_column_groups_.push_back(cg); }

void Updates::AppendFiles(const ColumnGroups& cgs) { appended_files_.push_back(cgs); }

void Updates::AddDeltaLog(const DeltaLog& delta_log) { added_delta_logs_.push_back(delta_log); }

void Updates::UpdateStat(const std::string& key, const std::vector<std::string>& files) { added_stats_[key] = files; }

const ColumnGroups& Updates::GetAddedColumnGroups() const { return added_column_groups_; }

const std::vector<ColumnGroups>& Updates::GetAppendedFiles() const { return appended_files_; }

const std::vector<DeltaLog>& Updates::GetAddedDeltaLogs() const { return added_delta_logs_; }

const std::map<std::string, std::vector<std::string>>& Updates::GetAddedStats() const { return added_stats_; }

// ==================== Helper Functions ====================

arrow::Result<std::shared_ptr<Manifest>> applyUpdates(const std::shared_ptr<Manifest>& manifest,
                                                      const Updates& updates) {
  // Get base manifest attributes
  const auto& base_column_groups = manifest->columnGroups();
  const auto& base_delta_logs = manifest->deltaLogs();
  const auto& base_stats = manifest->stats();

  // Validate: Check if adding column groups has existing column names
  for (const auto& new_cg : updates.GetAddedColumnGroups()) {
    if (!new_cg) {
      return arrow::Status::Invalid("Cannot add null column group");
    }
    for (const auto& column_name : new_cg->columns) {
      auto existing_cg = manifest->getColumnGroup(column_name);
      if (existing_cg != nullptr) {
        return arrow::Status::Invalid("Column '" + column_name + "' already exists in existing column groups");
      }
    }
  }

  // Validate: Check if appending files are aligned with existing column groups
  for (const auto& new_cgs : updates.GetAppendedFiles()) {
    // Check size alignment
    if (!base_column_groups.empty() && base_column_groups.size() != new_cgs.size()) {
      return arrow::Status::Invalid("Column group size mismatch: existing has ",
                                    std::to_string(base_column_groups.size()), " groups, but appended has ",
                                    std::to_string(new_cgs.size()));
    }

    // Check each column group alignment
    for (size_t i = 0; i < base_column_groups.size() && i < new_cgs.size(); ++i) {
      const auto& base_cg = base_column_groups[i];
      const auto& new_cg = new_cgs[i];

      if (!base_cg || !new_cg) {
        return arrow::Status::Invalid("Null column group at index ", std::to_string(i));
      }

      // Check column count
      if (base_cg->columns.size() != new_cg->columns.size()) {
        return arrow::Status::Invalid("Column count mismatch at index ", std::to_string(i), ": existing has ",
                                      std::to_string(base_cg->columns.size()), " columns, but appended has ",
                                      std::to_string(new_cg->columns.size()));
      }

      // Check format
      if (base_cg->format != new_cg->format) {
        return arrow::Status::Invalid("Format mismatch at index ", std::to_string(i), ": existing format is '",
                                      base_cg->format, "', but appended format is '", new_cg->format, "'");
      }

      // Check columns match
      std::set<std::string> base_cols(base_cg->columns.begin(), base_cg->columns.end());
      std::set<std::string> new_cols(new_cg->columns.begin(), new_cg->columns.end());
      if (base_cols != new_cols) {
        return arrow::Status::Invalid("Column mismatch at index ", std::to_string(i), ": columns do not match");
      }
    }
  }

  // Prepare delta logs (copy from base + add new ones)
  std::vector<DeltaLog> resolved_delta_logs = base_delta_logs;
  for (const auto& delta_log : updates.GetAddedDeltaLogs()) {
    resolved_delta_logs.push_back(delta_log);
  }

  // Prepare stats (copy from base + merge new ones, new values override)
  std::map<std::string, std::vector<std::string>> resolved_stats = base_stats;
  for (const auto& [key, files] : updates.GetAddedStats()) {
    resolved_stats[key] = files;  // Override existing or add new
  }

  // Create a copy of column groups to apply updates
  ColumnGroups resolved_column_groups = base_column_groups;

  // Apply updates: append files (merge files into existing column groups)
  for (const auto& new_cgs : updates.GetAppendedFiles()) {
    if (resolved_column_groups.empty()) {
      // If no existing column groups, directly assign
      resolved_column_groups = new_cgs;
    } else {
      // Merge files into existing column groups
      for (size_t i = 0; i < resolved_column_groups.size() && i < new_cgs.size(); ++i) {
        auto& base_cg = resolved_column_groups[i];
        const auto& new_cg = new_cgs[i];
        if (base_cg && new_cg) {
          // Append files from new_cg to base_cg
          for (const auto& file : new_cg->files) {
            base_cg->files.push_back(file);
          }
        }
      }
    }
  }

  // Apply updates: add column groups
  for (const auto& cg : updates.GetAddedColumnGroups()) {
    resolved_column_groups.push_back(cg);
  }

  // Create resolved manifest using the copy constructor with all attributes
  auto resolved = std::make_shared<Manifest>(std::move(resolved_column_groups), resolved_delta_logs, resolved_stats);

  return resolved;
}

// ==================== Helper Resolver Functions ====================

Resolver MergeResolver = [](const std::shared_ptr<Manifest>& /*read_manifest*/,
                            int64_t /*read_version*/,
                            const std::shared_ptr<Manifest>& seen_manifest,
                            int64_t /*seen_version*/,
                            const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  return applyUpdates(seen_manifest, updates);
};

Resolver OverwriteResolver = [](const std::shared_ptr<Manifest>& read_manifest,
                                int64_t /*read_version*/,
                                const std::shared_ptr<Manifest>& /*seen_manifest*/,
                                int64_t /*seen_version*/,
                                const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  return applyUpdates(read_manifest, updates);
};

Resolver FailResolver = [](const std::shared_ptr<Manifest>& /*read_manifest*/,
                           int64_t read_version,
                           const std::shared_ptr<Manifest>& seen_manifest,
                           int64_t seen_version,
                           const Updates& updates) -> arrow::Result<std::shared_ptr<Manifest>> {
  // Check if read_version equals seen_version (no concurrent changes)
  if (read_version == seen_version) {
    return applyUpdates(seen_manifest, updates);
  }

  return arrow::Status::Invalid("Commit failed: concurrent transaction detected");
};

// ==================== Transaction Implementation ====================

// Helper function that tries conditional write first, falls back to unsafe write if not supported
arrow::Status write_manifest_file(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                  const std::string& path,
                                  std::string_view data) {
  static std::mutex write_mutex;
  std::scoped_lock lock(write_mutex);

  // Try conditional write first if filesystem supports it
  auto res = open_condition_write_output_stream(fs, path);
  if (res.ok()) {
    auto output_stream = res.ValueOrDie();
    ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
    auto result = output_stream->Close();
    if (!result.ok()) {
      // already exist then return AlreadyExists
      if (result.code() == arrow::StatusCode::IOError) {
        return arrow::Status::AlreadyExists("File already exists: ", path);
      }
      // others return the error
      return result;
    }

    return arrow::Status::OK();
  }

  // Fall back to unsafe write
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
  if (file_info.type() != arrow::fs::FileType::NotFound) {
    return arrow::Status::AlreadyExists("File already exists: ", path);
  }

  auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
  if (!parent.empty()) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
  ARROW_RETURN_NOT_OK(output_stream->Close());

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> direct_read(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                          const std::string& path) {
  std::shared_ptr<arrow::Buffer> buffer;
  // Open input file and get size
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());

  // Read into an Arrow Buffer (makes memory management automatic)
  ARROW_ASSIGN_OR_RAISE(buffer, input_file->Read(file_size));

  // Ensure we read the expected size
  if (buffer->size() != file_size) {
    return arrow::Status::IOError("Failed to read the complete file, expected size =", file_size,
                                  ", actual size =", static_cast<int64_t>(buffer->size()));
  }

  ARROW_RETURN_NOT_OK(input_file->Close());
  return buffer;
}

arrow::Result<std::unique_ptr<Transaction>> Transaction::Open(const milvus_storage::ArrowFileSystemPtr& fs,
                                                              const std::string& base_path,
                                                              int64_t read_version,
                                                              const Resolver& resolver) {
  auto txn = std::unique_ptr<Transaction>(new Transaction(fs, base_path, read_version, resolver));

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
                         const Resolver& resolver)
    : read_version_(read_version),
      base_path_(base_path),
      read_manifest_(nullptr),
      updates_(),
      resolver_(resolver),
      fs_(fs) {}

arrow::Result<int64_t> Transaction::Commit() {
  assert(resolver_ != nullptr);

  // Fail if there are no updates
  if (!updates_.hasChanges()) {
    return arrow::Status::Invalid("Cannot commit: no updates recorded");
  }

  // Lambda to reload latest manifest and return version and manifest
  auto ReloadLatestManifest = [this]() -> arrow::Result<std::pair<int64_t, std::shared_ptr<Manifest>>> {
    ARROW_ASSIGN_OR_RAISE(auto latest_version, get_latest_version());

    std::shared_ptr<Manifest> seen_manifest;
    if (latest_version == read_version_) {
      // Latest version is the same as read version, use read_manifest as seen_manifest
      seen_manifest = read_manifest_;
    } else {
      // Latest version differs, load the latest manifest
      ARROW_ASSIGN_OR_RAISE(seen_manifest, read_manifest(latest_version));
    }

    return std::make_pair(latest_version, seen_manifest);
  };

  // Reload latest manifest
  ARROW_ASSIGN_OR_RAISE(auto latest_result, ReloadLatestManifest());
  int64_t latest_version = latest_result.first;
  std::shared_ptr<Manifest> seen_manifest = latest_result.second;

  // Always call resolver to get merged manifest
  auto resolved_manifest_result = resolver_(read_manifest_, read_version_, seen_manifest, latest_version, updates_);
  if (!resolved_manifest_result.ok()) {
    return arrow::Status::Invalid("Resolution failed: ", resolved_manifest_result.status().ToString());
  }
  auto resolved_manifest = resolved_manifest_result.ValueOrDie();

  // Determine committed version based on latest_version
  int64_t committed_version = latest_version + 1;

  // Try to commit the resolved manifest
  auto status = write_manifest(resolved_manifest, latest_version, committed_version);

  // If commit failed due to conflict (file already exists), retry with latest manifest
  // TODO: set a retry limit and return error if exceeded.
  if (!status.ok() && status.code() == arrow::StatusCode::AlreadyExists) {
    // Reload latest manifest (may have changed)
    ARROW_ASSIGN_OR_RAISE(auto retry_result, ReloadLatestManifest());
    int64_t new_latest_version = retry_result.first;
    std::shared_ptr<Manifest> new_seen_manifest = retry_result.second;

    // Re-resolve with new_seen_manifest (latest) and read_manifest (original)
    auto re_resolved_result = resolver_(read_manifest_, read_version_, new_seen_manifest, new_latest_version, updates_);
    if (!re_resolved_result.ok()) {
      return arrow::Status::Invalid("Resolver failed during retry: ", re_resolved_result.status().ToString());
    }
    resolved_manifest = re_resolved_result.ValueOrDie();

    // Update committed_version based on new_latest_version
    committed_version = new_latest_version + 1;
    ARROW_RETURN_NOT_OK(write_manifest(resolved_manifest, new_latest_version, committed_version));
  } else if (!status.ok()) {
    // Other errors (not conflict-related) should be returned
    return status;
  }

  return committed_version;
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
  auto manifest = std::make_shared<Manifest>();
  // If version is 0 or less, return empty manifest (no manifests exist yet)
  if (version <= 0) {
    return manifest;
  }

  ARROW_ASSIGN_OR_RAISE(auto manifest_buffer, direct_read(fs_, base_path_ + kSep + kManifestFilePrefix +
                                                                   std::to_string(version) + kManifestFileNameSuffix));
  std::string manifest_data(reinterpret_cast<const char*>(manifest_buffer->data()), manifest_buffer->size());
  std::istringstream in(manifest_data);
  ARROW_RETURN_NOT_OK(manifest->deserialize(in));

  // Resolve paths for column groups, delta logs, and stats
  ARROW_RETURN_NOT_OK(manifest->resolve_paths(base_path_));

  return manifest;
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

Transaction& Transaction::UpdateStat(const std::string& key, const std::vector<std::string>& files) {
  updates_.UpdateStat(key, files);
  return *this;
}

arrow::Result<int64_t> Transaction::get_latest_version() {
  // Check if metadata directory exists
  std::string metadata_dir = base_path_ + kSep + kMetadataDir;
  ARROW_ASSIGN_OR_RAISE(auto dir_info, fs_->GetFileInfo(metadata_dir));
  if (dir_info.type() == arrow::fs::FileType::NotFound) {
    return 0;  // No manifests yet, return 0 (next commit will be version 1)
  }

  arrow::fs::FileSelector selector;
  selector.base_dir = metadata_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  // list the objects in metadata directory and get the latest manifest file
  std::vector<std::string> file_names;
  {
    ARROW_ASSIGN_OR_RAISE(auto file_infos_result, fs_->GetFileInfo(selector));

    file_names.reserve(file_infos_result.size());
    for (const auto& file_info : file_infos_result) {
      file_names.push_back(file_info.base_name());
    }
  }

  int64_t latest_version = 0;
  for (const auto& file_name : file_names) {
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
  // Serialize new manifest to Avro
  std::ostringstream oss;
  ARROW_RETURN_NOT_OK(manifest->serialize(oss));
  std::string manifest_bytes = oss.str();

  // Write manifest file (tries conditional write, falls back to unsafe write if not supported)
  arrow::Status result = write_manifest_file(
      fs_, base_path_ + kSep + kManifestFilePrefix + std::to_string(new_version) + kManifestFileNameSuffix,
      manifest_bytes);

  return result;
}

}  // namespace milvus_storage::api::transaction
