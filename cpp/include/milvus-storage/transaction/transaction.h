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

#pragma once

#include <cstdint>
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <map>

#include <arrow/status.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"

namespace milvus_storage::api::transaction {

/**
 * @brief Version constant for fetching the latest manifest
 *
 * When used as a version parameter, LATEST means to fetch the greatest version number.
 * If no manifests exist, the system will generate version 1.
 */
static constexpr int64_t LATEST = -1;

/**
 * @brief Transaction updates tracking class
 *
 * Records all changes made during a transaction that can be used by the resolver
 * Implementation is in transaction.cpp
 */
class Updates {
  public:
  Updates();
  ~Updates();

  [[nodiscard]] bool hasChanges() const;
  void AddColumnGroup(const std::shared_ptr<ColumnGroup>& cg);
  void AppendFiles(const std::vector<std::shared_ptr<ColumnGroup>>& cgs);
  void AddDeltaLog(const DeltaLog& delta_log);
  void UpdateStat(const std::string& key, const std::vector<std::string>& files);
  [[nodiscard]] const std::vector<std::shared_ptr<ColumnGroup>>& GetAddedColumnGroups() const;
  [[nodiscard]] const std::vector<std::vector<std::shared_ptr<ColumnGroup>>>& GetAppendedFiles() const;
  [[nodiscard]] const std::vector<DeltaLog>& GetAddedDeltaLogs() const;
  [[nodiscard]] const std::map<std::string, std::vector<std::string>>& GetAddedStats() const;

  private:
  // Column group changes
  ColumnGroups added_column_groups_;          // New column groups to add
  std::vector<ColumnGroups> appended_files_;  // Column groups with files to append

  // Delta log changes
  std::vector<DeltaLog> added_delta_logs_;  // New delta logs to add

  // Stats changes
  std::map<std::string, std::vector<std::string>> added_stats_;  // New stats to add (key -> files)
};

/**
 * @brief Resolver function type
 *
 * Resolves conflicts between read manifest and seen manifest using recorded changes.
 *
 * @param read_manifest The manifest read when transaction began
 * @param read_version The version of the read manifest
 * @param seen_manifest The latest manifest seen (may differ from read_manifest if concurrent commits occurred)
 * @param seen_version The version of the seen manifest
 * @param updates The recorded changes in this transaction
 * @return Resolved manifest or error status
 */
using Resolver = std::function<arrow::Result<std::shared_ptr<Manifest>>(const std::shared_ptr<Manifest>& read_manifest,
                                                                        int64_t read_version,
                                                                        const std::shared_ptr<Manifest>& seen_manifest,
                                                                        int64_t seen_version,
                                                                        const Updates& updates)>;

// ==================== Helper Functions ====================

/**
 * @brief Apply updates to a manifest
 *
 * Creates a copy of the manifest and applies all updates (appended files, added column groups,
 * delta logs, stats) to it.
 *
 * @param manifest Base manifest to apply updates to
 * @param updates Updates to apply
 * @return New manifest with updates applied, or error status
 */
arrow::Result<std::shared_ptr<Manifest>> applyUpdates(const std::shared_ptr<Manifest>& manifest,
                                                      const Updates& updates);

// ==================== Helper Resolver Functions ====================

/**
 * @brief Unified resolver that merges changes with seen_manifest (latest manifest)
 *
 * This resolver merges all changes (appended files, added column groups, delta logs, stats)
 * into the seen_manifest, effectively merging concurrent changes.
 */
extern Resolver MergeResolver;

/**
 * @brief Resolver that applies updates to read_manifest, overwriting any concurrent changes
 *
 * This resolver applies all changes (appended files, added column groups, delta logs, stats)
 * into the read_manifest, ignoring any concurrent changes in seen_manifest.
 */
extern Resolver OverwriteResolver;

/**
 * @brief Resolver that fails if seen version differs from read version
 *
 * This resolver checks if there were concurrent changes by comparing versions.
 * If read_version equals seen_version, it applies updates to the manifest.
 */
extern Resolver FailResolver;

class Transaction {
  public:
  // Static factory method to open a transaction
  // @param fs Filesystem to use
  // @param base_path Base path for the storage
  // @param version Version to read from (default: LATEST = fetch greatest version)
  // @param resolver Resolver function for conflict resolution (default: FailResolver)
  // @param retry_limit Maximum number of retry attempts on commit conflicts (default: 1)
  static arrow::Result<std::unique_ptr<Transaction>> Open(const milvus_storage::ArrowFileSystemPtr& fs,
                                                          const std::string& base_path,
                                                          int64_t version = LATEST,
                                                          const Resolver& resolver = FailResolver,
                                                          uint32_t retry_limit = 1);

  ~Transaction() = default;

  // Commit using Updates (uses the resolver set via Open)
  // Returns the committed version number on success
  arrow::Result<int64_t> Commit();

  // Get manifest on read version
  // Returns empty manifest if read_version_ is 0 (no manifest exists)
  arrow::Result<std::shared_ptr<Manifest>> GetManifest();

  // Get the read version of this transaction
  [[nodiscard]] int64_t GetReadVersion() const;

  // ==================== Fluent-style Builder Methods ====================

  /**
   * @brief Add a new column group to the transaction updates
   * @param cg Column group to add
   * @return Reference to this transaction for method chaining
   */
  Transaction& AddColumnGroup(const std::shared_ptr<ColumnGroup>& cg);

  /**
   * @brief Append files to existing column groups
   * @param cgs Column groups containing files to append
   * @return Reference to this transaction for method chaining
   */
  Transaction& AppendFiles(const std::vector<std::shared_ptr<ColumnGroup>>& cgs);

  /**
   * @brief Add a delta log to the transaction updates
   * @param delta_log Delta log to add
   * @return Reference to this transaction for method chaining
   */
  Transaction& AddDeltaLog(const DeltaLog& delta_log);

  /**
   * @brief Add a stat entry to the transaction updates
   * @param key Stat key (e.g., "pk.delete", "bloomfilter", "bm25")
   * @param files List of file paths for this stat
   * @return Reference to this transaction for method chaining
   */
  Transaction& UpdateStat(const std::string& key, const std::vector<std::string>& files);

  private:
  // Private constructor - use Open() factory method instead
  Transaction(const milvus_storage::ArrowFileSystemPtr& fs,
              const std::string& base_path,
              int64_t read_version,
              const Resolver& resolver,
              uint32_t retry_limit);

  // Get latest version from filesystem
  arrow::Result<int64_t> get_latest_version();

  // Read manifest from filesystem at given version
  arrow::Result<std::shared_ptr<Manifest>> read_manifest(int64_t version);

  arrow::Status write_manifest(const std::shared_ptr<Manifest>& manifest, int64_t old_version, int64_t new_version);

  int64_t read_version_;
  std::shared_ptr<Manifest> read_manifest_;
  std::string base_path_;
  Updates updates_;    ///< Transaction updates tracked for resolver
  Resolver resolver_;  ///< Resolver function for conflict resolution
  milvus_storage::ArrowFileSystemPtr fs_;
  uint32_t retry_limit_;  ///< Maximum number of retry attempts on commit conflicts
};

}  // namespace milvus_storage::api::transaction
