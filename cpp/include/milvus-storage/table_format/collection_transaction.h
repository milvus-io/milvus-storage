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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/table_format/action.h"
#include "milvus-storage/table_format/metadata.h"
#include "milvus-storage/table_format/types.h"

namespace milvus_storage::api::table_format {

static constexpr int64_t LATEST_VERSION = -1;

class CollectionTransaction {
  public:
  static arrow::Result<std::unique_ptr<CollectionTransaction>> Open(
      const milvus_storage::ArrowFileSystemPtr& fs,
      const std::string& base_path,
      int64_t version = LATEST_VERSION,
      uint32_t retry_limit = 3);

  // Mutable access to metadata (for initial collection setup)
  Metadata& GetSnapshot();

  // Read methods
  const Metadata& GetMetadata() const;
  int64_t GetReadVersion() const;
  arrow::Result<const SnapshotEntry*> GetCurrentSnapshot() const;
  arrow::Result<const SnapshotEntry*> GetSnapshot(int64_t snapshot_id) const;
  arrow::Result<const SnapshotEntry*> GetSnapshotAtTime(int64_t timestamp_ms) const;
  arrow::Result<const SchemaInfo*> GetSchema(int32_t schema_id) const;
  arrow::Result<const IndexSpec*> GetIndexSpec(int32_t spec_id) const;
  arrow::Result<std::vector<ManifestListEntry>> ListSegments(const SnapshotEntry& snapshot) const;
  arrow::Result<std::vector<SegmentInfo>> ListSegments(const SnapshotEntry& snapshot, int64_t partition_id) const;

  arrow::Result<int64_t> Commit(std::shared_ptr<Action> action);

  private:
  CollectionTransaction(const milvus_storage::ArrowFileSystemPtr& fs,
                        const std::string& base_path,
                        int64_t read_version,
                        Metadata metadata,
                        uint32_t retry_limit);

  // Load ManifestListEntry from all manifest list files in the given snapshot
  arrow::Result<std::vector<ManifestListEntry>> LoadManifestEntries(
      const std::vector<ManifestListInfo>& manifest_lists) const;

  milvus_storage::ArrowFileSystemPtr fs_;
  std::string base_path_;
  int64_t read_version_;
  Metadata metadata_;
  uint32_t retry_limit_;
};

}  // namespace milvus_storage::api::table_format
