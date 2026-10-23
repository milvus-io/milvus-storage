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

#include <arrow/status.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/table_format/types.h"

namespace milvus_storage::api::table_format {

class Metadata;

class Action {
  public:
  virtual ~Action() = default;

  // Apply schema/index changes and segment operations to metadata.
  // Segment operations read/write manifest list files on-demand.
  virtual arrow::Status Apply(Metadata& md) = 0;
};

class ActionBuilder {
  public:
  ~ActionBuilder();

  static ActionBuilder Create(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& base_path);

  // Collection setup
  ActionBuilder& SetCollectionInfo(CollectionInfo info);
  ActionBuilder& SetSchema(SchemaInfo schema);

  // Schema evolution
  ActionBuilder& AddColumn(FieldSchema field);
  ActionBuilder& DropColumn(const std::string& field_name);

  // Partition management
  ActionBuilder& AddPartition(const std::string& partition_name);
  ActionBuilder& DropPartition(const std::string& partition_name);

  // Segment operations (partition_id=0 means auto-generate)
  ActionBuilder& AddSegment(const std::string& partition_name, SegmentInfo segment);
  ActionBuilder& AddSegment(int64_t partition_id, const std::string& partition_name, SegmentInfo segment);
  ActionBuilder& RemoveSegments(std::vector<int64_t> segment_ids);

  // Index management
  ActionBuilder& AddIndex(IndexInfo index);
  ActionBuilder& DropIndex(const std::string& index_name);

  // Snapshot rollback: create a new snapshot with the same state as a historical one
  ActionBuilder& SetCurrentSnapshot(int64_t snapshot_id);
  // Snapshot rollback by timestamp: find the latest snapshot at or before the given
  // timestamp and create a new snapshot with the same state
  ActionBuilder& SetCurrentSnapshotByTimestamp(int64_t timestamp_ms);

  std::shared_ptr<Action> Build();

  private:
  ActionBuilder();
  ActionBuilder(ActionBuilder&&) noexcept;
  ActionBuilder& operator=(ActionBuilder&&) noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace milvus_storage::api::table_format
