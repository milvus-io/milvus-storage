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

#include "milvus-storage/table_format/action.h"
#include "milvus-storage/table_format/manifest_list.h"
#include "milvus-storage/table_format/metadata.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

#include <fmt/format.h>

namespace milvus_storage::api::table_format {

struct SegmentAdd {
  int64_t partition_id;
  std::string partition_name;
  SegmentInfo segment;
};

struct ActionParams {
  milvus_storage::ArrowFileSystemPtr fs;
  std::string base_path;
  std::optional<CollectionInfo> collection_info;
  std::optional<SchemaInfo> schema;
  std::vector<SegmentAdd> segment_adds;
  std::vector<int64_t> segment_removes;
  std::vector<FieldSchema> columns_to_add;
  std::vector<std::string> columns_to_drop;
  std::vector<std::string> partitions_to_add;
  std::vector<std::string> partitions_to_drop;
  std::vector<IndexInfo> indexes_to_add;
  std::vector<std::string> indexes_to_drop;
  std::optional<int64_t> rollback_snapshot_id;
  std::optional<int64_t> rollback_timestamp_ms;
};

// ActionBuilder::Impl inherits ActionParams so builder methods can access
// fields directly, and Build() can slice the base into ActionImpl.
struct ActionBuilder::Impl : ActionParams {};

namespace {

// ---- ActionImpl ----

class ActionImpl : public Action {
  public:
  explicit ActionImpl(ActionParams params) : params_(std::move(params)) {}

  arrow::Status Apply(Metadata& md) override {
    // Mutual exclusivity: rollback by ID and by timestamp cannot both be set
    if (params_.rollback_snapshot_id.has_value() && params_.rollback_timestamp_ms.has_value()) {
      return arrow::Status::Invalid("Cannot set both rollback_snapshot_id and rollback_timestamp_ms");
    }

    if (params_.rollback_timestamp_ms.has_value()) {
      return ApplyRollbackByTimestamp(md);
    }
    if (params_.rollback_snapshot_id.has_value()) {
      return ApplyRollback(md);
    }

    if (params_.collection_info.has_value()) {
      md.mutable_collection() = std::move(*params_.collection_info);
    }
    if (params_.schema.has_value()) {
      md.mutable_schemas().push_back(std::move(*params_.schema));
      md.set_current_schema_id(md.schemas().back().schema_id);
    }

    ARROW_RETURN_NOT_OK(ApplySchemaChanges(md));
    ARROW_RETURN_NOT_OK(ApplyIndexChanges(md));

    // Determine manifest_lists for the new snapshot
    bool has_manifest_changes = !params_.segment_adds.empty() || !params_.segment_removes.empty() ||
                                !params_.partitions_to_add.empty() || !params_.partitions_to_drop.empty();
    std::vector<ManifestListInfo> manifest_lists;
    if (has_manifest_changes) {
      ARROW_ASSIGN_OR_RAISE(manifest_lists, ApplyManifestChanges(md));
    } else {
      // Copy from current snapshot
      for (const auto& snap : md.snapshots()) {
        if (snap.snapshot_id == md.current_snapshot_id()) {
          manifest_lists = snap.manifest_lists;
          break;
        }
      }
    }

    // Create default partition if none exists
    if (manifest_lists.empty()) {
      ManifestListEntry default_entry;
      default_entry.partition_id = 1;
      default_entry.partition_name = "_default";

      ManifestList ml({std::move(default_entry)});
      ARROW_ASSIGN_OR_RAISE(auto path, WriteManifestListToFile(params_.fs, params_.base_path, ml));
      manifest_lists.push_back({
          .manifest_list = path,
          .partition_ids = {1},
          .partition_names = {"_default"},
      });
    }

    // Create new snapshot entry using monotonic ID counter
    SnapshotEntry new_snap;
    new_snap.snapshot_id = md.allocate_snapshot_id();
    if (md.current_snapshot_id() > 0) {
      new_snap.parent_snapshot_id = md.current_snapshot_id();
    }
    auto now = std::chrono::system_clock::now();
    new_snap.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    new_snap.schema_id = md.current_schema_id();
    new_snap.index_spec_id = md.current_index_spec_id();
    new_snap.manifest_lists = std::move(manifest_lists);

    md.mutable_snapshots().push_back(std::move(new_snap));
    md.set_current_snapshot_id(md.snapshots().back().snapshot_id);

    ARROW_RETURN_NOT_OK(Validate(md));
    return arrow::Status::OK();
  }

  private:
  ActionParams params_;

  arrow::Status ApplyRollback(Metadata& md) {
    const SnapshotEntry* historical = nullptr;
    for (const auto& snap : md.snapshots()) {
      if (snap.snapshot_id == *params_.rollback_snapshot_id) {
        historical = &snap;
        break;
      }
    }
    if (!historical) {
      return arrow::Status::Invalid(fmt::format("Snapshot {} not found", *params_.rollback_snapshot_id));
    }
    ARROW_RETURN_NOT_OK(RollbackToSnapshot(md, *historical));
    return Validate(md);
  }

  arrow::Status ApplyRollbackByTimestamp(Metadata& md) {
    const SnapshotEntry* historical = nullptr;
    for (const auto& snap : md.snapshots()) {
      if (snap.timestamp_ms <= *params_.rollback_timestamp_ms) {
        if (!historical || snap.timestamp_ms > historical->timestamp_ms ||
            (snap.timestamp_ms == historical->timestamp_ms && snap.snapshot_id > historical->snapshot_id)) {
          historical = &snap;
        }
      }
    }
    if (!historical) {
      return arrow::Status::Invalid(
          fmt::format("No snapshot found at or before timestamp {}", *params_.rollback_timestamp_ms));
    }
    ARROW_RETURN_NOT_OK(RollbackToSnapshot(md, *historical));
    return Validate(md);
  }

  arrow::Status RollbackToSnapshot(Metadata& md, const SnapshotEntry& historical) {
    SnapshotEntry new_snap;
    new_snap.snapshot_id = md.allocate_snapshot_id();
    new_snap.parent_snapshot_id = md.current_snapshot_id();
    auto now = std::chrono::system_clock::now();
    new_snap.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    new_snap.schema_id = historical.schema_id;
    new_snap.index_spec_id = historical.index_spec_id;
    new_snap.manifest_lists = historical.manifest_lists;

    md.set_current_schema_id(historical.schema_id);
    md.set_current_index_spec_id(historical.index_spec_id);
    md.mutable_snapshots().push_back(std::move(new_snap));
    md.set_current_snapshot_id(md.snapshots().back().snapshot_id);

    return arrow::Status::OK();
  }

  arrow::Status Validate(const Metadata& md) {
    if (md.schemas().empty()) {
      return arrow::Status::Invalid("Collection must have at least one schema");
    }
    const auto& snap = md.snapshots().back();
    if (snap.manifest_lists.empty()) {
      return arrow::Status::Invalid("Snapshot must have at least one partition");
    }
    return arrow::Status::OK();
  }

  arrow::Status ApplySchemaChanges(Metadata& md) {
    if (params_.columns_to_add.empty() && params_.columns_to_drop.empty()) {
      return arrow::Status::OK();
    }

    const SchemaInfo* current = nullptr;
    for (const auto& s : md.schemas()) {
      if (s.schema_id == md.current_schema_id()) {
        current = &s;
        break;
      }
    }
    if (!current) {
      return arrow::Status::Invalid("Current schema not found");
    }

    SchemaInfo new_schema;
    new_schema.schema_id = current->schema_id + 1;
    new_schema.fields = current->fields;
    new_schema.functions = current->functions;

    for (const auto& name : params_.columns_to_drop) {
      auto it = std::remove_if(new_schema.fields.begin(), new_schema.fields.end(),
                               [&name](const FieldSchema& f) { return f.name == name; });
      if (it == new_schema.fields.end()) {
        return arrow::Status::Invalid(fmt::format("Column '{}' not found", name));
      }
      new_schema.fields.erase(it, new_schema.fields.end());
    }

    for (const auto& field : params_.columns_to_add) {
      new_schema.fields.push_back(field);
    }

    md.mutable_schemas().push_back(std::move(new_schema));
    md.set_current_schema_id(md.schemas().back().schema_id);
    return arrow::Status::OK();
  }

  arrow::Status ApplyIndexChanges(Metadata& md) {
    if (params_.indexes_to_add.empty() && params_.indexes_to_drop.empty()) {
      return arrow::Status::OK();
    }

    const IndexSpec* current = nullptr;
    for (const auto& s : md.index_specs()) {
      if (s.spec_id == md.current_index_spec_id()) {
        current = &s;
        break;
      }
    }

    IndexSpec new_spec;
    if (current) {
      new_spec.spec_id = current->spec_id + 1;
      new_spec.indexes = current->indexes;
    } else {
      new_spec.spec_id = 0;
    }

    for (const auto& name : params_.indexes_to_drop) {
      auto it = std::remove_if(new_spec.indexes.begin(), new_spec.indexes.end(),
                               [&name](const IndexInfo& idx) { return idx.index_name == name; });
      if (it == new_spec.indexes.end()) {
        return arrow::Status::Invalid(fmt::format("Index '{}' not found", name));
      }
      new_spec.indexes.erase(it, new_spec.indexes.end());
    }

    for (const auto& idx : params_.indexes_to_add) {
      new_spec.indexes.push_back(idx);
    }

    md.mutable_index_specs().push_back(std::move(new_spec));
    md.set_current_index_spec_id(md.index_specs().back().spec_id);
    return arrow::Status::OK();
  }

  arrow::Result<std::vector<ManifestListInfo>> ApplyManifestChanges(const Metadata& md) {
    // Load all entries from current snapshot's manifest lists
    std::vector<ManifestListEntry> entries;
    for (const auto& snap : md.snapshots()) {
      if (snap.snapshot_id == md.current_snapshot_id()) {
        for (const auto& ml_info : snap.manifest_lists) {
          ARROW_ASSIGN_OR_RAISE(auto ml, ReadManifestListFromFile(params_.fs, ml_info.manifest_list));
          for (auto& entry : ml.entries()) {
            entries.push_back(std::move(entry));
          }
        }
        break;
      }
    }

    int64_t max_id = 0;
    for (const auto& entry : entries) {
      max_id = std::max(max_id, entry.partition_id);
    }

    // Drop partitions
    for (const auto& name : params_.partitions_to_drop) {
      auto it = std::remove_if(entries.begin(), entries.end(),
                               [&name](const ManifestListEntry& e) { return e.partition_name == name; });
      if (it == entries.end()) {
        return arrow::Status::Invalid(fmt::format("Partition '{}' not found", name));
      }
      entries.erase(it, entries.end());
    }

    // Resolve partition identity for segment adds (using local copies)
    struct ResolvedAdd {
      int64_t partition_id;
      std::string partition_name;
      SegmentInfo segment;
    };

    std::vector<ResolvedAdd> resolved_adds;
    if (!params_.segment_adds.empty()) {
      for (const auto& add : params_.segment_adds) {
        ResolvedAdd r{add.partition_id, add.partition_name, add.segment};
        if (r.partition_name.empty()) {
          // Lookup name by id
          bool found = false;
          for (const auto& entry : entries) {
            if (entry.partition_id == r.partition_id) {
              r.partition_name = entry.partition_name;
              found = true;
              break;
            }
          }
          if (!found) {
            return arrow::Status::Invalid(fmt::format("Partition with id {} not found", r.partition_id));
          }
        } else if (r.partition_id == 0) {
          // Lookup id by name in existing entries
          bool found = false;
          for (const auto& entry : entries) {
            if (entry.partition_name == r.partition_name) {
              r.partition_id = entry.partition_id;
              found = true;
              break;
            }
          }
          // Also check previously resolved adds in this action
          if (!found) {
            for (const auto& prev : resolved_adds) {
              if (prev.partition_name == r.partition_name) {
                r.partition_id = prev.partition_id;
                found = true;
                break;
              }
            }
          }
          if (!found) {
            r.partition_id = ++max_id;
          }
        }
        resolved_adds.push_back(std::move(r));
      }
    }

    // Apply segment removals
    for (auto& entry : entries) {
      entry.segments.erase(std::remove_if(entry.segments.begin(), entry.segments.end(),
                                          [this](const SegmentInfo& seg) {
                                            return std::find(params_.segment_removes.begin(),
                                                             params_.segment_removes.end(),
                                                             seg.segment_id) != params_.segment_removes.end();
                                          }),
                           entry.segments.end());
    }

    // Apply segment additions
    for (const auto& add : resolved_adds) {
      bool found = false;
      for (auto& entry : entries) {
        if (entry.partition_id == add.partition_id) {
          entry.segments.push_back(add.segment);
          found = true;
          break;
        }
      }
      if (!found) {
        ManifestListEntry new_entry;
        new_entry.partition_id = add.partition_id;
        new_entry.partition_name = add.partition_name;
        new_entry.segments = {add.segment};
        entries.push_back(std::move(new_entry));
      }
    }

    // Add partitions
    for (const auto& name : params_.partitions_to_add) {
      bool exists = false;
      for (const auto& entry : entries) {
        if (entry.partition_name == name) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        ManifestListEntry new_entry;
        new_entry.partition_id = ++max_id;
        new_entry.partition_name = name;
        entries.push_back(std::move(new_entry));
      }
    }

    // Write new manifest list file and return ManifestListInfo
    std::vector<ManifestListInfo> result;
    if (!entries.empty()) {
      ManifestList ml(std::move(entries));
      ARROW_ASSIGN_OR_RAISE(auto path, WriteManifestListToFile(params_.fs, params_.base_path, ml));

      std::vector<int64_t> part_ids;
      std::vector<std::string> part_names;
      for (const auto& entry : ml.entries()) {
        part_ids.push_back(entry.partition_id);
        part_names.push_back(entry.partition_name);
      }

      result.push_back({
          .manifest_list = path,
          .partition_ids = std::move(part_ids),
          .partition_names = std::move(part_names),
      });
    }

    return result;
  }
};

}  // namespace

// ---- ActionBuilder ----

ActionBuilder::ActionBuilder() : impl_(std::make_unique<Impl>()) {}
ActionBuilder::~ActionBuilder() = default;
ActionBuilder::ActionBuilder(ActionBuilder&&) noexcept = default;
ActionBuilder& ActionBuilder::operator=(ActionBuilder&&) noexcept = default;

ActionBuilder ActionBuilder::Create(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& base_path) {
  ActionBuilder builder;
  builder.impl_->fs = fs;
  builder.impl_->base_path = base_path;
  return builder;
}

ActionBuilder& ActionBuilder::SetCollectionInfo(CollectionInfo info) {
  impl_->collection_info = std::move(info);
  return *this;
}

ActionBuilder& ActionBuilder::SetSchema(SchemaInfo schema) {
  impl_->schema = std::move(schema);
  return *this;
}

ActionBuilder& ActionBuilder::AddPartition(const std::string& partition_name) {
  impl_->partitions_to_add.push_back(partition_name);
  return *this;
}

ActionBuilder& ActionBuilder::DropPartition(const std::string& partition_name) {
  impl_->partitions_to_drop.push_back(partition_name);
  return *this;
}

ActionBuilder& ActionBuilder::AddSegment(const std::string& partition_name, SegmentInfo segment) {
  impl_->segment_adds.push_back({0, partition_name, std::move(segment)});
  return *this;
}

ActionBuilder& ActionBuilder::AddSegment(int64_t partition_id, const std::string& partition_name, SegmentInfo segment) {
  impl_->segment_adds.push_back({partition_id, partition_name, std::move(segment)});
  return *this;
}

ActionBuilder& ActionBuilder::RemoveSegments(std::vector<int64_t> segment_ids) {
  impl_->segment_removes.insert(impl_->segment_removes.end(), segment_ids.begin(), segment_ids.end());
  return *this;
}

ActionBuilder& ActionBuilder::AddColumn(FieldSchema field) {
  impl_->columns_to_add.push_back(std::move(field));
  return *this;
}

ActionBuilder& ActionBuilder::DropColumn(const std::string& field_name) {
  impl_->columns_to_drop.push_back(field_name);
  return *this;
}

ActionBuilder& ActionBuilder::AddIndex(IndexInfo index) {
  impl_->indexes_to_add.push_back(std::move(index));
  return *this;
}

ActionBuilder& ActionBuilder::DropIndex(const std::string& index_name) {
  impl_->indexes_to_drop.push_back(index_name);
  return *this;
}

ActionBuilder& ActionBuilder::SetCurrentSnapshot(int64_t snapshot_id) {
  impl_->rollback_snapshot_id = snapshot_id;
  return *this;
}

ActionBuilder& ActionBuilder::SetCurrentSnapshotByTimestamp(int64_t timestamp_ms) {
  impl_->rollback_timestamp_ms = timestamp_ms;
  return *this;
}

std::shared_ptr<Action> ActionBuilder::Build() {
  return std::make_shared<ActionImpl>(std::move(static_cast<ActionParams&>(*impl_)));
}

}  // namespace milvus_storage::api::table_format
