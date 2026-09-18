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
#include <string>
#include <vector>

#include <arrow/status.h>

#include "milvus-storage/table_format/types.h"

namespace avro {
template <typename T>
struct codec_traits;
}

namespace milvus_storage::api::table_format {

class Metadata final {
  public:
  Metadata() = default;

  // Format version
  int32_t format_version() const { return format_version_; }
  void set_format_version(int32_t v) { format_version_ = v; }

  // Collection info
  const CollectionInfo& collection() const { return collection_; }
  CollectionInfo& mutable_collection() { return collection_; }

  // Schemas
  const std::vector<SchemaInfo>& schemas() const { return schemas_; }
  std::vector<SchemaInfo>& mutable_schemas() { return schemas_; }
  int32_t current_schema_id() const { return current_schema_id_; }
  void set_current_schema_id(int32_t id) { current_schema_id_ = id; }

  // Index specs
  const std::vector<IndexSpec>& index_specs() const { return index_specs_; }
  std::vector<IndexSpec>& mutable_index_specs() { return index_specs_; }
  int32_t current_index_spec_id() const { return current_index_spec_id_; }
  void set_current_index_spec_id(int32_t id) { current_index_spec_id_ = id; }

  // Snapshots
  const std::vector<SnapshotEntry>& snapshots() const { return snapshots_; }
  std::vector<SnapshotEntry>& mutable_snapshots() { return snapshots_; }
  int64_t current_snapshot_id() const { return current_snapshot_id_; }
  void set_current_snapshot_id(int64_t id) { current_snapshot_id_ = id; }

  // Monotonically increasing snapshot ID counter (like Iceberg's last-sequence-number).
  // Use allocate_snapshot_id() to obtain the next ID and advance the counter.
  int64_t next_snapshot_id() const { return next_snapshot_id_; }
  void set_next_snapshot_id(int64_t id) { next_snapshot_id_ = id; }
  int64_t allocate_snapshot_id() { return next_snapshot_id_++; }

  // Stream I/O
  [[nodiscard]] arrow::Status serialize(std::ostream& output_stream) const;
  arrow::Status deserialize(std::istream& input_stream);

  private:
  int32_t format_version_ = 1;
  CollectionInfo collection_;
  std::vector<SchemaInfo> schemas_;
  int32_t current_schema_id_ = 0;
  std::vector<IndexSpec> index_specs_;
  int32_t current_index_spec_id_ = 0;
  std::vector<SnapshotEntry> snapshots_;
  int64_t current_snapshot_id_ = 0;
  int64_t next_snapshot_id_ = 1;
  friend struct avro::codec_traits<Metadata>;
};

}  // namespace milvus_storage::api::table_format
