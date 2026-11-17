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

#include <gtest/gtest.h>

#include <sstream>

#include "milvus-storage/table_format/metadata.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

static Metadata MakeTestMetadata(int64_t collection_id, const std::string& name) {
  Metadata md;
  md.set_format_version(1);
  md.mutable_collection() = {
      .collection_id = collection_id,
      .name = name,
      .db_id = 1,
      .created_at = 1700000000000,
  };

  FieldSchema f1;
  f1.field_id = 1;
  f1.name = "pk";
  f1.data_type = DataType::Int64;
  f1.is_primary_key = true;

  FieldSchema f2;
  f2.field_id = 2;
  f2.name = "vec";
  f2.data_type = DataType::FloatVector;
  f2.type_params = {{"dim", "128"}};

  SchemaInfo schema;
  schema.schema_id = 0;
  schema.fields = {f1, f2};
  md.mutable_schemas() = {schema};

  return md;
}

TEST(CollMetadataTest, SerializeDeserializeRoundtrip) {
  auto md = MakeTestMetadata(100, "roundtrip_test");

  // Add a snapshot entry with manifest list ref
  SnapshotEntry snap;
  snap.snapshot_id = 1;
  snap.timestamp_ms = 1700000001000;
  snap.manifest_lists = {
      {.manifest_list = "_manifests/ml1.avro",
       .partition_ids = {1},
       .partition_names = {"_default"}},
  };
  md.mutable_snapshots() = {snap};
  md.set_current_snapshot_id(1);

  std::stringstream ss;
  ASSERT_STATUS_OK(md.serialize(ss));

  Metadata restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  EXPECT_EQ(restored.collection().collection_id, 100);
  EXPECT_EQ(restored.collection().name, "roundtrip_test");
  ASSERT_EQ(restored.schemas().size(), 1u);
  ASSERT_EQ(restored.schemas()[0].fields.size(), 2u);
  ASSERT_EQ(restored.snapshots().size(), 1u);
  ASSERT_EQ(restored.snapshots()[0].manifest_lists[0].partition_ids.size(), 1u);
  EXPECT_EQ(restored.snapshots()[0].manifest_lists[0].partition_ids[0], 1);
}

}  // namespace milvus_storage::api::table_format
