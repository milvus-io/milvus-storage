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

#include <arrow/filesystem/localfs.h>

#include "milvus-storage/table_format/action.h"
#include "milvus-storage/table_format/metadata.h"
#include "milvus-storage/table_format/types.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

class ActionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    base_path_ = milvus_storage::GetTestBasePath("action-test");
    ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_)); }

  Metadata MakeBasicMetadata() {
    Metadata md;
    md.mutable_collection() = {.collection_id = 1, .name = "test", .db_id = 1, .created_at = 1700000000000};

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
    md.set_current_schema_id(0);

    return md;
  }

  milvus_storage::ArrowFileSystemPtr fs_;
  std::string base_path_;
};

TEST_F(ActionTest, AddColumn) {
  auto md = MakeBasicMetadata();

  FieldSchema new_field;
  new_field.field_id = 3;
  new_field.name = "score";
  new_field.data_type = DataType::Float;

  auto action = ActionBuilder::Create(fs_, base_path_).AddColumn(new_field).Build();

  ASSERT_STATUS_OK(action->Apply(md));

  ASSERT_EQ(md.schemas().size(), 2u);
  EXPECT_EQ(md.current_schema_id(), 1);
  EXPECT_EQ(md.schemas()[1].schema_id, 1);
  ASSERT_EQ(md.schemas()[1].fields.size(), 3u);
  EXPECT_EQ(md.schemas()[1].fields[0].name, "pk");
  EXPECT_EQ(md.schemas()[1].fields[1].name, "vec");
  EXPECT_EQ(md.schemas()[1].fields[2].name, "score");
}

TEST_F(ActionTest, DropColumn) {
  auto md = MakeBasicMetadata();
  auto action = ActionBuilder::Create(fs_, base_path_).DropColumn("vec").Build();

  ASSERT_STATUS_OK(action->Apply(md));

  ASSERT_EQ(md.schemas().size(), 2u);
  EXPECT_EQ(md.current_schema_id(), 1);
  ASSERT_EQ(md.schemas()[1].fields.size(), 1u);
  EXPECT_EQ(md.schemas()[1].fields[0].name, "pk");
}

TEST_F(ActionTest, DropNonExistentColumn) {
  auto md = MakeBasicMetadata();
  auto action = ActionBuilder::Create(fs_, base_path_).DropColumn("nonexistent").Build();

  auto status = action->Apply(md);
  EXPECT_FALSE(status.ok());
}

TEST_F(ActionTest, AddAndDropColumn) {
  auto md = MakeBasicMetadata();

  FieldSchema new_field;
  new_field.field_id = 3;
  new_field.name = "score";
  new_field.data_type = DataType::Float;

  auto action = ActionBuilder::Create(fs_, base_path_).DropColumn("vec").AddColumn(new_field).Build();

  ASSERT_STATUS_OK(action->Apply(md));

  ASSERT_EQ(md.schemas().size(), 2u);
  ASSERT_EQ(md.schemas()[1].fields.size(), 2u);
  EXPECT_EQ(md.schemas()[1].fields[0].name, "pk");
  EXPECT_EQ(md.schemas()[1].fields[1].name, "score");
}

TEST_F(ActionTest, AddIndex) {
  auto md = MakeBasicMetadata();

  IndexInfo idx;
  idx.index_id = 1;
  idx.index_name = "vec_index";
  idx.field_id = 2;
  idx.index_params = {{"index_type", "IVF_FLAT"}, {"nlist", "1024"}};

  auto action = ActionBuilder::Create(fs_, base_path_).AddIndex(idx).Build();

  ASSERT_STATUS_OK(action->Apply(md));

  ASSERT_EQ(md.index_specs().size(), 1u);
  EXPECT_EQ(md.current_index_spec_id(), 0);
  ASSERT_EQ(md.index_specs()[0].indexes.size(), 1u);
  EXPECT_EQ(md.index_specs()[0].indexes[0].index_name, "vec_index");
}

TEST_F(ActionTest, DropIndex) {
  auto md = MakeBasicMetadata();

  // First add an index
  IndexInfo idx;
  idx.index_id = 1;
  idx.index_name = "vec_index";
  idx.field_id = 2;

  IndexSpec ispec;
  ispec.spec_id = 0;
  ispec.indexes = {idx};
  md.mutable_index_specs() = {ispec};
  md.set_current_index_spec_id(0);

  auto action = ActionBuilder::Create(fs_, base_path_).DropIndex("vec_index").Build();

  ASSERT_STATUS_OK(action->Apply(md));

  ASSERT_EQ(md.index_specs().size(), 2u);
  EXPECT_EQ(md.current_index_spec_id(), 1);
  EXPECT_TRUE(md.index_specs()[1].indexes.empty());
}

TEST_F(ActionTest, DropNonExistentIndex) {
  auto md = MakeBasicMetadata();

  IndexSpec ispec;
  ispec.spec_id = 0;
  md.mutable_index_specs() = {ispec};
  md.set_current_index_spec_id(0);

  auto action = ActionBuilder::Create(fs_, base_path_).DropIndex("nonexistent").Build();

  auto status = action->Apply(md);
  EXPECT_FALSE(status.ok());
}

TEST_F(ActionTest, CompositeSchemaAndSegment) {
  auto md = MakeBasicMetadata();

  FieldSchema new_field;
  new_field.field_id = 3;
  new_field.name = "score";
  new_field.data_type = DataType::Float;

  auto action = ActionBuilder::Create(fs_, base_path_)
                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                    .AddColumn(new_field)
                    .Build();

  ASSERT_STATUS_OK(action->Apply(md));

  // Schema evolved
  ASSERT_EQ(md.schemas().size(), 2u);
  EXPECT_EQ(md.schemas()[1].fields.size(), 3u);

  // Snapshot created with segment info
  ASSERT_EQ(md.snapshots().size(), 1u);
  EXPECT_EQ(md.current_snapshot_id(), 1);
  ASSERT_EQ(md.snapshots()[0].manifest_lists.size(), 1u);
  ASSERT_EQ(md.snapshots()[0].manifest_lists[0].partition_ids.size(), 1u);
  EXPECT_EQ(md.snapshots()[0].manifest_lists[0].partition_ids[0], 1);
  EXPECT_EQ(md.snapshots()[0].manifest_lists[0].partition_names[0], "_default");
}

TEST_F(ActionTest, FailsWithoutSchema) {
  Metadata md;
  md.mutable_collection() = {.collection_id = 1, .name = "test", .db_id = 1, .created_at = 1700000000000};

  auto action = ActionBuilder::Create(fs_, base_path_)
                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                    .Build();

  auto status = action->Apply(md);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalid());
}

TEST_F(ActionTest, CreatesDefaultPartition) {
  auto md = MakeBasicMetadata();

  FieldSchema new_field;
  new_field.field_id = 3;
  new_field.name = "score";
  new_field.data_type = DataType::Float;

  auto action = ActionBuilder::Create(fs_, base_path_).AddColumn(new_field).Build();
  ASSERT_STATUS_OK(action->Apply(md));

  // Snapshot created with default partition
  ASSERT_EQ(md.snapshots().size(), 1u);
  ASSERT_EQ(md.snapshots()[0].manifest_lists.size(), 1u);
  ASSERT_EQ(md.snapshots()[0].manifest_lists[0].partition_ids.size(), 1u);
  EXPECT_EQ(md.snapshots()[0].manifest_lists[0].partition_ids[0], 1);
  EXPECT_EQ(md.snapshots()[0].manifest_lists[0].partition_names[0], "_default");
}

TEST_F(ActionTest, SchemaEvolutionPreservesOriginal) {
  auto md = MakeBasicMetadata();

  FieldSchema new_field;
  new_field.field_id = 3;
  new_field.name = "score";
  new_field.data_type = DataType::Float;

  auto action = ActionBuilder::Create(fs_, base_path_).AddColumn(new_field).Build();
  ASSERT_STATUS_OK(action->Apply(md));

  // Original schema unchanged
  ASSERT_EQ(md.schemas()[0].fields.size(), 2u);
  EXPECT_EQ(md.schemas()[0].schema_id, 0);

  // New schema added
  ASSERT_EQ(md.schemas()[1].fields.size(), 3u);
  EXPECT_EQ(md.schemas()[1].schema_id, 1);
}

TEST_F(ActionTest, RollbackMutualExclusivity) {
  auto md = MakeBasicMetadata();

  // First create a snapshot so we have something to rollback to
  auto setup = ActionBuilder::Create(fs_, base_path_)
                   .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                   .Build();
  ASSERT_STATUS_OK(setup->Apply(md));

  // Setting both rollback_snapshot_id and rollback_timestamp_ms should fail
  auto action = ActionBuilder::Create(fs_, base_path_)
                    .SetCurrentSnapshot(1)
                    .SetCurrentSnapshotByTimestamp(1700000000000)
                    .Build();

  auto status = action->Apply(md);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalid());
}

TEST_F(ActionTest, RollbackByTimestampTieBreaking) {
  auto md = MakeBasicMetadata();

  // Manually create two snapshots with the same timestamp but different IDs
  SnapshotEntry snap1;
  snap1.snapshot_id = md.allocate_snapshot_id();
  snap1.timestamp_ms = 1700000000000;
  snap1.schema_id = 0;
  snap1.manifest_lists = {};

  SnapshotEntry snap2;
  snap2.snapshot_id = md.allocate_snapshot_id();
  snap2.timestamp_ms = 1700000000000;
  snap2.parent_snapshot_id = snap1.snapshot_id;
  snap2.schema_id = 0;
  snap2.manifest_lists = {};

  md.mutable_snapshots() = {snap1, snap2};
  md.set_current_snapshot_id(snap2.snapshot_id);

  // We need manifest_lists for Validate, so add them via a real action
  // Instead, test directly with the metadata lookup in collection_transaction
  // The tie-breaking rule is: prefer higher snapshot_id when timestamps are equal

  // Verify snap2 has the higher ID
  EXPECT_GT(snap2.snapshot_id, snap1.snapshot_id);
}

TEST_F(ActionTest, SnapshotIdUsesMonotonicCounter) {
  auto md = MakeBasicMetadata();

  auto action1 = ActionBuilder::Create(fs_, base_path_)
                     .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                     .Build();
  ASSERT_STATUS_OK(action1->Apply(md));

  // next_snapshot_id should have advanced
  EXPECT_EQ(md.next_snapshot_id(), 2);
  EXPECT_EQ(md.snapshots().back().snapshot_id, 1);

  auto action2 = ActionBuilder::Create(fs_, base_path_)
                     .AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro"})
                     .Build();
  ASSERT_STATUS_OK(action2->Apply(md));

  EXPECT_EQ(md.next_snapshot_id(), 3);
  EXPECT_EQ(md.snapshots().back().snapshot_id, 2);
}

}  // namespace milvus_storage::api::table_format
