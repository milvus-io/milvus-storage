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
#include "milvus-storage/table_format/collection_transaction.h"
#include "milvus-storage/table_format/layout.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

static SchemaInfo MakeTestSchema() {
  FieldSchema pk;
  pk.field_id = 1;
  pk.name = "pk";
  pk.data_type = DataType::Int64;
  pk.is_primary_key = true;

  FieldSchema vec;
  vec.field_id = 2;
  vec.name = "vec";
  vec.data_type = DataType::FloatVector;
  vec.type_params = {{"dim", "128"}};

  SchemaInfo schema;
  schema.schema_id = 0;
  schema.fields = {pk, vec};
  return schema;
}

TEST(TableFormatIntegrationTest, EndToEnd) {
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  auto base_path = milvus_storage::GetTestBasePath("table-format-e2e");
  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs, base_path));

  // 1. Create collection and add first segment via actions
  int64_t v1;
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));

    ASSERT_AND_ASSIGN(v1, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCollectionInfo({.collection_id = 1, .name = "test_collection", .db_id = 1, .created_at = 1700000000000})
            .SetSchema(MakeTestSchema())
            .AddSegment("_default", {.segment_id = 1001,
                                     .manifest = "_manifests/1001.avro",
                                     .level = SegmentLevel::L1,
                                     .num_rows = 100000,
                                     .file_size = 512 * 1024 * 1024})
            .Build()));
    EXPECT_EQ(v1, 1);
  }

  // 2. Add second segment
  int64_t v2;
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(v2, txn->Commit(ActionBuilder::Create(fs, base_path)
                                    .AddSegment("_default", {.segment_id = 1002,
                                                             .manifest = "_manifests/1002.avro",
                                                             .level = SegmentLevel::L1,
                                                             .num_rows = 200000,
                                                             .file_size = 256 * 1024 * 1024})
                                    .Build()));
    EXPECT_EQ(v2, 2);
  }

  // 3. Read back via CollectionTransaction (latest = v2)
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    EXPECT_EQ(txn->GetReadVersion(), 2);
    EXPECT_EQ(txn->GetMetadata().collection().name, "test_collection");

    ASSERT_AND_ASSIGN(auto snapshot, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto segments, txn->ListSegments(*snapshot));
    ASSERT_EQ(segments.size(), 1u);              // 1 partition
    ASSERT_EQ(segments[0].segments.size(), 2u);  // 2 segments
    EXPECT_EQ(segments[0].partition_name, "_default");
  }

  // 4. Time travel: read v1
  {
    ASSERT_AND_ASSIGN(auto txn_v1, CollectionTransaction::Open(fs, base_path, v1));
    EXPECT_EQ(txn_v1->GetReadVersion(), 1);

    ASSERT_AND_ASSIGN(auto snap_v1, txn_v1->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto segs_v1, txn_v1->ListSegments(*snap_v1));
    ASSERT_EQ(segs_v1.size(), 1u);
    EXPECT_EQ(segs_v1[0].segments.size(), 1u);  // only 1 segment in v1
  }

  // 5. Remove segment (simulating compaction) and add new one
  int64_t v3;
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(v3, txn->Commit(ActionBuilder::Create(fs, base_path)
                                    .RemoveSegments({1001})
                                    .AddSegment("_default", {.segment_id = 2001,
                                                             .manifest = "_manifests/2001.avro",
                                                             .level = SegmentLevel::L2,
                                                             .num_rows = 300000,
                                                             .sorted = true})
                                    .Build()));
    EXPECT_EQ(v3, 3);
  }

  // 6. Verify current state after compaction
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    EXPECT_EQ(txn->GetReadVersion(), 3);

    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto segments, txn->ListSegments(*snap));
    ASSERT_EQ(segments.size(), 1u);
    ASSERT_EQ(segments[0].segments.size(), 2u);  // 1002 + 2001 (1001 removed)

    bool found_1002 = false, found_2001 = false;
    for (const auto& seg : segments[0].segments) {
      if (seg.segment_id == 1002) {
        found_1002 = true;
      }
      if (seg.segment_id == 2001) {
        found_2001 = true;
        EXPECT_EQ(seg.level, SegmentLevel::L2);
        EXPECT_TRUE(seg.sorted);
      }
    }
    EXPECT_TRUE(found_1002);
    EXPECT_TRUE(found_2001);

    ASSERT_AND_ASSIGN(auto schema, txn->GetSchema(0));
    EXPECT_EQ(schema->fields.size(), 2u);
    EXPECT_EQ(schema->fields[0].name, "pk");
    EXPECT_EQ(schema->fields[1].name, "vec");
  }

  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
}

TEST(TableFormatIntegrationTest, PartitionManagement) {
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  auto base_path = milvus_storage::GetTestBasePath("table-format-partition");
  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs, base_path));

  // 1. Create collection with explicit partitions
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));

    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCollectionInfo({.collection_id = 2, .name = "partitioned_collection", .db_id = 1, .created_at = 1700000000000})
            .SetSchema(MakeTestSchema())
            .AddPartition("_default")
            .AddPartition("hot")
            .AddPartition("cold")
            .Build()));
    EXPECT_EQ(v, 1);
  }

  // 2. Verify 3 empty partitions
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].partition_name, "_default");
    EXPECT_EQ(entries[1].partition_name, "hot");
    EXPECT_EQ(entries[2].partition_name, "cold");
    EXPECT_TRUE(entries[0].segments.empty());
    EXPECT_TRUE(entries[1].segments.empty());
    EXPECT_TRUE(entries[2].segments.empty());
  }

  // 3. Add segments to partitions
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .AddSegment("_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
            .AddSegment("hot", {.segment_id = 2001, .manifest = "_manifests/2001.avro"})
            .AddSegment("cold", {.segment_id = 3001, .manifest = "_manifests/3001.avro"})
            .Build()));
    EXPECT_EQ(v, 2);
  }

  // 4. Drop partition "cold"
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .DropPartition("cold")
            .Build()));
    EXPECT_EQ(v, 3);
  }

  // 5. Verify: 2 partitions remain, cold is gone
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 2u);

    bool found_default = false, found_hot = false;
    for (const auto& e : entries) {
      if (e.partition_name == "_default") {
        found_default = true;
        ASSERT_EQ(e.segments.size(), 1u);
        EXPECT_EQ(e.segments[0].segment_id, 1001);
      }
      if (e.partition_name == "hot") {
        found_hot = true;
        ASSERT_EQ(e.segments.size(), 1u);
        EXPECT_EQ(e.segments[0].segment_id, 2001);
      }
    }
    EXPECT_TRUE(found_default);
    EXPECT_TRUE(found_hot);
  }

  // 6. Drop non-existent partition fails at commit
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    auto result = txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .DropPartition("nonexistent")
            .Build());
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsInvalid());
  }

  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
}

TEST(TableFormatIntegrationTest, SnapshotRollback) {
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  auto base_path = milvus_storage::GetTestBasePath("table-format-rollback");
  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs, base_path));

  // 1. Create collection with 2 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCollectionInfo({.collection_id = 1, .name = "test", .db_id = 1, .created_at = 1700000000000})
            .SetSchema(MakeTestSchema())
            .AddSegment("_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro", .num_rows = 100})
            .AddSegment("_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro", .num_rows = 200})
            .Build()));
    EXPECT_EQ(v, 1);
  }

  // 2. Add a third segment (v2 has 3 segments)
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .AddSegment("_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro", .num_rows = 300})
            .Build()));
    EXPECT_EQ(v, 2);
  }

  // Verify v2 has 3 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].segments.size(), 3u);
  }

  // 3. Rollback to snapshot 1 (v1 state: 2 segments)
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCurrentSnapshot(1)
            .Build()));
    EXPECT_EQ(v, 3);
  }

  // 4. Verify current state is back to 2 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    EXPECT_EQ(txn->GetReadVersion(), 3);

    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 1u);
    ASSERT_EQ(entries[0].segments.size(), 2u);
    EXPECT_EQ(entries[0].segments[0].segment_id, 1001);
    EXPECT_EQ(entries[0].segments[1].segment_id, 1002);

    // History preserved: 3 snapshots exist
    EXPECT_EQ(txn->GetMetadata().snapshots().size(), 3u);
  }

  // 5. Rollback to non-existent snapshot fails at commit
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    auto result = txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCurrentSnapshot(999)
            .Build());
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsInvalid());
  }

  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
}

TEST(TableFormatIntegrationTest, SnapshotRollbackByTimestamp) {
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  auto base_path = milvus_storage::GetTestBasePath("table-format-rollback-ts");
  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs, base_path));

  // 1. Create collection with 2 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCollectionInfo({.collection_id = 1, .name = "test", .db_id = 1, .created_at = 1700000000000})
            .SetSchema(MakeTestSchema())
            .AddSegment("_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro", .num_rows = 100})
            .AddSegment("_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro", .num_rows = 200})
            .Build()));
    EXPECT_EQ(v, 1);
  }

  // Record the timestamp after v1 is created
  int64_t ts_after_v1;
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ts_after_v1 = snap->timestamp_ms;
  }

  // 2. Add a third segment (v2 has 3 segments)
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .AddSegment("_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro", .num_rows = 300})
            .Build()));
    EXPECT_EQ(v, 2);
  }

  // Verify v2 has 3 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].segments.size(), 3u);
  }

  // 3. Rollback to timestamp of v1 (should restore 2 segments)
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCurrentSnapshotByTimestamp(ts_after_v1)
            .Build()));
    EXPECT_EQ(v, 3);
  }

  // 4. Verify current state is back to 2 segments
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    EXPECT_EQ(txn->GetReadVersion(), 3);

    ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
    ASSERT_AND_ASSIGN(auto entries, txn->ListSegments(*snap));
    ASSERT_EQ(entries.size(), 1u);
    ASSERT_EQ(entries[0].segments.size(), 2u);
    EXPECT_EQ(entries[0].segments[0].segment_id, 1001);
    EXPECT_EQ(entries[0].segments[1].segment_id, 1002);

    // History preserved: 3 snapshots exist
    EXPECT_EQ(txn->GetMetadata().snapshots().size(), 3u);
  }

  // 5. Rollback to timestamp before any snapshot fails
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs, base_path));
    auto result = txn->Commit(
        ActionBuilder::Create(fs, base_path)
            .SetCurrentSnapshotByTimestamp(0)
            .Build());
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsInvalid());
  }

  ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs, base_path));
}

}  // namespace milvus_storage::api::table_format
