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
#include "test_env.h"

namespace milvus_storage::api::table_format {

class CollectionTransactionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    base_path_ = milvus_storage::GetTestBasePath("coll-txn-test");
    ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(milvus_storage::CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(milvus_storage::DeleteTestDir(fs_, base_path_)); }

  void SetupCollection(CollectionTransaction& txn) {
    auto& md = txn.GetSnapshot();
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

    SchemaInfo schema0;
    schema0.schema_id = 0;
    schema0.fields = {f1, f2};

    SchemaInfo schema1;
    schema1.schema_id = 1;
    schema1.fields = {f1};

    md.mutable_schemas() = {schema0, schema1};
    md.set_current_schema_id(0);
  }

  milvus_storage::ArrowFileSystemPtr fs_;
  std::string base_path_;
};

// ---- Write / Commit tests ----

TEST_F(CollectionTransactionTest, FirstCommit) {
  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  SetupCollection(*txn);

  ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                  .AddSegment(1, "_default", {.segment_id = 1001,
                                                  .manifest = "_manifests/1001.avro",
                                                  .level = SegmentLevel::L1,
                                                  .num_rows = 100000,
                                                  .file_size = 512 * 1024 * 1024})
                                  .Build()));
  EXPECT_EQ(v, 1);

  // Verify by re-opening
  ASSERT_AND_ASSIGN(auto txn2, CollectionTransaction::Open(fs_, base_path_));
  EXPECT_EQ(txn2->GetReadVersion(), 1);
  EXPECT_EQ(txn2->GetMetadata().collection().name, "test");
  ASSERT_EQ(txn2->GetMetadata().snapshots().size(), 1u);
  EXPECT_EQ(txn2->GetMetadata().current_snapshot_id(), 1);
}

TEST_F(CollectionTransactionTest, SecondCommit) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro", .num_rows = 100}).Build()));
    EXPECT_EQ(v, 1);
  }
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(
        ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro", .num_rows = 200}).Build()));
    EXPECT_EQ(v, 2);
  }

  ASSERT_AND_ASSIGN(auto txn3, CollectionTransaction::Open(fs_, base_path_));
  EXPECT_EQ(txn3->GetReadVersion(), 2);
  ASSERT_EQ(txn3->GetMetadata().snapshots().size(), 2u);
}

TEST_F(CollectionTransactionTest, CompositeAction) {
  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  SetupCollection(*txn);

  ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                  .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                                  .AddSegment(2, "part_a", {.segment_id = 2001, .manifest = "_manifests/2001.avro"})
                                  .AddSegment(3, "part_b", {.segment_id = 3001, .manifest = "_manifests/3001.avro"})
                                  .Build()));
  EXPECT_EQ(v, 1);
}

TEST_F(CollectionTransactionTest, RemoveSegment) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                                    .AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro"})
                                    .AddSegment(1, "_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro"})
                                    .Build()));
    EXPECT_EQ(v, 1);
  }
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_).RemoveSegments({1002}).Build()));
    EXPECT_EQ(v, 2);
  }
}

TEST_F(CollectionTransactionTest, ConcurrentCommitWithReplay) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"}).Build()));
    EXPECT_EQ(v, 1);
  }

  ASSERT_AND_ASSIGN(auto txn1, CollectionTransaction::Open(fs_, base_path_, LATEST_VERSION, 0));
  ASSERT_AND_ASSIGN(auto txn2, CollectionTransaction::Open(fs_, base_path_, LATEST_VERSION, 0));

  ASSERT_AND_ASSIGN(auto v1, txn1->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro"}).Build()));
  EXPECT_EQ(v1, 2);

  ASSERT_AND_ASSIGN(auto v2, txn2->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro"}).Build()));
  EXPECT_EQ(v2, 3);
}

TEST_F(CollectionTransactionTest, ConcurrentCommitRetrySucceeds) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"}).Build()));
    EXPECT_EQ(v, 1);
  }

  ASSERT_AND_ASSIGN(auto txn1, CollectionTransaction::Open(fs_, base_path_));
  ASSERT_AND_ASSIGN(auto txn2, CollectionTransaction::Open(fs_, base_path_));

  ASSERT_AND_ASSIGN(auto v1, txn1->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro"}).Build()));
  EXPECT_EQ(v1, 2);

  ASSERT_AND_ASSIGN(auto v2, txn2->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro"}).Build()));
  EXPECT_EQ(v2, 3);
}

// ---- Read / Query tests (merged from collection_reader_test.cpp) ----

TEST_F(CollectionTransactionTest, OpenAndReadCurrent) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro", .num_rows = 100})
                                    .AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro", .num_rows = 200})
                                    .AddSegment(2, "part_a", {.segment_id = 2001, .manifest = "_manifests/2001.avro", .num_rows = 300})
                                    .Build()));
    EXPECT_EQ(v, 1);
  }

  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  EXPECT_EQ(txn->GetReadVersion(), 1);
  EXPECT_EQ(txn->GetMetadata().collection().name, "test");

  ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
  ASSERT_AND_ASSIGN(auto segments, txn->ListSegments(*snap));

  ASSERT_EQ(segments.size(), 2u);

  int total_segments = 0;
  for (const auto& entry : segments) {
    total_segments += static_cast<int>(entry.segments.size());
  }
  EXPECT_EQ(total_segments, 3);
}

TEST_F(CollectionTransactionTest, MultipleSnapshots) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                                    .AddSegment(1, "_default", {.segment_id = 1002, .manifest = "_manifests/1002.avro"})
                                    .Build()));
    EXPECT_EQ(v, 1);
  }
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1003, .manifest = "_manifests/1003.avro"}).Build()));
    EXPECT_EQ(v, 2);
  }

  // Read latest (v2) - should have 3 segments
  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());
  ASSERT_AND_ASSIGN(auto segments, txn->ListSegments(*snap));
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0].segments.size(), 3u);

  // Read v1 - should have 2 segments
  ASSERT_AND_ASSIGN(auto txn_v1, CollectionTransaction::Open(fs_, base_path_, 1));
  ASSERT_AND_ASSIGN(auto snap_v1, txn_v1->GetCurrentSnapshot());
  ASSERT_AND_ASSIGN(auto segments_v1, txn_v1->ListSegments(*snap_v1));
  ASSERT_EQ(segments_v1.size(), 1u);
  EXPECT_EQ(segments_v1[0].segments.size(), 2u);
}

TEST_F(CollectionTransactionTest, ListByPartition) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_)
                                    .AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"})
                                    .AddSegment(2, "part_a", {.segment_id = 2001, .manifest = "_manifests/2001.avro"})
                                    .AddSegment(2, "part_a", {.segment_id = 2002, .manifest = "_manifests/2002.avro"})
                                    .AddSegment(3, "part_b", {.segment_id = 3001, .manifest = "_manifests/3001.avro"})
                                    .Build()));
    EXPECT_EQ(v, 1);
  }

  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  ASSERT_AND_ASSIGN(auto snap, txn->GetCurrentSnapshot());

  ASSERT_AND_ASSIGN(auto segs, txn->ListSegments(*snap, 2));
  ASSERT_EQ(segs.size(), 2u);
  EXPECT_EQ(segs[0].segment_id, 2001);
  EXPECT_EQ(segs[1].segment_id, 2002);

  ASSERT_AND_ASSIGN(auto segs1, txn->ListSegments(*snap, 1));
  ASSERT_EQ(segs1.size(), 1u);

  ASSERT_AND_ASSIGN(auto segs_empty, txn->ListSegments(*snap, 999));
  EXPECT_TRUE(segs_empty.empty());
}

TEST_F(CollectionTransactionTest, SchemaResolution) {
  {
    ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
    SetupCollection(*txn);
    ASSERT_AND_ASSIGN(auto v, txn->Commit(ActionBuilder::Create(fs_, base_path_).AddSegment(1, "_default", {.segment_id = 1001, .manifest = "_manifests/1001.avro"}).Build()));
  }

  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));

  ASSERT_AND_ASSIGN(auto schema0, txn->GetSchema(0));
  EXPECT_EQ(schema0->schema_id, 0);
  ASSERT_EQ(schema0->fields.size(), 2u);

  ASSERT_AND_ASSIGN(auto schema1, txn->GetSchema(1));
  EXPECT_EQ(schema1->schema_id, 1);
  ASSERT_EQ(schema1->fields.size(), 1u);

  auto bad = txn->GetSchema(99);
  EXPECT_FALSE(bad.ok());
}

TEST_F(CollectionTransactionTest, EmptyCollection) {
  ASSERT_AND_ASSIGN(auto txn, CollectionTransaction::Open(fs_, base_path_));
  EXPECT_EQ(txn->GetReadVersion(), 0);
}

}  // namespace milvus_storage::api::table_format
