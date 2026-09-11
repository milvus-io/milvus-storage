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
#include "milvus-storage/table_format/types.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

TEST(TypesTest, MetadataRoundtrip) {
  Metadata md;
  md.set_format_version(1);
  md.mutable_collection() = {
      .collection_id = 100,
      .name = "test_collection",
      .db_id = 1,
      .created_at = 1700000000000,
      .properties = {{"key1", "value1"}, {"key2", "value2"}},
  };

  // Schema 0
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
  f2.description = "vector field";

  FieldSchema f3;
  f3.field_id = 3;
  f3.name = "text";
  f3.data_type = DataType::VarChar;
  f3.nullable = true;
  f3.default_value = "hello";
  f3.is_partition_key = true;

  SchemaInfo schema0;
  schema0.schema_id = 0;
  schema0.fields = {f1, f2, f3};

  FunctionSchema func1;
  func1.function_id = 1;
  func1.name = "bm25";
  func1.description = "BM25 scoring function";
  func1.type = "BM25";
  func1.input_field_names = {"text"};
  func1.input_field_ids = {3};
  func1.output_field_names = {"score"};
  func1.output_field_ids = {4};
  func1.params = {{"k1", "1.2"}, {"b", "0.75"}};
  schema0.functions = {func1};

  // Schema 1
  SchemaInfo schema1;
  schema1.schema_id = 1;
  schema1.fields = {f1, f2};

  md.mutable_schemas() = {schema0, schema1};
  md.set_current_schema_id(0);

  // Index spec
  IndexInfo idx1;
  idx1.index_id = 1;
  idx1.index_name = "vec_index";
  idx1.field_id = 2;
  idx1.index_params = {{"index_type", "HNSW"}, {"M", "16"}};
  idx1.type_params = {{"dim", "128"}};
  idx1.auto_index = true;
  idx1.user_index_params = std::map<std::string, std::string>{{"nlist", "100"}};
  idx1.created_at = 1700000002000;

  IndexSpec ispec;
  ispec.spec_id = 0;
  ispec.indexes = {idx1};
  md.mutable_index_specs() = {ispec};
  md.set_current_index_spec_id(0);

  // Snapshots
  SnapshotEntry snap1;
  snap1.snapshot_id = 1;
  snap1.timestamp_ms = 1700000003000;
  snap1.schema_id = 0;
  snap1.index_spec_id = 0;
  snap1.manifest_lists = {{.manifest_list = "_manifests/ml1.avro"}};

  SnapshotEntry snap2;
  snap2.snapshot_id = 2;
  snap2.parent_snapshot_id = 1;
  snap2.timestamp_ms = 1700000004000;
  snap2.schema_id = 0;
  snap2.index_spec_id = 0;
  snap2.manifest_lists = {
      {.manifest_list = "_manifests/ml2.avro",
       .partition_ids = {1, 2},
       .partition_names = {"_default", "part_a"}},
  };

  SnapshotEntry snap3;
  snap3.snapshot_id = 3;
  snap3.parent_snapshot_id = 2;
  snap3.timestamp_ms = 1700000005000;
  snap3.schema_id = 1;
  snap3.index_spec_id = 0;
  snap3.manifest_lists = {{.manifest_list = "_manifests/ml3.avro"}};

  md.mutable_snapshots() = {snap1, snap2, snap3};
  md.set_current_snapshot_id(3);

  // Serialize
  std::stringstream ss;
  ASSERT_STATUS_OK(md.serialize(ss));

  // Deserialize
  Metadata restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  // Verify
  EXPECT_EQ(restored.format_version(), 1);
  EXPECT_EQ(restored.collection().collection_id, 100);
  EXPECT_EQ(restored.collection().name, "test_collection");
  EXPECT_EQ(restored.collection().db_id, 1);
  EXPECT_EQ(restored.collection().created_at, 1700000000000);
  EXPECT_EQ(restored.collection().properties.size(), 2u);
  EXPECT_EQ(restored.collection().properties.at("key1"), "value1");

  ASSERT_EQ(restored.schemas().size(), 2u);
  EXPECT_EQ(restored.schemas()[0].schema_id, 0);
  ASSERT_EQ(restored.schemas()[0].fields.size(), 3u);
  EXPECT_EQ(restored.schemas()[0].fields[0].name, "pk");
  EXPECT_TRUE(restored.schemas()[0].fields[0].is_primary_key);
  EXPECT_EQ(restored.schemas()[0].fields[1].type_params.at("dim"), "128");
  EXPECT_EQ(restored.schemas()[0].fields[1].description.value(), "vector field");
  EXPECT_TRUE(restored.schemas()[0].fields[2].nullable);
  EXPECT_EQ(restored.schemas()[0].fields[2].default_value.value(), "hello");
  EXPECT_TRUE(restored.schemas()[0].fields[2].is_partition_key);
  EXPECT_FALSE(restored.schemas()[0].fields[0].element_type.has_value());
  EXPECT_FALSE(restored.schemas()[0].fields[0].external_field.has_value());

  ASSERT_EQ(restored.schemas()[0].functions.size(), 1u);
  EXPECT_EQ(restored.schemas()[0].functions[0].function_id, 1);
  EXPECT_EQ(restored.schemas()[0].functions[0].name, "bm25");
  EXPECT_EQ(restored.schemas()[0].functions[0].description.value(), "BM25 scoring function");
  EXPECT_EQ(restored.schemas()[0].functions[0].params.at("k1"), "1.2");

  EXPECT_EQ(restored.current_schema_id(), 0);

  ASSERT_EQ(restored.index_specs().size(), 1u);
  ASSERT_EQ(restored.index_specs()[0].indexes.size(), 1u);
  EXPECT_EQ(restored.index_specs()[0].indexes[0].index_name, "vec_index");
  EXPECT_TRUE(restored.index_specs()[0].indexes[0].auto_index);
  ASSERT_TRUE(restored.index_specs()[0].indexes[0].user_index_params.has_value());
  EXPECT_EQ(restored.index_specs()[0].indexes[0].user_index_params->at("nlist"), "100");

  ASSERT_EQ(restored.snapshots().size(), 3u);
  EXPECT_EQ(restored.snapshots()[0].snapshot_id, 1);
  EXPECT_FALSE(restored.snapshots()[0].parent_snapshot_id.has_value());
  EXPECT_EQ(restored.snapshots()[1].parent_snapshot_id.value(), 1);

  // Check partition info
  ASSERT_EQ(restored.snapshots()[1].manifest_lists[0].partition_ids.size(), 2u);
  EXPECT_EQ(restored.snapshots()[1].manifest_lists[0].partition_ids[0], 1);
  EXPECT_EQ(restored.snapshots()[1].manifest_lists[0].partition_ids[1], 2);
  EXPECT_EQ(restored.snapshots()[1].manifest_lists[0].partition_names[0], "_default");
  EXPECT_EQ(restored.snapshots()[1].manifest_lists[0].partition_names[1], "part_a");

  EXPECT_EQ(restored.current_snapshot_id(), 3);
}

TEST(TypesTest, FieldSchemaDefaults) {
  Metadata md;
  md.mutable_collection() = {.collection_id = 1, .name = "minimal"};

  FieldSchema f;
  f.field_id = 1;
  f.name = "pk";
  f.data_type = DataType::Int64;

  SchemaInfo schema;
  schema.schema_id = 0;
  schema.fields = {f};
  md.mutable_schemas() = {schema};

  std::stringstream ss;
  ASSERT_STATUS_OK(md.serialize(ss));

  Metadata restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  auto& rf = restored.schemas()[0].fields[0];
  EXPECT_FALSE(rf.is_primary_key);
  EXPECT_FALSE(rf.is_partition_key);
  EXPECT_FALSE(rf.is_clustering_key);
  EXPECT_FALSE(rf.nullable);
  EXPECT_FALSE(rf.is_dynamic);
  EXPECT_FALSE(rf.is_function_output);
  EXPECT_FALSE(rf.element_type.has_value());
  EXPECT_FALSE(rf.default_value.has_value());
  EXPECT_FALSE(rf.description.has_value());
  EXPECT_FALSE(rf.external_field.has_value());
}

TEST(TypesTest, ManifestListInfoEmptyPartitions) {
  Metadata md;
  md.mutable_collection() = {.collection_id = 1, .name = "test"};
  md.mutable_schemas() = {{.schema_id = 0}};

  SnapshotEntry snap;
  snap.snapshot_id = 1;
  snap.manifest_lists = {{.manifest_list = "_manifests/ml0.avro"}};
  md.mutable_snapshots() = {snap};
  md.set_current_snapshot_id(1);

  std::stringstream ss;
  ASSERT_STATUS_OK(md.serialize(ss));

  Metadata restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  ASSERT_EQ(restored.snapshots()[0].manifest_lists.size(), 1u);
  EXPECT_TRUE(restored.snapshots()[0].manifest_lists[0].partition_ids.empty());
  EXPECT_TRUE(restored.snapshots()[0].manifest_lists[0].partition_names.empty());
}

TEST(TypesTest, IndexInfoNullUserParams) {
  Metadata md;
  md.mutable_collection() = {.collection_id = 1, .name = "test"};
  md.mutable_schemas() = {{.schema_id = 0}};

  IndexInfo idx;
  idx.index_id = 1;
  idx.index_name = "test_idx";
  idx.field_id = 2;
  // user_index_params is nullopt by default

  IndexSpec ispec;
  ispec.spec_id = 0;
  ispec.indexes = {idx};
  md.mutable_index_specs() = {ispec};

  std::stringstream ss;
  ASSERT_STATUS_OK(md.serialize(ss));

  Metadata restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  ASSERT_EQ(restored.index_specs().size(), 1u);
  EXPECT_FALSE(restored.index_specs()[0].indexes[0].user_index_params.has_value());
}

}  // namespace milvus_storage::api::table_format
