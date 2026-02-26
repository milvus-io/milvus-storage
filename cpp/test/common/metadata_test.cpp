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

#include <gtest/gtest.h>

#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/constants.h"

namespace milvus_storage::test {

class MetadataTest : public testing::Test {};

TEST_F(MetadataTest, TestGroupFieldIDListSerde) {
  GroupFieldIDList list({{0, 1, 2}, {3, 4}, {5, 6, 7, 8}});
  std::string serialized = list.Serialize();
  EXPECT_EQ(serialized, "0,1,2;3,4;5,6,7,8");
  GroupFieldIDList deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, list);

  // Test case: Empty input
  GroupFieldIDList empty_list = {};
  serialized = empty_list.Serialize();
  EXPECT_EQ(serialized, "");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_TRUE(deserialized.empty());

  // Test case: Single group
  GroupFieldIDList single_group({{1, 2, 3}});
  serialized = single_group.Serialize();
  EXPECT_EQ(serialized, "1,2,3");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, single_group);

  // Test case: Single column in each group
  GroupFieldIDList single_column_groups({{0}, {1}, {2}});
  serialized = single_column_groups.Serialize();
  EXPECT_EQ(serialized, "0;1;2");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, single_column_groups);
}

TEST_F(MetadataTest, TestFieldIDList) {
  // Basic operations
  {
    FieldIDList list;
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0);

    list.Add(10);
    list.Add(20);
    list.Add(30);
    EXPECT_FALSE(list.empty());
    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(list.Get(0), 10);
    EXPECT_EQ(list.Get(1), 20);
    EXPECT_EQ(list.Get(2), 30);
  }

  // GetOutOfRange
  {
    FieldIDList list;
    list.Add(1);
    EXPECT_THROW(list.Get(5), std::out_of_range);
  }

  // Equality
  {
    FieldIDList a;
    a.Add(1);
    a.Add(2);

    FieldIDList b;
    b.Add(1);
    b.Add(2);
    EXPECT_TRUE(a == b);

    FieldIDList c;
    c.Add(1);
    c.Add(3);
    EXPECT_FALSE(a == c);
  }

  // ToString
  {
    FieldIDList list;
    list.Add(10);
    list.Add(20);
    list.Add(30);
    EXPECT_EQ(list.ToString(), "10,20,30");

    FieldIDList empty;
    EXPECT_EQ(empty.ToString(), "");
  }

  // Make from schema — valid
  {
    auto field0 = arrow::field("col0", arrow::int32(), arrow::KeyValueMetadata::Make({ARROW_FIELD_ID_KEY}, {"100"}));
    auto field1 = arrow::field("col1", arrow::utf8(), arrow::KeyValueMetadata::Make({ARROW_FIELD_ID_KEY}, {"200"}));
    auto schema = arrow::schema({field0, field1});

    auto result = FieldIDList::Make(schema);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 2);
    EXPECT_EQ(result->Get(0), 100);
    EXPECT_EQ(result->Get(1), 200);
  }

  // Make from schema — missing key
  {
    auto field0 = arrow::field("col0", arrow::int32());
    auto result = FieldIDList::Make(arrow::schema({field0}));
    EXPECT_FALSE(result.ok());
  }

  // Make from schema — invalid field id
  {
    auto field0 =
        arrow::field("col0", arrow::int32(), arrow::KeyValueMetadata::Make({ARROW_FIELD_ID_KEY}, {"not_a_number"}));
    auto result = FieldIDList::Make(arrow::schema({field0}));
    EXPECT_FALSE(result.ok());
  }
}

TEST_F(MetadataTest, TestGroupFieldIDList) {
  // Make
  {
    FieldIDList field_ids;
    field_ids.Add(100);
    field_ids.Add(200);
    field_ids.Add(300);

    auto group = GroupFieldIDList::Make({{0, 1}, {2}}, field_ids);
    EXPECT_EQ(group.num_groups(), 2);

    auto g0 = group.GetFieldIDList(0);
    EXPECT_EQ(g0.size(), 2);
    EXPECT_EQ(g0.Get(0), 100);
    EXPECT_EQ(g0.Get(1), 200);

    auto g1 = group.GetFieldIDList(1);
    EXPECT_EQ(g1.size(), 1);
    EXPECT_EQ(g1.Get(0), 300);
  }

  // GetOutOfRange
  {
    GroupFieldIDList list({{0, 1}});
    EXPECT_THROW(list.GetFieldIDList(5), std::out_of_range);
  }

  // Equality
  {
    GroupFieldIDList a({{1, 2}, {3}});
    GroupFieldIDList b({{1, 2}, {3}});
    EXPECT_TRUE(a == b);

    GroupFieldIDList c({{1, 2}, {4}});
    EXPECT_FALSE(a == c);
  }
}

TEST_F(MetadataTest, TestRowGroupMetadata) {
  // Basic
  {
    RowGroupMetadata meta(1024, 100, 0);
    EXPECT_EQ(meta.memory_size(), 1024);
    EXPECT_EQ(meta.row_num(), 100);
    EXPECT_EQ(meta.row_offset(), 0);
  }

  // ToString
  {
    RowGroupMetadata meta(1024, 100, 50);
    auto str = meta.ToString();
    EXPECT_NE(str.find("memory_size=1024"), std::string::npos);
    EXPECT_NE(str.find("row_num=100"), std::string::npos);
    EXPECT_NE(str.find("row_offset=50"), std::string::npos);
  }

  // Serde
  {
    RowGroupMetadata meta(2048, 200, 100);
    EXPECT_EQ(meta.Serialize(), "2048|200|100");

    auto deserialized = RowGroupMetadata::Deserialize("2048|200|100");
    EXPECT_EQ(deserialized.memory_size(), 2048);
    EXPECT_EQ(deserialized.row_num(), 200);
    EXPECT_EQ(deserialized.row_offset(), 100);
  }

  // Invalid deserialization
  { EXPECT_THROW(RowGroupMetadata::Deserialize("invalid"), std::runtime_error); }
}

TEST_F(MetadataTest, TestRowGroupMetadataVector) {
  // Basic
  {
    RowGroupMetadataVector vec;
    vec.Add(RowGroupMetadata(1024, 100, 0));
    vec.Add(RowGroupMetadata(2048, 200, 100));

    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec.row_num(), 300);
    EXPECT_EQ(vec.memory_size(), 3072);
    EXPECT_EQ(vec.Get(0).row_num(), 100);
    EXPECT_EQ(vec.Get(1).row_num(), 200);
  }

  // GetOutOfRange
  {
    RowGroupMetadataVector vec;
    EXPECT_THROW(vec.Get(0), std::out_of_range);
  }

  // Clear
  {
    RowGroupMetadataVector vec;
    vec.Add(RowGroupMetadata(1024, 100, 0));
    EXPECT_EQ(vec.size(), 1);
    vec.clear();
    EXPECT_EQ(vec.size(), 0);
  }

  // Serde
  {
    RowGroupMetadataVector vec;
    vec.Add(RowGroupMetadata(1024, 100, 0));
    vec.Add(RowGroupMetadata(2048, 200, 100));
    EXPECT_EQ(vec.Serialize(), "1024|100|0;2048|200|100");

    auto deserialized = RowGroupMetadataVector::Deserialize("1024|100|0;2048|200|100");
    EXPECT_EQ(deserialized.size(), 2);
    EXPECT_EQ(deserialized.Get(0).memory_size(), 1024);
    EXPECT_EQ(deserialized.Get(1).row_num(), 200);
  }

  // ToString
  {
    RowGroupMetadataVector vec;
    vec.Add(RowGroupMetadata(100, 10, 0));
    EXPECT_NE(vec.ToString().find("memory_size=100"), std::string::npos);
  }

  // Empty serde
  {
    RowGroupMetadataVector vec;
    EXPECT_EQ(vec.Serialize(), "");

    auto deserialized = RowGroupMetadataVector::Deserialize("");
    EXPECT_EQ(deserialized.size(), 0);
  }
}

}  // namespace milvus_storage::test
