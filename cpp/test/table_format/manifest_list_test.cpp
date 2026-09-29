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

#include "milvus-storage/table_format/manifest_list.h"
#include "test_env.h"

namespace milvus_storage::api::table_format {

TEST(ManifestListTest, WriteReadRoundtrip) {
  std::vector<ManifestListEntry> entries;

  // Partition 1: 2 segments
  ManifestListEntry p1;
  p1.partition_id = 1;
  p1.partition_name = "_default";
  p1.segments = {
      {.segment_id = 1001,
       .manifest = "_manifests/1001.avro",
       .level = SegmentLevel::L1,
       .num_rows = 100000,
       .file_size = 512 * 1024 * 1024},
      {.segment_id = 1002,
       .manifest = "_manifests/1002.avro",
       .level = SegmentLevel::L2,
       .num_rows = 200000,
       .file_size = 1024 * 1024 * 1024,
       .index_size = 128 * 1024 * 1024,
       .sorted = true},
  };
  entries.push_back(p1);

  // Partition 2: 3 segments
  ManifestListEntry p2;
  p2.partition_id = 2;
  p2.partition_name = "part_a";
  p2.segments = {
      {.segment_id = 2001, .manifest = "_manifests/2001.avro", .level = SegmentLevel::L1, .num_rows = 50000},
      {.segment_id = 2002, .manifest = "_manifests/2002.avro", .level = SegmentLevel::L1, .num_rows = 60000},
      {.segment_id = 2003,
       .manifest = "_manifests/2003.avro",
       .level = SegmentLevel::L2,
       .num_rows = 110000,
       .partition_key_sorted = true},
  };
  entries.push_back(p2);

  // Partition 3: 2 segments
  ManifestListEntry p3;
  p3.partition_id = 3;
  p3.partition_name = "part_b";
  p3.segments = {
      {.segment_id = 3001, .manifest = "_manifests/3001.avro"},
      {.segment_id = 3002, .manifest = "_manifests/3002.avro"},
  };
  entries.push_back(p3);

  ManifestList ml(std::move(entries));

  // Serialize
  std::stringstream ss;
  ASSERT_STATUS_OK(ml.serialize(ss));

  // Deserialize
  ManifestList restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  // Verify
  ASSERT_EQ(restored.entries().size(), 3u);

  EXPECT_EQ(restored.entries()[0].partition_id, 1);
  ASSERT_EQ(restored.entries()[0].segments.size(), 2u);
  EXPECT_EQ(restored.entries()[0].segments[0].segment_id, 1001);
  EXPECT_EQ(restored.entries()[0].segments[0].level, SegmentLevel::L1);
  EXPECT_EQ(restored.entries()[0].segments[0].num_rows, 100000);
  EXPECT_EQ(restored.entries()[0].segments[1].level, SegmentLevel::L2);
  EXPECT_TRUE(restored.entries()[0].segments[1].sorted);

  EXPECT_EQ(restored.entries()[1].partition_id, 2);
  ASSERT_EQ(restored.entries()[1].segments.size(), 3u);
  EXPECT_TRUE(restored.entries()[1].segments[2].partition_key_sorted);

  EXPECT_EQ(restored.entries()[2].partition_id, 3);
  ASSERT_EQ(restored.entries()[2].segments.size(), 2u);
}

TEST(ManifestListTest, EmptyManifestList) {
  ManifestList ml;

  std::stringstream ss;
  ASSERT_STATUS_OK(ml.serialize(ss));

  ManifestList restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  EXPECT_TRUE(restored.entries().empty());
}

TEST(ManifestListTest, SegmentInfoFields) {
  SegmentInfo seg;
  seg.segment_id = 42;
  seg.manifest = "_manifests/0042.avro";
  seg.level = SegmentLevel::L2;
  seg.num_rows = 999999;
  seg.file_size = 2048;
  seg.index_size = 1024;
  seg.sorted = true;
  seg.partition_key_sorted = true;

  ManifestListEntry entry;
  entry.partition_id = 10;
  entry.partition_name = "test_part";
  entry.segments = {seg};

  ManifestList ml({entry});

  std::stringstream ss;
  ASSERT_STATUS_OK(ml.serialize(ss));

  ManifestList restored;
  ASSERT_STATUS_OK(restored.deserialize(ss));

  ASSERT_EQ(restored.entries().size(), 1u);
  auto& rs = restored.entries()[0].segments[0];
  EXPECT_EQ(rs.segment_id, 42);
  EXPECT_EQ(rs.manifest, "_manifests/0042.avro");
  EXPECT_EQ(rs.level, SegmentLevel::L2);
  EXPECT_EQ(rs.num_rows, 999999);
  EXPECT_EQ(rs.file_size, 2048);
  EXPECT_EQ(rs.index_size, 1024);
  EXPECT_TRUE(rs.sorted);
  EXPECT_TRUE(rs.partition_key_sorted);
}

TEST(ManifestListTest, MoveSemantics) {
  ManifestListEntry entry;
  entry.partition_id = 1;
  entry.partition_name = "_default";
  entry.segments = {{.segment_id = 100, .manifest = "_manifests/0100.avro"}};

  ManifestList ml1({entry});
  ASSERT_EQ(ml1.entries().size(), 1u);

  // Move construct
  ManifestList ml2(std::move(ml1));
  ASSERT_EQ(ml2.entries().size(), 1u);
  EXPECT_EQ(ml2.entries()[0].segments[0].segment_id, 100);

  // Move assign
  ManifestList ml3;
  ml3 = std::move(ml2);
  ASSERT_EQ(ml3.entries().size(), 1u);
  EXPECT_EQ(ml3.entries()[0].partition_id, 1);
}

}  // namespace milvus_storage::api::table_format
