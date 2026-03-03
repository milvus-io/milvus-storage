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
#include <cstdint>
#include <sstream>
#include <random>

#include <avro/Specific.hh>
#include <avro/Stream.hh>
#include <avro/Encoder.hh>
#include <avro/Decoder.hh>

#include "milvus-storage/manifest.h"
#include "milvus-storage/common/config.h"

#include "test_env.h"

using namespace milvus_storage::api;

class ColumnGroupsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Create test column groups
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = {"id", "name", "age"};
    // Initialize files using brace initialization (aggregates)
    // ColumnGroupFile has path, start_index, end_index.
    // Optional members default to nullopt.
    cg1->files = {{"/data/cg1_part1.parquet"}, {"/data/cg1_part2.parquet"}};
    cg1->format = LOON_FORMAT_PARQUET;

    auto cg2 = std::make_shared<ColumnGroup>();
    cg2->columns = {"embedding", "metadata"};
    cg2->files = {{"/data/cg2_vectors.vortex"}};
    cg2->format = LOON_FORMAT_VORTEX;

    ColumnGroups column_groups = {cg1, cg2};
    test_cgs_ = std::move(column_groups);
  }

  ColumnGroups test_cgs_;
};

TEST_F(ColumnGroupsTest, SerializeDeserialize) {
  // Create Manifest with test column groups
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>());

  // Serialize to Avro
  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  std::string avro_str = oss.str();

  EXPECT_FALSE(avro_str.empty());

  // Deserialize from Avro
  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(avro_str);
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));
  const auto& groups = deserialized_manifest->columnGroups();
  const auto& expected_groups = test_cgs_;

  EXPECT_EQ(groups.size(), expected_groups.size());

  for (size_t i = 0; i < groups.size(); ++i) {
    EXPECT_EQ(groups[i]->columns, expected_groups[i]->columns);
    EXPECT_EQ(groups[i]->format, expected_groups[i]->format);

    ASSERT_EQ(groups[i]->files.size(), expected_groups[i]->files.size());
    for (size_t j = 0; j < groups[i]->files.size(); ++j) {
      EXPECT_EQ(groups[i]->files[j].path, expected_groups[i]->files[j].path);
      EXPECT_EQ(groups[i]->files[j].start_index, expected_groups[i]->files[j].start_index);
      EXPECT_EQ(groups[i]->files[j].end_index, expected_groups[i]->files[j].end_index);
    }
  }
}

TEST_F(ColumnGroupsTest, EmptyColumnGroups) {
  // Test empty column groups
  ColumnGroups column_groups = {};
  auto manifest =
      std::make_shared<Manifest>(column_groups, std::vector<DeltaLog>(), std::map<std::string, Statistics>());

  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  std::string avro_str = oss.str();

  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(avro_str);
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  EXPECT_EQ(deserialized_manifest->columnGroups().size(), 0);
}

TEST_F(ColumnGroupsTest, ColumnLookup) {
  // Serialize and deserialize
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>());
  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  std::string avro_str = oss.str();

  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(avro_str);
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  // Test column lookup functionality
  const auto& expected_groups = test_cgs_;
  if (!expected_groups.empty() && !expected_groups[0]->columns.empty()) {
    std::string test_col = expected_groups[0]->columns[0];
    auto cg = deserialized_manifest->getColumnGroup(test_col);
    ASSERT_NE(cg, nullptr);
    EXPECT_EQ(cg->format, expected_groups[0]->format);
  }

  auto missing_cg = deserialized_manifest->getColumnGroup("nonexistent_column_name_xyz");
  EXPECT_EQ(missing_cg, nullptr);
}

TEST_F(ColumnGroupsTest, InvalidAvro) {
  auto deserialized_manifest = std::make_shared<Manifest>();

  // Empty stream should fail (too short to read header)
  {
    std::string empty_str = "";
    std::istringstream in1(empty_str);
    auto status = deserialized_manifest->deserialize(in1);
    EXPECT_FALSE(status.ok());
  }

  // Garbage data falls through to legacy path and fails
  {
    std::string garbage = "garbage_data_12345";
    std::istringstream in2(garbage);
    auto status = deserialized_manifest->deserialize(in2);
    EXPECT_FALSE(status.ok());
  }

  // Data starting with OCF magic but truncated should fail
  {
    std::string truncated_ocf = "Obj\x01";
    std::istringstream in3(truncated_ocf);
    auto status = deserialized_manifest->deserialize(in3);
    EXPECT_FALSE(status.ok());
  }
}

TEST_F(ColumnGroupsTest, TestPrivateData) {
  uint8_t private_data[] = {0x01, 0x02, 0x03, 0x04};
  auto pvec = std::vector<uint8_t>(private_data, private_data + sizeof(private_data));
  auto cg1 = std::make_shared<ColumnGroup>();
  cg1->columns = {"test_column"};
  cg1->files.emplace_back(ColumnGroupFile{
      .path = "test_path",
      .metadata = pvec,
  });
  cg1->format = LOON_FORMAT_PARQUET;

  ColumnGroups column_groups = {cg1};
  auto manifest = std::make_shared<Manifest>(std::move(column_groups), std::vector<DeltaLog>(),
                                             std::map<std::string, Statistics>());

  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  std::string avro_str = oss.str();

  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(avro_str);
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  auto deserialized_cg = deserialized_manifest->getColumnGroup("test_column");
  ASSERT_EQ(deserialized_cg->files[0].metadata,
            std::vector<uint8_t>(private_data, private_data + sizeof(private_data)));
}

// ==================== Index Serialization Tests ====================

TEST_F(ColumnGroupsTest, IndexSerializeDeserialize) {
  // Create indexes
  std::vector<Index> indexes;

  Index idx1;
  idx1.column_name = "embedding";
  idx1.index_type = "hnsw";
  idx1.path = "/data/_index/embedding_hnsw.idx";
  idx1.properties = {{"ef_construction", "128"}, {"M", "16"}};
  indexes.push_back(idx1);

  Index idx2;
  idx2.column_name = "id";
  idx2.index_type = "inverted";
  idx2.path = "/data/_index/id_inverted.idx";
  idx2.properties = {};
  indexes.push_back(idx2);

  // Create manifest with indexes
  auto manifest =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  // Serialize
  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  std::string avro_str = oss.str();
  EXPECT_FALSE(avro_str.empty());

  // Deserialize
  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(avro_str);
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  // Verify indexes
  const auto& deserialized_indexes = deserialized_manifest->indexes();
  ASSERT_EQ(deserialized_indexes.size(), 2);

  // Check first index
  const Index* found1 = deserialized_manifest->getIndex("embedding", "hnsw");
  ASSERT_NE(found1, nullptr);
  EXPECT_EQ(found1->column_name, "embedding");
  EXPECT_EQ(found1->index_type, "hnsw");
  EXPECT_EQ(found1->path, "/data/_index/embedding_hnsw.idx");
  EXPECT_EQ(found1->properties.size(), 2);
  EXPECT_EQ(found1->properties.at("ef_construction"), "128");
  EXPECT_EQ(found1->properties.at("M"), "16");

  // Check second index
  const Index* found2 = deserialized_manifest->getIndex("id", "inverted");
  ASSERT_NE(found2, nullptr);
  EXPECT_EQ(found2->column_name, "id");
  EXPECT_EQ(found2->index_type, "inverted");
  EXPECT_EQ(found2->path, "/data/_index/id_inverted.idx");
  EXPECT_TRUE(found2->properties.empty());
}

TEST_F(ColumnGroupsTest, IndexLookupNotFound) {
  // Create manifest with one index
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = "/data/_index/embedding_hnsw.idx";
  idx.properties = {};
  indexes.push_back(idx);

  auto manifest =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  // Serialize and deserialize
  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));
  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  // Test getIndex with non-existent keys
  EXPECT_EQ(deserialized_manifest->getIndex("nonexistent", "hnsw"), nullptr);
  EXPECT_EQ(deserialized_manifest->getIndex("embedding", "nonexistent"), nullptr);
  EXPECT_EQ(deserialized_manifest->getIndex("nonexistent", "nonexistent"), nullptr);

  // Test getIndex with correct key
  EXPECT_NE(deserialized_manifest->getIndex("embedding", "hnsw"), nullptr);
}

TEST_F(ColumnGroupsTest, EmptyIndexes) {
  // Create manifest without indexes (empty vector)
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(),
                                             std::vector<Index>());

  // Serialize
  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));

  // Deserialize
  auto deserialized_manifest = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized_manifest->deserialize(in));

  // Verify empty indexes
  EXPECT_TRUE(deserialized_manifest->indexes().empty());
}

// ==================== Stats & DeltaLog Serialization Tests ====================

TEST_F(ColumnGroupsTest, StatsRoundTrip) {
  std::map<std::string, Statistics> stats;
  Statistics stat1;
  stat1.paths = {"/stats/bloom_filter_100.bin", "/stats/bloom_filter_101.bin"};
  stat1.metadata = {{"type", "bloom_filter"}, {"num_bits", "1024"}};
  stats["bloom_filter.100"] = stat1;

  Statistics stat2;
  stat2.paths = {"/stats/bm25_200.bin"};
  stats["bm25.200"] = stat2;

  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), stats);

  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  const auto& ds = deserialized->stats();
  ASSERT_EQ(ds.size(), 2);

  ASSERT_EQ(ds.at("bloom_filter.100").paths.size(), 2);
  EXPECT_EQ(ds.at("bloom_filter.100").paths[0], "/stats/bloom_filter_100.bin");
  EXPECT_EQ(ds.at("bloom_filter.100").metadata.at("type"), "bloom_filter");
  EXPECT_EQ(ds.at("bloom_filter.100").metadata.at("num_bits"), "1024");

  ASSERT_EQ(ds.at("bm25.200").paths.size(), 1);
  EXPECT_TRUE(ds.at("bm25.200").metadata.empty());
}

TEST_F(ColumnGroupsTest, DeltaLogRoundTrip) {
  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back({"/delta/pk_delete_1.bin", DeltaLogType::PRIMARY_KEY, 100});
  delta_logs.push_back({"/delta/pos_delete_2.bin", DeltaLogType::POSITIONAL, 50});

  auto manifest = std::make_shared<Manifest>(test_cgs_, delta_logs, std::map<std::string, Statistics>());

  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  const auto& dl = deserialized->deltaLogs();
  ASSERT_EQ(dl.size(), 2);
  EXPECT_EQ(dl[0].path, "/delta/pk_delete_1.bin");
  EXPECT_EQ(dl[0].type, DeltaLogType::PRIMARY_KEY);
  EXPECT_EQ(dl[0].num_entries, 100);
  EXPECT_EQ(dl[1].path, "/delta/pos_delete_2.bin");
  EXPECT_EQ(dl[1].type, DeltaLogType::POSITIONAL);
  EXPECT_EQ(dl[1].num_entries, 50);
}

TEST_F(ColumnGroupsTest, AllFieldsRoundTrip) {
  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back({"/delta/del.bin", DeltaLogType::EQUALITY, 10});

  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {"/stats/s.bin"};
  stat.metadata = {{"k", "v"}};
  stats["key"] = stat;

  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = "/index/emb.idx";
  idx.properties = {{"M", "16"}};
  indexes.push_back(idx);

  auto manifest = std::make_shared<Manifest>(test_cgs_, delta_logs, stats, indexes);

  std::ostringstream oss;
  ASSERT_STATUS_OK(manifest->serialize(oss));

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_EQ(deserialized->deltaLogs().size(), 1);
  EXPECT_EQ(deserialized->deltaLogs()[0].type, DeltaLogType::EQUALITY);
  EXPECT_EQ(deserialized->stats().size(), 1);
  EXPECT_EQ(deserialized->stats().at("key").metadata.at("k"), "v");
  EXPECT_EQ(deserialized->indexes().size(), 1);
  EXPECT_EQ(deserialized->indexes()[0].properties.at("M"), "16");
}

// ==================== Legacy Format Deserialization Tests ====================

// Helper: produce a legacy MILV-format binary using raw Avro binary encoder.
// Encodes each field manually to match the old serialize() order, since
// codec_traits for custom types are not visible from this translation unit.
static void encodeColumnGroupFile(avro::Encoder& e, const ColumnGroupFile& file) {
  avro::encode(e, file.path);
  avro::encode(e, file.start_index);
  avro::encode(e, file.end_index);
  avro::encode(e, file.metadata);
}

static void encodeColumnGroup(avro::Encoder& e, const ColumnGroup& group) {
  avro::encode(e, group.columns);
  e.arrayStart();
  if (!group.files.empty()) {
    e.setItemCount(group.files.size());
    for (const auto& f : group.files) {
      e.startItem();
      encodeColumnGroupFile(e, f);
    }
  }
  e.arrayEnd();
  avro::encode(e, group.format);
}

static void encodeDeltaLog(avro::Encoder& e, const DeltaLog& dl) {
  avro::encode(e, dl.path);
  avro::encode(e, static_cast<int32_t>(dl.type));
  avro::encode(e, dl.num_entries);
}

static void encodeIndex(avro::Encoder& e, const Index& idx) {
  avro::encode(e, idx.column_name);
  avro::encode(e, idx.index_type);
  avro::encode(e, idx.path);
  avro::encode(e, idx.properties);
}

static void encodeStatistics(avro::Encoder& e, const Statistics& stat) {
  avro::encode(e, stat.paths);
  avro::encode(e, stat.metadata);
}

static std::string encodeLegacyManifest(int32_t version,
                                        const ColumnGroups& cgs,
                                        const std::vector<DeltaLog>& delta_logs,
                                        const std::map<std::string, Statistics>& stats,
                                        const std::vector<Index>& indexes) {
  std::ostringstream oss;
  auto avro_output = avro::ostreamOutputStream(oss);
  auto encoder = avro::binaryEncoder();
  encoder->init(*avro_output);

  constexpr int32_t MILV_MAGIC = 0x4D494C56;
  avro::encode(*encoder, MILV_MAGIC);
  avro::encode(*encoder, version);

  // Encode column groups
  encoder->arrayStart();
  if (!cgs.empty()) {
    encoder->setItemCount(cgs.size());
    for (const auto& cg : cgs) {
      encoder->startItem();
      encodeColumnGroup(*encoder, *cg);
    }
  }
  encoder->arrayEnd();

  // Encode delta logs
  encoder->arrayStart();
  if (!delta_logs.empty()) {
    encoder->setItemCount(delta_logs.size());
    for (const auto& dl : delta_logs) {
      encoder->startItem();
      encodeDeltaLog(*encoder, dl);
    }
  }
  encoder->arrayEnd();

  // Encode stats
  if (version >= 3) {
    encoder->mapStart();
    if (!stats.empty()) {
      encoder->setItemCount(stats.size());
      for (const auto& [key, stat] : stats) {
        encoder->startItem();
        avro::encode(*encoder, key);
        encodeStatistics(*encoder, stat);
      }
    }
    encoder->mapEnd();
  } else {
    // v1/v2: stats is map<string, vector<string>>
    std::map<std::string, std::vector<std::string>> legacy_stats;
    for (const auto& [key, stat] : stats) {
      legacy_stats[key] = stat.paths;
    }
    avro::encode(*encoder, legacy_stats);
  }

  // Encode indexes (v2+ only)
  if (version >= 2) {
    encoder->arrayStart();
    if (!indexes.empty()) {
      encoder->setItemCount(indexes.size());
      for (const auto& idx : indexes) {
        encoder->startItem();
        encodeIndex(*encoder, idx);
      }
    }
    encoder->arrayEnd();
  }

  encoder->flush();
  return oss.str();
}

TEST_F(ColumnGroupsTest, LegacyV3Deserialize) {
  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {"/stats/s.bin"};
  stat.metadata = {{"k", "v"}};
  stats["key"] = stat;

  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "col";
  idx.index_type = "hnsw";
  idx.path = "/index/i.idx";
  idx.properties = {{"M", "8"}};
  indexes.push_back(idx);

  std::string legacy_bytes = encodeLegacyManifest(3, test_cgs_, {}, stats, indexes);

  // Verify it does NOT start with OCF magic
  ASSERT_NE(legacy_bytes.substr(0, 4), std::string("Obj\x01", 4));

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(legacy_bytes);
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_EQ(deserialized->stats().at("key").paths[0], "/stats/s.bin");
  EXPECT_EQ(deserialized->stats().at("key").metadata.at("k"), "v");
  EXPECT_EQ(deserialized->indexes().size(), 1);
  EXPECT_EQ(deserialized->indexes()[0].properties.at("M"), "8");
}

TEST_F(ColumnGroupsTest, LegacyV1Deserialize) {
  // v1: no indexes, stats as map<string, vector<string>>
  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {"/stats/old.bin"};
  stats["old_key"] = stat;

  std::string legacy_bytes = encodeLegacyManifest(1, test_cgs_, {}, stats, {});

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(legacy_bytes);
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_EQ(deserialized->stats().at("old_key").paths[0], "/stats/old.bin");
  EXPECT_TRUE(deserialized->stats().at("old_key").metadata.empty());
  EXPECT_TRUE(deserialized->indexes().empty());
}

TEST_F(ColumnGroupsTest, LegacyV2Deserialize) {
  // v2: has indexes, stats as map<string, vector<string>>
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "col";
  idx.index_type = "ivf";
  idx.path = "/index/v2.idx";
  idx.properties = {};
  indexes.push_back(idx);

  std::string legacy_bytes = encodeLegacyManifest(2, test_cgs_, {}, {}, indexes);

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(legacy_bytes);
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_TRUE(deserialized->stats().empty());
  EXPECT_EQ(deserialized->indexes().size(), 1);
  EXPECT_EQ(deserialized->indexes()[0].index_type, "ivf");
}

TEST_F(ColumnGroupsTest, IndexRoundTripPreservesData) {
  // Create manifest with indexes
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = "/data/_index/embedding_hnsw.idx";
  idx.properties = {{"key", "value"}};
  indexes.push_back(idx);

  auto original =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  // Serialize and deserialize to test data preservation
  std::ostringstream oss;
  ASSERT_STATUS_OK(original->serialize(oss));

  auto deserialized = std::make_shared<Manifest>();
  std::istringstream in(oss.str());
  ASSERT_STATUS_OK(deserialized->deserialize(in));

  // Verify deserialized has same indexes
  ASSERT_EQ(deserialized->indexes().size(), 1);
  const Index* found = deserialized->getIndex("embedding", "hnsw");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->properties.at("key"), "value");

  // Modify original indexes, deserialized should be independent
  original->indexes().clear();
  EXPECT_EQ(original->indexes().size(), 0);
  EXPECT_EQ(deserialized->indexes().size(), 1);  // Deserialized unchanged
}
