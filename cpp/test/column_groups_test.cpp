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
#include "milvus-storage/common/layout.h"

#include "test_env.h"

using namespace milvus_storage;
using namespace milvus_storage::api;

class ColumnGroupsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    Manifest::CleanCache();
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("column-groups-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    // Create test column groups
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = {"id", "name", "age"};
    cg1->files = {{base_path_ + "/_data/cg1_part1.parquet"}, {base_path_ + "/_data/cg1_part2.parquet"}};
    cg1->format = LOON_FORMAT_PARQUET;

    auto cg2 = std::make_shared<ColumnGroup>();
    cg2->columns = {"embedding", "metadata"};
    cg2->files = {{base_path_ + "/_data/cg2_vectors.vortex"}};
    cg2->format = LOON_FORMAT_VORTEX;

    ColumnGroups column_groups = {cg1, cg2};
    test_cgs_ = std::move(column_groups);
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  // Helper: write manifest to file and read back
  arrow::Result<std::shared_ptr<Manifest>> WriteAndReadBack(const Manifest& manifest, size_t version = 1) {
    std::string path = milvus_storage::get_manifest_filepath(base_path_, version);
    ARROW_RETURN_NOT_OK(Manifest::WriteTo(fs_, path, manifest));
    Manifest::CleanCache();
    return Manifest::ReadFrom(fs_, path);
  }

  ColumnGroups test_cgs_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  Properties properties_;
};

TEST_F(ColumnGroupsTest, SerializeDeserialize) {
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));
  const auto& groups = deserialized->columnGroups();
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
  ColumnGroups column_groups = {};
  auto manifest =
      std::make_shared<Manifest>(column_groups, std::vector<DeltaLog>(), std::map<std::string, Statistics>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));
  EXPECT_EQ(deserialized->columnGroups().size(), 0);
}

TEST_F(ColumnGroupsTest, ColumnLookup) {
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  const auto& expected_groups = test_cgs_;
  if (!expected_groups.empty() && !expected_groups[0]->columns.empty()) {
    std::string test_col = expected_groups[0]->columns[0];
    auto cg = deserialized->getColumnGroup(test_col);
    ASSERT_NE(cg, nullptr);
    EXPECT_EQ(cg->format, expected_groups[0]->format);
  }

  auto missing_cg = deserialized->getColumnGroup("nonexistent_column_name_xyz");
  EXPECT_EQ(missing_cg, nullptr);
}

TEST_F(ColumnGroupsTest, TestProperties) {
  auto cg1 = std::make_shared<ColumnGroup>();
  cg1->columns = {"test_column"};
  cg1->files.emplace_back(ColumnGroupFile{
      .path = base_path_ + "/_data/test_path",
      .properties = {{"key1", "val1"}, {"key2", "val2"}},
  });
  cg1->format = LOON_FORMAT_PARQUET;

  ColumnGroups column_groups = {cg1};
  auto manifest = std::make_shared<Manifest>(std::move(column_groups), std::vector<DeltaLog>(),
                                             std::map<std::string, Statistics>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  auto deserialized_cg = deserialized->getColumnGroup("test_column");
  ASSERT_EQ(deserialized_cg->files[0].properties.at("key1"), "val1");
  ASSERT_EQ(deserialized_cg->files[0].properties.at("key2"), "val2");
}

// ==================== Index Serialization Tests ====================

TEST_F(ColumnGroupsTest, IndexSerializeDeserialize) {
  std::vector<Index> indexes;

  Index idx1;
  idx1.column_name = "embedding";
  idx1.index_type = "hnsw";
  idx1.path = base_path_ + "/_index/embedding_hnsw.idx";
  idx1.properties = {{"ef_construction", "128"}, {"M", "16"}};
  indexes.push_back(idx1);

  Index idx2;
  idx2.column_name = "id";
  idx2.index_type = "inverted";
  idx2.path = base_path_ + "/_index/id_inverted.idx";
  idx2.properties = {};
  indexes.push_back(idx2);

  auto manifest =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  const auto& deserialized_indexes = deserialized->indexes();
  ASSERT_EQ(deserialized_indexes.size(), 2);

  const Index* found1 = deserialized->getIndex("embedding", "hnsw");
  ASSERT_NE(found1, nullptr);
  EXPECT_EQ(found1->column_name, "embedding");
  EXPECT_EQ(found1->index_type, "hnsw");
  EXPECT_EQ(found1->path, base_path_ + "/_index/embedding_hnsw.idx");
  EXPECT_EQ(found1->properties.size(), 2);
  EXPECT_EQ(found1->properties.at("ef_construction"), "128");
  EXPECT_EQ(found1->properties.at("M"), "16");

  const Index* found2 = deserialized->getIndex("id", "inverted");
  ASSERT_NE(found2, nullptr);
  EXPECT_EQ(found2->column_name, "id");
  EXPECT_EQ(found2->index_type, "inverted");
  EXPECT_EQ(found2->path, base_path_ + "/_index/id_inverted.idx");
  EXPECT_TRUE(found2->properties.empty());
}

TEST_F(ColumnGroupsTest, IndexLookupNotFound) {
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = base_path_ + "/_index/embedding_hnsw.idx";
  idx.properties = {};
  indexes.push_back(idx);

  auto manifest =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  EXPECT_EQ(deserialized->getIndex("nonexistent", "hnsw"), nullptr);
  EXPECT_EQ(deserialized->getIndex("embedding", "nonexistent"), nullptr);
  EXPECT_EQ(deserialized->getIndex("nonexistent", "nonexistent"), nullptr);
  EXPECT_NE(deserialized->getIndex("embedding", "hnsw"), nullptr);
}

TEST_F(ColumnGroupsTest, EmptyIndexes) {
  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(),
                                             std::vector<Index>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));
  EXPECT_TRUE(deserialized->indexes().empty());
}

// ==================== Stats & DeltaLog Serialization Tests ====================

TEST_F(ColumnGroupsTest, StatsRoundTrip) {
  std::map<std::string, Statistics> stats;
  Statistics stat1;
  stat1.paths = {base_path_ + "/_stats/bloom_filter_100.bin", base_path_ + "/_stats/bloom_filter_101.bin"};
  stat1.metadata = {{"type", "bloom_filter"}, {"num_bits", "1024"}};
  stats["bloom_filter.100"] = stat1;

  Statistics stat2;
  stat2.paths = {base_path_ + "/_stats/bm25_200.bin"};
  stats["bm25.200"] = stat2;

  auto manifest = std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), stats);

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  const auto& ds = deserialized->stats();
  ASSERT_EQ(ds.size(), 2);

  ASSERT_EQ(ds.at("bloom_filter.100").paths.size(), 2);
  EXPECT_EQ(ds.at("bloom_filter.100").paths[0], base_path_ + "/_stats/bloom_filter_100.bin");
  EXPECT_EQ(ds.at("bloom_filter.100").metadata.at("type"), "bloom_filter");
  EXPECT_EQ(ds.at("bloom_filter.100").metadata.at("num_bits"), "1024");

  ASSERT_EQ(ds.at("bm25.200").paths.size(), 1);
  EXPECT_TRUE(ds.at("bm25.200").metadata.empty());
}

TEST_F(ColumnGroupsTest, DeltaLogRoundTrip) {
  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back({base_path_ + "/_delta/pk_delete_1.bin", DeltaLogType::PRIMARY_KEY, 100});
  delta_logs.push_back({base_path_ + "/_delta/pos_delete_2.bin", DeltaLogType::POSITIONAL, 50});

  auto manifest = std::make_shared<Manifest>(test_cgs_, delta_logs, std::map<std::string, Statistics>());

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

  const auto& dl = deserialized->deltaLogs();
  ASSERT_EQ(dl.size(), 2);
  EXPECT_EQ(dl[0].path, base_path_ + "/_delta/pk_delete_1.bin");
  EXPECT_EQ(dl[0].type, DeltaLogType::PRIMARY_KEY);
  EXPECT_EQ(dl[0].num_entries, 100);
  EXPECT_EQ(dl[1].path, base_path_ + "/_delta/pos_delete_2.bin");
  EXPECT_EQ(dl[1].type, DeltaLogType::POSITIONAL);
  EXPECT_EQ(dl[1].num_entries, 50);
}

TEST_F(ColumnGroupsTest, AllFieldsRoundTrip) {
  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back({base_path_ + "/_delta/del.bin", DeltaLogType::EQUALITY, 10});

  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {base_path_ + "/_stats/s.bin"};
  stat.metadata = {{"k", "v"}};
  stats["key"] = stat;

  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = base_path_ + "/_index/emb.idx";
  idx.properties = {{"M", "16"}};
  indexes.push_back(idx);

  auto manifest = std::make_shared<Manifest>(test_cgs_, delta_logs, stats, indexes);

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*manifest));

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
static void encodeColumnGroupFile(avro::Encoder& e, const ColumnGroupFile& file) {
  avro::encode(e, file.path);
  avro::encode(e, file.start_index);
  avro::encode(e, file.end_index);
  // Legacy format only had metadata (bytes), no file_size/footer_size
  avro::encode(e, std::vector<uint8_t>());  // metadata
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

  encoder->arrayStart();
  if (!cgs.empty()) {
    encoder->setItemCount(cgs.size());
    for (const auto& cg : cgs) {
      encoder->startItem();
      encodeColumnGroup(*encoder, *cg);
    }
  }
  encoder->arrayEnd();

  encoder->arrayStart();
  if (!delta_logs.empty()) {
    encoder->setItemCount(delta_logs.size());
    for (const auto& dl : delta_logs) {
      encoder->startItem();
      encodeDeltaLog(*encoder, dl);
    }
  }
  encoder->arrayEnd();

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
    std::map<std::string, std::vector<std::string>> legacy_stats;
    for (const auto& [key, stat] : stats) {
      legacy_stats[key] = stat.paths;
    }
    avro::encode(*encoder, legacy_stats);
  }

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

// Helper: strip base_path + dir prefix from file paths to create relative paths for legacy encoding
static ColumnGroups MakeLegacyCgs(const ColumnGroups& cgs, const std::string& base_path) {
  ColumnGroups legacy_cgs;
  for (const auto& cg : cgs) {
    auto copy = std::make_shared<ColumnGroup>(*cg);
    for (auto& f : copy->files) {
      std::string prefix = base_path + "/_data/";
      if (f.path.find(prefix) == 0) {
        f.path = f.path.substr(prefix.size());
      }
    }
    legacy_cgs.push_back(copy);
  }
  return legacy_cgs;
}

// Helper: write raw bytes to a manifest file path and read back via ReadFrom
arrow::Result<std::shared_ptr<Manifest>> WriteLegacyAndReadBack(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                                const std::string& base_path,
                                                                const std::string& legacy_bytes,
                                                                size_t version = 1) {
  std::string path = milvus_storage::get_manifest_filepath(base_path, version);
  auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
  if (!parent.empty()) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
  }
  ARROW_ASSIGN_OR_RAISE(auto output, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output->Write(legacy_bytes.data(), legacy_bytes.size()));
  ARROW_RETURN_NOT_OK(output->Close());
  Manifest::CleanCache();
  return Manifest::ReadFrom(fs, path);
}

TEST_F(ColumnGroupsTest, LegacyV3Deserialize) {
  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {"s.bin"};
  stat.metadata = {{"k", "v"}};
  stats["key"] = stat;

  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "col";
  idx.index_type = "hnsw";
  idx.path = "i.idx";
  idx.properties = {{"M", "8"}};
  indexes.push_back(idx);

  auto legacy_cgs = MakeLegacyCgs(test_cgs_, base_path_);
  std::string legacy_bytes = encodeLegacyManifest(3, legacy_cgs, {}, stats, indexes);

  ASSERT_NE(legacy_bytes.substr(0, 4), std::string("Obj\x01", 4));

  ASSERT_AND_ASSIGN(auto deserialized, WriteLegacyAndReadBack(fs_, base_path_, legacy_bytes));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_EQ(deserialized->stats().at("key").metadata.at("k"), "v");
  EXPECT_EQ(deserialized->indexes().size(), 1);
  EXPECT_EQ(deserialized->indexes()[0].properties.at("M"), "8");
}

TEST_F(ColumnGroupsTest, LegacyV1Deserialize) {
  std::map<std::string, Statistics> stats;
  Statistics stat;
  stat.paths = {"old.bin"};
  stats["old_key"] = stat;

  auto legacy_cgs = MakeLegacyCgs(test_cgs_, base_path_);
  std::string legacy_bytes = encodeLegacyManifest(1, legacy_cgs, {}, stats, {});

  ASSERT_AND_ASSIGN(auto deserialized, WriteLegacyAndReadBack(fs_, base_path_, legacy_bytes, 2));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_TRUE(deserialized->stats().at("old_key").metadata.empty());
  EXPECT_TRUE(deserialized->indexes().empty());
}

TEST_F(ColumnGroupsTest, LegacyV2Deserialize) {
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "col";
  idx.index_type = "ivf";
  idx.path = "v2.idx";
  idx.properties = {};
  indexes.push_back(idx);

  auto legacy_cgs = MakeLegacyCgs(test_cgs_, base_path_);
  std::string legacy_bytes = encodeLegacyManifest(2, legacy_cgs, {}, {}, indexes);

  ASSERT_AND_ASSIGN(auto deserialized, WriteLegacyAndReadBack(fs_, base_path_, legacy_bytes, 3));

  EXPECT_EQ(deserialized->columnGroups().size(), 2);
  EXPECT_TRUE(deserialized->stats().empty());
  EXPECT_EQ(deserialized->indexes().size(), 1);
  EXPECT_EQ(deserialized->indexes()[0].index_type, "ivf");
}

TEST_F(ColumnGroupsTest, IndexRoundTripPreservesData) {
  std::vector<Index> indexes;
  Index idx;
  idx.column_name = "embedding";
  idx.index_type = "hnsw";
  idx.path = base_path_ + "/_index/embedding_hnsw.idx";
  idx.properties = {{"key", "value"}};
  indexes.push_back(idx);

  auto original =
      std::make_shared<Manifest>(test_cgs_, std::vector<DeltaLog>(), std::map<std::string, Statistics>(), indexes);

  ASSERT_AND_ASSIGN(auto deserialized, WriteAndReadBack(*original));

  ASSERT_EQ(deserialized->indexes().size(), 1);
  const Index* found = deserialized->getIndex("embedding", "hnsw");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->properties.at("key"), "value");

  original->indexes().clear();
  EXPECT_EQ(original->indexes().size(), 0);
  EXPECT_EQ(deserialized->indexes().size(), 1);
}
