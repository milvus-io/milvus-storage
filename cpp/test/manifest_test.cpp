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

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/manifest.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class ManifestTest : public ::testing::Test {
  protected:
  void SetUp() override {
    Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("manifest-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  // Helper: write manifest and read it back via WriteTo/ReadFrom
  std::shared_ptr<Manifest> RoundTrip(const Manifest& manifest, size_t version = 1) {
    std::string path = get_manifest_filepath(base_path_, version);
    auto status = Manifest::WriteTo(fs_, path, manifest);
    EXPECT_TRUE(status.ok()) << status.ToString();
    Manifest::CleanCache();
    auto result = Manifest::ReadFrom(fs_, path);
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    return std::move(result).ValueOrDie();
  }

  // Helper: make a column group with specified columns, format, and files
  static std::shared_ptr<ColumnGroup> MakeCG(std::vector<std::string> columns,
                                             const std::string& format,
                                             std::vector<ColumnGroupFile> files) {
    auto cg = std::make_shared<ColumnGroup>();
    cg->columns = std::move(columns);
    cg->format = format;
    cg->files = std::move(files);
    return cg;
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  api::Properties properties_;
  std::string base_path_;
};

// ---------- Read-Write Roundtrip Tests ----------

TEST_F(ManifestTest, EmptyManifestRoundTrip) {
  Manifest manifest;
  auto read_back = RoundTrip(manifest);
  ASSERT_NE(read_back, nullptr);
  EXPECT_TRUE(read_back->columnGroups().empty());
  EXPECT_TRUE(read_back->deltaLogs().empty());
  EXPECT_TRUE(read_back->stats().empty());
  EXPECT_TRUE(read_back->indexes().empty());
}

TEST_F(ManifestTest, ColumnGroupsRoundTrip) {
  auto cg = MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
                   {{.path = get_data_filepath(base_path_, "file1.parquet"), .start_index = 0, .end_index = 100}});

  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 1);
  auto& rcg = read_back->columnGroups()[0];
  EXPECT_EQ(rcg->columns.size(), 2);
  EXPECT_EQ(rcg->columns[0], "id");
  EXPECT_EQ(rcg->columns[1], "name");
  EXPECT_EQ(rcg->format, LOON_FORMAT_PARQUET);
  ASSERT_EQ(rcg->files.size(), 1);
  EXPECT_EQ(rcg->files[0].start_index, 0);
  EXPECT_EQ(rcg->files[0].end_index, 100);
}

TEST_F(ManifestTest, DeltaLogsRoundTrip) {
  DeltaLog d1{
      .path = get_delta_filepath(base_path_, "del1.parquet"), .type = DeltaLogType::PRIMARY_KEY, .num_entries = 50};
  DeltaLog d2{
      .path = get_delta_filepath(base_path_, "del2.parquet"), .type = DeltaLogType::POSITIONAL, .num_entries = 30};
  DeltaLog d3{
      .path = get_delta_filepath(base_path_, "del3.parquet"), .type = DeltaLogType::EQUALITY, .num_entries = 10};

  Manifest manifest({}, {d1, d2, d3});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->deltaLogs().size(), 3);
  EXPECT_EQ(read_back->deltaLogs()[0].type, DeltaLogType::PRIMARY_KEY);
  EXPECT_EQ(read_back->deltaLogs()[0].num_entries, 50);
  EXPECT_EQ(read_back->deltaLogs()[1].type, DeltaLogType::POSITIONAL);
  EXPECT_EQ(read_back->deltaLogs()[1].num_entries, 30);
  EXPECT_EQ(read_back->deltaLogs()[2].type, DeltaLogType::EQUALITY);
  EXPECT_EQ(read_back->deltaLogs()[2].num_entries, 10);
}

TEST_F(ManifestTest, StatsRoundTrip) {
  Statistics stat1;
  stat1.paths = {get_stats_filepath(base_path_, "bloom_100.bin"), get_stats_filepath(base_path_, "bloom_101.bin")};
  stat1.metadata = {{"type", "bloom_filter"}, {"fpp", "0.01"}};

  Statistics stat2;
  stat2.paths = {get_stats_filepath(base_path_, "bm25_101.bin")};
  stat2.metadata = {};

  std::map<std::string, Statistics> stats = {{"bloom_filter.100", stat1}, {"bm25.101", stat2}};

  Manifest manifest({}, {}, stats);
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->stats().size(), 2);

  auto& rs1 = read_back->stats().at("bloom_filter.100");
  EXPECT_EQ(rs1.paths.size(), 2);
  EXPECT_EQ(rs1.metadata.at("type"), "bloom_filter");
  EXPECT_EQ(rs1.metadata.at("fpp"), "0.01");

  auto& rs2 = read_back->stats().at("bm25.101");
  EXPECT_EQ(rs2.paths.size(), 1);
  EXPECT_TRUE(rs2.metadata.empty());
}

TEST_F(ManifestTest, IndexesRoundTrip) {
  Index idx1{.column_name = "vector",
             .index_type = "hnsw",
             .path = get_index_filepath(base_path_, "vec_hnsw.idx"),
             .properties = {{"M", "16"}, {"ef_construction", "128"}}};

  Index idx2{.column_name = "id",
             .index_type = "inverted",
             .path = get_index_filepath(base_path_, "id_inverted.idx"),
             .properties = {}};

  Manifest manifest({}, {}, {}, {idx1, idx2});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->indexes().size(), 2);

  const Index* found_hnsw = read_back->getIndex("vector", "hnsw");
  ASSERT_NE(found_hnsw, nullptr);
  EXPECT_EQ(found_hnsw->properties.at("M"), "16");
  EXPECT_EQ(found_hnsw->properties.at("ef_construction"), "128");

  const Index* found_inv = read_back->getIndex("id", "inverted");
  ASSERT_NE(found_inv, nullptr);
  EXPECT_TRUE(found_inv->properties.empty());
}

TEST_F(ManifestTest, FullManifestRoundTrip) {
  // Populate all fields
  auto cg1 =
      MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "cg1_part0.parquet"), .start_index = 0, .end_index = 500},
              {.path = get_data_filepath(base_path_, "cg1_part1.parquet"), .start_index = 500, .end_index = 1000}});
  auto cg2 = MakeCG({"value", "vector"}, LOON_FORMAT_PARQUET,
                    {{.path = get_data_filepath(base_path_, "cg2.parquet"), .start_index = 0, .end_index = 1000}});

  std::vector<DeltaLog> deltas = {
      {.path = get_delta_filepath(base_path_, "del.parquet"), .type = DeltaLogType::PRIMARY_KEY, .num_entries = 20}};

  Statistics stat;
  stat.paths = {get_stats_filepath(base_path_, "bloom.bin")};
  stat.metadata = {{"version", "1"}};
  std::map<std::string, Statistics> stats = {{"bloom_filter.100", stat}};

  std::vector<Index> indexes = {{.column_name = "vector",
                                 .index_type = "hnsw",
                                 .path = get_index_filepath(base_path_, "vec.idx"),
                                 .properties = {{"M", "16"}}}};

  Manifest manifest({cg1, cg2}, deltas, stats, indexes);
  auto read_back = RoundTrip(manifest);

  EXPECT_EQ(read_back->columnGroups().size(), 2);
  EXPECT_EQ(read_back->deltaLogs().size(), 1);
  EXPECT_EQ(read_back->stats().size(), 1);
  EXPECT_EQ(read_back->indexes().size(), 1);

  // Verify multi-file column group
  EXPECT_EQ(read_back->columnGroups()[0]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[0]->files[0].end_index, 500);
  EXPECT_EQ(read_back->columnGroups()[0]->files[1].start_index, 500);
}

// ---------- Column Group Policy Tests ----------

TEST_F(ManifestTest, SingleColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema));

  auto groups = policy->get_column_groups();
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0]->columns.size(), 4);
  EXPECT_EQ(groups[0]->format, LOON_FORMAT_PARQUET);

  // RoundTrip with policy-generated column groups
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), 1);
  EXPECT_EQ(read_back->columnGroups()[0]->columns.size(), 4);
}

TEST_F(ManifestTest, SchemaBasedColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  // "id|value" in group 1, "name" in group 2, "vector" in group 3
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy("id|value,name,vector", LOON_FORMAT_PARQUET, schema));

  auto groups = policy->get_column_groups();
  ASSERT_EQ(groups.size(), 3);

  // Verify each group has the expected columns
  EXPECT_EQ(groups[0]->columns.size(), 2);  // id, value
  EXPECT_EQ(groups[1]->columns.size(), 1);  // name
  EXPECT_EQ(groups[2]->columns.size(), 1);  // vector

  // RoundTrip
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), 3);

  // Verify column names survived roundtrip
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(read_back->columnGroups()[i]->columns.size(), groups[i]->columns.size());
    for (size_t j = 0; j < groups[i]->columns.size(); ++j) {
      EXPECT_EQ(read_back->columnGroups()[i]->columns[j], groups[i]->columns[j]);
    }
  }
}

TEST_F(ManifestTest, SizeBasedColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto test_batch, CreateTestData(schema));
  // max_avg_column_size = 1, max_columns_in_group = 2 -> should split into multiple groups
  ASSERT_AND_ASSIGN(auto policy, CreateSizeBasePolicy(1, 2, LOON_FORMAT_PARQUET, schema));
  // SizeBasedPolicy requires a sample before get_column_groups()
  ASSERT_STATUS_OK(policy->sample(test_batch));

  auto groups = policy->get_column_groups();
  EXPECT_GT(groups.size(), 1);

  // Every group should have at most 2 columns
  for (const auto& g : groups) {
    EXPECT_LE(g->columns.size(), 2);
    EXPECT_EQ(g->format, LOON_FORMAT_PARQUET);
  }

  // RoundTrip
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), groups.size());
}

// ---------- Hybrid Format Tests ----------

TEST_F(ManifestTest, HybridFormatsInSingleManifest) {
  auto cg_parquet =
      MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "cg_parquet.parquet"), .start_index = 0, .end_index = 100}});

  auto cg_iceberg = MakeCG({"value"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/warehouse/table/data/file1.parquet",
                             .start_index = 0,
                             .end_index = 100,
                             .properties = {{api::kPropertyMetadata, std::string({'\x01', '\x02', '\x03'})}}}});

  auto cg_lance = MakeCG({"vector"}, LOON_FORMAT_LANCE_TABLE,
                         {{.path = "s3://bucket/lance/table.lance", .start_index = 0, .end_index = 100}});

  Manifest manifest({cg_parquet, cg_iceberg, cg_lance});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 3);
  EXPECT_EQ(read_back->columnGroups()[0]->format, LOON_FORMAT_PARQUET);
  EXPECT_EQ(read_back->columnGroups()[1]->format, LOON_FORMAT_ICEBERG_TABLE);
  EXPECT_EQ(read_back->columnGroups()[2]->format, LOON_FORMAT_LANCE_TABLE);

  // Verify iceberg column group metadata bytes survived roundtrip
  EXPECT_EQ(read_back->columnGroups()[1]->files[0].properties.at(api::kPropertyMetadata),
            std::string({'\x01', '\x02', '\x03'}));

  // Verify external table paths are preserved as-is (absolute URIs)
  EXPECT_EQ(read_back->columnGroups()[1]->files[0].path, "s3://bucket/warehouse/table/data/file1.parquet");
  EXPECT_EQ(read_back->columnGroups()[2]->files[0].path, "s3://bucket/lance/table.lance");
}

TEST_F(ManifestTest, HybridFormatsWithGetColumnGroup) {
  auto cg_parquet = MakeCG({"id"}, LOON_FORMAT_PARQUET,
                           {{.path = get_data_filepath(base_path_, "p.parquet"), .start_index = 0, .end_index = 50}});
  auto cg_iceberg = MakeCG({"name", "value"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/iceberg/data.parquet",
                             .start_index = 0,
                             .end_index = 50,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\xAB')}}}});

  Manifest manifest({cg_parquet, cg_iceberg});
  auto read_back = RoundTrip(manifest);

  // getColumnGroup should locate each column in the correct group
  auto id_cg = read_back->getColumnGroup("id");
  ASSERT_NE(id_cg, nullptr);
  EXPECT_EQ(id_cg->format, LOON_FORMAT_PARQUET);

  auto name_cg = read_back->getColumnGroup("name");
  ASSERT_NE(name_cg, nullptr);
  EXPECT_EQ(name_cg->format, LOON_FORMAT_ICEBERG_TABLE);

  auto value_cg = read_back->getColumnGroup("value");
  ASSERT_NE(value_cg, nullptr);
  EXPECT_EQ(value_cg.get(), name_cg.get());  // same column group

  EXPECT_EQ(read_back->getColumnGroup("nonexistent"), nullptr);
}

// ---------- Multiple Files in One Column Group ----------

TEST_F(ManifestTest, MultipleFilesInOneColumnGroup) {
  std::vector<ColumnGroupFile> files;
  files.reserve(5);
  for (int i = 0; i < 5; ++i) {
    files.push_back({.path = get_data_filepath(base_path_, "part_" + std::to_string(i) + ".parquet"),
                     .start_index = i * 1000,
                     .end_index = (i + 1) * 1000});
  }

  auto cg = MakeCG({"id", "name", "value", "vector"}, LOON_FORMAT_PARQUET, files);
  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 1);
  auto& rcg = read_back->columnGroups()[0];
  ASSERT_EQ(rcg->files.size(), 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(rcg->files[i].start_index, i * 1000);
    EXPECT_EQ(rcg->files[i].end_index, (i + 1) * 1000);
  }
}

TEST_F(ManifestTest, MultipleFilesWithMetadata) {
  std::string meta1 = {'\x10', '\x20', '\x30'};
  std::string meta2 = {'\xAA', '\xBB'};
  std::string meta3 = {};

  auto cg = MakeCG({"id"}, LOON_FORMAT_ICEBERG_TABLE,
                   {{.path = "s3://bucket/data/part0.parquet",
                     .start_index = 0,
                     .end_index = 500,
                     .properties = {{api::kPropertyMetadata, meta1}}},
                    {.path = "s3://bucket/data/part1.parquet",
                     .start_index = 500,
                     .end_index = 1000,
                     .properties = {{api::kPropertyMetadata, meta2}}},
                    {.path = "s3://bucket/data/part2.parquet", .start_index = 1000, .end_index = 1500}});

  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  auto& rcg = read_back->columnGroups()[0];
  ASSERT_EQ(rcg->files.size(), 3);
  EXPECT_EQ(rcg->files[0].properties.at(api::kPropertyMetadata), meta1);
  EXPECT_EQ(rcg->files[1].properties.at(api::kPropertyMetadata), meta2);
  EXPECT_TRUE(rcg->files[2].properties.find(api::kPropertyMetadata) == rcg->files[2].properties.end());
}

// ---------- Edge Cases ----------

TEST_F(ManifestTest, WriteToExistingPathFails) {
  Manifest manifest;
  std::string path = get_manifest_filepath(base_path_, 1);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path, manifest));

  // Second write to the same path should fail with AlreadyExists
  auto status = Manifest::WriteTo(fs_, path, manifest);
  EXPECT_TRUE(status.IsAlreadyExists()) << status.ToString();
}

TEST_F(ManifestTest, ReadFromCachesResult) {
  auto cg = MakeCG({"id"}, LOON_FORMAT_PARQUET,
                   {{.path = get_data_filepath(base_path_, "cached.parquet"), .start_index = 0, .end_index = 10}});

  Manifest manifest({cg});
  std::string path = get_manifest_filepath(base_path_, 1);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path, manifest));

  Manifest::CleanCache();
  ASSERT_AND_ASSIGN(auto m1, Manifest::ReadFrom(fs_, path));
  ASSERT_AND_ASSIGN(auto m2, Manifest::ReadFrom(fs_, path));
  EXPECT_EQ(m1.get(), m2.get());  // same pointer from cache
}

TEST_F(ManifestTest, ManifestVersion) {
  Manifest manifest;
  EXPECT_EQ(manifest.version(), MANIFEST_VERSION);

  auto read_back = RoundTrip(manifest);
  EXPECT_EQ(read_back->version(), MANIFEST_VERSION);
}

TEST_F(ManifestTest, ColumnGroupsXFormatsXFiles) {
  // A realistic manifest with all 4 supported formats, multiple files per group,
  // plus delta logs, stats, and indexes.
  auto cg_parquet =
      MakeCG({"id"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "p0.parquet"), .start_index = 0, .end_index = 1000},
              {.path = get_data_filepath(base_path_, "p1.parquet"), .start_index = 1000, .end_index = 2000},
              {.path = get_data_filepath(base_path_, "p2.parquet"), .start_index = 2000, .end_index = 3000}});

  auto cg_vortex =
      MakeCG({"name"}, LOON_FORMAT_VORTEX,
             {{.path = get_data_filepath(base_path_, "v0.vortex"), .start_index = 0, .end_index = 1500},
              {.path = get_data_filepath(base_path_, "v1.vortex"), .start_index = 1500, .end_index = 3000}});

  auto cg_lance = MakeCG({"value"}, LOON_FORMAT_LANCE_TABLE,
                         {{.path = "s3://bucket/lance/table.lance", .start_index = 0, .end_index = 3000}});

  auto cg_iceberg = MakeCG({"vector"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/iceberg/data/i0.parquet",
                             .start_index = 0,
                             .end_index = 1500,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\x01')}}},
                            {.path = "s3://bucket/iceberg/data/i1.parquet",
                             .start_index = 1500,
                             .end_index = 3000,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\x02')}}}});

  DeltaLog delta{
      .path = get_delta_filepath(base_path_, "del.parquet"), .type = DeltaLogType::POSITIONAL, .num_entries = 100};

  Statistics stat;
  stat.paths = {get_stats_filepath(base_path_, "bloom.bin")};
  stat.metadata = {{"fpp", "0.001"}};

  Index idx{.column_name = "vector",
            .index_type = "hnsw",
            .path = get_index_filepath(base_path_, "vec_hnsw.idx"),
            .properties = {{"M", "32"}, {"ef_construction", "256"}}};

  Manifest manifest({cg_parquet, cg_vortex, cg_lance, cg_iceberg}, {delta}, {{"bloom_filter.100", stat}}, {idx});
  auto read_back = RoundTrip(manifest);

  // All 4 column groups present with correct formats
  ASSERT_EQ(read_back->columnGroups().size(), 4);
  EXPECT_EQ(read_back->columnGroups()[0]->format, LOON_FORMAT_PARQUET);
  EXPECT_EQ(read_back->columnGroups()[0]->files.size(), 3);
  EXPECT_EQ(read_back->columnGroups()[1]->format, LOON_FORMAT_VORTEX);
  EXPECT_EQ(read_back->columnGroups()[1]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[2]->format, LOON_FORMAT_LANCE_TABLE);
  EXPECT_EQ(read_back->columnGroups()[2]->files.size(), 1);
  EXPECT_EQ(read_back->columnGroups()[3]->format, LOON_FORMAT_ICEBERG_TABLE);
  EXPECT_EQ(read_back->columnGroups()[3]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[3]->files[0].properties.at(api::kPropertyMetadata), std::string(1, '\x01'));

  // External table paths preserved as absolute URIs
  EXPECT_EQ(read_back->columnGroups()[2]->files[0].path, "s3://bucket/lance/table.lance");
  EXPECT_EQ(read_back->columnGroups()[3]->files[0].path, "s3://bucket/iceberg/data/i0.parquet");

  // Delta logs
  ASSERT_EQ(read_back->deltaLogs().size(), 1);
  EXPECT_EQ(read_back->deltaLogs()[0].type, DeltaLogType::POSITIONAL);

  // Stats
  ASSERT_EQ(read_back->stats().size(), 1);
  EXPECT_EQ(read_back->stats().at("bloom_filter.100").metadata.at("fpp"), "0.001");

  // Indexes
  ASSERT_EQ(read_back->indexes().size(), 1);
  EXPECT_EQ(read_back->indexes()[0].properties.at("M"), "32");
}

}  // namespace milvus_storage::test
