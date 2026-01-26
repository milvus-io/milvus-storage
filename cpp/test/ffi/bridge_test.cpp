
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

#include <sstream>
#include <random>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class BridgeTest : public ::testing::Test {};

TEST_F(BridgeTest, ExportImportColumnGroups) {
  // create column groups
  ColumnGroups cgs;

  ColumnGroupFile cgf1{
      .path = "test_file1", .start_index = 0, .end_index = 10, .metadata = std::vector<uint8_t>({1, 2, 3, 4})};

  ColumnGroupFile cgf2{
      .path = "test_file2",
      .start_index = 2000,
      .end_index = 10000,
      .metadata = std::vector<uint8_t>(),
  };

  ColumnGroupFile cgf3{
      .path = "test_file2",
      .start_index = 100,
      .end_index = 200,
      .metadata = std::vector<uint8_t>({1, 3, 4, 5}),
  };

  ColumnGroup cg1{
      .columns = std::vector<std::string>({"col1", "col2"}),
      .format = "parquet",
      .files = std::vector<ColumnGroupFile>({cgf1, cgf2}),
  };

  ColumnGroup cg2{
      .columns = std::vector<std::string>({"col3"}),
      .format = "parquet",
      .files = std::vector<ColumnGroupFile>({cgf3}),
  };

  cgs.push_back(std::make_shared<ColumnGroup>(cg1));
  cgs.push_back(std::make_shared<ColumnGroup>(cg2));

  // test export and import
  LoonColumnGroups* ccgs = nullptr;
  ASSERT_STATUS_OK(column_groups_export(cgs, &ccgs));

  // verify export C struct
  {
    ASSERT_NE(ccgs, nullptr);
    ASSERT_NE(ccgs->column_group_array, nullptr);
    ASSERT_EQ(ccgs->num_of_column_groups, 2);
    ASSERT_NE(ccgs->column_group_array[0].columns, nullptr);
    ASSERT_EQ(ccgs->column_group_array[0].num_of_columns, 2);
    ASSERT_EQ(ccgs->column_group_array[0].columns[0], cg1.columns[0]);
    ASSERT_EQ(ccgs->column_group_array[0].columns[1], cg1.columns[1]);

    ASSERT_EQ(ccgs->column_group_array[0].format, cg1.format);
    ASSERT_NE(ccgs->column_group_array[0].files, nullptr);
    ASSERT_EQ(ccgs->column_group_array[0].num_of_files, 2);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].path, cg1.files[0].path);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].start_index, cg1.files[0].start_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].end_index, cg1.files[0].end_index);
    ASSERT_EQ(std::vector<uint8_t>(
                  ccgs->column_group_array[0].files[0].metadata,
                  ccgs->column_group_array[0].files[0].metadata + ccgs->column_group_array[0].files[0].metadata_size),
              cg1.files[0].metadata);

    ASSERT_EQ(ccgs->column_group_array[0].files[1].path, cg1.files[1].path);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].start_index, cg1.files[1].start_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].end_index, cg1.files[1].end_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].metadata, nullptr);

    ASSERT_NE(ccgs->column_group_array[1].columns, nullptr);
    ASSERT_EQ(ccgs->column_group_array[1].num_of_columns, 1);
    ASSERT_EQ(ccgs->column_group_array[1].columns[0], cg2.columns[0]);

    ASSERT_EQ(ccgs->column_group_array[1].format, cg2.format);
    ASSERT_NE(ccgs->column_group_array[1].files, nullptr);
    ASSERT_EQ(ccgs->column_group_array[1].num_of_files, 1);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].path, cgf3.path);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].start_index, cgf3.start_index);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].end_index, cgf3.end_index);
    ASSERT_EQ(std::vector<uint8_t>(
                  ccgs->column_group_array[1].files[0].metadata,
                  ccgs->column_group_array[1].files[0].metadata + ccgs->column_group_array[1].files[0].metadata_size),
              cgf3.metadata);
  }

  ColumnGroups imported_cgs;
  ASSERT_STATUS_OK(column_groups_import(ccgs, &imported_cgs));

  // verify import column groups should same with origin one
  {
    ASSERT_EQ(cgs.size(), imported_cgs.size());
    for (size_t i = 0; i < cgs.size(); ++i) {
      auto left_cg = cgs[i];
      auto right_cg = imported_cgs[i];

      ASSERT_EQ(left_cg->columns, right_cg->columns);
      ASSERT_EQ(left_cg->format, right_cg->format);
      ASSERT_EQ(left_cg->files.size(), right_cg->files.size());
      for (size_t j = 0; j < left_cg->files.size(); ++j) {
        ASSERT_EQ(left_cg->files[j].path, right_cg->files[j].path);
        ASSERT_EQ(left_cg->files[j].start_index, right_cg->files[j].start_index);
        ASSERT_EQ(left_cg->files[j].end_index, right_cg->files[j].end_index);
        ASSERT_EQ(left_cg->files[j].metadata, right_cg->files[j].metadata);
      }
    }
  }
  loon_column_groups_destroy(ccgs);
}

// Test empty column groups import
TEST_F(BridgeTest, ImportEmptyColumnGroups) {
  LoonColumnGroups ccgs;
  ccgs.column_group_array = nullptr;
  ccgs.num_of_column_groups = 0;

  ColumnGroups out_cgs;
  ASSERT_STATUS_OK(column_groups_import(&ccgs, &out_cgs));
  ASSERT_TRUE(out_cgs.empty());
}

// Test import with null column_group_array but num > 0 (error case)
TEST_F(BridgeTest, ImportInvalidColumnGroups) {
  LoonColumnGroups ccgs;
  ccgs.column_group_array = nullptr;
  ccgs.num_of_column_groups = 1;  // Invalid: array is null but count > 0

  ColumnGroups out_cgs;
  auto status = column_groups_import(&ccgs, &out_cgs);
  ASSERT_FALSE(status.ok());
}

// Test export and import manifest with delta logs and stats
TEST_F(BridgeTest, ExportImportManifestWithDeltaLogsAndStats) {
  // Create column groups
  ColumnGroups cgs;
  ColumnGroupFile cgf{.path = "data_file.parquet", .start_index = 0, .end_index = 100, .metadata = {}};
  ColumnGroup cg{.columns = {"col1", "col2"}, .format = "parquet", .files = {cgf}};
  cgs.push_back(std::make_shared<ColumnGroup>(cg));

  // Create delta logs
  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back(DeltaLog{.path = "delta_log_1.log", .type = DeltaLogType::PRIMARY_KEY, .num_entries = 100});
  delta_logs.push_back(DeltaLog{.path = "delta_log_2.log", .type = DeltaLogType::PRIMARY_KEY, .num_entries = 200});

  // Create stats
  std::map<std::string, std::vector<std::string>> stats;
  stats["stat_key_1"] = {"stat_file_1.parquet", "stat_file_2.parquet"};
  stats["stat_key_2"] = {"stat_file_3.parquet"};

  // Create manifest with all components
  auto manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats);

  // Export manifest
  LoonManifest* cmanifest = nullptr;
  ASSERT_STATUS_OK(manifest_export(manifest, &cmanifest));

  // Verify exported manifest
  ASSERT_NE(cmanifest, nullptr);

  // Verify column groups
  ASSERT_EQ(cmanifest->column_groups.num_of_column_groups, 1);
  ASSERT_NE(cmanifest->column_groups.column_group_array, nullptr);

  // Verify delta logs
  ASSERT_EQ(cmanifest->delta_logs.num_delta_logs, 2);
  ASSERT_NE(cmanifest->delta_logs.delta_log_paths, nullptr);
  ASSERT_NE(cmanifest->delta_logs.delta_log_num_entries, nullptr);
  ASSERT_STREQ(cmanifest->delta_logs.delta_log_paths[0], "delta_log_1.log");
  ASSERT_STREQ(cmanifest->delta_logs.delta_log_paths[1], "delta_log_2.log");
  ASSERT_EQ(cmanifest->delta_logs.delta_log_num_entries[0], 100);
  ASSERT_EQ(cmanifest->delta_logs.delta_log_num_entries[1], 200);

  // Verify stats
  ASSERT_EQ(cmanifest->stats.num_stats, 2);
  ASSERT_NE(cmanifest->stats.stat_keys, nullptr);
  ASSERT_NE(cmanifest->stats.stat_files, nullptr);
  ASSERT_NE(cmanifest->stats.stat_file_counts, nullptr);

  // Import manifest back
  std::shared_ptr<Manifest> imported_manifest;
  ASSERT_STATUS_OK(manifest_import(cmanifest, &imported_manifest));

  // Verify imported manifest
  ASSERT_NE(imported_manifest, nullptr);

  // Verify imported column groups
  ASSERT_EQ(imported_manifest->columnGroups().size(), 1);

  // Verify imported delta logs
  ASSERT_EQ(imported_manifest->deltaLogs().size(), 2);
  ASSERT_EQ(imported_manifest->deltaLogs()[0].path, "delta_log_1.log");
  ASSERT_EQ(imported_manifest->deltaLogs()[0].num_entries, 100);
  ASSERT_EQ(imported_manifest->deltaLogs()[1].path, "delta_log_2.log");
  ASSERT_EQ(imported_manifest->deltaLogs()[1].num_entries, 200);

  // Verify imported stats
  const auto& imported_stats = imported_manifest->stats();
  ASSERT_EQ(imported_stats.size(), 2);
  ASSERT_TRUE(imported_stats.count("stat_key_1") > 0);
  ASSERT_TRUE(imported_stats.count("stat_key_2") > 0);
  ASSERT_EQ(imported_stats.at("stat_key_1").size(), 2);
  ASSERT_EQ(imported_stats.at("stat_key_2").size(), 1);

  // Clean up
  loon_manifest_destroy(cmanifest);
}

// Test export manifest with empty delta logs and stats
TEST_F(BridgeTest, ExportImportManifestEmpty) {
  // Create empty manifest
  ColumnGroups cgs;
  std::vector<DeltaLog> delta_logs;
  std::map<std::string, std::vector<std::string>> stats;

  auto manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats);

  // Export manifest
  LoonManifest* cmanifest = nullptr;
  ASSERT_STATUS_OK(manifest_export(manifest, &cmanifest));

  // Verify exported manifest
  ASSERT_NE(cmanifest, nullptr);
  ASSERT_EQ(cmanifest->column_groups.num_of_column_groups, 0);
  ASSERT_EQ(cmanifest->delta_logs.num_delta_logs, 0);
  ASSERT_EQ(cmanifest->stats.num_stats, 0);

  // Import back
  std::shared_ptr<Manifest> imported_manifest;
  ASSERT_STATUS_OK(manifest_import(cmanifest, &imported_manifest));

  ASSERT_NE(imported_manifest, nullptr);
  ASSERT_TRUE(imported_manifest->columnGroups().empty());
  ASSERT_TRUE(imported_manifest->deltaLogs().empty());
  ASSERT_TRUE(imported_manifest->stats().empty());

  // Clean up
  loon_manifest_destroy(cmanifest);
}

// Test export column groups with empty input
TEST_F(BridgeTest, ExportEmptyColumnGroups) {
  ColumnGroups cgs;

  LoonColumnGroups* ccgs = nullptr;
  ASSERT_STATUS_OK(column_groups_export(cgs, &ccgs));

  ASSERT_NE(ccgs, nullptr);
  ASSERT_EQ(ccgs->num_of_column_groups, 0);

  loon_column_groups_destroy(ccgs);
}

// Test column_groups_debug_string with null input
TEST_F(BridgeTest, ColumnGroupsDebugStringNull) {
  std::string result = column_groups_debug_string(nullptr);
  ASSERT_EQ(result, "LoonColumnGroups(null)");
}

// Test column_groups_debug_string with valid input
TEST_F(BridgeTest, ColumnGroupsDebugStringValid) {
  ColumnGroups cgs;
  ColumnGroupFile cgf{.path = "test.parquet", .start_index = 0, .end_index = 100, .metadata = {1, 2, 3}};
  ColumnGroup cg{.columns = {"col1", "col2"}, .format = "parquet", .files = {cgf}};
  cgs.push_back(std::make_shared<ColumnGroup>(cg));

  LoonColumnGroups* ccgs = nullptr;
  ASSERT_STATUS_OK(column_groups_export(cgs, &ccgs));

  std::string result = column_groups_debug_string(ccgs);
  ASSERT_TRUE(result.find("LoonColumnGroups(num_of_column_groups=1)") != std::string::npos);
  ASSERT_TRUE(result.find("col1") != std::string::npos);
  ASSERT_TRUE(result.find("test.parquet") != std::string::npos);

  loon_column_groups_destroy(ccgs);
}

// Test manifest_debug_string with null input
TEST_F(BridgeTest, ManifestDebugStringNull) {
  std::string result = manifest_debug_string(nullptr);
  ASSERT_EQ(result, "LoonManifest(null)");
}

// Test manifest_debug_string with valid input
TEST_F(BridgeTest, ManifestDebugStringValid) {
  ColumnGroups cgs;
  ColumnGroupFile cgf{.path = "data.parquet", .start_index = 0, .end_index = 50, .metadata = {}};
  ColumnGroup cg{.columns = {"col1"}, .format = "parquet", .files = {cgf}};
  cgs.push_back(std::make_shared<ColumnGroup>(cg));

  std::vector<DeltaLog> delta_logs;
  delta_logs.push_back(DeltaLog{.path = "delta.log", .type = DeltaLogType::PRIMARY_KEY, .num_entries = 10});

  std::map<std::string, std::vector<std::string>> stats;
  stats["stat_key"] = {"stat_file.parquet"};

  auto manifest = std::make_shared<Manifest>(std::move(cgs), delta_logs, stats);

  LoonManifest* cmanifest = nullptr;
  ASSERT_STATUS_OK(manifest_export(manifest, &cmanifest));

  std::string result = manifest_debug_string(cmanifest);
  ASSERT_TRUE(result.find("LoonManifest:") != std::string::npos);
  ASSERT_TRUE(result.find("DeltaLogs") != std::string::npos);
  ASSERT_TRUE(result.find("delta.log") != std::string::npos);
  ASSERT_TRUE(result.find("Stats") != std::string::npos);

  loon_manifest_destroy(cmanifest);
}

}  // namespace milvus_storage::test
