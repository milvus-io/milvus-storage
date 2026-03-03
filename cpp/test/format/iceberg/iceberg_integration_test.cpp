// Copyright 2025 Zilliz
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

#include <arrow/api.h>

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/common/config.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage::iceberg {
namespace {

using namespace milvus_storage::api;

// Helper: build a ColumnGroupFile from IcebergFileInfo, converting delete_metadata_json bytes to properties.
static ColumnGroupFile MakeCgFile(const IcebergFileInfo& info) {
  std::unordered_map<std::string, std::string> props;
  if (!info.delete_metadata_json.empty()) {
    props[kPropertyMetadata] = std::string(info.delete_metadata_json.begin(), info.delete_metadata_json.end());
  }
  return ColumnGroupFile{info.data_file_path, 0, static_cast<int64_t>(info.record_count), std::move(props)};
}

class IcebergIntegrationTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    if (IsCloudEnv()) {
      GTEST_SKIP() << "Iceberg integration tests require local filesystem";
    }
  }

  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));

    // Override root_path to "/" so absolute local paths work directly
    // with the SubTreeFileSystem. Iceberg returns absolute paths for
    // local files, and all path resolution (data files, delete files)
    // needs to work with these absolute paths.
    api::SetValue(properties_, PROPERTY_FS_ROOT_PATH, "/");
    // Clear the filesystem cache to pick up the new root
    FilesystemCache::getInstance().clean();

    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // Use an absolute temp path within /tmp for the test table
    abs_table_dir_ = "/tmp/iceberg-integration-test/test_table";
  }

  void TearDown() override {
    // Clean up test directory
    auto status = fs_->DeleteDirContents("/tmp/iceberg-integration-test");
    if (!status.ok()) {
      // May already be deleted, that's fine
    }
    (void)fs_->DeleteDir("/tmp/iceberg-integration-test");
    FilesystemCache::getInstance().clean();
  }

  Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string abs_table_dir_;
};

// End-to-end: create Iceberg table → explore via PlanFiles → read via IcebergFormatReader
TEST_F(IcebergIntegrationTest, ExploreAndReadBasic) {
  const uint64_t num_rows = 50;

  // 1. Create a standard Iceberg table via Rust bridge
  auto table_info = CreateTestTable(abs_table_dir_, num_rows, false, {});

  // 2. Explore: plan files from the Iceberg metadata
  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);

  ASSERT_EQ(file_infos.size(), 1);
  ASSERT_EQ(file_infos[0].record_count, num_rows);
  ASSERT_EQ(file_infos[0].data_file_path, table_info.data_file_uri);
  ASSERT_TRUE(file_infos[0].delete_metadata_json.empty());

  // 3. Read: create FormatReader and read all data
  auto cg_file = MakeCgFile(file_infos[0]);

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, cg_file, properties_, columns, nullptr));

  // 4. Read all row groups
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ASSERT_GE(rg_infos.size(), 1);

  int64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_GT(batch->num_rows(), 0);
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));

  // 5. Verify data content from first chunk
  ASSERT_AND_ASSIGN(auto first_batch, reader->get_chunk(0));
  ASSERT_EQ(first_batch->num_columns(), 3);

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(first_batch->column(0));
  auto name_array = std::dynamic_pointer_cast<arrow::StringArray>(first_batch->column(1));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(first_batch->column(2));
  ASSERT_NE(id_array, nullptr);
  ASSERT_NE(name_array, nullptr);
  ASSERT_NE(value_array, nullptr);

  // Verify first few rows
  ASSERT_EQ(id_array->Value(0), 0);
  ASSERT_EQ(name_array->GetString(0), "row_0");
  ASSERT_DOUBLE_EQ(value_array->Value(0), 0.0);

  if (first_batch->num_rows() > 1) {
    ASSERT_EQ(id_array->Value(1), 1);
    ASSERT_EQ(name_array->GetString(1), "row_1");
    ASSERT_DOUBLE_EQ(value_array->Value(1), 1.5);
  }
}

// End-to-end: create Iceberg table with positional deletes → explore → read with filtering
TEST_F(IcebergIntegrationTest, ExploreAndReadWithPositionalDeletes) {
  const uint64_t num_rows = 20;
  std::vector<int64_t> deleted_positions = {3, 7, 15};

  // 1. Create Iceberg table with positional deletes
  auto table_info = CreateTestTable(abs_table_dir_, num_rows, true, deleted_positions);

  // 2. Explore
  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);

  ASSERT_EQ(file_infos.size(), 1);
  ASSERT_EQ(file_infos[0].record_count, num_rows);
  // Should have delete metadata
  ASSERT_FALSE(file_infos[0].delete_metadata_json.empty());

  // 3. Read with delete filtering
  auto cg_file = MakeCgFile(file_infos[0]);

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, cg_file, properties_, columns, nullptr));

  // 4. Read and count rows — should be num_rows - deleted_positions.size()
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  int64_t total_rows = 0;
  std::vector<int64_t> all_ids;

  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      all_ids.push_back(id_array->Value(j));
    }
  }

  auto expected_rows = static_cast<int64_t>(num_rows - deleted_positions.size());
  ASSERT_EQ(total_rows, expected_rows);

  // 5. Verify deleted rows are NOT present
  std::unordered_set<int64_t> deleted_set(deleted_positions.begin(), deleted_positions.end());
  for (auto id : all_ids) {
    ASSERT_EQ(deleted_set.count(id), 0) << "Deleted row id=" << id << " should not appear in results";
  }

  // 6. Verify all non-deleted rows ARE present
  std::unordered_set<int64_t> seen_ids(all_ids.begin(), all_ids.end());
  for (int64_t i = 0; i < static_cast<int64_t>(num_rows); ++i) {
    if (deleted_set.count(i) == 0) {
      ASSERT_EQ(seen_ids.count(i), 1) << "Non-deleted row id=" << i << " should appear in results";
    }
  }
}

// End-to-end: take() maps logical doc IDs to physical positions.
// With 30 rows and deletes at {5,10,20}, post-delete view has 27 rows:
//   logical 0  -> physical 0  (id=0)
//   logical 5  -> physical 6  (id=6)   [skip delete at 5]
//   logical 9  -> physical 11 (id=11)  [skip deletes at 5,10]
//   logical 18 -> physical 21 (id=21)  [skip deletes at 5,10,20]
TEST_F(IcebergIntegrationTest, TakeWithPositionalDeletes) {
  const uint64_t num_rows = 30;
  std::vector<int64_t> deleted_positions = {5, 10, 20};

  auto table_info = CreateTestTable(abs_table_dir_, num_rows, true, deleted_positions);

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
  ASSERT_EQ(file_infos.size(), 1);

  auto cg_file = MakeCgFile(file_infos[0]);

  std::vector<std::string> columns = {"id", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, cg_file, properties_, columns, nullptr));

  // Logical doc IDs (post-delete indices)
  std::vector<int64_t> take_positions = {0, 5, 9, 18};
  ASSERT_AND_ASSIGN(auto table, reader->take(take_positions));

  ASSERT_EQ(table->num_rows(), 4);

  auto id_col = std::dynamic_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
  ASSERT_NE(id_col, nullptr);
  ASSERT_EQ(id_col->Value(0), 0);   // logical 0  -> physical 0
  ASSERT_EQ(id_col->Value(1), 6);   // logical 5  -> physical 6
  ASSERT_EQ(id_col->Value(2), 11);  // logical 9  -> physical 11
  ASSERT_EQ(id_col->Value(3), 21);  // logical 18 -> physical 21

  auto val_col = std::dynamic_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
  ASSERT_NE(val_col, nullptr);
  ASSERT_DOUBLE_EQ(val_col->Value(0), 0.0);
  ASSERT_DOUBLE_EQ(val_col->Value(1), 6.0 * 1.5);
  ASSERT_DOUBLE_EQ(val_col->Value(2), 11.0 * 1.5);
  ASSERT_DOUBLE_EQ(val_col->Value(3), 21.0 * 1.5);
}

// Column projection: read only a subset of columns
TEST_F(IcebergIntegrationTest, ColumnProjection) {
  const uint64_t num_rows = 10;

  auto table_info = CreateTestTable(abs_table_dir_, num_rows, false, {});

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
  ASSERT_EQ(file_infos.size(), 1);

  auto cg_file = MakeCgFile(file_infos[0]);

  // Read only "name" column
  std::vector<std::string> columns = {"name"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, cg_file, properties_, columns, nullptr));

  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  ASSERT_EQ(batch->num_columns(), 1);
  ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(num_rows));

  auto name_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
  ASSERT_NE(name_array, nullptr);
  ASSERT_EQ(name_array->GetString(0), "row_0");
  ASSERT_EQ(name_array->GetString(9), "row_9");
}

// clone_reader() shares delete state
TEST_F(IcebergIntegrationTest, CloneReaderSharesDeletes) {
  const uint64_t num_rows = 10;
  std::vector<int64_t> deleted_positions = {2, 8};

  auto table_info = CreateTestTable(abs_table_dir_, num_rows, true, deleted_positions);

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);

  auto cg_file = MakeCgFile(file_infos[0]);

  std::vector<std::string> columns = {"id"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, cg_file, properties_, columns, nullptr));

  ASSERT_AND_ASSIGN(auto cloned, reader->clone_reader());

  // Both original and clone should show 8 rows (10 - 2 deleted)
  ASSERT_AND_ASSIGN(auto batch1, reader->get_chunk(0));
  ASSERT_AND_ASSIGN(auto batch2, cloned->get_chunk(0));
  ASSERT_EQ(batch1->num_rows(), 8);
  ASSERT_EQ(batch2->num_rows(), 8);
}

}  // namespace
}  // namespace milvus_storage::iceberg
