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

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/transaction/transaction.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage {
namespace {

using milvus_storage::api::ColumnGroup;
using milvus_storage::api::ColumnGroupFile;
using milvus_storage::api::ColumnGroups;
using milvus_storage::api::kPropertyMetadata;
using milvus_storage::api::Manifest;
using milvus_storage::api::Properties;
using milvus_storage::api::SetValue;
using milvus_storage::api::transaction::Transaction;
using milvus_storage::iceberg::CreateTestTable;
using milvus_storage::iceberg::IcebergStorageOptions;
using milvus_storage::iceberg::PlanFiles;

class LoonTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    if (IsCloudEnv()) {
      GTEST_SKIP() << "Loon tests require local filesystem";
    }
  }

  void SetUp() override {
    Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    SetValue(properties_, PROPERTY_FS_ROOT_PATH, "/");
    FilesystemCache::getInstance().clean();
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_dir_ = "/tmp/loon-test";
    table_dir_ = base_dir_ + "/iceberg_source";
    target_dir_ = base_dir_ + "/target";
  }

  void TearDown() override {
    auto status = fs_->DeleteDirContents(base_dir_);
    (void)status;
    (void)fs_->DeleteDir(base_dir_);
    FilesystemCache::getInstance().clean();
  }

  Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_dir_;
  std::string table_dir_;
  std::string target_dir_;
};

// Create manifest from Iceberg table and read back sequentially
TEST_F(LoonTest, CreateAndReadIceberg) {
  const uint64_t num_rows = 30;

  // 1. Create Iceberg test table
  auto table_info = CreateTestTable(table_dir_, num_rows, false, {});

  // 2. Explore via PlanFiles
  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
  ASSERT_EQ(file_infos.size(), 1);

  // 3. Build ColumnGroup and commit manifest via Transaction
  std::vector<ColumnGroupFile> files;
  files.reserve(file_infos.size());
  for (const auto& info : file_infos) {
    std::unordered_map<std::string, std::string> file_props;
    if (!info.delete_metadata_json.empty()) {
      file_props[kPropertyMetadata] = std::string(info.delete_metadata_json.begin(), info.delete_metadata_json.end());
    }
    files.emplace_back(
        ColumnGroupFile{info.data_file_path, 0, static_cast<int64_t>(info.record_count), std::move(file_props)});
  }

  std::vector<std::string> columns = {"id", "name", "value"};
  ColumnGroups cgs;
  cgs.push_back(std::make_shared<ColumnGroup>(
      ColumnGroup{.columns = columns, .format = LOON_FORMAT_ICEBERG_TABLE, .files = files}));

  ASSERT_AND_ASSIGN(auto tx, Transaction::Open(fs_, target_dir_));
  tx->AppendFiles(cgs);
  ASSERT_AND_ASSIGN(auto version, tx->Commit());
  ASSERT_EQ(version, 1);

  // 4. Read manifest back
  auto manifest_path = get_manifest_filepath(target_dir_, version);
  ASSERT_AND_ASSIGN(auto manifest, Manifest::ReadFrom(fs_, manifest_path));

  auto& read_cgs = manifest->columnGroups();
  ASSERT_EQ(read_cgs.size(), 1);
  ASSERT_EQ(read_cgs[0]->format, LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_EQ(read_cgs[0]->files.size(), 1);

  // 5. Read data via FormatReader
  auto& f = read_cgs[0]->files[0];
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, read_cgs[0]->format, f, properties_, columns, nullptr));

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ASSERT_GE(rg_infos.size(), 1);

  int64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_GT(batch->num_rows(), 0);
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
}

// Create manifest from Iceberg table with deletes, read with take().
// take() maps logical doc IDs to physical positions.
// With 20 rows and deletes at {3,7,15}, post-delete view has 17 rows:
//   logical 0  -> physical 0  (id=0)
//   logical 3  -> physical 4  (id=4)   [skip delete at 3]
//   logical 6  -> physical 8  (id=8)   [skip deletes at 3,7]
//   logical 13 -> physical 16 (id=16)  [skip deletes at 3,7,15]
//   logical 16 -> physical 19 (id=19)  [skip all 3 deletes]
TEST_F(LoonTest, CreateAndTakeWithDeletes) {
  const uint64_t num_rows = 20;
  std::vector<int64_t> deleted_positions = {3, 7, 15};

  auto table_info = CreateTestTable(table_dir_, num_rows, true, deleted_positions);

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
  ASSERT_EQ(file_infos.size(), 1);
  ASSERT_FALSE(file_infos[0].delete_metadata_json.empty());

  // Build manifest
  std::vector<ColumnGroupFile> files;
  std::unordered_map<std::string, std::string> props;
  if (!file_infos[0].delete_metadata_json.empty()) {
    props[kPropertyMetadata] =
        std::string(file_infos[0].delete_metadata_json.begin(), file_infos[0].delete_metadata_json.end());
  }
  files.emplace_back(ColumnGroupFile{file_infos[0].data_file_path, 0, static_cast<int64_t>(file_infos[0].record_count),
                                     std::move(props)});

  std::vector<std::string> columns = {"id", "value"};
  ColumnGroups cgs;
  cgs.push_back(std::make_shared<ColumnGroup>(
      ColumnGroup{.columns = columns, .format = LOON_FORMAT_ICEBERG_TABLE, .files = files}));

  ASSERT_AND_ASSIGN(auto tx, Transaction::Open(fs_, target_dir_));
  tx->AppendFiles(cgs);
  ASSERT_AND_ASSIGN(auto version, tx->Commit());

  // Read manifest
  auto manifest_path = get_manifest_filepath(target_dir_, version);
  ASSERT_AND_ASSIGN(auto manifest, Manifest::ReadFrom(fs_, manifest_path));

  // Take with logical doc IDs
  auto& f = manifest->columnGroups()[0]->files[0];
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, f, properties_, columns, nullptr));

  std::vector<int64_t> take_positions = {0, 3, 6, 13, 16};
  ASSERT_AND_ASSIGN(auto table, reader->take(take_positions));

  ASSERT_EQ(table->num_rows(), 5);

  auto id_col = std::dynamic_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
  ASSERT_NE(id_col, nullptr);
  ASSERT_EQ(id_col->Value(0), 0);   // logical 0  -> physical 0
  ASSERT_EQ(id_col->Value(1), 4);   // logical 3  -> physical 4
  ASSERT_EQ(id_col->Value(2), 8);   // logical 6  -> physical 8
  ASSERT_EQ(id_col->Value(3), 16);  // logical 13 -> physical 16
  ASSERT_EQ(id_col->Value(4), 19);  // logical 16 -> physical 19
}

// Sequential read filters out deleted rows
TEST_F(LoonTest, SequentialReadFiltersDeletes) {
  const uint64_t num_rows = 15;
  std::vector<int64_t> deleted_positions = {0, 5, 14};

  auto table_info = CreateTestTable(table_dir_, num_rows, true, deleted_positions);

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);

  std::vector<ColumnGroupFile> files;
  {
    std::unordered_map<std::string, std::string> file_props;
    if (!file_infos[0].delete_metadata_json.empty()) {
      file_props[kPropertyMetadata] =
          std::string(file_infos[0].delete_metadata_json.begin(), file_infos[0].delete_metadata_json.end());
    }
    files.emplace_back(ColumnGroupFile{file_infos[0].data_file_path, 0,
                                       static_cast<int64_t>(file_infos[0].record_count), std::move(file_props)});
  }

  std::vector<std::string> columns = {"id", "name", "value"};
  ColumnGroups cgs;
  cgs.push_back(std::make_shared<ColumnGroup>(
      ColumnGroup{.columns = columns, .format = LOON_FORMAT_ICEBERG_TABLE, .files = files}));

  ASSERT_AND_ASSIGN(auto tx, Transaction::Open(fs_, target_dir_));
  tx->AppendFiles(cgs);
  ASSERT_AND_ASSIGN(auto version, tx->Commit());

  // Read manifest back
  auto manifest_path = get_manifest_filepath(target_dir_, version);
  ASSERT_AND_ASSIGN(auto manifest, Manifest::ReadFrom(fs_, manifest_path));

  auto& f = manifest->columnGroups()[0]->files[0];
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, f, properties_, columns, nullptr));

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

  // 15 rows - 3 deleted = 12
  ASSERT_EQ(total_rows, 12);

  // Verify no deleted IDs appear
  std::unordered_set<int64_t> deleted_set(deleted_positions.begin(), deleted_positions.end());
  for (auto id : all_ids) {
    ASSERT_EQ(deleted_set.count(id), 0) << "Deleted row id=" << id << " should not appear";
  }
}

// Manifest preserves delete metadata through serialization round-trip
TEST_F(LoonTest, ManifestPreservesDeleteMetadata) {
  const uint64_t num_rows = 10;
  std::vector<int64_t> deleted_positions = {2, 8};

  auto table_info = CreateTestTable(table_dir_, num_rows, true, deleted_positions);

  IcebergStorageOptions storage_options;
  auto file_infos = PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);

  // Commit manifest with delete metadata
  std::vector<ColumnGroupFile> files;
  auto& original_metadata = file_infos[0].delete_metadata_json;
  ASSERT_FALSE(original_metadata.empty());

  std::string original_metadata_str(original_metadata.begin(), original_metadata.end());
  std::unordered_map<std::string, std::string> file_props;
  file_props[kPropertyMetadata] = original_metadata_str;
  files.emplace_back(ColumnGroupFile{file_infos[0].data_file_path, 0, static_cast<int64_t>(file_infos[0].record_count),
                                     std::move(file_props)});

  std::vector<std::string> columns = {"id"};
  ColumnGroups cgs;
  cgs.push_back(std::make_shared<ColumnGroup>(
      ColumnGroup{.columns = columns, .format = LOON_FORMAT_ICEBERG_TABLE, .files = files}));

  ASSERT_AND_ASSIGN(auto tx, Transaction::Open(fs_, target_dir_));
  tx->AppendFiles(cgs);
  ASSERT_AND_ASSIGN(auto version, tx->Commit());

  // Read manifest and verify metadata is preserved
  auto manifest_path = get_manifest_filepath(target_dir_, version);
  ASSERT_AND_ASSIGN(auto manifest, Manifest::ReadFrom(fs_, manifest_path));

  auto& round_tripped_props = manifest->columnGroups()[0]->files[0].properties;
  auto it = round_tripped_props.find(kPropertyMetadata);
  ASSERT_NE(it, round_tripped_props.end());
  ASSERT_EQ(it->second, original_metadata_str);
}

}  // namespace
}  // namespace milvus_storage
