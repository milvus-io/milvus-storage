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

// Integration tests for reading external tables (Lance, Iceberg) on S3.
// These tests require:
//   STORAGE_TYPE=remote  CLOUD_PROVIDER=aws
//   ADDRESS=<s3-endpoint>  BUCKET_NAME=<bucket>  REGION=<region>

#include <gtest/gtest.h>
#include <chrono>

#include <iostream>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "lance_bridge.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage {

// Validate row group infos: offsets are contiguous and sum to expected_logical_rows.
static void ValidateRowGroupInfos(const std::vector<RowGroupInfo>& rg_infos, uint64_t expected_logical_rows) {
  ASSERT_FALSE(rg_infos.empty());
  ASSERT_EQ(rg_infos.front().start_offset, 0u);
  for (size_t i = 1; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, rg_infos[i - 1].end_offset)
        << "Row group " << i << " start_offset is not contiguous with previous end_offset";
  }
  ASSERT_EQ(rg_infos.back().end_offset, expected_logical_rows);
}

// Holds everything needed to read back a written table via FormatReader
struct WriteResult {
  api::ColumnGroupFile cgfile;
  std::shared_ptr<arrow::Schema> schema;  // nullptr for Iceberg
  uint64_t num_rows;
};

class ExternalTableTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "External table tests require cloud environment (STORAGE_TYPE=remote)";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties_));

    auto address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    auto region = GetEnvVar(ENV_VAR_REGION).ValueOr("");
    api::SetValue(properties_, "extfs.iam.storage_type", "remote");
    api::SetValue(properties_, "extfs.iam.cloud_provider", "aws");
    api::SetValue(properties_, "extfs.iam.address", address.c_str());
    api::SetValue(properties_, "extfs.iam.bucket_name", bucket.c_str());
    api::SetValue(properties_, "extfs.iam.region", region.c_str());
    auto use_iam_str = GetEnvVar(ENV_VAR_USE_IAM).ValueOr("false");
    api::SetValue(properties_, "extfs.iam.use_iam", use_iam_str.c_str());
    if (use_iam_str != "true" && use_iam_str != "1") {
      auto ak = GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("minioadmin");
      auto sk = GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("minioadmin");
      api::SetValue(properties_, "extfs.iam.access_key_id", ak.c_str());
      api::SetValue(properties_, "extfs.iam.access_key_value", sk.c_str());
    }
    FilesystemCache::getInstance().clean();

    // External table tests currently only support S3-compatible storage (AWS, MinIO).
    if (fs_config_.cloud_provider != kCloudProviderAWS) {
      GTEST_SKIP() << "External table tests require S3-compatible storage (CLOUD_PROVIDER=aws)";
    }

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "exttable-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (fs_) {
      (void)DeleteTestDir(fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  // Create a table on S3 and return the ColumnGroupFile for reading
  arrow::Result<WriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  // Create a table with some rows deleted, return ColumnGroupFile for reading
  arrow::Result<WriteResult> CreateTestTableWithDeletes(const std::string& format,
                                                        uint64_t num_rows,
                                                        const std::vector<int64_t>& deleted_ids) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTableWithDeletes(num_rows, deleted_ids);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTableWithDeletes(num_rows, deleted_ids);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  std::string MakeTableUri(const std::string& bucket, const std::string& path) { return "s3://" + bucket + "/" + path; }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  ArrowFileSystemConfig fs_config_;
  std::string test_base_;

  private:
  arrow::Result<WriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, properties_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    return WriteResult{std::move(cgfile), schema, num_rows};
  }

  arrow::Result<WriteResult> CreateLanceTableWithDeletes(uint64_t num_rows, const std::vector<int64_t>& deleted_ids) {
    // Write the full dataset first
    ARROW_ASSIGN_OR_RAISE(auto result, CreateLanceTable(num_rows));

    // Open the dataset and delete rows by predicate
    ARROW_ASSIGN_OR_RAISE(auto parsed, lance::ParseLanceUri(result.cgfile.path));
    auto lance_uri = parsed.first;

    ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
    auto storage_options = lance::ToStorageOptions(fs_config);

    auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

    // Build predicate like "id in (3, 10, 25)"
    std::string predicate = "id in (";
    for (size_t i = 0; i < deleted_ids.size(); ++i) {
      if (i > 0)
        predicate += ", ";
      predicate += std::to_string(deleted_ids[i]);
    }
    predicate += ")";
    dataset->DeleteRows(predicate);

    return WriteResult{std::move(result.cgfile), result.schema, num_rows};
  }

  arrow::Result<WriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    if (bucket.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set");
    }
    auto path = test_base_ + "/iceberg";
    auto table_uri = MakeTableUri(bucket, path);
    auto storage_options = iceberg::ToStorageOptions(fs_config_);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }

    auto milvus_path = iceberg::ToMilvusUri(file_infos[0].data_file_path, fs_config_.address);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(file_infos[0].record_count), {}};
    return WriteResult{std::move(cg_file), nullptr, num_rows};
  }

  arrow::Result<WriteResult> CreateIcebergTableWithDeletes(uint64_t num_rows, const std::vector<int64_t>& deleted_ids) {
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    if (bucket.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set");
    }
    auto path = test_base_ + "/iceberg-deletes";
    auto table_uri = MakeTableUri(bucket, path);
    auto storage_options = iceberg::ToStorageOptions(fs_config_);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, true, deleted_ids, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }

    auto milvus_path = iceberg::ToMilvusUri(file_infos[0].data_file_path, fs_config_.address);
    std::unordered_map<std::string, std::string> file_props;
    file_props[api::kPropertyMetadata] =
        iceberg::ConvertDeleteMetadataPaths(file_infos[0].delete_metadata_json, fs_config_.address);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(file_infos[0].record_count),
                                 std::move(file_props)};
    return WriteResult{std::move(cg_file), nullptr, num_rows};
  }
};

// ---------------------------------------------------------------------------
// Parameterized: write table to S3, read back via FormatReader, verify data
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, WriteAndRead) {
  const auto& format = GetParam();
  const uint64_t num_rows = 100;

  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  // Read all row groups and count total rows
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, num_rows);

  int64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));

  // Verify data content from the first chunk
  ASSERT_AND_ASSIGN(auto first_batch, reader->get_chunk(0));
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(first_batch->column(0));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(first_batch->column(2));
  ASSERT_NE(id_array, nullptr);
  ASSERT_NE(value_array, nullptr);

  ASSERT_EQ(id_array->Value(0), 0);
  ASSERT_DOUBLE_EQ(value_array->Value(0), 0.0);
  if (first_batch->num_rows() > 1) {
    ASSERT_EQ(id_array->Value(1), 1);
    ASSERT_DOUBLE_EQ(value_array->Value(1), 1.5);
  }
}

INSTANTIATE_TEST_SUITE_P(Formats,
                         ExternalTableTest,
                         ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE));

// ---------------------------------------------------------------------------
// Parameterized: write table with deletes, read back, verify deletions
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, WriteAndReadWithDeletes) {
  const auto& format = GetParam();
  const uint64_t num_rows = 30;
  std::vector<int64_t> deleted_ids = {3, 10, 25};

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  int64_t total_rows = 0;
  std::vector<int64_t> all_ids;

  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      all_ids.push_back(id_array->Value(j));
    }
  }

  ASSERT_EQ(total_rows, expected_rows);

  std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
  for (auto id : all_ids) {
    ASSERT_EQ(deleted_set.count(id), 0) << "Deleted row id=" << id << " should not appear in results";
  }

  std::unordered_set<int64_t> seen_ids(all_ids.begin(), all_ids.end());
  for (int64_t i = 0; i < static_cast<int64_t>(num_rows); ++i) {
    if (deleted_set.count(i) == 0) {
      ASSERT_EQ(seen_ids.count(i), 1u) << "Non-deleted row id=" << i << " should appear in results";
    }
  }

  // --- get_chunks: read all row groups in one call ---
  {
    std::vector<int> all_rg_indices;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      all_rg_indices.push_back(static_cast<int>(i));
    }
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(all_rg_indices));
    int64_t chunks_total = 0;
    for (const auto& b : batches) {
      chunks_total += b->num_rows();
    }
    ASSERT_EQ(chunks_total, static_cast<int64_t>(num_rows - deleted_ids.size()));
  }

  // --- take: logical indices ---
  // Logical indices {0, 1, 2, 3}: the first 4 non-deleted rows.
  // Deleted ids are {3, 10, 25}, so logical row 3 = id 4 (skipping deleted id=3).
  {
    std::vector<int64_t> take_indices = {0, 1, 2, 3};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 4);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 0);
    ASSERT_EQ(take_ids->Value(1), 1);
    ASSERT_EQ(take_ids->Value(2), 2);
    ASSERT_EQ(take_ids->Value(3), 4);  // logical row 3 = physical row 4 (id=3 was deleted)
  }

  // --- read_with_range: logical range [0, 5) ---
  // Returns the first 5 non-deleted rows: ids 0,1,2,4,5
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 5));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 5);
    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));
    ASSERT_EQ(range_ids->Value(0), 0);
    ASSERT_EQ(range_ids->Value(1), 1);
    ASSERT_EQ(range_ids->Value(2), 2);
    ASSERT_EQ(range_ids->Value(3), 4);
    ASSERT_EQ(range_ids->Value(4), 5);
  }
}

// ---------------------------------------------------------------------------
// Parameterized: large table with many deletes spanning multiple row groups
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, LargeTableWithDeletesAcrossRowGroups) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Delete rows spanning multiple row groups (default logical_chunk_rows = 8192).
  // Covers: first/last row, row group boundaries, dense clusters, scattered.
  std::vector<int64_t> deleted_ids = {
      0,                                       // first row
      1,      2,      3,      4,      5,       // dense cluster at start
      8191,   8192,                            // row group 0/1 boundary
      16383,  16384,  16385,                   // row group 1/2 boundary cluster
      50000,  50001,  50002,  50003,  50004,   // dense cluster mid-file
      100000, 200000, 300000, 400000,          // scattered across file
      500000, 600000, 700000, 800000, 900000,  // scattered continued
      999998, 999999,                          // last rows
  };

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  // Read all chunks, collect all ids
  int64_t total_rows = 0;
  std::unordered_set<int64_t> seen_ids;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      seen_ids.insert(id_array->Value(j));
    }
  }

  ASSERT_EQ(total_rows, expected_rows);

  // Verify deleted rows absent, non-deleted rows present
  std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
  for (auto id : deleted_ids) {
    ASSERT_EQ(seen_ids.count(id), 0u) << "Deleted id=" << id << " should not appear";
  }
  ASSERT_EQ(static_cast<int64_t>(seen_ids.size()), expected_rows);

  // get_chunks: read all row groups in one call
  {
    std::vector<int> all_rg_indices;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      all_rg_indices.push_back(static_cast<int>(i));
    }
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(all_rg_indices));
    int64_t chunks_total = 0;
    for (const auto& b : batches) {
      chunks_total += b->num_rows();
    }
    ASSERT_EQ(chunks_total, expected_rows);
  }

  // take: logical indices around deletion boundaries
  {
    // Logical rows 0..4 should skip deleted ids {0,1,2,3,4,5} → ids 6,7,8,9,10
    std::vector<int64_t> take_indices = {0, 1, 2, 3, 4};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 5);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 6);
    ASSERT_EQ(take_ids->Value(1), 7);
    ASSERT_EQ(take_ids->Value(2), 8);
    ASSERT_EQ(take_ids->Value(3), 9);
    ASSERT_EQ(take_ids->Value(4), 10);
  }

  // read_with_range: logical range [0, 5)
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 5));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 5);
    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));
    ASSERT_EQ(range_ids->Value(0), 6);
    ASSERT_EQ(range_ids->Value(1), 7);
    ASSERT_EQ(range_ids->Value(2), 8);
    ASSERT_EQ(range_ids->Value(3), 9);
    ASSERT_EQ(range_ids->Value(4), 10);
  }
}

// ---------------------------------------------------------------------------
// Parameterized: delete entire first row group worth of rows (contiguous block)
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, DeleteEntireFirstRowGroup) {
  const auto& format = GetParam();
  const uint64_t num_rows = 20000;  // ~2.4 row groups at chunk_rows=8192

  // Delete rows 0-8191: the entire first logical row group
  std::vector<int64_t> deleted_ids;
  for (int64_t i = 0; i < 8192; ++i) {
    deleted_ids.push_back(i);
  }

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  int64_t total_rows = 0;
  std::vector<int64_t> all_ids;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      all_ids.push_back(id_array->Value(j));
    }
  }
  ASSERT_EQ(total_rows, expected_rows);

  // First surviving row should be id=8192
  ASSERT_FALSE(all_ids.empty());
  ASSERT_EQ(all_ids.front(), 8192);
  ASSERT_EQ(all_ids.back(), static_cast<int64_t>(num_rows - 1));

  // No deleted ids should appear
  for (auto id : all_ids) {
    ASSERT_GE(id, 8192) << "Deleted id=" << id << " should not appear";
  }
}

// ---------------------------------------------------------------------------
// Parameterized: delete ALL rows in the file — reader should return empty data
// Lance removes the fragment from manifest on full delete, so skip Lance.
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, DeleteAllRows) {
  const auto& format = GetParam();
  if (format == LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "Lance removes fully-deleted fragments from manifest; not reachable in production";
  }
  const uint64_t num_rows = 100;

  // Delete every row
  std::vector<int64_t> deleted_ids;
  for (int64_t i = 0; i < static_cast<int64_t>(num_rows); ++i) {
    deleted_ids.push_back(i);
  }

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  // Row group infos should sum to 0 logical rows
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  uint64_t total_logical = 0;
  for (const auto& rg : rg_infos) {
    total_logical += rg.end_offset - rg.start_offset;
  }
  ASSERT_EQ(total_logical, 0u);

  // read_with_range [0, 0) should return empty
  ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 0));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
  ASSERT_EQ(range_table->num_rows(), 0);

  // take with empty indices should return empty
  ASSERT_AND_ASSIGN(auto take_result, reader->take({}));
  ASSERT_EQ(take_result->num_rows(), 0);
}

// ---------------------------------------------------------------------------
// Parameterized: read_with_range spanning multiple row groups
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, ReadWithRangeAcrossRowGroups) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Deletions scattered across multiple row groups
  std::vector<int64_t> deleted_ids = {
      100,   4000,  8000,   // in row group 0
      8192,  10000, 12000,  // in row group 1
      16384, 18000,         // in row group 2
  };

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_total = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_total));

  // read_with_range [0, 20000): spans ~3 row groups (8192 each), crosses deletion boundaries
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 20000));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 20000);

    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));

    // Verify no deleted ids in the range
    std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
    for (int64_t j = 0; j < range_batch->num_rows(); ++j) {
      ASSERT_EQ(deleted_set.count(range_ids->Value(j)), 0u)
          << "Deleted id=" << range_ids->Value(j) << " at logical position " << j;
    }

    // Verify ordering: ids should be monotonically increasing
    for (int64_t j = 1; j < range_batch->num_rows(); ++j) {
      ASSERT_GT(range_ids->Value(j), range_ids->Value(j - 1)) << "IDs not monotonically increasing at position " << j;
    }
  }
}

// ---------------------------------------------------------------------------
// Parameterized: take with logical indices crossing row group boundaries
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, TakeAcrossRowGroupBoundary) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Delete some rows near the row group boundary (logical chunk_rows=8192)
  // Physical ids 8190, 8191, 8193 will be deleted.
  // After deletion, physical id 8192 shifts to a different logical index.
  std::vector<int64_t> deleted_ids = {8190, 8191, 8193};

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_total = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_total));

  // Physical ids near boundary: ..., 8188, 8189, [8190 del], [8191 del], 8192, [8193 del], 8194, ...
  // Logical mapping:
  //   logical 8188 → physical 8188 (id=8188)
  //   logical 8189 → physical 8189 (id=8189)
  //   logical 8190 → physical 8192 (id=8192, skipping 8190 & 8191)
  //   logical 8191 → physical 8194 (id=8194, skipping 8193)
  //   logical 8192 → physical 8195 (id=8195)
  {
    std::vector<int64_t> take_indices = {8188, 8189, 8190, 8191, 8192};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 5);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 8188);
    ASSERT_EQ(take_ids->Value(1), 8189);
    ASSERT_EQ(take_ids->Value(2), 8192);  // skipped 8190, 8191
    ASSERT_EQ(take_ids->Value(3), 8194);  // skipped 8193
    ASSERT_EQ(take_ids->Value(4), 8195);
  }
}

}  // namespace milvus_storage
