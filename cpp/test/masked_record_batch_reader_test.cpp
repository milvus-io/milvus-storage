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

#include <map>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/testing/gtest_util.h>

#include "include/test_env.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage {
namespace {

class MaskedRecordBatchReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    api::Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("masked-record-batch-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  std::shared_ptr<api::Manifest> WriteManifest() { return WriteManifest(schema_, test_batch_); }

  std::shared_ptr<api::Manifest> WriteManifest(const std::shared_ptr<arrow::Schema>& schema,
                                               const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto policy_result = CreateSinglePolicy(LOON_FORMAT_PARQUET, schema);
    if (!policy_result.ok()) {
      ADD_FAILURE() << policy_result.status().ToString();
      return nullptr;
    }

    auto writer = api::Writer::create(base_path_, schema, std::move(policy_result).ValueOrDie(), properties_);
    if (!writer) {
      ADD_FAILURE() << "Writer::create returned nullptr";
      return nullptr;
    }

    auto status = writer->write(batch);
    if (!status.ok()) {
      ADD_FAILURE() << status.ToString();
      return nullptr;
    }

    auto cgs_result = writer->close();
    if (!cgs_result.ok()) {
      ADD_FAILURE() << cgs_result.status().ToString();
      return nullptr;
    }
    return std::make_shared<api::Manifest>(*std::move(cgs_result).ValueOrDie());
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> CreatePkSchema(std::shared_ptr<arrow::DataType> pk_type) {
    return arrow::schema(
        {arrow::field("pk", std::move(pk_type), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("Timestamp", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"1"}))});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateInt64PkBatch(const std::shared_ptr<arrow::Schema>& schema) {
    arrow::Int64Builder pk_builder;
    arrow::Int64Builder ts_builder;
    for (auto value : {1, 2, 3, 4}) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(value));
    }
    for (auto value : {10, 20, 30, 40}) {
      ARROW_RETURN_NOT_OK(ts_builder.Append(value));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    return arrow::RecordBatch::Make(schema, 4, {pk_array, ts_array});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateStringPkBatch(const std::shared_ptr<arrow::Schema>& schema) {
    arrow::StringBuilder pk_builder;
    arrow::Int64Builder ts_builder;
    for (const auto& value : {"a", "b", "c"}) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(value));
    }
    for (auto value : {10, 20, 30}) {
      ARROW_RETURN_NOT_OK(ts_builder.Append(value));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    return arrow::RecordBatch::Make(schema, 3, {pk_array, ts_array});
  }

  arrow::Result<api::ColumnGroupFile> WriteDeltaLog(const std::string& path,
                                                    const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto policy_result = CreateSinglePolicy(LOON_FORMAT_PARQUET, batch->schema());
    ARROW_RETURN_NOT_OK(policy_result.status());
    auto writer =
        api::Writer::create(base_path_ + path, batch->schema(), std::move(policy_result).ValueOrDie(), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Writer::create returned nullptr");
    }
    ARROW_RETURN_NOT_OK(writer->write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
    if (!cgs || cgs->empty() || !(*cgs)[0] || (*cgs)[0]->files.empty()) {
      return arrow::Status::Invalid("Delta writer did not produce a column group file");
    }
    return (*cgs)[0]->files[0];
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateInt64DeltaBatch(
      const std::vector<int64_t>& pks, const std::vector<int64_t>& delete_timestamps) {
    if (pks.size() != delete_timestamps.size()) {
      return arrow::Status::Invalid("pks and delete_timestamps size mismatch");
    }
    auto schema = arrow::schema({arrow::field("pk", arrow::int64(), false), arrow::field("ts", arrow::int64(), false)});
    arrow::Int64Builder pk_builder;
    arrow::Int64Builder ts_builder;
    for (size_t i = 0; i < pks.size(); ++i) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(pks[i]));
      ARROW_RETURN_NOT_OK(ts_builder.Append(delete_timestamps[i]));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(pks.size()), {pk_array, ts_array});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateStringDeltaBatch(
      const std::vector<std::string>& pks, const std::vector<int64_t>& delete_timestamps) {
    if (pks.size() != delete_timestamps.size()) {
      return arrow::Status::Invalid("pks and delete_timestamps size mismatch");
    }
    auto schema = arrow::schema({arrow::field("pk", arrow::utf8(), false), arrow::field("ts", arrow::int64(), false)});
    arrow::StringBuilder pk_builder;
    arrow::Int64Builder ts_builder;
    for (size_t i = 0; i < pks.size(); ++i) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(pks[i]));
      ARROW_RETURN_NOT_OK(ts_builder.Append(delete_timestamps[i]));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(pks.size()), {pk_array, ts_array});
  }

  void AddPkDeltaLog(api::Manifest* manifest,
                     const api::ColumnGroupFile& file,
                     int64_t num_entries,
                     int64_t pk_field_id = 100) {
    ASSERT_NE(manifest, nullptr);
    manifest->deltaLogs().push_back(
        api::DeltaLog{.path = file.path, .type = api::DeltaLogType::PRIMARY_KEY, .num_entries = num_entries});
    manifest->stats()["bloom_filter." + std::to_string(pk_field_id)] = api::Statistics{};
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  std::string base_path_;
};

TEST_F(MaskedRecordBatchReaderTest, ColumnGroupsReaderRejectsMaskedRead) {
  auto manifest = WriteManifest();
  auto column_groups = std::make_shared<ColumnGroups>(manifest->columnGroups());
  auto reader = api::Reader::create(column_groups, schema_, nullptr, properties_);

  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("manifest-aware"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderReturnsAllTrueMaskAndEof) {
  auto manifest = WriteManifest();
  auto reader = api::Reader::create(manifest, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  EXPECT_EQ(masked_batch.keep_mask->length(), masked_batch.batch->num_rows());
  EXPECT_EQ(masked_batch.batch->num_rows(), test_batch_->num_rows());

  for (int64_t i = 0; i < masked_batch.keep_mask->length(); ++i) {
    EXPECT_TRUE(masked_batch.keep_mask->Value(i));
  }

  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  EXPECT_EQ(masked_batch.batch, nullptr);
  EXPECT_EQ(masked_batch.keep_mask, nullptr);
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderAppliesInt64PrimaryKeyDeletes) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::int64()));
  ASSERT_AND_ASSIGN(auto batch, CreateInt64PkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2, 2, 3, 4}, {15, 25, 35, 39}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-int64", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 4);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));  // pk=2 uses max delete ts 25, so row_ts 20 is deleted.
  EXPECT_FALSE(masked_batch.keep_mask->Value(2));
  EXPECT_TRUE(masked_batch.keep_mask->Value(3));  // row_ts 40 is newer than delete_ts 39.
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderAppliesStringPrimaryKeyDeletes) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::utf8()));
  ASSERT_AND_ASSIGN(auto batch, CreateStringPkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateStringDeltaBatch({"b", "c"}, {20, 25}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-string", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 3);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));
}

TEST_F(MaskedRecordBatchReaderTest, PrimaryKeyDeletesRequireTimestampField) {
  auto pk_only_schema = arrow::schema(
      {arrow::field("pk", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  arrow::Int64Builder pk_builder;
  for (auto value : {1, 2, 3}) {
    ASSERT_OK(pk_builder.Append(value));
  }
  ASSERT_AND_ASSIGN(auto pk_array, pk_builder.Finish());
  auto batch = arrow::RecordBatch::Make(pk_only_schema, 3, {pk_array});
  auto manifest = WriteManifest(pk_only_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2}, {20}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-missing-ts", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, pk_only_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Row timestamp field id 1"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, PrimaryKeyDeletesRequirePrimaryKeyStatsMetadata) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::int64()));
  ASSERT_AND_ASSIGN(auto batch, CreateInt64PkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2}, {20}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-missing-pk-stats", delta_batch));
  manifest->deltaLogs().push_back(api::DeltaLog{
      .path = delta_file.path, .type = api::DeltaLogType::PRIMARY_KEY, .num_entries = delta_batch->num_rows()});

  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("bloom_filter.<field_id>"), std::string::npos);
}

}  // namespace
}  // namespace milvus_storage
