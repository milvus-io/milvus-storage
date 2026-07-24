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

#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/testing/gtest_util.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

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
      const std::vector<int64_t>& pks,
      const std::vector<int64_t>& delete_timestamps,
      std::string pk_column_name = "pk",
      std::string ts_column_name = "ts") {
    if (pks.size() != delete_timestamps.size()) {
      return arrow::Status::Invalid("pks and delete_timestamps size mismatch");
    }
    auto schema = arrow::schema({arrow::field(std::move(pk_column_name), arrow::int64(), false),
                                 arrow::field(std::move(ts_column_name), arrow::int64(), false)});
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

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreatePredicateDeltaBatch(
      const std::vector<int64_t>& delete_timestamps, const std::vector<std::string>& predicate_sqls) {
    if (delete_timestamps.size() != predicate_sqls.size()) {
      return arrow::Status::Invalid("delete_timestamps and predicate_sqls size mismatch");
    }
    // Column order matches the V3 contract: predicate (0), delete_timestamp (1).
    auto schema = arrow::schema(
        {arrow::field("predicate", arrow::utf8(), false), arrow::field("delete_timestamp", arrow::int64(), false)});
    arrow::Int64Builder delete_ts_builder;
    arrow::StringBuilder predicate_sql_builder;
    for (size_t i = 0; i < delete_timestamps.size(); ++i) {
      ARROW_RETURN_NOT_OK(delete_ts_builder.Append(delete_timestamps[i]));
      ARROW_RETURN_NOT_OK(predicate_sql_builder.Append(predicate_sqls[i]));
    }
    ARROW_ASSIGN_OR_RAISE(auto delete_ts_array, delete_ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto predicate_sql_array, predicate_sql_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(delete_timestamps.size()),
                                    {predicate_sql_array, delete_ts_array});
  }

  void AddPkDeltaLog(api::Manifest* manifest, const api::ColumnGroupFile& file, int64_t num_entries) {
    ASSERT_NE(manifest, nullptr);
    manifest->deltaLogs().push_back(
        api::DeltaLog{.path = file.path, .type = api::DeltaLogType::PRIMARY_KEY, .num_entries = num_entries});
  }

  void AddPredicateDeltaLog(api::Manifest* manifest, const api::ColumnGroupFile& file, int64_t num_entries) {
    ASSERT_NE(manifest, nullptr);
    manifest->deltaLogs().push_back(
        api::DeltaLog{.path = file.path, .type = api::DeltaLogType::PREDICATE, .num_entries = num_entries});
  }

  void ExpectRowMask(const std::shared_ptr<arrow::BooleanArray>& keep_mask, const std::vector<bool>& expected) {
    ASSERT_NE(keep_mask, nullptr);
    ASSERT_EQ(keep_mask->length(), static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < keep_mask->length(); ++i) {
      EXPECT_EQ(keep_mask->Value(i), expected[static_cast<size_t>(i)]) << "row " << i;
    }
  }

  void ExpectInt64Column(const std::shared_ptr<arrow::RecordBatch>& batch,
                         const std::string& column,
                         const std::vector<int64_t>& expected) {
    ASSERT_NE(batch, nullptr);
    auto array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->GetColumnByName(column));
    ASSERT_NE(array, nullptr) << column;
    ASSERT_EQ(array->length(), static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < array->length(); ++i) {
      ASSERT_FALSE(array->IsNull(i)) << column << " row " << i;
      EXPECT_EQ(array->Value(i), expected[static_cast<size_t>(i)]) << column << " row " << i;
    }
  }

  void ExpectStringColumn(const std::shared_ptr<arrow::RecordBatch>& batch,
                          const std::string& column,
                          const std::vector<std::string>& expected) {
    ASSERT_NE(batch, nullptr);
    auto array = std::dynamic_pointer_cast<arrow::StringArray>(batch->GetColumnByName(column));
    ASSERT_NE(array, nullptr) << column;
    ASSERT_EQ(array->length(), static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < array->length(); ++i) {
      ASSERT_FALSE(array->IsNull(i)) << column << " row " << i;
      EXPECT_EQ(array->GetString(i), expected[static_cast<size_t>(i)]) << column << " row " << i;
    }
  }

  void ExpectMaskedReaderEof(const std::shared_ptr<api::MaskedRecordBatchReader>& reader) {
    api::MaskedRecordBatch eof_batch;
    ASSERT_OK(reader->ReadNext(&eof_batch));
    EXPECT_EQ(eof_batch.batch, nullptr);
    EXPECT_EQ(eof_batch.keep_mask, nullptr);
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  std::string base_path_;
};

class MaskedRecordBatchReaderPredicateSqlTest : public MaskedRecordBatchReaderTest {
  protected:
  std::shared_ptr<arrow::Schema> CreatePredicateSchema() {
    return arrow::schema(
        {arrow::field("pk", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("Timestamp", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"1"})),
         arrow::field("value", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
         arrow::field("tag", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
         arrow::field("nullable_value", arrow::int64(), true,
                      arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"103"}))});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreatePredicateDataBatch(
      const std::shared_ptr<arrow::Schema>& schema) {
    const std::vector<int64_t> pks = {1, 2, 3, 4, 5};
    const std::vector<int64_t> timestamps = {10, 20, 30, 40, 50};
    const std::vector<int64_t> values = {50, 150, 250, 350, 450};
    const std::vector<std::string> tags = {"keep", "drop", "drop", "keep", "drop"};
    const std::vector<int64_t> nullable_values = {0, 5, 0, 7, 5};
    const std::vector<bool> nullable_valid = {false, true, false, true, true};

    arrow::Int64Builder pk_builder;
    arrow::Int64Builder ts_builder;
    arrow::Int64Builder value_builder;
    arrow::StringBuilder tag_builder;
    arrow::Int64Builder nullable_builder;
    for (size_t i = 0; i < pks.size(); ++i) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(pks[i]));
      ARROW_RETURN_NOT_OK(ts_builder.Append(timestamps[i]));
      ARROW_RETURN_NOT_OK(value_builder.Append(values[i]));
      ARROW_RETURN_NOT_OK(tag_builder.Append(tags[i]));
      if (nullable_valid[i]) {
        ARROW_RETURN_NOT_OK(nullable_builder.Append(nullable_values[i]));
      } else {
        ARROW_RETURN_NOT_OK(nullable_builder.AppendNull());
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto value_array, value_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto tag_array, tag_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto nullable_array, nullable_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(pks.size()),
                                    {pk_array, ts_array, value_array, tag_array, nullable_array});
  }

  void ExpectOutputColumns(const std::shared_ptr<arrow::RecordBatch>& batch,
                           const std::vector<std::string>& expected_columns) {
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->num_columns(), static_cast<int>(expected_columns.size()));
    for (int i = 0; i < batch->num_columns(); ++i) {
      EXPECT_EQ(batch->schema()->field(i)->name(), expected_columns[static_cast<size_t>(i)]);
    }
  }
};

TEST_F(MaskedRecordBatchReaderTest, ColumnGroupsReaderRejectsMaskedRead) {
  auto manifest = WriteManifest();
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest->columnGroups());
  auto reader = api::Reader::create(column_groups, schema_, nullptr, properties_);

  auto result = reader->get_masked_record_batch_reader(api::MaskedReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("manifest-aware"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderReturnsAllTrueMaskAndEof) {
  auto manifest = WriteManifest();
  auto reader = api::Reader::create(manifest, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::MaskedReadOptions{}));

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
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(
                                            api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectInt64Column(masked_batch.batch, "pk", {1, 2, 3, 4});
  ExpectInt64Column(masked_batch.batch, "Timestamp", {10, 20, 30, 40});
  ExpectRowMask(masked_batch.keep_mask, {true, false, false, true});
  ExpectMaskedReaderEof(masked_reader);
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
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(
                                            api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectStringColumn(masked_batch.batch, "pk", {"a", "b", "c"});
  ExpectInt64Column(masked_batch.batch, "Timestamp", {10, 20, 30});
  ExpectRowMask(masked_batch.keep_mask, {true, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, PrimaryKeyDeltaLogReadsByPhysicalColumnOrder) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::int64()));
  ASSERT_AND_ASSIGN(auto batch, CreateInt64PkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2, 3}, {25, 29}, "0", "1"));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-field-id-column-names", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(
                                            api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectRowMask(masked_batch.keep_mask, {true, false, true, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, MaskedReaderRejectsUnsupportedDeltaLogTypes) {
  // POSITIONAL is a known-but-unsupported type; value 2 is the retired EQUALITY slot,
  // which must now be rejected as an unknown type rather than coerced to PRIMARY_KEY.
  for (const auto delta_type : {api::DeltaLogType::POSITIONAL, static_cast<api::DeltaLogType>(2)}) {
    auto manifest = WriteManifest();
    ASSERT_NE(manifest, nullptr);
    manifest->deltaLogs().push_back(api::DeltaLog{.path = "unsupported-delta", .type = delta_type, .num_entries = 1});

    auto reader = api::Reader::create(manifest, schema_, nullptr, properties_);
    ASSERT_NE(reader, nullptr);
    auto result = reader->get_masked_record_batch_reader(api::MaskedReadOptions{});
    ASSERT_FALSE(result.ok());
    const auto message = result.status().ToString();
    EXPECT_TRUE(message.find("Unsupported delta log") != std::string::npos ||
                message.find("Unknown delta log type") != std::string::npos);
  }
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeltaRequiresTimestampField) {
  auto schema = arrow::schema(
      {arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  arrow::Int64Builder id_builder;
  ASSERT_OK(id_builder.AppendValues({1, 2, 3}));
  ASSERT_AND_ASSIGN(auto id_array, id_builder.Finish());
  auto batch = arrow::RecordBatch::Make(schema, 3, {id_array});
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({20}, {"id = 2"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-missing-ts", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Row timestamp field id 1"), std::string::npos);
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
  auto result =
      reader->get_masked_record_batch_reader(api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Row timestamp field id 1"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, PrimaryKeyDeletesRequirePkFieldIdOption) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::int64()));
  ASSERT_AND_ASSIGN(auto batch, CreateInt64PkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2}, {20}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-missing-pk-field-id", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  // No pk_field_id supplied in MaskedReadOptions -> PK delete cannot resolve which
  // schema field is the primary key.
  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::MaskedReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("pk_field_id"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, PrimaryKeyDeletesRejectNegativeDeleteTimestamp) {
  ASSERT_AND_ASSIGN(auto pk_schema, CreatePkSchema(arrow::int64()));
  ASSERT_AND_ASSIGN(auto batch, CreateInt64PkBatch(pk_schema));
  auto manifest = WriteManifest(pk_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreateInt64DeltaBatch({2}, {-1}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-negative-pk-ts", delta_batch));
  AddPkDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, pk_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result =
      reader->get_masked_record_batch_reader(api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Delete timestamp must be non-negative"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, PredicateOnlyDeletesRowsBySqlAndTimestamp) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({35}, {"value > 100 and tag = 'drop'"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-only", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto output_columns = std::make_shared<std::vector<std::string>>(std::initializer_list<std::string>{"pk", "tag"});
  auto reader = api::Reader::create(manifest, schema, output_columns, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader,
                    reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectOutputColumns(masked_batch.batch, {"pk", "tag"});
  ExpectInt64Column(masked_batch.batch, "pk", {1, 2, 3, 4, 5});
  ExpectStringColumn(masked_batch.batch, "tag", {"keep", "drop", "drop", "keep", "drop"});
  ExpectRowMask(masked_batch.keep_mask, {true, false, false, true, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, SamePredicateMultipleTimestampsUseMaxTimestamp) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  // Two delete rows share the same predicate with different timestamps; they are
  // deduplicated to a single `value > 100 AND row_ts <= max(30, 45) = 45`.
  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({30, 45}, {"value > 100", "value > 100"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-dedup", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader,
                    reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  // value>100 matches pk 2..5 (ts 20,30,40,50); row_ts<=45 deletes pk 2,3,4; pk5 (ts 50) survives.
  ExpectRowMask(masked_batch.keep_mask, {true, false, false, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, PrimaryKeyAndPredicateDeletesAreMerged) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto pk_delta_batch, CreateInt64DeltaBatch({4, 5}, {45, 49}));
  ASSERT_AND_ASSIGN(auto pk_delta_file, WriteDeltaLog("/delta-pk-and-predicate-pk", pk_delta_batch));
  AddPkDeltaLog(manifest.get(), pk_delta_file, pk_delta_batch->num_rows());

  ASSERT_AND_ASSIGN(auto predicate_delta_batch, CreatePredicateDeltaBatch({35}, {"value > 100 and tag = 'drop'"}));
  ASSERT_AND_ASSIGN(auto predicate_delta_file, WriteDeltaLog("/delta-pk-and-predicate-sql", predicate_delta_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_delta_file, predicate_delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(
                                            api::MaskedReadOptions{.pk_field_id = 100, .row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectRowMask(masked_batch.keep_mask, {true, false, false, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, MultiplePredicateRowsUseOrSemantics) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({35, 55}, {"value = 50", "tag = 'drop'"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-or", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader,
                    reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectRowMask(masked_batch.keep_mask, {false, false, false, true, false});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, NullablePredicateResultDoesNotDeleteUnknownRows) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({100}, {"nullable_value = 5"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-nullable", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader,
                    reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectRowMask(masked_batch.keep_mask, {true, false, true, true, false});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderPredicateSqlTest, UnsupportedPredicateSqlFailsReaderCreation) {
  auto schema = CreatePredicateSchema();
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(schema));
  auto manifest = WriteManifest(schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto delta_batch, CreatePredicateDeltaBatch({100}, {"missing = 1"}));
  ASSERT_AND_ASSIGN(auto delta_file, WriteDeltaLog("/delta-predicate-unsupported", delta_batch));
  AddPredicateDeltaLog(manifest.get(), delta_file, delta_batch->num_rows());

  auto reader = api::Reader::create(manifest, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::MaskedReadOptions{.row_timestamp_field_id = 1});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Unknown column"), std::string::npos);
}

}  // namespace
}  // namespace milvus_storage
