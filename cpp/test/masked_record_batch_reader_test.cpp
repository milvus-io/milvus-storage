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


void AppendVarint(uint64_t value, std::string* out) {
  while (value >= 0x80) {
    out->push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out->push_back(static_cast<char>(value));
}

void AppendTag(int field_number, int wire_type, std::string* out) {
  AppendVarint((static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type), out);
}

void AppendVarintField(int field_number, uint64_t value, std::string* out) {
  AppendTag(field_number, 0, out);
  AppendVarint(value, out);
}

void AppendBytesField(int field_number, const std::string& value, std::string* out) {
  AppendTag(field_number, 2, out);
  AppendVarint(value.size(), out);
  out->append(value);
}

void AppendFixed64Field(int field_number, double value, std::string* out) {
  uint64_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  AppendTag(field_number, 1, out);
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<char>((raw >> (i * 8)) & 0xFF));
  }
}

std::string PlanColumnInfo(int64_t field_id) {
  std::string out;
  AppendVarintField(1, static_cast<uint64_t>(field_id), &out);
  return out;
}

std::string PlanBoolValue(bool value) {
  std::string out;
  AppendVarintField(1, value ? 1 : 0, &out);
  return out;
}

std::string PlanInt64Value(int64_t value) {
  std::string out;
  AppendVarintField(2, static_cast<uint64_t>(value), &out);
  return out;
}

std::string PlanStringValue(const std::string& value) {
  std::string out;
  AppendBytesField(4, value, &out);
  return out;
}

std::string PlanDoubleValue(double value) {
  std::string out;
  AppendFixed64Field(3, value, &out);
  return out;
}

std::string PlanUnaryRangeExpr(int64_t field_id, int op, const std::string& value) {
  std::string unary_range;
  AppendBytesField(1, PlanColumnInfo(field_id), &unary_range);
  AppendVarintField(2, op, &unary_range);
  AppendBytesField(3, value, &unary_range);

  std::string expr;
  AppendBytesField(5, unary_range, &expr);
  return expr;
}

std::string PlanTermExpr(int64_t field_id, const std::vector<std::string>& values) {
  std::string term;
  AppendBytesField(1, PlanColumnInfo(field_id), &term);
  for (const auto& value : values) {
    AppendBytesField(2, value, &term);
  }

  std::string expr;
  AppendBytesField(1, term, &expr);
  return expr;
}

std::string PlanBinaryRangeExpr(int64_t field_id,
                                bool lower_inclusive,
                                bool upper_inclusive,
                                const std::string& lower_value,
                                const std::string& upper_value) {
  std::string binary_range;
  AppendBytesField(1, PlanColumnInfo(field_id), &binary_range);
  AppendVarintField(2, lower_inclusive ? 1 : 0, &binary_range);
  AppendVarintField(3, upper_inclusive ? 1 : 0, &binary_range);
  AppendBytesField(4, lower_value, &binary_range);
  AppendBytesField(5, upper_value, &binary_range);

  std::string expr;
  AppendBytesField(6, binary_range, &expr);
  return expr;
}

std::string PlanNullExpr(int64_t field_id, int op) {
  std::string null_expr;
  AppendBytesField(1, PlanColumnInfo(field_id), &null_expr);
  AppendVarintField(2, op, &null_expr);

  std::string expr;
  AppendBytesField(15, null_expr, &expr);
  return expr;
}

std::string PlanAlwaysTrueExpr() {
  std::string expr;
  AppendBytesField(12, std::string{}, &expr);
  return expr;
}

std::string PlanUnaryExpr(int op, const std::string& child) {
  std::string unary;
  AppendVarintField(1, op, &unary);
  AppendBytesField(2, child, &unary);

  std::string expr;
  AppendBytesField(2, unary, &expr);
  return expr;
}

std::string PlanBinaryExpr(int op, const std::string& left, const std::string& right) {
  std::string binary;
  AppendVarintField(1, op, &binary);
  AppendBytesField(2, left, &binary);
  AppendBytesField(3, right, &binary);

  std::string expr;
  AppendBytesField(3, binary, &expr);
  return expr;
}

std::string PlanWithQueryPredicate(const std::string& expr) {
  std::string query;
  AppendBytesField(1, expr, &query);

  std::string plan;
  AppendBytesField(4, query, &plan);
  return plan;
}

std::string PlanWithDirectPredicate(const std::string& expr) {
  std::string plan;
  AppendBytesField(2, expr, &plan);
  return plan;
}

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

  arrow::Result<api::ColumnGroupFile> WritePredicateDeltaLog(const std::string& path,
                                                             const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto full_path = base_path_ + path + "/_data/0_predicate.parquet";
    ARROW_RETURN_NOT_OK(fs_->CreateDir(base_path_ + path + "/_data"));
    ARROW_ASSIGN_OR_RAISE(auto sink, fs_->OpenOutputStream(full_path));

    ::parquet::WriterProperties::Builder writer_props_builder;
    writer_props_builder.compression(::parquet::Compression::ZSTD);
    ::parquet::ArrowWriterProperties::Builder arrow_props_builder;
    arrow_props_builder.store_schema();

    ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(batch->schema(), {batch}));
    ARROW_RETURN_NOT_OK(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, batch->num_rows(),
                                                     writer_props_builder.build(), arrow_props_builder.build()));
    ARROW_RETURN_NOT_OK(sink->Close());

    ARROW_ASSIGN_OR_RAISE(auto file_info, fs_->GetFileInfo(full_path));
    return api::ColumnGroupFile{
        .path = full_path,
        .start_index = 0,
        .end_index = batch->num_rows(),
        .properties = {{api::kPropertyFileSize, std::to_string(file_info.size())}},
    };
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


  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreatePredicateBatch(
      const std::vector<int64_t>& delete_timestamps,
      const std::vector<std::string>& serialized_plans,
      bool include_format_metadata = true) {
    if (delete_timestamps.size() != serialized_plans.size()) {
      return arrow::Status::Invalid("predicate event vector size mismatch");
    }
    auto metadata = include_format_metadata
                        ? arrow::key_value_metadata({"milvus.predicate_delta.format_version"}, {"1"})
                        : nullptr;
    auto schema = arrow::schema({arrow::field("source_msg_id", arrow::int64(), false),
                                 arrow::field("delete_timestamp", arrow::int64(), false),
                                 arrow::field("schema_version", arrow::int32(), false),
                                 arrow::field("expr", arrow::utf8(), true),
                                 arrow::field("serialized_expr_plan", arrow::binary(), false)},
                                metadata);
    arrow::Int64Builder source_builder;
    arrow::Int64Builder delete_ts_builder;
    arrow::Int32Builder schema_version_builder;
    arrow::StringBuilder expr_builder;
    arrow::BinaryBuilder plan_builder;
    for (size_t i = 0; i < delete_timestamps.size(); ++i) {
      ARROW_RETURN_NOT_OK(source_builder.Append(static_cast<int64_t>(i + 1)));
      ARROW_RETURN_NOT_OK(delete_ts_builder.Append(delete_timestamps[i]));
      ARROW_RETURN_NOT_OK(schema_version_builder.Append(1));
      ARROW_RETURN_NOT_OK(expr_builder.Append(""));
      ARROW_RETURN_NOT_OK(plan_builder.Append(reinterpret_cast<const uint8_t*>(serialized_plans[i].data()),
                                             static_cast<int32_t>(serialized_plans[i].size())));
    }
    ARROW_ASSIGN_OR_RAISE(auto source_array, source_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto delete_ts_array, delete_ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto schema_version_array, schema_version_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto expr_array, expr_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto plan_array, plan_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(delete_timestamps.size()),
                                    {source_array, delete_ts_array, schema_version_array, expr_array, plan_array});
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> CreatePredicateSchema() {
    return arrow::schema(
        {arrow::field("pk", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("Timestamp", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"1"})),
         arrow::field("age", arrow::int64(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
         arrow::field("city", arrow::utf8(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
         arrow::field("score", arrow::float64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"103"})),
         arrow::field("active", arrow::boolean(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"104"})),
         arrow::field("level", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"105"})),
         arrow::field("ratio", arrow::float32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"106"}))});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreatePredicateDataBatch(
      const std::shared_ptr<arrow::Schema>& schema) {
    arrow::Int64Builder pk_builder;
    arrow::Int64Builder ts_builder;
    arrow::Int64Builder age_builder;
    arrow::StringBuilder city_builder;
    arrow::DoubleBuilder score_builder;
    arrow::BooleanBuilder active_builder;
    arrow::Int32Builder level_builder;
    arrow::FloatBuilder ratio_builder;
    for (auto value : {1, 2, 3, 4, 5}) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(value));
    }
    for (auto value : {10, 20, 30, 40, 50}) {
      ARROW_RETURN_NOT_OK(ts_builder.Append(value));
    }
    ARROW_RETURN_NOT_OK(age_builder.Append(5));
    ARROW_RETURN_NOT_OK(age_builder.Append(15));
    ARROW_RETURN_NOT_OK(age_builder.AppendNull());
    ARROW_RETURN_NOT_OK(age_builder.Append(30));
    ARROW_RETURN_NOT_OK(age_builder.Append(30));
    ARROW_RETURN_NOT_OK(city_builder.Append("beijing"));
    ARROW_RETURN_NOT_OK(city_builder.Append("shanghai"));
    ARROW_RETURN_NOT_OK(city_builder.AppendNull());
    ARROW_RETURN_NOT_OK(city_builder.Append("beijing"));
    ARROW_RETURN_NOT_OK(city_builder.Append("beijing"));
    for (auto value : {0.5, 2.5, 3.5, 4.5, 5.5}) {
      ARROW_RETURN_NOT_OK(score_builder.Append(value));
    }
    for (auto value : {true, false, true, true, false}) {
      ARROW_RETURN_NOT_OK(active_builder.Append(value));
    }
    for (auto value : {1, 2, 3, 4, 5}) {
      ARROW_RETURN_NOT_OK(level_builder.Append(value));
    }
    for (auto value : {0.5F, 1.5F, 2.5F, 3.5F, 4.5F}) {
      ARROW_RETURN_NOT_OK(ratio_builder.Append(value));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto age_array, age_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto city_array, city_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto score_array, score_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto active_array, active_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto level_array, level_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ratio_array, ratio_builder.Finish());
    return arrow::RecordBatch::Make(
        schema, 5, {pk_array, ts_array, age_array, city_array, score_array, active_array, level_array, ratio_array});
  }

  void AddPredicateDeltaLog(api::Manifest* manifest, const api::ColumnGroupFile& file, int64_t num_entries) {
    ASSERT_NE(manifest, nullptr);
    manifest->deltaLogs().push_back(
        api::DeltaLog{.path = file.path, .type = api::DeltaLogType::PREDICATE, .num_entries = num_entries});
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

  void ExpectAliveMask(const std::shared_ptr<arrow::BooleanArray>& keep_mask, const std::vector<bool>& expected) {
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

  void ExpectPredicateOutputBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->num_rows(), 5);
    ExpectInt64Column(batch, "pk", {1, 2, 3, 4, 5});
    ExpectInt64Column(batch, "Timestamp", {10, 20, 30, 40, 50});
    auto city = std::dynamic_pointer_cast<arrow::StringArray>(batch->GetColumnByName("city"));
    ASSERT_NE(city, nullptr);
    ASSERT_EQ(city->length(), 5);
    EXPECT_EQ(city->GetString(0), "beijing");
    EXPECT_EQ(city->GetString(1), "shanghai");
    EXPECT_TRUE(city->IsNull(2));
    EXPECT_EQ(city->GetString(3), "beijing");
    EXPECT_EQ(city->GetString(4), "beijing");
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

TEST_F(MaskedRecordBatchReaderTest, ColumnGroupsReaderRejectsMaskedRead) {
  auto manifest = WriteManifest();
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest->columnGroups());
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
  ExpectInt64Column(masked_batch.batch, "pk", {1, 2, 3, 4});
  ExpectInt64Column(masked_batch.batch, "Timestamp", {10, 20, 30, 40});
  ExpectAliveMask(masked_batch.keep_mask, {true, false, false, true});
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
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ExpectStringColumn(masked_batch.batch, "pk", {"a", "b", "c"});
  ExpectInt64Column(masked_batch.batch, "Timestamp", {10, 20, 30});
  ExpectAliveMask(masked_batch.keep_mask, {true, false, true});
  ExpectMaskedReaderEof(masked_reader);
}


TEST_F(MaskedRecordBatchReaderTest, ManifestReaderAppliesPredicateDeletesWithTimestampAndNullSemantics) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_gt_10 = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(10));
  const auto city_eq_beijing = PlanUnaryRangeExpr(102, 5 /* Equal */, PlanStringValue("beijing"));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({35, 45}, {PlanWithQueryPredicate(age_gt_10), PlanWithQueryPredicate(city_eq_beijing)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ExpectPredicateOutputBatch(masked_batch.batch);
  ExpectAliveMask(masked_batch.keep_mask, {false, false, true, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderSupportsPredicateNullExpr) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_is_null = PlanNullExpr(101, 1 /* IsNull */);
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({35}, {PlanWithQueryPredicate(age_is_null)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-null", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 5);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));
  EXPECT_TRUE(masked_batch.keep_mask->Value(1));
  EXPECT_FALSE(masked_batch.keep_mask->Value(2));
  EXPECT_TRUE(masked_batch.keep_mask->Value(3));
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesCanReadExtraEvaluationColumns) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_gt_10 = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(10));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({35}, {PlanWithQueryPredicate(age_gt_10)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-extra-columns", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto needed_columns = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"pk"});
  auto reader = api::Reader::create(manifest, predicate_schema, needed_columns, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  EXPECT_EQ(masked_batch.batch->num_columns(), 1);
  EXPECT_NE(masked_batch.batch->schema()->GetFieldIndex("pk"), -1);
  EXPECT_EQ(masked_batch.batch->schema()->GetFieldIndex("Timestamp"), -1);
  EXPECT_EQ(masked_batch.batch->schema()->GetFieldIndex("age"), -1);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesRespectVisibleUntilTs) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_gt_10 = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(10));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({35, 45}, {PlanWithQueryPredicate(age_gt_10), PlanWithQueryPredicate(PlanAlwaysTrueExpr())}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-visible-ts", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  api::AliveReadOptions options;
  options.visible_until_ts = 35;
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(options));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 5);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));
  EXPECT_TRUE(masked_batch.keep_mask->Value(3));
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderCombinesPrimaryKeyAndPredicateDeletes) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto pk_delta_batch, CreateInt64DeltaBatch({2}, {25}));
  ASSERT_AND_ASSIGN(auto pk_delta_file, WriteDeltaLog("/delta-pk-and-predicate", pk_delta_batch));
  AddPkDeltaLog(manifest.get(), pk_delta_file, pk_delta_batch->num_rows());

  const auto age_gt_20 = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(20));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({45}, {PlanWithQueryPredicate(age_gt_20)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-with-pk", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ExpectPredicateOutputBatch(masked_batch.batch);
  ExpectAliveMask(masked_batch.keep_mask, {true, false, true, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportLogicalOpsAndAlwaysTrue) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_ge_30 = PlanUnaryRangeExpr(101, 2 /* GreaterEqual */, PlanInt64Value(30));
  const auto city_eq_beijing = PlanUnaryRangeExpr(102, 5 /* Equal */, PlanStringValue("beijing"));
  const auto not_city_beijing = PlanUnaryExpr(1 /* Not */, city_eq_beijing);
  const auto age_and_city = PlanBinaryExpr(1 /* And */, age_ge_30, city_eq_beijing);
  const auto expr = PlanBinaryExpr(2 /* Or */, age_and_city, not_city_beijing);
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({45, 10}, {PlanWithQueryPredicate(expr), PlanWithQueryPredicate(PlanAlwaysTrueExpr())}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-logical", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  EXPECT_FALSE(masked_batch.keep_mask->Value(0));  // AlwaysTrue at delete_ts=10 deletes row_ts=10.
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));  // NOT city == beijing is true.
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));   // city null makes NOT null and AND null, not delete.
  EXPECT_FALSE(masked_batch.keep_mask->Value(3));  // age >= 30 AND city == beijing.
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));   // row_ts 50 is newer than delete_ts 45.
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportDoubleBoolAndNegativeInt64) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto score_gt_4 = PlanUnaryRangeExpr(103, 1 /* GreaterThan */, PlanDoubleValue(4.0));
  const auto active_eq_true = PlanUnaryRangeExpr(104, 5 /* Equal */, PlanBoolValue(true));
  const auto age_gt_negative = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(-1));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({45, 25, 12}, {PlanWithQueryPredicate(score_gt_4), PlanWithQueryPredicate(active_eq_true), PlanWithQueryPredicate(age_gt_negative)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-scalar-types", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  EXPECT_FALSE(masked_batch.keep_mask->Value(0));  // 5 > -1 proves negative int64 literal decoding.
  EXPECT_TRUE(masked_batch.keep_mask->Value(1));   // active=false and score<=4.
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));   // age null yields null, not delete.
  EXPECT_FALSE(masked_batch.keep_mask->Value(3));  // score > 4 and active=true.
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));   // row_ts 50 is newer than delete_ts 45.
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportTermExpr) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto city_in = PlanTermExpr(102, {PlanStringValue("beijing"), PlanStringValue("shanghai")});
  ASSERT_AND_ASSIGN(auto predicate_batch, CreatePredicateBatch({50}, {PlanWithQueryPredicate(city_in)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-term", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 5);
  EXPECT_FALSE(masked_batch.keep_mask->Value(0));  // beijing
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));  // shanghai
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));   // city null yields predicate null, not delete.
  EXPECT_FALSE(masked_batch.keep_mask->Value(3));  // beijing
  EXPECT_FALSE(masked_batch.keep_mask->Value(4));  // shanghai
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportNotInViaUnaryNotTermExpr) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto city_not_in = PlanUnaryExpr(1 /* Not */, PlanTermExpr(102, {PlanStringValue("beijing")}));
  ASSERT_AND_ASSIGN(auto predicate_batch, CreatePredicateBatch({50}, {PlanWithQueryPredicate(city_not_in)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-not-in", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 5);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));    // beijing
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));   // shanghai
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));    // null yields predicate null, not delete.
  EXPECT_TRUE(masked_batch.keep_mask->Value(3));    // beijing
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));    // beijing
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportBinaryRangeExpr) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_between = PlanBinaryRangeExpr(101, true, false, PlanInt64Value(15), PlanInt64Value(30));
  ASSERT_AND_ASSIGN(auto predicate_batch, CreatePredicateBatch({50}, {PlanWithQueryPredicate(age_between)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-binary-range", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  ASSERT_EQ(masked_batch.keep_mask->length(), 5);
  EXPECT_TRUE(masked_batch.keep_mask->Value(0));   // age 5
  EXPECT_FALSE(masked_batch.keep_mask->Value(1));  // age 15, lower inclusive
  EXPECT_TRUE(masked_batch.keep_mask->Value(2));   // age null yields predicate null, not delete.
  EXPECT_TRUE(masked_batch.keep_mask->Value(3));   // age 30, upper exclusive
  EXPECT_TRUE(masked_batch.keep_mask->Value(4));   // age 30, upper exclusive
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesSupportIsNotNullAndDirectPlanPredicate) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto city_is_not_null = PlanNullExpr(102, 2 /* IsNotNull */);
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({45}, {PlanWithDirectPredicate(city_is_not_null)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-is-not-null", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ExpectPredicateOutputBatch(masked_batch.batch);
  ExpectAliveMask(masked_batch.keep_mask, {false, false, true, false, true});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesCoverRemainingCompareOpsAndPhysicalTypes) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_le_5 = PlanUnaryRangeExpr(101, 4 /* LessEqual */, PlanInt64Value(5));
  const auto score_lt_3 = PlanUnaryRangeExpr(103, 3 /* LessThan */, PlanInt64Value(3));
  const auto city_ne_beijing = PlanUnaryRangeExpr(102, 6 /* NotEqual */, PlanStringValue("beijing"));
  const auto active_ne_true = PlanUnaryRangeExpr(104, 6 /* NotEqual */, PlanBoolValue(true));
  const auto level_ge_4 = PlanUnaryRangeExpr(105, 2 /* GreaterEqual */, PlanDoubleValue(4.0));
  const auto ratio_lt_1 = PlanUnaryRangeExpr(106, 3 /* LessThan */, PlanDoubleValue(1.0));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({50, 50, 50, 50, 50, 50},
                                         {PlanWithQueryPredicate(age_le_5), PlanWithQueryPredicate(score_lt_3),
                                          PlanWithQueryPredicate(city_ne_beijing), PlanWithQueryPredicate(active_ne_true),
                                          PlanWithQueryPredicate(level_ge_4), PlanWithQueryPredicate(ratio_lt_1)}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-remaining-branches", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ExpectPredicateOutputBatch(masked_batch.batch);
  ExpectAliveMask(masked_batch.keep_mask, {false, false, true, false, false});
  ExpectMaskedReaderEof(masked_reader);
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesRequireFormatVersionMetadata) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  const auto age_gt_10 = PlanUnaryRangeExpr(101, 1 /* GreaterThan */, PlanInt64Value(10));
  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({35}, {PlanWithQueryPredicate(age_gt_10)}, false /* include_format_metadata */));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-predicate-missing-format", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("format_version=1"), std::string::npos);
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
  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Delete timestamp must be non-negative"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, PredicateDeletesRejectNegativeDeleteTimestamp) {
  ASSERT_AND_ASSIGN(auto predicate_schema, CreatePredicateSchema());
  ASSERT_AND_ASSIGN(auto batch, CreatePredicateDataBatch(predicate_schema));
  auto manifest = WriteManifest(predicate_schema, batch);
  ASSERT_NE(manifest, nullptr);

  ASSERT_AND_ASSIGN(auto predicate_batch,
                    CreatePredicateBatch({-1}, {PlanWithQueryPredicate(PlanAlwaysTrueExpr())}));
  ASSERT_AND_ASSIGN(auto predicate_file, WritePredicateDeltaLog("/delta-negative-predicate-ts", predicate_batch));
  AddPredicateDeltaLog(manifest.get(), predicate_file, predicate_batch->num_rows());

  auto reader = api::Reader::create(manifest, predicate_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Delete timestamp must be non-negative"), std::string::npos);
}

}  // namespace
}  // namespace milvus_storage
