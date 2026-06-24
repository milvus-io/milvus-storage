// Copyright 2026 Zilliz
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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/builder.h>

#include "delete_evaluator.h"
#include "include/test_env.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage {
namespace {

class PredicateDeltaSqlTest : public ::testing::Test {
  protected:
  void SetUp() override {
    api::Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("predicate-delta-sql-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  std::shared_ptr<arrow::Schema> DataSchema(std::string timestamp_field_name = "Timestamp") {
    return arrow::schema(
        {arrow::field("pk", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field(std::move(timestamp_field_name), arrow::int64(), false,
                      arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"1"})),
         arrow::field("value", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
         arrow::field("tag", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"}))});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> DataBatch(const std::shared_ptr<arrow::Schema>& schema,
                                                               const std::vector<int64_t>& pks = {1, 2, 3}) {
    arrow::Int64Builder pk_builder;
    arrow::Int64Builder ts_builder;
    arrow::Int64Builder value_builder;
    arrow::StringBuilder tag_builder;
    for (auto pk : pks) {
      ARROW_RETURN_NOT_OK(pk_builder.Append(pk));
      ARROW_RETURN_NOT_OK(ts_builder.Append(pk * 10));
      ARROW_RETURN_NOT_OK(value_builder.Append(pk * 100));
      ARROW_RETURN_NOT_OK(tag_builder.Append("tag_" + std::to_string(pk)));
    }
    ARROW_ASSIGN_OR_RAISE(auto pk_array, pk_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto ts_array, ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto value_array, value_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto tag_array, tag_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(pks.size()),
                                    {pk_array, ts_array, value_array, tag_array});
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> PredicateDeltaBatch(
      const std::vector<int64_t>& delete_timestamps,
      const std::vector<std::string>& predicate_sqls,
      const std::shared_ptr<arrow::DataType>& predicate_sql_type = arrow::utf8(),
      const std::string& predicate_field_name = "predicate",
      const std::string& delete_ts_field_name = "delete_timestamp") {
    if (delete_timestamps.size() != predicate_sqls.size()) {
      return arrow::Status::Invalid("delete_timestamps and predicate_sqls size mismatch");
    }
    // Column order matches the V3 contract: predicate (0), delete_timestamp (1).
    auto schema = arrow::schema({arrow::field(predicate_field_name, predicate_sql_type, false),
                                 arrow::field(delete_ts_field_name, arrow::int64(), false)});
    arrow::Int64Builder delete_ts_builder;
    std::shared_ptr<arrow::Array> predicate_sql_array;
    for (auto ts : delete_timestamps) {
      ARROW_RETURN_NOT_OK(delete_ts_builder.Append(ts));
    }
    if (predicate_sql_type->id() == arrow::Type::STRING) {
      arrow::StringBuilder builder;
      for (const auto& predicate_sql : predicate_sqls) {
        ARROW_RETURN_NOT_OK(builder.Append(predicate_sql));
      }
      ARROW_RETURN_NOT_OK(builder.Finish(&predicate_sql_array));
    } else if (predicate_sql_type->id() == arrow::Type::BINARY) {
      arrow::BinaryBuilder builder;
      for (const auto& predicate_sql : predicate_sqls) {
        ARROW_RETURN_NOT_OK(builder.Append(predicate_sql));
      }
      ARROW_RETURN_NOT_OK(builder.Finish(&predicate_sql_array));
    } else {
      arrow::Int64Builder builder;
      for (size_t i = 0; i < predicate_sqls.size(); ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(static_cast<int64_t>(i)));
      }
      ARROW_RETURN_NOT_OK(builder.Finish(&predicate_sql_array));
    }
    ARROW_ASSIGN_OR_RAISE(auto delete_ts_array, delete_ts_builder.Finish());
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(delete_timestamps.size()),
                                    {predicate_sql_array, delete_ts_array});
  }

  arrow::Result<std::shared_ptr<arrow::Array>> OneRowArray(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
      case arrow::Type::INT64: {
        arrow::Int64Builder builder;
        ARROW_RETURN_NOT_OK(builder.Append(20));
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
      }
      case arrow::Type::INT32: {
        arrow::Int32Builder builder;
        ARROW_RETURN_NOT_OK(builder.Append(20));
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
      }
      case arrow::Type::STRING: {
        arrow::StringBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Append("value > 100"));
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
      }
      case arrow::Type::BINARY: {
        arrow::BinaryBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Append("value > 100"));
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
      }
      default:
        return arrow::Status::Invalid("Unsupported test array type: ", type->ToString());
    }
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> OneRowBatch(const std::shared_ptr<arrow::Schema>& schema) {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(schema->num_fields());
    for (int i = 0; i < schema->num_fields(); ++i) {
      ARROW_ASSIGN_OR_RAISE(auto array, OneRowArray(schema->field(i)->type()));
      arrays.push_back(std::move(array));
    }
    return arrow::RecordBatch::Make(schema, 1, std::move(arrays));
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> PredicateDeltaBatchWithNullPredicateSql() {
    auto schema = arrow::schema(
        {arrow::field("predicate", arrow::utf8(), false), arrow::field("delete_timestamp", arrow::int64(), false)});
    arrow::Int64Builder delete_ts_builder;
    arrow::StringBuilder predicate_sql_builder;
    ARROW_RETURN_NOT_OK(delete_ts_builder.Append(20));
    ARROW_RETURN_NOT_OK(predicate_sql_builder.AppendNull());
    ARROW_ASSIGN_OR_RAISE(auto delete_ts_array, delete_ts_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto predicate_sql_array, predicate_sql_builder.Finish());
    return arrow::RecordBatch::Make(schema, 1, {predicate_sql_array, delete_ts_array});
  }

  arrow::Result<api::ColumnGroupFile> WriteBatchAsDeltaLog(const std::string& path,
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

  arrow::Result<std::shared_ptr<api::Manifest>> ManifestWithPredicateDelta(
      const std::shared_ptr<arrow::Schema>& data_schema,
      const std::shared_ptr<arrow::RecordBatch>& delta_batch,
      api::DeltaLogType type = api::DeltaLogType::PREDICATE,
      int64_t num_entries_override = -1) {
    ARROW_ASSIGN_OR_RAISE(auto data_batch, DataBatch(data_schema));
    auto policy_result = CreateSinglePolicy(LOON_FORMAT_PARQUET, data_schema);
    ARROW_RETURN_NOT_OK(policy_result.status());
    auto writer = api::Writer::create(base_path_, data_schema, std::move(policy_result).ValueOrDie(), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Writer::create returned nullptr");
    }
    ARROW_RETURN_NOT_OK(writer->write(data_batch));
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
    auto manifest = std::make_shared<api::Manifest>(*cgs);

    if (num_entries_override != 0) {
      ARROW_ASSIGN_OR_RAISE(auto delta_file, WriteBatchAsDeltaLog("/predicate-delta", delta_batch));
      manifest->deltaLogs().push_back(
          api::DeltaLog{.path = delta_file.path,
                        .type = type,
                        .num_entries = num_entries_override >= 0 ? num_entries_override : delta_batch->num_rows()});
    } else {
      manifest->deltaLogs().push_back(api::DeltaLog{.path = "unused-predicate-delta", .type = type, .num_entries = 0});
    }
    return manifest;
  }

  arrow::Result<std::shared_ptr<api::DeleteEvaluator>> CreateEvaluator(const std::shared_ptr<api::Manifest>& manifest,
                                                                       const std::shared_ptr<arrow::Schema>& schema,
                                                                       api::AliveReadOptions options = {}) {
    return api::CreateDeleteEvaluator(manifest, schema, properties_, options, nullptr);
  }

  std::vector<std::string> NeededColumns(const std::shared_ptr<api::DeleteEvaluator>& evaluator) {
    return api::DeleteEvaluatorNeededColumns(evaluator);
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

TEST_F(PredicateDeltaSqlTest, LoadsPredicateRowsAndReportsNeededColumns) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"value > 100 and tag = 'tag_2'"}));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));

  ASSERT_AND_ASSIGN(auto evaluator, CreateEvaluator(manifest, schema));
  auto needed_columns = NeededColumns(evaluator);
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "Timestamp"), needed_columns.end());
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "value"), needed_columns.end());
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "tag"), needed_columns.end());
}

TEST_F(PredicateDeltaSqlTest, RejectsBinaryPredicateColumn) {
  auto schema = DataSchema();
  // The predicate column must be string; binary is rejected.
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"value > 100"}, arrow::binary()));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));
  EXPECT_FALSE(CreateEvaluator(manifest, schema).ok());
}

TEST_F(PredicateDeltaSqlTest, FiltersRowsByVisibleUntilTimestampBeforeCompiling) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({20, 30}, {"value > 100", "missing = 1"}));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));

  api::AliveReadOptions options;
  options.visible_until_ts = 20;
  ASSERT_AND_ASSIGN(auto evaluator, CreateEvaluator(manifest, schema, options));
  auto needed_columns = NeededColumns(evaluator);
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "value"), needed_columns.end());
  EXPECT_EQ(std::find(needed_columns.begin(), needed_columns.end(), "missing"), needed_columns.end());
}

TEST_F(PredicateDeltaSqlTest, ResolvesTimestampByFieldIdMetadataNotName) {
  auto schema = DataSchema("row_time");
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"value > 100"}));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));

  ASSERT_AND_ASSIGN(auto evaluator, CreateEvaluator(manifest, schema));
  auto needed_columns = NeededColumns(evaluator);
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "row_time"), needed_columns.end());
  EXPECT_EQ(std::find(needed_columns.begin(), needed_columns.end(), "Timestamp"), needed_columns.end());
}

TEST_F(PredicateDeltaSqlTest, IgnoresEmptyPredicateDeltaLog) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"missing = 1"}));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch, api::DeltaLogType::PREDICATE, 0));

  ASSERT_AND_ASSIGN(auto evaluator, CreateEvaluator(manifest, schema));
  EXPECT_TRUE(NeededColumns(evaluator).empty());
}

TEST_F(PredicateDeltaSqlTest, RejectsMissingAndWrongPredicateSchema) {
  auto schema = DataSchema();
  // Columns are validated by position (0 = predicate string/binary, 1 =
  // delete_timestamp int64); names are irrelevant. Each schema below is invalid.
  const std::vector<std::shared_ptr<arrow::Schema>> invalid_schemas = {
      // single column
      arrow::schema({arrow::field("predicate", arrow::utf8(), false)}),
      // delete_timestamp (column 1) wrong type
      arrow::schema(
          {arrow::field("predicate", arrow::utf8(), false), arrow::field("delete_timestamp", arrow::int32(), false)}),
      // single column
      arrow::schema({arrow::field("delete_timestamp", arrow::int64(), false)}),
      // predicate (column 0) wrong type
      arrow::schema(
          {arrow::field("predicate", arrow::int64(), false), arrow::field("delete_timestamp", arrow::int64(), false)}),
      // delete_timestamp (column 1) nullable
      arrow::schema(
          {arrow::field("predicate", arrow::utf8(), false), arrow::field("delete_timestamp", arrow::int64(), true)}),
      // predicate (column 0) nullable
      arrow::schema(
          {arrow::field("predicate", arrow::utf8(), true), arrow::field("delete_timestamp", arrow::int64(), false)}),
  };

  for (size_t i = 0; i < invalid_schemas.size(); ++i) {
    ASSERT_AND_ASSIGN(auto invalid_batch, OneRowBatch(invalid_schemas[i]));
    ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, invalid_batch));
    auto result = CreateEvaluator(manifest, schema);
    EXPECT_FALSE(result.ok()) << i;
  }
}

TEST_F(PredicateDeltaSqlTest, ReadsPredicateDeltaByPositionIgnoringColumnNames) {
  auto schema = DataSchema();
  // Mirror the V3/packed writer, whose delta columns are not named
  // "predicate"/"delete_timestamp"; the reader must locate them by position
  // (0 = predicate, 1 = delete_timestamp), not by name.
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"value > 100"}, arrow::utf8(), "0", "1"));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));

  ASSERT_AND_ASSIGN(auto evaluator, CreateEvaluator(manifest, schema));
  auto needed_columns = NeededColumns(evaluator);
  EXPECT_NE(std::find(needed_columns.begin(), needed_columns.end(), "value"), needed_columns.end());
}

TEST_F(PredicateDeltaSqlTest, RejectsNullPredicateSqlValue) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto null_sql_batch, PredicateDeltaBatchWithNullPredicateSql());
  ASSERT_AND_ASSIGN(auto null_sql_manifest, ManifestWithPredicateDelta(schema, null_sql_batch));
  EXPECT_FALSE(CreateEvaluator(null_sql_manifest, schema).ok());
}

TEST_F(PredicateDeltaSqlTest, RejectsMalformedPredicateSqlAndUnknownColumns) {
  auto schema = DataSchema();
  const std::vector<std::string> invalid_predicates = {"value >", "missing = 1", ""};
  for (const auto& predicate_sql : invalid_predicates) {
    ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {predicate_sql}));
    ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));
    auto result = CreateEvaluator(manifest, schema);
    EXPECT_FALSE(result.ok()) << predicate_sql;
  }
}

TEST_F(PredicateDeltaSqlTest, RejectsNegativeDeleteTimestamp) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({-1}, {"value > 100"}));
  ASSERT_AND_ASSIGN(auto manifest, ManifestWithPredicateDelta(schema, delta_batch));

  auto result = CreateEvaluator(manifest, schema);
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Delete timestamp must be non-negative"), std::string::npos);
}

TEST_F(PredicateDeltaSqlTest, LoadsOnlyManifestPredicateDeltaType) {
  auto schema = DataSchema();
  ASSERT_AND_ASSIGN(auto delta_batch, PredicateDeltaBatch({25}, {"value > 100"}));
  ASSERT_AND_ASSIGN(auto equality_manifest,
                    ManifestWithPredicateDelta(schema, delta_batch, api::DeltaLogType::EQUALITY));
  auto equality_result = CreateEvaluator(equality_manifest, schema);
  ASSERT_FALSE(equality_result.ok());
  EXPECT_NE(equality_result.status().ToString().find("Unsupported delta log"), std::string::npos);
}

}  // namespace
}  // namespace milvus_storage
