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
#include <cstdint>
#include <memory>
#include <string>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>

#include "test_env.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class PredicatePushdownTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("predicate-pushdown-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    format_ = GetParam();

    // Build schema: age (int64), name (utf8)
    schema_ = arrow::schema({
        arrow::field("age", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),
        arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"101"})),
    });

    // Build test data: 100 rows, age=0..99, name="name_0".."name_99"
    arrow::Int64Builder age_builder;
    arrow::StringBuilder name_builder;
    for (int64_t i = 0; i < kNumRows; ++i) {
      ASSERT_TRUE(age_builder.Append(i).ok());
      ASSERT_TRUE(name_builder.Append("name_" + std::to_string(i)).ok());
    }
    std::shared_ptr<arrow::Array> age_array, name_array;
    ASSERT_TRUE(age_builder.Finish(&age_array).ok());
    ASSERT_TRUE(name_builder.Finish(&name_array).ok());
    test_batch_ = arrow::RecordBatch::Make(schema_, kNumRows, {age_array, name_array});

    // Write data using the Writer API
    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format_, schema_));
    auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
    ASSERT_NE(writer, nullptr);
    ASSERT_TRUE(writer->write(test_batch_).ok());
    auto cgs_result = writer->close();
    ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
    cgs_ = std::move(cgs_result).ValueOrDie();
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  int64_t CountRowsWithPredicate(const std::string& predicate) {
    auto reader = Reader::create(cgs_, schema_, nullptr, properties_);
    auto batch_reader_result = reader->get_record_batch_reader(predicate);
    if (!batch_reader_result.ok()) {
      ADD_FAILURE() << "get_record_batch_reader failed: " << batch_reader_result.status().ToString();
      return -1;
    }
    auto batch_reader = std::move(batch_reader_result).ValueOrDie();

    int64_t total_rows = 0;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      auto status = batch_reader->ReadNext(&batch);
      if (!status.ok()) {
        ADD_FAILURE() << "ReadNext failed: " << status.ToString();
        return -1;
      }
      if (batch == nullptr) {
        break;
      }
      total_rows += batch->num_rows();
    }
    return total_rows;
  }

  static constexpr int64_t kNumRows = 100;
  std::string format_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  std::shared_ptr<ColumnGroups> cgs_;
  Properties properties_;
};

// age > 90 -> rows 91..99 = 9 rows (vortex only)
TEST_P(PredicatePushdownTest, GreaterThanFilter) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("age > 90"), 9);
}

// age >= 20 AND age < 30 -> 10 rows (vortex only)
TEST_P(PredicatePushdownTest, RangeFilter) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("age >= 20 AND age < 30"), 10);
}

// age = 42 -> 1 row (vortex only)
TEST_P(PredicatePushdownTest, EqualsFilter) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("age = 42"), 1);
}

// age IN (10, 20, 30) -> 3 rows (vortex only)
TEST_P(PredicatePushdownTest, InFilter) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("age IN (10, 20, 30)"), 3);
}

// name = 'name_50' -> 1 row (vortex only)
TEST_P(PredicatePushdownTest, StringFilter) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("name = 'name_50'"), 1);
}

// empty predicate -> all 100 rows (both formats)
TEST_P(PredicatePushdownTest, NoPredicateReturnsAll) { EXPECT_EQ(CountRowsWithPredicate(""), kNumRows); }

// age > 1000 -> 0 rows (vortex only)
TEST_P(PredicatePushdownTest, NoMatchReturnsEmpty) {
  if (format_ != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Predicate pushdown only supported for vortex format";
  }
  EXPECT_EQ(CountRowsWithPredicate("age > 1000"), 0);
}

// parquet ignores predicate -> still returns all 100 rows
TEST_P(PredicatePushdownTest, ParquetIgnoresPredicate) {
  if (format_ != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "This test is parquet-specific";
  }
  EXPECT_EQ(CountRowsWithPredicate("age > 90"), kNumRows);
}

INSTANTIATE_TEST_SUITE_P(Formats, PredicatePushdownTest, ::testing::Values(LOON_FORMAT_VORTEX, LOON_FORMAT_PARQUET));

}  // namespace milvus_storage::test
