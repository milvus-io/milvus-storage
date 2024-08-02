// Copyright 2024 Zilliz
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
#include "writer/splitter/indices_based_splitter.h"
#include "writer/column_group.h"

namespace milvus_storage {

class IndicesBasedSplitterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Set up schema
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    ASSERT_TRUE(int_builder.AppendValues({1, 2, 3}).ok());
    ASSERT_TRUE(int64_builder.AppendValues({1, 2, 3}).ok());
    ASSERT_TRUE(str_builder.AppendValues({"a", "b", "c"}).ok());

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> int64_array;
    std::shared_ptr<arrow::Array> str_array;

    ASSERT_TRUE(int_builder.Finish(&int_array).ok());
    ASSERT_TRUE(int64_builder.Finish(&int64_array).ok());
    ASSERT_TRUE(str_builder.Finish(&str_array).ok());

    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field("int", arrow::int32()), arrow::field("int64", arrow::int64()), arrow::field("str", arrow::utf8())};

    schema_ = arrow::schema(fields);

    std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
    record_batch_ = arrow::RecordBatch::Make(schema_, 3, arrays);

    column_indices_ = {{1}, {0, 2}};
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::vector<std::vector<int>> column_indices_;
};

TEST_F(IndicesBasedSplitterTest, SplitColumnsTest) {
  IndicesBasedSplitter splitter(column_indices_);
  std::vector<ColumnGroup> column_groups = splitter.Split(record_batch_);

  ASSERT_EQ(column_groups.size(), 2);

  ASSERT_EQ(column_groups[0].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT64);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT32);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(1)->type()->id(), arrow::Type::STRING);
}

}  // namespace milvus_storage
