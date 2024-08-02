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
#include "writer/column_group.h"
#include "common/arrow_util.h"

namespace milvus_storage {

class ColumnGroupTest : public ::testing::Test {
  protected:
  std::shared_ptr<arrow::RecordBatch> CreateRecordBatch(int num_columns, int num_rows) {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> schema_fields;

    for (int i = 0; i < num_columns; ++i) {
      std::vector<int32_t> values(num_rows, i);
      auto array_data = arrow::ArrayData::Make(arrow::int32(), num_rows, {nullptr, arrow::Buffer::Wrap(values)});
      arrays.push_back(arrow::MakeArray(array_data));

      std::string column_name = "column_" + std::to_string(i);
      schema_fields.push_back(arrow::field(column_name, arrow::int32()));
    }

    auto schema = std::make_shared<arrow::Schema>(schema_fields);
    return arrow::RecordBatch::Make(schema, num_rows, arrays);
  }
};

TEST_F(ColumnGroupTest, AddAndRetrieveBatches) {
  GroupId group_id = 1;
  ColumnGroup column_group(group_id, {1});

  auto record_batch1 = CreateRecordBatch(3, 5);
  column_group.AddRecordBatch(record_batch1);

  auto record_batch2 = CreateRecordBatch(3, 4);
  column_group.AddRecordBatch(record_batch2);

  ASSERT_EQ(column_group.size(), 2);

  auto retrieved_batch1 = column_group.GetRecordBatch(0);
  ASSERT_EQ(retrieved_batch1->num_columns(), 3);
  ASSERT_EQ(retrieved_batch1->num_rows(), 5);

  auto retrieved_batch2 = column_group.GetRecordBatch(1);
  ASSERT_EQ(retrieved_batch2->num_columns(), 3);
  ASSERT_EQ(retrieved_batch2->num_rows(), 4);
}

TEST_F(ColumnGroupTest, MemoryUsageCalculation) {
  GroupId group_id = 1;
  ColumnGroup column_group(group_id, {1});

  auto record_batch = CreateRecordBatch(3, 5);
  column_group.AddRecordBatch(record_batch);

  size_t expected_memory_usage = 0;
  for (int i = 0; i < record_batch->num_columns(); ++i) {
    expected_memory_usage += GetArrowArrayMemorySize(record_batch->column(i));
  }

  ASSERT_EQ(column_group.GetMemoryUsage(), expected_memory_usage);
}

TEST_F(ColumnGroupTest, CreateTable) {
  GroupId group_id = 1;
  ColumnGroup column_group(group_id, {1});

  auto record_batch = CreateRecordBatch(3, 5);
  column_group.AddRecordBatch(record_batch);

  auto table = column_group.Table();

  // Check the properties of the table
  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), 5);
}

TEST_F(ColumnGroupTest, ColumnGroupMaxHeap) {
  ColumnGroupMaxHeap max_heap;

  ColumnGroup column_group1(1, {0, 1, 2});
  auto record_batch1 = CreateRecordBatch(3, 5);
  column_group1.AddRecordBatch(record_batch1);
  max_heap.push(column_group1);

  ColumnGroup column_group2(2, {0, 1, 2});
  auto record_batch2 = CreateRecordBatch(3, 4);
  column_group2.AddRecordBatch(record_batch2);
  max_heap.push(column_group2);

  ColumnGroup column_group3(3, {0, 1, 2});
  auto record_batch3 = CreateRecordBatch(3, 6);
  column_group3.AddRecordBatch(record_batch3);
  max_heap.push(column_group3);

  ColumnGroup extracted_group = max_heap.top();
  max_heap.pop();
  EXPECT_EQ(extracted_group.group_id(), 3);

  extracted_group = max_heap.top();
  max_heap.pop();
  EXPECT_EQ(extracted_group.group_id(), 1);

  extracted_group = max_heap.top();
  max_heap.pop();
  EXPECT_EQ(extracted_group.group_id(), 2);
}

}  // namespace milvus_storage