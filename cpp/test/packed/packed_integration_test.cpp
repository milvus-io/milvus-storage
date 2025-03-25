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

#include "packed_test_base.h"

namespace milvus_storage {

class PackedIntegrationTest : public PackedTestBase {};

TEST_F(PackedIntegrationTest, TestOneFile) {
  int batch_size = 100;

  auto paths = std::vector<std::string>{path_.string() + "/10000.parquet"};
  auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  PackedRecordBatchReader pr(fs_, paths, schema_, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

TEST_F(PackedIntegrationTest, TestSplitColumnGroup) {
  int batch_size = 1000;

  auto paths = std::vector<std::string>{path_.string() + "/10000.parquet", path_.string() + "/10001.parquet"};
  auto column_groups = std::vector<std::vector<int>>{{2}, {0, 1}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  PackedRecordBatchReader pr(fs_, paths, schema_, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

TEST_F(PackedIntegrationTest, SchemaEvolutionFewerColumns) {
  int batch_size = 1000;

  auto paths = std::vector<std::string>{path_.string() + "/10000.parquet", path_.string() + "/10001.parquet"};
  auto column_groups = std::vector<std::vector<int>>{{2}, {0, 1}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::shared_ptr<arrow::Schema> partial_schema = arrow::schema({schema_->field(0)->Copy(), schema_->field(2)->Copy()});

  PackedRecordBatchReader pr(fs_, paths, partial_schema, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_EQ(table->fields()[0]->name(), schema_->field(0)->name());
  ASSERT_EQ(table->fields()[1]->name(), schema_->field(2)->name());
  ASSERT_EQ(table->schema(), pr.schema());
}

TEST_F(PackedIntegrationTest, SchemaEvolutionMoreColumns) {
  int batch_size = 1000;

  auto paths = std::vector<std::string>{path_.string() + "/10000.parquet", path_.string() + "/10001.parquet"};
  auto column_groups = std::vector<std::vector<int>>{{2}, {0, 1}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::shared_ptr<arrow::Schema> added_field_schema = arrow::schema(
      {schema_->field(1)->Copy(), schema_->field(0)->Copy(),
       arrow::field("float", arrow::float32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"400"})),
       schema_->field(2)->Copy()});

  PackedRecordBatchReader pr(fs_, paths, added_field_schema, reader_memory_);

  std::shared_ptr<RecordBatch> batch;
  int total_size = 0;
  while (true) {
    ASSERT_STATUS_OK(pr.ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    total_size += batch->num_rows();
    ASSERT_EQ(batch->num_columns(), 4);
    ASSERT_EQ(batch->schema()->field(0)->name(), "int64");
    ASSERT_EQ(batch->schema()->field(1)->name(), "int32");
    ASSERT_EQ(batch->schema()->field(2)->name(), "float");
    ASSERT_EQ(batch->schema()->field(3)->name(), "str");
    ASSERT_EQ(batch->column(0)->null_count(), 0);
    ASSERT_EQ(batch->column(1)->null_count(), 0);
    ASSERT_EQ(batch->column(2)->null_count(), batch->num_rows());
    ASSERT_EQ(batch->column(3)->null_count(), 0);
  }
  ASSERT_EQ(total_size, batch_size * 3);
}

}  // namespace milvus_storage