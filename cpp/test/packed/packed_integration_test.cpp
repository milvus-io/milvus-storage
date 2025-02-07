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

  PackedRecordBatchWriter writer(writer_memory_, schema_, fs_, file_path_, storage_config_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::set<int> needed_columns = {0, 1, 2};

  PackedRecordBatchReader pr(*fs_, file_path_, schema_, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

TEST_F(PackedIntegrationTest, TestSplitColumnGroup) {
  int batch_size = 1000;

  PackedRecordBatchWriter writer(writer_memory_, schema_, fs_, file_path_, storage_config_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::set<int> needed_columns = {0, 1, 2};

  PackedRecordBatchReader pr(*fs_, file_path_, schema_, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

TEST_F(PackedIntegrationTest, TestPartialField) {
  int batch_size = 1000;

  PackedRecordBatchWriter writer(writer_memory_, schema_, fs_, file_path_, storage_config_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::set<int> needed_columns = {0, 2};
  PackedRecordBatchReader pr(*fs_, file_path_, schema_, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_EQ(table->fields()[0], schema_->field(0));
  ASSERT_EQ(table->fields()[1], schema_->field(2));
  ASSERT_EQ(table->schema(), pr.schema());
}

}  // namespace milvus_storage