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
  auto column_offset_mapping = writer.Close();
  EXPECT_EQ(column_offset_mapping->Size(), schema_->num_fields());

  std::vector<ColumnOffset> column_offsets;
  for (auto& field : schema_->fields()) {
    column_offsets.push_back(column_offset_mapping->GetColumnOffset(field->name()));
  }

  std::vector<std::string> paths = {file_path_ + "/0"};

  std::set<int> needed_columns = {0, 1, 2};

  PackedRecordBatchReader pr(*fs_, paths, schema_, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

TEST_F(PackedIntegrationTest, TestSplitColumnGroup) {
  int batch_size = 100;

  PackedRecordBatchWriter writer(writer_memory_, schema_, fs_, file_path_, storage_config_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  auto column_offset_mapping = writer.Close();
  EXPECT_EQ(column_offset_mapping->Size(), schema_->num_fields());

  std::vector<ColumnOffset> column_offsets;
  for (auto& field : schema_->fields()) {
    column_offsets.push_back(column_offset_mapping->GetColumnOffset(field->name()));
  }

  std::vector<std::string> paths = {file_path_ + "/0", file_path_ + "/1"};

  std::set<int> needed_columns = {0, 1, 2};

  PackedRecordBatchReader pr(*fs_, paths, schema_, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  ValidateTableData(table);
}

}  // namespace milvus_storage