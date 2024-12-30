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
#include "packed/mem_record_reader.h"
#include "packed/async_mem_record_reader.h"

namespace milvus_storage {

class OneFileTest : public PackedTestBase {};

TEST_F(OneFileTest, WriteAndRead) {
  int batch_size = 100;

  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };

  std::vector<std::string> paths = {file_path_ + "/0"};

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
      arrow::field("str", arrow::utf8()),
  };

  TestWriteAndRead(batch_size, paths, fields, column_offsets);
}

TEST_F(OneFileTest, MemRecordBatchReader) {
  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
      arrow::field("str", arrow::utf8()),
  };
  auto schema = arrow::schema(fields);

  // exeed row group range, should throw out_of_range
  std::string path = file_path_ + "/0";
  EXPECT_THROW(MemRecordBatchReader mr(*fs_, path, schema, 100, 1, reader_memory_), std::out_of_range);

  // file not exist, should throw runtime_error
  path = file_path_ + "/file_not_exist";
  EXPECT_THROW(MemRecordBatchReader mr(*fs_, path, schema, 0, 1, reader_memory_), std::runtime_error);

  // read all row groups
  path = file_path_ + "/0";
  MemRecordBatchReader mr(*fs_, path, schema, 0, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto m_table, mr.ToTable());
  ASSERT_STATUS_OK(mr.Close());

  // read all row groups async
  AsyncMemRecordBatchReader amr(*fs_, path, schema, reader_memory_);
  ASSERT_STATUS_OK(amr.Execute());
  auto amr_table = amr.Table();
  ASSERT_EQ(m_table->num_rows(), amr_table->num_rows());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };
  std::vector<std::string> paths = {file_path_ + "/0"};
  PackedRecordBatchReader pr(*fs_, paths, schema, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto p_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(m_table->num_rows(), p_table->num_rows());

  // read row group 1
  path = file_path_ + "/0";
  MemRecordBatchReader mr2(*fs_, path, schema, 1, 1, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto rg_table, mr2.ToTable());
  ASSERT_STATUS_OK(mr.Close());
  ASSERT_GT(m_table->num_rows(), rg_table->num_rows());
}

}  // namespace milvus_storage