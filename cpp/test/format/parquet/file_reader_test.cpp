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

#include "../../packed/packed_test_base.h"
#include "format/parquet/file_reader.h"
namespace milvus_storage {

class FileReaderTest : public PackedTestBase {};

TEST_F(FileReaderTest, FileRecordBatchReader) {
  int batch_size = 100;

  PackedRecordBatchWriter writer(writer_memory_, schema_, fs_, file_path_, storage_config_, props_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
      arrow::field("str", arrow::utf8()),
  };
  auto schema = arrow::schema(fields);

  // exeed row group range, should throw out_of_range
  std::string path = file_path_ + "/0";
  EXPECT_THROW(FileRecordBatchReader fr(*fs_, path, schema, reader_memory_, 100), std::out_of_range);

  // file not exist, should throw runtime_error
  path = file_path_ + "/file_not_exist";
  EXPECT_THROW(FileRecordBatchReader fr(*fs_, path, schema, reader_memory_), std::runtime_error);

  // read all row groups
  path = file_path_ + "/0";
  FileRecordBatchReader fr(*fs_, path, schema, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, fr.ToTable());
  ASSERT_STATUS_OK(fr.Close());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };
  PackedRecordBatchReader pr(*fs_, {path}, schema, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(fr_table->num_rows(), pr_table->num_rows());

  // read row group 1
  path = file_path_ + "/0";
  FileRecordBatchReader rgr(*fs_, path, schema, reader_memory_, 1, 1);
  ASSERT_AND_ARROW_ASSIGN(auto rg_table, rgr.ToTable());
  ASSERT_STATUS_OK(rgr.Close());
  ASSERT_GT(fr_table->num_rows(), rg_table->num_rows());
}

}  // namespace milvus_storage