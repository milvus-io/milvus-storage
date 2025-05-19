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
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include <gtest/gtest.h>
#include <cstdint>
#include "arrow/table.h"
#include "milvus-storage/common/type_fwd.h"
#include "test_util.h"

namespace milvus_storage {

class FileReaderTest : public PackedTestBase {};

TEST_F(FileReaderTest, ReadAllColumnsWithEnoughMemory) {
  // read all row groups with enough memory. Only 1 readRowGroups() call.
  SetupOneFile();
  // read all row groups
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Read and validate row counts
  int64_t total_rows = 0;
  std::shared_ptr<arrow::Table> table;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
    ASSERT_EQ(table->num_rows(), row_group_sizes.Get(i).row_num());
    total_rows += table->num_rows();
  }

  // Verify total rows match metadata
  auto metadata = fr.file_metadata()->GetParquetMetadata();
  ASSERT_EQ(total_rows, metadata->num_rows());

  // Verify row group counts
  int64_t expected_rows = 0;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    expected_rows += row_group_sizes.Get(i).row_num();
  }
  ASSERT_EQ(total_rows, expected_rows);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadAllColumnsWithFewerMemory) {
  // read all row groups with fewer memory. Need more than 1 readRowGroups() call.
  SetupOneFile();
  // read all row groups
  size_t fewer_memory = 2 * 1024 * 1024;
  FileRowGroupReader fr(fs_, one_file_path_, schema_, fewer_memory);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Read and validate row counts
  int64_t total_rows = 0;
  std::shared_ptr<arrow::Table> table;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
    ASSERT_EQ(table->num_rows(), row_group_sizes.Get(i).row_num());
    total_rows += table->num_rows();
  }

  // Verify total rows match metadata
  auto metadata = fr.file_metadata()->GetParquetMetadata();
  ASSERT_EQ(total_rows, metadata->num_rows());

  // Verify row group counts
  int64_t expected_rows = 0;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    expected_rows += row_group_sizes.Get(i).row_num();
  }
  ASSERT_EQ(total_rows, expected_rows);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadPartialRowGroup) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(1, 1));

  // Read and validate row counts
  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));

  // Verify row counts match metadata
  auto metadata = fr.file_metadata()->GetParquetMetadata();
  auto row_group_metadata = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_EQ(table->num_rows(), metadata->RowGroup(1)->num_rows());
  ASSERT_EQ(table->num_rows(), row_group_metadata.Get(1).row_num());

  // Verify no more rows to read
  std::shared_ptr<arrow::Table> next_table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&next_table));
  ASSERT_EQ(next_table, nullptr);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, NonExistedRowGroup) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(100, 1).ok());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadNoRowGroup) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(0, 0).ok());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, SchemaEvolutionMoreColumns) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema = arrow::schema(
      {schema_->field(0)->Copy(), schema_->field(1)->Copy(),
       arrow::field("float", arrow::float32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"400"})),
       schema_->field(2)->Copy()});

  FileRowGroupReader fr(fs_, one_file_path_, new_schema, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));

  ASSERT_EQ(table->num_columns(), new_schema->num_fields());
  ASSERT_EQ(table->column(2)->null_count(), table->num_rows());  // Check if extra column has nulls

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, SchemaEvolutionFewerColumns) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema = arrow::schema({schema_->field(1)->Copy(), schema_->field(0)->Copy()});

  FileRowGroupReader fr(fs_, one_file_path_, new_schema, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));

  ASSERT_EQ(table->num_columns(), 2);
  ASSERT_EQ(table->schema()->field(0)->name(), "int64");
  ASSERT_EQ(table->schema()->field(1)->name(), "int32");

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, SchemaEvolutionColumnOrder) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema =
      arrow::schema({schema_->field(2)->Copy(), schema_->field(1)->Copy(), schema_->field(0)->Copy()});

  FileRowGroupReader fr(fs_, one_file_path_, new_schema, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));

  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->schema()->field(0)->name(), "str");
  ASSERT_EQ(table->schema()->field(1)->name(), "int64");
  ASSERT_EQ(table->schema()->field(2)->name(), "int32");

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, RowGroupMetadata) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);

  auto row_group_metadata = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_GT(row_group_metadata.size(), 0);

  int64_t expected_row_offset = 0;
  for (size_t i = 0; i < row_group_metadata.size(); ++i) {
    const auto& metadata = row_group_metadata.Get(i);
    ASSERT_EQ(metadata.row_offset(), expected_row_offset);

    auto parquet_metadata = fr.file_metadata()->GetParquetMetadata();
    ASSERT_EQ(metadata.row_num(), parquet_metadata->RowGroup(i)->num_rows());
    ASSERT_GT(metadata.memory_size(), 0);

    expected_row_offset += metadata.row_num();
  }

  ASSERT_EQ(expected_row_offset, fr.file_metadata()->GetParquetMetadata()->num_rows());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadWithoutSchema) {
  SetupOneFile();

  // Read without providing schema
  FileRowGroupReader fr(fs_, one_file_path_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  int64_t total_rows = 0;
  std::shared_ptr<arrow::Table> table;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
    ASSERT_EQ(table->num_rows(), row_group_sizes.Get(i).row_num());
    total_rows += table->num_rows();
  }

  // Verify schema matches the original file schema
  auto file_schema = fr.schema();
  ASSERT_EQ(file_schema->num_fields(), schema_->num_fields());
  ASSERT_EQ(FieldIDList::Make(file_schema).value(), FieldIDList::Make(schema_).value());

  // Verify data matches
  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader pr(fs_, paths, schema_, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(total_rows, pr_table->num_rows());

  ASSERT_STATUS_OK(fr.Close());
}

}  // namespace milvus_storage