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
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/arrow_util.h"
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

TEST_F(FileReaderTest, ReadWithNotEnoughMemory) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, 1024);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_EQ(table->num_rows(), row_group_sizes.Get(0).row_num());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileNotExists) {
  // Test reading non-existent file
  std::string non_existent_path = "/tmp/non_existent_file.parquet";
  EXPECT_THROW(FileRowGroupReader(fs_, non_existent_path, schema_, reader_memory_), std::runtime_error);
}

TEST_F(FileReaderTest, InvalidBufferSize) {
  SetupOneFile();
  // Test with negative buffer size
  FileRowGroupReader fr(fs_, one_file_path_, schema_, -1);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, 1));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_NE(table, nullptr);
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ZeroBufferSize) {
  SetupOneFile();
  // Test with zero buffer size
  FileRowGroupReader fr(fs_, one_file_path_, schema_, 0);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, 1));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_NE(table, nullptr);
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, NegativeRowGroupOffset) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(-1, 1).ok());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, InvalidRowGroupRange) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();

  // Test with offset + count exceeding total row groups
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(row_group_sizes.size(), 1).ok());

  // Test with offset + count > total row groups
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(row_group_sizes.size() - 1, 2).ok());

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadAfterClose) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_STATUS_OK(fr.Close());

  std::shared_ptr<arrow::Table> table;
  // Should handle gracefully after close
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_EQ(table, nullptr);
}

TEST_F(FileReaderTest, MultipleCloseCalls) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  ASSERT_STATUS_OK(fr.Close());
  ASSERT_STATUS_OK(fr.Close());  // Multiple close calls should be safe
}

TEST_F(FileReaderTest, ReadWithoutSettingRowGroupRange) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_EQ(table, nullptr);  // Should return nullptr when no range is set
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ReadAfterAllRowGroups) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Read all row groups
  std::shared_ptr<arrow::Table> table;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
    ASSERT_NE(table, nullptr);
  }

  // Try to read one more time
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_EQ(table, nullptr);  // Should return nullptr when no more row groups

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, SchemaEvolutionWithInvalidFieldID) {
  SetupOneFile();

  // Create schema with invalid field ID (non-numeric)
  auto invalid_field =
      arrow::field("invalid", arrow::int32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"invalid_id"}));
  std::shared_ptr<arrow::Schema> invalid_schema = arrow::schema({invalid_field});

  // Should handle gracefully - expect exception for invalid field ID
  EXPECT_THROW(FileRowGroupReader(fs_, one_file_path_, invalid_schema, reader_memory_), std::invalid_argument);
}

TEST_F(FileReaderTest, SchemaEvolutionWithMissingFieldIDMetadata) {
  SetupOneFile();

  // Create schema without field ID metadata
  auto field_without_id = arrow::field("no_id", arrow::int32(), true);
  std::shared_ptr<arrow::Schema> schema_without_id = arrow::schema({field_without_id});

  // Should handle gracefully - expect exception for missing field ID
  EXPECT_THROW(FileRowGroupReader(fs_, one_file_path_, schema_without_id, reader_memory_), std::runtime_error);
}

TEST_F(FileReaderTest, MemoryPressureWithLargeRowGroups) {
  SetupOneFile();

  // Test with very small memory limit to force multiple reads
  int64_t tiny_memory = 1024;  // 1KB
  FileRowGroupReader fr(fs_, one_file_path_, schema_, tiny_memory);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Should still be able to read all row groups
  int64_t total_rows = 0;
  std::shared_ptr<arrow::Table> table;
  for (int i = 0; i < row_group_sizes.size(); ++i) {
    ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
    ASSERT_NE(table, nullptr);
    total_rows += table->num_rows();
  }

  // Verify total rows match metadata
  auto metadata = fr.file_metadata()->GetParquetMetadata();
  ASSERT_EQ(total_rows, metadata->num_rows());

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ConcurrentReadOperations) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Test that multiple read operations work correctly
  std::shared_ptr<arrow::Table> table1, table2;

  // Read first row group
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table1));
  ASSERT_NE(table1, nullptr);

  // Read second row group
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table2));
  ASSERT_NE(table2, nullptr);

  // Verify both tables have data (they might have same row count if all row groups are same size)
  ASSERT_GT(table1->num_rows(), 0);
  ASSERT_GT(table2->num_rows(), 0);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, ResetRowGroupRange) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();

  // Set initial range
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, 1));
  std::shared_ptr<arrow::Table> table1;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table1));
  ASSERT_NE(table1, nullptr);

  // Reset to different range
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(1, 1));
  std::shared_ptr<arrow::Table> table2;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table2));
  ASSERT_NE(table2, nullptr);

  // Verify both tables have data (they might have same row count if all row groups are same size)
  ASSERT_GT(table1->num_rows(), 0);
  ASSERT_GT(table2->num_rows(), 0);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, EmptyRowGroupRange) {
  SetupOneFile();
  FileRowGroupReader fr(fs_, one_file_path_, schema_, reader_memory_);

  // Test with empty range (should fail)
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(0, 0).ok());

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, SchemaEvolutionWithDuplicateFieldIDs) {
  SetupOneFile();

  // Create schema with duplicate field IDs
  auto field1 = arrow::field("field1", arrow::int32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));
  auto field2 = arrow::field("field2", arrow::int64(), true,
                             arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));  // Same ID as field1
  std::shared_ptr<arrow::Schema> duplicate_schema = arrow::schema({field1, field2});

  // Should handle gracefully (first field with ID 0 will be used)
  FileRowGroupReader fr(fs_, one_file_path_, duplicate_schema, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, 1));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_EQ(table->num_columns(), 2);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, VeryLargeBufferSize) {
  SetupOneFile();

  // Test with very large buffer size
  int64_t huge_memory = INT64_MAX;
  FileRowGroupReader fr(fs_, one_file_path_, schema_, huge_memory);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));

  // Should read all row groups in one go
  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_NE(table, nullptr);

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, NullSchemaPointer) {
  SetupOneFile();

  // Test with null schema pointer (should use file schema)
  FileRowGroupReader fr(fs_, one_file_path_, nullptr, reader_memory_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupMetadataVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, 1));

  std::shared_ptr<arrow::Table> table;
  ASSERT_STATUS_OK(fr.ReadNextRowGroup(&table));
  ASSERT_NE(table, nullptr);

  // Verify schema matches file schema
  auto file_schema = fr.schema();
  ASSERT_EQ(file_schema->num_fields(), schema_->num_fields());

  ASSERT_STATUS_OK(fr.Close());
}

}  // namespace milvus_storage