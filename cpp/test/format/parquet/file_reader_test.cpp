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

TEST_F(FileReaderTest, FileRecordBatchReader_ReadAllColumns) {
  SetupOneFile();
  // read all row groups
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);
  auto row_group_num = fr.file_metadata()->GetRowGroupMetadataVector().size();
  std::vector<int64_t> all_row_groups(row_group_num);
  std::iota(all_row_groups.begin(), all_row_groups.end(), 0);
  ASSERT_STATUS_OK(fr.SetRowGroups(all_row_groups));
  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));
  auto arrow_schema = fr.schema();
  ASSERT_EQ(arrow_schema->num_fields(), schema_->num_fields());
  ASSERT_EQ(FieldIDList::Make(arrow_schema).value(), FieldIDList::Make(schema_).value());
  ASSERT_STATUS_OK(fr.Close());

  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader pr(fs_, paths, schema_, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(fr_table->num_rows(), pr_table->num_rows());

  // read row group 1
  FileRecordBatchReader rgr(fs_, paths[0], schema_);
  std::vector<int64_t> single_row_group = {1};
  ASSERT_STATUS_OK(rgr.SetRowGroups(single_row_group));
  std::shared_ptr<RecordBatch> rg_batch;
  ASSERT_STATUS_OK(rgr.ReadNext(&rg_batch));
  ASSERT_STATUS_OK(rgr.Close());
  ASSERT_AND_ARROW_ASSIGN(auto rg_table, arrow::Table::FromRecordBatches({rg_batch}));
  ASSERT_GT(fr_table->num_rows(), rg_table->num_rows());
}

TEST_F(FileReaderTest, FileRecordBatchReader_ReadPartialRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);
  std::vector<int64_t> single_row_group = {1};
  ASSERT_STATUS_OK(fr.SetRowGroups(single_row_group));
  std::shared_ptr<RecordBatch> rg_batch;
  ASSERT_STATUS_OK(fr.ReadNext(&rg_batch));
  ASSERT_EQ(fr.file_metadata()->GetParquetMetadata()->RowGroup(1)->num_rows(), rg_batch->num_rows());
  ASSERT_EQ(fr.file_metadata()->GetRowGroupMetadataVector().Get(1).row_num(), rg_batch->num_rows());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_NonExistedRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);
  std::vector<int64_t> invalid_row_group = {100};
  ASSERT_FALSE(fr.SetRowGroups(invalid_row_group).ok());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_ReadNoRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);
  std::vector<int64_t> empty_row_groups;
  ASSERT_STATUS_OK(fr.SetRowGroups(empty_row_groups));
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_ReadNonContinuousRowGroups) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);

  // Get total number of row groups
  auto total_row_groups = fr.file_metadata()->GetRowGroupMetadataVector().size();
  ASSERT_EQ(total_row_groups, 3);

  // Set row groups to read
  std::vector<int64_t> row_groups = {0, 2};
  ASSERT_STATUS_OK(fr.SetRowGroups(row_groups));

  // Read all batches and verify total row count
  std::vector<std::shared_ptr<RecordBatch>> batches;
  std::shared_ptr<RecordBatch> batch;
  int64_t total_rows = 0;
  while (true) {
    ASSERT_STATUS_OK(fr.ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    batches.push_back(batch);
    total_rows += batch->num_rows();
  }
  ASSERT_STATUS_OK(fr.Close());

  // Verify total row count matches sum of selected row groups
  int64_t expected_total_rows = 0;
  for (int64_t rg : row_groups) {
    expected_total_rows += fr.file_metadata()->GetRowGroupMetadataVector().Get(rg).row_num();
  }
  ASSERT_EQ(total_rows, expected_total_rows);

  // Verify each row group's data
  int64_t current_row = 0;
  for (int64_t rg : row_groups) {
    // Read the row group directly
    FileRecordBatchReader single_rg_reader(fs_, one_file_path_, schema_);
    std::vector<int64_t> single_row_group = {rg};
    ASSERT_STATUS_OK(single_rg_reader.SetRowGroups(single_row_group));
    std::shared_ptr<RecordBatch> single_batch;
    ASSERT_STATUS_OK(single_rg_reader.ReadNext(&single_batch));
    ASSERT_STATUS_OK(single_rg_reader.Close());

    // Compare with the corresponding part in the combined batch
    int64_t rg_rows = single_batch->num_rows();
    ASSERT_EQ(rg_rows, fr.file_metadata()->GetRowGroupMetadataVector().Get(rg).row_num());

    // Compare each column
    for (int i = 0; i < schema_->num_fields(); ++i) {
      auto single_array = single_batch->column(i);
      auto combined_array = batches[0]->column(i)->Slice(current_row, rg_rows);
      ASSERT_TRUE(single_array->Equals(combined_array));
    }
    current_row += rg_rows;
  }
}

TEST_F(FileReaderTest, FileRecordBatchReader_SchemaEvolutionMoreColumns) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema = arrow::schema(
      {schema_->field(0)->Copy(), schema_->field(1)->Copy(),
       arrow::field("float", arrow::float32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"400"})),
       schema_->field(2)->Copy()});

  FileRecordBatchReader fr(fs_, one_file_path_, new_schema);
  auto row_group_num = fr.file_metadata()->GetRowGroupMetadataVector().size();
  std::vector<int64_t> all_row_groups(row_group_num);
  std::iota(all_row_groups.begin(), all_row_groups.end(), 0);
  ASSERT_STATUS_OK(fr.SetRowGroups(all_row_groups));

  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));

  ASSERT_EQ(fr_table->num_columns(), new_schema->num_fields());
  ASSERT_EQ(fr_table->column(2)->null_count(), fr_table->num_rows());  // Check if extra column has nulls

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_SchemaEvolutionFewerColumns) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema = arrow::schema({schema_->field(1)->Copy(), schema_->field(0)->Copy()});

  FileRecordBatchReader fr(fs_, one_file_path_, new_schema);
  auto row_group_num = fr.file_metadata()->GetRowGroupMetadataVector().size();
  std::vector<int64_t> all_row_groups(row_group_num);
  std::iota(all_row_groups.begin(), all_row_groups.end(), 0);
  ASSERT_STATUS_OK(fr.SetRowGroups(all_row_groups));

  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));

  ASSERT_EQ(fr_table->num_columns(), 2);
  ASSERT_EQ(fr_table->schema()->field(0)->name(), "int64");
  ASSERT_EQ(fr_table->schema()->field(1)->name(), "int32");

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_SchemaEvolutionColumnOrder) {
  SetupOneFile();

  std::shared_ptr<arrow::Schema> new_schema =
      arrow::schema({schema_->field(2)->Copy(), schema_->field(1)->Copy(), schema_->field(0)->Copy()});

  FileRecordBatchReader fr(fs_, one_file_path_, new_schema);
  auto row_group_num = fr.file_metadata()->GetRowGroupMetadataVector().size();
  std::vector<int64_t> all_row_groups(row_group_num);
  std::iota(all_row_groups.begin(), all_row_groups.end(), 0);
  ASSERT_STATUS_OK(fr.SetRowGroups(all_row_groups));

  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));

  ASSERT_EQ(fr_table->num_columns(), 3);
  ASSERT_EQ(fr_table->schema()->field(0)->name(), "str");
  ASSERT_EQ(fr_table->schema()->field(1)->name(), "int64");
  ASSERT_EQ(fr_table->schema()->field(2)->name(), "int32");

  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_RowGroupMetadata) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_, schema_);

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

TEST_F(FileReaderTest, FileRecordBatchReader_ReadWithoutSchema) {
  SetupOneFile();

  // Read without providing schema
  FileRecordBatchReader fr(fs_, one_file_path_);
  auto row_group_num = fr.file_metadata()->GetRowGroupMetadataVector().size();
  std::vector<int64_t> all_row_groups(row_group_num);
  std::iota(all_row_groups.begin(), all_row_groups.end(), 0);
  ASSERT_STATUS_OK(fr.SetRowGroups(all_row_groups));

  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));

  // Verify schema matches the original file schema
  auto file_schema = fr.schema();
  ASSERT_EQ(file_schema->num_fields(), schema_->num_fields());
  ASSERT_EQ(FieldIDList::Make(file_schema).value(), FieldIDList::Make(schema_).value());

  // Verify data matches
  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader pr(fs_, paths, schema_, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(fr_table->num_rows(), pr_table->num_rows());

  ASSERT_STATUS_OK(fr.Close());
}

}  // namespace milvus_storage