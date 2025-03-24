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
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include "arrow/table.h"
#include "milvus-storage/common/type_fwd.h"

namespace milvus_storage {

class FileReaderTest : public PackedTestBase {};

TEST_F(FileReaderTest, FileRecordBatchReader_ReadAllColumns) {
  SetupOneFile();
  // read all row groups
  FileRecordBatchReader fr(fs_, one_file_path_);
  auto row_group_sizes = fr.file_metadata()->GetRowGroupSizeVector();
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size()));
  std::shared_ptr<RecordBatch> batch;
  ASSERT_STATUS_OK(fr.ReadNext(&batch));
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));
  auto arrow_schema = fr.schema();
  ASSERT_EQ(arrow_schema->num_fields(), schema_->num_fields());
  ASSERT_EQ(FieldIDList::Make(arrow_schema).value(), FieldIDList::Make(schema_).value());
  ASSERT_STATUS_OK(fr.Close());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };
  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader pr(fs_, paths, schema_, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(fr_table->num_rows(), pr_table->num_rows());

  // read row group 1
  FileRecordBatchReader rgr(fs_, paths[0]);
  ASSERT_STATUS_OK(rgr.SetRowGroupOffsetAndCount(1, 1));
  std::shared_ptr<RecordBatch> rg_batch;
  ASSERT_STATUS_OK(rgr.ReadNext(&rg_batch));
  ASSERT_STATUS_OK(rgr.Close());
  ASSERT_AND_ARROW_ASSIGN(auto rg_table, arrow::Table::FromRecordBatches({rg_batch}));
  ASSERT_GT(fr_table->num_rows(), rg_table->num_rows());
}

TEST_F(FileReaderTest, FileRecordBatchReader_ReadPartialRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_);
  ASSERT_STATUS_OK(fr.SetRowGroupOffsetAndCount(1, 1));
  std::shared_ptr<RecordBatch> rg_batch;
  ASSERT_STATUS_OK(fr.ReadNext(&rg_batch));
  ASSERT_EQ(fr.file_metadata()->GetParquetMetadata()->RowGroup(1)->num_rows(), rg_batch->num_rows());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_NonExistedRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_);
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(100, 1).ok());
  ASSERT_STATUS_OK(fr.Close());
}

TEST_F(FileReaderTest, FileRecordBatchReader_ReadNoRowGroup) {
  SetupOneFile();
  FileRecordBatchReader fr(fs_, one_file_path_);
  ASSERT_FALSE(fr.SetRowGroupOffsetAndCount(0, 0).ok());
  ASSERT_STATUS_OK(fr.Close());
}

}  // namespace milvus_storage