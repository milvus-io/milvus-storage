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
#include "milvus-storage/format/parquet/file_reader.h"
#include "arrow/table.h"

namespace milvus_storage {

class FileReaderTest : public PackedTestBase {};

TEST_F(FileReaderTest, FileRecordBatchReader) {
  int batch_size = 100;

  auto paths = std::vector<std::string>{path_.string() + "/10000.parquet"};
  auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  auto column_index_groups = writer.Close();

  // read all row groups
  FileRecordBatchReader fr(fs_, paths[0]);
  auto row_group_sizes = fr.GetRowGroupSizes();
  fr.SetRowGroupOffsetAndCount(0, row_group_sizes.size());
  std::shared_ptr<RecordBatch> batch;
  fr.ReadNext(&batch);
  ASSERT_AND_ARROW_ASSIGN(auto fr_table, arrow::Table::FromRecordBatches({batch}));
  auto arrow_schema = fr.schema();
  ASSERT_EQ(arrow_schema->num_fields(), schema_->num_fields());
  for (int i = 0; i < arrow_schema->num_fields(); ++i) {
    ASSERT_EQ(arrow_schema->field(i)->metadata()->Get(ARROW_FIELD_ID_KEY), schema_->field(i)->metadata()->Get(ARROW_FIELD_ID_KEY));
  }
  ASSERT_STATUS_OK(fr.Close());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };
  PackedRecordBatchReader pr(fs_, paths, schema_, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto pr_table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());
  ASSERT_EQ(fr_table->num_rows(), pr_table->num_rows());

  // read row group 1
  FileRecordBatchReader rgr(fs_, paths[0]);
  rgr.SetRowGroupOffsetAndCount(1, 1);
  std::shared_ptr<RecordBatch> rg_batch;
  rgr.ReadNext(&rg_batch);
  ASSERT_STATUS_OK(rgr.Close());
  ASSERT_AND_ARROW_ASSIGN(auto rg_table, arrow::Table::FromRecordBatches({rg_batch}));
  ASSERT_GT(fr_table->num_rows(), rg_table->num_rows());
}

}  // namespace milvus_storage