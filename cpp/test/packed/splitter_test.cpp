

#include <gtest/gtest.h>
#include <arrow/api.h>
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/packed/splitter/size_based_splitter.h"
#include "milvus-storage/packed/column_group.h"
#include "packed_test_base.h"

namespace milvus_storage {

class SplitterTest : public PackedTestBase {
  protected:
  void SetUp() override { SetUpCommonData(); }
};

TEST_F(SplitterTest, IndicesBasedSplitterTest) {
  std::vector<std::vector<int>> column_indices_ = {{1}, {0, 2}};
  IndicesBasedSplitter splitter(column_indices_);
  std::vector<ColumnGroup> column_groups = splitter.Split(record_batch_);

  ASSERT_EQ(column_groups.size(), 2);

  ASSERT_EQ(column_groups[0].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT64);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT32);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(1)->type()->id(), arrow::Type::STRING);
}

TEST_F(SplitterTest, SizeBasedSplitterTest) {
  SizeBasedSplitter splitter(64);
  std::vector<ColumnGroup> column_groups = splitter.Split(record_batch_);

  ASSERT_EQ(column_groups.size(), 2);

  ASSERT_EQ(column_groups[0].GetRecordBatch(0)->num_columns(), 1);
  ASSERT_EQ(column_groups[0].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::STRING);

  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->num_columns(), 2);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT32);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(1)->type()->id(), arrow::Type::INT64);
}

}  // namespace milvus_storage
