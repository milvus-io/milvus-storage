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

#include <gtest/gtest.h>
#include <arrow/api.h>
#include "writer/splitter/indices_based_splitter.h"
#include "writer/column_group.h"
#include "splitter_test_base.h"

namespace milvus_storage {

class IndicesBasedSplitterTest : public SplitterTestBase {
  protected:
  void SetUp() override {
    SetUpCommonData();
    column_indices_ = {{1}, {0, 2}};
  }
  std::vector<std::vector<int>> column_indices_;
};

TEST_F(IndicesBasedSplitterTest, SplitColumnsTest) {
  IndicesBasedSplitter splitter(column_indices_);
  std::vector<ColumnGroup> column_groups = splitter.Split(record_batch_);

  ASSERT_EQ(column_groups.size(), 2);

  ASSERT_EQ(column_groups[0].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT64);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(0)->type()->id(), arrow::Type::INT32);
  ASSERT_EQ(column_groups[1].GetRecordBatch(0)->column(1)->type()->id(), arrow::Type::STRING);
}

}  // namespace milvus_storage
