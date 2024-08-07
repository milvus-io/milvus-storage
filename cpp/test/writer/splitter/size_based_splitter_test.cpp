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

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <gtest/gtest.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <memory>
#include "writer/splitter/size_based_splitter.h"
#include "splitter_test_base.h"

using namespace std;

namespace milvus_storage {

class SizeBasedSplitterTest : public SplitterTestBase {
  protected:
  void SetUp() override { SetUpCommonData(); }
};

TEST_F(SizeBasedSplitterTest, SplitColumnsTest) {
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
