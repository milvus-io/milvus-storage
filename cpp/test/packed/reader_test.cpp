// Copyright 2023 Zilliz
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

#include <memory>

#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/table.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/packed/reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "test_env.h"

namespace milvus_storage {

class PackedRecordBatchReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {}
};

TEST_F(PackedRecordBatchReaderTest, RowOffsetMinHeapTest) {
  RowOffsetMinHeap minHeap;

  minHeap.emplace(1, 30);
  minHeap.emplace(2, 20);
  minHeap.emplace(3, 40);
  minHeap.emplace(4, 10);

  EXPECT_EQ(minHeap.top().second, 10);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 20);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 30);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 40);
}

}  // namespace milvus_storage
