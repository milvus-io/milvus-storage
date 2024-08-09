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

#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "arrow/table.h"
#include "packed/chunk_manager.h"
#include "packed_test_base.h"

namespace milvus_storage {

class ChunkManagerTest : public PackedTestBase {
  protected:
  void SetUp() override {
    SetUpCommonData();

    // int32 and int64 columns
    tables_.emplace_back(table_->SelectColumns({0, 1}).ValueOrDie());
    // large string column
    tables_.emplace_back(table_->SelectColumns({2}).ValueOrDie());

    column_offsets_ = {ColumnOffset(0, 0), ColumnOffset(0, 1), ColumnOffset(1, 0)};

    chunk_manager_ = std::make_unique<ChunkManager>(column_offsets_, chunksize_);
  }

  std::vector<ColumnOffset> column_offsets_;
  std::unique_ptr<ChunkManager> chunk_manager_;
  std::vector<std::shared_ptr<arrow::Table>> tables_;
  int chunksize_ = 2;
};

TEST_F(ChunkManagerTest, GetMaxContiguousSlice) {
  auto chunks = chunk_manager_->GetMaxContiguousSlice(tables_);
  ASSERT_EQ(chunks.size(), column_offsets_.size());
  for (auto& chunk : chunks) {
    EXPECT_EQ(chunk->length(), 3);
  }
  auto batches = chunk_manager_->SliceChunks(chunks);
  for (auto& batch : batches) {
    EXPECT_EQ(batch.get()->length, 2);
  }
}

}  // namespace milvus_storage