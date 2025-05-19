

#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "arrow/table.h"
#include "milvus-storage/packed/chunk_manager.h"
#include "packed_test_base.h"

namespace milvus_storage {

class ChunkManagerTest : public PackedTestBase {
  protected:
  void SetUp() override {
    SetUpCommonData();
    tables_.resize(2, std::queue<std::shared_ptr<arrow::Table>>());
    // int32 and int64 columns
    tables_[0].push(table_->SelectColumns({0, 1}).ValueOrDie());
    // large string column
    tables_[1].push(table_->SelectColumns({2}).ValueOrDie());

    column_offsets_ = {ColumnOffset(0, 0), ColumnOffset(0, 1), ColumnOffset(1, 0)};

    chunk_manager_ = std::make_unique<ChunkManager>(column_offsets_, chunksize_);
  }

  std::vector<ColumnOffset> column_offsets_;
  std::unique_ptr<ChunkManager> chunk_manager_;
  std::vector<std::queue<std::shared_ptr<arrow::Table>>> tables_;
  int chunksize_ = 2;
};

TEST_F(ChunkManagerTest, SliceChunksByMaxContiguousSlice) {
  auto chunks = chunk_manager_->SliceChunksByMaxContiguousSlice(chunksize_, tables_);
  ASSERT_EQ(chunks.size(), column_offsets_.size());
}

}  // namespace milvus_storage