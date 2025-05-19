

#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "milvus-storage/packed/reader.h"
#include "test_util.h"
#include "arrow/table.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"

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
