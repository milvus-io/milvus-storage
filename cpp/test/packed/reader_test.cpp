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

#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "packed/reader.h"
#include "test_util.h"
#include "arrow/table.h"
#include "filesystem/fs.h"
#include "common/config.h"

namespace milvus_storage {

class PackedRecordBatchReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    auto arrow_schema = CreateArrowSchema({"pk_field"}, {arrow::int64()});
    arrow::Int64Builder pk_builder;
    ASSERT_STATUS_OK(pk_builder.AppendValues({1, 2, 3}));
    std::shared_ptr<arrow::Array> pk_array;
    ASSERT_STATUS_OK(pk_builder.Finish(&pk_array));
    auto rec_batch = arrow::RecordBatch::Make(arrow_schema, 3, {pk_array});
    std::string path;
    auto factory = std::make_shared<FileSystemFactory>();
    auto conf = StorageConfig();
    conf.uri = "file:///tmp/";
    ASSERT_AND_ASSIGN(fs_, factory->BuildFileSystem(conf, &path));
    ASSERT_AND_ARROW_ASSIGN(auto f1, fs_->OpenOutputStream("/tmp/f1"));
    ASSERT_AND_ARROW_ASSIGN(auto w1, parquet::arrow::FileWriter::Open(*arrow_schema, arrow::default_memory_pool(), f1));
    ASSERT_STATUS_OK(w1->WriteRecordBatch(*rec_batch));
    ASSERT_STATUS_OK(w1->Close());
    ASSERT_STATUS_OK(f1->Close());

    arrow_schema = CreateArrowSchema({"json_field"}, {arrow::utf8()});
    arrow::StringBuilder builder;
    ASSERT_STATUS_OK(builder.AppendValues({"foo", "bar", "foo"}));
    std::shared_ptr<arrow::Array> json_array;
    ASSERT_STATUS_OK(builder.Finish(&json_array));
    rec_batch = arrow::RecordBatch::Make(arrow_schema, 3, {json_array});

    ASSERT_AND_ARROW_ASSIGN(auto f2, fs_->OpenOutputStream("/tmp/f2"));
    ASSERT_AND_ARROW_ASSIGN(auto w2, parquet::arrow::FileWriter::Open(*arrow_schema, arrow::default_memory_pool(), f2));
    ASSERT_STATUS_OK(w2->WriteRecordBatch(*rec_batch));
    ASSERT_STATUS_OK(w2->Close());
    ASSERT_STATUS_OK(f2->Close());
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
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
