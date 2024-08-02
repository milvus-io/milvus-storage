#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <gtest/gtest.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <memory>
#include "writer/splitter/size_based_splitter.h"

using namespace milvus_storage::writer;

namespace milvus_storage {
namespace writer {

class SizeBasedSplitterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    int_builder.AppendValues({1, 2, 3});
    int64_builder.AppendValues({1, 2, 3});
    str_builder.AppendValues({std::string(10000, 'a'), std::string(10000, 'b'), std::string(10000, 'c')});

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> bool_array;
    std::shared_ptr<arrow::Array> str_array;

    int_builder.Finish(&int_array);
    int64_builder.Finish(&bool_array);
    str_builder.Finish(&str_array);

    columns_ = {int_array, str_array, bool_array};
    schema_ = arrow::schema({arrow::field("int", arrow::int32()), arrow::field("large_str", arrow::utf8()),
                             arrow::field("bool", arrow::int64())});

    record_batch_ = arrow::RecordBatch::Make(schema_, 3, columns_);
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::vector<std::shared_ptr<arrow::Array>> columns_;
};

TEST_F(SizeBasedSplitterTest, SplitColumnsTest) {
  SizeBasedSplitter splitter(64);
  std::vector<std::shared_ptr<arrow::RecordBatch>> result = splitter.Split(record_batch_);

  ASSERT_EQ(result.size(), 2);

  EXPECT_EQ(result[0]->column(0)->type()->id(), arrow::Type::STRING);
  EXPECT_EQ(result[1]->column(0)->type()->id(), arrow::Type::INT32);
  EXPECT_EQ(result[1]->column(1)->type()->id(), arrow::Type::INT64);
}

}  // namespace writer
}  // namespace milvus_storage
