#ifdef BUILD_VORTEX_BRIDGE

#include <memory>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/table.h>

#include "test_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/constants.h"
#include <gtest/gtest.h>
#include "milvus-storage/format/vortex/vortex_writer.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

class VortexBasicTest : public ::testing::Test {
  protected:
  void SetUp() override {
    config_.address = "http://localhost:9000";
    config_.bucket_name = "rust-bucket";
    table_ = arrow::Table::FromRecordBatches({randomRecordBatch()}).ValueOrDie();
    schema_ = table_->schema();
  }

  protected:
  std::shared_ptr<arrow::RecordBatch> randomRecordBatch() {
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    int32_values = {rand() % 10000, rand() % 10000, rand() % 10000};
    int64_values = {rand() % 10000000, rand() % 10000000, rand() % 10000000};
    str_values = {random_string(10000), random_string(10000), random_string(10000)};

    int_builder.AppendValues(int32_values).ok();
    int64_builder.AppendValues(int64_values).ok();
    str_builder.AppendValues(str_values).ok();

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> int64_array;
    std::shared_ptr<arrow::Array> str_array;

    int_builder.Finish(&int_array).ok();
    int64_builder.Finish(&int64_array).ok();
    str_builder.Finish(&str_array).ok();

    std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
    auto schema = arrow::schema(
        {arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
         arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"}))});
    return arrow::RecordBatch::Make(schema, 3, arrays);
  }

  std::string random_string(size_t length) {
    auto randchar = []() -> char {
      const char charset[] =
          "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
  }

  protected:
  ArrowFileSystemConfig config_;  // default one
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Table> table_;

  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<std::basic_string<char>> str_values;
};

TEST_F(VortexBasicTest, TestBasicWrite) {
  auto vw = vortex::VortexFileWriter(config_, schema_, "test-file.vx", api::Properties());
  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(vw.Write(randomRecordBatch()).ok());
  }

  ASSERT_TRUE(vw.Flush().ok());
  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(vw.Write(randomRecordBatch()).ok());
  }
  ASSERT_TRUE(vw.Flush().ok());

  ASSERT_EQ(20 * 3, vw.count());
  ASSERT_TRUE(vw.Close().ok());
}

}  // namespace milvus_storage

#endif