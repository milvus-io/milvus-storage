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
#include "milvus-storage/vortex/VortexWriter.h"
#include "milvus-storage/vortex/VortexReader.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

class VortexBasicTest : public ::testing::Test {
  protected:
  void SetUp() override {
    path_ = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    auto conf = ArrowFileSystemConfig();
    conf.storage_type = "local";
    conf.root_path = path_.string();

    ArrowFileSystemSingleton::GetInstance().Init(conf);
    fs_ = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();

    record_batch_ = randomRecordBatch();
    table_ = arrow::Table::FromRecordBatches({record_batch_}).ValueOrDie();
    schema_ = table_->schema();
  }

  private:
  std::shared_ptr<arrow::RecordBatch> randomRecordBatch() {
    auto fields = {
        arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
        arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"}))};
    // limit by arrow-ffi
    auto struct_field = ::arrow::field("struct", arrow::struct_(fields));
    auto schema = arrow::schema({struct_field});
    std::shared_ptr<arrow::Array> struct_array;

    auto int_builder = std::make_shared<arrow::Int32Builder>();
    auto int64_builder = std::make_shared<arrow::Int64Builder>();
    auto str_builder = std::make_shared<arrow::StringBuilder>();
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders = {int_builder, int64_builder, str_builder};
    arrow::StructBuilder struct_builder(arrow::struct_(fields), arrow::default_memory_pool(), builders);

    int32_values = {1, 2, 3};
    int64_values = {11, 12, 13};
    str_values = {random_string(10000), random_string(10000), random_string(10000)};

    int_builder->AppendValues(int32_values).ok();
    int64_builder->AppendValues(int64_values).ok();
    str_builder->AppendValues(str_values).ok();
    struct_builder.Append(true).ok();

    struct_builder.Finish(&struct_array).ok();

    return arrow::RecordBatch::Make(schema, 1, {struct_array});
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
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  boost::filesystem::path path_;

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::shared_ptr<arrow::Table> table_;

  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<std::basic_string<char>> str_values;
};

TEST_F(VortexBasicTest, TestBasicWriteRead) {
  auto vw = VortexWriter(fs_, path_.string() + "/test.vortex", schema_);
  vw.write(record_batch_);
  vw.flush();
  vw.close();

  auto vr = VortexReader(fs_, path_.string() + "/test.vortex");
  uint64_t idx = 0;
  auto rbs = vr.TakeToRecordBatchs(&idx, 1);
  ASSERT_TRUE(rbs.ok());
  ASSERT_EQ((*rbs).size(), 1);
  ASSERT_EQ((*rbs)[0]->num_rows(), 1);

  // take the row1
  auto tb = vr.TakeToTable(&idx, 1);
  ASSERT_TRUE(tb.ok());
  ASSERT_EQ((*tb)->fields().size(), 3);
  auto f1 = (*tb)->field(0);

  auto chunks = (*tb)->GetColumnByName("int32");
  ASSERT_EQ(chunks->num_chunks(), 1);
  auto int32_array = std::dynamic_pointer_cast<arrow::Int32Array>(chunks->chunk(0));
  ASSERT_EQ(int32_array->length(), 1);
  ASSERT_EQ(int32_array->Value(0), 1);  // row1

  vr.close();
}

}  // namespace milvus_storage