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

#include <memory>
#include <vector>
#include <string>

#include <parquet/properties.h>
#include <arrow/filesystem/filesystem.h>
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
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/packed/column_group.h"

#include "test_env.h"

namespace milvus_storage {

class PackedTestBase : public ::testing::Test {
  protected:
  void SetUp() override {
    api::Properties properties;
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties));
    path_ = GetTestBasePath("packed-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, path_));

    SetUpCommonData();
    writer_memory_ = (22 + 16) * 1024 * 1024;  // 22 MB for S3FS part upload
    reader_memory_ = 5 * 1024 * 1024;
  }

  void TearDown() override {
    if (fs_ != nullptr) {
      ASSERT_STATUS_OK(DeleteTestDir(fs_, path_));
    }
  }

  void ValidateTableData(const std::shared_ptr<arrow::Table>& table) {
    auto chunks = table->GetColumnByName("str");
    int64_t count = 0;
    for (int i = 0; i < chunks->num_chunks(); ++i) {
      auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(chunks->chunk(i));
      for (int j = 0; j < str_array->length(); ++j) {
        int expected_index = (count++) % str_values.size();
        ASSERT_EQ(str_array->GetString(j), str_values[expected_index]);
      }
    }

    chunks = table->GetColumnByName("int32");
    count = 0;
    for (int i = 0; i < chunks->num_chunks(); ++i) {
      auto int32_array = std::dynamic_pointer_cast<arrow::Int32Array>(chunks->chunk(i));
      for (int j = 0; j < int32_array->length(); ++j) {
        int expected_index = (count++) % int32_values.size();
        ASSERT_EQ(int32_array->Value(j), int32_values[expected_index]);
      }
    }

    chunks = table->GetColumnByName("int64");
    count = 0;
    for (int i = 0; i < chunks->num_chunks(); ++i) {
      auto int64_array = std::dynamic_pointer_cast<arrow::Int64Array>(chunks->chunk(i));
      for (int j = 0; j < int64_array->length(); ++j) {
        int expected_index = (count++) % int64_values.size();
        ASSERT_EQ(int64_array->Value(j), int64_values[expected_index]);
      }
    }
  }

  void SetupOneFile() {
    one_file_path_ = path_ + "/10000.parquet";
    std::vector<std::string> paths = {one_file_path_};
    int batch_size = 100;
    auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};
    ASSERT_AND_ASSIGN(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, column_groups, writer_memory_));
    for (int i = 0; i < batch_size; ++i) {
      EXPECT_TRUE(writer->Write(record_batch_).ok());
    }
    auto column_index_groups = writer->Close();
  }

  void SetUpCommonData() {
    record_batch_ = randomRecordBatch();
    table_ = arrow::Table::FromRecordBatches({record_batch_}).ValueOrDie();
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

  arrow::Result<std::shared_ptr<arrow::Table>> ReadToTable(PackedRecordBatchReader& reader) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(reader.ReadNext(&batch));
      if (!batch)
        break;
      batches.push_back(batch);
    }
    return arrow::Table::FromRecordBatches(reader.schema(), batches);
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

  size_t writer_memory_;
  size_t reader_memory_;
  ArrowFileSystemPtr fs_;
  std::string path_;

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::shared_ptr<arrow::Table> table_;

  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<std::basic_string<char>> str_values;

  std::string one_file_path_;
};

}  // namespace milvus_storage