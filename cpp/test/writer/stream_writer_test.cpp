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

#include <arrow/filesystem/localfs.h>
#include <gtest/gtest.h>
#include <arrow/api.h>
#include <parquet/api/writer.h>
#include <writer/stream_writer.h>
#include <parquet/properties.h>
#include <common/status.h>
#include <filesystem>

namespace milvus_storage {

class StreamWriterTest : public ::testing::Test {
  protected:
  StreamWriterTest() : props_(*parquet::default_writer_properties()) {}

  void SetUp() override {
    file_path_ = std::filesystem::temp_directory_path() / "stream_writer_test";
    std::filesystem::create_directory(file_path_);

    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    for (int i = 0; i < 10; ++i) {
      ASSERT_TRUE(int64_builder.AppendValues({1, 2, 3}).ok());
    }

    ASSERT_TRUE(
        str_builder.AppendValues({std::string(10000, 'a'), std::string(10000, 'b'), std::string(10000, 'c')}).ok());
    std::vector<std::shared_ptr<arrow::Array>> int64_columns;
    std::shared_ptr<arrow::Array> str_array;
    std::shared_ptr<arrow::Array> int64_array;

    for (int i = 0; i < 10; ++i) {
      ASSERT_TRUE(int64_builder.Finish(&int64_array).ok());
      int64_columns.push_back(int64_array);
    }

    ASSERT_TRUE(str_builder.Finish(&str_array).ok());

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < 10; ++i) {
      fields.push_back(arrow::field("int64_" + std::to_string(i), arrow::int64()));
    }
    fields.push_back(arrow::field("large_str", arrow::utf8()));

    schema_ = arrow::schema(fields);
    columns_ = int64_columns;
    columns_.push_back(str_array);

    record_batch_ = arrow::RecordBatch::Make(schema_, 3, columns_);

    fs_ = arrow::fs::LocalFileSystem();
    props_ = *parquet::default_writer_properties();
    memory_limit_ = 1024 * 1024;  // 1MB for testing
  }

  void TearDown() override { std::filesystem::remove_all(file_path_); }

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::vector<std::shared_ptr<arrow::Array>> columns_;
  arrow::fs::LocalFileSystem fs_;
  std::string file_path_;
  parquet::WriterProperties props_;
  size_t memory_limit_;
  std::unique_ptr<StreamWriter> writer_;
};

TEST_F(StreamWriterTest, Write) {
  StreamWriter writer(memory_limit_, schema_, fs_, file_path_, props_);
  auto status = writer.Init(record_batch_);
  EXPECT_TRUE(status.ok());

  status = writer.Write(record_batch_);
  EXPECT_TRUE(status.ok());

  status = writer.Close();
  EXPECT_TRUE(status.ok());
}

}  // namespace milvus_storage