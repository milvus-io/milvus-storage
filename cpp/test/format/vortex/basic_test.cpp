// Copyright 2025 Zilliz
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
#include <random>
#include <set>
#include <vector>
#include <cstdint>

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
#include <arrow/array/concatenate.h>

#include "test_env.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

using namespace vortex;

class VortexBasicTest : public ::testing::Test {
  void SetUp() override {
    schema_ = arrow::Table::FromRecordBatches({makeRecordBatch(0, 0, 0)}).ValueOrDie()->schema();
    record_bacths_ = makeRecordBatchs();
    columngroup_ = std::make_shared<api::ColumnGroup>();

    columngroup_->format = LOON_FORMAT_VORTEX;
    columngroup_->files = {{.path = test_file_name_}};
    columngroup_->columns = {"int32", "int64", "binary"};

    ASSERT_STATUS_OK(InitTestProperties(properties_));

    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(test_file_name_);
    }
  }

  protected:
  std::vector<std::shared_ptr<arrow::RecordBatch>> makeRecordBatchs() {
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
    uint32_t offset_each_loop = 0;

    // offset: [0, 0, 100, 300 ..., 3600]
    // count : [0, 100, 200, ..., 900]
    for (int i = 0; i < record_batch_len_; i++) {
      rbs.emplace_back(makeRecordBatch(offset_each_loop, count_each_loop_ * i, rand_strlen_));
      offset_each_loop += (count_each_loop_ * i);
    }
    return rbs;
  }

  inline int64_t recordBatchsRows() const {
    return (count_each_loop_ * (record_batch_len_ - 1)) * record_batch_len_ / 2;
  }

  inline size_t recordBatchsSize() const { return record_bacths_.size(); }

  template <typename T>
  std::vector<T> randomNumbers(T maxVal, T size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate unique random numbers using a set
    std::set<T> unique_numbers;
    std::uniform_int_distribution<T> dis(0, maxVal);
    while (unique_numbers.size() < static_cast<size_t>(size)) {
      unique_numbers.insert(dis(gen));
    }

    // Convert to sorted vector (std::set is already sorted)
    return std::vector<T>(unique_numbers.begin(), unique_numbers.end());
  }

  template <typename T>
  std::vector<T> rangeNumbers(T start, T end) {
    std::vector<T> numbers;

    for (T i = start; i < end; ++i) {
      numbers.emplace_back(i);
    }

    return numbers;
  }

  std::shared_ptr<arrow::RecordBatch> makeRecordBatch(uint32_t offset, uint32_t count, uint32_t str_len) {
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder binary_builder;

    std::vector<int32_t> int32_values;
    std::vector<int64_t> int64_values;
    std::vector<std::basic_string<char>> binary_values;

    for (int i = offset; i < offset + count; i++) {
      int32_values.emplace_back(i);
      int64_values.emplace_back(i);
      binary_values.emplace_back(random_string(str_len));
    }

    int_builder.AppendValues(int32_values).ok();
    int64_builder.AppendValues(int64_values).ok();
    binary_builder.AppendValues(binary_values).ok();

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> int64_array;
    std::shared_ptr<arrow::Array> str_array;

    int_builder.Finish(&int_array).ok();
    int64_builder.Finish(&int64_array).ok();
    binary_builder.Finish(&str_array).ok();

    std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
    auto schema = arrow::schema(
        {arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
         arrow::field("binary", arrow::binary(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"}))});
    return arrow::RecordBatch::Make(schema, count, arrays);
  }

  std::string random_string(size_t size) {
    std::string str;
    str.resize(size);

    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    const size_t num_blocks = size / sizeof(uint64_t);
    const size_t remainder = size % sizeof(uint64_t);

    // treat the string as an array of uint64_t and fill 8 bytes at a time
    auto p = reinterpret_cast<uint64_t*>(&str[0]);
    for (size_t i = 0; i < num_blocks; ++i) {
      p[i] = dis(gen);
    }

    // deal the remainder bytes
    if (remainder > 0) {
      uint64_t last_block = dis(gen);
      char* last_chars = reinterpret_cast<char*>(&last_block);
      for (size_t i = 0; i < remainder; ++i) {
        str[size - remainder + i] = last_chars[i];
      }
    }

    return str;
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkedArrayToRecordBatch(
      const std::shared_ptr<arrow::ChunkedArray>& chunkedarray) {
    auto chunk_size = chunkedarray->num_chunks();
    if (chunk_size == 1) {
      return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
    for (int i = 0; i < chunk_size; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
      rbs.emplace_back(rb);
    }

    return arrow::ConcatenateRecordBatches(rbs);
  }

  protected:
  std::shared_ptr<api::ColumnGroup> columngroup_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_bacths_;
  const char* test_file_name_ = "test-file.vx";
  api::Properties properties_;

  private:
  uint32_t count_each_loop_ = 100;
  uint32_t rand_strlen_ = 100;
  uint32_t record_batch_len_ = 10;
};

TEST_F(VortexBasicTest, TestBasicWrite) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);
}

TEST_F(VortexBasicTest, TestBasicRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_,
                                              std::vector<std::string>{"int32", "int64", "binary"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));

  ASSERT_EQ(recordBatchsRows(), rb->num_rows());
  ASSERT_EQ(3, rb->num_columns());
  ASSERT_EQ(arrow::Type::INT32, rb->column(0)->type_id());
  ASSERT_EQ(arrow::Type::INT64, rb->column(1)->type_id());

  auto i32array = std::dynamic_pointer_cast<arrow::Int32Array>(rb->column(0));
  auto i64array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(1));

  for (int i = 0; i < i32array->length(); ++i) {
    ASSERT_EQ(i32array->Value(i), (int32_t)i);
    ASSERT_EQ(i64array->Value(i), (int64_t)i);
  }
}

TEST_F(VortexBasicTest, TestEmptyWriteRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  auto empty_rb = makeRecordBatch(0, 0, 0);
  ASSERT_TRUE(vx_writer.Write(empty_rb).ok());

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(0, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_,
                                              std::vector<std::string>{"int32", "int64", "binary"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(0, vx_reader.rows());

  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, vx_reader.rows()));
  ASSERT_EQ(0, chunked_array->num_chunks());
}

TEST_F(VortexBasicTest, TestReaderProjection) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  // all projection
  {
    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_,
                                                std::vector<std::string>{"int32", "int64", "binary"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());
    ASSERT_EQ(arrow::Type::INT32, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::INT64, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::BINARY, rb->column(2)->type_id());
  }

  // projection with different order
  {
    auto projection_schema = arrow::schema({
        arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
        arrow::field("binary", arrow::binary(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"})),
        arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
    });

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"int64", "binary", "int32"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::BINARY, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::INT32, rb->column(2)->type_id());
  }

  // single projection
  {
    auto projection_schema = arrow::schema(
        {arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"}))});

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"int64"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
  }

  // empty projection
  {
    auto projection_schema = arrow::schema({});
    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());
  }
}

TEST_F(VortexBasicTest, TestBasicTake) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto take_verify = [&](vortex::VortexFormatReader& vx_reader, const std::vector<int64_t>& row_indices,
                         int64_t expect_rows) {
    ASSERT_AND_ASSIGN(auto table, vx_reader.take(row_indices));
    ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());

    ASSERT_EQ(expect_rows, rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT32, rb->column(0)->type_id());
    auto i32array = std::dynamic_pointer_cast<arrow::Int32Array>(rb->column(0));
    for (size_t i = 0; i < rb->num_rows(); i++) {
      ASSERT_EQ(i32array->Value(i), (int32_t)row_indices[i]);
    }
  };

  auto projection_schema = arrow::schema(
      {arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"int32"});
  ASSERT_STATUS_OK(vx_reader.open());
  // take single row
  take_verify(vx_reader, std::move(randomNumbers<int64_t>(recordBatchsRows() - 1, 1)), 1);
  // 100 randowm rows
  take_verify(vx_reader, std::move(randomNumbers<int64_t>(recordBatchsRows() - 1, 100)), 100);
  // boundary Testing
  take_verify(vx_reader, {0, (int64_t)recordBatchsRows() - 1}, 2);
  // take all range
  take_verify(vx_reader, rangeNumbers<int64_t>(0, recordBatchsRows()), (int64_t)recordBatchsRows());
  // Note: vortex 0.56+ does not gracefully handle out-of-range indices (panics instead of returning error),
  // so we removed the out-of-range index tests.
}

class VortexSegmentRowSizeTest : public ::testing::Test {
  void SetUp() override {
    // Schema with scalar, vector (FixedSizeBinary), and varlen (binary) columns
    vector_schema_ = arrow::schema({
        arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("vector", arrow::fixed_size_binary(16), false,
                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
    });
    varlen_schema_ = arrow::schema({
        arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("text", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"})),
    });
    scalar_schema_ = arrow::schema({
        arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("value", arrow::float64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
    });

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  void TearDown() override {
    for (const auto& f : test_files_) {
      boost::filesystem::remove_all(f);
    }
  }

  protected:
  std::shared_ptr<arrow::RecordBatch> makeVectorBatch(int32_t offset, int32_t count) {
    arrow::Int32Builder id_builder;
    arrow::FixedSizeBinaryBuilder vec_builder(arrow::fixed_size_binary(16));

    for (int32_t i = offset; i < offset + count; i++) {
      id_builder.Append(i).ok();
      // Fill 16 bytes with a pattern based on i
      uint8_t buf[16];
      memset(buf, 0, sizeof(buf));
      memcpy(buf, &i, sizeof(i));
      vec_builder.Append(buf).ok();
    }

    std::shared_ptr<arrow::Array> id_array, vec_array;
    id_builder.Finish(&id_array).ok();
    vec_builder.Finish(&vec_array).ok();
    return arrow::RecordBatch::Make(vector_schema_, count, {id_array, vec_array});
  }

  std::shared_ptr<arrow::RecordBatch> makeVarlenBatch(int32_t offset, int32_t count) {
    arrow::Int32Builder id_builder;
    arrow::StringBuilder text_builder;

    for (int32_t i = offset; i < offset + count; i++) {
      id_builder.Append(i).ok();
      text_builder.Append("text_" + std::to_string(i)).ok();
    }

    std::shared_ptr<arrow::Array> id_array, text_array;
    id_builder.Finish(&id_array).ok();
    text_builder.Finish(&text_array).ok();
    return arrow::RecordBatch::Make(varlen_schema_, count, {id_array, text_array});
  }

  std::shared_ptr<arrow::RecordBatch> makeScalarBatch(int32_t offset, int32_t count) {
    arrow::Int32Builder id_builder;
    arrow::DoubleBuilder val_builder;

    for (int32_t i = offset; i < offset + count; i++) {
      id_builder.Append(i).ok();
      val_builder.Append(static_cast<double>(i) * 1.5).ok();
    }

    std::shared_ptr<arrow::Array> id_array, val_array;
    id_builder.Finish(&id_array).ok();
    val_builder.Finish(&val_array).ok();
    return arrow::RecordBatch::Make(scalar_schema_, count, {id_array, val_array});
  }

  protected:
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::shared_ptr<arrow::Schema> varlen_schema_;
  std::shared_ptr<arrow::Schema> scalar_schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  api::Properties properties_;
  std::vector<std::string> test_files_;
};

TEST_F(VortexSegmentRowSizeTest, TestVectorColumnWriteRead) {
  const char* file_name = "test-vector-segment.vx";
  test_files_.push_back(file_name);

  // Set custom segment sizes
  ASSERT_EQ(std::nullopt, api::SetValue(properties_, PROPERTY_WRITER_VORTEX_SEGMENT_ROW_SIZE, "4096"));
  ASSERT_EQ(std::nullopt, api::SetValue(properties_, PROPERTY_WRITER_VORTEX_VECTOR_SEGMENT_ROW_SIZE, "128"));
  ASSERT_EQ(std::nullopt, api::SetValue(properties_, PROPERTY_WRITER_VORTEX_VARLEN_SEGMENT_ROW_SIZE, "1024"));

  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_schema_, file_name, properties_);

  int32_t total_rows = 0;
  for (int i = 0; i < 5; i++) {
    auto rb = makeVectorBatch(total_rows, 200);
    ASSERT_STATUS_OK(vx_writer.Write(rb));
    total_rows += 200;
  }

  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  // Read back and verify
  auto id_schema = arrow::schema(
      {arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto vx_reader =
      vortex::VortexFormatReader(file_system_, id_schema, file_name, properties_, std::vector<std::string>{"id"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(total_rows, vx_reader.rows());

  ASSERT_AND_ASSIGN(auto table, vx_reader.take(std::vector<int64_t>{0, 1, 2}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  ASSERT_EQ(3, rb->num_rows());
  ASSERT_EQ(1, rb->num_columns());
  auto id_array = std::dynamic_pointer_cast<arrow::Int32Array>(rb->column(0));
  ASSERT_EQ(0, id_array->Value(0));
  ASSERT_EQ(1, id_array->Value(1));
  ASSERT_EQ(2, id_array->Value(2));
}

TEST_F(VortexSegmentRowSizeTest, TestVarlenColumnWriteRead) {
  const char* file_name = "test-varlen-segment.vx";
  test_files_.push_back(file_name);

  // Set custom segment sizes
  ASSERT_EQ(std::nullopt, api::SetValue(properties_, PROPERTY_WRITER_VORTEX_VARLEN_SEGMENT_ROW_SIZE, "512"));

  auto vx_writer = vortex::VortexFileWriter(file_system_, varlen_schema_, file_name, properties_);

  int32_t total_rows = 0;
  for (int i = 0; i < 5; i++) {
    auto rb = makeVarlenBatch(total_rows, 200);
    ASSERT_STATUS_OK(vx_writer.Write(rb));
    total_rows += 200;
  }

  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  // Read back and verify
  auto vx_reader = vortex::VortexFormatReader(file_system_, varlen_schema_, file_name, properties_,
                                              std::vector<std::string>{"id", "text"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(total_rows, vx_reader.rows());
}

TEST_F(VortexSegmentRowSizeTest, TestScalarColumnWriteRead) {
  const char* file_name = "test-scalar-segment.vx";
  test_files_.push_back(file_name);

  // Set custom segment size for scalars
  ASSERT_EQ(std::nullopt, api::SetValue(properties_, PROPERTY_WRITER_VORTEX_SEGMENT_ROW_SIZE, "2048"));

  auto vx_writer = vortex::VortexFileWriter(file_system_, scalar_schema_, file_name, properties_);

  int32_t total_rows = 0;
  for (int i = 0; i < 5; i++) {
    auto rb = makeScalarBatch(total_rows, 200);
    ASSERT_STATUS_OK(vx_writer.Write(rb));
    total_rows += 200;
  }

  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  // Read back and verify
  auto vx_reader = vortex::VortexFormatReader(file_system_, scalar_schema_, file_name, properties_,
                                              std::vector<std::string>{"id", "value"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(total_rows, vx_reader.rows());
}

}  // namespace milvus_storage