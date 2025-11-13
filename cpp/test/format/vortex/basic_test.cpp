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

#ifdef BUILD_VORTEX_BRIDGE

#include <gtest/gtest.h>
#include <memory>
#include <random>
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

#include "test_util.h"
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
    columngroup_->paths = {test_file_name_};
    columngroup_->columns = {"int32", "int64", "binary"};

    InitTestProperties(properties_);

    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, config));
    auto& cache = LRUCache<ArrowFileSystemConfig, std::shared_ptr<ObjectStoreWrapper>>::getInstance();
    ASSERT_AND_ASSIGN(file_system_, cache.get(config, [](const ArrowFileSystemConfig& config) {
      return std::make_shared<ObjectStoreWrapper>(
          ObjectStoreWrapper::OpenObjectStore(config.storage_type, config.address, config.access_key_id,
                                              config.access_key_value, config.region, config.bucket_name));
    }));
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(test_file_name_);
    }
  }

  protected:
  ObjectStoreWrapper createVortexObjectStoreWrapper(const api::Properties& properties) {
    return ObjectStoreWrapper::OpenObjectStore(
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_STORAGE_TYPE).c_str(),
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_ADDRESS).c_str(),
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_ACCESS_KEY_ID).c_str(),
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_ACCESS_KEY_VALUE).c_str(),
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_REGION).c_str(),
        api::GetValueNoError<std::string>(properties, PROPERTY_FS_BUCKET_NAME).c_str());
  }

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
    std::uniform_int_distribution<T> dis(0, maxVal);

    std::vector<T> numbers;
    numbers.reserve(size);
    for (T i = 0; i < size; ++i) {
      numbers.emplace_back(dis(gen));
    }

    std::sort(numbers.begin(), numbers.end());
    return numbers;
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
  std::shared_ptr<ObjectStoreWrapper> file_system_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_bacths_;
  const char* test_file_name_ = "test-file.vx";
  api::Properties properties_;

  private:
  uint32_t count_each_loop_ = 100;
  uint32_t rand_strlen_ = 100;
  uint32_t record_batch_len_ = 10;
};

TEST_F(VortexBasicTest, TestBasicWrite) {
  auto vx_writer = vortex::VortexFileWriter(columngroup_, file_system_, schema_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_EQ(recordBatchsRows(), vx_writer.count());
  ASSERT_TRUE(vx_writer.Close().ok());
}

TEST_F(VortexBasicTest, TestBasicRead) {
  auto vx_writer = vortex::VortexFileWriter(columngroup_, file_system_, schema_, properties_);
  auto obs = createVortexObjectStoreWrapper(properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_EQ(recordBatchsRows(), vx_writer.count());
  ASSERT_TRUE(vx_writer.Close().ok());

  auto vx_reader =
      vortex::VortexFormatReader(obs, schema_, test_file_name_, std::vector<std::string>{"int32", "int64", "binary"});

  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, recordBatchsRows()));
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
  auto vx_writer = vortex::VortexFileWriter(columngroup_, file_system_, schema_, properties_);
  auto obs = createVortexObjectStoreWrapper(properties_);

  auto empty_rb = makeRecordBatch(0, 0, 0);
  ASSERT_TRUE(vx_writer.Write(empty_rb).ok());

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_EQ(0, vx_writer.count());
  ASSERT_TRUE(vx_writer.Close().ok());

  auto vx_reader =
      vortex::VortexFormatReader(obs, schema_, test_file_name_, std::vector<std::string>{"int32", "int64", "binary"});

  ASSERT_EQ(0, vx_reader.rows());

  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, vx_reader.rows()));
  ASSERT_EQ(0, chunked_array->num_chunks());
}

TEST_F(VortexBasicTest, TestReaderProjection) {
  auto vx_writer = vortex::VortexFileWriter(columngroup_, file_system_, schema_, properties_);
  auto obs = createVortexObjectStoreWrapper(properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_EQ(recordBatchsRows(), vx_writer.count());
  ASSERT_TRUE(vx_writer.Close().ok());

  // all projection
  {
    auto vx_reader =
        vortex::VortexFormatReader(obs, schema_, test_file_name_, std::vector<std::string>{"int32", "int64", "binary"});
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, recordBatchsRows()));
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

    auto vx_reader = vortex::VortexFormatReader(obs, projection_schema, test_file_name_,
                                                std::vector<std::string>{"int64", "binary", "int32"});
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, recordBatchsRows()));
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

    auto vx_reader =
        vortex::VortexFormatReader(obs, projection_schema, test_file_name_, std::vector<std::string>{"int64"});
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
  }

  // empty projection
  {
    auto projection_schema = arrow::schema({});
    auto vx_reader = vortex::VortexFormatReader(obs, projection_schema, test_file_name_, std::vector<std::string>{});
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(0, rb->num_columns());
  }
}

TEST_F(VortexBasicTest, TestBasicTake) {
  auto vx_writer = vortex::VortexFileWriter(columngroup_, file_system_, schema_, properties_);
  auto obs = createVortexObjectStoreWrapper(properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_EQ(recordBatchsRows(), vx_writer.count());
  ASSERT_TRUE(vx_writer.Close().ok());

  auto take_verify = [&](vortex::VortexFormatReader& vx_reader, const std::vector<int64_t>& row_indices,
                         int64_t expect_rows) {
    auto rb_status = vx_reader.take(row_indices);
    ASSERT_TRUE(rb_status.ok());
    auto rb = rb_status.ValueUnsafe();
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

  auto vx_reader =
      vortex::VortexFormatReader(obs, projection_schema, test_file_name_, std::vector<std::string>{"int32"});

  // take single row
  take_verify(vx_reader, std::move(randomNumbers<int64_t>(recordBatchsRows() - 1, 1)), 1);
  // 100 randowm rows
  take_verify(vx_reader, std::move(randomNumbers<int64_t>(recordBatchsRows() - 1, 100)), 100);
  // boundary Testing
  take_verify(vx_reader, {0, (int64_t)recordBatchsRows() - 1}, 2);
  // all index out of range
  ASSERT_FALSE(vx_reader.take({recordBatchsRows(), (int64_t)recordBatchsRows() + 1000}).ok());
  // one of index out of range, will be success
  take_verify(vx_reader, {0, (int64_t)recordBatchsRows() - 1, (int64_t)recordBatchsRows() + 1000}, 2);
  // take all range
  take_verify(vx_reader, rangeNumbers<int64_t>(0, recordBatchsRows()), recordBatchsRows());
}

}  // namespace milvus_storage

#endif