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
#include <cstring>

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
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

using namespace vortex;

class VortexTestBase : public ::testing::Test {
  protected:
  void CommonSetUp(uint32_t format_version) {
    schema_ = arrow::Table::FromRecordBatches({makeRecordBatch(0, 0, 0)}).ValueOrDie()->schema();
    record_bacths_ = makeRecordBatchs();
    columngroup_ = std::make_shared<api::ColumnGroup>();

    columngroup_->format = LOON_FORMAT_VORTEX;
    columngroup_->files = {{.path = test_file_name_}};
    columngroup_->columns = {"int32", "int64", "binary"};

    ASSERT_STATUS_OK(InitTestProperties(properties_));

    format_version_ = format_version;
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, std::to_string(format_version_).c_str());

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

  [[nodiscard]] inline int64_t recordBatchsRows() const {
    return (count_each_loop_ * (record_batch_len_ - 1)) * record_batch_len_ / 2;
  }

  [[nodiscard]] inline size_t recordBatchsSize() const { return record_bacths_.size(); }

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
  uint32_t format_version_ = 1;

  private:
  uint32_t count_each_loop_ = 100;
  uint32_t rand_strlen_ = 100;
  uint32_t record_batch_len_ = 10;
};

// Parameterized fixture: runs each test for both V1 and V2 format
class VortexBasicTest : public VortexTestBase, public ::testing::WithParamInterface<uint32_t> {
  void SetUp() override { CommonSetUp(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(V1V2,
                         VortexBasicTest,
                         ::testing::Values(1, 2),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return "V" + std::to_string(info.param);
                         });

TEST_P(VortexBasicTest, TestBasicWrite) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);
}

TEST_P(VortexBasicTest, TestBasicRead) {
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

TEST_P(VortexBasicTest, TestEmptyWriteRead) {
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

TEST_P(VortexBasicTest, TestReaderProjection) {
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

TEST_P(VortexBasicTest, TestBasicTake) {
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
  take_verify(vx_reader, randomNumbers<int64_t>(recordBatchsRows() - 1, 1), 1);
  // 100 randowm rows
  take_verify(vx_reader, randomNumbers<int64_t>(recordBatchsRows() - 1, 100), 100);
  // boundary Testing
  take_verify(vx_reader, {0, recordBatchsRows() - 1}, 2);
  // take all range
  take_verify(vx_reader, rangeNumbers<int64_t>(0, recordBatchsRows()), recordBatchsRows());
  // Note: vortex 0.56+ does not gracefully handle out-of-range indices (panics instead of returning error),
  // so we removed the out-of-range index tests.
}

TEST_P(VortexBasicTest, FooterSizeMatchesActualFile) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(vx_footer_size, 0u);
  ASSERT_GT(vx_file_size, vx_footer_size);

  // Verify file_size matches actual file
  ASSERT_AND_ASSIGN(auto file, file_system_->OpenInputFile(test_file_name_));
  ASSERT_AND_ASSIGN(auto actual_file_size, file->GetSize());
  EXPECT_EQ(vx_file_size, static_cast<uint64_t>(actual_file_size));

  // Read EOF (last 8 bytes): [version 2B][postscript_len 2B][magic "VTXF" 4B]
  ASSERT_AND_ASSIGN(auto eof_buf, file->ReadAt(actual_file_size - 8, 8));
  const uint8_t* eof = eof_buf->data();

  // Verify magic
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(eof + 4), 4), "VTXF");

  // Get postscript_len from EOF
  uint16_t postscript_len = 0;
  std::memcpy(&postscript_len, eof + 2, 2);

  // Read postscript to get the earliest segment offset (= start of footer)
  int64_t postscript_offset = actual_file_size - 8 - postscript_len;
  ASSERT_GT(postscript_offset, 0);

  std::cout << "vx_footer_size: " << vx_footer_size << std::endl;
  std::cout << "postscript_len: " << postscript_len << std::endl;

  // The footer spans from the earliest segment to the end of file.
  // The postscript contains segment descriptors with absolute offsets.
  // We can parse the postscript flatbuffer to find the earliest offset,
  // but a simpler sanity check: footer_size must be > postscript_len + 8 (postscript + EOF)
  // and footer_size must be < file_size - 4 (exclude file header magic).
  EXPECT_GT(vx_footer_size, static_cast<uint64_t>(postscript_len) + 8)
      << "footer_size should include postscript + EOF + segment data";
  EXPECT_LT(vx_footer_size, vx_file_size - 4) << "footer_size should be less than file_size minus header magic";
}

TEST_P(VortexBasicTest, FooterSizeNotMatch) {
  // Write a vortex file
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  auto vx_footer_size2 = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size2 = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(vx_footer_size2, 0u);
  ASSERT_GT(vx_file_size2, vx_footer_size2);

  auto verify_read = [&](uint64_t footer_size) {
    auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
    auto vxfile = VortexFile::Open((uint8_t*)fs_holder.get(), test_file_name_, vx_file_size2, footer_size);
    ASSERT_EQ(vxfile.RowCount(), static_cast<uint64_t>(recordBatchsRows()));

    auto vx_reader =
        vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_,
                                   std::vector<std::string>{"int32", "int64", "binary"}, vx_file_size2, footer_size);
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    auto i32array = std::dynamic_pointer_cast<arrow::Int32Array>(rb->column(0));
    for (int i = 0; i < i32array->length(); ++i) {
      ASSERT_EQ(i32array->Value(i), (int32_t)i);
    }
  };

  verify_read(1);

  // Case 2: footer_size too large (= file_size, reads entire file as initial read).
  // Vortex clamps to min(initial_read_size, file_size). Extra bytes get cached as segments.
  verify_read(vx_file_size2);
}

// V2-specific fixture: always uses format_version=2
class VortexV2Test : public VortexTestBase {
  void SetUp() override { CommonSetUp(2); }
};

TEST_F(VortexV2Test, TestV2RowGroupWrite) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_bacths_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto total_rows = recordBatchsRows();

  // --- blocking_read: full scan ---
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_,
                                              std::vector<std::string>{"int32", "int64", "binary"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, total_rows));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));

  ASSERT_EQ(total_rows, rb->num_rows());
  ASSERT_EQ(3, rb->num_columns());

  auto i32array = std::dynamic_pointer_cast<arrow::Int32Array>(rb->column(0));
  auto i64array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(1));
  for (int i = 0; i < i32array->length(); ++i) {
    ASSERT_EQ(i32array->Value(i), (int32_t)i);
    ASSERT_EQ(i64array->Value(i), (int64_t)i);
  }

  // --- get_chunk: per row-group read ---
  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  ASSERT_GT(rg_infos.size(), 1u);
  uint64_t offset = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
    auto chunk_i32 = std::dynamic_pointer_cast<arrow::Int32Array>(chunk_rb->column(0));
    ASSERT_EQ(chunk_i32->Value(0), static_cast<int32_t>(offset)) << "rg[" << i << "] first value mismatch";
    offset += chunk_rb->num_rows();
  }
  ASSERT_EQ(offset, total_rows);

  // --- take: random access ---
  auto proj_schema = arrow::schema(
      {arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto take_reader = vortex::VortexFormatReader(file_system_, proj_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"int32"});
  ASSERT_STATUS_OK(take_reader.open());

  std::vector<int64_t> take_indices = {0, 42, total_rows / 2, total_rows - 1};
  ASSERT_AND_ASSIGN(auto table, take_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), static_cast<int64_t>(take_indices.size()));
  auto take_i32 = std::dynamic_pointer_cast<arrow::Int32Array>(take_rb->column(0));
  for (size_t i = 0; i < take_indices.size(); ++i) {
    ASSERT_EQ(take_i32->Value(i), static_cast<int32_t>(take_indices[i]));
  }

  // --- read_with_range: partial range read ---
  uint64_t range_start = rg_infos[0].end_offset;
  uint64_t range_end = rg_infos[1].end_offset;
  ASSERT_AND_ASSIGN(auto range_reader, vx_reader.read_with_range(range_start, range_end));
  std::shared_ptr<arrow::RecordBatch> range_batch;
  int64_t range_rows = 0;
  while (true) {
    ASSERT_STATUS_OK(range_reader->ReadNext(&range_batch));
    if (!range_batch)
      break;
    if (range_rows == 0) {
      auto range_i32 = std::dynamic_pointer_cast<arrow::Int32Array>(range_batch->column(0));
      ASSERT_EQ(range_i32->Value(0), static_cast<int32_t>(range_start));
    }
    range_rows += range_batch->num_rows();
  }
  ASSERT_EQ(range_rows, static_cast<int64_t>(range_end - range_start));
}

// Test that inline_array_node enables sub-segment range reads.
// Writes FSB(512) data, takes 1 row, and asserts IO read bytes are small.
// Only runs in cloud (S3) environment where FilesystemMetrics are available.
TEST_P(VortexBasicTest, TestInlineArrayNodeSubSegmentRead) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "Sub-segment IO test requires cloud environment with FilesystemMetrics";
  }

  // Get cloud filesystem with metrics
  ASSERT_AND_ASSIGN(auto cloud_fs, GetFileSystem(properties_));
  auto observable = std::dynamic_pointer_cast<Observable>(cloud_fs);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  // Build schema with a single FSB(512) column
  auto fsb_schema = arrow::schema({arrow::field("embedding", arrow::fixed_size_binary(512), false,
                                                arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  // Write 10000 rows of FSB(512) data (~5MB)
  const int64_t num_rows = 10000;
  const int fsb_width = 512;
  std::string test_path = GetTestBasePath("vortex-inline-test") + "/test-fsb.vx";

  {
    auto vx_writer = vortex::VortexFileWriter(cloud_fs, fsb_schema, test_path, properties_);

    arrow::FixedSizeBinaryBuilder fsb_builder(arrow::fixed_size_binary(fsb_width));
    std::string row_data(fsb_width, '\0');
    std::mt19937_64 rng(42);
    for (int64_t i = 0; i < num_rows; ++i) {
      // Fill with random data to prevent compression from shrinking the file
      auto* p = reinterpret_cast<uint64_t*>(row_data.data());
      for (size_t j = 0; j < fsb_width / sizeof(uint64_t); ++j) {
        p[j] = rng();
      }
      ASSERT_TRUE(fsb_builder.Append(row_data).ok());
    }
    std::shared_ptr<arrow::Array> fsb_array;
    ASSERT_TRUE(fsb_builder.Finish(&fsb_array).ok());

    auto rb = arrow::RecordBatch::Make(fsb_schema, num_rows, {fsb_array});
    ASSERT_TRUE(vx_writer.Write(rb).ok());
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(num_rows, cgfile.end_index);
  }

  // Open reader and take 1 row
  auto vx_reader =
      vortex::VortexFormatReader(cloud_fs, fsb_schema, test_path, properties_, std::vector<std::string>{"embedding"});
  ASSERT_STATUS_OK(vx_reader.open());

  // Reset metrics before take
  metrics->Reset();

  std::vector<int64_t> indices = {42};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(indices));
  ASSERT_EQ(1, table->num_rows());

  // Check IO: reading 1 row of 512 bytes should read far less than the full file
  // Full file is ~5MB; sub-segment read should be well under 1MB
  auto read_bytes = metrics->GetReadBytes();
  ASSERT_EQ(read_bytes, fsb_width) << "Sub-segment read of 1 row should be exactly " << fsb_width << " bytes, got "
                                   << read_bytes;
}

// Test V2 row group splits with variable-size string data.
// Writes two batches of 8192 rows with different string sizes (128B and 512B),
// then verifies that row group splits reflect the byte-size-based partitioning.
TEST_F(VortexV2Test, TestV2RowGroupSplitsBySize) {
  const int64_t rows_per_batch = 8192;
  const size_t small_str_len = 128;
  const size_t large_str_len = 512;
  const uint64_t row_group_max_size = 1024 * 1024;  // 1MB

  auto str_schema = arrow::schema(
      {arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  std::string test_path = "test-v2-rg-splits.vx";

  // Helper to build a batch of fixed-length random strings
  auto make_string_batch = [&](size_t str_len, int64_t num_rows) {
    arrow::StringBuilder builder;
    std::string s(str_len, 'x');
    for (int64_t i = 0; i < num_rows; ++i) {
      EXPECT_TRUE(builder.Append(s).ok());
    }
    std::shared_ptr<arrow::Array> arr;
    EXPECT_TRUE(builder.Finish(&arr).ok());
    return arrow::RecordBatch::Make(str_schema, num_rows, {arr});
  };

  // Write: batch1 = 8192 rows * 128B, batch2 = 8192 rows * 512B
  {
    auto vx_writer = vortex::VortexFileWriter(file_system_, str_schema, test_path, properties_);
    ASSERT_TRUE(vx_writer.Write(make_string_batch(small_str_len, rows_per_batch)).ok());
    ASSERT_TRUE(vx_writer.Write(make_string_batch(large_str_len, rows_per_batch)).ok());
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(rows_per_batch * 2, cgfile.end_index);
  }

  // Read back and check row group infos
  auto vx_reader =
      vortex::VortexFormatReader(file_system_, str_schema, test_path, properties_, std::vector<std::string>{"str"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());

  // batch1: 8192 rows * 128B utf8 (132 B/row with offsets) ≈ 1.03 MB
  // batch2: 8192 rows * 512B utf8 (516 B/row with offsets) ≈ 4.03 MB
  // row_group_max_size = 1MB → expect 6 row groups:
  //   rg[0]: batch1 first ~1MB
  //   rg[1]: batch1 tail + batch2 start (cross-batch merge)
  //   rg[2..4]: batch2 middle chunks (~1MB each)
  //   rg[5]: batch2 tail (EOF remainder)
  ASSERT_EQ(rg_infos.size(), 6u);

  // Verify exact row ranges
  struct Expected {
    uint64_t start;
    uint64_t end;
  };
  std::vector<Expected> expected = {{0, 7944},      {7944, 10161},  {10161, 12194},
                                    {12194, 14227}, {14227, 16260}, {16260, 16384}};
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, expected[i].start) << "rg[" << i << "] start_offset mismatch";
    ASSERT_EQ(rg_infos[i].end_offset, expected[i].end) << "rg[" << i << "] end_offset mismatch";
  }

  // Verify memory_size is non-zero for all row groups
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
  }

  // Verify total rows
  uint64_t total_rows = 0;
  for (const auto& rg : rg_infos) {
    total_rows += (rg.end_offset - rg.start_offset);
  }
  ASSERT_EQ(total_rows, static_cast<uint64_t>(rows_per_batch * 2));

  // --- blocking_read: full scan and verify string lengths ---
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, rows_per_batch * 2));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), rows_per_batch * 2);
  auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(0));
  // batch1 rows should have small_str_len, batch2 rows should have large_str_len
  ASSERT_EQ(str_array->GetView(0).size(), small_str_len);
  ASSERT_EQ(str_array->GetView(rows_per_batch).size(), large_str_len);
  ASSERT_EQ(str_array->GetView(rows_per_batch * 2 - 1).size(), large_str_len);

  // --- get_chunk: per row-group read ---
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
  }

  // --- take: spot-check boundary rows ---
  std::vector<int64_t> take_indices = {0, rows_per_batch - 1, rows_per_batch, rows_per_batch * 2 - 1};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), 4);
  auto take_str = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(0));
  ASSERT_EQ(take_str->GetView(0).size(), small_str_len);  // first row of batch1
  ASSERT_EQ(take_str->GetView(1).size(), small_str_len);  // last row of batch1
  ASSERT_EQ(take_str->GetView(2).size(), large_str_len);  // first row of batch2
  ASSERT_EQ(take_str->GetView(3).size(), large_str_len);  // last row of batch2
}

TEST_F(VortexV2Test, TestV2RowGroupMultiColumnSplitsBySize) {
  const int64_t rows_per_batch = 8192;
  const size_t str_len_a = 256;
  const size_t str_len_b = 512;
  const size_t str_len_c = 1024;
  const uint64_t row_group_max_size = 10 * 1024 * 1024;  // 10MB

  auto multi_schema = arrow::schema({
      arrow::field("col_a", arrow::utf8(), false),
      arrow::field("col_b", arrow::utf8(), false),
      arrow::field("col_c", arrow::utf8(), false),
  });

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  std::string test_path = "test-v2-rg-multi-col.vx";

  // Helper to build a multi-column batch with fixed-length strings per column
  auto make_batch = [&](int64_t num_rows) {
    auto build_col = [&](size_t str_len, int64_t n) {
      arrow::StringBuilder builder;
      std::string s(str_len, 'x');
      for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(builder.Append(s).ok());
      }
      std::shared_ptr<arrow::Array> arr;
      EXPECT_TRUE(builder.Finish(&arr).ok());
      return arr;
    };
    return arrow::RecordBatch::Make(
        multi_schema, num_rows,
        {build_col(str_len_a, num_rows), build_col(str_len_b, num_rows), build_col(str_len_c, num_rows)});
  };

  // Write 4 batches of 8192 rows each
  // Per-row uncompressed: 256+4 + 512+4 + 1024+4 ≈ 1804 B/row (with offsets)
  // Per batch: 8192 * 1804 ≈ 14.1 MB → each batch > 10MB limit → gets split
  // Total: 4 * 8192 = 32768 rows ≈ 56.4 MB
  {
    auto vx_writer = vortex::VortexFileWriter(file_system_, multi_schema, test_path, properties_);
    for (int i = 0; i < 4; ++i) {
      ASSERT_TRUE(vx_writer.Write(make_batch(rows_per_batch)).ok());
    }
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(rows_per_batch * 4, cgfile.end_index);
  }

  // Read back and check row group infos
  auto vx_reader = vortex::VortexFormatReader(file_system_, multi_schema, test_path, properties_,
                                              std::vector<std::string>{"col_a", "col_b", "col_c"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());

  // Each batch ≈ 14.1 MB, limit 10MB → each batch should be split into ~2 row groups
  // 4 batches → expect ~6-8 row groups (with cross-batch merging and EOF tail)
  ASSERT_GT(rg_infos.size(), 4u) << "Should have more row groups than input batches";

  // Verify contiguous ranges and total rows
  ASSERT_EQ(rg_infos[0].start_offset, 0u);
  uint64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    uint64_t rg_rows = rg_infos[i].end_offset - rg_infos[i].start_offset;
    ASSERT_GT(rg_rows, 0u) << "rg[" << i << "] should have rows";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    total_rows += rg_rows;
    if (i > 0) {
      ASSERT_EQ(rg_infos[i].start_offset, rg_infos[i - 1].end_offset) << "rg[" << i << "] should be contiguous";
    }
  }
  ASSERT_EQ(total_rows, static_cast<uint64_t>(rows_per_batch * 4));

  // --- blocking_read: full scan and verify data ---
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, rows_per_batch * 4));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), rows_per_batch * 4);
  ASSERT_EQ(rb->num_columns(), 3);
  // Verify string lengths for each column
  auto col_a = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(0));
  auto col_b = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(1));
  auto col_c = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(2));
  ASSERT_EQ(col_a->GetView(0).size(), str_len_a);
  ASSERT_EQ(col_b->GetView(0).size(), str_len_b);
  ASSERT_EQ(col_c->GetView(0).size(), str_len_c);

  // --- get_chunk: per row-group read ---
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
    ASSERT_EQ(chunk_rb->num_columns(), 3);
  }

  // --- take: random access across batch boundaries ---
  std::vector<int64_t> take_indices = {0, rows_per_batch - 1, rows_per_batch, rows_per_batch * 3,
                                       rows_per_batch * 4 - 1};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), static_cast<int64_t>(take_indices.size()));
  // All rows should have the same string lengths (uniform data)
  auto take_col_a = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(0));
  auto take_col_c = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(2));
  for (int i = 0; i < take_rb->num_rows(); ++i) {
    ASSERT_EQ(take_col_a->GetView(i).size(), str_len_a) << "take row " << i << " col_a size mismatch";
    ASSERT_EQ(take_col_c->GetView(i).size(), str_len_c) << "take row " << i << " col_c size mismatch";
  }
}

}  // namespace milvus_storage