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
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

class VortexTestBase : public ::testing::Test {
  protected:
  void CommonSetUp(uint32_t format_version) {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, std::to_string(format_version).c_str());

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema(needed_columns_));
    for (int64_t batch_idx = 0; batch_idx < batch_count_; ++batch_idx) {
      ASSERT_AND_ASSIGN(auto rb, MakeTestData(batch_idx * rows_per_batch_, rows_per_batch_));
      record_batches_.emplace_back(std::move(rb));
    }

    ASSERT_AND_ASSIGN(file_system_, GetFileSystem(properties_));
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(test_file_name_);
    }
  }

  protected:
  [[nodiscard]] inline int64_t recordBatchsRows() const { return batch_count_ * rows_per_batch_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeTestData(int64_t start_offset, size_t num_rows) {
    return CreateTestData(schema_, start_offset, false, num_rows, 4, 50, needed_columns_);
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

  const std::vector<std::string>& data_columns() const { return data_columns_; }

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
  const char* test_file_name_ = "test-file.vx";
  api::Properties properties_;

  private:
  const std::array<bool, 4> needed_columns_ = {true, true, true, false};
  const std::vector<std::string> data_columns_ = {"id", "name", "value"};
  const int64_t rows_per_batch_ = 1024;
  const int64_t batch_count_ = 4;
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

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);
}

TEST_P(VortexBasicTest, TestBasicRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));

  ASSERT_EQ(recordBatchsRows(), rb->num_rows());
  ASSERT_EQ(3, rb->num_columns());
  ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
  ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
  ASSERT_EQ(arrow::Type::DOUBLE, rb->column(2)->type_id());

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(rb->column(2));

  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
    ASSERT_DOUBLE_EQ(value_array->Value(i), static_cast<double>(i) * 1.5);
  }
}

TEST_P(VortexBasicTest, TestEmptyWriteRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  ASSERT_AND_ASSIGN(auto empty_rb, MakeTestData(0, 0));
  ASSERT_TRUE(vx_writer.Write(empty_rb).ok());

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(0, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(0, vx_reader.rows());

  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, vx_reader.rows()));
  ASSERT_EQ(0, chunked_array->num_chunks());
}

TEST_P(VortexBasicTest, TestReaderProjection) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  // all projection
  {
    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());
    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::DOUBLE, rb->column(2)->type_id());
  }

  // projection with different order
  {
    auto projection_schema = arrow::schema({
        schema_->field(2),
        schema_->field(1),
        schema_->field(0),
    });

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"value", "name", "id"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    ASSERT_EQ(arrow::Type::DOUBLE, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::INT64, rb->column(2)->type_id());
  }

  // single projection
  {
    auto projection_schema = arrow::schema({schema_->field(0)});

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"id"});
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

  for (const auto& rb : record_batches_) {
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

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (size_t i = 0; i < rb->num_rows(); i++) {
      ASSERT_EQ(id_array->Value(i), row_indices[i]);
    }
  };

  auto projection_schema = arrow::schema({schema_->field(0)});

  auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"id"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto one_random_row, GenerateSortedUniqueArray(1, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto random_rows, GenerateSortedUniqueArray(100, recordBatchsRows()));
  std::vector<int64_t> all_rows(recordBatchsRows());
  std::iota(all_rows.begin(), all_rows.end(), 0);

  // take single row
  take_verify(vx_reader, one_random_row, 1);
  // 100 random rows
  take_verify(vx_reader, random_rows, 100);
  // boundary Testing
  take_verify(vx_reader, {0, recordBatchsRows() - 1}, 2);
  // take all range
  take_verify(vx_reader, all_rows, recordBatchsRows());
  // Note: vortex 0.56+ does not gracefully handle out-of-range indices (panics instead of returning error),
  // so we removed the out-of-range index tests.
}

TEST_P(VortexBasicTest, FooterSizeMatchesActualFile) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
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
  for (const auto& rb : record_batches_) {
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
    auto vxfile =
        VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size2, footer_size);
    ASSERT_EQ(vxfile.RowCount(), static_cast<uint64_t>(recordBatchsRows()));

    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                                vx_file_size2, footer_size);
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (int i = 0; i < id_array->length(); ++i) {
      ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
    }
  };

  verify_read(1);

  // Case 2: footer_size too large (= file_size, reads entire file as initial read).
  // Vortex clamps to min(initial_read_size, file_size). Extra bytes get cached as segments.
  verify_read(vx_file_size2);
}

}  // namespace milvus_storage
