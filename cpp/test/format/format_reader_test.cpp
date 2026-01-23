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

#include <gtest/gtest.h>

#include <random>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/testing/gtest_util.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class FormatReaderTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("format-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    // Create a simple test schema with field IDs required by packed writer
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());

    // Create test data
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_P(FormatReaderTest, ReadParquetWithoutMeta) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  // using arrow origin writer to write a parquet file
  auto parquet_writer_props = parquet::WriterProperties::Builder()
                                  .compression(parquet::Compression::ZSTD)
                                  ->enable_dictionary()
                                  ->enable_statistics()
                                  ->build();

  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(base_path_ + "/test.parquet"));

  ASSERT_AND_ASSIGN(auto parquet_writer, ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(),
                                                                            sink, parquet_writer_props));

  for (size_t i = 0; i < 10; i++) {
    ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
    ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*test_batch_));
  }

  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  ASSERT_AND_ASSIGN(auto format_reader,
                    FormatReader::create(schema_, LOON_FORMAT_PARQUET,
                                         api::ColumnGroupFile{.path = base_path_ + "/test.parquet",
                                                              .start_index = 0,
                                                              .end_index = test_batch_->num_rows() * 10,
                                                              .metadata = {}},
                                         properties_, std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto row_group_infos, format_reader->get_row_group_infos());
  ASSERT_EQ(row_group_infos.size(), 10);

  for (size_t i = 0; i < row_group_infos.size(); i++) {
    ASSERT_EQ(row_group_infos[i].start_offset, i * test_batch_->num_rows());
    ASSERT_EQ(row_group_infos[i].end_offset, (i + 1) * test_batch_->num_rows());
    ASSERT_GT(row_group_infos[i].memory_size, 0);
    ASSERT_AND_ASSIGN(auto rb, format_reader->get_chunk(i));
    ASSERT_EQ(rb->num_rows(), test_batch_->num_rows());
  }

  std::vector<int> rg_indices_in_file(row_group_infos.size());
  std::iota(rg_indices_in_file.begin(), rg_indices_in_file.end(), 0);
  ASSERT_AND_ASSIGN(auto rbs, format_reader->get_chunks(rg_indices_in_file));

  size_t total_size = 0;
  for (size_t i = 0; i < rbs.size(); i++) {
    total_size += rbs[i]->num_rows();
  }
  ASSERT_EQ(total_size, test_batch_->num_rows() * 10);
}

TEST_P(FormatReaderTest, TestReadWithRange) {
  std::string format = GetParam();
  ASSERT_AND_ASSIGN(auto two_cols_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, true}));
  ASSERT_AND_ASSIGN(auto id_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, false}));

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, two_cols_schema));
  auto writer = Writer::create(base_path_, two_cols_schema, std::move(policy), properties_);

  ASSERT_AND_ASSIGN(
      auto rb_2560_rows,
      CreateTestData(two_cols_schema /* schema */, 0, false /* randdata */, 2560 /* num_rows */, 1024 /* vector_dim */,
                     0 /* str_length */, std::array<bool, 4>{true, false, false, true} /*needed_columns */));
  ASSERT_OK(writer->write(rb_2560_rows));
  ASSERT_OK(writer->flush());

  auto cgs_result = writer->close();

  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();
  auto cg = std::find_if(cgs->begin(), cgs->end(),
                         [](const std::shared_ptr<ColumnGroup>& cg) { return cg->columns[0] == "id"; });
  ASSERT_NE(cg, cgs->end());
  ASSERT_EQ((*cg)->files.size(), 1);

  ASSERT_AND_ASSIGN(auto format_reader, FormatReader::create(id_schema, format, (*cg)->files[0], properties_,
                                                             std::vector<std::string>{"id"}, nullptr));

  for (int i = 0; i < 10; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis1(0, 2560 - 1);
    int start = dis1(gen);
    std::uniform_int_distribution<> dis2(start + 1, 2560);
    int end = dis2(gen);

    ASSERT_AND_ASSIGN(auto rb_reader, format_reader->read_with_range(start, end));
    ASSERT_AND_ASSIGN(auto rbs, rb_reader->ToRecordBatches());
    ASSERT_AND_ASSIGN(auto rb, arrow::ConcatenateRecordBatches(rbs));
    // verify rb
    ASSERT_EQ(end - start, rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    auto i64array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (size_t i = 0; i < rb->num_rows(); i++) {
      ASSERT_EQ(i64array->Value(i), (int64_t)(start + i));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(FormatReaderTestP,
                         FormatReaderTest,
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE));

}  // namespace milvus_storage::test