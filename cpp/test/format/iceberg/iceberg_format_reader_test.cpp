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

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/common/config.h"
#include "test_env.h"

namespace milvus_storage::iceberg {
namespace {

using namespace milvus_storage::api;

class IcebergFormatReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("iceberg-format-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    data_file_path_ = base_path_ + "/data.parquet";
    delete_file_path_ = base_path_ + "/pos-delete.parquet";
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  // Write a data file with columns: id (int64), value (string)
  void WriteDataFile(int64_t num_rows) {
    auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("value", arrow::utf8())});

    arrow::Int64Builder id_builder;
    arrow::StringBuilder value_builder;
    for (int64_t i = 0; i < num_rows; ++i) {
      ASSERT_STATUS_OK(id_builder.Append(i));
      ASSERT_STATUS_OK(value_builder.Append("row_" + std::to_string(i)));
    }

    std::shared_ptr<arrow::Array> id_array, value_array;
    ASSERT_STATUS_OK(id_builder.Finish(&id_array));
    ASSERT_STATUS_OK(value_builder.Finish(&value_array));

    auto batch = arrow::RecordBatch::Make(schema, num_rows, {id_array, value_array});

    ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(data_file_path_));
    ASSERT_AND_ASSIGN(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
    ASSERT_STATUS_OK(writer->WriteRecordBatch(*batch));
    ASSERT_STATUS_OK(writer->Close());
    ASSERT_STATUS_OK(sink->Close());

    data_num_rows_ = num_rows;
  }

  // Write a positional delete file with columns: file_path (string), pos (int64)
  void WritePositionalDeleteFile(const std::string& data_file_uri, const std::vector<int64_t>& deleted_positions) {
    auto schema = arrow::schema({arrow::field("file_path", arrow::utf8()), arrow::field("pos", arrow::int64())});

    arrow::StringBuilder path_builder;
    arrow::Int64Builder pos_builder;
    for (auto pos : deleted_positions) {
      ASSERT_STATUS_OK(path_builder.Append(data_file_uri));
      ASSERT_STATUS_OK(pos_builder.Append(pos));
    }

    std::shared_ptr<arrow::Array> path_array, pos_array;
    ASSERT_STATUS_OK(path_builder.Finish(&path_array));
    ASSERT_STATUS_OK(pos_builder.Finish(&pos_array));

    auto batch = arrow::RecordBatch::Make(schema, deleted_positions.size(), {path_array, pos_array});

    ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(delete_file_path_));
    ASSERT_AND_ASSIGN(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
    ASSERT_STATUS_OK(writer->WriteRecordBatch(*batch));
    ASSERT_STATUS_OK(writer->Close());
    ASSERT_STATUS_OK(sink->Close());
  }

  // Build delete metadata JSON for a positional delete file
  std::vector<uint8_t> MakeDeleteMetadataJson(const std::string& pos_delete_path) {
    std::string json = R"([{"path":")" + pos_delete_path + R"(","file_type":"position"}])";
    return {json.begin(), json.end()};
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  Properties properties_;
  std::string base_path_;
  std::string data_file_path_;
  std::string delete_file_path_;
  int64_t data_num_rows_ = 0;
};

// No deletes — IcebergFormatReader should return all rows
TEST_F(IcebergFormatReaderTest, NoDeletes) {
  WriteDataFile(10);

  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, {}};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ASSERT_GE(rg_infos.size(), 1);

  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  ASSERT_EQ(batch->num_rows(), data_num_rows_);
}

// Positional deletes — deleted rows should be excluded from get_chunk()
TEST_F(IcebergFormatReaderTest, PositionalDeleteGetChunk) {
  WriteDataFile(10);

  // Delete rows at positions 2, 5, 8
  std::vector<int64_t> deleted_pos = {2, 5, 8};
  WritePositionalDeleteFile(data_file_path_, deleted_pos);

  auto metadata = MakeDeleteMetadataJson(delete_file_path_);
  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));

  // Should have 10 - 3 = 7 rows
  ASSERT_EQ(batch->num_rows(), 7);

  // Verify the remaining row IDs
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
  ASSERT_NE(id_array, nullptr);
  std::vector<int64_t> expected_ids = {0, 1, 3, 4, 6, 7, 9};
  ASSERT_EQ(id_array->length(), static_cast<int64_t>(expected_ids.size()));
  for (size_t i = 0; i < expected_ids.size(); ++i) {
    ASSERT_EQ(id_array->Value(i), expected_ids[i]) << "Mismatch at index " << i;
  }
}

// Positional deletes — take() maps logical doc IDs to physical positions.
// With 10 rows [0..9] and deletes at {2,5,8}, the post-delete view is:
//   logical 0 -> physical 0 (id=0)
//   logical 1 -> physical 1 (id=1)
//   logical 2 -> physical 3 (id=3)  [physical 2 deleted]
//   logical 3 -> physical 4 (id=4)
//   logical 4 -> physical 6 (id=6)  [physical 5 deleted]
//   logical 5 -> physical 7 (id=7)
//   logical 6 -> physical 9 (id=9)  [physical 8 deleted]
TEST_F(IcebergFormatReaderTest, PositionalDeleteTake) {
  WriteDataFile(10);

  std::vector<int64_t> deleted_pos = {2, 5, 8};
  WritePositionalDeleteFile(data_file_path_, deleted_pos);

  auto metadata = MakeDeleteMetadataJson(delete_file_path_);
  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  // Logical doc IDs from the index perspective
  std::vector<int64_t> take_positions = {1, 2, 4, 6};
  ASSERT_AND_ASSIGN(auto table, reader->take(take_positions));

  ASSERT_EQ(table->num_rows(), 4);

  auto id_col = std::dynamic_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
  ASSERT_NE(id_col, nullptr);
  // logical 1 -> physical 1 (id=1)
  ASSERT_EQ(id_col->Value(0), 1);
  // logical 2 -> physical 3 (id=3)
  ASSERT_EQ(id_col->Value(1), 3);
  // logical 4 -> physical 6 (id=6)
  ASSERT_EQ(id_col->Value(2), 6);
  // logical 6 -> physical 9 (id=9)
  ASSERT_EQ(id_col->Value(3), 9);
}

// No deletes — take() passes through logical == physical
TEST_F(IcebergFormatReaderTest, TakeNoDeletes) {
  WriteDataFile(10);

  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, {}};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  std::vector<int64_t> take_positions = {0, 3, 9};
  ASSERT_AND_ASSIGN(auto table, reader->take(take_positions));
  ASSERT_EQ(table->num_rows(), 3);

  auto id_col = std::dynamic_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
  ASSERT_NE(id_col, nullptr);
  ASSERT_EQ(id_col->Value(0), 0);
  ASSERT_EQ(id_col->Value(1), 3);
  ASSERT_EQ(id_col->Value(2), 9);
}

// Delete file references a different data file — no rows deleted
TEST_F(IcebergFormatReaderTest, DeleteForDifferentFile) {
  WriteDataFile(5);

  // Delete file references a different data file URI
  WritePositionalDeleteFile("s3://other-bucket/other-file.parquet", {0, 1, 2});

  auto metadata = MakeDeleteMetadataJson(delete_file_path_);
  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  ASSERT_EQ(batch->num_rows(), 5);
}

// clone_reader() shares deletion state
TEST_F(IcebergFormatReaderTest, CloneSharesDeletes) {
  WriteDataFile(10);

  std::vector<int64_t> deleted_pos = {3, 7};
  WritePositionalDeleteFile(data_file_path_, deleted_pos);

  auto metadata = MakeDeleteMetadataJson(delete_file_path_);
  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                                      std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto cloned, reader->clone_reader());

  ASSERT_AND_ASSIGN(auto batch, cloned->get_chunk(0));
  ASSERT_EQ(batch->num_rows(), 8);  // 10 - 2 deleted
}

// Equality delete metadata should cause an error at open()
TEST_F(IcebergFormatReaderTest, EqualityDeleteRejected) {
  WriteDataFile(5);

  std::string json = R"([{"path":"eq-delete.parquet","file_type":"equality","equality_ids":[1]}])";
  std::vector<uint8_t> metadata(json.begin(), json.end());

  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  auto result = FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                     std::vector<std::string>{"id"}, nullptr);

  ASSERT_FALSE(result.ok());
  ASSERT_TRUE(result.status().ToString().find("Equality") != std::string::npos)
      << "Error should mention equality deletes: " << result.status().ToString();
}

// Invalid JSON metadata should cause an error
TEST_F(IcebergFormatReaderTest, InvalidJsonMetadata) {
  WriteDataFile(5);

  std::string json = "not-valid-json";
  std::vector<uint8_t> metadata(json.begin(), json.end());

  ColumnGroupFile file{data_file_path_, 0, data_num_rows_, metadata};
  auto result = FormatReader::create(nullptr, LOON_FORMAT_ICEBERG_TABLE, file, properties_,
                                     std::vector<std::string>{"id"}, nullptr);

  ASSERT_FALSE(result.ok());
}

}  // namespace
}  // namespace milvus_storage::iceberg
