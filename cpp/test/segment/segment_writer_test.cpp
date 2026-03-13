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

#ifdef BUILD_VORTEX_BRIDGE

#include <gtest/gtest.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <boost/filesystem.hpp>

#include <random>
#include <string>
#include <vector>

#include "test_env.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/segment/segment_writer.h"
#include "milvus-storage/segment/segment_reader.h"

namespace milvus_storage::segment {

class SegmentWriterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/segment_writer_test_" + std::to_string(std::random_device{}());
    boost::filesystem::create_directories(test_dir_);

    // create filesystem
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

    // create test schema with TEXT column
    // field IDs: id=100, content=101, value=102
    schema_ = arrow::schema({
        arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("content", arrow::utf8(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
        arrow::field("value", arrow::float64(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
    });

    // configure writer
    // lob_base_path is for LOB files, segment_path is for manifest and parquet data
    config_.lob_base_path = test_dir_ + "/lobs";
    config_.segment_path = test_dir_ + "/segments/seg-001";

    // TEXT column config (field_id=101 is the TEXT column)
    text_column::TextColumnConfig text_config;
    text_config.lob_base_path = test_dir_ + "/lobs/101";
    text_config.field_id = 101;
    text_config.inline_threshold = 20;  // small threshold for testing
    text_config.max_lob_file_bytes = 1024 * 1024;
    text_config.flush_threshold_bytes = 64 * 1024;
    ASSERT_STATUS_OK(InitTestProperties(text_config.properties));

    config_.text_columns[101] = text_config;

    // properties for ColumnGroupPolicy (single policy - all columns in one group)
    ASSERT_STATUS_OK(InitTestProperties(config_.properties));
    config_.properties[PROPERTY_WRITER_POLICY] = LOON_COLUMN_GROUP_POLICY_SINGLE;
    config_.properties[PROPERTY_FORMAT] = LOON_FORMAT_PARQUET;
  }

  void TearDown() override {
    // cleanup test directory
    boost::filesystem::remove_all(test_dir_);
  }

  std::string GenerateRandomString(size_t length) {
    static const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; i++) {
      result.push_back(charset[dis(gen)]);
    }
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateTestBatch(int64_t num_rows, int64_t start_id = 0) {
    // build id array
    arrow::Int64Builder id_builder;
    ARROW_RETURN_NOT_OK(id_builder.Reserve(num_rows));
    for (int64_t i = 0; i < num_rows; i++) {
      ARROW_RETURN_NOT_OK(id_builder.Append(start_id + i));
    }
    std::shared_ptr<arrow::Int64Array> id_array;
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));

    // build content array (mix of inline and LOB)
    arrow::StringBuilder content_builder;
    ARROW_RETURN_NOT_OK(content_builder.Reserve(num_rows));
    for (int64_t i = 0; i < num_rows; i++) {
      if (i % 3 == 0) {
        // inline text
        ARROW_RETURN_NOT_OK(content_builder.Append("short" + std::to_string(i)));
      } else {
        // LOB text
        ARROW_RETURN_NOT_OK(content_builder.Append(GenerateRandomString(50 + i % 50)));
      }
    }
    std::shared_ptr<arrow::StringArray> content_array;
    ARROW_RETURN_NOT_OK(content_builder.Finish(&content_array));

    // build value array
    arrow::DoubleBuilder value_builder;
    ARROW_RETURN_NOT_OK(value_builder.Reserve(num_rows));
    for (int64_t i = 0; i < num_rows; i++) {
      ARROW_RETURN_NOT_OK(value_builder.Append(static_cast<double>(i) * 0.1));
    }
    std::shared_ptr<arrow::DoubleArray> value_array;
    ARROW_RETURN_NOT_OK(value_builder.Finish(&value_array));

    return arrow::RecordBatch::Make(schema_, num_rows, {id_array, content_array, value_array});
  }

  protected:
  std::string test_dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  SegmentWriterConfig config_;
};

// test writer creation
TEST_F(SegmentWriterTest, CreateWriter) {
  auto result = SegmentWriter::Create(fs_, schema_, config_);
  ASSERT_TRUE(result.ok()) << result.status().message();

  auto writer = std::move(result).ValueOrDie();
  ASSERT_NE(writer, nullptr);
  ASSERT_FALSE(writer->IsClosed());
  ASSERT_EQ(writer->WrittenRows(), 0);

  // verify storage schema has binary type for TEXT column
  auto storage_schema = writer->GetStorageSchema();
  ASSERT_EQ(storage_schema->field(1)->type()->id(), arrow::Type::BINARY);

  // original schema should have utf8
  auto original_schema = writer->GetOriginalSchema();
  ASSERT_EQ(original_schema->field(1)->type()->id(), arrow::Type::STRING);
}

// test writer creation with invalid config
TEST_F(SegmentWriterTest, CreateWriterInvalidConfig) {
  // null filesystem
  auto result1 = SegmentWriter::Create(nullptr, schema_, config_);
  ASSERT_FALSE(result1.ok());

  // null schema
  auto result2 = SegmentWriter::Create(fs_, nullptr, config_);
  ASSERT_FALSE(result2.ok());

  // empty lob_base_path when text_columns is not empty
  SegmentWriterConfig invalid_config = config_;
  invalid_config.lob_base_path = "";
  auto result3 = SegmentWriter::Create(fs_, schema_, invalid_config);
  ASSERT_FALSE(result3.ok());

  // empty segment path
  invalid_config = config_;
  invalid_config.segment_path = "";
  auto result4 = SegmentWriter::Create(fs_, schema_, invalid_config);
  ASSERT_FALSE(result4.ok());

  // missing writer policy
  invalid_config = config_;
  invalid_config.properties.erase(PROPERTY_WRITER_POLICY);
  auto result5 = SegmentWriter::Create(fs_, schema_, invalid_config);
  ASSERT_FALSE(result5.ok());
}

// test writing single batch
TEST_F(SegmentWriterTest, WriteSingleBatch) {
  auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
  ASSERT_TRUE(writer_result.ok()) << writer_result.status().message();
  auto writer = std::move(writer_result).ValueOrDie();

  // create test batch
  auto batch_result = CreateTestBatch(10);
  ASSERT_TRUE(batch_result.ok()) << batch_result.status().message();
  auto batch = std::move(batch_result).ValueOrDie();

  // write batch
  auto write_status = writer->Write(batch);
  ASSERT_TRUE(write_status.ok()) << write_status.message();
  ASSERT_EQ(writer->WrittenRows(), 10);

  // close writer (commits transaction)
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok()) << close_result.status().message();
  ASSERT_TRUE(writer->IsClosed());

  auto result = std::move(close_result).ValueOrDie();

  // verify result
  ASSERT_EQ(result.rows_written, 10);
  ASSERT_GT(result.committed_version, 0);
  ASSERT_FALSE(result.manifest_path.empty());

  // verify manifest file exists
  ASSERT_TRUE(boost::filesystem::exists(result.manifest_path));

  // check stats
  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_rows, 10);
  ASSERT_GE(stats.parquet_files_created, 1);
}

// test writing multiple batches
TEST_F(SegmentWriterTest, WriteMultipleBatches) {
  auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write multiple batches
  for (int i = 0; i < 5; i++) {
    auto batch_result = CreateTestBatch(20, i * 20);
    ASSERT_TRUE(batch_result.ok());
    auto batch = std::move(batch_result).ValueOrDie();

    auto write_status = writer->Write(batch);
    ASSERT_TRUE(write_status.ok()) << write_status.message();
  }

  ASSERT_EQ(writer->WrittenRows(), 100);

  // close writer
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok()) << close_result.status().message();

  auto result = std::move(close_result).ValueOrDie();
  ASSERT_EQ(result.rows_written, 100);

  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_rows, 100);
}

// test write and read round trip using Open (from manifest)
TEST_F(SegmentWriterTest, WriteAndReadFromManifest) {
  // prepare test data
  auto batch_result = CreateTestBatch(50);
  ASSERT_TRUE(batch_result.ok());
  auto original_batch = std::move(batch_result).ValueOrDie();

  int64_t committed_version = 0;

  // write
  {
    auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
    ASSERT_TRUE(writer_result.ok()) << writer_result.status().message();
    auto writer = std::move(writer_result).ValueOrDie();

    ASSERT_STATUS_OK(writer->Write(original_batch));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok()) << close_result.status().message();

    auto result = std::move(close_result).ValueOrDie();
    committed_version = result.committed_version;
    ASSERT_GT(committed_version, 0);
  }

  // read using Open (from manifest)
  {
    SegmentReaderConfig reader_config;
    reader_config.text_columns = config_.text_columns;
    ASSERT_STATUS_OK(InitTestProperties(reader_config.properties));

    // open from manifest at segment path (latest version)
    auto reader_result = SegmentReader::Open(fs_, config_.segment_path, -1, schema_, {}, reader_config);
    ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
    auto reader = std::move(reader_result).ValueOrDie();

    // verify version
    ASSERT_EQ(reader->GetVersion(), committed_version);

    // read all batches
    std::vector<std::shared_ptr<arrow::RecordBatch>> read_batches;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      auto read_status = reader->ReadNext(&batch);
      ASSERT_TRUE(read_status.ok()) << read_status.message();

      if (!batch) {
        break;
      }
      read_batches.push_back(batch);
    }

    ASSERT_FALSE(read_batches.empty());

    // concatenate read batches and verify
    int64_t total_rows = 0;
    for (const auto& batch : read_batches) {
      total_rows += batch->num_rows();
    }
    ASSERT_EQ(total_rows, original_batch->num_rows());

    // verify schema
    ASSERT_TRUE(reader->schema()->Equals(schema_));

    // verify data for first batch
    auto first_batch = read_batches[0];
    ASSERT_EQ(first_batch->schema()->field(1)->type()->id(), arrow::Type::STRING);

    // close reader
    ASSERT_STATUS_OK(reader->Close());
  }
}

// test incremental writes (multiple commits)
TEST_F(SegmentWriterTest, IncrementalWrites) {
  int64_t version1 = 0, version2 = 0;

  // first write
  {
    auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto batch_result = CreateTestBatch(10);
    ASSERT_TRUE(batch_result.ok());
    ASSERT_STATUS_OK(writer->Write(batch_result.ValueOrDie()));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    version1 = close_result.ValueOrDie().committed_version;
  }

  // second write (reads from version1, writes version2)
  {
    SegmentWriterConfig config2 = config_;
    config2.read_version = version1;

    auto writer_result = SegmentWriter::Create(fs_, schema_, config2);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    ASSERT_EQ(writer->GetReadVersion(), version1);

    auto batch_result = CreateTestBatch(20, 10);
    ASSERT_TRUE(batch_result.ok());
    ASSERT_STATUS_OK(writer->Write(batch_result.ValueOrDie()));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    version2 = close_result.ValueOrDie().committed_version;
  }

  ASSERT_GT(version2, version1);

  // read version2 - should have data from both writes
  {
    SegmentReaderConfig reader_config;
    reader_config.text_columns = config_.text_columns;
    ASSERT_STATUS_OK(InitTestProperties(reader_config.properties));

    auto reader_result = SegmentReader::Open(fs_, config_.segment_path, version2, schema_, {}, reader_config);
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    ASSERT_EQ(reader->GetVersion(), version2);

    // count total rows
    int64_t total_rows = 0;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_STATUS_OK(reader->ReadNext(&batch));
      if (!batch)
        break;
      total_rows += batch->num_rows();
    }

    // should have 10 + 20 = 30 rows
    ASSERT_EQ(total_rows, 30);

    ASSERT_STATUS_OK(reader->Close());
  }
}

// test abort
TEST_F(SegmentWriterTest, WriterAbort) {
  auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write some data
  auto batch_result = CreateTestBatch(20);
  ASSERT_TRUE(batch_result.ok());
  auto batch = std::move(batch_result).ValueOrDie();

  ASSERT_STATUS_OK(writer->Write(batch));

  // abort instead of close
  auto abort_status = writer->Abort();
  ASSERT_TRUE(abort_status.ok());
  ASSERT_TRUE(writer->IsClosed());

  // verify no manifest was created (transaction was not committed)
  // manifest directory at segment level might exist but should be empty or not contain new version
}

// test writing with null values in TEXT column
TEST_F(SegmentWriterTest, WriteWithNullText) {
  auto writer_result = SegmentWriter::Create(fs_, schema_, config_);
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // build batch with null TEXT values
  arrow::Int64Builder id_builder;
  arrow::StringBuilder content_builder;
  arrow::DoubleBuilder value_builder;

  ASSERT_STATUS_OK(id_builder.Append(1));
  ASSERT_STATUS_OK(content_builder.Append("hello"));
  ASSERT_STATUS_OK(value_builder.Append(1.0));

  ASSERT_STATUS_OK(id_builder.Append(2));
  ASSERT_STATUS_OK(content_builder.AppendNull());  // null TEXT
  ASSERT_STATUS_OK(value_builder.Append(2.0));

  ASSERT_STATUS_OK(id_builder.Append(3));
  ASSERT_STATUS_OK(content_builder.Append(GenerateRandomString(100)));  // LOB
  ASSERT_STATUS_OK(value_builder.Append(3.0));

  std::shared_ptr<arrow::Int64Array> id_array;
  std::shared_ptr<arrow::StringArray> content_array;
  std::shared_ptr<arrow::DoubleArray> value_array;

  ASSERT_STATUS_OK(id_builder.Finish(&id_array));
  ASSERT_STATUS_OK(content_builder.Finish(&content_array));
  ASSERT_STATUS_OK(value_builder.Finish(&value_array));

  auto batch = arrow::RecordBatch::Make(schema_, 3, {id_array, content_array, value_array});

  // write batch
  ASSERT_STATUS_OK(writer->Write(batch));

  // close
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok()) << close_result.status().message();
}

// test schema without TEXT columns
TEST_F(SegmentWriterTest, SchemaWithoutTextColumns) {
  // create schema without TEXT columns
  auto no_text_schema = arrow::schema({
      arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("value", arrow::float64(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
  });

  // config without TEXT columns (lob_base_path can be empty)
  SegmentWriterConfig no_text_config;
  no_text_config.segment_path = test_dir_ + "/no_text/segments/seg-001";
  boost::filesystem::create_directories(test_dir_ + "/no_text");
  ASSERT_STATUS_OK(InitTestProperties(no_text_config.properties));
  no_text_config.properties[PROPERTY_WRITER_POLICY] = LOON_COLUMN_GROUP_POLICY_SINGLE;
  no_text_config.properties[PROPERTY_FORMAT] = LOON_FORMAT_PARQUET;

  auto writer_result = SegmentWriter::Create(fs_, no_text_schema, no_text_config);
  ASSERT_TRUE(writer_result.ok()) << writer_result.status().message();
  auto writer = std::move(writer_result).ValueOrDie();

  // storage schema should be same as original (no TEXT columns)
  ASSERT_TRUE(writer->GetStorageSchema()->Equals(writer->GetOriginalSchema()));

  // write data
  arrow::Int64Builder id_builder;
  arrow::DoubleBuilder value_builder;
  for (int i = 0; i < 10; i++) {
    ASSERT_STATUS_OK(id_builder.Append(i));
    ASSERT_STATUS_OK(value_builder.Append(i * 0.5));
  }
  std::shared_ptr<arrow::Int64Array> id_array;
  std::shared_ptr<arrow::DoubleArray> value_array;
  ASSERT_STATUS_OK(id_builder.Finish(&id_array));
  ASSERT_STATUS_OK(value_builder.Finish(&value_array));

  auto batch = arrow::RecordBatch::Make(no_text_schema, 10, {id_array, value_array});
  ASSERT_STATUS_OK(writer->Write(batch));

  // close
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok()) << close_result.status().message();

  auto result = std::move(close_result).ValueOrDie();
  ASSERT_GT(result.committed_version, 0);
}

}  // namespace milvus_storage::segment

#endif  // BUILD_VORTEX_BRIDGE
