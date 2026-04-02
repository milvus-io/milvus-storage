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
#include "milvus-storage/transaction/transaction.h"

namespace milvus_storage::segment {

class SegmentReaderTextTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/segment_reader_text_test_" + std::to_string(std::random_device{}());
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
    writer_config_.segment_path = test_dir_ + "/segments/seg-001";

    // TEXT column config
    lob_column::LobColumnConfig text_config;
    text_config.lob_base_path = test_dir_ + "/lobs/101";
    text_config.field_id = 101;
    text_config.inline_threshold = 20;
    text_config.max_lob_file_bytes = 1024 * 1024;
    text_config.flush_threshold_bytes = 64 * 1024;
    ASSERT_STATUS_OK(InitTestProperties(text_config.properties));

    writer_config_.lob_columns[101] = text_config;
    ASSERT_STATUS_OK(InitTestProperties(writer_config_.properties));
    writer_config_.properties[PROPERTY_WRITER_POLICY] = LOON_COLUMN_GROUP_POLICY_SINGLE;
    writer_config_.properties[PROPERTY_FORMAT] = LOON_FORMAT_PARQUET;

    // configure reader
    reader_config_.lob_columns = writer_config_.lob_columns;
    ASSERT_STATUS_OK(InitTestProperties(reader_config_.properties));
  }

  void TearDown() override { boost::filesystem::remove_all(test_dir_); }

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

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateTestBatch(int64_t num_rows,
                                                                     int64_t start_id = 0,
                                                                     bool use_lob = true) {
    arrow::Int64Builder id_builder;
    arrow::StringBuilder content_builder;
    arrow::DoubleBuilder value_builder;

    ARROW_RETURN_NOT_OK(id_builder.Reserve(num_rows));
    ARROW_RETURN_NOT_OK(content_builder.Reserve(num_rows));
    ARROW_RETURN_NOT_OK(value_builder.Reserve(num_rows));

    for (int64_t i = 0; i < num_rows; i++) {
      ARROW_RETURN_NOT_OK(id_builder.Append(start_id + i));
      ARROW_RETURN_NOT_OK(value_builder.Append(static_cast<double>(start_id + i) * 0.1));

      if (use_lob && i % 3 != 0) {
        // LOB text
        ARROW_RETURN_NOT_OK(content_builder.Append(GenerateRandomString(50 + i % 50)));
      } else {
        // inline text
        ARROW_RETURN_NOT_OK(content_builder.Append("short" + std::to_string(start_id + i)));
      }
    }

    std::shared_ptr<arrow::Int64Array> id_array;
    std::shared_ptr<arrow::StringArray> content_array;
    std::shared_ptr<arrow::DoubleArray> value_array;

    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
    ARROW_RETURN_NOT_OK(content_builder.Finish(&content_array));
    ARROW_RETURN_NOT_OK(value_builder.Finish(&value_array));

    return arrow::RecordBatch::Make(schema_, num_rows, {id_array, content_array, value_array});
  }

  // write test data and return committed version
  int64_t WriteTestData(int64_t num_rows) {
    auto writer_result = SegmentWriter::Create(fs_, schema_, writer_config_);
    EXPECT_TRUE(writer_result.ok()) << writer_result.status().message();
    auto writer = std::move(writer_result).ValueOrDie();

    auto batch_result = CreateTestBatch(num_rows);
    EXPECT_TRUE(batch_result.ok());
    EXPECT_TRUE(writer->Write(batch_result.ValueOrDie()).ok());

    auto close_result = writer->Close();
    EXPECT_TRUE(close_result.ok()) << close_result.status().message();
    auto output = std::move(close_result).ValueOrDie();

    auto txn_result = api::transaction::Transaction::Open(fs_, writer_config_.segment_path);
    EXPECT_TRUE(txn_result.ok()) << txn_result.status().message();
    auto txn = std::move(txn_result).ValueOrDie();
    txn->AppendFiles(*output.column_groups);
    for (const auto& lob : output.lob_files) {
      txn->AddLobFile(lob);
    }
    auto commit_result = txn->Commit();
    EXPECT_TRUE(commit_result.ok()) << commit_result.status().message();
    return commit_result.ValueOrDie();
  }

  // helper: open transaction and get manifest
  std::shared_ptr<api::Manifest> OpenManifest(const std::string& segment_path, int64_t version) {
    auto txn_result =
        api::transaction::Transaction::Open(fs_, segment_path, version, api::transaction::FailResolver, 1);
    EXPECT_TRUE(txn_result.ok()) << txn_result.status().message();
    auto txn = std::move(txn_result).ValueOrDie();
    auto manifest_result = txn->GetManifest();
    EXPECT_TRUE(manifest_result.ok()) << manifest_result.status().message();
    return std::move(manifest_result).ValueOrDie();
  }

  protected:
  std::string test_dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  SegmentWriterConfig writer_config_;
  SegmentReaderConfig reader_config_;
};

// ==================== Basic Read Tests ====================

// test reading all columns including TEXT
TEST_F(SegmentReaderTextTest, ReadAllColumns) {
  int64_t num_rows = 50;
  int64_t version = WriteTestData(num_rows);
  ASSERT_GT(version, 0);

  // open reader
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // read all batches
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;

    total_rows += batch->num_rows();

    // verify schema - TEXT column should be utf8 (resolved from binary)
    ASSERT_EQ(batch->schema()->field(1)->type()->id(), arrow::Type::STRING);

    // verify TEXT data is readable
    auto content_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    for (int64_t i = 0; i < batch->num_rows(); i++) {
      if (!content_array->IsNull(i)) {
        ASSERT_FALSE(content_array->GetString(i).empty());
      }
    }
  }

  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// test reading only TEXT column
TEST_F(SegmentReaderTextTest, ReadOnlyLobColumn) {
  int64_t num_rows = 30;
  int64_t version = WriteTestData(num_rows);

  // open reader with only TEXT column
  std::vector<std::string> columns = {"content"};
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // verify schema has only one column
  ASSERT_EQ(reader->schema()->num_fields(), 1);
  ASSERT_EQ(reader->schema()->field(0)->name(), "content");
  ASSERT_EQ(reader->schema()->field(0)->type()->id(), arrow::Type::STRING);

  // read all batches
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;

    total_rows += batch->num_rows();
    ASSERT_EQ(batch->num_columns(), 1);
  }

  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// test reading TEXT column with other columns
TEST_F(SegmentReaderTextTest, ReadTextWithOtherColumns) {
  int64_t num_rows = 40;
  int64_t version = WriteTestData(num_rows);

  // open reader with TEXT and id columns
  std::vector<std::string> columns = {"id", "content"};
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // verify schema
  ASSERT_EQ(reader->schema()->num_fields(), 2);
  ASSERT_EQ(reader->schema()->field(0)->name(), "id");
  ASSERT_EQ(reader->schema()->field(1)->name(), "content");

  // read and verify
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;

    total_rows += batch->num_rows();
    ASSERT_EQ(batch->num_columns(), 2);

    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto content_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));

    // verify data consistency
    for (int64_t i = 0; i < batch->num_rows(); i++) {
      int64_t id = id_array->Value(i);
      std::string content = content_array->GetString(i);

      // inline texts start with "short"
      if (id % 3 == 0) {
        ASSERT_TRUE(content.find("short") == 0) << "id=" << id << ", content=" << content;
      }
    }
  }

  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// test reading without TEXT column
TEST_F(SegmentReaderTextTest, ReadWithoutLobColumn) {
  int64_t num_rows = 25;
  int64_t version = WriteTestData(num_rows);

  // open reader without TEXT column
  std::vector<std::string> columns = {"id", "value"};
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // verify schema - no TEXT column
  ASSERT_EQ(reader->schema()->num_fields(), 2);
  ASSERT_EQ(reader->schema()->field(0)->name(), "id");
  ASSERT_EQ(reader->schema()->field(1)->name(), "value");

  // read all batches
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;

    total_rows += batch->num_rows();
    ASSERT_EQ(batch->num_columns(), 2);
  }

  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Take API Tests ====================

// test Take with TEXT column
TEST_F(SegmentReaderTextTest, TakeWithLobColumn) {
  int64_t num_rows = 100;
  int64_t version = WriteTestData(num_rows);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // take specific rows
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 99};
  auto take_result = reader->Take(row_indices);
  ASSERT_TRUE(take_result.ok()) << take_result.status().message();
  auto table = std::move(take_result).ValueOrDie();

  ASSERT_EQ(table->num_rows(), static_cast<int64_t>(row_indices.size()));
  ASSERT_EQ(table->num_columns(), 3);

  // verify TEXT column
  auto content_column = table->column(1);
  ASSERT_EQ(content_column->type()->id(), arrow::Type::STRING);

  ASSERT_STATUS_OK(reader->Close());
}

// test Take with only TEXT column selected
TEST_F(SegmentReaderTextTest, TakeOnlyLobColumn) {
  int64_t num_rows = 50;
  int64_t version = WriteTestData(num_rows);

  std::vector<std::string> columns = {"content"};
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<int64_t> row_indices = {5, 15, 25, 35, 45};
  auto take_result = reader->Take(row_indices);
  ASSERT_TRUE(take_result.ok()) << take_result.status().message();
  auto table = std::move(take_result).ValueOrDie();

  ASSERT_EQ(table->num_rows(), static_cast<int64_t>(row_indices.size()));
  ASSERT_EQ(table->num_columns(), 1);
  ASSERT_EQ(table->schema()->field(0)->name(), "content");

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Multiple Writes and Reads ====================

// test incremental writes and reads
TEST_F(SegmentReaderTextTest, IncrementalWritesAndReads) {
  int64_t version1 = 0, version2 = 0;

  // first write
  {
    auto writer_result = SegmentWriter::Create(fs_, schema_, writer_config_);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto batch_result = CreateTestBatch(20);
    ASSERT_TRUE(batch_result.ok());
    ASSERT_STATUS_OK(writer->Write(batch_result.ValueOrDie()));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    auto output = std::move(close_result).ValueOrDie();

    auto txn_result = api::transaction::Transaction::Open(fs_, writer_config_.segment_path);
    ASSERT_TRUE(txn_result.ok());
    auto txn = std::move(txn_result).ValueOrDie();
    txn->AppendFiles(*output.column_groups);
    for (const auto& lob : output.lob_files) {
      txn->AddLobFile(lob);
    }
    auto commit_result = txn->Commit();
    ASSERT_TRUE(commit_result.ok());
    version1 = commit_result.ValueOrDie();
  }

  // second write (incremental)
  {
    SegmentWriterConfig config2 = writer_config_;

    auto writer_result = SegmentWriter::Create(fs_, schema_, config2);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto batch_result = CreateTestBatch(30, 20);
    ASSERT_TRUE(batch_result.ok());
    ASSERT_STATUS_OK(writer->Write(batch_result.ValueOrDie()));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    auto output = std::move(close_result).ValueOrDie();

    auto txn_result = api::transaction::Transaction::Open(fs_, writer_config_.segment_path, version1);
    ASSERT_TRUE(txn_result.ok());
    auto txn = std::move(txn_result).ValueOrDie();
    txn->AppendFiles(*output.column_groups);
    for (const auto& lob : output.lob_files) {
      txn->AddLobFile(lob);
    }
    auto commit_result = txn->Commit();
    ASSERT_TRUE(commit_result.ok());
    version2 = commit_result.ValueOrDie();
  }

  ASSERT_GT(version2, version1);

  // read version1 - should have 20 rows
  {
    auto manifest1 = OpenManifest(writer_config_.segment_path, version1);
    auto reader_result = SegmentReader::Open(fs_, manifest1, schema_, {}, reader_config_);
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    int64_t total = 0;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_STATUS_OK(reader->ReadNext(&batch));
      if (!batch)
        break;
      total += batch->num_rows();
    }
    ASSERT_EQ(total, 20);
    ASSERT_STATUS_OK(reader->Close());
  }

  // read version2 - should have 50 rows (20 + 30)
  {
    auto manifest2 = OpenManifest(writer_config_.segment_path, version2);
    auto reader_result = SegmentReader::Open(fs_, manifest2, schema_, {}, reader_config_);
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    int64_t total = 0;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_STATUS_OK(reader->ReadNext(&batch));
      if (!batch)
        break;
      total += batch->num_rows();
    }
    ASSERT_EQ(total, 50);
    ASSERT_STATUS_OK(reader->Close());
  }
}

// ==================== NULL Handling Tests ====================

// test reading with NULL TEXT values
TEST_F(SegmentReaderTextTest, ReadWithNullTextValues) {
  // write batch with NULL TEXT values
  {
    auto writer_result = SegmentWriter::Create(fs_, schema_, writer_config_);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    // create batch with NULLs
    arrow::Int64Builder id_builder;
    arrow::StringBuilder content_builder;
    arrow::DoubleBuilder value_builder;

    ASSERT_STATUS_OK(id_builder.Append(1));
    ASSERT_STATUS_OK(content_builder.Append("text1"));
    ASSERT_STATUS_OK(value_builder.Append(1.0));

    ASSERT_STATUS_OK(id_builder.Append(2));
    ASSERT_STATUS_OK(content_builder.AppendNull());
    ASSERT_STATUS_OK(value_builder.Append(2.0));

    ASSERT_STATUS_OK(id_builder.Append(3));
    ASSERT_STATUS_OK(content_builder.Append(GenerateRandomString(50)));
    ASSERT_STATUS_OK(value_builder.Append(3.0));

    ASSERT_STATUS_OK(id_builder.Append(4));
    ASSERT_STATUS_OK(content_builder.AppendNull());
    ASSERT_STATUS_OK(value_builder.Append(4.0));

    std::shared_ptr<arrow::Int64Array> id_array;
    std::shared_ptr<arrow::StringArray> content_array;
    std::shared_ptr<arrow::DoubleArray> value_array;

    ASSERT_STATUS_OK(id_builder.Finish(&id_array));
    ASSERT_STATUS_OK(content_builder.Finish(&content_array));
    ASSERT_STATUS_OK(value_builder.Finish(&value_array));

    auto batch = arrow::RecordBatch::Make(schema_, 4, {id_array, content_array, value_array});
    ASSERT_STATUS_OK(writer->Write(batch));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    auto output = std::move(close_result).ValueOrDie();

    auto txn_result = api::transaction::Transaction::Open(fs_, writer_config_.segment_path);
    ASSERT_TRUE(txn_result.ok());
    auto txn = std::move(txn_result).ValueOrDie();
    txn->AppendFiles(*output.column_groups);
    for (const auto& lob : output.lob_files) {
      txn->AddLobFile(lob);
    }
    auto commit_result = txn->Commit();
    ASSERT_TRUE(commit_result.ok());
  }

  // read and verify NULLs are preserved
  {
    auto manifest = OpenManifest(writer_config_.segment_path, -1);
    auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->num_rows(), 4);

    auto content_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    ASSERT_FALSE(content_array->IsNull(0));
    ASSERT_TRUE(content_array->IsNull(1));
    ASSERT_FALSE(content_array->IsNull(2));
    ASSERT_TRUE(content_array->IsNull(3));

    ASSERT_STATUS_OK(reader->Close());
  }
}

// ==================== Large Data Tests ====================

// test reading large TEXT data
TEST_F(SegmentReaderTextTest, ReadLargeTextData) {
  // write large texts
  {
    auto writer_result = SegmentWriter::Create(fs_, schema_, writer_config_);
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    arrow::Int64Builder id_builder;
    arrow::StringBuilder content_builder;
    arrow::DoubleBuilder value_builder;

    for (int i = 0; i < 10; i++) {
      ASSERT_STATUS_OK(id_builder.Append(i));
      // 100KB text per row
      ASSERT_STATUS_OK(content_builder.Append(GenerateRandomString(100 * 1024)));
      ASSERT_STATUS_OK(value_builder.Append(i * 0.1));
    }

    std::shared_ptr<arrow::Int64Array> id_array;
    std::shared_ptr<arrow::StringArray> content_array;
    std::shared_ptr<arrow::DoubleArray> value_array;

    ASSERT_STATUS_OK(id_builder.Finish(&id_array));
    ASSERT_STATUS_OK(content_builder.Finish(&content_array));
    ASSERT_STATUS_OK(value_builder.Finish(&value_array));

    auto batch = arrow::RecordBatch::Make(schema_, 10, {id_array, content_array, value_array});
    ASSERT_STATUS_OK(writer->Write(batch));

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    auto output = std::move(close_result).ValueOrDie();

    auto txn_result = api::transaction::Transaction::Open(fs_, writer_config_.segment_path);
    ASSERT_TRUE(txn_result.ok());
    auto txn = std::move(txn_result).ValueOrDie();
    txn->AppendFiles(*output.column_groups);
    for (const auto& lob : output.lob_files) {
      txn->AddLobFile(lob);
    }
    auto commit_result = txn->Commit();
    ASSERT_TRUE(commit_result.ok());
  }

  // read and verify
  {
    auto manifest = OpenManifest(writer_config_.segment_path, -1);
    auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    int64_t total_rows = 0;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_STATUS_OK(reader->ReadNext(&batch));
      if (!batch)
        break;

      total_rows += batch->num_rows();

      auto content_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
      for (int64_t i = 0; i < batch->num_rows(); i++) {
        ASSERT_EQ(content_array->GetString(i).size(), 100 * 1024);
      }
    }

    ASSERT_EQ(total_rows, 10);
    ASSERT_STATUS_OK(reader->Close());
  }
}

// ==================== Create Factory Tests ====================

// test Create from ColumnGroups directly
TEST_F(SegmentReaderTextTest, CreateFromColumnGroups) {
  int64_t num_rows = 30;
  int64_t version = WriteTestData(num_rows);

  // open transaction to get column groups
  auto txn_result =
      api::transaction::Transaction::Open(fs_, writer_config_.segment_path, version, api::transaction::FailResolver, 1);
  ASSERT_TRUE(txn_result.ok()) << txn_result.status().message();
  auto txn = std::move(txn_result).ValueOrDie();

  auto manifest_result = txn->GetManifest();
  ASSERT_TRUE(manifest_result.ok());
  auto manifest = std::move(manifest_result).ValueOrDie();
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest->columnGroups());

  // create reader from ColumnGroups
  auto reader_result = SegmentReader::Create(fs_, column_groups, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  // read all rows
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// test Create with column selection
TEST_F(SegmentReaderTextTest, CreateWithColumnSelection) {
  int64_t num_rows = 20;
  int64_t version = WriteTestData(num_rows);

  auto txn_result =
      api::transaction::Transaction::Open(fs_, writer_config_.segment_path, version, api::transaction::FailResolver, 1);
  ASSERT_TRUE(txn_result.ok());
  auto txn = std::move(txn_result).ValueOrDie();
  auto manifest_result = txn->GetManifest();
  ASSERT_TRUE(manifest_result.ok());
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest_result.ValueOrDie()->columnGroups());

  std::vector<std::string> columns = {"id", "content"};
  auto reader_result = SegmentReader::Create(fs_, column_groups, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_EQ(reader->schema()->num_fields(), 2);
  ASSERT_EQ(reader->schema()->field(0)->name(), "id");
  ASSERT_EQ(reader->schema()->field(1)->name(), "content");

  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, num_rows);
  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Error Path Tests ====================

// test Create with null filesystem
TEST_F(SegmentReaderTextTest, CreateWithNullFs) {
  int64_t version = WriteTestData(10);

  auto txn_result =
      api::transaction::Transaction::Open(fs_, writer_config_.segment_path, version, api::transaction::FailResolver, 1);
  ASSERT_TRUE(txn_result.ok());
  auto txn = std::move(txn_result).ValueOrDie();
  auto manifest_result = txn->GetManifest();
  ASSERT_TRUE(manifest_result.ok());
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest_result.ValueOrDie()->columnGroups());

  auto result = SegmentReader::Create(nullptr, column_groups, schema_, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test Create with null schema
TEST_F(SegmentReaderTextTest, CreateWithNullSchema) {
  int64_t version = WriteTestData(10);

  auto txn_result =
      api::transaction::Transaction::Open(fs_, writer_config_.segment_path, version, api::transaction::FailResolver, 1);
  ASSERT_TRUE(txn_result.ok());
  auto txn = std::move(txn_result).ValueOrDie();
  auto manifest_result = txn->GetManifest();
  ASSERT_TRUE(manifest_result.ok());
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest_result.ValueOrDie()->columnGroups());

  auto result = SegmentReader::Create(fs_, column_groups, nullptr, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test Create with null column_groups
TEST_F(SegmentReaderTextTest, CreateWithNullColumnGroups) {
  auto result = SegmentReader::Create(fs_, nullptr, schema_, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test Open with null filesystem
TEST_F(SegmentReaderTextTest, OpenWithNullFs) {
  int64_t version = WriteTestData(10);
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto result = SegmentReader::Open(nullptr, manifest, schema_, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test Open with null schema
TEST_F(SegmentReaderTextTest, OpenWithNullSchema) {
  int64_t version = WriteTestData(10);
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto result = SegmentReader::Open(fs_, manifest, nullptr, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test Open with null manifest
TEST_F(SegmentReaderTextTest, OpenWithNullManifest) {
  auto result = SegmentReader::Open(fs_, nullptr, schema_, {}, reader_config_);
  ASSERT_FALSE(result.ok());
}

// test ReadNext on closed reader
TEST_F(SegmentReaderTextTest, ReadNextAfterClose) {
  int64_t version = WriteTestData(10);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());
  ASSERT_TRUE(reader->IsClosed());

  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = reader->ReadNext(&batch);
  ASSERT_FALSE(status.ok());
}

// test Take on closed reader
TEST_F(SegmentReaderTextTest, TakeAfterClose) {
  int64_t version = WriteTestData(10);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());

  std::vector<int64_t> indices = {0, 1};
  auto result = reader->Take(indices);
  ASSERT_FALSE(result.ok());
}

// test double Close is safe
TEST_F(SegmentReaderTextTest, DoubleClose) {
  int64_t version = WriteTestData(10);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());
  ASSERT_STATUS_OK(reader->Close());  // second close should be OK
}

// ==================== Take Edge Case Tests ====================

// test Take with empty indices returns empty table
TEST_F(SegmentReaderTextTest, TakeEmptyIndices) {
  int64_t version = WriteTestData(20);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<int64_t> empty_indices;
  auto take_result = reader->Take(empty_indices);
  ASSERT_TRUE(take_result.ok()) << take_result.status().message();
  auto table = std::move(take_result).ValueOrDie();

  ASSERT_EQ(table->num_rows(), 0);
  ASSERT_EQ(table->num_columns(), 3);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Accessor Tests ====================

// test GetOriginalSchema, GetExtractedColumns, GetTotalRows, IsClosed
TEST_F(SegmentReaderTextTest, Accessors) {
  int64_t num_rows = 20;
  int64_t version = WriteTestData(num_rows);

  // open with column subset
  std::vector<std::string> columns = {"id", "content"};
  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // GetOriginalSchema should return full schema
  auto original_schema = reader->GetOriginalSchema();
  ASSERT_EQ(original_schema->num_fields(), 3);
  ASSERT_EQ(original_schema->field(0)->name(), "id");
  ASSERT_EQ(original_schema->field(1)->name(), "content");
  ASSERT_EQ(original_schema->field(2)->name(), "value");

  // schema() should return extracted schema
  auto extracted_schema = reader->schema();
  ASSERT_EQ(extracted_schema->num_fields(), 2);

  // GetExtractedColumns
  auto& extracted_columns = reader->GetExtractedColumns();
  ASSERT_EQ(extracted_columns.size(), 2u);
  ASSERT_EQ(extracted_columns[0], "id");
  ASSERT_EQ(extracted_columns[1], "content");

  // IsClosed should be false before close
  ASSERT_FALSE(reader->IsClosed());

  // GetTotalRows should be 0 before reading
  ASSERT_EQ(reader->GetTotalRows(), 0);

  // read all data
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch)
      break;
  }

  // GetTotalRows should reflect total read
  ASSERT_EQ(reader->GetTotalRows(), num_rows);

  ASSERT_STATUS_OK(reader->Close());
  ASSERT_TRUE(reader->IsClosed());
}

// ==================== GetStream Tests ====================

// test GetStream without predicate
TEST_F(SegmentReaderTextTest, GetStreamNoPredicate) {
  int64_t num_rows = 30;
  int64_t version = WriteTestData(num_rows);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto stream_result = reader->GetStream();
  ASSERT_TRUE(stream_result.ok()) << stream_result.status().message();
  auto stream = std::move(stream_result).ValueOrDie();

  // read all batches from stream
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(stream->ReadNext(&batch));
    if (!batch)
      break;

    total_rows += batch->num_rows();

    // verify TEXT column is resolved (utf8 type)
    ASSERT_EQ(batch->schema()->field(1)->type()->id(), arrow::Type::STRING);
  }
  ASSERT_EQ(total_rows, num_rows);

  ASSERT_STATUS_OK(reader->Close());
}

// test GetStream on closed reader
TEST_F(SegmentReaderTextTest, GetStreamAfterClose) {
  int64_t version = WriteTestData(10);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());

  auto stream_result = reader->GetStream();
  ASSERT_FALSE(stream_result.ok());
}

// test GetStream without LOB columns (no wrapping)
TEST_F(SegmentReaderTextTest, GetStreamWithoutLobColumns) {
  int64_t num_rows = 20;
  int64_t version = WriteTestData(num_rows);

  // open reader with only non-LOB columns
  std::vector<std::string> columns = {"id", "value"};
  SegmentReaderConfig config_no_lob;
  ASSERT_STATUS_OK(InitTestProperties(config_no_lob.properties));
  // no lob_columns configured

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, columns, config_no_lob);
  ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
  auto reader = std::move(reader_result).ValueOrDie();

  auto stream_result = reader->GetStream();
  ASSERT_TRUE(stream_result.ok()) << stream_result.status().message();
  auto stream = std::move(stream_result).ValueOrDie();

  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(stream->ReadNext(&batch));
    if (!batch)
      break;
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, num_rows);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== GetChunkReader Tests ====================

// test GetChunkReader
TEST_F(SegmentReaderTextTest, GetChunkReader) {
  int64_t num_rows = 30;
  int64_t version = WriteTestData(num_rows);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // get chunk reader for column group 0
  auto chunk_reader_result = reader->GetChunkReader(0);
  ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().message();
  auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();

  // verify chunk reader is usable
  auto total_chunks = chunk_reader->total_number_of_chunks();
  ASSERT_GT(total_chunks, 0u);

  // read first chunk
  auto chunk_result = chunk_reader->get_chunk(0);
  ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().message();
  auto chunk = std::move(chunk_result).ValueOrDie();
  ASSERT_GT(chunk->num_rows(), 0);

  ASSERT_STATUS_OK(reader->Close());
}

// test GetChunkReader on closed reader
TEST_F(SegmentReaderTextTest, GetChunkReaderAfterClose) {
  int64_t version = WriteTestData(10);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());

  auto chunk_reader_result = reader->GetChunkReader(0);
  ASSERT_FALSE(chunk_reader_result.ok());
}

// ==================== GetColumnGroups Test ====================

// test GetColumnGroups returns valid column groups
TEST_F(SegmentReaderTextTest, GetColumnGroups) {
  int64_t version = WriteTestData(20);

  auto manifest = OpenManifest(writer_config_.segment_path, version);
  auto reader_result = SegmentReader::Open(fs_, manifest, schema_, {}, reader_config_);
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto column_groups = reader->GetColumnGroups();
  ASSERT_NE(column_groups, nullptr);

  ASSERT_STATUS_OK(reader->Close());
}

}  // namespace milvus_storage::segment
