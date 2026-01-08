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

#include <unistd.h>
#include <algorithm>
#include <memory>
#include <random>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/api.h>
#include <arrow/testing/gtest_util.h>

#include "include/test_env.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/column_group_writer.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

class APIWriterReaderTest : public ::testing::TestWithParam<std::tuple<std::string, size_t>> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("api-writer-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    // Create a simple test schema with field IDs required by packed writer
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());

    // Create test data
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));

    // Get format
    format = std::get<0>(GetParam());

    // Initialize thread pool
    ThreadPoolHolder::WithSingleton(std::get<1>(GetParam()));
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ThreadPoolHolder::Release();
  }

  protected:
  std::string format;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_P(APIWriterReaderTest, SingleColumnGroupWriteRead) {
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  EXPECT_EQ(cgs->size(), 1);
  auto column_groups = *cgs;
  EXPECT_EQ(column_groups[0]->format, format);
  EXPECT_EQ(column_groups[0]->columns.size(), 4);

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);

  ASSERT_NE(reader, nullptr);

  // Test get_record_batch_reader (uses PackedRecordBatchReader internally)
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_OK(batch_reader->ReadNext(&batch));
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_rows(), 100);
  EXPECT_EQ(batch->num_columns(), 4);

  // Verify data content matches
  // EXPECT_TRUE(batch->Equals(*test_batch_, false));

  // Read until end to verify complete data
  ASSERT_OK(batch_reader->ReadNext(&batch));
  EXPECT_EQ(batch, nullptr);  // Should be at end
}

TEST_P(APIWriterReaderTest, SchemaBasedColumnGroupWriteRead) {
  // Test writing with SchemaBasedColumnGroupPolicy
  std::string patterns = "id|value,name,vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify cgs has multiple column groups
  auto column_groups = *cgs;
  EXPECT_EQ(column_groups.size(), 3);

  // Test reading without column projection first (column groups may not contain all columns)
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Test chunk reader
  auto chunk_reader_result = reader->get_chunk_reader(0);
  ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
  auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();
  ASSERT_NE(chunk_reader, nullptr);
  auto chunk_result = chunk_reader->get_chunk(0);
  ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().ToString();
  auto chunk = std::move(chunk_result).ValueOrDie();
  ASSERT_NE(chunk, nullptr);
  EXPECT_GT(chunk->num_rows(), 0);
}

TEST_P(APIWriterReaderTest, SizeBasedColumnGroupPolicy) {
  // Test SizeBasedColumnGroupPolicy
  int64_t max_avg_column_size = 1000;  // bytes
  int64_t max_columns_in_group = 2;

  ASSERT_AND_ASSIGN(auto policy, CreateSizeBasePolicy(max_avg_column_size, max_columns_in_group, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data (this should trigger sampling)
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify that policy created multiple groups based on size
  auto column_groups = *cgs;
  EXPECT_GE(column_groups.size(), 1);

  // Verify that no group exceeds max columns
  for (const auto& group : column_groups) {
    EXPECT_LE(group->columns.size(), static_cast<size_t>(max_columns_in_group));
  }
}

TEST_P(APIWriterReaderTest, TestWriteNotExistPath) {
  auto verify_writer = [&](std::string base_path, api::Properties& properties) {
    ASSERT_AND_ASSIGN(auto temp_fs, GetFileSystem(properties));
    ASSERT_STATUS_OK(DeleteTestDir(temp_fs, base_path));

    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
    auto writer = Writer::create(base_path, schema_, std::move(policy), properties);

    ASSERT_NE(writer, nullptr);
    ASSERT_OK(writer->write(test_batch_));
    auto cgs_result = writer->close();
    ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
    auto cgs = std::move(cgs_result).ValueOrDie();

    auto reader = Reader::create(cgs, schema_, nullptr, properties);
    ASSERT_NE(reader, nullptr);
    ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_OK(batch_reader->ReadNext(&batch));
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->num_rows(), 100);
    EXPECT_EQ(batch->num_columns(), 4);
    ASSERT_STATUS_OK(DeleteTestDir(temp_fs, base_path));
  };

  // test local with root_path
  api::Properties local_properties;
  api::SetValue(local_properties, PROPERTY_FS_STORAGE_TYPE, "local");
  api::SetValue(local_properties, PROPERTY_FS_ROOT_PATH, "/tmp/");
  verify_writer("not-exist-path1", local_properties);
  verify_writer("not-exist-path2", local_properties);
  verify_writer("not-exist-path3", local_properties);
  verify_writer("not-exist-path4/path1/path2/path3/path4", local_properties);
  verify_writer("not-exist-path5///", local_properties);  // will do lexically_normal

  if (!IsCloudEnv()) {
    GTEST_SKIP() << "No cloud env, skipped.";
  }

  // use the properties_ which is the cloud env
  verify_writer("not-exist-path1", properties_);
  verify_writer("not-exist-path2/path1/path2/path3/path4", properties_);
  verify_writer("not-exist-path5///", properties_);  // will do lexically_normal
}

TEST_P(APIWriterReaderTest, WriteWithTransactionAppendFiles) {
  int loop_times = 5;
  auto write_file = [&]() -> arrow::Result<std::shared_ptr<ColumnGroups>> {
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));
    auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
    // Write test data
    ARROW_RETURN_NOT_OK(writer->write(test_batch_));

    // Close and get cgs
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
    return cgs;
  };

  auto append_files = [&](std::shared_ptr<ColumnGroups>& cgs) -> arrow::Result<bool> {
    ARROW_ASSIGN_OR_RAISE(auto fs, GetFileSystem(properties_));
    ARROW_ASSIGN_OR_RAISE(auto transaction, Transaction::Open(fs, base_path_));
    transaction->AppendFiles(*cgs);
    ARROW_ASSIGN_OR_RAISE(auto committed_version, transaction->Commit());
    return committed_version > 0;
  };

  for (int i = 0; i < loop_times; ++i) {
    ASSERT_AND_ASSIGN(auto manifest_ptr, write_file());
    ASSERT_AND_ASSIGN(auto commit_result, append_files(manifest_ptr));
    ASSERT_TRUE(commit_result);
  }

  // verify paths and data
  ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

  ASSERT_AND_ASSIGN(auto manifest, transaction->GetManifest());
  auto cgs = manifest->columnGroups();
  ASSERT_EQ(cgs.size(), 1);
  ASSERT_EQ(cgs[0]->files.size(), loop_times);

  // verify data
  auto cgs_ptr = std::make_shared<ColumnGroups>(cgs);
  auto reader = Reader::create(cgs_ptr, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
  ASSERT_NE(batch_reader, nullptr);

  ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
  ASSERT_STATUS_OK(batch_reader->Close());

  ASSERT_EQ(table->num_rows(), 100 * loop_times);
  ASSERT_AND_ASSIGN(auto combined_batch, table->CombineChunksToBatch());
  ASSERT_STATUS_OK(ValidateRowAlignment(combined_batch));
}

TEST_P(APIWriterReaderTest, RandomAccessReading) {
  // Write data first
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  ASSERT_OK(writer->write(test_batch_));
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test random access reading
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Test take with specific row indices
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99};
  ASSERT_AND_ASSIGN(auto table, reader->take(row_indices));
  ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test

  // Verify result structure
  ASSERT_NE(result_batch, nullptr);
  EXPECT_GT(result_batch->num_rows(), 0);
  EXPECT_EQ(result_batch->num_columns(), 4);

  // Verify data correctness by checking specific values
  // Note: The current take implementation may return full chunks rather than exact rows
  // So we verify that the expected data is present somewhere in the result

  // Get the ID column (first column) to verify values
  auto id_column = std::static_pointer_cast<arrow::Int64Array>(result_batch->column(0));
  auto name_column = std::static_pointer_cast<arrow::StringArray>(result_batch->column(1));
  auto value_column = std::static_pointer_cast<arrow::DoubleArray>(result_batch->column(2));
  auto vector_column = std::static_pointer_cast<arrow::ListArray>(result_batch->column(3));

  // Verify we have some data
  EXPECT_GT(result_batch->num_rows(), 0);

  // Check that the data follows our expected pattern (id = i, name = "name_i", value = i * 1.5)
  // Since take may return chunks rather than exact rows, we verify the pattern holds
  for (int64_t i = 0; i < result_batch->num_rows(); ++i) {
    if (!id_column->IsNull(i)) {
      int64_t id_value = id_column->Value(i);

      // Verify this is a valid ID from our original data (0-99)
      EXPECT_GE(id_value, 0);
      EXPECT_LE(id_value, 99);

      // Verify name matches pattern
      if (!name_column->IsNull(i)) {
        std::string name_value = name_column->GetString(i);
        std::string expected_name = "name_" + std::to_string(id_value);
        EXPECT_EQ(name_value, expected_name);
      }

      // Verify value matches pattern
      if (!value_column->IsNull(i)) {
        double value_val = value_column->Value(i);
        double expected_value = id_value * 1.5;
        EXPECT_DOUBLE_EQ(value_val, expected_value);
      }
    }
  }
}

TEST_P(APIWriterReaderTest, FormatIntegration) {
  // Test that FileFormat uses packed reader/writer correctly
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write multiple batches
  for (int i = 0; i < 5; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify all column groups are format
  auto column_groups = *cgs;
  for (const auto& cg : column_groups) {
    EXPECT_EQ(cg->format, format);
  }

  // Test reading with packed reader integration
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  int total_rows = 0;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    ASSERT_OK(batch_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    total_rows += batch->num_rows();
    EXPECT_EQ(batch->num_columns(), 4);
  }

  EXPECT_EQ(total_rows, 5 * 100);  // 5 batches × 100 rows each
}

TEST_P(APIWriterReaderTest, ColumnProjection) {
  // Test column projection with packed reader - simplified to avoid memory issues
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  ASSERT_OK(writer->write(test_batch_));
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test reader with projection
  {
    std::vector<std::vector<std::string>> valid_projections = {{"id"},
                                                               {"value"},
                                                               {"name"},
                                                               {"vector"},
                                                               {"id", "name"},
                                                               {"value", "vector"},
                                                               {"id", "value", "name", "vector"},
                                                               {"value", "id"},
                                                               {"name", "id"}};
    for (const auto& col_names : valid_projections) {
      std::shared_ptr<std::vector<std::string>> needed_columns = std::make_shared<std::vector<std::string>>(col_names);
      auto reader = Reader::create(cgs, schema_, needed_columns, properties_);

      // record batch reader
      {
        auto batch_reader_result = reader->get_record_batch_reader();
        ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
        auto batch_reader = std::move(batch_reader_result).ValueOrDie();
        std::shared_ptr<arrow::RecordBatch> batch;
        ASSERT_OK(batch_reader->ReadNext(&batch));
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->num_columns(), col_names.size());
        for (int i = 0; i < batch->num_columns(); ++i) {
          EXPECT_EQ(batch->schema()->field(i)->name(), col_names[i]);
        }
      }

      // chunk reader
      {
        auto chunk_reader_result = reader->get_chunk_reader(0);
        ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
        auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();
        ASSERT_NE(chunk_reader, nullptr);
        auto chunk_result = chunk_reader->get_chunk(0);
        ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().ToString();
        auto chunk = std::move(chunk_result).ValueOrDie();
        ASSERT_NE(chunk, nullptr);
        EXPECT_EQ(chunk->num_columns(), col_names.size());
        for (int i = 0; i < chunk->num_columns(); ++i) {
          EXPECT_EQ(chunk->schema()->field(i)->name(), col_names[i]);
        }
      }

      // take
      {
        std::vector<int64_t> row_indices = {10};
        ASSERT_AND_ASSIGN(auto table, reader->take(row_indices));
        ASSERT_AND_ASSIGN(auto batch, table->CombineChunksToBatch());  // for test

        ASSERT_EQ(batch->num_rows(), 1);
        for (int i = 0; i < batch->num_columns(); ++i) {
          EXPECT_EQ(batch->schema()->field(i)->name(), col_names[i]);
        }
      }
    }
  }

  // Test basic reading without column projection for now
  {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);

    // record batch reader
    {
      auto batch_reader_result = reader->get_record_batch_reader();
      ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
      auto batch_reader = std::move(batch_reader_result).ValueOrDie();

      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_OK(batch_reader->ReadNext(&batch));
      ASSERT_NE(batch, nullptr);

      // Verify basic functionality
      EXPECT_EQ(batch->num_columns(), 4);
      EXPECT_GT(batch->num_rows(), 0);

      // Verify that all columns are present
      EXPECT_EQ(batch->schema()->field(0)->name(), "id");
      EXPECT_EQ(batch->schema()->field(1)->name(), "name");
      EXPECT_EQ(batch->schema()->field(2)->name(), "value");
      EXPECT_EQ(batch->schema()->field(3)->name(), "vector");
    }

    // chunk reader
    {
      auto chunk_reader_result = reader->get_chunk_reader(0);
      ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
      auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();
      ASSERT_NE(chunk_reader, nullptr);
      auto chunk_result = chunk_reader->get_chunk(0);
      ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().ToString();
      auto chunk = std::move(chunk_result).ValueOrDie();
      ASSERT_NE(chunk, nullptr);

      // Verify basic functionality
      EXPECT_EQ(chunk->num_columns(), 4);
      EXPECT_GT(chunk->num_rows(), 0);
      // Verify that all columns are present
      EXPECT_EQ(chunk->schema()->field(0)->name(), "id");
      EXPECT_EQ(chunk->schema()->field(1)->name(), "name");
      EXPECT_EQ(chunk->schema()->field(2)->name(), "value");
      EXPECT_EQ(chunk->schema()->field(3)->name(), "vector");
    }

    // take
    {
      std::vector<int64_t> row_indices = {10};
      ASSERT_AND_ASSIGN(auto table, reader->take(row_indices));
      ASSERT_AND_ASSIGN(auto batch, table->CombineChunksToBatch());  // for test

      ASSERT_EQ(batch->num_rows(), 1);
      ASSERT_EQ(batch->num_columns(), 4);  // All columns

      EXPECT_EQ(batch->schema()->field(0)->name(), "id");
      EXPECT_EQ(batch->schema()->field(1)->name(), "name");
      EXPECT_EQ(batch->schema()->field(2)->name(), "value");
      EXPECT_EQ(batch->schema()->field(3)->name(), "vector");
    }
  }
}

TEST_P(APIWriterReaderTest, MultipleWritesWithFlush) {
  // Test multiple writes with explicit flush operations
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write and flush multiple times
  ASSERT_OK(writer->write(test_batch_));
  ASSERT_OK(writer->flush());

  ASSERT_OK(writer->write(test_batch_));
  ASSERT_OK(writer->flush());

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify data integrity after multiple flushes
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  int total_rows = 0;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    ASSERT_OK(batch_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    total_rows += batch->num_rows();
  }

  EXPECT_EQ(total_rows, 2 * 100);  // 2 batches × 100 rows each
}

TEST_P(APIWriterReaderTest, RowAlignmentMultiColumnGroups) {
  // Test row alignment across multiple column groups
  int batch_size = 1000;

  // Create multiple column groups to test row alignment
  std::string patterns = "id, name|value, vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify we have multiple column groups
  auto column_groups = *cgs;
  EXPECT_EQ(column_groups.size(), 3);

  // Test row alignment with get_record_batch_reader
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  std::shared_ptr<arrow::RecordBatch> batch;
  int total_rows = 0;

  while (true) {
    ASSERT_OK(batch_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    total_rows += batch->num_rows();

    // Verify row alignment - all columns should have same number of rows
    EXPECT_EQ(batch->num_columns(), 4);
    for (int i = 0; i < batch->num_columns(); ++i) {
      EXPECT_EQ(batch->column(i)->length(), batch->num_rows());
    }

    // Verify data consistency across columns for same rows
    ASSERT_STATUS_OK(ValidateRowAlignment(batch));
  }

  EXPECT_EQ(total_rows, batch_size);
}

TEST_P(APIWriterReaderTest, RowAlignmentWithTakeOperation) {
  int batch_size = 500;

  // Create multiple column groups
  std::string patterns = "id, name|value, vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test take operation with specific row indices
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99, 150, 250, 350, 450};

  ASSERT_AND_ASSIGN(auto table, reader->take(row_indices));
  ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test

  // Verify row alignment in result
  ASSERT_NE(result_batch, nullptr);
  EXPECT_GT(result_batch->num_rows(), 0);
  EXPECT_EQ(result_batch->num_columns(), 4);

  // Verify all columns have same number of rows
  for (int i = 0; i < result_batch->num_columns(); ++i) {
    EXPECT_EQ(result_batch->column(i)->length(), result_batch->num_rows());
  }

  // Verify data consistency across columns
  ASSERT_STATUS_OK(ValidateRowAlignment(result_batch));
}

TEST_P(APIWriterReaderTest, RowAlignmentWithChunkReader) {
  // Test row alignment using individual chunk readers
  int batch_size = 200;

  // Create multiple column groups
  std::string patterns = "id, name, value, vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test chunk readers from different column groups
  auto column_groups = *cgs;
  EXPECT_EQ(column_groups.size(), 4);

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);

  // Get chunk readers for each column group
  std::vector<std::shared_ptr<ChunkReader>> chunk_readers;
  for (int i = 0; i < column_groups.size(); ++i) {
    const auto& cg = column_groups[i];
    auto chunk_reader_result = reader->get_chunk_reader(i);
    ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
    chunk_readers.push_back(std::move(chunk_reader_result).ValueOrDie());
  }

  // Read chunk 0 from each column group and verify row alignment
  std::vector<std::shared_ptr<arrow::RecordBatch>> chunks;
  for (auto& chunk_reader : chunk_readers) {
    auto chunk_result = chunk_reader->get_chunk(0);
    ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().ToString();
    auto chunk = std::move(chunk_result).ValueOrDie();
    ASSERT_NE(chunk, nullptr);
    chunks.push_back(chunk);
  }

  // Verify all chunks have same number of rows (row alignment)
  int64_t expected_rows = chunks[0]->num_rows();
  for (size_t i = 1; i < chunks.size(); ++i) {
    EXPECT_EQ(chunks[i]->num_rows(), expected_rows)
        << "Row count mismatch between column groups " << i - 1 << " and " << i;
  }

  // Verify data consistency by combining chunks and checking alignment
  std::vector<std::shared_ptr<arrow::Array>> combined_arrays;
  std::vector<std::shared_ptr<arrow::Field>> combined_fields;

  for (const auto& chunk : chunks) {
    for (int i = 0; i < chunk->num_columns(); ++i) {
      combined_arrays.push_back(chunk->column(i));
      combined_fields.push_back(chunk->schema()->field(i));
    }
  }

  auto combined_schema = arrow::schema(combined_fields);
  auto combined_batch = arrow::RecordBatch::Make(combined_schema, expected_rows, combined_arrays);

  ASSERT_STATUS_OK(ValidateRowAlignment(combined_batch));
}

TEST_P(APIWriterReaderTest, RowAlignmentWithMultipleRowGroups) {
  // Test row alignment when data spans multiple row groups
  int batch_size = 10000;                // Large amount of data
  const char* small_buffer = "1048576";  // 1MB buffer, forcing multiple row groups

  // Create multiple column groups
  std::string patterns = "id|name, value|vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto properties = milvus_storage::api::Properties{};
  ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties));
  SetValue(properties, PROPERTY_WRITER_BUFFER_SIZE, small_buffer);
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Read and verify row alignment across multiple row groups
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  int total_rows = 0;
  int batch_count = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    ASSERT_OK(batch_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }

    batch_count++;
    total_rows += batch->num_rows();

    // Verify row alignment in each batch
    EXPECT_EQ(batch->num_columns(), 4);
    for (int i = 0; i < batch->num_columns(); ++i) {
      EXPECT_EQ(batch->column(i)->length(), batch->num_rows());
    }

    ASSERT_STATUS_OK(ValidateRowAlignment(batch));
  }

  EXPECT_EQ(total_rows, batch_size);
  EXPECT_GT(batch_count, 1);
}

TEST_P(APIWriterReaderTest, TakeMethodTest) {
  std::string patterns = "id, name, value|vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);

  auto do_take = [](const auto& reader, const auto& row_indices) -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(row_indices));
    ARROW_ASSIGN_OR_RAISE(auto batch, table->CombineChunksToBatch());  // for test
    return batch;
  };

  auto verify_take_result = [](const auto& test_batch, const auto& take_result_batch, const auto& row_indices) {
    ASSERT_EQ(take_result_batch->num_rows(), row_indices.size());
    ASSERT_EQ(take_result_batch->num_columns(), test_batch->num_columns());
    for (size_t i = 0; i < row_indices.size(); ++i) {
      auto id_array = std::static_pointer_cast<arrow::Int64Array>(take_result_batch->column(0));
      auto name_array = std::static_pointer_cast<arrow::StringArray>(take_result_batch->column(1));
      auto value_array = std::static_pointer_cast<arrow::DoubleArray>(take_result_batch->column(2));

      EXPECT_EQ(id_array->Value(i),
                std::static_pointer_cast<arrow::Int64Array>(test_batch->column(0))->Value(row_indices[i]));
      EXPECT_EQ(name_array->GetString(i),
                std::static_pointer_cast<arrow::StringArray>(test_batch->column(1))->GetString(row_indices[i]));
      EXPECT_EQ(value_array->Value(i),
                std::static_pointer_cast<arrow::DoubleArray>(test_batch->column(2))->Value(row_indices[i]));
    }
  };

  std::vector<int64_t> all_row_indices(test_batch_->num_rows());
  std::iota(all_row_indices.begin(), all_row_indices.end(), 0);
  std::vector<std::vector<int64_t>> test_row_indices = {// take single row
                                                        {42},
                                                        // take multiple rows
                                                        {10, 50, 90},
                                                        // take all rows with order and uniqued
                                                        all_row_indices};

  for (const auto& row_indices : test_row_indices) {
    ASSERT_AND_ASSIGN(auto batch, do_take(reader, row_indices));
    verify_take_result(test_batch_, batch, row_indices);
  }
}

TEST_P(APIWriterReaderTest, TakeWithMultiFiles) {
  std::string patterns = "id, name|value, vector";

  size_t written_rows = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_written;
  for (int i = 0; i < 10; ++i) {
    ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema_, written_rows, false, i * 50 /* num_of_rows */));
    batches_written.emplace_back(batch);

    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));
    auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
    ASSERT_STATUS_OK(writer->write(batch));
    ASSERT_AND_ASSIGN(auto cgs, writer->close());
    transaction->AppendFiles(*cgs);
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_GT(committed_version, 0);
    written_rows += batch->num_rows();
  }
  ASSERT_AND_ASSIGN(auto verify_batch, ConcatenateRecordBatches(batches_written));

  ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));
  ASSERT_AND_ASSIGN(auto manifest, transaction->GetManifest());
  auto cgs = manifest->columnGroups();

  auto do_take = [](const auto& reader, const auto& row_indices) -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(row_indices));
    ARROW_ASSIGN_OR_RAISE(auto batch, table->CombineChunksToBatch());  // for test
    return batch;
  };

  auto verify_take_result = [](const auto& test_batch, const auto& take_result_batch, const auto& row_indices) {
    ASSERT_EQ(take_result_batch->num_rows(), row_indices.size());
    ASSERT_EQ(take_result_batch->num_columns(), 2);  // only take the id and name
    for (size_t i = 0; i < row_indices.size(); ++i) {
      auto id_array = std::static_pointer_cast<arrow::Int64Array>(take_result_batch->column(0));
      auto name_array = std::static_pointer_cast<arrow::StringArray>(take_result_batch->column(1));

      EXPECT_EQ(id_array->Value(i),
                std::static_pointer_cast<arrow::Int64Array>(test_batch->column(0))->Value(row_indices[i]));
      EXPECT_EQ(name_array->GetString(i),
                std::static_pointer_cast<arrow::StringArray>(test_batch->column(1))->GetString(row_indices[i]));
    }
  };

  std::vector<std::string> projection = {"id", "name"};
  auto cgs_ptr = std::make_shared<ColumnGroups>(cgs);
  auto reader = Reader::create(cgs_ptr, schema_, std::make_shared<std::vector<std::string>>(projection), properties_);

  // all rows
  std::vector<int64_t> all_row_indices(written_rows);
  std::iota(all_row_indices.begin(), all_row_indices.end(), 0);

  // random rows
  ASSERT_AND_ASSIGN(auto random_indices, GenerateSortedUniqueArray(50, written_rows, true));

  ASSERT_EQ(written_rows, 2250);
  std::vector<std::vector<int64_t>> test_row_indices = {
      // take single row
      {(int64_t)written_rows - 1},
      {(int64_t)written_rows / 2},
      {0},
      // take multiple rows
      {10, 500, 900},
      // take all rows with order and uniqued
      all_row_indices,
      // take random rows
      random_indices,
  };

  for (const auto& row_indices : test_row_indices) {
    ASSERT_AND_ASSIGN(auto batch, do_take(reader, row_indices));
    verify_take_result(verify_batch, batch, row_indices);
  }
}

TEST_P(APIWriterReaderTest, GetChucksTest) {
  // Test SizeBasedColumnGroupPolicy
  int64_t max_avg_column_size = 1000;  // bytes
  int64_t max_columns_in_group = 2;

  ASSERT_AND_ASSIGN(auto policy, CreateSizeBasePolicy(max_avg_column_size, max_columns_in_group, format, schema_));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data (this should trigger sampling)
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Test chunk reader
  {  // chunk reader 0
    auto chunk_reader_result0 = reader->get_chunk_reader(0);
    ASSERT_TRUE(chunk_reader_result0.ok()) << chunk_reader_result0.status().ToString();
    auto chunk_reader0 = std::move(chunk_reader_result0).ValueOrDie();
    ASSERT_NE(chunk_reader0, nullptr);

    auto chunk_indices_result0 = chunk_reader0->get_chunk_indices({0, 1, 11, 99, 1000});
    ASSERT_FALSE(chunk_indices_result0.ok());
  }

  {  // chunk reader 1
    auto chunk_reader_result1 = reader->get_chunk_reader(1);
    ASSERT_TRUE(chunk_reader_result1.ok()) << chunk_reader_result1.status().ToString();
    auto chunk_reader1 = std::move(chunk_reader_result1).ValueOrDie();
    ASSERT_NE(chunk_reader1, nullptr);
  }

  {  // chunk reader N not exist
    auto chunk_reader_resultn = reader->get_chunk_reader(10);
    ASSERT_FALSE(chunk_reader_resultn.ok());
  }
}

TEST_P(APIWriterReaderTest, EnrypytionWriterReaderTest) {
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "CMEK test is only applicable for Parquet format currently.";
  }

  // Test CMEK integrationfile_reader.cc:304
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto properties = milvus_storage::api::Properties{};
  ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties));

  SetValue(properties, PROPERTY_WRITER_ENC_ENABLE, "true");
  SetValue(properties, PROPERTY_WRITER_ENC_KEY, "footer_key_16B__");  // must be 16/24/32 bytes
  SetValue(properties, PROPERTY_WRITER_ENC_META, "encryption_meta_data");
  SetValue(properties, PROPERTY_WRITER_ENC_ALGORITHM, ENCRYPTION_ALGORITHM_AES_GCM_V1);

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties);
  ASSERT_NE(writer, nullptr);

  // Write test data
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test reading with encryption
  auto reader = Reader::create(cgs, schema_, nullptr, properties);
  ASSERT_NE(reader, nullptr);
  int called_keyretriever = 0;
  std::string key_id_used;
  reader->set_keyretriever([&called_keyretriever, &key_id_used](const std::string& key_id) -> std::string {
    called_keyretriever++;
    key_id_used = key_id;
    return "footer_key_16B__";
  });

  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();
  ASSERT_NE(batch_reader, nullptr);
  ASSERT_EQ(called_keyretriever, 1);
  ASSERT_EQ(key_id_used, "encryption_meta_data");

  auto chunk_reader_result = reader->get_chunk_reader(0);
  ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
  ASSERT_EQ(called_keyretriever, 2);
  ASSERT_EQ(key_id_used, "encryption_meta_data");
}

// port by packed/tests
TEST_P(APIWriterReaderTest, TestNullableFields) {
  // Create schema with nullable fields
  auto nullable_schema = arrow::schema(
      {arrow::field("int32", arrow::int32(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),
       arrow::field("int64", arrow::int64(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"200"})),
       arrow::field("str", arrow::utf8(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"300"}))});

  // Create record batch with null values
  arrow::Int32Builder int_builder;
  arrow::Int64Builder int64_builder;
  arrow::StringBuilder str_builder;

  // Add some null values
  ASSERT_STATUS_OK(int_builder.Append(100));
  ASSERT_STATUS_OK(int_builder.AppendNull());
  ASSERT_STATUS_OK(int_builder.Append(300));
  ASSERT_STATUS_OK(int64_builder.Append(1000));
  ASSERT_STATUS_OK(int64_builder.Append(2000));
  ASSERT_STATUS_OK(int64_builder.AppendNull());
  ASSERT_STATUS_OK(str_builder.Append("hello"));
  ASSERT_STATUS_OK(str_builder.AppendNull());
  ASSERT_STATUS_OK(str_builder.Append("world"));

  std::shared_ptr<arrow::Array> int_array, int64_array, str_array;
  ASSERT_STATUS_OK(int_builder.Finish(&int_array));
  ASSERT_STATUS_OK(int64_builder.Finish(&int64_array));
  ASSERT_STATUS_OK(str_builder.Finish(&str_array));

  std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
  auto nullable_batch = arrow::RecordBatch::Make(nullable_schema, 3, arrays);

  int batch_size = 500;
  auto column_groups = std::vector<std::vector<int>>{{2}, {0, 1}};

  // writer
  std::string patterns = "int32|int64,str";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, nullable_schema));
  auto writer = Writer::create(base_path_, nullable_schema, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  for (int i = 0; i < batch_size; ++i) {
    ASSERT_TRUE(writer->write(nullable_batch).ok());
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // reader
  auto reader = Reader::create(cgs, nullable_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();
  // Read and validate null values
  ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
  ASSERT_STATUS_OK(batch_reader->Close());

  ASSERT_EQ(table->num_rows(), batch_size * 3);
  ASSERT_EQ(table->num_columns(), 3);

  // Validate null counts
  ASSERT_EQ(table->column(0)->null_count(), batch_size);  // 1 out of every 3 values is null
  ASSERT_EQ(table->column(1)->null_count(), batch_size);  // 1 out of every 3 values is null
  ASSERT_EQ(table->column(2)->null_count(), batch_size);  // 1 out of every 3 values is null
}

TEST_P(APIWriterReaderTest, TestMixedNullableAndNonNullable) {
  // Create schema with mixed nullable and non-nullable fields
  auto mixed_schema = arrow::schema({
      arrow::field("int32", arrow::int32(), false,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),  // non-nullable
      arrow::field("int64", arrow::int64(), true,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"200"})),  // nullable
      arrow::field("str", arrow::utf8(), false,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"300"}))  // non-nullable
  });

  // Create record batch
  arrow::Int32Builder int_builder;
  arrow::Int64Builder int64_builder;
  arrow::StringBuilder str_builder;

  ASSERT_STATUS_OK(int_builder.AppendValues({100, 200, 300}));
  ASSERT_STATUS_OK(int64_builder.Append(1000));
  ASSERT_STATUS_OK(int64_builder.AppendNull());
  ASSERT_STATUS_OK(int64_builder.Append(3000));
  ASSERT_STATUS_OK(str_builder.AppendValues({"hello", "world", "test"}));

  std::shared_ptr<arrow::Array> int_array, int64_array, str_array;
  ASSERT_STATUS_OK(int_builder.Finish(&int_array));
  ASSERT_STATUS_OK(int64_builder.Finish(&int64_array));
  ASSERT_STATUS_OK(str_builder.Finish(&str_array));

  std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
  auto mixed_batch = arrow::RecordBatch::Make(mixed_schema, 3, arrays);

  int batch_size = 300;

  // writer
  std::string patterns = "str,int32|int64";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, mixed_schema));
  auto writer = Writer::create(base_path_, mixed_schema, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer->write(mixed_batch).ok());
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test reading full schema
  auto reader1 = Reader::create(cgs, mixed_schema, nullptr, properties_);
  ASSERT_NE(reader1, nullptr);

  auto batch_reader_result1 = reader1->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result1.ok()) << batch_reader_result1.status().ToString();
  auto batch_reader1 = std::move(batch_reader_result1).ValueOrDie();

  ASSERT_AND_ASSIGN(auto table1, batch_reader1->ToTable());
  ASSERT_STATUS_OK(batch_reader1->Close());

  ASSERT_EQ(table1->num_rows(), batch_size * 3);
  ASSERT_EQ(table1->column(0)->null_count(), 0);           // non-nullable
  ASSERT_EQ(table1->column(1)->null_count(), batch_size);  // nullable, 1 out of every 3 values is null
  ASSERT_EQ(table1->column(2)->null_count(), 0);           // non-nullable

  // Test schema evolution - change non-nullable field to nullable for reading
  auto evolved_schema = arrow::schema({
      arrow::field("int32", arrow::int32(), true,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),  // changed to nullable
      arrow::field("int64", arrow::int64(), true,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"200"})),  // kept nullable
      arrow::field("str", arrow::utf8(), true,
                   arrow::key_value_metadata({"PARQUET:field_id"}, {"300"}))  // changed to nullable
  });

  auto reader2 = Reader::create(cgs, evolved_schema, nullptr, properties_);
  ASSERT_NE(reader2, nullptr);

  auto batch_reader_result2 = reader2->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result2.ok()) << batch_reader_result2.status().ToString();
  auto batch_reader2 = std::move(batch_reader_result2).ValueOrDie();

  ASSERT_AND_ASSIGN(auto table2, batch_reader2->ToTable());
  ASSERT_STATUS_OK(batch_reader2->Close());

  ASSERT_EQ(table2->num_rows(), batch_size * 3);
  ASSERT_EQ(table2->column(0)->null_count(),
            0);  // Although schema changed to nullable, data itself does not have nulls
  ASSERT_EQ(table2->column(1)->null_count(), batch_size);
  ASSERT_EQ(table2->column(2)->null_count(), 0);
}

TEST_P(APIWriterReaderTest, TestAllNullFields) {
  int batch_size = 400;
  int number_of_rows = 20;

  // Create schema with all nullable fields
  auto all_nullable_schema = arrow::schema(
      {arrow::field("int32", arrow::int32(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),
       arrow::field("int64", arrow::int64(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"200"})),
       arrow::field("str", arrow::utf8(), true, arrow::key_value_metadata({"PARQUET:field_id"}, {"300"}))});

  // Create record batch with all null values
  arrow::Int32Builder int_builder;
  arrow::Int64Builder int64_builder;
  arrow::StringBuilder str_builder;

  // Append nulls
  for (int i = 0; i < number_of_rows; ++i) {
    ASSERT_STATUS_OK(int_builder.AppendNull());
    ASSERT_STATUS_OK(int64_builder.AppendNull());
    ASSERT_STATUS_OK(str_builder.AppendNull());
  }

  std::shared_ptr<arrow::Array> int_array, int64_array, str_array;
  ASSERT_STATUS_OK(int_builder.Finish(&int_array));
  ASSERT_STATUS_OK(int64_builder.Finish(&int64_array));
  ASSERT_STATUS_OK(str_builder.Finish(&str_array));

  std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
  auto all_null_batch = arrow::RecordBatch::Make(all_nullable_schema, number_of_rows, arrays);

  // writer
  std::string patterns = "int32|int64,str";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, all_nullable_schema));
  auto writer = Writer::create(base_path_, all_nullable_schema, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(writer->write(all_null_batch).ok());
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // reader
  auto reader = Reader::create(cgs, all_nullable_schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  // Read and validate all null values
  ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
  ASSERT_STATUS_OK(batch_reader->Close());
  ASSERT_EQ(table->num_rows(), batch_size * number_of_rows);
  ASSERT_EQ(table->num_columns(), 3);
  // Validate all columns have all nulls
  ASSERT_EQ(table->column(0)->null_count(), batch_size * number_of_rows);
  ASSERT_EQ(table->column(1)->null_count(), batch_size * number_of_rows);
  ASSERT_EQ(table->column(2)->null_count(), batch_size * number_of_rows);
}

TEST_P(APIWriterReaderTest, TestLargeBatch) {
  // Test writing and reading a large record batch
  int large_batch_size = 100000;  // 100k rows

  // Create large record batch by repeating test_batch_
  std::vector<std::shared_ptr<arrow::Array>> large_arrays;
  for (int i = 0; i < test_batch_->num_columns(); ++i) {
    arrow::ArrayVector arrays_to_concat;
    for (int j = 0; j < large_batch_size / test_batch_->num_rows(); ++j) {
      arrays_to_concat.push_back(test_batch_->column(i));
    }
    std::shared_ptr<arrow::Array> large_array;
    auto concat_result = arrow::Concatenate(arrays_to_concat);
    ASSERT_TRUE(concat_result.ok()) << concat_result.status().ToString();
    large_arrays.push_back(concat_result.ValueOrDie());
  }
  auto large_batch = arrow::RecordBatch::Make(schema_, large_batch_size, large_arrays);

  // writer
  std::string patterns = "id|value, name, vector";
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  EXPECT_TRUE(writer->write(large_batch).ok());

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // reader
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  auto batch_reader_result = reader->get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();
  // Read and validate large batch
  ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
  ASSERT_STATUS_OK(batch_reader->Close());
  ASSERT_EQ(table->num_rows(), large_batch_size);
  ASSERT_EQ(table->num_columns(), 4);
  // Validate data in each column
  for (int i = 0; i < table->num_columns(); ++i) {
    ASSERT_EQ(table->column(i)->null_count(), 0);
  }

  ASSERT_TRUE(large_batch->Equals(*table->CombineChunksToBatch().ValueOrDie()));
}

INSTANTIATE_TEST_SUITE_P(APIWriterReaderTestP,
                         APIWriterReaderTest,
#ifdef BUILD_VORTEX_BRIDGE
                         ::testing::Combine(::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX),
                                            ::testing::Values(1, 4))
#else
                         ::testing::Combine(::testing::Values(LOON_FORMAT_PARQUET), ::testing::Values(1, 4))
#endif
);

}  // namespace milvus_storage::test