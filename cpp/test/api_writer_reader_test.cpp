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
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/api.h>
#include <arrow/testing/gtest_util.h>
#include <unistd.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/column_groups.h"
#include "test_util.h"

using namespace milvus_storage::api;

// Helper function to get temporary directory
std::string GetTempDir() { return "/tmp/milvus_storage_test_" + std::to_string(getpid()); }

class APIWriterReaderTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

    // Create a simple test schema with field IDs required by packed writer
    schema_ = arrow::schema(
        {arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),
         arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"101"})),
         arrow::field("value", arrow::float64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"102"})),
         arrow::field("vector", arrow::list(arrow::float32()), false,
                      arrow::key_value_metadata({"PARQUET:field_id"}, {"103"}))});

    base_path_ = GetTempDir() + "/api_test";
    ASSERT_OK(fs_->CreateDir(base_path_));

    // Create test data
    CreateTestData();

    milvus_storage::InitTestProperties(properties_, "/", base_path_);
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_OK(fs_->DeleteDirContents(GetTempDir()));
  }

  void CreateTestData() {
    arrow::Int64Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::DoubleBuilder value_builder;
    arrow::ListBuilder vector_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());

    for (int64_t i = 0; i < 100; ++i) {
      ASSERT_OK(id_builder.Append(i));
      ASSERT_OK(name_builder.Append("name_" + std::to_string(i)));
      ASSERT_OK(value_builder.Append(i * 1.5));

      // Create vector data
      auto vector_element_builder = static_cast<arrow::FloatBuilder*>(vector_builder.value_builder());
      ASSERT_OK(vector_builder.Append());
      for (int j = 0; j < 4; ++j) {
        ASSERT_OK(vector_element_builder->Append(i * 0.1f + j));
      }
    }

    std::shared_ptr<arrow::Array> id_array, name_array, value_array, vector_array;
    ASSERT_OK(id_builder.Finish(&id_array));
    ASSERT_OK(name_builder.Finish(&name_array));
    ASSERT_OK(value_builder.Finish(&value_array));
    ASSERT_OK(vector_builder.Finish(&vector_array));

    test_batch_ = arrow::RecordBatch::Make(schema_, 100, {id_array, name_array, value_array, vector_array});
  }

  std::shared_ptr<arrow::fs::LocalFileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;

  void ValidateRowAlignment(const std::shared_ptr<arrow::RecordBatch>& batch) {
    // Validate that data is properly aligned across columns
    // This checks that for each row, the data follows the expected pattern
    std::shared_ptr<arrow::StringArray> name_str_column = nullptr;
    std::shared_ptr<arrow::StringViewArray> name_strview_column = nullptr;

    // Get columns
    auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    if (batch->column(1)->type()->id() == arrow::Type::STRING) {
      name_str_column = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    } else if (batch->column(1)->type()->id() == arrow::Type::STRING_VIEW) {
      name_strview_column = std::static_pointer_cast<arrow::StringViewArray>(batch->column(1));
    } else {
      ASSERT_TRUE(false) << "Column 1 is not of type STRING";
    }

    auto value_column = std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));
    auto vector_column = std::static_pointer_cast<arrow::ListArray>(batch->column(3));

    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      if (!id_column->IsNull(row)) {
        int64_t id_value = id_column->Value(row);
        int64_t original_id = id_value % 100;  // Original ID in test data (0-99)

        // Verify name matches expected pattern
        if (name_str_column && !name_str_column->IsNull(row)) {
          std::string name_value = name_str_column->GetString(row);
          std::string expected_name = "name_" + std::to_string(original_id);
          EXPECT_EQ(name_value, expected_name) << "Row " << row << ": name mismatch for id " << id_value;
        }

        if (name_strview_column && !name_strview_column->IsNull(row)) {
          std::string name_value = name_strview_column->GetString(row);
          std::string expected_name = "name_" + std::to_string(original_id);
          EXPECT_EQ(name_value, expected_name) << "Row " << row << ": name mismatch for id " << id_value;
        }

        // Verify value matches expected pattern
        if (!value_column->IsNull(row)) {
          double value_val = value_column->Value(row);
          double expected_value = original_id * 1.5;
          EXPECT_DOUBLE_EQ(value_val, expected_value) << "Row " << row << ": value mismatch for id " << id_value;
        }

        // Verify vector has expected structure (4 elements)
        if (!vector_column->IsNull(row)) {
          auto vector_slice = vector_column->value_slice(row);
          auto float_array = std::static_pointer_cast<arrow::FloatArray>(vector_slice);
          EXPECT_EQ(float_array->length(), 4) << "Row " << row << ": vector length mismatch for id " << id_value;

          // Check vector values
          for (int j = 0; j < 4; ++j) {
            if (!float_array->IsNull(j)) {
              float expected_vector_value = original_id * 0.1f + j;
              EXPECT_FLOAT_EQ(float_array->Value(j), expected_vector_value)
                  << "Row " << row << ", vector[" << j << "]: value mismatch for id " << id_value;
            }
          }
        }
      }
    }
  }
};

TEST_P(APIWriterReaderTest, SingleColumnGroupWriteRead) {
  std::string format = GetParam();
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_, format);
  auto writer = Writer::create(base_path_ + "/" + format, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  EXPECT_EQ(cgs->get_all().size(), 1);
  auto column_groups = cgs->get_all();
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
  std::string format = GetParam();
  // Test writing with SchemaBasedColumnGroupPolicy
  std::vector<std::string> patterns = {"id|value", "name", "vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns, format);

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify cgs has multiple column groups
  auto column_groups = cgs->get_all();
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
  std::string format = GetParam();

  // Test SizeBasedColumnGroupPolicy
  int64_t max_avg_column_size = 1000;  // bytes
  int64_t max_columns_in_group = 2;

  auto policy =
      std::make_unique<SizeBasedColumnGroupPolicy>(schema_, max_avg_column_size, max_columns_in_group, format);

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);

  // Write test data (this should trigger sampling)
  ASSERT_OK(writer->write(test_batch_));

  // Close and get cgs
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify that policy created multiple groups based on size
  auto column_groups = cgs->get_all();
  EXPECT_GE(column_groups.size(), 1);

  // Verify that no group exceeds max columns
  for (const auto& group : column_groups) {
    EXPECT_LE(group->columns.size(), static_cast<size_t>(max_columns_in_group));
  }
}

TEST_P(APIWriterReaderTest, RandomAccessReading) {
  // Ignore this test for now, it is not implemented yet
  return;

  // Write data first
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  auto writer = Writer::create(base_path_, schema_, std::move(policy));
  ASSERT_NE(writer, nullptr);

  ASSERT_OK(writer->write(test_batch_));
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test random access reading
  auto reader = Reader::create(cgs, schema_);
  ASSERT_NE(reader, nullptr);

  // Test take with specific row indices
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99};
  auto take_result = reader->take(row_indices);
  ASSERT_TRUE(take_result.ok()) << take_result.status().ToString();
  auto result_batch = std::move(take_result).ValueOrDie();

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

TEST_P(APIWriterReaderTest, ErrorHandling) {
  // Ignore this test for now, it is not implemented yet
  return;
  // Test error handling
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  auto writer = Writer::create(base_path_, schema_, std::move(policy));

  // Test writing after close
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs1 = std::move(cgs_result).ValueOrDie();
  auto status = writer->write(test_batch_);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.message().find("closed") != std::string::npos);

  // Test invalid row indices in Reader
  auto reader = Reader::create(cgs1, schema_);
  std::vector<int64_t> invalid_indices = {-1, 200};
  auto result = reader->take(invalid_indices);
  EXPECT_FALSE(result.ok());
}

TEST_P(APIWriterReaderTest, FormatIntegration) {
  std::string format = GetParam();
  // Test that FileFormat uses packed reader/writer correctly
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_, format);
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write multiple batches
  for (int i = 0; i < 5; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify all column groups are format
  auto column_groups = cgs->get_all();
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
  std::string format = GetParam();
  // Test column projection with packed reader - simplified to avoid memory issues
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_, format);
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
  }

  // Test invalid projection
  // should we throw exception or just ignore the invalid column?
  {
    std::vector<std::vector<std::string>> invalid_projections = {
        {"non_existent_column1"}, {"id", "non_existent_column"}, {"name", "value", "invalid_col"}};
    for (const auto& col_names : invalid_projections) {
      std::shared_ptr<std::vector<std::string>> needed_columns = std::make_shared<std::vector<std::string>>(col_names);
      bool throw_caught = false;
      try {
        auto reader = Reader::create(cgs, schema_, needed_columns, properties_);
      } catch (const std::exception& e) {
        throw_caught = true;
      }
      EXPECT_TRUE(throw_caught);
    }
  }
}

TEST_P(APIWriterReaderTest, MultipleWritesWithFlush) {
  std::string format = GetParam();
  // Test multiple writes with explicit flush operations
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_, format);
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
  std::string format = GetParam();
  // Test row alignment across multiple column groups
  int batch_size = 1000;

  // Create multiple column groups to test row alignment
  std::vector<std::string> patterns = {"id", "name|value", "vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns, format);

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Verify we have multiple column groups
  auto column_groups = cgs->get_all();
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
    ValidateRowAlignment(batch);
  }

  EXPECT_EQ(total_rows, batch_size);
}

TEST_P(APIWriterReaderTest, RowAlignmentWithTakeOperation) {
  // Ignore this test for now, it is not implemented yet
  return;
  // Test row alignment with random access (take operation)
  int batch_size = 500;

  // Create multiple column groups
  std::vector<std::string> patterns = {"id|name", "value|vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns);

  auto writer = Writer::create(base_path_, schema_, std::move(policy));

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test take operation with specific row indices
  auto reader = Reader::create(cgs, schema_);
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99, 150, 250, 350, 450};

  auto take_result = reader->take(row_indices);
  ASSERT_TRUE(take_result.ok()) << take_result.status().ToString();
  auto result_batch = std::move(take_result).ValueOrDie();

  // Verify row alignment in result
  ASSERT_NE(result_batch, nullptr);
  EXPECT_GT(result_batch->num_rows(), 0);
  EXPECT_EQ(result_batch->num_columns(), 4);

  // Verify all columns have same number of rows
  for (int i = 0; i < result_batch->num_columns(); ++i) {
    EXPECT_EQ(result_batch->column(i)->length(), result_batch->num_rows());
  }

  // Verify data consistency across columns
  ValidateRowAlignment(result_batch);
}

TEST_P(APIWriterReaderTest, RowAlignmentWithChunkReader) {
  std::string format = GetParam();
  // Test row alignment using individual chunk readers
  int batch_size = 200;

  // Create multiple column groups
  std::vector<std::string> patterns = {"id", "name", "value", "vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns, format);

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer->write(test_batch_));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Test chunk readers from different column groups
  auto column_groups = cgs->get_all();
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

  ValidateRowAlignment(combined_batch);
}

TEST_P(APIWriterReaderTest, RowAlignmentWithMultipleRowGroups) {
  std::string format = GetParam();
  // Test row alignment when data spans multiple row groups
  int batch_size = 5000;                 // Large amount of data
  const char* small_buffer = "1048576";  // 1MB buffer, forcing multiple row groups

  // Create multiple column groups
  std::vector<std::string> patterns = {"id|name", "value|vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns, format);

  auto properties = milvus_storage::api::Properties{};
  milvus_storage::InitTestProperties(properties, "/", base_path_);
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

    ValidateRowAlignment(batch);
  }

  EXPECT_EQ(total_rows, batch_size);
  EXPECT_GT(batch_count, 1);  // Should have multiple batches due to small buffer
}

TEST_P(APIWriterReaderTest, TakeMethodTest) {
  // Ignore this test for now, it is not implemented yet
  return;
  // Create multi-column group data for take testing
  std::vector<std::string> patterns = {"id|value", "name", "vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns);

  auto writer = Writer::create(base_path_ + "_take", schema_, std::move(policy));
  ASSERT_OK(writer->write(test_batch_));

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << "Writer close failed: " << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  auto reader = Reader::create(cgs, schema_);

  // Test single row take
  std::vector<int64_t> row_indices = {42};
  auto result = reader->take(row_indices);
  ASSERT_TRUE(result.ok()) << "Take failed: " << result.status().ToString();

  auto batch = result.ValueOrDie();
  ASSERT_EQ(batch->num_rows(), 1);
  ASSERT_EQ(batch->num_columns(), 4);  // All columns

  // Verify data correctness
  auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
  auto name_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
  auto value_array = std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));

  EXPECT_EQ(id_array->Value(0), 42);
  EXPECT_EQ(name_array->GetString(0), "name_42");
  EXPECT_EQ(value_array->Value(0), 42 * 1.5);

  // Test multiple rows
  std::vector<int64_t> multi_indices = {10, 50, 90};
  result = reader->take(multi_indices);
  ASSERT_TRUE(result.ok()) << "Multi-row take failed: " << result.status().ToString();

  batch = result.ValueOrDie();
  ASSERT_EQ(batch->num_rows(), 3);

  auto id_array2 = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
  auto name_array2 = std::static_pointer_cast<arrow::StringArray>(batch->column(1));

  EXPECT_EQ(id_array2->Value(0), 10);
  EXPECT_EQ(id_array2->Value(1), 50);
  EXPECT_EQ(id_array2->Value(2), 90);

  EXPECT_EQ(name_array2->GetString(0), "name_10");
  EXPECT_EQ(name_array2->GetString(1), "name_50");
  EXPECT_EQ(name_array2->GetString(2), "name_90");
}

TEST_P(APIWriterReaderTest, GetChucksTest) {
  std::string format = GetParam();
  // Test SizeBasedColumnGroupPolicy
  int64_t max_avg_column_size = 1000;  // bytes
  int64_t max_columns_in_group = 2;

  auto policy =
      std::make_unique<SizeBasedColumnGroupPolicy>(schema_, max_avg_column_size, max_columns_in_group, format);

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
  std::string format = GetParam();

  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "CMEK test is only applicable for Parquet format currently.";
  }

  // Test CMEK integrationfile_reader.cc:304
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_, format);
  auto properties = milvus_storage::api::Properties{};
  milvus_storage::InitTestProperties(properties, "/", base_path_);

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
  std::string format = GetParam();

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
  std::vector<std::string> patterns = {"int32|int64", "str"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(nullable_schema, patterns, format);
  auto writer = Writer::create(base_path_ + "/" + format, nullable_schema, std::move(policy), properties_);
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
  std::string format = GetParam();
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
  std::vector<std::string> patterns = {"str", "int32|int64"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(mixed_schema, patterns, format);
  auto writer = Writer::create(base_path_ + "/" + format, mixed_schema, std::move(policy), properties_);
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
  std::string format = GetParam();
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
  std::vector<std::string> patterns = {"int32|int64", "str"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(all_nullable_schema, patterns, format);
  auto writer = Writer::create(base_path_ + "/" + format, all_nullable_schema, std::move(policy), properties_);
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
  std::string format = GetParam();
  if (format == LOON_FORMAT_VORTEX) {
    GTEST_SKIP();
  }

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
  std::vector<std::string> patterns = {"id|value", "name", "vector"};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, patterns, format);
  auto writer = Writer::create(base_path_ + "/" + format, schema_, std::move(policy), properties_);
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
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX)
#else
                         ::testing::Values(LOON_FORMAT_PARQUET)
#endif
);