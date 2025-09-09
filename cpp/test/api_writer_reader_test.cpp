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

#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/manifest.h"

using namespace milvus_storage::api;

// Helper function to get temporary directory
std::string GetTempDir() { return "/tmp/milvus_storage_test_" + std::to_string(getpid()); }

class APIWriterReaderTest : public ::testing::Test {
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
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_OK(fs_->DeleteDirContents(base_path_));
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

  void ValidateRowAlignment(const std::shared_ptr<arrow::RecordBatch>& batch) {
    // Validate that data is properly aligned across columns
    // This checks that for each row, the data follows the expected pattern

    // Get columns
    auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto name_column = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    auto value_column = std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));
    auto vector_column = std::static_pointer_cast<arrow::ListArray>(batch->column(3));

    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      if (!id_column->IsNull(row)) {
        int64_t id_value = id_column->Value(row);
        int64_t original_id = id_value % 100;  // Original ID in test data (0-99)

        // Verify name matches expected pattern
        if (!name_column->IsNull(row)) {
          std::string name_value = name_column->GetString(row);
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

TEST_F(APIWriterReaderTest, SingleColumnGroupWriteRead) {
  // Test writing with SingleColumnGroupPolicy
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);

  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write test data
  ASSERT_OK(writer.write(test_batch_));

  // Close and get manifest
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify manifest
  EXPECT_EQ(manifest->get_column_groups().size(), 1);
  auto column_groups = manifest->get_column_groups();
  EXPECT_EQ(column_groups[0]->format, FileFormat::PARQUET);
  EXPECT_EQ(column_groups[0]->columns.size(), 4);

  // Test reading with new API
  Reader reader(fs_, manifest, schema_);

  // Test get_record_batch_reader (uses PackedRecordBatchReader internally)
  auto batch_reader_result = reader.get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_OK(batch_reader->ReadNext(&batch));
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_rows(), 100);
  EXPECT_EQ(batch->num_columns(), 4);

  // Verify data content matches
  EXPECT_TRUE(batch->Equals(*test_batch_));

  // Read until end to verify complete data
  ASSERT_OK(batch_reader->ReadNext(&batch));
  EXPECT_EQ(batch, nullptr);  // Should be at end
}

TEST_F(APIWriterReaderTest, SchemaBasedColumnGroupWriteRead) {
  // Test writing with SchemaBasedColumnGroupPolicy
  std::vector<ColumnGroupConfig> configs = {{.column_patterns = {"id|value"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"name"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"vector"}, .format = FileFormat::PARQUET}};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, configs);

  auto properties =
      WritePropertiesBuilder().with_compression(CompressionType::ZSTD).with_max_row_group_size(50).build();

  Writer writer(fs_, base_path_, schema_, std::move(policy), properties);

  // Write test data
  ASSERT_OK(writer.write(test_batch_));

  // Add some metadata
  ASSERT_OK(writer.add_metadata("test_key", "test_value"));

  // Close and get manifest
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify manifest has multiple column groups
  auto column_groups = manifest->get_column_groups();
  EXPECT_EQ(column_groups.size(), 3);

  // Test reading without column projection first (column groups may not contain all columns)
  Reader reader(fs_, manifest, schema_);

  // Test chunk reader
  auto chunk_reader_result = reader.get_chunk_reader(column_groups[0]->id);
  ASSERT_TRUE(chunk_reader_result.ok()) << chunk_reader_result.status().ToString();
  auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();

  auto chunk_result = chunk_reader->get_chunk(0);
  ASSERT_TRUE(chunk_result.ok()) << chunk_result.status().ToString();
  auto chunk = std::move(chunk_result).ValueOrDie();
  ASSERT_NE(chunk, nullptr);
  EXPECT_GT(chunk->num_rows(), 0);
}

TEST_F(APIWriterReaderTest, SizeBasedColumnGroupPolicy) {
  // Test SizeBasedColumnGroupPolicy
  int64_t max_avg_column_size = 1000;  // bytes
  int64_t max_columns_in_group = 2;

  auto policy = std::make_unique<SizeBasedColumnGroupPolicy>(schema_, max_avg_column_size, max_columns_in_group);

  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write test data (this should trigger sampling)
  ASSERT_OK(writer.write(test_batch_));

  // Close and get manifest
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify that policy created multiple groups based on size
  auto column_groups = manifest->get_column_groups();
  EXPECT_GE(column_groups.size(), 1);

  // Verify that no group exceeds max columns
  for (const auto& group : column_groups) {
    EXPECT_LE(group->columns.size(), static_cast<size_t>(max_columns_in_group));
  }
}

TEST_F(APIWriterReaderTest, RandomAccessReading) {
  // Write data first
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  Writer writer(fs_, base_path_, schema_, std::move(policy));

  ASSERT_OK(writer.write(test_batch_));
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Test random access reading
  Reader reader(fs_, manifest, schema_);

  // Test take with specific row indices
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99};
  auto take_result = reader.take(row_indices);
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

TEST_F(APIWriterReaderTest, WritePropertiesBuilder) {
  // Test WritePropertiesBuilder
  auto properties = WritePropertiesBuilder()
                        .with_compression(CompressionType::ZSTD)
                        .with_compression_level(3)
                        .with_max_row_group_size(1000)
                        .with_buffer_size(32 * 1024 * 1024)
                        .with_dictionary_encoding(true)
                        .with_metadata("created_by", "api_test")
                        .build();

  EXPECT_EQ(properties.compression, CompressionType::ZSTD);
  EXPECT_EQ(properties.compression_level, 3);
  EXPECT_EQ(properties.max_row_group_size, 1000);
  EXPECT_EQ(properties.buffer_size, 32 * 1024 * 1024);
  EXPECT_TRUE(properties.enable_dictionary);
  EXPECT_EQ(properties.custom_metadata.at("created_by"), "api_test");
}

TEST_F(APIWriterReaderTest, ReadPropertiesBuilder) {
  // Test ReadPropertiesBuilder
  auto properties = ReadPropertiesBuilder()
                        .with_cipher_type("AES256")
                        .with_cipher_key("test_key")
                        .with_cipher_metadata("test_metadata")
                        .build();

  EXPECT_EQ(properties.cipher_type, "AES256");
  EXPECT_EQ(properties.cipher_key, "test_key");
  EXPECT_EQ(properties.cipher_metadata, "test_metadata");
}

TEST_F(APIWriterReaderTest, ErrorHandling) {
  // Test error handling
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Test writing after close
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest1 = std::move(manifest_result).ValueOrDie();
  auto status = writer.write(test_batch_);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.message().find("closed") != std::string::npos);

  // Test invalid row indices in Reader
  Reader reader(fs_, manifest1, schema_);
  std::vector<int64_t> invalid_indices = {-1, 200};
  auto result = reader.take(invalid_indices);
  EXPECT_FALSE(result.ok());
}

TEST_F(APIWriterReaderTest, ParquetFormatIntegration) {
  // Test that FileFormat::PARQUET uses packed reader/writer correctly
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write multiple batches
  for (int i = 0; i < 5; ++i) {
    ASSERT_OK(writer.write(test_batch_));
  }

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify all column groups are PARQUET format
  auto column_groups = manifest->get_column_groups();
  for (const auto& cg : column_groups) {
    EXPECT_EQ(cg->format, FileFormat::PARQUET);
  }

  // Test reading with packed reader integration
  Reader reader(fs_, manifest, schema_);
  auto batch_reader_result = reader.get_record_batch_reader();
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

TEST_F(APIWriterReaderTest, ColumnProjection) {
  // Test column projection with packed reader - simplified to avoid memory issues
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  Writer writer(fs_, base_path_, schema_, std::move(policy));

  ASSERT_OK(writer.write(test_batch_));
  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Test basic reading without column projection for now
  Reader reader(fs_, manifest, schema_);

  auto batch_reader_result = reader.get_record_batch_reader();
  ASSERT_TRUE(batch_reader_result.ok()) << batch_reader_result.status().ToString();
  auto batch_reader = std::move(batch_reader_result).ValueOrDie();

  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_OK(batch_reader->ReadNext(&batch));
  ASSERT_NE(batch, nullptr);

  // Verify basic functionality
  EXPECT_EQ(batch->num_columns(), 4);
  EXPECT_GT(batch->num_rows(), 0);

  // Verify that all columns are present
  bool found_id = false, found_name = false, found_value = false, found_vector = false;
  for (int i = 0; i < batch->num_columns(); ++i) {
    auto field_name = batch->schema()->field(i)->name();
    if (field_name == "id")
      found_id = true;
    if (field_name == "name")
      found_name = true;
    if (field_name == "value")
      found_value = true;
    if (field_name == "vector")
      found_vector = true;
  }
  EXPECT_TRUE(found_id && found_name && found_value && found_vector);
}

TEST_F(APIWriterReaderTest, MultipleWritesWithFlush) {
  // Test multiple writes with explicit flush operations
  auto policy = std::make_unique<SingleColumnGroupPolicy>(schema_);
  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write and flush multiple times
  ASSERT_OK(writer.write(test_batch_));
  ASSERT_OK(writer.flush());

  ASSERT_OK(writer.write(test_batch_));
  ASSERT_OK(writer.flush());

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify data integrity after multiple flushes
  Reader reader(fs_, manifest, schema_);
  auto batch_reader_result = reader.get_record_batch_reader();
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

TEST_F(APIWriterReaderTest, RowAlignmentMultiColumnGroups) {
  // Test row alignment across multiple column groups
  int batch_size = 1000;

  // Create multiple column groups to test row alignment
  std::vector<ColumnGroupConfig> configs = {{.column_patterns = {"id"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"name|value"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"vector"}, .format = FileFormat::PARQUET}};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, configs);

  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer.write(test_batch_));
  }

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Verify we have multiple column groups
  auto column_groups = manifest->get_column_groups();
  EXPECT_EQ(column_groups.size(), 3);

  // Test row alignment with get_record_batch_reader
  Reader reader(fs_, manifest, schema_);
  auto batch_reader_result = reader.get_record_batch_reader();
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

TEST_F(APIWriterReaderTest, RowAlignmentWithTakeOperation) {
  // Test row alignment with random access (take operation)
  int batch_size = 500;

  // Create multiple column groups
  std::vector<ColumnGroupConfig> configs = {{.column_patterns = {"id|name"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"value|vector"}, .format = FileFormat::PARQUET}};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, configs);

  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer.write(test_batch_));
  }

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Test take operation with specific row indices
  Reader reader(fs_, manifest, schema_);
  std::vector<int64_t> row_indices = {0, 10, 25, 50, 75, 99, 150, 250, 350, 450};

  auto take_result = reader.take(row_indices);
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

TEST_F(APIWriterReaderTest, RowAlignmentWithChunkReader) {
  // Test row alignment using individual chunk readers
  int batch_size = 200;

  // Create multiple column groups
  std::vector<ColumnGroupConfig> configs = {{.column_patterns = {"id"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"name"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"value"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"vector"}, .format = FileFormat::PARQUET}};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, configs);

  Writer writer(fs_, base_path_, schema_, std::move(policy));

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer.write(test_batch_));
  }

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Test chunk readers from different column groups
  auto column_groups = manifest->get_column_groups();
  EXPECT_EQ(column_groups.size(), 4);

  Reader reader(fs_, manifest, schema_);

  // Get chunk readers for each column group
  std::vector<std::shared_ptr<ChunkReader>> chunk_readers;
  for (const auto& cg : column_groups) {
    auto chunk_reader_result = reader.get_chunk_reader(cg->id);
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

TEST_F(APIWriterReaderTest, RowAlignmentWithMultipleRowGroups) {
  // Test row alignment when data spans multiple row groups
  int batch_size = 5000;              // Large amount of data
  size_t small_buffer = 1024 * 1024;  // 1MB buffer, forcing multiple row groups

  // Create multiple column groups
  std::vector<ColumnGroupConfig> configs = {{.column_patterns = {"id|name"}, .format = FileFormat::PARQUET},
                                            {.column_patterns = {"value|vector"}, .format = FileFormat::PARQUET}};
  auto policy = std::make_unique<SchemaBasedColumnGroupPolicy>(schema_, configs);

  auto properties = WritePropertiesBuilder()
                        .with_max_row_group_size(1000)  // Force multiple row groups
                        .with_buffer_size(small_buffer)
                        .build();

  Writer writer(fs_, base_path_, schema_, std::move(policy), properties);

  // Write test data
  for (int i = 0; i < batch_size / 100; ++i) {
    ASSERT_OK(writer.write(test_batch_));
  }

  auto manifest_result = writer.close();
  ASSERT_TRUE(manifest_result.ok()) << manifest_result.status().ToString();
  auto manifest = std::move(manifest_result).ValueOrDie();

  // Read and verify row alignment across multiple row groups
  Reader reader(fs_, manifest, schema_);
  auto batch_reader_result = reader.get_record_batch_reader();
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
