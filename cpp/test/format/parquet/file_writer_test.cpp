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
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/io/memory.h>
#include <arrow/io/file.h>
#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <sys/stat.h>

#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/packed/writer.h"

namespace milvus_storage {

class ParquetFileWriterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Create schema with mixed data types
    // Current test case exist some nullable columns
    // should set all field `nullable` to true.
    auto id_field =
        arrow::field("id", arrow::int64(), true /*nullable*/, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));
    auto text_field = arrow::field("text", arrow::utf8(), true /*nullable*/,
                                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}));
    auto vector_field = arrow::field("vector", arrow::fixed_size_binary(128), true /*nullable*/,
                                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"}));

    schema_ = arrow::schema({id_field, text_field, vector_field});

    // Create file system
    auto result = arrow::fs::FileSystemFromUri("file:///");
    fs_ = result.ValueOrDie();
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

TEST_F(ParquetFileWriterTest, LargeRecordBatchSplitting) {
  // Create a large record batch with mixed data sizes
  const int64_t num_rows = 1000;

  // Create ID array (small, uniform size)
  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();

  // Create text array (mixed sizes - some very large)
  arrow::StringBuilder text_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    if (i % 20 == 0) {
      // Every 20th row has a very large text (simulating large text field)
      std::string large_text(50000, 'x');  // 50KB text
      ASSERT_TRUE(text_builder.Append(large_text).ok());
    } else {
      // Normal rows have small text
      std::string small_text = "row_" + std::to_string(i);
      ASSERT_TRUE(text_builder.Append(small_text).ok());
    }
  }
  auto text_array = text_builder.Finish().ValueOrDie();

  // Create vector array (uniform size)
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));
  std::vector<uint8_t> vector_data(128, 0);
  for (int64_t i = 0; i < num_rows; ++i) {
    // Fill with some pattern
    for (int j = 0; j < 128; ++j) {
      vector_data[j] = static_cast<uint8_t>((i + j) % 256);
    }
    ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
  }
  auto vector_array = vector_builder.Finish().ValueOrDie();

  // Create record batch
  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  // Create temporary file path
  std::string temp_file = "/tmp/test_large_batch.parquet";

  // Create packed writer and write record batch
  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 2 * 1024 * 1024);  // 2MB buffer
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(writer.Write(record_batch).ok());
  }
  ASSERT_TRUE(writer.Close().ok());

  // Read back and verify
  FileRowGroupReader reader(fs_, temp_file, schema_);

  // Get metadata
  auto file_metadata = reader.file_metadata();
  auto row_group_metadata = file_metadata->GetRowGroupMetadataVector();
  int num_row_groups = row_group_metadata.size();

  // Verify each row group size
  for (int i = 0; i < num_row_groups; ++i) {
    const auto& metadata = row_group_metadata.Get(i);
    int64_t row_group_size = metadata.memory_size();

    // Verify that row group size is reasonable (should be around 1MB)
    EXPECT_LE(row_group_size, DEFAULT_MAX_ROW_GROUP_SIZE * 1.1);  // Allow some tolerance

    // only the last row group should be less than 1MB
    if (i < num_row_groups - 1) {
      EXPECT_GT(row_group_size, DEFAULT_MAX_ROW_GROUP_SIZE);
    }
  }

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, EmptyRecordBatch) {
  // Test writing empty record batch
  // Create empty arrays for each column in the schema
  auto id_array = arrow::MakeArrayOfNull(arrow::int64(), 0).ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), 0).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), 0).ValueOrDie();

  auto empty_batch = arrow::RecordBatch::Make(schema_, 0, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_empty_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024 * 1024);

  ASSERT_TRUE(writer.Write(empty_batch).ok());
  ASSERT_TRUE(writer.Close().ok());

  // Verify file was created
  struct stat buffer;
  ASSERT_EQ(stat(temp_file.c_str(), &buffer), 0);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, NullRecordBatch) {
  // Test writing null record batch
  std::string temp_file = "/tmp/test_null_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024 * 1024);

  // Should handle null batch gracefully
  ASSERT_TRUE(writer.Write(nullptr).ok());
  ASSERT_TRUE(writer.Close().ok());

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, VerySmallBufferSize) {
  // Test with very small buffer size
  const int64_t num_rows = 100;

  // Create simple record batch
  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
    ASSERT_TRUE(text_builder.Append("row_" + std::to_string(i)).ok());

    std::vector<uint8_t> vector_data(128, static_cast<uint8_t>(i % 256));
    ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
  }

  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = text_builder.Finish().ValueOrDie();
  auto vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_small_buffer.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024);  // 1KB buffer

  ASSERT_TRUE(writer.Write(record_batch).ok());
  ASSERT_TRUE(writer.Close().ok());

  // Verify file was created and can be read
  FileRowGroupReader reader(fs_, temp_file, schema_);
  auto file_metadata = reader.file_metadata();
  ASSERT_GT(file_metadata->GetRowGroupMetadataVector().size(), 0);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, ZeroBufferSize) {
  // Test with zero buffer size
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_zero_buffer.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 0);  // Zero buffer

  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(writer.Write(record_batch).ok());
  }
  ASSERT_TRUE(writer.Close().ok());

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, NegativeBufferSize) {
  // Test with negative buffer size
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_negative_buffer.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, -1);  // Negative buffer

  ASSERT_TRUE(writer.Write(record_batch).ok());
  ASSERT_TRUE(writer.Close().ok());

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, LargeNumberOfSmallBatches) {
  // Test writing many small batches
  const int64_t batch_size = 10;
  const int num_batches = 100;

  std::string temp_file = "/tmp/test_many_small_batches.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024 * 1024);

  for (int batch = 0; batch < num_batches; ++batch) {
    arrow::Int64Builder id_builder;
    arrow::StringBuilder text_builder;
    arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

    for (int64_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(id_builder.Append(batch * batch_size + i).ok());
      ASSERT_TRUE(text_builder.Append("batch_" + std::to_string(batch) + "_row_" + std::to_string(i)).ok());

      std::vector<uint8_t> vector_data(128, static_cast<uint8_t>((batch + i) % 256));
      ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
    }

    auto id_array = id_builder.Finish().ValueOrDie();
    auto text_array = text_builder.Finish().ValueOrDie();
    auto vector_array = vector_builder.Finish().ValueOrDie();

    auto record_batch = arrow::RecordBatch::Make(schema_, batch_size, {id_array, text_array, vector_array});
    ASSERT_TRUE(writer.Write(record_batch).ok());
  }

  ASSERT_TRUE(writer.Close().ok());

  // Verify file was created
  struct stat buffer;
  ASSERT_EQ(stat(temp_file.c_str(), &buffer), 0);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, WriteWithNullArrays) {
  // Test writing record batch with null arrays
  const int64_t num_rows = 100;

  // Create null arrays using builders instead of MakeArrayOfNull
  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  // Append nulls for all rows
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.AppendNull().ok());
    ASSERT_TRUE(text_builder.AppendNull().ok());
    // For FixedSizeBinary, we append zero vectors instead of nulls
    std::vector<uint8_t> zero_vector(128, 0);
    ASSERT_TRUE(vector_builder.Append(zero_vector.data()).ok());
  }

  auto null_id_array = id_builder.Finish().ValueOrDie();
  auto null_text_array = text_builder.Finish().ValueOrDie();
  auto null_vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {null_id_array, null_text_array, null_vector_array});

  std::string temp_file = "/tmp/test_null_arrays.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024 * 1024);

  ASSERT_TRUE(writer.Write(record_batch).ok());
  ASSERT_TRUE(writer.Close().ok());

  // Verify file was created
  struct stat buffer;
  ASSERT_EQ(stat(temp_file.c_str(), &buffer), 0);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, WriteWithMixedNullAndValidData) {
  // Test writing record batch with mixed null and valid data
  const int64_t num_rows = 100;

  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  for (int64_t i = 0; i < num_rows; ++i) {
    if (i % 3 == 0) {
      ASSERT_TRUE(id_builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(id_builder.Append(i).ok());
    }

    if (i % 5 == 0) {
      ASSERT_TRUE(text_builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(text_builder.Append("row_" + std::to_string(i)).ok());
    }

    if (i % 7 == 0) {
      // FixedSizeBinaryBuilder doesn't support AppendNull, so we append a zero vector instead
      std::vector<uint8_t> zero_vector(128, 0);
      ASSERT_TRUE(vector_builder.Append(zero_vector.data()).ok());
    } else {
      std::vector<uint8_t> vector_data(128, static_cast<uint8_t>(i % 256));
      ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
    }
  }

  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = text_builder.Finish().ValueOrDie();
  auto vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_mixed_data.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  PackedRecordBatchWriter writer(fs_, paths, schema_, config, column_groups, 1024 * 1024);

  ASSERT_TRUE(writer.Write(record_batch).ok());
  ASSERT_TRUE(writer.Close().ok());

  // Verify file was created
  struct stat buffer;
  ASSERT_EQ(stat(temp_file.c_str(), &buffer), 0);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidSchema) {
  // Test writing with invalid schema (null schema)
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, id_array, id_array});

  std::string temp_file = "/tmp/test_invalid_schema.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};

  // Should throw exception for null schema
  EXPECT_THROW(PackedRecordBatchWriter(fs_, paths, nullptr, config, column_groups, 1024 * 1024), std::runtime_error);

  std::remove(temp_file.c_str());
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidColumnGroups) {
  // Test writing with invalid column groups (out of range indices)
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_invalid_column_groups.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> invalid_column_groups = {{100, 200, 300}};  // Out of range

  // Should throw exception for invalid column groups
  EXPECT_THROW(PackedRecordBatchWriter(fs_, paths, schema_, config, invalid_column_groups, 1024 * 1024),
               std::out_of_range);
}

TEST_F(ParquetFileWriterTest, WriteWithNullFileSystem) {
  // Test writing with null filesystem
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = "/tmp/test_null_filesystem.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should throw exception for null file system
  EXPECT_THROW(PackedRecordBatchWriter(nullptr, paths, schema_, config, column_groups, 1024 * 1024),
               std::runtime_error);
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidFilePath) {
  // Test writing with invalid file path
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string invalid_path = "/invalid/path/test.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {invalid_path};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should throw exception for invalid file path
  EXPECT_THROW(PackedRecordBatchWriter(fs_, paths, schema_, config, column_groups, 1024 * 1024), std::runtime_error);
}

}  // namespace milvus_storage