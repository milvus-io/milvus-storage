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
#include <cstring>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/io/memory.h>
#include <arrow/io/file.h>
#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>

#include "test_env.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

namespace milvus_storage::test {

class ParquetFileWriterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("parquet-file-writer-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

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
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  milvus_storage::api::Properties properties_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
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
  std::string temp_file = base_path_ + "/data/test_large_batch.parquet";

  // Create packed writer and write record batch
  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 2 * 1024 * 1024));
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(writer->Write(record_batch).ok());
  }
  ASSERT_TRUE(writer->Close().ok());

  // Read back and verify
  ASSERT_AND_ASSIGN(auto reader, FileRowGroupReader::Make(fs_, temp_file, schema_));

  // Get metadata
  auto file_metadata = reader->file_metadata();
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
}

TEST_F(ParquetFileWriterTest, EmptyRecordBatch) {
  // Test writing empty record batch
  // Create empty arrays for each column in the schema
  auto id_array = arrow::MakeArrayOfNull(arrow::int64(), 0).ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), 0).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), 0).ValueOrDie();

  auto empty_batch = arrow::RecordBatch::Make(schema_, 0, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_empty_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(empty_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
}

TEST_F(ParquetFileWriterTest, NullRecordBatch) {
  // Test writing null record batch
  std::string temp_file = base_path_ + "/data/test_null_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  // Should handle null batch gracefully
  ASSERT_TRUE(writer->Write(nullptr).ok());
  ASSERT_TRUE(writer->Close().ok());
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

  std::string temp_file = base_path_ + "/data/test_small_buffer.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created and can be read
  ASSERT_AND_ASSIGN(auto reader, FileRowGroupReader::Make(fs_, temp_file, schema_));
  auto file_metadata = reader->file_metadata();
  ASSERT_GT(file_metadata->GetRowGroupMetadataVector().size(), 0);
}

TEST_F(ParquetFileWriterTest, LargeNumberOfSmallBatches) {
  // Test writing many small batches
  const int64_t batch_size = 10;
  const int num_batches = 100;

  std::string temp_file = base_path_ + "/data/test_many_small_batches.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

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
    ASSERT_TRUE(writer->Write(record_batch).ok());
  }

  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
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

  std::string temp_file = base_path_ + "/data/test_null_arrays.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
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

  std::string temp_file = base_path_ + "/data/test_mixed_data.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
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

  std::string temp_file = base_path_ + "/data/test_invalid_schema.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};

  // Should throw exception for null schema
  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, nullptr, config, column_groups, 1024 * 1024).ok());
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

  std::string temp_file = base_path_ + "/data/test_invalid_column_groups.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> invalid_column_groups = {{100, 200, 300}};  // Out of range

  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, schema_, config, invalid_column_groups, 1024 * 1024).ok());
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

  std::string temp_file = base_path_ + "/data/test_null_filesystem.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should throw exception for null file system
  ASSERT_FALSE(PackedRecordBatchWriter::Make(nullptr, paths, schema_, config, column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidFilePath) {
  // Test writing with invalid file path (empty path)
  StorageConfig config;
  std::vector<std::string> paths = {""};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should fail for empty file path
  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, TellBeforeAndAfterClose) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_tell.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  // Write data and flush
  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_STATUS_OK(writer->Flush());

  // Tell after flush should be > 0
  ASSERT_AND_ASSIGN(auto tell_before_close, writer->Tell());
  ASSERT_GT(tell_before_close, 0);

  // Close
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  // Tell after close should return cached value >= tell before close
  ASSERT_AND_ASSIGN(auto tell_after_close, writer->Tell());
  ASSERT_GE(tell_after_close, tell_before_close);

  // Verify tell matches actual file size
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(tell_after_close, static_cast<size_t>(file_info.size()));
}

TEST_F(ParquetFileWriterTest, PackedWriterTell) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file1 = base_path_ + "/data/test_packed_tell_1.parquet";
  std::string temp_file2 = base_path_ + "/data/test_packed_tell_2.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file1, temp_file2};
  // Split: columns 0,1 in group 0, columns 2,3 in group 1
  std::vector<std::vector<int>> column_groups = {{0, 1}, {2, 3}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, test_schema, config, column_groups, 1024 * 1024));

  // Write data
  ASSERT_STATUS_OK(writer->Write(record_batch));

  // Close
  ASSERT_STATUS_OK(writer->Close());

  // Tell after close
  ASSERT_AND_ASSIGN(auto positions, writer->Tell());
  ASSERT_EQ(positions.size(), 2);
  ASSERT_GT(positions[0], 0);
  ASSERT_GT(positions[1], 0);

  // Verify tell matches actual file sizes
  ASSERT_AND_ASSIGN(auto file_info1, fs_->GetFileInfo(temp_file1));
  ASSERT_EQ(positions[0], static_cast<size_t>(file_info1.size()));

  ASSERT_AND_ASSIGN(auto file_info2, fs_->GetFileInfo(temp_file2));
  ASSERT_EQ(positions[1], static_cast<size_t>(file_info2.size()));
}

TEST_F(ParquetFileWriterTest, FooterSizeMatchesActualFile) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_footer_size.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  auto cached_footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(cached_footer_size, 0u);

  // Read actual footer size from the file:
  // Parquet tail: [Thrift metadata][4B footer_length LE][4B magic "PAR1"]
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(temp_file));
  ASSERT_AND_ASSIGN(auto file_size, file->GetSize());

  // Read last 8 bytes
  ASSERT_AND_ASSIGN(auto tail_buf, file->ReadAt(file_size - 8, 8));
  const uint8_t* tail = tail_buf->data();

  uint32_t footer_length = 0;
  std::memcpy(&footer_length, tail, 4);
  // Verify magic
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(tail + 4), 4), "PAR1");

  uint64_t actual_footer_size = static_cast<uint64_t>(footer_length) + 8;
  EXPECT_EQ(cached_footer_size, actual_footer_size)
      << "cached footer_size=" << cached_footer_size << " actual=" << actual_footer_size;

  // Also verify file_size
  EXPECT_EQ(close_result.Get<uint64_t>(api::kPropertyFileSize), static_cast<uint64_t>(file_size));
}

// Helper: write a small record batch through ParquetFileWriter::Make and
// return the parquet column-chunk compression codecs (one entry per column,
// in schema order) for the first row group.
namespace {
arrow::Result<std::vector<::parquet::Compression::type>> WriteAndReadColumnCompression(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::string& file_path,
    bool use_properties_make,
    const milvus_storage::api::Properties& properties) {
  if (use_properties_make) {
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          milvus_storage::parquet::ParquetFileWriter::Make(fs, schema, file_path, properties));
    ARROW_RETURN_NOT_OK(writer->Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto _close, writer->Close());
    (void)_close;
  } else {
    milvus_storage::StorageConfig config;
    ARROW_ASSIGN_OR_RAISE(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(schema, fs, file_path, config));
    ARROW_RETURN_NOT_OK(writer->Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto _close, writer->Close());
    (void)_close;
  }

  ARROW_ASSIGN_OR_RAISE(auto file, fs->OpenInputFile(file_path));
  auto reader = ::parquet::ParquetFileReader::Open(file);
  auto metadata = reader->metadata();
  std::vector<::parquet::Compression::type> codecs;
  codecs.reserve(schema->num_fields());
  auto rg = metadata->RowGroup(0);
  for (int i = 0; i < rg->num_columns(); ++i) {
    codecs.push_back(rg->ColumnChunk(i)->compression());
  }
  return codecs;
}
}  // namespace

// Dense vector columns (FIXED_SIZE_BINARY) should land UNCOMPRESSED in the
// file. BINARY columns may carry LOB / sparse-vector payloads where
// compression can still help, so they inherit the file-level codec. Other
// columns also follow the file-level codec. Verified through both
// ParquetFileWriter::Make overloads.
TEST_F(ParquetFileWriterTest, FixedSizeBinaryColumnsAreUncompressed) {
  const int64_t num_rows = 16;
  auto schema = arrow::schema({
      arrow::field("id", arrow::int64(), false /*nullable*/, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("dense_vec", arrow::fixed_size_binary(128), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
      arrow::field("blob", arrow::binary(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
  });

  arrow::Int64Builder id_builder;
  arrow::FixedSizeBinaryBuilder dense_builder(arrow::fixed_size_binary(128));
  arrow::BinaryBuilder blob_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
    std::vector<uint8_t> v(128, static_cast<uint8_t>(i));
    ASSERT_TRUE(dense_builder.Append(v.data()).ok());
    std::string s(32, static_cast<char>('a' + (i % 26)));
    ASSERT_TRUE(blob_builder.Append(s).ok());
  }
  ASSERT_AND_ASSIGN(auto id_array, id_builder.Finish());
  ASSERT_AND_ASSIGN(auto dense_array, dense_builder.Finish());
  ASSERT_AND_ASSIGN(auto blob_array, blob_builder.Finish());
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {id_array, dense_array, blob_array});

  // Legacy Make (parquet::WriterProperties default => UNCOMPRESSED at file
  // level): constructor falls back to ZSTD-3 default. FIXED_SIZE_BINARY is
  // forced UNCOMPRESSED per-column; BINARY follows the file default (ZSTD).
  {
    auto file_path = base_path_ + "/data/vector_uncompressed_legacy.parquet";
    ASSERT_AND_ASSIGN(auto codecs, WriteAndReadColumnCompression(fs_, schema, batch, file_path,
                                                                 /*use_properties_make=*/false, properties_));
    ASSERT_EQ(codecs.size(), 3u);
    EXPECT_EQ(codecs[0], ::parquet::Compression::ZSTD) << "id should be ZSTD";
    EXPECT_EQ(codecs[1], ::parquet::Compression::UNCOMPRESSED) << "dense_vec should be UNCOMPRESSED";
    EXPECT_EQ(codecs[2], ::parquet::Compression::ZSTD) << "blob (BINARY) should follow file-level ZSTD";
  }

  // Properties-based Make (registry default zstd / level 3): same expectation.
  {
    auto file_path = base_path_ + "/data/vector_uncompressed_props.parquet";
    ASSERT_AND_ASSIGN(auto codecs, WriteAndReadColumnCompression(fs_, schema, batch, file_path,
                                                                 /*use_properties_make=*/true, properties_));
    ASSERT_EQ(codecs.size(), 3u);
    EXPECT_EQ(codecs[0], ::parquet::Compression::ZSTD) << "id should be ZSTD";
    EXPECT_EQ(codecs[1], ::parquet::Compression::UNCOMPRESSED) << "dense_vec should be UNCOMPRESSED";
    EXPECT_EQ(codecs[2], ::parquet::Compression::ZSTD) << "blob (BINARY) should follow file-level ZSTD";
  }
}

// File-level ZSTD setting does not leak into vector columns — they are
// always emitted UNCOMPRESSED regardless of the caller's WriterProperties.
TEST_F(ParquetFileWriterTest, FileLevelCompressionDoesNotPreventVectorUncompressed) {
  const int64_t num_rows = 8;
  auto schema = arrow::schema({
      arrow::field("dense_vec", arrow::fixed_size_binary(64), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
  });

  arrow::FixedSizeBinaryBuilder b(arrow::fixed_size_binary(64));
  for (int64_t i = 0; i < num_rows; ++i) {
    std::vector<uint8_t> v(64, static_cast<uint8_t>(i));
    ASSERT_TRUE(b.Append(v.data()).ok());
  }
  ASSERT_AND_ASSIGN(auto arr, b.Finish());
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {arr});

  // File-level ZSTD-7 — vector column still emitted UNCOMPRESSED.
  auto props =
      ::parquet::WriterProperties::Builder().compression(::parquet::Compression::ZSTD)->compression_level(7)->build();
  auto file_path = base_path_ + "/data/vector_filelevel_only.parquet";
  milvus_storage::StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer,
                    milvus_storage::parquet::ParquetFileWriter::Make(schema, fs_, file_path, config, props));
  ASSERT_TRUE(writer->Write(batch).ok());
  ASSERT_AND_ASSIGN(auto _close, writer->Close());
  (void)_close;

  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(file_path));
  auto reader = ::parquet::ParquetFileReader::Open(file);
  auto metadata = reader->metadata();
  EXPECT_EQ(metadata->RowGroup(0)->ColumnChunk(0)->compression(), ::parquet::Compression::UNCOMPRESSED);
}

TEST_F(ParquetFileWriterTest, FooterSizeNotMatch) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_footer_size_mismatch.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());
  auto cached_footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  auto cached_file_size = close_result.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(cached_footer_size, 0u);
  ASSERT_GT(cached_file_size, cached_footer_size);

  // Test reading with different footer_size values passed to ParquetFormatReader.
  // The reader uses footer_size to pre-read the footer in a single IO;
  // if the size is wrong, it falls back to Arrow's normal 2-step footer read.
  auto verify_read = [&](uint64_t footer_size) {
    auto reader =
        milvus_storage::parquet::ParquetFormatReader(fs_, temp_file, properties_, /*needed_columns=*/{},
                                                     /*key_retriever=*/nullptr, cached_file_size, footer_size);
    ASSERT_STATUS_OK(reader.open());

    ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
    ASSERT_GT(row_group_infos.size(), 0u);

    // Read first row group to verify data integrity
    ASSERT_AND_ASSIGN(auto rb, reader.get_chunk(0));
    ASSERT_GT(rb->num_rows(), 0);
  };

  // Case 1: footer_size too small (1 byte).
  // Pre-read can't cover the Thrift metadata → falls back to Arrow's normal 2-step footer read.
  verify_read(1);

  // Case 2: footer_size too large (= file_size).
  // Pre-reads entire file as suffix. Correctly locates footer_length and magic at the end.
  verify_read(cached_file_size);
}

}  // namespace milvus_storage::test
