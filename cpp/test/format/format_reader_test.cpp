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

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class FormatReaderTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // Mirror fs.* into extfs.default.* so resolve_config can match Lance's Milvus-format URI
    ArrowFileSystemConfig fs_config;
    if (ArrowFileSystemConfig::create_file_system_config(properties_, fs_config).ok() &&
        fs_config.storage_type == "remote") {
      api::SetValue(properties_, "extfs.default.storage_type", "remote");
      api::SetValue(properties_, "extfs.default.cloud_provider", fs_config.cloud_provider.c_str());
      api::SetValue(properties_, "extfs.default.address", fs_config.address.c_str());
      api::SetValue(properties_, "extfs.default.bucket_name", fs_config.bucket_name.c_str());
      api::SetValue(properties_, "extfs.default.region", fs_config.region.c_str());
      api::SetValue(properties_, "extfs.default.access_key_id", fs_config.access_key_id.c_str());
      api::SetValue(properties_, "extfs.default.access_key_value", fs_config.access_key_value.c_str());
      if (fs_config.use_ssl) {
        api::SetValue(properties_, "extfs.default.use_ssl", "true");
      }
      if (fs_config.use_iam) {
        api::SetValue(properties_, "extfs.default.use_iam", "true");
      }
    }

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
  auto parquet_writer_props = ::parquet::WriterProperties::Builder()
                                  .compression(::parquet::Compression::ZSTD)
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
                                                              .end_index = test_batch_->num_rows() * 10},
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
  for (const auto& rb : rbs) {
    total_size += rb->num_rows();
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

#ifdef BUILD_WITH_FIU
TEST_P(FormatReaderTest, S3FilesystemWriterCloseFailureShouldPropagate) {
  std::string format = GetParam();
  if (format == LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "Test parquet and vortex only.";
  }

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  const auto is_s3_provider =
      fs_config.cloud_provider == kCloudProviderAWS || fs_config.cloud_provider == kCloudProviderAliyun ||
      fs_config.cloud_provider == kCloudProviderTencent || fs_config.cloud_provider == kCloudProviderHuawei;
  if (fs_config.storage_type != "remote" || !is_s3_provider) {
    GTEST_SKIP() << "Test requires S3-backed remote filesystem.";
  }

  const char* fault_keys[] = {
      FIUKEY_S3FS_WRITER_WRITE_FAIL,
      FIUKEY_S3FS_WRITER_FLUSH_FAIL,
      FIUKEY_S3FS_WRITER_CLOSE_FAIL,
  };

  ASSERT_EQ(0, InitFiuOnce());
  for (const auto* fault_key : fault_keys) {
    SCOPED_TRACE(fault_key);
    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
    auto writer = Writer::create(base_path_ + "/" + fault_key, schema_, std::move(policy), properties_);
    ASSERT_OK(writer->write(test_batch_));

    ASSERT_EQ(0, FIU_ENABLE_FAULT_ONETIME(fault_key));
    auto close_result = writer->close();
    EXPECT_FALSE(close_result.ok()) << "Writer::close unexpectedly succeeded after filesystem fault " << fault_key
                                    << " for format=" << format;
  }
}

TEST_P(FormatReaderTest, S3FilesystemReaderFailureShouldPropagate) {
  std::string format = GetParam();
  if (format == LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "Test parquet and vortex only.";
  }

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  const auto is_s3_provider =
      fs_config.cloud_provider == kCloudProviderAWS || fs_config.cloud_provider == kCloudProviderAliyun ||
      fs_config.cloud_provider == kCloudProviderTencent || fs_config.cloud_provider == kCloudProviderHuawei;
  if (fs_config.storage_type != "remote" || !is_s3_provider) {
    GTEST_SKIP() << "Test requires S3-backed remote filesystem.";
  }

  ASSERT_EQ(0, InitFiuOnce());

  if (format == LOON_FORMAT_PARQUET) {
    const std::string raw_path = base_path_ + "/reader-fiu-raw";
    const std::string raw_data = "reader-fiu";
    ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(raw_path));
    ASSERT_STATUS_OK(sink->Write(raw_data.data(), raw_data.size()));
    ASSERT_STATUS_OK(sink->Close());

    {
      ASSERT_AND_ASSIGN(auto input, fs_->OpenInputFile(raw_path));
      ASSERT_EQ(0, FIU_ENABLE_FAULT_ONETIME(FIUKEY_S3FS_READER_READ_FAIL));
      auto read_result = input->Read(1);
      EXPECT_FALSE(read_result.ok()) << "Read unexpectedly succeeded after filesystem fault "
                                     << FIUKEY_S3FS_READER_READ_FAIL;
    }

    {
      ASSERT_AND_ASSIGN(auto input, fs_->OpenInputFile(raw_path));
      ASSERT_EQ(0, FIU_ENABLE_FAULT_ONETIME(FIUKEY_S3FS_READER_READAT_FAIL));
      auto read_result = input->ReadAt(0, 1);
      EXPECT_FALSE(read_result.ok()) << "ReadAt unexpectedly succeeded after filesystem fault "
                                     << FIUKEY_S3FS_READER_READAT_FAIL;
    }

    return;
  }

  ASSERT_AND_ASSIGN(auto id_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, false}));
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema_));
  auto writer = Writer::create(base_path_ + "/reader-fiu-" + format, schema_, std::move(policy), properties_);
  ASSERT_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());
  auto cg = std::find_if(cgs->begin(), cgs->end(),
                         [](const std::shared_ptr<ColumnGroup>& cg) { return cg->columns[0] == "id"; });
  ASSERT_NE(cg, cgs->end());
  ASSERT_EQ((*cg)->files.size(), 1);

  ASSERT_EQ(0, FIU_ENABLE_FAULT_ONETIME(FIUKEY_S3FS_READER_READAT_FAIL));
  try {
    auto format_reader_result =
        FormatReader::create(id_schema, format, (*cg)->files[0], properties_, std::vector<std::string>{"id"}, nullptr);
    EXPECT_FALSE(format_reader_result.ok()) << "FormatReader::create unexpectedly succeeded after filesystem fault "
                                            << FIUKEY_S3FS_READER_READAT_FAIL << " for format=" << format;
  } catch (const std::exception& e) {
    FAIL() << "FormatReader::create threw instead of returning status after filesystem fault "
           << FIUKEY_S3FS_READER_READAT_FAIL << " for format=" << format << ": " << e.what();
  }
}
#endif

TEST_P(FormatReaderTest, ParquetReadWithRangeReaderOutlivesFormatReader) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  auto parquet_writer_props = ::parquet::WriterProperties::Builder()
                                  .compression(::parquet::Compression::ZSTD)
                                  ->enable_dictionary()
                                  ->enable_statistics()
                                  ->build();

  const auto file_path = base_path_ + "/range_outlives_reader.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(file_path));
  ASSERT_AND_ASSIGN(auto parquet_writer, ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(),
                                                                            sink, parquet_writer_props));
  ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
  ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*test_batch_));
  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  std::shared_ptr<arrow::RecordBatchReader> rb_reader;
  {
    std::shared_ptr<FormatReader> format_reader;
    ASSERT_AND_ASSIGN(format_reader, FormatReader::create(
                                         schema_, LOON_FORMAT_PARQUET,
                                         api::ColumnGroupFile{
                                             .path = file_path, .start_index = 0, .end_index = test_batch_->num_rows()},
                                         properties_, std::vector<std::string>{"id"}, nullptr));
    ASSERT_AND_ASSIGN(rb_reader, format_reader->read_with_range(1, 4));
  }

  ASSERT_AND_ASSIGN(auto rbs, rb_reader->ToRecordBatches());
  ASSERT_AND_ASSIGN(auto rb, arrow::ConcatenateRecordBatches(rbs));
  ASSERT_EQ(3, rb->num_rows());
  ASSERT_EQ(1, rb->num_columns());
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  ASSERT_NE(nullptr, id_array);
  EXPECT_EQ(1, id_array->Value(0));
  EXPECT_EQ(2, id_array->Value(1));
  EXPECT_EQ(3, id_array->Value(2));
}

TEST_P(FormatReaderTest, ParquetCreateFromMetadataReappliesProjection) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  ASSERT_AND_ASSIGN(auto two_cols_schema, CreateTestSchema(std::array<bool, 4>{true, false, true, false}));
  ASSERT_AND_ASSIGN(auto two_cols_batch, CreateTestData(two_cols_schema, 0, false, 10, 4, 50,
                                                        std::array<bool, 4>{true, false, true, false}));

  const auto file_path = base_path_ + "/cached_projection.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(file_path));
  ASSERT_AND_ASSIGN(auto parquet_writer,
                    ::parquet::arrow::FileWriter::Open(*two_cols_schema, arrow::default_memory_pool(), sink));
  ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
  ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*two_cols_batch));
  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  api::ColumnGroupFile file{.path = file_path, .start_index = 0, .end_index = two_cols_batch->num_rows()};
  ASSERT_AND_ASSIGN(auto metadata,
                    FormatReader::load_metadata<parquet::ParquetFormatReader>(file, properties_, nullptr));
  auto id_metadata = metadata;
  auto value_metadata = metadata;
  ASSERT_EQ(id_metadata.get(), value_metadata.get());

  ASSERT_AND_ASSIGN(auto id_reader, FormatReader::create_from_metadata<parquet::ParquetFormatReader>(
                                        id_metadata, file, two_cols_schema, {"id"}, ""));
  ASSERT_AND_ASSIGN(auto id_batch, id_reader->get_chunk(0));
  ASSERT_EQ(id_batch->num_rows(), two_cols_batch->num_rows());
  ASSERT_EQ(id_batch->num_columns(), 1);
  ASSERT_EQ(id_batch->schema()->field(0)->name(), "id");
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(id_batch->column(0));
  ASSERT_NE(id_array, nullptr);
  for (int64_t i = 0; i < id_batch->num_rows(); ++i) {
    ASSERT_EQ(id_array->Value(i), i);
  }

  ASSERT_AND_ASSIGN(auto value_reader, FormatReader::create_from_metadata<parquet::ParquetFormatReader>(
                                           value_metadata, file, two_cols_schema, {"value"}, ""));
  ASSERT_AND_ASSIGN(auto value_batch, value_reader->get_chunk(0));
  ASSERT_EQ(value_batch->num_rows(), two_cols_batch->num_rows());
  ASSERT_EQ(value_batch->num_columns(), 1);
  ASSERT_EQ(value_batch->schema()->field(0)->name(), "value");
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(value_batch->column(0));
  ASSERT_NE(value_array, nullptr);
  for (int64_t i = 0; i < value_batch->num_rows(); ++i) {
    ASSERT_DOUBLE_EQ(value_array->Value(i), i * 1.5);
  }
}

INSTANTIATE_TEST_SUITE_P(FormatReaderTestP,
                         FormatReaderTest,
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE));

}  // namespace milvus_storage::test
