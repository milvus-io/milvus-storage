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
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/array/array_dict.h>
#include <arrow/array/concatenate.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

class VortexTestBase : public ::testing::Test {
  protected:
  void CommonSetUp(uint32_t format_version) {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, std::to_string(format_version).c_str());

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema(needed_columns_));
    for (int64_t batch_idx = 0; batch_idx < batch_count_; ++batch_idx) {
      ASSERT_AND_ASSIGN(auto rb, MakeTestData(batch_idx * rows_per_batch_, rows_per_batch_));
      record_batches_.emplace_back(std::move(rb));
    }

    ASSERT_AND_ASSIGN(file_system_, GetFileSystem(properties_));
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(test_file_name_);
    }
  }

  protected:
  [[nodiscard]] inline int64_t recordBatchsRows() const { return batch_count_ * rows_per_batch_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeTestData(int64_t start_offset, size_t num_rows) {
    return CreateTestData(schema_, start_offset, false, num_rows, 4, 50, needed_columns_);
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkedArrayToRecordBatch(
      const std::shared_ptr<arrow::ChunkedArray>& chunkedarray) {
    auto chunk_size = chunkedarray->num_chunks();
    if (chunk_size == 1) {
      return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
    for (int i = 0; i < chunk_size; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
      rbs.emplace_back(rb);
    }

    return arrow::ConcatenateRecordBatches(rbs);
  }

  const std::vector<std::string>& data_columns() const { return data_columns_; }

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
  const char* test_file_name_ = "test-file.vx";
  api::Properties properties_;

  private:
  const std::array<bool, 4> needed_columns_ = {true, true, true, false};
  const std::vector<std::string> data_columns_ = {"id", "name", "value"};
  const int64_t rows_per_batch_ = 1024;
  const int64_t batch_count_ = 4;
};

// Parameterized fixture: runs each test for both V1 and V2 format
class VortexBasicTest : public VortexTestBase, public ::testing::WithParamInterface<uint32_t> {
  void SetUp() override { CommonSetUp(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(V1V2,
                         VortexBasicTest,
                         ::testing::Values(1, 2),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return "V" + std::to_string(info.param);
                         });

namespace {

constexpr int kDictionaryValueGroup = 3;

std::string MakeFixedSizeBinaryValue(int64_t row, int64_t group, int64_t index, int byte_width) {
  std::string value(byte_width, '\0');
  for (int byte_index = 0; byte_index < byte_width; ++byte_index) {
    value[byte_index] = static_cast<char>((row * 31 + group * 17 + index * 7 + byte_index) & 0xff);
  }
  return value;
}

arrow::Status AppendFixedSizeBinaryValue(
    arrow::FixedSizeBinaryBuilder* builder, int64_t row, int64_t group, int64_t index, int byte_width) {
  return builder->Append(MakeFixedSizeBinaryValue(row, group, index, byte_width));
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeFixedSizeBinaryArray(int64_t num_rows, int vector_width) {
  arrow::FixedSizeBinaryBuilder fsb_builder(arrow::fixed_size_binary(vector_width));

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(&fsb_builder, row, 0, 0, vector_width));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(fsb_builder.Finish(&array));
  return array;
}

void AssertFixedSizeBinaryValue(const std::shared_ptr<arrow::FixedSizeBinaryArray>& array,
                                int64_t index,
                                const std::string& expected) {
  ASSERT_NE(array, nullptr);
  ASSERT_FALSE(array->IsNull(index));
  ASSERT_EQ(static_cast<int32_t>(expected.size()), array->byte_width());
  ASSERT_EQ(0, std::memcmp(array->GetValue(index), expected.data(), expected.size()));
}

void AssertFixedSizeBinaryArray(const std::shared_ptr<arrow::FixedSizeBinaryArray>& array,
                                int64_t source_row_offset,
                                int64_t num_rows,
                                int vector_width) {
  ASSERT_NE(array, nullptr);
  ASSERT_EQ(num_rows, array->length());
  ASSERT_EQ(vector_width, array->byte_width());

  for (int64_t row = 0; row < num_rows; ++row) {
    AssertFixedSizeBinaryValue(array, row, MakeFixedSizeBinaryValue(source_row_offset + row, 0, 0, vector_width));
  }
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeListOfFixedSizeBinaryArray(int64_t num_rows,
                                                                            int vectors_per_row,
                                                                            int vector_width) {
  auto value_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_width));
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(value_builder.get(), row, 0, vector_index, vector_width));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeLargeListOfFixedSizeBinaryArray(int64_t num_rows,
                                                                                 int vectors_per_row,
                                                                                 int vector_width) {
  auto value_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_width));
  arrow::LargeListBuilder list_builder(arrow::default_memory_pool(), value_builder);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(value_builder.get(), row, 0, vector_index, vector_width));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeFixedSizeListOfFixedSizeBinaryArray(int64_t num_rows,
                                                                                     int vectors_per_row,
                                                                                     int vector_width) {
  auto value_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_width));
  arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(), value_builder, vectors_per_row);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(value_builder.get(), row, 0, vector_index, vector_width));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&array));
  return array;
}

template <typename ListArrayType>
void AssertFixedSizeBinaryListValues(const std::shared_ptr<ListArrayType>& list_array,
                                     int64_t source_row_offset,
                                     int64_t num_rows,
                                     int vectors_per_row,
                                     int vector_width) {
  ASSERT_NE(list_array, nullptr);
  ASSERT_EQ(num_rows, list_array->length());

  for (int64_t row = 0; row < num_rows; ++row) {
    ASSERT_FALSE(list_array->IsNull(row));
    auto values = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(list_array->value_slice(row));
    ASSERT_NE(values, nullptr);
    ASSERT_EQ(vectors_per_row, values->length());
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      AssertFixedSizeBinaryValue(values, vector_index,
                                 MakeFixedSizeBinaryValue(source_row_offset + row, 0, vector_index, vector_width));
    }
  }
}

void AssertListOfFixedSizeBinaryArray(const std::shared_ptr<arrow::ListArray>& list_array,
                                      int64_t source_row_offset,
                                      int64_t num_rows,
                                      int vectors_per_row,
                                      int vector_width) {
  AssertFixedSizeBinaryListValues(list_array, source_row_offset, num_rows, vectors_per_row, vector_width);
}

void AssertLargeListOfFixedSizeBinaryArray(const std::shared_ptr<arrow::LargeListArray>& list_array,
                                           int64_t source_row_offset,
                                           int64_t num_rows,
                                           int vectors_per_row,
                                           int vector_width) {
  AssertFixedSizeBinaryListValues(list_array, source_row_offset, num_rows, vectors_per_row, vector_width);
}

void AssertFixedSizeListOfFixedSizeBinaryArray(const std::shared_ptr<arrow::FixedSizeListArray>& list_array,
                                               int64_t source_row_offset,
                                               int64_t num_rows,
                                               int vectors_per_row,
                                               int vector_width) {
  AssertFixedSizeBinaryListValues(list_array, source_row_offset, num_rows, vectors_per_row, vector_width);
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeStructOfFixedSizeBinaryArray(int64_t num_rows, int vector_width) {
  arrow::FixedSizeBinaryBuilder fsb_builder(arrow::fixed_size_binary(vector_width));
  arrow::Int32Builder int_builder;

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(&fsb_builder, row, 1, 0, vector_width));
    ARROW_RETURN_NOT_OK(int_builder.Append(static_cast<int32_t>(row * 13)));
  }

  std::shared_ptr<arrow::Array> fsb_array;
  std::shared_ptr<arrow::Array> int_array;
  ARROW_RETURN_NOT_OK(fsb_builder.Finish(&fsb_array));
  ARROW_RETURN_NOT_OK(int_builder.Finish(&int_array));

  return arrow::StructArray::Make({fsb_array, int_array},
                                  {arrow::field("fsb", arrow::fixed_size_binary(vector_width), false),
                                   arrow::field("other", arrow::int32(), false)});
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeNullableFixedSizeBinaryArray(int64_t num_rows, int vector_width) {
  arrow::FixedSizeBinaryBuilder fsb_builder(arrow::fixed_size_binary(vector_width));

  for (int64_t row = 0; row < num_rows; ++row) {
    if (row == 1 || row == 4) {
      ARROW_RETURN_NOT_OK(fsb_builder.AppendNull());
    } else {
      ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(&fsb_builder, row, 2, 0, vector_width));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(fsb_builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeNestedListOfFixedSizeBinaryArray(int64_t num_rows,
                                                                                  int groups_per_row,
                                                                                  int vectors_per_group,
                                                                                  int vector_width) {
  auto value_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_width));
  auto inner_list_builder = std::make_shared<arrow::ListBuilder>(arrow::default_memory_pool(), value_builder);
  arrow::ListBuilder outer_list_builder(arrow::default_memory_pool(), inner_list_builder);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(outer_list_builder.Append());
    for (int group = 0; group < groups_per_row; ++group) {
      ARROW_RETURN_NOT_OK(inner_list_builder->Append());
      for (int vector_index = 0; vector_index < vectors_per_group; ++vector_index) {
        ARROW_RETURN_NOT_OK(AppendFixedSizeBinaryValue(value_builder.get(), row, group, vector_index, vector_width));
      }
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(outer_list_builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeDictionaryOfFixedSizeBinaryArray(const std::vector<int32_t>& indices,
                                                                                  int dictionary_size,
                                                                                  int vector_width) {
  arrow::FixedSizeBinaryBuilder dictionary_builder(arrow::fixed_size_binary(vector_width));
  for (int dict_index = 0; dict_index < dictionary_size; ++dict_index) {
    ARROW_RETURN_NOT_OK(
        AppendFixedSizeBinaryValue(&dictionary_builder, dict_index, kDictionaryValueGroup, 0, vector_width));
  }

  arrow::Int32Builder indices_builder;
  for (auto index : indices) {
    if (index < 0) {
      ARROW_RETURN_NOT_OK(indices_builder.AppendNull());
    } else {
      ARROW_RETURN_NOT_OK(indices_builder.Append(index));
    }
  }

  std::shared_ptr<arrow::Array> dictionary_values;
  std::shared_ptr<arrow::Array> index_array;
  ARROW_RETURN_NOT_OK(dictionary_builder.Finish(&dictionary_values));
  ARROW_RETURN_NOT_OK(indices_builder.Finish(&index_array));

  return arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::fixed_size_binary(vector_width)),
                                            index_array, dictionary_values);
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeFixedSizeListUInt8Array(int64_t num_rows, int value_width) {
  auto value_builder = std::make_shared<arrow::UInt8Builder>();
  arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(), value_builder, value_width);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    for (int byte_index = 0; byte_index < value_width; ++byte_index) {
      ARROW_RETURN_NOT_OK(value_builder->Append(static_cast<uint8_t>((row * 17 + byte_index) & 0xff)));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeFixedSizeListUInt8ArrayWithChildNull(int64_t num_rows,
                                                                                      int value_width) {
  auto value_builder = std::make_shared<arrow::UInt8Builder>();
  auto list_type = arrow::fixed_size_list(arrow::field("item", arrow::uint8(), true), value_width);
  arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(), value_builder, list_type);

  for (int64_t row = 0; row < num_rows; ++row) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    for (int byte_index = 0; byte_index < value_width; ++byte_index) {
      if (row == 1 && byte_index == 2) {
        ARROW_RETURN_NOT_OK(value_builder->AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(value_builder->Append(static_cast<uint8_t>((row * 17 + byte_index) & 0xff)));
      }
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&array));
  return array;
}

}  // namespace

TEST_P(VortexBasicTest, TestBasicWrite) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);
}

TEST_P(VortexBasicTest, FlushAllowsSubsequentWrites) {
  ASSERT_AND_ASSIGN(auto first_batch, MakeTestData(0, 1));
  ASSERT_AND_ASSIGN(auto second_batch, MakeTestData(1, 1));
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(first_batch));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_STATUS_OK(vx_writer.Write(second_batch));

  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(first_batch->num_rows() + second_batch->num_rows(), cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, cgfile.end_index, kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(cgfile.end_index, rb->num_rows());
}

#ifdef BUILD_WITH_FIU
TEST_P(VortexBasicTest, S3FlushFailureCloseReturnsErrorAndLeavesNoObject) {
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  const auto is_s3_provider =
      fs_config.cloud_provider == kCloudProviderAWS || fs_config.cloud_provider == kCloudProviderAliyun ||
      fs_config.cloud_provider == kCloudProviderTencent || fs_config.cloud_provider == kCloudProviderHuawei;
  if (fs_config.storage_type != "remote" || !is_s3_provider) {
    GTEST_SKIP() << "Test requires S3-backed remote filesystem.";
  }

  ASSERT_EQ(0, InitFiuOnce());

  const std::string test_path =
      GetTestBasePath("vortex-s3-flush-fiu") + "/flush-fail-v" + std::to_string(GetParam()) + ".vx";
  (void)file_system_->DeleteFile(test_path);

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_path, properties_);
  const auto rows_per_batch = record_batches_.front()->num_rows();
  for (int i = 0; i < 10; ++i) {
    ASSERT_AND_ASSIGN(auto rb, MakeTestData(i * rows_per_batch, rows_per_batch));
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }

  arrow::Status write_status = arrow::Status::OK();
  {
    ScopedFiuFault fault(FIUKEY_S3FS_WRITER_FLUSH_FAIL, /*one_time=*/false);
    ASSERT_EQ(0, fault.enable_result());

    for (int i = 10; i < 20; ++i) {
      ASSERT_AND_ASSIGN(auto rb, MakeTestData(i * rows_per_batch, rows_per_batch));
      write_status = vx_writer.Write(rb);
      if (!write_status.ok()) {
        EXPECT_NE(write_status.ToString().find(FIUKEY_S3FS_WRITER_FLUSH_FAIL), std::string::npos)
            << write_status.ToString();
        break;
      }
    }

    auto close_result = vx_writer.Close();
    ASSERT_FALSE(close_result.ok());
  }

  ASSERT_AND_ASSIGN(auto file_info, file_system_->GetFileInfo(test_path));
  EXPECT_EQ(arrow::fs::FileType::NotFound, file_info.type()) << file_info.ToString();
}
#endif

TEST_P(VortexBasicTest, TestListOfFixedSizeBinary512WriteRead) {
  const int64_t num_rows = 8;
  const int vectors_per_row = 2;
  const int vector_width = 512;

  auto vector_array_field =
      arrow::field("vector_array", arrow::list(arrow::field("item", arrow::fixed_size_binary(vector_width), false)),
                   false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}));
  auto vector_array_schema = arrow::schema({vector_array_field});

  auto value_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_width));
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);
  std::vector<std::string> expected_values;
  expected_values.reserve(num_rows * vectors_per_row);

  for (int64_t row = 0; row < num_rows; ++row) {
    ASSERT_STATUS_OK(list_builder.Append());
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      std::string vector(vector_width, '\0');
      for (int byte_index = 0; byte_index < vector_width; ++byte_index) {
        vector[byte_index] = static_cast<char>((row * vectors_per_row + vector_index + byte_index) & 0xff);
      }
      ASSERT_STATUS_OK(value_builder->Append(vector));
      expected_values.push_back(vector);
    }
  }

  std::shared_ptr<arrow::Array> vector_array;
  ASSERT_STATUS_OK(list_builder.Finish(&vector_array));

  auto rb = arrow::RecordBatch::Make(vector_array_schema, num_rows, {vector_array});
  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_array_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, vector_array_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"vector_array"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(num_rows, read_rb->num_rows());
  ASSERT_EQ(1, read_rb->num_columns());

  auto read_list = std::dynamic_pointer_cast<arrow::ListArray>(read_rb->column(0));
  ASSERT_NE(read_list, nullptr);
  auto read_values = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_list->values());
  ASSERT_NE(read_values, nullptr);
  ASSERT_EQ(vector_width, read_values->byte_width());

  for (int64_t row = 0; row < num_rows; ++row) {
    ASSERT_EQ(vectors_per_row, read_list->value_length(row));
    for (int vector_index = 0; vector_index < vectors_per_row; ++vector_index) {
      auto value_index = read_list->value_offset(row) + vector_index;
      const auto& expected = expected_values[static_cast<size_t>(value_index)];
      ASSERT_EQ(0, std::memcmp(read_values->GetValue(value_index), expected.data(), vector_width));
    }
  }
}

TEST_P(VortexBasicTest, TestSlicedListOfFixedSizeBinaryWriteRead) {
  const int64_t total_rows = 7;
  const int64_t slice_offset = 2;
  const int64_t slice_length = 4;
  const int vectors_per_row = 3;
  const int vector_width = 32;

  ASSERT_AND_ASSIGN(auto full_vector_array, MakeListOfFixedSizeBinaryArray(total_rows, vectors_per_row, vector_width));
  auto sliced_vector_array = full_vector_array->Slice(slice_offset, slice_length);
  ASSERT_GT(sliced_vector_array->offset(), 0);

  auto vector_array_schema = arrow::schema({arrow::field("vector_array", sliced_vector_array->type(), false,
                                                         arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(vector_array_schema, slice_length, {sliced_vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_array_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(slice_length, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, vector_array_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"vector_array"});
  ASSERT_STATUS_OK(vx_reader.open());

  {
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, slice_length, kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(slice_length, read_rb->num_rows());
    auto read_list = std::dynamic_pointer_cast<arrow::ListArray>(read_rb->column(0));
    AssertListOfFixedSizeBinaryArray(read_list, slice_offset, slice_length, vectors_per_row, vector_width);
  }

  {
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(1, slice_length, kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(slice_length - 1, read_rb->num_rows());
    auto read_list = std::dynamic_pointer_cast<arrow::ListArray>(read_rb->column(0));
    AssertListOfFixedSizeBinaryArray(read_list, slice_offset + 1, slice_length - 1, vectors_per_row, vector_width);
  }
}

TEST_P(VortexBasicTest, TestLargeListOfFixedSizeBinaryWriteRead) {
  const int64_t num_rows = 6;
  const int vectors_per_row = 3;
  const int vector_width = 32;

  ASSERT_AND_ASSIGN(auto vector_array, MakeLargeListOfFixedSizeBinaryArray(num_rows, vectors_per_row, vector_width));
  auto vector_schema = arrow::schema({arrow::field("large_vector_array", vector_array->type(), false,
                                                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(vector_schema, num_rows, {vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, vector_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"large_vector_array"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(num_rows, read_rb->num_rows());
  auto read_list = std::dynamic_pointer_cast<arrow::LargeListArray>(read_rb->column(0));
  AssertLargeListOfFixedSizeBinaryArray(read_list, 0, num_rows, vectors_per_row, vector_width);
}

TEST_P(VortexBasicTest, TestFixedSizeListOfFixedSizeBinaryWriteRead) {
  const int64_t num_rows = 6;
  const int vectors_per_row = 3;
  const int vector_width = 32;

  ASSERT_AND_ASSIGN(auto vector_array,
                    MakeFixedSizeListOfFixedSizeBinaryArray(num_rows, vectors_per_row, vector_width));
  auto vector_schema = arrow::schema({arrow::field("fixed_vector_array", vector_array->type(), false,
                                                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(vector_schema, num_rows, {vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, vector_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"fixed_vector_array"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(num_rows, read_rb->num_rows());
  auto read_list = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(read_rb->column(0));
  AssertFixedSizeListOfFixedSizeBinaryArray(read_list, 0, num_rows, vectors_per_row, vector_width);
}

TEST_P(VortexBasicTest, TestSlicedFixedSizeBinaryWriteRead) {
  const int64_t total_rows = 7;
  const int64_t slice_offset = 2;
  const int64_t slice_length = 4;
  const int vector_width = 32;

  ASSERT_AND_ASSIGN(auto full_vector_array, MakeFixedSizeBinaryArray(total_rows, vector_width));
  auto sliced_vector_array = full_vector_array->Slice(slice_offset, slice_length);
  ASSERT_GT(sliced_vector_array->offset(), 0);

  auto vector_schema = arrow::schema({arrow::field("embedding", sliced_vector_array->type(), false,
                                                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(vector_schema, slice_length, {sliced_vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, vector_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(slice_length, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, vector_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"embedding"});
  ASSERT_STATUS_OK(vx_reader.open());

  {
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, slice_length, kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(slice_length, read_rb->num_rows());
    auto read_array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_rb->column(0));
    AssertFixedSizeBinaryArray(read_array, slice_offset, slice_length, vector_width);
  }

  {
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(1, slice_length, kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(slice_length - 1, read_rb->num_rows());
    auto read_array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_rb->column(0));
    AssertFixedSizeBinaryArray(read_array, slice_offset + 1, slice_length - 1, vector_width);
  }
}

TEST_P(VortexBasicTest, TestNestedFixedSizeBinaryWriteRead) {
  const int64_t num_rows = 6;
  const int vector_width = 16;
  const int groups_per_row = 2;
  const int vectors_per_group = 2;

  ASSERT_AND_ASSIGN(auto struct_array, MakeStructOfFixedSizeBinaryArray(num_rows, vector_width));
  ASSERT_AND_ASSIGN(auto nullable_array, MakeNullableFixedSizeBinaryArray(num_rows, vector_width));
  ASSERT_AND_ASSIGN(auto nested_list_array,
                    MakeNestedListOfFixedSizeBinaryArray(num_rows, groups_per_row, vectors_per_group, vector_width));

  auto nested_schema = arrow::schema({
      arrow::field("entity", struct_array->type(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("nullable_embedding", nullable_array->type(), true,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
      arrow::field("nested_vectors", nested_list_array->type(), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
  });

  auto rb = arrow::RecordBatch::Make(nested_schema, num_rows, {struct_array, nullable_array, nested_list_array});
  auto vx_writer = vortex::VortexFileWriter(file_system_, nested_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto vx_reader =
      vortex::VortexFormatReader(file_system_, nested_schema, test_file_name_, properties_,
                                 std::vector<std::string>{"entity", "nullable_embedding", "nested_vectors"});
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto read_rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(num_rows, read_rb->num_rows());
  ASSERT_EQ(3, read_rb->num_columns());

  auto read_struct = std::dynamic_pointer_cast<arrow::StructArray>(read_rb->column(0));
  ASSERT_NE(read_struct, nullptr);
  auto read_struct_fsb = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_struct->field(0));
  ASSERT_NE(read_struct_fsb, nullptr);
  auto read_struct_other = std::dynamic_pointer_cast<arrow::Int32Array>(read_struct->field(1));
  ASSERT_NE(read_struct_other, nullptr);
  for (int64_t row = 0; row < num_rows; ++row) {
    AssertFixedSizeBinaryValue(read_struct_fsb, row, MakeFixedSizeBinaryValue(row, 1, 0, vector_width));
    ASSERT_EQ(static_cast<int32_t>(row * 13), read_struct_other->Value(row));
  }

  auto read_nullable = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_rb->column(1));
  ASSERT_NE(read_nullable, nullptr);
  ASSERT_EQ(2, read_nullable->null_count());
  for (int64_t row = 0; row < num_rows; ++row) {
    if (row == 1 || row == 4) {
      ASSERT_TRUE(read_nullable->IsNull(row));
    } else {
      AssertFixedSizeBinaryValue(read_nullable, row, MakeFixedSizeBinaryValue(row, 2, 0, vector_width));
    }
  }

  auto read_outer_list = std::dynamic_pointer_cast<arrow::ListArray>(read_rb->column(2));
  ASSERT_NE(read_outer_list, nullptr);
  ASSERT_EQ(num_rows, read_outer_list->length());
  for (int64_t row = 0; row < num_rows; ++row) {
    auto read_inner_list = std::dynamic_pointer_cast<arrow::ListArray>(read_outer_list->value_slice(row));
    ASSERT_NE(read_inner_list, nullptr);
    ASSERT_EQ(groups_per_row, read_inner_list->length());
    for (int group = 0; group < groups_per_row; ++group) {
      auto read_vectors = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(read_inner_list->value_slice(group));
      ASSERT_NE(read_vectors, nullptr);
      ASSERT_EQ(vectors_per_group, read_vectors->length());
      for (int vector_index = 0; vector_index < vectors_per_group; ++vector_index) {
        AssertFixedSizeBinaryValue(read_vectors, vector_index,
                                   MakeFixedSizeBinaryValue(row, group, vector_index, vector_width));
      }
    }
  }
}

TEST_P(VortexBasicTest, TestDictionaryOfFixedSizeBinaryWriteFails) {
  const std::vector<int32_t> indices = {0, 2, 1, -1, 2, 0, 1};
  const int64_t num_rows = static_cast<int64_t>(indices.size());
  const int dictionary_size = 3;
  const int vector_width = 24;

  ASSERT_AND_ASSIGN(auto dictionary_array,
                    MakeDictionaryOfFixedSizeBinaryArray(indices, dictionary_size, vector_width));

  auto dictionary_schema = arrow::schema({arrow::field("dictionary_embedding", dictionary_array->type(), true,
                                                       arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(dictionary_schema, num_rows, {dictionary_array});
  auto vx_writer = vortex::VortexFileWriter(file_system_, dictionary_schema, test_file_name_, properties_);

  auto status = vx_writer.Write(rb);
  if (status.ok()) {
    status = vx_writer.Flush();
  }

  ASSERT_FALSE(status.ok());
  const std::string message = status.ToString();
  EXPECT_NE(message.find("Dictionary"), std::string::npos) << message;
  EXPECT_NE(message.find("FixedSizeBinary"), std::string::npos) << message;
  EXPECT_NE(message.find("not supported"), std::string::npos) << message;

  auto close_result = vx_writer.Close();
  ASSERT_FALSE(close_result.ok());
  EXPECT_TRUE(close_result.status().IsInvalid()) << close_result.status().ToString();
}

TEST_P(VortexBasicTest, TestFixedSizeListWidthMismatchReadFails) {
  const int64_t num_rows = 4;
  const int written_width = 4;
  const int read_width = 3;

  ASSERT_AND_ASSIGN(auto vector_array, MakeFixedSizeListUInt8Array(num_rows, written_width));

  auto write_schema = arrow::schema({arrow::field("embedding", vector_array->type(), false,
                                                  arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(write_schema, num_rows, {vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, write_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto read_schema = arrow::schema({arrow::field("embedding", arrow::fixed_size_binary(read_width), false,
                                                 arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto vx_reader = vortex::VortexFormatReader(file_system_, read_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"embedding"});
  ASSERT_STATUS_OK(vx_reader.open());

  auto read_result = vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow);
  ASSERT_FALSE(read_result.ok());
}

TEST_P(VortexBasicTest, TestFixedSizeListChildNullReadFails) {
  const int64_t num_rows = 4;
  const int vector_width = 4;

  ASSERT_AND_ASSIGN(auto vector_array, MakeFixedSizeListUInt8ArrayWithChildNull(num_rows, vector_width));
  ASSERT_EQ(vector_array->type()->id(), arrow::Type::FIXED_SIZE_LIST);
  ASSERT_TRUE(vector_array->type()->field(0)->nullable());

  auto write_schema = arrow::schema({arrow::field("embedding", vector_array->type(), false,
                                                  arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto rb = arrow::RecordBatch::Make(write_schema, num_rows, {vector_array});

  auto vx_writer = vortex::VortexFileWriter(file_system_, write_schema, test_file_name_, properties_);
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(num_rows, cgfile.end_index);

  auto read_schema = arrow::schema({arrow::field("embedding", arrow::fixed_size_binary(vector_width), false,
                                                 arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});
  auto vx_reader = vortex::VortexFormatReader(file_system_, read_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"embedding"});
  ASSERT_STATUS_OK(vx_reader.open());

  auto read_result = vx_reader.blocking_read(0, num_rows, kSmallCoalescingWindow);
  // FixedSizeBinary can only represent row-level nullability, so a FixedSizeList<UInt8>
  // with byte-level child nulls must fail instead of silently dropping those nulls.
  ASSERT_FALSE(read_result.ok());
}

TEST_P(VortexBasicTest, TestBasicRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));

  ASSERT_EQ(recordBatchsRows(), rb->num_rows());
  ASSERT_EQ(3, rb->num_columns());
  ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
  ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
  ASSERT_EQ(arrow::Type::DOUBLE, rb->column(2)->type_id());

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(rb->column(2));

  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
    ASSERT_DOUBLE_EQ(value_array->Value(i), static_cast<double>(i) * 1.5);
  }
}

TEST_P(VortexBasicTest, TestEmptyWriteRead) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  ASSERT_AND_ASSIGN(auto empty_rb, MakeTestData(0, 0));
  ASSERT_TRUE(vx_writer.Write(empty_rb).ok());

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(0, cgfile.end_index);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_EQ(0, vx_reader.rows());

  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, vx_reader.rows(), kSmallCoalescingWindow));
  ASSERT_EQ(0, chunked_array->num_chunks());
}

TEST_P(VortexBasicTest, TestReaderProjection) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  // all projection
  {
    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());
    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::DOUBLE, rb->column(2)->type_id());
  }

  // projection with different order
  {
    auto projection_schema = arrow::schema({
        schema_->field(2),
        schema_->field(1),
        schema_->field(0),
    });

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"value", "name", "id"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    ASSERT_EQ(arrow::Type::DOUBLE, rb->column(0)->type_id());
    ASSERT_EQ(arrow::Type::STRING, rb->column(1)->type_id());
    ASSERT_EQ(arrow::Type::INT64, rb->column(2)->type_id());
  }

  // single projection
  {
    auto projection_schema = arrow::schema({schema_->field(0)});

    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"id"});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
  }

  // empty projection
  {
    auto projection_schema = arrow::schema({});
    auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                                std::vector<std::string>{});
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_EQ(recordBatchsRows(), vx_reader.rows());

    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());
  }
}

TEST_P(VortexBasicTest, CachedCreateReaderKeepsProjectionForEmptyPredicateResult) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }

  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  KeyRetriever key_retriever;
  ASSERT_AND_ASSIGN(auto metadata,
                    vortex::VortexFormatReader::MetaTrait::load_metadata(cgfile, properties_, key_retriever));

  auto id_schema = arrow::schema({schema_->field(0)});
  ASSERT_AND_ASSIGN(auto filtered_reader,
                    vortex::VortexFormatReader::MetaTrait::create_from_metadata(
                        metadata, cgfile, properties_, id_schema, std::vector<std::string>{"id"}, "id > 1000000"));
  ASSERT_AND_ASSIGN(auto empty_batch, filtered_reader->get_chunk(0));
  ASSERT_EQ(0, empty_batch->num_rows());
  ASSERT_EQ(1, empty_batch->num_columns());
  ASSERT_EQ("id", empty_batch->schema()->field(0)->name());

  auto value_schema = arrow::schema({schema_->field(2)});
  ASSERT_AND_ASSIGN(auto unfiltered_reader,
                    vortex::VortexFormatReader::MetaTrait::create_from_metadata(
                        metadata, cgfile, properties_, value_schema, std::vector<std::string>{"value"}, ""));
  ASSERT_AND_ASSIGN(auto value_batch, unfiltered_reader->get_chunk(0));
  ASSERT_GT(value_batch->num_rows(), 0);
  ASSERT_EQ(1, value_batch->num_columns());
  ASSERT_EQ("value", value_batch->schema()->field(0)->name());
}

TEST_P(VortexBasicTest, CachedMetadataFallsBackWhenFooterSizeAccountingFails) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());

  const auto footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(footer_size, 0u);
  cgfile.Set(api::kPropertyFooterSize, uint64_t{0});

  KeyRetriever key_retriever;
  ScopedFiuFault fault(FIUKEY_VORTEX_METADATA_CACHE_SIZE_FAIL, /*one_time=*/true);
  ASSERT_AND_ASSIGN(auto metadata,
                    vortex::VortexFormatReader::MetaTrait::load_metadata(cgfile, properties_, key_retriever));
  EXPECT_EQ(metadata->cache_size, 0);
}

TEST_P(VortexBasicTest, TestBasicTake) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto take_verify = [&](vortex::VortexFormatReader& vx_reader, const std::vector<int64_t>& row_indices,
                         int64_t expect_rows) {
    ASSERT_AND_ASSIGN(auto table, vx_reader.take(row_indices));
    ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());

    ASSERT_EQ(expect_rows, rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (size_t i = 0; i < rb->num_rows(); i++) {
      ASSERT_EQ(id_array->Value(i), row_indices[i]);
    }
  };

  auto projection_schema = arrow::schema({schema_->field(0)});

  auto vx_reader = vortex::VortexFormatReader(file_system_, projection_schema, test_file_name_, properties_,
                                              std::vector<std::string>{"id"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto one_random_row, GenerateSortedUniqueArray(1, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto random_rows, GenerateSortedUniqueArray(100, recordBatchsRows()));
  std::vector<int64_t> all_rows(recordBatchsRows());
  std::iota(all_rows.begin(), all_rows.end(), 0);

  // take single row
  take_verify(vx_reader, one_random_row, 1);
  // 100 random rows
  take_verify(vx_reader, random_rows, 100);
  // boundary Testing
  take_verify(vx_reader, {0, recordBatchsRows() - 1}, 2);
  // take all range
  take_verify(vx_reader, all_rows, recordBatchsRows());
  // Note: vortex 0.56+ does not gracefully handle out-of-range indices (panics instead of returning error),
  // so we removed the out-of-range index tests.
}

TEST_P(VortexBasicTest, FooterSizeMatchesActualFile) {
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);

  // Zero-row output intentionally covers footer-size computation with no data segments.
  auto zero_row_batch = record_batches_.front()->Slice(0, 0);
  ASSERT_TRUE(vx_writer.Write(zero_row_batch).ok());

  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(vx_footer_size, 0u);
  ASSERT_GT(vx_file_size, vx_footer_size);

  // Verify file_size matches actual file
  ASSERT_AND_ASSIGN(auto file, file_system_->OpenInputFile(test_file_name_));
  ASSERT_AND_ASSIGN(auto actual_file_size, file->GetSize());
  EXPECT_EQ(vx_file_size, static_cast<uint64_t>(actual_file_size));

  // Read EOF (last 8 bytes): [version 2B][postscript_len 2B][magic "VTXF" 4B]
  ASSERT_AND_ASSIGN(auto eof_buf, file->ReadAt(actual_file_size - 8, 8));
  const uint8_t* eof = eof_buf->data();

  // Verify magic
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(eof + 4), 4), "VTXF");

  constexpr uint64_t kVortexHeaderSize = 4;
  const auto eof_size = VortexEofSize();
  ASSERT_GE(static_cast<uint64_t>(actual_file_size), kVortexHeaderSize + eof_size);
  EXPECT_EQ(vx_footer_size, static_cast<uint64_t>(actual_file_size) - kVortexHeaderSize - eof_size)
      << "zero-row Vortex output should contain only the header before the footer body";

  auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
  auto vxfile =
      VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size, vx_footer_size);
  auto footer_range = vxfile.FooterByteRange(vx_file_size);
  ASSERT_EQ(footer_range.size(), 2u);
  ASSERT_LE(footer_range[0], vx_file_size);
  ASSERT_LE(footer_range[1], vx_file_size - footer_range[0]);

  ASSERT_GT(footer_range[1], eof_size);
  const auto actual_footer_body_size = footer_range[1] - eof_size;

  EXPECT_EQ(vx_footer_size, actual_footer_body_size)
      << "cached footer_size should match the actual Vortex footer body; normal open adds EOF_SIZE";
}

TEST_P(VortexBasicTest, FooterSizeNotMatch) {
  // Write a vortex file
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  auto vx_footer_size2 = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size2 = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(vx_footer_size2, 0u);
  ASSERT_GT(vx_file_size2, vx_footer_size2);

  auto verify_read = [&](uint64_t footer_size) {
    auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
    auto vxfile =
        VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size2, footer_size);
    ASSERT_EQ(vxfile.RowCount(), static_cast<uint64_t>(recordBatchsRows()));

    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                                vx_file_size2, footer_size);
    ASSERT_STATUS_OK(vx_reader.open());
    ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows(), kSmallCoalescingWindow));
    ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
    ASSERT_EQ(recordBatchsRows(), rb->num_rows());
    ASSERT_EQ(3, rb->num_columns());

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (int i = 0; i < id_array->length(); ++i) {
      ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
    }
  };

  verify_read(1);

  // Case 2: footer_size too large (= file_size, reads entire file as initial read).
  // Vortex clamps to min(initial_read_size, file_size). Extra bytes get cached as segments.
  verify_read(vx_file_size2);
}

}  // namespace milvus_storage
