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
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_dict.h>
#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/key_value_metadata.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

class VortexV2RowGroupSizeTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (IsCloudEnv()) {
      GTEST_SKIP() << "Vortex row-group size regression matrix is local-fs only.";
    }

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, "2");
    ASSERT_AND_ASSIGN(file_system_, GetFileSystem(properties_));
  }

  void TearDown() override {
    if (file_system_ != nullptr) {
      for (const auto& path : generated_paths_) {
        (void)file_system_->DeleteFile(path);
      }
    }

    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      for (const auto& path : generated_paths_) {
        boost::filesystem::remove_all(path);
      }
    }
  }

  std::string RegisterTestPath(const std::string& name) {
    auto path = "test-v2-rg-size-" + name + ".vx";
    generated_paths_.push_back(path);
    if (file_system_ != nullptr) {
      (void)file_system_->DeleteFile(path);
    }
    boost::filesystem::remove_all(path);
    return path;
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  api::Properties properties_;
  std::vector<std::string> generated_paths_;
};

namespace {

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1) / divisor; }

constexpr uint64_t kMaxRowsPerRowGroup = 8192;

uint64_t MinExpectedRowGroups(uint64_t logical_bytes, uint64_t row_group_max_size, int64_t rows) {
  auto byte_groups = CeilDiv(logical_bytes, row_group_max_size);
  auto row_groups = CeilDiv(static_cast<uint64_t>(rows), kMaxRowsPerRowGroup);
  return std::max(byte_groups, row_groups);
}

size_t MaxExpectedRowGroups(uint64_t logical_bytes, uint64_t row_group_max_size, int64_t rows) {
  return std::max<uint64_t>(4, MinExpectedRowGroups(logical_bytes, row_group_max_size, rows) + 2);
}

uint64_t BoolLogicalBytes(int64_t rows) { return CeilDiv(static_cast<uint64_t>(rows), 8); }

uint64_t ValidityLogicalBytes(int64_t rows) { return CeilDiv(static_cast<uint64_t>(rows), 8); }

uint64_t AddValidityBytes(uint64_t bytes, int64_t rows, bool nullable) {
  return nullable ? bytes + ValidityLogicalBytes(rows) : bytes;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> BuildValidityBitmap(int64_t rows, int64_t* null_count) {
  ARROW_ASSIGN_OR_RAISE(auto bitmap, arrow::AllocateBitmap(rows));
  *null_count = 0;
  auto* bits = bitmap->mutable_data();
  for (int64_t i = 0; i < rows; ++i) {
    const bool is_valid = i % 7 != 3;
    arrow::bit_util::SetBitTo(bits, i, is_valid);
    if (!is_valid) {
      ++(*null_count);
    }
  }
  return bitmap;
}

arrow::Result<std::shared_ptr<arrow::Array>> MakeNullableArray(const std::shared_ptr<arrow::Array>& array) {
  if (array->type_id() == arrow::Type::NA) {
    return array;
  }

  int64_t null_count = 0;
  ARROW_ASSIGN_OR_RAISE(auto bitmap, BuildValidityBitmap(array->length(), &null_count));
  auto data = array->data()->Copy();
  if (data->buffers.empty()) {
    data->buffers.resize(1);
  }
  data->buffers[0] = std::move(bitmap);
  data->SetNullCount(null_count);
  return arrow::MakeArray(data);
}

template <typename Builder, typename ValueFn>
arrow::Result<std::shared_ptr<arrow::Array>> BuildNumericArray(const std::shared_ptr<arrow::DataType>& type,
                                                               int64_t rows,
                                                               ValueFn value_fn) {
  Builder builder(type, arrow::default_memory_pool());
  for (int64_t i = 0; i < rows; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(value_fn(i)));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildBoolArray(int64_t rows) {
  arrow::BooleanBuilder builder;
  for (int64_t i = 0; i < rows; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append((i & 1) == 0));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

template <typename Builder>
arrow::Result<std::shared_ptr<arrow::Array>> BuildBytesArray(int64_t rows, size_t value_len) {
  Builder builder;
  std::string value(value_len, 'x');
  for (int64_t i = 0; i < rows; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(value));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildDecimal128Array(int64_t rows) {
  arrow::Decimal128Builder builder(arrow::decimal128(20, 2));
  for (int64_t i = 0; i < rows; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(arrow::Decimal128(i)));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildDecimal256Array(int64_t rows) {
  arrow::Decimal256Builder builder(arrow::decimal256(40, 2));
  for (int64_t i = 0; i < rows; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(arrow::Decimal256(arrow::Decimal128(i))));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildFixedSizeBinaryArray(int64_t rows, int32_t byte_width) {
  arrow::FixedSizeBinaryBuilder builder(arrow::fixed_size_binary(byte_width));
  std::string value(byte_width, 'x');
  for (int64_t i = 0; i < rows; ++i) {
    value[0] = static_cast<char>('a' + (i % 26));
    ARROW_RETURN_NOT_OK(builder.Append(value));
  }
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

template <typename OffsetBuilder, typename OffsetType>
arrow::Result<std::shared_ptr<arrow::Array>> BuildOffsets(int64_t outer_rows,
                                                          int64_t values_per_row,
                                                          bool include_final_offset) {
  auto rows = include_final_offset ? outer_rows + 1 : outer_rows;
  return BuildNumericArray<OffsetBuilder>(
      std::is_same<OffsetType, int64_t>::value ? arrow::int64() : arrow::int32(), rows,
      [values_per_row](int64_t i) { return static_cast<OffsetType>(i * values_per_row); });
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildListArray(const std::shared_ptr<arrow::Array>& values,
                                                            int64_t outer_rows,
                                                            int64_t values_per_row,
                                                            bool values_nullable) {
  ARROW_ASSIGN_OR_RAISE(auto offsets, (BuildOffsets<arrow::Int32Builder, int32_t>(outer_rows, values_per_row, true)));
  auto type = arrow::list(arrow::field("item", values->type(), values_nullable));
  ARROW_ASSIGN_OR_RAISE(auto list, arrow::ListArray::FromArrays(type, *offsets, *values));
  return std::static_pointer_cast<arrow::Array>(list);
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildLargeListArray(const std::shared_ptr<arrow::Array>& values,
                                                                 int64_t outer_rows,
                                                                 int64_t values_per_row,
                                                                 bool values_nullable) {
  ARROW_ASSIGN_OR_RAISE(auto offsets, (BuildOffsets<arrow::Int64Builder, int64_t>(outer_rows, values_per_row, true)));
  auto type = arrow::large_list(arrow::field("item", values->type(), values_nullable));
  ARROW_ASSIGN_OR_RAISE(auto list, arrow::LargeListArray::FromArrays(type, *offsets, *values));
  return std::static_pointer_cast<arrow::Array>(list);
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildFixedSizeListArray(const std::shared_ptr<arrow::Array>& values,
                                                                     int32_t values_per_row,
                                                                     bool values_nullable) {
  auto type = arrow::fixed_size_list(arrow::field("item", values->type(), values_nullable), values_per_row);
  return arrow::FixedSizeListArray::FromArrays(values, type);
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildStructArray(const std::shared_ptr<arrow::Array>& values,
                                                              bool values_nullable) {
  ARROW_ASSIGN_OR_RAISE(auto struct_array,
                        arrow::StructArray::Make({values}, {arrow::field("value", values->type(), values_nullable)}));
  return std::static_pointer_cast<arrow::Array>(struct_array);
}

}  // namespace

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupCountUsesLogicalSizeForSlicedUtf8Batches) {
  const int64_t total_rows = 4096;
  const int64_t rows_per_slice = 16;
  const size_t str_len = 4096;
  const uint64_t row_group_max_size = 1024 * 1024;

  auto str_schema = arrow::schema(
      {arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  arrow::StringBuilder builder;
  std::string value(str_len, 'x');
  for (int64_t i = 0; i < total_rows; ++i) {
    ASSERT_STATUS_OK(builder.Append(value));
  }

  std::shared_ptr<arrow::Array> full_array;
  ASSERT_STATUS_OK(builder.Finish(&full_array));

  auto test_path = RegisterTestPath("sliced-utf8");
  auto vx_writer = vortex::VortexFileWriter(file_system_, str_schema, test_path, properties_);
  for (int64_t offset = 0; offset < total_rows; offset += rows_per_slice) {
    auto slice_len = std::min<int64_t>(rows_per_slice, total_rows - offset);
    auto rb = arrow::RecordBatch::Make(str_schema, slice_len, {full_array->Slice(offset, slice_len)});
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader = vortex::VortexFormatReader(file_system_, str_schema, test_path, properties_,
                                              std::vector<std::string>{"str"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes = static_cast<uint64_t>(total_rows) * (str_len + 16);
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);
  ASSERT_LT(rg_infos.size(), static_cast<size_t>(total_rows / rows_per_slice / 2));

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count) << "rg[" << i << "] should be contiguous";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupCountUsesLogicalSizeForSlicedStringViewBatches) {
  const int64_t total_rows = 4096;
  const int64_t rows_per_slice = 16;
  const size_t str_len = 4096;
  const uint64_t row_group_max_size = 1024 * 1024;

  auto schema = arrow::schema(
      {arrow::field("str", arrow::utf8_view(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  arrow::StringViewBuilder builder;
  std::string value(str_len, 'x');
  for (int64_t i = 0; i < total_rows; ++i) {
    ASSERT_STATUS_OK(builder.Append(value));
  }

  std::shared_ptr<arrow::Array> full_array;
  ASSERT_STATUS_OK(builder.Finish(&full_array));

  auto test_path = RegisterTestPath("sliced-utf8-view");
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
  for (int64_t offset = 0; offset < total_rows; offset += rows_per_slice) {
    auto slice_len = std::min<int64_t>(rows_per_slice, total_rows - offset);
    auto rb = arrow::RecordBatch::Make(schema, slice_len, {full_array->Slice(offset, slice_len)});
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                              std::vector<std::string>{"str"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes = static_cast<uint64_t>(total_rows) * (str_len + 16);
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);
  ASSERT_LT(rg_infos.size(), static_cast<size_t>(total_rows / rows_per_slice / 2));

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count);
    ASSERT_GT(rg_infos[i].memory_size, 0u);
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupCountUsesAverageSampleForSkewedStringViewBatches) {
  const int64_t total_rows = 4096;
  const int64_t rows_per_slice = 16;
  const size_t small_len = 8;
  const size_t large_len = 1024 * 1024;
  const uint64_t row_group_max_size = 1024 * 1024;

  auto schema = arrow::schema(
      {arrow::field("str", arrow::utf8_view(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  arrow::StringViewBuilder builder;
  std::string small_value(small_len, 'x');
  std::string large_value(large_len, 'y');
  for (int64_t i = 0; i < total_rows; ++i) {
    ASSERT_STATUS_OK(builder.Append(i == 0 ? large_value : small_value));
  }

  std::shared_ptr<arrow::Array> full_array;
  ASSERT_STATUS_OK(builder.Finish(&full_array));

  auto test_path = RegisterTestPath("skewed-utf8-view");
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
  for (int64_t offset = 0; offset < total_rows; offset += rows_per_slice) {
    auto slice_len = std::min<int64_t>(rows_per_slice, total_rows - offset);
    auto rb = arrow::RecordBatch::Make(schema, slice_len, {full_array->Slice(offset, slice_len)});
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                              std::vector<std::string>{"str"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes = static_cast<uint64_t>(total_rows) * (small_len + 16) + large_len - small_len;
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);
  ASSERT_LT(rg_infos.size(), static_cast<size_t>(total_rows / rows_per_slice / 2));

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count);
    ASSERT_GT(rg_infos[i].memory_size, 0u);
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupReestimatesSkewedSingleBatchRemainder) {
  const int64_t total_rows = 4096;
  const size_t small_len = 8;
  const size_t large_len = 1024 * 1024;
  const uint64_t row_group_max_size = 1024 * 1024;

  auto schema = arrow::schema(
      {arrow::field("str", arrow::utf8_view(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  arrow::StringViewBuilder builder;
  std::string small_value(small_len, 'x');
  std::string large_value(large_len, 'y');
  for (int64_t i = 0; i < total_rows; ++i) {
    ASSERT_STATUS_OK(builder.Append(i == 0 ? large_value : small_value));
  }

  std::shared_ptr<arrow::Array> array;
  ASSERT_STATUS_OK(builder.Finish(&array));

  auto test_path = RegisterTestPath("skewed-single-batch-remainder");
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
  auto rb = arrow::RecordBatch::Make(schema, total_rows, {array});
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                              std::vector<std::string>{"str"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes = static_cast<uint64_t>(total_rows) * (small_len + 16) + large_len - small_len;
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count) << "rg[" << i << "] should be contiguous";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupSplitsSingleLargeBatch) {
  const int64_t total_rows = 4096;
  const size_t str_len = 2048;
  const uint64_t row_group_max_size = 512 * 1024;

  auto schema = arrow::schema(
      {arrow::field("str", arrow::utf8_view(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  ASSERT_AND_ASSIGN(auto array, BuildBytesArray<arrow::StringViewBuilder>(total_rows, str_len));

  auto test_path = RegisterTestPath("single-large-batch");
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
  auto rb = arrow::RecordBatch::Make(schema, total_rows, {array});
  ASSERT_STATUS_OK(vx_writer.Write(rb));
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                              std::vector<std::string>{"str"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes = static_cast<uint64_t>(total_rows) * (str_len + 16);
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count) << "rg[" << i << "] should be contiguous";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupSizes) {
  const uint64_t row_group_max_size = 128 * 1024;
  const uint64_t target_logical_bytes = 256 * 1024;
  const int64_t nested_outer_rows = 2048;
  const size_t bytes_value_len = 256;

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  struct LeafFactory {
    std::string name;
    std::function<arrow::Result<std::shared_ptr<arrow::Array>>(int64_t)> build;
    std::function<uint64_t(int64_t)> logical_bytes;
    int64_t direct_rows;
    int64_t nested_values_per_row;
    bool supports_nested;
  };

  auto verify_sliced_case = [&](const std::string& name, const std::shared_ptr<arrow::Array>& array,
                                uint64_t logical_bytes, int64_t rows_per_slice, bool nullable) {
    SCOPED_TRACE(name);
    const auto field_name = "value";
    auto case_schema = arrow::schema({arrow::field(field_name, array->type(), nullable)});
    auto test_path = RegisterTestPath(name);

    auto vx_writer = vortex::VortexFileWriter(file_system_, case_schema, test_path, properties_);
    for (int64_t offset = 0; offset < array->length(); offset += rows_per_slice) {
      auto slice_len = std::min<int64_t>(rows_per_slice, array->length() - offset);
      auto rb = arrow::RecordBatch::Make(case_schema, slice_len, {array->Slice(offset, slice_len)});
      ASSERT_STATUS_OK(vx_writer.Write(rb));
    }
    ASSERT_STATUS_OK(vx_writer.Flush());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(array->length(), cgfile.end_index);

    auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
    auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
    auto vx_reader = vortex::VortexFormatReader(file_system_, case_schema, test_path, properties_,
                                                std::vector<std::string>{field_name}, vx_file_size, vx_footer_size);
    ASSERT_STATUS_OK(vx_reader.open());

    ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
    const auto max_expected_row_groups = MaxExpectedRowGroups(logical_bytes, row_group_max_size, array->length());
    ASSERT_LE(rg_infos.size(), max_expected_row_groups)
        << name << " should split by logical data size, not retained slice backing buffers";
    if (logical_bytes > row_group_max_size * 3 / 2) {
      ASSERT_GT(rg_infos.size(), 1u) << name << " should split data larger than the row-group target";
    }

    const auto slice_count = static_cast<size_t>((array->length() + rows_per_slice - 1) / rows_per_slice);
    const auto slice_guard = std::max<size_t>(2, slice_count / 2);
    const auto min_expected_row_groups = MinExpectedRowGroups(logical_bytes, row_group_max_size, array->length());
    if (slice_count >= 8 && logical_bytes > 0 && min_expected_row_groups + 2 < slice_guard) {
      ASSERT_LT(rg_infos.size(), slice_guard) << name << " should not produce one row group per input slice";
    }

    uint64_t row_count = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_EQ(rg_infos[i].start_offset, row_count) << name << " rg[" << i << "] should be contiguous";
      if (logical_bytes > 0) {
        ASSERT_GT(rg_infos[i].memory_size, 0u) << name << " rg[" << i << "] memory_size should be > 0";
      }
      row_count = rg_infos[i].end_offset;
    }
    ASSERT_EQ(row_count, static_cast<uint64_t>(array->length()));

    ASSERT_STATUS_OK(file_system_->DeleteFile(test_path));
  };

  auto rows_for_width = [&](uint64_t width) { return static_cast<int64_t>(CeilDiv(target_logical_bytes, width)); };
  auto nested_items_for_width = [&](uint64_t width) {
    return static_cast<int64_t>(std::max<uint64_t>(1, CeilDiv(target_logical_bytes, width * nested_outer_rows)));
  };
  auto direct_rows_per_slice = [](int64_t rows) { return std::max<int64_t>(1, rows / 32); };

  std::vector<LeafFactory> leaves;
  auto add_leaf = [&](std::string name, auto build, auto logical_bytes, int64_t direct_rows,
                      int64_t nested_values_per_row, bool supports_nested = true) {
    leaves.push_back(
        LeafFactory{std::move(name), build, logical_bytes, direct_rows, nested_values_per_row, supports_nested});
  };

  add_leaf(
      "null",
      [](int64_t rows) -> arrow::Result<std::shared_ptr<arrow::Array>> {
        return std::static_pointer_cast<arrow::Array>(std::make_shared<arrow::NullArray>(rows));
      },
      [](int64_t) { return 0; }, 32768, 1);
  add_leaf(
      "bool", [](int64_t rows) { return BuildBoolArray(rows); }, [](int64_t rows) { return BoolLogicalBytes(rows); },
      static_cast<int64_t>(target_logical_bytes * 8),
      static_cast<int64_t>(target_logical_bytes * 8 / nested_outer_rows));
  add_leaf(
      "uint8",
      [](int64_t rows) {
        return BuildNumericArray<arrow::UInt8Builder>(arrow::uint8(), rows,
                                                      [](int64_t i) { return static_cast<uint8_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows); }, rows_for_width(1), nested_items_for_width(1));
  add_leaf(
      "uint16",
      [](int64_t rows) {
        return BuildNumericArray<arrow::UInt16Builder>(arrow::uint16(), rows,
                                                       [](int64_t i) { return static_cast<uint16_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 2; }, rows_for_width(2), nested_items_for_width(2));
  add_leaf(
      "uint32",
      [](int64_t rows) {
        return BuildNumericArray<arrow::UInt32Builder>(arrow::uint32(), rows,
                                                       [](int64_t i) { return static_cast<uint32_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "uint64",
      [](int64_t rows) {
        return BuildNumericArray<arrow::UInt64Builder>(arrow::uint64(), rows,
                                                       [](int64_t i) { return static_cast<uint64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "int8",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Int8Builder>(arrow::int8(), rows,
                                                     [](int64_t i) { return static_cast<int8_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows); }, rows_for_width(1), nested_items_for_width(1));
  add_leaf(
      "int16",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Int16Builder>(arrow::int16(), rows,
                                                      [](int64_t i) { return static_cast<int16_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 2; }, rows_for_width(2), nested_items_for_width(2));
  add_leaf(
      "int32",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Int32Builder>(arrow::int32(), rows,
                                                      [](int64_t i) { return static_cast<int32_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "int64",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Int64Builder>(arrow::int64(), rows,
                                                      [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "float16",
      [](int64_t rows) {
        return BuildNumericArray<arrow::HalfFloatBuilder>(arrow::float16(), rows,
                                                          [](int64_t i) { return static_cast<uint16_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 2; }, rows_for_width(2), nested_items_for_width(2));
  add_leaf(
      "float32",
      [](int64_t rows) {
        return BuildNumericArray<arrow::FloatBuilder>(arrow::float32(), rows,
                                                      [](int64_t i) { return static_cast<float>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "float64",
      [](int64_t rows) {
        return BuildNumericArray<arrow::DoubleBuilder>(arrow::float64(), rows,
                                                       [](int64_t i) { return static_cast<double>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "utf8", [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::StringBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * bytes_value_len + (rows + 1) * 4; }, 1024,
      1);
  add_leaf(
      "large-utf8",
      [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::LargeStringBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * bytes_value_len + (rows + 1) * 8; }, 1024,
      1);
  add_leaf(
      "binary",
      [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::BinaryBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * bytes_value_len + (rows + 1) * 4; }, 1024,
      1);
  add_leaf(
      "large-binary",
      [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::LargeBinaryBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * bytes_value_len + (rows + 1) * 8; }, 1024,
      1);
  add_leaf(
      "utf8-view",
      [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::StringViewBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * (bytes_value_len + 16); }, 1024, 1);
  add_leaf(
      "binary-view",
      [bytes_value_len](int64_t rows) { return BuildBytesArray<arrow::BinaryViewBuilder>(rows, bytes_value_len); },
      [bytes_value_len](int64_t rows) { return static_cast<uint64_t>(rows) * (bytes_value_len + 16); }, 1024, 1);
  add_leaf(
      "date32",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Date32Builder>(arrow::date32(), rows,
                                                       [](int64_t i) { return static_cast<int32_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "date64",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Date64Builder>(arrow::date64(), rows,
                                                       [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "time32-s",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Time32Builder>(arrow::time32(arrow::TimeUnit::SECOND), rows,
                                                       [](int64_t i) { return static_cast<int32_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "time32-ms",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Time32Builder>(arrow::time32(arrow::TimeUnit::MILLI), rows,
                                                       [](int64_t i) { return static_cast<int32_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 4; }, rows_for_width(4), nested_items_for_width(4));
  add_leaf(
      "time64-us",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Time64Builder>(arrow::time64(arrow::TimeUnit::MICRO), rows,
                                                       [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "time64-ns",
      [](int64_t rows) {
        return BuildNumericArray<arrow::Time64Builder>(arrow::time64(arrow::TimeUnit::NANO), rows,
                                                       [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "timestamp-s",
      [](int64_t rows) {
        return BuildNumericArray<arrow::TimestampBuilder>(arrow::timestamp(arrow::TimeUnit::SECOND), rows,
                                                          [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "timestamp-ms",
      [](int64_t rows) {
        return BuildNumericArray<arrow::TimestampBuilder>(arrow::timestamp(arrow::TimeUnit::MILLI), rows,
                                                          [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "timestamp-us",
      [](int64_t rows) {
        return BuildNumericArray<arrow::TimestampBuilder>(arrow::timestamp(arrow::TimeUnit::MICRO), rows,
                                                          [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "timestamp-ns",
      [](int64_t rows) {
        return BuildNumericArray<arrow::TimestampBuilder>(arrow::timestamp(arrow::TimeUnit::NANO), rows,
                                                          [](int64_t i) { return static_cast<int64_t>(i); });
      },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 8; }, rows_for_width(8), nested_items_for_width(8));
  add_leaf(
      "decimal128", [](int64_t rows) { return BuildDecimal128Array(rows); },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 16; }, rows_for_width(16), nested_items_for_width(16));
  add_leaf(
      "decimal256", [](int64_t rows) { return BuildDecimal256Array(rows); },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 32; }, rows_for_width(32), nested_items_for_width(32));
  add_leaf(
      "fixed-size-binary", [](int64_t rows) { return BuildFixedSizeBinaryArray(rows, 64); },
      [](int64_t rows) { return static_cast<uint64_t>(rows) * 64; }, rows_for_width(64), nested_items_for_width(64),
      false);

  for (const auto& leaf : leaves) {
    ASSERT_AND_ASSIGN(auto array, leaf.build(leaf.direct_rows));
    const bool direct_nullable = array->type_id() == arrow::Type::NA;
    verify_sliced_case(leaf.name, array, leaf.logical_bytes(leaf.direct_rows), direct_rows_per_slice(leaf.direct_rows),
                       direct_nullable);
    if (!direct_nullable) {
      ASSERT_AND_ASSIGN(auto nullable_array, MakeNullableArray(array));
      verify_sliced_case("nullable-" + leaf.name, nullable_array,
                         AddValidityBytes(leaf.logical_bytes(leaf.direct_rows), leaf.direct_rows, true),
                         direct_rows_per_slice(leaf.direct_rows), true);
    }

    if (!leaf.supports_nested) {
      continue;
    }

    auto add_struct_case = [&](bool child_nullable, bool struct_nullable) {
      std::shared_ptr<arrow::Array> struct_values = array;
      uint64_t struct_value_bytes = leaf.logical_bytes(leaf.direct_rows);
      if (child_nullable && array->type_id() != arrow::Type::NA) {
        ASSERT_AND_ASSIGN(auto nullable_values, MakeNullableArray(array));
        struct_values = nullable_values;
        struct_value_bytes = AddValidityBytes(struct_value_bytes, leaf.direct_rows, true);
      }

      ASSERT_AND_ASSIGN(auto struct_array,
                        BuildStructArray(struct_values, child_nullable || array->type_id() == arrow::Type::NA));
      auto logical_bytes = struct_value_bytes;
      if (struct_nullable) {
        ASSERT_AND_ASSIGN(auto nullable_struct_array, MakeNullableArray(struct_array));
        struct_array = nullable_struct_array;
        logical_bytes = AddValidityBytes(logical_bytes, struct_array->length(), true);
      }

      std::string name = struct_nullable ? "nullable-struct-" : "struct-";
      name += leaf.name;
      if (child_nullable && array->type_id() != arrow::Type::NA) {
        name += "-nullable-field";
      }
      verify_sliced_case(name, struct_array, logical_bytes, direct_rows_per_slice(leaf.direct_rows), struct_nullable);
    };

    add_struct_case(false, false);
    add_struct_case(false, true);
    if (array->type_id() != arrow::Type::NA) {
      add_struct_case(true, false);
      add_struct_case(true, true);
    }

    const int64_t values_per_row = leaf.nested_values_per_row;
    const int64_t nested_value_rows = nested_outer_rows * values_per_row;
    auto add_nested_cases = [&](bool child_nullable, bool container_nullable) {
      ASSERT_AND_ASSIGN(auto nested_values, leaf.build(nested_value_rows));
      uint64_t nested_value_bytes = leaf.logical_bytes(nested_value_rows);
      if (child_nullable && nested_values->type_id() != arrow::Type::NA) {
        ASSERT_AND_ASSIGN(auto nullable_nested_values, MakeNullableArray(nested_values));
        nested_values = nullable_nested_values;
        nested_value_bytes = AddValidityBytes(nested_value_bytes, nested_value_rows, true);
      }
      const bool values_nullable = child_nullable || nested_values->type_id() == arrow::Type::NA;

      auto verify_container_case = [&](const std::string& shape, const std::shared_ptr<arrow::Array>& base_array,
                                       uint64_t base_logical_bytes) {
        auto nested_array = base_array;
        auto logical_bytes = base_logical_bytes;
        if (container_nullable) {
          ASSERT_AND_ASSIGN(auto nullable_nested_array, MakeNullableArray(base_array));
          nested_array = nullable_nested_array;
          logical_bytes = AddValidityBytes(logical_bytes, nested_array->length(), true);
        }

        std::string name = container_nullable ? "nullable-" : "";
        name += shape + "-" + leaf.name;
        if (child_nullable && nested_values->type_id() != arrow::Type::NA) {
          name += "-nullable-values";
        }
        verify_sliced_case(name, nested_array, logical_bytes, 128, container_nullable);
      };

      ASSERT_AND_ASSIGN(auto list_array,
                        BuildListArray(nested_values, nested_outer_rows, values_per_row, values_nullable));
      verify_container_case("list", list_array, nested_value_bytes + (nested_outer_rows + 1) * 4);

      ASSERT_AND_ASSIGN(auto large_list_array,
                        BuildLargeListArray(nested_values, nested_outer_rows, values_per_row, values_nullable));
      verify_container_case("large-list", large_list_array, nested_value_bytes + (nested_outer_rows + 1) * 8);

      ASSERT_AND_ASSIGN(auto fixed_size_list_array,
                        BuildFixedSizeListArray(nested_values, static_cast<int32_t>(values_per_row), values_nullable));
      verify_container_case("fixed-size-list", fixed_size_list_array, nested_value_bytes);
    };

    add_nested_cases(false, false);
    add_nested_cases(false, true);
    if (leaf.name != "null") {
      add_nested_cases(true, false);
      add_nested_cases(true, true);
    }
  }

  {
    const int64_t rows = 1024;
    ASSERT_AND_ASSIGN(auto dictionary, BuildBytesArray<arrow::StringBuilder>(16, bytes_value_len));
    auto verify_dictionary_case = [&](const std::string& name, const std::shared_ptr<arrow::Array>& indices) {
      ASSERT_AND_ASSIGN(auto dict_array,
                        arrow::DictionaryArray::FromArrays(arrow::dictionary(indices->type(), dictionary->type()),
                                                           indices, dictionary));
      const auto logical_bytes = static_cast<uint64_t>(rows) * bytes_value_len;
      verify_sliced_case(name, dict_array, logical_bytes, 32, false);

      ASSERT_AND_ASSIGN(auto nullable_dict_array, MakeNullableArray(dict_array));
      verify_sliced_case("nullable-" + name, nullable_dict_array, AddValidityBytes(logical_bytes, rows, true), 32,
                         true);
    };

    ASSERT_AND_ASSIGN(auto uint8_indices, BuildNumericArray<arrow::UInt8Builder>(arrow::uint8(), rows, [](int64_t i) {
                        return static_cast<uint8_t>(i % 16);
                      }));
    verify_dictionary_case("dict-uint8-utf8", uint8_indices);
    ASSERT_AND_ASSIGN(auto uint16_indices,
                      BuildNumericArray<arrow::UInt16Builder>(arrow::uint16(), rows,
                                                              [](int64_t i) { return static_cast<uint16_t>(i % 16); }));
    verify_dictionary_case("dict-uint16-utf8", uint16_indices);
    ASSERT_AND_ASSIGN(auto uint32_indices,
                      BuildNumericArray<arrow::UInt32Builder>(arrow::uint32(), rows,
                                                              [](int64_t i) { return static_cast<uint32_t>(i % 16); }));
    verify_dictionary_case("dict-uint32-utf8", uint32_indices);
    ASSERT_AND_ASSIGN(auto uint64_indices,
                      BuildNumericArray<arrow::UInt64Builder>(arrow::uint64(), rows,
                                                              [](int64_t i) { return static_cast<uint64_t>(i % 16); }));
    verify_dictionary_case("dict-uint64-utf8", uint64_indices);
  }

  {
    const int64_t inner_rows = nested_outer_rows * 2;
    ASSERT_AND_ASSIGN(auto utf8_values, BuildBytesArray<arrow::StringBuilder>(inner_rows, bytes_value_len));
    ASSERT_AND_ASSIGN(auto inner_list, BuildListArray(utf8_values, inner_rows, 1, false));
    ASSERT_AND_ASSIGN(auto outer_list, BuildListArray(inner_list, nested_outer_rows, 2, false));
    const uint64_t utf8_bytes = static_cast<uint64_t>(inner_rows) * bytes_value_len + (inner_rows + 1) * 4;
    const uint64_t inner_list_bytes = utf8_bytes + (inner_rows + 1) * 4;
    verify_sliced_case("list-list-utf8", outer_list, inner_list_bytes + (nested_outer_rows + 1) * 4, 128, false);

    ASSERT_AND_ASSIGN(auto nullable_outer_list, MakeNullableArray(outer_list));
    verify_sliced_case("nullable-list-list-utf8", nullable_outer_list,
                       AddValidityBytes(inner_list_bytes + (nested_outer_rows + 1) * 4, nested_outer_rows, true), 128,
                       true);

    ASSERT_AND_ASSIGN(auto struct_array, BuildStructArray(inner_list, false));
    verify_sliced_case("struct-list-utf8", struct_array, inner_list_bytes, 128, false);

    ASSERT_AND_ASSIGN(auto nullable_struct_array, MakeNullableArray(struct_array));
    verify_sliced_case("nullable-struct-list-utf8", nullable_struct_array,
                       AddValidityBytes(inner_list_bytes, nullable_struct_array->length(), true), 128, true);
  }
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupSizesForListOfStructValues) {
  const int64_t outer_rows = 2048;
  const int64_t values_per_row = 4;
  const int64_t value_rows = outer_rows * values_per_row;
  const int64_t rows_per_slice = 64;
  const size_t str_len = 512;
  const uint64_t row_group_max_size = 512 * 1024;

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  ASSERT_AND_ASSIGN(auto ids, BuildNumericArray<arrow::Int64Builder>(
                                  arrow::int64(), value_rows, [](int64_t i) { return static_cast<int64_t>(i); }));
  ASSERT_AND_ASSIGN(auto strings, BuildBytesArray<arrow::StringViewBuilder>(value_rows, str_len));
  ASSERT_AND_ASSIGN(auto struct_values,
                    arrow::StructArray::Make({ids, strings}, {arrow::field("id", ids->type(), false),
                                                              arrow::field("str", strings->type(), false)}));

  auto verify_case = [&](const std::string& name, const std::shared_ptr<arrow::Array>& array, uint64_t logical_bytes,
                         bool nullable) {
    SCOPED_TRACE(name);
    auto schema = arrow::schema({arrow::field("value", array->type(), nullable)});
    auto test_path = RegisterTestPath(name);

    auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
    for (int64_t offset = 0; offset < array->length(); offset += rows_per_slice) {
      auto slice_len = std::min<int64_t>(rows_per_slice, array->length() - offset);
      auto rb = arrow::RecordBatch::Make(schema, slice_len, {array->Slice(offset, slice_len)});
      ASSERT_STATUS_OK(vx_writer.Write(rb));
    }
    ASSERT_STATUS_OK(vx_writer.Flush());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(array->length(), cgfile.end_index);

    auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
    auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
    auto vx_reader = vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                                std::vector<std::string>{"value"}, vx_file_size, vx_footer_size);
    ASSERT_STATUS_OK(vx_reader.open());

    ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
    ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, array->length()));
    ASSERT_GT(rg_infos.size(), 1u);

    uint64_t row_count = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_EQ(rg_infos[i].start_offset, row_count) << name << " rg[" << i << "] should be contiguous";
      ASSERT_GT(rg_infos[i].memory_size, 0u) << name << " rg[" << i << "] memory_size should be > 0";
      row_count = rg_infos[i].end_offset;
    }
    ASSERT_EQ(row_count, static_cast<uint64_t>(array->length()));
  };

  const auto struct_value_bytes = static_cast<uint64_t>(value_rows) * (sizeof(int64_t) + str_len + 16);

  ASSERT_AND_ASSIGN(auto list_array, BuildListArray(struct_values, outer_rows, values_per_row, false));
  const auto list_bytes = struct_value_bytes + (outer_rows + 1) * 4;
  verify_case("list-struct-mixed", list_array, list_bytes, false);

  ASSERT_AND_ASSIGN(auto nullable_list_array, MakeNullableArray(list_array));
  verify_case("nullable-list-struct-mixed", nullable_list_array, AddValidityBytes(list_bytes, outer_rows, true), true);

  ASSERT_AND_ASSIGN(auto large_list_array, BuildLargeListArray(struct_values, outer_rows, values_per_row, false));
  const auto large_list_bytes = struct_value_bytes + (outer_rows + 1) * 8;
  verify_case("large-list-struct-mixed", large_list_array, large_list_bytes, false);

  ASSERT_AND_ASSIGN(auto nullable_large_list_array, MakeNullableArray(large_list_array));
  verify_case("nullable-large-list-struct-mixed", nullable_large_list_array,
              AddValidityBytes(large_list_bytes, outer_rows, true), true);

  ASSERT_AND_ASSIGN(auto nullable_strings, MakeNullableArray(strings));
  ASSERT_AND_ASSIGN(auto struct_values_with_nullable_field,
                    arrow::StructArray::Make({ids, nullable_strings}, {arrow::field("id", ids->type(), false),
                                                                       arrow::field("str", strings->type(), true)}));
  const auto struct_value_bytes_with_nullable_field = AddValidityBytes(struct_value_bytes, value_rows, true);

  ASSERT_AND_ASSIGN(auto list_array_with_nullable_field,
                    BuildListArray(struct_values_with_nullable_field, outer_rows, values_per_row, true));
  verify_case("list-struct-mixed-nullable-field", list_array_with_nullable_field,
              struct_value_bytes_with_nullable_field + (outer_rows + 1) * 4, false);

  ASSERT_AND_ASSIGN(auto large_list_array_with_nullable_field,
                    BuildLargeListArray(struct_values_with_nullable_field, outer_rows, values_per_row, true));
  verify_case("large-list-struct-mixed-nullable-field", large_list_array_with_nullable_field,
              struct_value_bytes_with_nullable_field + (outer_rows + 1) * 8, false);
}

TEST_F(VortexV2RowGroupSizeTest, TestV2RowGroupSizesForMixedTopLevelColumns) {
  const int64_t total_rows = 4096;
  const int64_t rows_per_slice = 32;
  const size_t str_len = 2048;
  const uint64_t row_group_max_size = 512 * 1024;

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  ASSERT_AND_ASSIGN(auto ids, BuildNumericArray<arrow::Int64Builder>(
                                  arrow::int64(), total_rows, [](int64_t i) { return static_cast<int64_t>(i); }));
  ASSERT_AND_ASSIGN(auto strings, BuildBytesArray<arrow::StringViewBuilder>(total_rows, str_len));

  const int64_t values_per_row = 8;
  ASSERT_AND_ASSIGN(auto values,
                    BuildNumericArray<arrow::FloatBuilder>(arrow::float32(), total_rows * values_per_row,
                                                           [](int64_t i) { return static_cast<float>(i); }));
  ASSERT_AND_ASSIGN(auto vectors, BuildFixedSizeListArray(values, static_cast<int32_t>(values_per_row), false));

  auto schema = arrow::schema({
      arrow::field("id", ids->type(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("str", strings->type(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
      arrow::field("vector", vectors->type(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
  });

  auto test_path = RegisterTestPath("mixed-top-level");
  auto vx_writer = vortex::VortexFileWriter(file_system_, schema, test_path, properties_);
  for (int64_t offset = 0; offset < total_rows; offset += rows_per_slice) {
    auto slice_len = std::min<int64_t>(rows_per_slice, total_rows - offset);
    auto rb = arrow::RecordBatch::Make(
        schema, slice_len,
        {ids->Slice(offset, slice_len), strings->Slice(offset, slice_len), vectors->Slice(offset, slice_len)});
    ASSERT_STATUS_OK(vx_writer.Write(rb));
  }
  ASSERT_STATUS_OK(vx_writer.Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(total_rows, cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto vx_reader =
      vortex::VortexFormatReader(file_system_, schema, test_path, properties_,
                                 std::vector<std::string>{"id", "str", "vector"}, vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  const auto logical_bytes =
      static_cast<uint64_t>(total_rows) * (sizeof(int64_t) + str_len + 16 + values_per_row * sizeof(float));
  ASSERT_LE(rg_infos.size(), MaxExpectedRowGroups(logical_bytes, row_group_max_size, total_rows));
  ASSERT_GT(rg_infos.size(), 1u);
  ASSERT_LT(rg_infos.size(), static_cast<size_t>(total_rows / rows_per_slice / 2));

  uint64_t row_count = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, row_count) << "rg[" << i << "] should be contiguous";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    row_count = rg_infos[i].end_offset;
  }
  ASSERT_EQ(row_count, static_cast<uint64_t>(total_rows));
}

}  // namespace milvus_storage
