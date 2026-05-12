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
#include <memory>
#include <random>
#include <vector>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/concatenate.h>
#include <arrow/builder.h>
#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

// V2-specific fixture: always uses format_version=2
class VortexV2Test : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, "2");

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema(needed_columns_));
    for (int64_t batch_idx = 0; batch_idx < batch_count_; ++batch_idx) {
      ASSERT_AND_ASSIGN(auto rb, CreateTestData(schema_, batch_idx * rows_per_batch_, false, rows_per_batch_, 4, 50,
                                                needed_columns_));
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

  [[nodiscard]] int64_t recordBatchsRows() const { return batch_count_ * rows_per_batch_; }

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
  // Keep the common test_env schema/data shape, but make each V2 test file large
  // enough to cross the minimum validated row-group size (128KB).
  const int64_t rows_per_batch_ = 8192;
  const int64_t batch_count_ = 4;
};

TEST_F(VortexV2Test, TestV2StatsEnabledUsesRowGroupZoneMapLayout) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
  auto vxfile =
      VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size, vx_footer_size);

  ASSERT_EQ(vxfile.RootLayoutEncoding(), "milvus.v2_zoned_row_group");
  ASSERT_TRUE(vxfile.RowGroupZoneMapDataBeforeZones());
  auto splits = vxfile.Splits();
  ASSERT_GT(splits.size(), 1u);
  ASSERT_EQ(vxfile.RowGroupZoneMapCount(), splits.size());

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                              vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(recordBatchsRows(), rb->num_rows());
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
  }
}

TEST_F(VortexV2Test, TestV2StatsDisabledUsesPlainRowGroupLayout) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "false");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
  auto vxfile =
      VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size, vx_footer_size);

  ASSERT_NE(vxfile.RootLayoutEncoding(), "milvus.v2_zoned_row_group");
  ASSERT_FALSE(vxfile.RowGroupZoneMapDataBeforeZones());
  ASSERT_EQ(vxfile.RowGroupZoneMapCount(), 0u);
  ASSERT_GT(vxfile.Splits().size(), 1u);

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                              vx_file_size, vx_footer_size);
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, recordBatchsRows()));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(recordBatchsRows(), rb->num_rows());
}

TEST_F(VortexV2Test, TestV2RowGroupZoneMapFilterScan) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());

  auto vx_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto vx_file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
  auto vxfile =
      VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_, vx_file_size, vx_footer_size);
  ASSERT_EQ(vxfile.RootLayoutEncoding(), "milvus.v2_zoned_row_group");

  auto scan_builder = vxfile.CreateScanBuilder();
  scan_builder.WithFilter(expr::and_(expr::gt_eq(expr::column("id"), expr::literal(scalar::int64(1200))),
                                     expr::lt(expr::column("id"), expr::literal(scalar::int64(1300)))));
  scan_builder.WithProjection(expr::select(std::vector<std::string_view>{"id"}, expr::root()));
  scan_builder.WithRowRange(1000, 1500);

  ArrowArrayStream array_stream = std::move(scan_builder).IntoStream();
  ASSERT_AND_ASSIGN(auto chunked_array, arrow::ImportChunkedArray(&array_stream));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), 100);
  ASSERT_EQ(rb->num_columns(), 1);

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(1200 + i));
  }
}

TEST_F(VortexV2Test, TestV2RowGroupWrite) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());

  auto vx_writer = vortex::VortexFileWriter(file_system_, schema_, test_file_name_, properties_);
  for (const auto& rb : record_batches_) {
    ASSERT_TRUE(vx_writer.Write(rb).ok());
  }
  ASSERT_TRUE(vx_writer.Flush().ok());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
  ASSERT_EQ(recordBatchsRows(), cgfile.end_index);

  auto total_rows = recordBatchsRows();

  // --- blocking_read: full scan ---
  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns());
  ASSERT_STATUS_OK(vx_reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, total_rows));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));

  ASSERT_EQ(total_rows, rb->num_rows());
  ASSERT_EQ(3, rb->num_columns());

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(rb->column(2));
  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(i));
    ASSERT_DOUBLE_EQ(value_array->Value(i), static_cast<double>(i) * 1.5);
  }

  // --- get_chunk: per row-group read ---
  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());
  ASSERT_GT(rg_infos.size(), 1u);
  uint64_t offset = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
    auto chunk_id = std::dynamic_pointer_cast<arrow::Int64Array>(chunk_rb->column(0));
    ASSERT_EQ(chunk_id->Value(0), static_cast<int64_t>(offset)) << "rg[" << i << "] first value mismatch";
    offset += chunk_rb->num_rows();
  }
  ASSERT_EQ(offset, total_rows);

  // --- take: random access ---
  auto proj_schema = arrow::schema({schema_->field(0)});
  auto take_reader = vortex::VortexFormatReader(file_system_, proj_schema, test_file_name_, properties_,
                                                std::vector<std::string>{"id"});
  ASSERT_STATUS_OK(take_reader.open());

  std::vector<int64_t> take_indices = {0, 42, total_rows / 2, total_rows - 1};
  ASSERT_AND_ASSIGN(auto table, take_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), static_cast<int64_t>(take_indices.size()));
  auto take_id = std::dynamic_pointer_cast<arrow::Int64Array>(take_rb->column(0));
  for (size_t i = 0; i < take_indices.size(); ++i) {
    ASSERT_EQ(take_id->Value(i), take_indices[i]);
  }

  // --- read_with_range: partial range read ---
  uint64_t range_start = rg_infos[0].end_offset;
  uint64_t range_end = rg_infos[1].end_offset;
  ASSERT_AND_ASSIGN(auto range_reader, vx_reader.read_with_range(range_start, range_end));
  std::shared_ptr<arrow::RecordBatch> range_batch;
  int64_t range_rows = 0;
  while (true) {
    ASSERT_STATUS_OK(range_reader->ReadNext(&range_batch));
    if (!range_batch)
      break;
    if (range_rows == 0) {
      auto range_id = std::dynamic_pointer_cast<arrow::Int64Array>(range_batch->column(0));
      ASSERT_EQ(range_id->Value(0), static_cast<int64_t>(range_start));
    }
    range_rows += range_batch->num_rows();
  }
  ASSERT_EQ(range_rows, static_cast<int64_t>(range_end - range_start));
}

// Test that inline_array_node enables sub-segment range reads.
// Writes FSB(512) data, takes 1 row, and asserts IO read bytes are small.
// Only runs in cloud (S3) environment where FilesystemMetrics are available.
TEST_F(VortexV2Test, TestInlineArrayNodeSubSegmentRead) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "Sub-segment IO test requires cloud environment with FilesystemMetrics";
  }

  // Get cloud filesystem with metrics
  ASSERT_AND_ASSIGN(auto cloud_fs, GetFileSystem(properties_));
  auto observable = std::dynamic_pointer_cast<Observable>(cloud_fs);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  // Build schema with a single FSB(512) column
  auto fsb_schema = arrow::schema({arrow::field("embedding", arrow::fixed_size_binary(512), false,
                                                arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  // Write 10000 rows of FSB(512) data (~5MB)
  const int64_t num_rows = 10000;
  const int fsb_width = 512;
  std::string test_path = GetTestBasePath("vortex-inline-test") + "/test-fsb.vx";

  {
    auto vx_writer = vortex::VortexFileWriter(cloud_fs, fsb_schema, test_path, properties_);

    arrow::FixedSizeBinaryBuilder fsb_builder(arrow::fixed_size_binary(fsb_width));
    std::string row_data(fsb_width, '\0');
    std::mt19937_64 rng(42);
    for (int64_t i = 0; i < num_rows; ++i) {
      // Fill with random data to prevent compression from shrinking the file
      auto* p = reinterpret_cast<uint64_t*>(row_data.data());
      for (size_t j = 0; j < fsb_width / sizeof(uint64_t); ++j) {
        p[j] = rng();
      }
      ASSERT_TRUE(fsb_builder.Append(row_data).ok());
    }
    std::shared_ptr<arrow::Array> fsb_array;
    ASSERT_TRUE(fsb_builder.Finish(&fsb_array).ok());

    auto rb = arrow::RecordBatch::Make(fsb_schema, num_rows, {fsb_array});
    ASSERT_TRUE(vx_writer.Write(rb).ok());
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(num_rows, cgfile.end_index);
  }

  // Open reader and take 1 row
  auto vx_reader =
      vortex::VortexFormatReader(cloud_fs, fsb_schema, test_path, properties_, std::vector<std::string>{"embedding"});
  ASSERT_STATUS_OK(vx_reader.open());

  // Reset metrics before take
  metrics->Reset();

  std::vector<int64_t> indices = {42};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(indices));
  ASSERT_EQ(1, table->num_rows());

  // Check IO: reading 1 row of 512 bytes should read far less than the full file
  // Full file is ~5MB; sub-segment read should be well under 1MB
  auto read_bytes = metrics->GetReadBytes();
  ASSERT_EQ(read_bytes, fsb_width) << "Sub-segment read of 1 row should be exactly " << fsb_width << " bytes, got "
                                   << read_bytes;
}

// Test V2 row group splits with variable-size string data.
// Writes two batches of 8192 rows with different string sizes (128B and 512B),
// then verifies that row group splits reflect the byte-size-based partitioning.
TEST_F(VortexV2Test, TestV2RowGroupSplitsBySize) {
  const int64_t rows_per_batch = 8192;
  const size_t small_str_len = 128;
  const size_t large_str_len = 512;
  const uint64_t row_group_max_size = 1024 * 1024;  // 1MB

  auto str_schema = arrow::schema(
      {arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))});

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  std::string test_path = "test-v2-rg-splits.vx";

  // Helper to build a batch of fixed-length random strings
  auto make_string_batch = [&](size_t str_len, int64_t num_rows) {
    arrow::StringBuilder builder;
    std::string s(str_len, 'x');
    for (int64_t i = 0; i < num_rows; ++i) {
      EXPECT_TRUE(builder.Append(s).ok());
    }
    std::shared_ptr<arrow::Array> arr;
    EXPECT_TRUE(builder.Finish(&arr).ok());
    return arrow::RecordBatch::Make(str_schema, num_rows, {arr});
  };

  // Write: batch1 = 8192 rows * 128B, batch2 = 8192 rows * 512B
  {
    auto vx_writer = vortex::VortexFileWriter(file_system_, str_schema, test_path, properties_);
    ASSERT_TRUE(vx_writer.Write(make_string_batch(small_str_len, rows_per_batch)).ok());
    ASSERT_TRUE(vx_writer.Write(make_string_batch(large_str_len, rows_per_batch)).ok());
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(rows_per_batch * 2, cgfile.end_index);
  }

  // Read back and check row group infos
  auto vx_reader =
      vortex::VortexFormatReader(file_system_, str_schema, test_path, properties_, std::vector<std::string>{"str"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());

  // batch1: 8192 rows * 128B utf8 (132 B/row with offsets) ≈ 1.03 MB
  // batch2: 8192 rows * 512B utf8 (516 B/row with offsets) ≈ 4.03 MB
  // row_group_max_size = 1MB → expect 6 row groups:
  //   rg[0]: batch1 first ~1MB
  //   rg[1]: batch1 tail + batch2 start (cross-batch merge)
  //   rg[2..4]: batch2 middle chunks (~1MB each)
  //   rg[5]: batch2 tail (EOF remainder)
  ASSERT_EQ(rg_infos.size(), 6u);

  // Verify exact row ranges
  struct Expected {
    uint64_t start;
    uint64_t end;
  };
  std::vector<Expected> expected = {{0, 7944},      {7944, 10161},  {10161, 12194},
                                    {12194, 14227}, {14227, 16260}, {16260, 16384}};
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, expected[i].start) << "rg[" << i << "] start_offset mismatch";
    ASSERT_EQ(rg_infos[i].end_offset, expected[i].end) << "rg[" << i << "] end_offset mismatch";
  }

  // Verify memory_size is non-zero for all row groups
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
  }

  // Verify total rows
  uint64_t total_rows = 0;
  for (const auto& rg : rg_infos) {
    total_rows += (rg.end_offset - rg.start_offset);
  }
  ASSERT_EQ(total_rows, static_cast<uint64_t>(rows_per_batch * 2));

  // --- blocking_read: full scan and verify string lengths ---
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, rows_per_batch * 2));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), rows_per_batch * 2);
  auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(0));
  // batch1 rows should have small_str_len, batch2 rows should have large_str_len
  ASSERT_EQ(str_array->GetView(0).size(), small_str_len);
  ASSERT_EQ(str_array->GetView(rows_per_batch).size(), large_str_len);
  ASSERT_EQ(str_array->GetView(rows_per_batch * 2 - 1).size(), large_str_len);

  // --- get_chunk: per row-group read ---
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
  }

  // --- take: spot-check boundary rows ---
  std::vector<int64_t> take_indices = {0, rows_per_batch - 1, rows_per_batch, rows_per_batch * 2 - 1};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), 4);
  auto take_str = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(0));
  ASSERT_EQ(take_str->GetView(0).size(), small_str_len);  // first row of batch1
  ASSERT_EQ(take_str->GetView(1).size(), small_str_len);  // last row of batch1
  ASSERT_EQ(take_str->GetView(2).size(), large_str_len);  // first row of batch2
  ASSERT_EQ(take_str->GetView(3).size(), large_str_len);  // last row of batch2
}

TEST_F(VortexV2Test, TestV2RowGroupMultiColumnSplitsBySize) {
  const int64_t rows_per_batch = 8192;
  const size_t str_len_a = 256;
  const size_t str_len_b = 512;
  const size_t str_len_c = 1024;
  const uint64_t row_group_max_size = 10 * 1024 * 1024;  // 10MB

  auto multi_schema = arrow::schema({
      arrow::field("col_a", arrow::utf8(), false),
      arrow::field("col_b", arrow::utf8(), false),
      arrow::field("col_c", arrow::utf8(), false),
  });

  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(row_group_max_size).c_str());

  std::string test_path = "test-v2-rg-multi-col.vx";

  // Helper to build a multi-column batch with fixed-length strings per column
  auto make_batch = [&](int64_t num_rows) {
    auto build_col = [&](size_t str_len, int64_t n) {
      arrow::StringBuilder builder;
      std::string s(str_len, 'x');
      for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(builder.Append(s).ok());
      }
      std::shared_ptr<arrow::Array> arr;
      EXPECT_TRUE(builder.Finish(&arr).ok());
      return arr;
    };
    return arrow::RecordBatch::Make(
        multi_schema, num_rows,
        {build_col(str_len_a, num_rows), build_col(str_len_b, num_rows), build_col(str_len_c, num_rows)});
  };

  // Write 4 batches of 8192 rows each
  // Per-row uncompressed: 256+4 + 512+4 + 1024+4 ≈ 1804 B/row (with offsets)
  // Per batch: 8192 * 1804 ≈ 14.1 MB → each batch > 10MB limit → gets split
  // Total: 4 * 8192 = 32768 rows ≈ 56.4 MB
  {
    auto vx_writer = vortex::VortexFileWriter(file_system_, multi_schema, test_path, properties_);
    for (int i = 0; i < 4; ++i) {
      ASSERT_TRUE(vx_writer.Write(make_batch(rows_per_batch)).ok());
    }
    ASSERT_TRUE(vx_writer.Flush().ok());
    ASSERT_AND_ASSIGN(auto cgfile, vx_writer.Close());
    ASSERT_EQ(rows_per_batch * 4, cgfile.end_index);
  }

  // Read back and check row group infos
  auto vx_reader = vortex::VortexFormatReader(file_system_, multi_schema, test_path, properties_,
                                              std::vector<std::string>{"col_a", "col_b", "col_c"});
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto rg_infos, vx_reader.get_row_group_infos());

  // Each batch ≈ 14.1 MB, limit 10MB → each batch should be split into ~2 row groups
  // 4 batches → expect ~6-8 row groups (with cross-batch merging and EOF tail)
  ASSERT_GT(rg_infos.size(), 4u) << "Should have more row groups than input batches";

  // Verify contiguous ranges and total rows
  ASSERT_EQ(rg_infos[0].start_offset, 0u);
  uint64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    uint64_t rg_rows = rg_infos[i].end_offset - rg_infos[i].start_offset;
    ASSERT_GT(rg_rows, 0u) << "rg[" << i << "] should have rows";
    ASSERT_GT(rg_infos[i].memory_size, 0u) << "rg[" << i << "] memory_size should be > 0";
    total_rows += rg_rows;
    if (i > 0) {
      ASSERT_EQ(rg_infos[i].start_offset, rg_infos[i - 1].end_offset) << "rg[" << i << "] should be contiguous";
    }
  }
  ASSERT_EQ(total_rows, static_cast<uint64_t>(rows_per_batch * 4));

  // --- blocking_read: full scan and verify data ---
  ASSERT_AND_ASSIGN(auto chunked_array, vx_reader.blocking_read(0, rows_per_batch * 4));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), rows_per_batch * 4);
  ASSERT_EQ(rb->num_columns(), 3);
  // Verify string lengths for each column
  auto col_a = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(0));
  auto col_b = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(1));
  auto col_c = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(2));
  ASSERT_EQ(col_a->GetView(0).size(), str_len_a);
  ASSERT_EQ(col_b->GetView(0).size(), str_len_b);
  ASSERT_EQ(col_c->GetView(0).size(), str_len_c);

  // --- get_chunk: per row-group read ---
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto chunk_rb, vx_reader.get_chunk(static_cast<int>(i)));
    ASSERT_EQ(chunk_rb->num_rows(), rg_infos[i].end_offset - rg_infos[i].start_offset)
        << "rg[" << i << "] row count mismatch";
    ASSERT_EQ(chunk_rb->num_columns(), 3);
  }

  // --- take: random access across batch boundaries ---
  std::vector<int64_t> take_indices = {0, rows_per_batch - 1, rows_per_batch, rows_per_batch * 3,
                                       rows_per_batch * 4 - 1};
  ASSERT_AND_ASSIGN(auto table, vx_reader.take(take_indices));
  ASSERT_AND_ASSIGN(auto take_rb, table->CombineChunksToBatch());
  ASSERT_EQ(take_rb->num_rows(), static_cast<int64_t>(take_indices.size()));
  // All rows should have the same string lengths (uniform data)
  auto take_col_a = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(0));
  auto take_col_c = std::dynamic_pointer_cast<arrow::StringArray>(take_rb->column(2));
  for (int i = 0; i < take_rb->num_rows(); ++i) {
    ASSERT_EQ(take_col_a->GetView(i).size(), str_len_a) << "take row " << i << " col_a size mismatch";
    ASSERT_EQ(take_col_c->GetView(i).size(), str_len_c) << "take row " << i << " col_c size mismatch";
  }
}

}  // namespace milvus_storage
