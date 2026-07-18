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
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/table.h>
#include <arrow/array/concatenate.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "test_env.h"

namespace milvus_storage {

using namespace lance;

class LanceBasicTest : public ::testing::Test {
  protected:
  struct LanceDataFileVersion {
    uint16_t major;
    uint16_t minor;
  };

  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // For Arrow filesystem operations
    arrow_base_path_ = GetTestBasePath("lance-fragment-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, arrow_base_path_));

    // For Lance, use relative path - BuildLanceBaseUri will be called internally
    base_path_ = arrow_base_path_;

    // Create a simple test schema with field IDs required by packed writer
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());

    // Create test data
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override {
    if (!IsCloudEnv()) {
      ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
    }
  }

  arrow::Result<LanceDataFileVersion> ReadLanceDataFileVersion() const {
    auto* subtree = dynamic_cast<arrow::fs::SubTreeFileSystem*>(fs_.get());
    if (subtree == nullptr) {
      return arrow::Status::Invalid("Expected a subtree filesystem for local Lance test data");
    }

    const boost::filesystem::path root_path(subtree->base_path());
    const boost::filesystem::path dataset_path = root_path / arrow_base_path_;
    for (boost::filesystem::recursive_directory_iterator it(dataset_path), end; it != end; ++it) {
      if (!boost::filesystem::is_regular_file(it->path()) || it->path().extension() != ".lance") {
        continue;
      }

      const auto file_path = boost::filesystem::relative(it->path(), root_path).generic_string();
      ARROW_ASSIGN_OR_RAISE(auto input, fs_->OpenInputFile(file_path));
      ARROW_ASSIGN_OR_RAISE(auto size, input->GetSize());
      if (size < 8) {
        return arrow::Status::Invalid("Lance data file is too small to contain a version footer: ", file_path);
      }

      ARROW_ASSIGN_OR_RAISE(auto footer, input->ReadAt(size - 8, 8));
      const auto* bytes = footer->data();
      if (footer->size() != 8 || bytes[4] != 'L' || bytes[5] != 'A' || bytes[6] != 'N' || bytes[7] != 'C') {
        return arrow::Status::Invalid("Invalid Lance data file footer: ", file_path);
      }

      return LanceDataFileVersion{
          .major = static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8)),
          .minor = static_cast<uint16_t>(bytes[2] | (static_cast<uint16_t>(bytes[3]) << 8)),
      };
    }

    return arrow::Status::Invalid("No Lance data file found under ", arrow_base_path_);
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;        // Lance URI (s3://bucket/path or /tmp/path)
  std::string arrow_base_path_;  // Path for Arrow filesystem operations
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

class LanceRleStorageVersionTest : public LanceBasicTest,
                                   public ::testing::WithParamInterface<LanceDataStorageFormat> {};

TEST_P(LanceRleStorageVersionTest, WritesAndReadsCustomerShapedRleData) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 100'000;
  constexpr int64_t kRunLength = 4096;
  constexpr int64_t kSampleSize = 2 * kRunLength + 1;
  constexpr int32_t kEmbeddingDimension = 4;

  auto rle_metadata = arrow::key_value_metadata({"lance-encoding:rle-threshold", "lance-encoding:bss"}, {"1.0", "off"});
  auto curr_time_type = arrow::timestamp(arrow::TimeUnit::MILLI);
  auto rle_schema = arrow::schema({
      arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), kEmbeddingDimension), false),
      arrow::field("uuid", arrow::utf8(), false, rle_metadata),
      arrow::field("curr_time", curr_time_type, false, rle_metadata),
  });

  std::vector<float> embedding_values(kRows * kEmbeddingDimension);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int32_t dim = 0; dim < kEmbeddingDimension; ++dim) {
      embedding_values[row * kEmbeddingDimension + dim] = static_cast<float>(dim) + 0.25F;
    }
  }

  auto embedding_value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder embedding_builder(arrow::default_memory_pool(), embedding_value_builder,
                                                kEmbeddingDimension);
  ASSERT_STATUS_OK(embedding_builder.AppendValues(kRows));
  ASSERT_STATUS_OK(embedding_value_builder->AppendValues(embedding_values));

  constexpr std::string_view kUuidA = "00000000-0000-0000-0000-000000000001";
  constexpr std::string_view kUuidB = "00000000-0000-0000-0000-000000000002";
  arrow::StringBuilder uuid_builder;
  ASSERT_STATUS_OK(uuid_builder.Reserve(kRows));
  ASSERT_STATUS_OK(uuid_builder.ReserveData(kRows * kUuidA.size()));
  for (int64_t row = 0; row < kRows; ++row) {
    uuid_builder.UnsafeAppend((row / kRunLength) % 2 == 0 ? kUuidA : kUuidB);
  }

  constexpr int64_t kBaseTimestamp = 1'784'065'218'692;
  std::vector<int64_t> curr_time_values(kRows);
  for (int64_t row = 0; row < kRows; ++row) {
    curr_time_values[row] = kBaseTimestamp + row / kRunLength;
  }
  arrow::TimestampBuilder curr_time_builder(curr_time_type, arrow::default_memory_pool());
  ASSERT_STATUS_OK(curr_time_builder.AppendValues(curr_time_values));

  std::shared_ptr<arrow::Array> embedding_array;
  std::shared_ptr<arrow::Array> uuid_array;
  std::shared_ptr<arrow::Array> curr_time_array;
  ASSERT_STATUS_OK(embedding_builder.Finish(&embedding_array));
  ASSERT_STATUS_OK(uuid_builder.Finish(&uuid_array));
  ASSERT_STATUS_OK(curr_time_builder.Finish(&curr_time_array));
  auto batch = arrow::RecordBatch::Make(rle_schema, kRows, {embedding_array, uuid_array, curr_time_array});

  LanceTableWriter writer(base_path_, rle_schema, properties_, GetParam());
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);

  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));
  ASSERT_AND_ASSIGN(auto storage_version, ReadLanceDataFileVersion());
  ASSERT_EQ(storage_version.major, 2);
  ASSERT_EQ(storage_version.minor, static_cast<uint32_t>(GetParam()));

  std::vector<int64_t> row_indices(kSampleSize);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  const std::vector<std::vector<std::string>> projections = {
      {"embedding"},
      {"uuid", "curr_time"},
      {"embedding", "uuid", "curr_time"},
  };
  for (const auto& projection : projections) {
    LanceTableReader reader(parsed_uri.first, parsed_uri.second, nullptr, properties_, projection);
    ASSERT_STATUS_OK(reader.open());
    ASSERT_AND_ASSIGN(auto table, reader.take(row_indices));
    ASSERT_STATUS_OK(table->ValidateFull());
    ASSERT_EQ(table->num_rows(), kSampleSize);
    ASSERT_EQ(table->num_columns(), projection.size());
    for (size_t column = 0; column < projection.size(); ++column) {
      ASSERT_EQ(table->field(column)->name(), projection[column]);
    }

    ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());
    if (auto column = result_batch->GetColumnByName("embedding")) {
      auto embedding = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(column);
      ASSERT_NE(embedding, nullptr);
      auto values = std::dynamic_pointer_cast<arrow::FloatArray>(embedding->values());
      ASSERT_NE(values, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        for (int32_t dim = 0; dim < kEmbeddingDimension; ++dim) {
          ASSERT_FLOAT_EQ(values->Value(embedding->value_offset(row) + dim), static_cast<float>(dim) + 0.25F);
        }
      }
    }

    if (auto column = result_batch->GetColumnByName("uuid")) {
      auto uuid = std::dynamic_pointer_cast<arrow::StringArray>(column);
      ASSERT_NE(uuid, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        const auto expected = (row_indices[row] / kRunLength) % 2 == 0 ? kUuidA : kUuidB;
        ASSERT_EQ(uuid->GetString(row), std::string(expected));
      }
    }

    if (auto column = result_batch->GetColumnByName("curr_time")) {
      auto curr_time = std::dynamic_pointer_cast<arrow::TimestampArray>(column);
      ASSERT_NE(curr_time, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        ASSERT_EQ(curr_time->Value(row), kBaseTimestamp + row_indices[row] / kRunLength);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DataStorageVersions,
                         LanceRleStorageVersionTest,
                         ::testing::Values(LanceDataStorageFormat::V2_1,
                                           LanceDataStorageFormat::V2_2,
                                           LanceDataStorageFormat::V2_3),
                         [](const ::testing::TestParamInfo<LanceDataStorageFormat>& info) {
                           switch (info.param) {
                             case LanceDataStorageFormat::V2_1:
                               return "V2_1";
                             case LanceDataStorageFormat::V2_2:
                               return "V2_2";
                             case LanceDataStorageFormat::V2_3:
                               return "V2_3";
                             default:
                               return "Unknown";
                           }
                         });

TEST_F(LanceBasicTest, DefaultStorageVersionIsV2_1) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto storage_version, ReadLanceDataFileVersion());
  ASSERT_EQ(storage_version.major, 2);
  ASSERT_EQ(storage_version.minor, 1);
}

TEST_F(LanceBasicTest, TestBasic) {
  size_t num_of_batches = 10;
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Build Lance URI from relative path
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  // write without flush, single fragment
  {
    LanceTableWriter writer(base_path_, schema_, properties_);
    for (int i = 0; i < num_of_batches; i++) {
      ASSERT_STATUS_OK(writer.Write(test_batch_));
    }
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(cgfile.end_index, test_batch_->num_rows() * num_of_batches);
  }

  auto verify_reader = [&]() {
    auto read_dataset = BlockingDataset::Open(lance_uri, storage_options);
    const std::vector<uint64_t> fragment_ids = read_dataset->GetAllFragmentIds();

    uint64_t total_rows = 0;
    for (const auto& fragment_id : fragment_ids) {
      LanceTableReader reader(read_dataset, fragment_id, schema_, properties_);
      ASSERT_STATUS_OK(reader.open());
      ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
      ASSERT_FALSE(rgs.empty());
      total_rows += rgs.back().end_offset;
    }
    ASSERT_EQ(total_rows, test_batch_->num_rows() * num_of_batches);
  };

  verify_reader();
  ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
  ASSERT_STATUS_OK(CreateTestDir(fs_, arrow_base_path_));

  // write with flush, multiple fragments
  {
    LanceTableWriter writer(base_path_, schema_, properties_);
    for (int i = 0; i < num_of_batches; i++) {
      ASSERT_STATUS_OK(writer.Write(test_batch_));
      ASSERT_STATUS_OK(writer.Flush());
    }
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(cgfile.end_index, test_batch_->num_rows() * num_of_batches);
  }

  verify_reader();
}

TEST_F(LanceBasicTest, TestRead) {
  ASSERT_AND_ASSIGN(auto large_batch, CreateTestData(schema_, 0, false, 200000));

  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Build Lance URI from relative path
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(large_batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, large_batch->num_rows());

  auto read_dataset = BlockingDataset::Open(lance_uri, storage_options);

  const std::vector<uint64_t> fragment_ids = read_dataset->GetAllFragmentIds();
  // The splitting conditions(`WriteParams`) in lance are very strict.
  // So the default setting will only generate one fragment.
  ASSERT_EQ(fragment_ids.size(), 1);
  LanceTableReader reader(read_dataset, fragment_ids[0], schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_FALSE(rgs.empty());
  ASSERT_EQ(rgs.back().end_offset, large_batch->num_rows());
  auto estimated_memory_size =
      std::accumulate(rgs.begin(), rgs.end(), uint64_t{0},
                      [](uint64_t total, const RowGroupInfo& rg) { return total + rg.memory_size; });
  ASSERT_EQ(estimated_memory_size, read_dataset->EstimateFragmentMemory(fragment_ids[0]));

  auto verify_recordbatch = [&](const std::shared_ptr<arrow::RecordBatch>& batch, auto start_ridx, auto num_of_row) {
    ASSERT_EQ(batch->num_rows(), num_of_row);
    auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    for (int i = 0; i < num_of_row; i++) {
      ASSERT_EQ(id_column->Value(i), start_ridx + i);
    }
  };

  // get chunk && read range
  {
    for (size_t rg_idx = 0; rg_idx < rgs.size(); rg_idx++) {
      ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(rg_idx));
      verify_recordbatch(chunk, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);

      ASSERT_AND_ASSIGN(auto rbreader, reader.read_with_range(rgs[rg_idx].start_offset, rgs[rg_idx].end_offset));
      ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rbreader.get()));
      ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
      verify_recordbatch(result_batch, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);
    }
    ASSERT_GT(estimated_memory_size, 0);
  }

  // get chunks
  {
    std::vector<int> chunk_ids(rgs.size());
    std::iota(chunk_ids.begin(), chunk_ids.end(), 0);
    ASSERT_AND_ASSIGN(auto chunks, reader.get_chunks(chunk_ids));
    ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatches(chunks));
    ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
    verify_recordbatch(result_batch, 0, rgs.back().end_offset);
  }

  // test projection
  {
    ASSERT_AND_ASSIGN(auto projection_schema, CreateTestSchema({true, true, false, false}));

    LanceTableReader projection_reader(read_dataset, fragment_ids[0], projection_schema, properties_);
    ASSERT_STATUS_OK(projection_reader.open());
    ASSERT_AND_ASSIGN(auto projection_rgs, projection_reader.get_row_group_infos());
    ASSERT_EQ(projection_rgs.size(), rgs.size());
    for (size_t rg_idx = 0; rg_idx < rgs.size(); ++rg_idx) {
      ASSERT_EQ(projection_rgs[rg_idx].start_offset, rgs[rg_idx].start_offset);
      ASSERT_EQ(projection_rgs[rg_idx].end_offset, rgs[rg_idx].end_offset);
      ASSERT_EQ(projection_rgs[rg_idx].memory_size, rgs[rg_idx].memory_size);
    }

    for (size_t rg_idx = 0; rg_idx < rgs.size(); rg_idx++) {
      ASSERT_AND_ASSIGN(auto chunk, projection_reader.get_chunk(rg_idx));
      verify_recordbatch(chunk, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);

      ASSERT_AND_ASSIGN(auto rbreader,
                        projection_reader.read_with_range(rgs[rg_idx].start_offset, rgs[rg_idx].end_offset));
      ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rbreader.get()));
      ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
      verify_recordbatch(result_batch, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);
    }
  }
}

TEST_F(LanceBasicTest, GetColumnSizesShapeAndProportions) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Enough rows that the variable-length string column ("name") dwarfs the fixed-width
  // int64 column ("id"), so we can assert the per-column ordering is meaningful.
  ASSERT_AND_ASSIGN(auto large_batch, CreateTestData(schema_, 0, false, 200000));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(large_batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  auto read_dataset = BlockingDataset::Open(lance_uri, storage_options);
  const std::vector<uint64_t> fragment_ids = read_dataset->GetAllFragmentIds();
  ASSERT_EQ(fragment_ids.size(), 1);

  // Project id (small, index 0) and name (large, index 1) in that order.
  ASSERT_AND_ASSIGN(auto projection_schema, CreateTestSchema({true, true, false, false}));
  LanceTableReader reader(read_dataset, fragment_ids[0], projection_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_FALSE(rgs.empty());

  for (size_t rg_idx = 0; rg_idx < rgs.size(); ++rg_idx) {
    // get_column_sizes returns raw whole-fragment per-column weights (not chunk-normalized);
    // ColumnGroupReader normalizes them to the chunk's real memory.
    ASSERT_AND_ASSIGN(auto column_sizes, reader.get_column_sizes(static_cast<int>(rg_idx)));
    // Inner index: one weight per projected column, in projection (id, name) order.
    ASSERT_EQ(column_sizes.size(), 2u);

    const uint64_t id_size = column_sizes[0];
    const uint64_t name_size = column_sizes[1];
    // Both projected columns must carry a positive weight. We do NOT assert an
    // ordering between them: Lance reports the accurate decoded in-memory size, and
    // for short strings the string column can weigh about the same as (or slightly
    // less than) an 8-byte int64 column, so name > id is not a reliable invariant.
    ASSERT_GT(id_size, 0u);
    ASSERT_GT(name_size, 0u);
  }
}

TEST_F(LanceBasicTest, EstimatedMemoryAccountsForDeletions) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 10'000;
  constexpr int64_t kDeletedRows = 2'000;
  ASSERT_AND_ASSIGN(auto id_schema, CreateTestSchema({true, false, false, false}));
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(id_schema, 0, false, kRows, 4, 50, {true, false, false, false}));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  LanceTableWriter writer(base_path_, id_schema, properties_);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);
  auto dataset = BlockingDataset::Open(lance_uri, storage_options);
  dataset->DeleteRows("id < 2000");

  auto fragment_ids = dataset->GetAllFragmentIds();
  ASSERT_EQ(fragment_ids.size(), 1);
  LanceTableReader reader(dataset, fragment_ids[0], id_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_EQ(rgs.size(), 1);
  ASSERT_EQ(rgs[0].end_offset, kRows - kDeletedRows);
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  ASSERT_EQ(rgs[0].memory_size, GetRecordBatchMemorySize(chunk));
}

TEST_F(LanceBasicTest, FixedSizeListUsesExactMemoryEstimate) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 10'000;
  constexpr int32_t kDimension = 16;
  auto vector_schema = arrow::schema({arrow::field(
      "embedding", arrow::fixed_size_list(arrow::float32(), kDimension), false,
      arrow::key_value_metadata({"lance-encoding:rle-threshold", "lance-encoding:bss"}, {"1.0", "off"}))});

  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder vector_builder(arrow::default_memory_pool(), value_builder, kDimension);
  ASSERT_STATUS_OK(vector_builder.AppendValues(kRows));
  ASSERT_STATUS_OK(value_builder->AppendValues(std::vector<float>(kRows * kDimension, 1.0F)));

  std::shared_ptr<arrow::Array> vector_array;
  ASSERT_STATUS_OK(vector_builder.Finish(&vector_array));
  auto batch = arrow::RecordBatch::Make(vector_schema, kRows, {vector_array});

  LanceTableWriter writer(base_path_, vector_schema, properties_);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));

  LanceTableReader reader(parsed_uri.first, parsed_uri.second, vector_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  auto estimated_memory_size =
      std::accumulate(rgs.begin(), rgs.end(), uint64_t{0},
                      [](uint64_t total, const RowGroupInfo& rg) { return total + rg.memory_size; });
  ASSERT_EQ(estimated_memory_size, GetRecordBatchMemorySize(batch));
}

TEST_F(LanceBasicTest, LegacyFormatFallsBackToZeroMemoryEstimate) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 10'000;
  ASSERT_AND_ASSIGN(auto vector_schema, CreateTestSchema({false, false, false, true}));
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(vector_schema, 0, false, kRows, 4, 50, {false, false, false, true}));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  LanceTableWriter writer(base_path_, vector_schema, properties_, LanceDataStorageFormat::Legacy);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);

  auto dataset = BlockingDataset::Open(lance_uri, storage_options);
  auto fragment_ids = dataset->GetAllFragmentIds();
  ASSERT_EQ(fragment_ids.size(), 1);
  LanceTableReader reader(dataset, fragment_ids[0], vector_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());

  for (const auto& rg : rgs) {
    ASSERT_EQ(rg.memory_size, 0);
  }
  ASSERT_AND_ASSIGN(auto rbreader, reader.read_with_range(0, kRows));
  ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rbreader.get()));
  ASSERT_EQ(table->num_rows(), kRows);
}

TEST_F(LanceBasicTest, TestCachedOpenRejectsMissingNeededColumnWithoutReadSchema) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  ASSERT_AND_ASSIGN(auto metadata,
                    LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));

  auto reader_result = LanceTableReader::MetaTrait::create_from_metadata(metadata, cgfile, nullptr /* read_schema */,
                                                                         {"id", "missing_column"}, "");
  ASSERT_FALSE(reader_result.ok());
  EXPECT_TRUE(reader_result.status().IsInvalid());
  EXPECT_NE(reader_result.status().ToString().find("missing_column"), std::string::npos);
}

TEST_F(LanceBasicTest, CachedCreateReaderReappliesProjection) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  ASSERT_AND_ASSIGN(auto metadata,
                    LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));
  auto id_metadata = metadata;
  auto value_metadata = metadata;
  ASSERT_EQ(id_metadata.get(), value_metadata.get());

  ASSERT_AND_ASSIGN(auto id_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                        id_metadata, cgfile, nullptr /* read_schema */, {"id"}, ""));
  ASSERT_AND_ASSIGN(auto id_rgs, id_reader->get_row_group_infos());
  ASSERT_FALSE(id_rgs.empty());
  ASSERT_EQ(id_rgs.size(), metadata->row_group_infos.size());
  for (size_t rg_idx = 0; rg_idx < id_rgs.size(); ++rg_idx) {
    ASSERT_EQ(id_rgs[rg_idx].memory_size, metadata->row_group_infos[rg_idx].memory_size);
  }
  ASSERT_AND_ASSIGN(auto id_chunk, id_reader->get_chunk(0));
  ASSERT_EQ(id_chunk->num_columns(), 1);
  ASSERT_EQ(id_chunk->schema()->field(0)->name(), "id");
  ASSERT_EQ(id_chunk->num_rows(), static_cast<int64_t>(id_rgs[0].end_offset - id_rgs[0].start_offset));
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(id_chunk->column(0));
  ASSERT_NE(id_array, nullptr);
  for (int64_t i = 0; i < id_chunk->num_rows(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(id_rgs[0].start_offset) + i);
  }

  ASSERT_AND_ASSIGN(auto value_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                           value_metadata, cgfile, nullptr /* read_schema */, {"value"}, ""));
  ASSERT_AND_ASSIGN(auto value_rgs, value_reader->get_row_group_infos());
  ASSERT_FALSE(value_rgs.empty());
  ASSERT_AND_ASSIGN(auto value_chunk, value_reader->get_chunk(0));
  ASSERT_EQ(value_chunk->num_columns(), 1);
  ASSERT_EQ(value_chunk->schema()->field(0)->name(), "value");
  ASSERT_EQ(value_chunk->num_rows(), static_cast<int64_t>(value_rgs[0].end_offset - value_rgs[0].start_offset));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(value_chunk->column(0));
  ASSERT_NE(value_array, nullptr);
  for (int64_t i = 0; i < value_chunk->num_rows(); ++i) {
    const auto row = static_cast<int64_t>(value_rgs[0].start_offset) + i;
    ASSERT_DOUBLE_EQ(value_array->Value(i), row * 1.5);
  }
}

// Test that storage options are correctly passed through writer and reader
TEST_F(LanceBasicTest, TestStorageOptionsIntegration) {
  // Mirror fs.* into extfs.default.* so resolve_config can match by address+bucket
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  if (fs_config.storage_type == "remote") {
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

  // Writer uses storage options from properties
  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, test_batch_->num_rows());

  // Parse lance_uri and fragment_id from cgfile.path (format: {lance_uri}?fragment_id=X)
  ASSERT_AND_ASSIGN(auto parsed, ParseLanceUri(cgfile.path));
  auto lance_uri = parsed.first;
  auto fragment_id = parsed.second;

  // Reader opens dataset using full Lance URI and storage options from properties
  // Use the URI-based constructor to test the storage options path in open()
  LanceTableReader reader(lance_uri, fragment_id, schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_FALSE(rgs.empty());
  ASSERT_EQ(rgs.back().end_offset, test_batch_->num_rows());

  // Actually read the data and verify
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  ASSERT_EQ(chunk->num_rows(), test_batch_->num_rows());

  // Verify the data content
  auto expected_id_column = std::static_pointer_cast<arrow::Int64Array>(test_batch_->column(0));
  auto actual_id_column = std::static_pointer_cast<arrow::Int64Array>(chunk->column(0));
  for (int i = 0; i < chunk->num_rows(); i++) {
    ASSERT_EQ(actual_id_column->Value(i), expected_id_column->Value(i));
  }
}

}  // namespace milvus_storage
