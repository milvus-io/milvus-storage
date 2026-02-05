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
#include <random>
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

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;        // Lance URI (s3://bucket/path or /tmp/path)
  std::string arrow_base_path_;  // Path for Arrow filesystem operations
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_F(LanceBasicTest, TestBasic) {
  size_t num_of_batches = 10;
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Build Lance URI from relative path
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  auto storage_options = ToLanceStorageOptions(fs_config);

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
  auto storage_options = ToLanceStorageOptions(fs_config);

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

// Test that storage options are correctly passed through writer and reader
TEST_F(LanceBasicTest, TestStorageOptionsIntegration) {
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
