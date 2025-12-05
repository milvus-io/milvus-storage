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

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/column_group_writer.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class ColumnGroupsWRTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("column-group-writer-reader-test");
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

  arrow::Result<std::unique_ptr<ColumnGroupPolicy>> CreateSinglePolicy(
      const std::string& format, std::shared_ptr<arrow::Schema> schema = nullptr) {
    auto properties = milvus_storage::api::Properties{};
    SetValue(properties, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SINGLE);
    SetValue(properties, PROPERTY_FORMAT, format.c_str());

    return ColumnGroupPolicy::create_column_group_policy(properties, schema ? schema : schema_);
  }

  arrow::Result<std::unique_ptr<ColumnGroupPolicy>> CreateSchemaBasePolicy(
      const std::string& patterns, const std::string& format, std::shared_ptr<arrow::Schema> schema = nullptr) {
    auto properties = milvus_storage::api::Properties{};
    SetValue(properties, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED);
    SetValue(properties, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS, patterns.c_str());
    SetValue(properties, PROPERTY_FORMAT, format.c_str());

    return ColumnGroupPolicy::create_column_group_policy(properties, schema ? schema : schema_);
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_P(ColumnGroupsWRTest, TestGetChunksSliced) {
  std::string format = GetParam();

  // Test writing and reading with parallelism
  int total_rows = 1000000;

  // Create multiple column groups
  std::vector<std::string> patterns = {"id|name|value|vector"};
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns[0], format));

  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write test data in parallel
  for (int i = 0; i < total_rows / 1000; ++i) {
    ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema_, true, 1000, (i % 24) + 1));
    ASSERT_OK(writer->write(batch));
  }

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();

  // Read and verify data
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  std::vector<int64_t> row_indices;
  for (int i = 0; i < total_rows; i += 500) {
    row_indices.emplace_back(i);
  }

  ASSERT_AND_ASSIGN(auto chunk_rows_for_check, chunk_reader->get_chunk_rows());
  ASSERT_AND_ASSIGN(auto chunk_indices, chunk_reader->get_chunk_indices(row_indices));

  // all
  {
    ASSERT_AND_ASSIGN(auto chunks, chunk_reader->get_chunks(chunk_indices, 0 /* parallelism */));
    ASSERT_EQ(chunks.size(), chunk_indices.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto& chunk = chunks[i];
      ASSERT_NE(chunk, nullptr);
      EXPECT_EQ(chunk->num_rows(), chunk_rows_for_check[chunk_indices[i]]);
    }
  }

  // random test
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t random_times = 5;
    for (size_t i = 1; i <= random_times; ++i) {
      std::vector<int64_t> chunkidx_samples;

      float fraction = static_cast<float>(i) / random_times;
      size_t samples_size = static_cast<size_t>(chunk_indices.size() * fraction);

      std::sample(chunk_indices.begin(), chunk_indices.end(), std::back_inserter(chunkidx_samples), samples_size, gen);

      std::cout << "sample indices: ";
      for (int sample_id : chunkidx_samples) std::cout << sample_id << " ";
      std::cout << std::endl;

      ASSERT_AND_ASSIGN(auto chunks, chunk_reader->get_chunks(chunkidx_samples, 0 /* parallelism */));
      ASSERT_EQ(chunks.size(), chunkidx_samples.size());

      for (size_t j = 0; j < chunks.size(); ++j) {
        const auto& chunk = chunks[j];
        ASSERT_NE(chunk, nullptr);
        EXPECT_EQ(chunk->num_rows(), chunk_rows_for_check[chunkidx_samples[j]]);
      }
    }
  }
}

TEST_P(ColumnGroupsWRTest, TestStartEndIndex) {
  std::string format = GetParam();

  ASSERT_AND_ASSIGN(auto two_cols_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, true}));

  // Test writing with SinglePolicy
  // make sure parquet will make new column group for each write
  ASSERT_AND_ASSIGN(
      auto rb_256_rows,
      CreateTestData(two_cols_schema /* schema */, false /* randdata */, 256 /* num_rows */, 1024 /* vector_dim */,
                     0 /* str_length */, std::array<bool, 4>{true, false, false, true} /*needed_columns */));

  std::vector<std::shared_ptr<ColumnGroups>> cgsvec;
  // two files, col0 is [0~25600], col1 always (1024 * 4)B used to fill row groups
  for (int i = 0; i < 2; i++) {
    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, two_cols_schema));
    auto writer = Writer::create(base_path_, two_cols_schema, std::move(policy), properties_);
    ASSERT_NE(writer, nullptr);

    // Write test data
    for (int j = 0; j < 100; ++j) {
      arrow::Int64Builder builder;
      ASSERT_STATUS_OK(builder.Reserve(256));
      for (int k = 0; k < 256; ++k) {
        builder.UnsafeAppend(j * 256 + k);
      }
      std::shared_ptr<arrow::Array> id_array;
      ASSERT_STATUS_OK(builder.Finish(&id_array));

      auto new_batch_result = rb_256_rows->SetColumn(0, arrow::field("id", arrow::int64()), id_array);
      ASSERT_TRUE(new_batch_result.ok());
      ASSERT_STATUS_OK(writer->write(*new_batch_result));
    }

    // Close and get cgs
    auto cgs_result = writer->close();
    ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
    auto cgs = std::move(cgs_result).ValueOrDie();
    cgsvec.emplace_back(cgs);
  }

  // reconstruct column group
  std::shared_ptr<ColumnGroup> file_cg;
  {
    ASSERT_EQ(cgsvec[0]->size(), 1);
    ASSERT_EQ(cgsvec[0]->get_column_group(0)->files.size(), 1);

    ASSERT_EQ(cgsvec[1]->size(), 1);
    ASSERT_EQ(cgsvec[1]->get_column_group(0)->files.size(), 1);
    auto origin_cg0 = cgsvec[0]->get_column_group(0);
    auto origin_cg1 = cgsvec[1]->get_column_group(0);

    file_cg = std::make_shared<ColumnGroup>();
    file_cg->columns = origin_cg0->columns;
    file_cg->format = origin_cg0->format;
    file_cg->files = {
        ColumnGroupFile{
            .path = origin_cg0->files[0].path,
            .start_index = 1000,
            .end_index = 2000,
        },
        ColumnGroupFile{
            .path = origin_cg0->files[0].path,
            .start_index = 2000,
            .end_index = 18789,
        },
        ColumnGroupFile{
            .path = origin_cg1->files[0].path,
            .start_index = 0,
            .end_index = 5000,
        },
        ColumnGroupFile{
            .path = origin_cg1->files[0].path,
            .start_index = 5000,
            .end_index = 20000,
        },
        ColumnGroupFile{
            .path = origin_cg1->files[0].path,
            .start_index = std::nullopt,
            .end_index = std::nullopt,
        },
    };
  }

  // make sure vortex chunk rows is 256
  EXPECT_EQ(SetValue(properties_, PROPERTY_READER_VORTEX_CHUNK_ROWS, "256"), std::nullopt);

  // reader
  ASSERT_AND_ASSIGN(auto chunk_reader, ColumnGroupReader::create(two_cols_schema, file_cg, {"id"}, properties_,
                                                                 nullptr /* key_retriever */));
  auto total_number_of_chunks = chunk_reader->total_number_of_chunks();
  auto total_rows = chunk_reader->total_rows();
  // 25600 + 15000 + 5000 + 16789 + 1000
  EXPECT_EQ(total_rows, 63389);

  std::vector<int64_t> expected_ids;
  expected_ids.reserve(total_rows);
  // 1 & 2. origin_cg0: 1000-18789
  for (int i = 1000; i < 18789; ++i) expected_ids.emplace_back(i);
  // 3 & 4. origin_cg1: 0-20000
  for (int i = 0; i < 20000; ++i) expected_ids.emplace_back(i);
  // 5. origin_cg1: all (0-25600)
  for (int i = 0; i < 25600; ++i) expected_ids.emplace_back(i);

  ASSERT_EQ(expected_ids.size(), total_rows);

  // verification get_chunk
  {
    int64_t current_row = 0;
    for (size_t i = 0; i < total_number_of_chunks; ++i) {
      ASSERT_AND_ASSIGN(auto chunk, chunk_reader->get_chunk(i));
      ASSERT_NE(chunk, nullptr);
      auto id_array = std::static_pointer_cast<arrow::Int64Array>(chunk->column(0));
      for (int64_t j = 0; j < chunk->num_rows(); ++j) {
        ASSERT_EQ(id_array->Value(j), expected_ids[current_row])
            << "Row " << current_row << " mismatch. Chunk " << i << " row " << j;
        current_row++;
      }
    }
  }

  // verification get_chunks
  {
    std::vector<int64_t> chunk_indices;
    int64_t current_row = 0;
    for (size_t i = 0; i < total_number_of_chunks; ++i) {
      chunk_indices.push_back(i);
    }
    ASSERT_AND_ASSIGN(auto chunks, chunk_reader->get_chunks(chunk_indices));
    size_t total_rows = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      total_rows += chunks[i]->num_rows();
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto& chunk = chunks[i];
      ASSERT_NE(chunk, nullptr);
      auto id_array = std::static_pointer_cast<arrow::Int64Array>(chunk->column(0));
      for (int64_t j = 0; j < chunk->num_rows(); ++j) {
        ASSERT_EQ(id_array->Value(j), expected_ids[current_row])
            << "Row " << current_row << " mismatch. Chunk " << i << " row " << j;
        current_row++;
      }
    }

    ASSERT_EQ(current_row, total_rows);
  }
}

INSTANTIATE_TEST_SUITE_P(ColumnGroupsWRTestP,
                         ColumnGroupsWRTest,
#ifdef BUILD_VORTEX_BRIDGE
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX)
#else
                         ::testing::Values(LOON_FORMAT_PARQUET)
#endif
);

}  // namespace milvus_storage::test
