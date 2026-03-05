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

// Only compile this test file when FIU is enabled
#ifdef BUILD_WITH_FIU

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>

#include "test_env.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

static std::once_flag fiu_init_flag;

class FaultInjectionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Initialize FIU once (thread-safe, only on first test run)
    std::call_once(fiu_init_flag, []() { FIU_INIT(); });
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("fiu-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));

    ThreadPoolHolder::WithSingleton(4);
  }

  void TearDown() override {
    // Disable all fault points
    FIU_DISABLE_FAULT(FIUKEY_WRITER_WRITE_FAIL);
    FIU_DISABLE_FAULT(FIUKEY_WRITER_FLUSH_FAIL);
    FIU_DISABLE_FAULT(FIUKEY_WRITER_CLOSE_FAIL);
    FIU_DISABLE_FAULT(FIUKEY_COLUMN_GROUP_READ_FAIL);
    FIU_DISABLE_FAULT(FIUKEY_TAKE_ROWS_FAIL);
    FIU_DISABLE_FAULT(FIUKEY_COLUMN_GROUP_WRITE_FAIL);

    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ThreadPoolHolder::Release();
  }

  // Helper to write test data and return column groups
  arrow::Result<std::shared_ptr<ColumnGroups>> WriteTestData() {
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
    auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);
    ARROW_RETURN_NOT_OK(writer->write(test_batch_));
    return writer->close();
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_F(FaultInjectionTest, WriterWriteFail) {
  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_WRITE_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // First write should fail
  auto status = writer->write(test_batch_);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(status.ToString().find("Injected fault") != std::string::npos);

  // Second write should succeed (FIU_ONETIME exhausted)
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, WriterFlushFail) {
  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_FLUSH_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  ASSERT_STATUS_OK(writer->write(test_batch_));

  // First flush should fail
  auto status = writer->flush();
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(status.ToString().find("Injected fault") != std::string::npos);

  // Second flush should succeed
  ASSERT_STATUS_OK(writer->flush());
  ASSERT_AND_ASSIGN(auto cgs, writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, WriterCloseFail) {
  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_CLOSE_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  ASSERT_STATUS_OK(writer->write(test_batch_));

  // First close should fail
  auto result = writer->close();
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("Injected fault") != std::string::npos);
}

TEST_F(FaultInjectionTest, ColumnGroupReadFail) {
  // First write valid data
  ASSERT_AND_ASSIGN(auto cgs, WriteTestData());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_COLUMN_GROUP_READ_FAIL);

  // get_chunk_reader should succeed, but get_chunk should fail
  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  auto chunk_result = chunk_reader->get_chunk(0);
  ASSERT_FALSE(chunk_result.ok());
  EXPECT_TRUE(chunk_result.status().ToString().find("Injected fault") != std::string::npos);

  // Second attempt should succeed (failnum=1 exhausted)
  ASSERT_AND_ASSIGN(auto chunk, chunk_reader->get_chunk(0));
  EXPECT_GT(chunk->num_rows(), 0);
}

TEST_F(FaultInjectionTest, ColumnGroupReadFailMultiple) {
  // First write valid data
  ASSERT_AND_ASSIGN(auto cgs, WriteTestData());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Enable fault point with failnum=-1 (fail forever)
  FIU_ENABLE_FAULT_ALWAYS(FIUKEY_COLUMN_GROUP_READ_FAIL);

  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  // Multiple reads should all fail
  for (int i = 0; i < 3; ++i) {
    auto chunk_result = chunk_reader->get_chunk(0);
    ASSERT_FALSE(chunk_result.ok());
  }

  // Disable fault point and verify reads succeed
  FIU_DISABLE_FAULT(FIUKEY_COLUMN_GROUP_READ_FAIL);
  ASSERT_AND_ASSIGN(auto chunk, chunk_reader->get_chunk(0));
  EXPECT_GT(chunk->num_rows(), 0);
}

TEST_F(FaultInjectionTest, TakeRowsFail) {
  // First write valid data
  ASSERT_AND_ASSIGN(auto cgs, WriteTestData());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_TAKE_ROWS_FAIL);

  std::vector<int64_t> row_indices = {0, 10, 50};

  // First take should fail
  auto result = reader->take(row_indices);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("Injected fault") != std::string::npos);

  // Second take should succeed
  ASSERT_AND_ASSIGN(auto table, reader->take(row_indices));
  ASSERT_AND_ASSIGN(auto batch, table->CombineChunksToBatch());
  EXPECT_EQ(batch->num_rows(), row_indices.size());
}

TEST_F(FaultInjectionTest, ColumnGroupWriteFail) {
  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_COLUMN_GROUP_WRITE_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_, schema_, std::move(policy), properties_);

  // Write should fail due to column group write failure
  auto status = writer->write(test_batch_);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(status.ToString().find("Injected fault") != std::string::npos);

  // Second write should succeed
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, RecoveryAfterWriterFault) {
  // Test that system recovers properly after writer fault injection
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_WRITE_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_ + "/recovery1", schema_, std::move(policy), properties_);

  // First write fails
  ASSERT_FALSE(writer->write(test_batch_).ok());

  // But we can continue using the writer
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());

  // And data is readable
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
  ASSERT_AND_ASSIGN(auto table, batch_reader->ToTable());
  EXPECT_EQ(table->num_rows(), test_batch_->num_rows());
}

TEST_F(FaultInjectionTest, RecoveryAfterReaderFault) {
  // Write valid data first
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_ + "/recovery2", schema_, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, writer->close());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  // Enable fault
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_COLUMN_GROUP_READ_FAIL);

  // First read fails
  ASSERT_FALSE(chunk_reader->get_chunk(0).ok());

  // But retry succeeds
  ASSERT_AND_ASSIGN(auto chunk, chunk_reader->get_chunk(0));
  EXPECT_EQ(chunk->num_rows(), test_batch_->num_rows());
}

TEST_F(FaultInjectionTest, GetChunksFail) {
  // First write valid data
  ASSERT_AND_ASSIGN(auto cgs, WriteTestData());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  // Enable fault point
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_COLUMN_GROUP_READ_FAIL);

  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  // get_chunks should fail
  std::vector<int64_t> chunk_indices = {0};
  auto chunks_result = chunk_reader->get_chunks(chunk_indices);
  ASSERT_FALSE(chunks_result.ok());
  EXPECT_TRUE(chunks_result.status().ToString().find("Injected fault") != std::string::npos);

  // Second attempt should succeed
  ASSERT_AND_ASSIGN(auto chunks, chunk_reader->get_chunks(chunk_indices));
  EXPECT_EQ(chunks.size(), 1);
  EXPECT_GT(chunks[0]->num_rows(), 0);
}

}  // namespace milvus_storage::test

#endif  // BUILD_WITH_FIU
