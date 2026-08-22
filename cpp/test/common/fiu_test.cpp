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
#include "milvus-storage/packed/writer.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class FaultInjectionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Initialize FIU once (thread-safe, only on first test run)
    ASSERT_EQ(0, InitFiuOnce());
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

  // A failed writer is terminal and must never be retried in place.
  EXPECT_TRUE(writer->write(test_batch_).Equals(status));

  // Retry with a new writer, which also generates new file paths.
  ASSERT_AND_ASSIGN(auto retry_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto retry_writer = Writer::create(base_path_ + "/write-retry", schema_, std::move(retry_policy), properties_);
  ASSERT_STATUS_OK(retry_writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, retry_writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, ScopedFiuFaultDisablesOnScopeExit) {
  {
    ScopedFiuFault fault(FIUKEY_WRITER_WRITE_FAIL, /*one_time=*/true);
    ASSERT_EQ(0, fault.enable_result());
    EXPECT_NE(0, fiu_fail(FIUKEY_WRITER_WRITE_FAIL));
  }

  EXPECT_EQ(0, fiu_fail(FIUKEY_WRITER_WRITE_FAIL));
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

  // The failed instance returns its first failure without doing more work.
  EXPECT_TRUE(writer->flush().Equals(status));

  ASSERT_AND_ASSIGN(auto retry_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto retry_writer = Writer::create(base_path_ + "/flush-retry", schema_, std::move(retry_policy), properties_);
  ASSERT_STATUS_OK(retry_writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, retry_writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, PackedWriterCloseFailureIsTerminal) {
  std::vector<std::string> paths = {base_path_ + "/terminal-packed.parquet"};
  std::vector<int> all_columns;
  for (int column = 0; column < schema_->num_fields(); ++column) {
    all_columns.push_back(column);
  }
  std::vector<std::vector<int>> column_groups = {std::move(all_columns)};
  StorageConfig storage_config;
  ASSERT_AND_ASSIGN(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config, column_groups));
  ASSERT_STATUS_OK(writer->Write(test_batch_));

  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_FLUSH_FAIL);
  auto first_close = writer->Close();
  ASSERT_FALSE(first_close.ok());
  EXPECT_NE(first_close.ToString().find(FIUKEY_WRITER_FLUSH_FAIL), std::string::npos);

  EXPECT_TRUE(writer->Write(test_batch_).Equals(first_close));
  EXPECT_TRUE(writer->AddUserMetadata("key", "value").Equals(first_close));
  auto tell_result = writer->Tell();
  ASSERT_FALSE(tell_result.ok());
  EXPECT_TRUE(tell_result.status().Equals(first_close));
  // Close() returns the stored failure without additional cleanup I/O.
  EXPECT_TRUE(writer->Close().Equals(first_close));
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

  // The failed instance cannot be reused.
  EXPECT_TRUE(writer->write(test_batch_).Equals(status));

  ASSERT_AND_ASSIGN(auto retry_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto retry_writer = Writer::create(base_path_ + "/column-group-retry", schema_, std::move(retry_policy), properties_);
  ASSERT_STATUS_OK(retry_writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, retry_writer->close());
  EXPECT_EQ(cgs->size(), 1);
}

TEST_F(FaultInjectionTest, RecoveryAfterWriterFaultReplacesWriter) {
  // Recovery creates a new writer and therefore a new set of file paths.
  FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_WRITE_FAIL);

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto writer = Writer::create(base_path_ + "/recovery1", schema_, std::move(policy), properties_);

  // First write fails
  ASSERT_FALSE(writer->write(test_batch_).ok());

  ASSERT_AND_ASSIGN(auto retry_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
  auto retry_writer = Writer::create(base_path_ + "/recovery2", schema_, std::move(retry_policy), properties_);
  ASSERT_STATUS_OK(retry_writer->write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgs, retry_writer->close());

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
