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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <future>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/api.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/testing/gtest_util.h>

#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "test_env.h"

namespace milvus_storage::test {
using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

class TransactionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Clean up test directory
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("transaction-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    schema_ = arrow::schema(
        {arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})),
         arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"101"})),
         arrow::field("value", arrow::float64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"102"})),
         arrow::field("vector", arrow::list(arrow::float32()), false,
                      arrow::key_value_metadata({"PARQUET:field_id"}, {"103"}))});
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
  }

  ManifestPtr CreateSampleManifest(const std::string& dummy_name, std::vector<std::string> cols = {"id", "name"}) {
    ManifestPtr manifest = std::make_shared<Manifest>();
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = cols;
    cg1->files = {
        {.path = base_path_ + dummy_name},
    };
    cg1->format = LOON_FORMAT_PARQUET;

    manifest->add_column_group(cg1);
    return manifest;
  }

  void VerifyLatestReadVersion(int64_t expected_version) {
    auto read_transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);

    ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->get_latest_manifest());
    ASSERT_EQ(read_transaction->read_version(), expected_version);
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  api::Properties properties_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
};

class TransactionAtomicHandlerTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("transaction-test-atomic-handler");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
  }

  ManifestPtr CreateSampleManifest(const std::string& dummy_name, std::vector<std::string> cols = {"id", "name"}) {
    ManifestPtr manifest = std::make_shared<Manifest>();
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = cols;
    cg1->files = {{.path = base_path_ + dummy_name}};
    cg1->format = LOON_FORMAT_PARQUET;

    manifest->add_column_group(cg1);
    return manifest;
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  api::Properties properties_;
  std::string base_path_;
};

TEST_F(TransactionTest, EmptyManifestTest) {
  // read latest manifest with empty directory
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_INIT);
    ASSERT_AND_ASSIGN(auto latest_manifest, transaction->get_latest_manifest());
    ASSERT_NE(latest_manifest, nullptr);
    ASSERT_EQ(transaction->read_version(), 0);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_READ);

    auto reader = Reader::create(latest_manifest, schema_, nullptr, properties_);
    ASSERT_NE(reader, nullptr);

    // Test get_record_batch_reader with empty manifest
    {
      ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());

      std::shared_ptr<arrow::RecordBatch> batch;
      ASSERT_OK(batch_reader->ReadNext(&batch));
      ASSERT_EQ(batch, nullptr);
    }

    // Test get_record_batch_reader with empty manifest
    {
      auto chunk_reader_result = reader->get_chunk_reader(0);
      ASSERT_STATUS_NOT_OK(chunk_reader_result.status());
    }
  }

  // write latest manifest with empty directory
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_INIT);
    ASSERT_OK(transaction->begin());
    ASSERT_EQ(transaction->read_version(), 0);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_BEGIN);

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest("/dummy1.parquet");
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(1);
  }
}

TEST_F(TransactionTest, AppendFileTest) {
  size_t loop_times = 10;
  int64_t last_valid_read_version = 0;

  for (size_t i = 0; i <= loop_times; ++i) {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_INIT);
    ASSERT_OK(transaction->begin());
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_BEGIN);

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest(("/dummy" + std::to_string(i) + ".parquet").c_str());
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(commit_result.read_version, i);
    ASSERT_EQ(commit_result.committed_version, i + 1);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    auto read_transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);

    ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->get_latest_manifest());
    ASSERT_EQ(read_transaction->read_version(), i + 1);
    ASSERT_NE(latest_manifest, nullptr);
    ASSERT_EQ(latest_manifest->size(), 1);
    ASSERT_EQ(latest_manifest->get_column_group(0)->files.size(), i + 1);

    last_valid_read_version = read_transaction->read_version();
  }

  // failed to commit with invalid append files, can't apply
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest = std::make_shared<Manifest>();  // empty manifest
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_FALSE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_ABORTED);

    // mismatch columns
    transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());
    manifest = CreateSampleManifest("/dummy_invalid.parquet", {"mismatched_col"});
    ASSERT_AND_ASSIGN(commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_FALSE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_ABORTED);

    // the manifest should be unchanged
    VerifyLatestReadVersion(last_valid_read_version);
  }

  // try abort
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());
    ASSERT_STATUS_OK(transaction->abort());
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_ABORTED);

    // the manifest should be unchanged
    VerifyLatestReadVersion(last_valid_read_version);
  }

  // duplicate paths now allowed in APPENDFILES
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest =
        CreateSampleManifest("/dummy" + std::to_string(last_valid_read_version - 1) + ".parquet");  // duplicate path
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // the manifest should be changed
    VerifyLatestReadVersion(last_valid_read_version + 1);
  }
}

TEST_F(TransactionTest, AddFieldTest) {
  // initial commit with one column group
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest("/dummy_initial.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(1);
  }

  // add field commit
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest with one column group for new field
    ManifestPtr new_field_manifest = CreateSampleManifest("/dummy_new_field.parquet", {"value", "vector"});
    ASSERT_AND_ASSIGN(auto commit_result, transaction->commit(new_field_manifest, UpdateType::ADDFIELD,
                                                              TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    {
      auto read_transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);

      ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->get_latest_manifest());
      ASSERT_EQ(read_transaction->read_version(), 2);
      ASSERT_NE(latest_manifest, nullptr);
      ASSERT_EQ(latest_manifest->size(), 2);
    }
  }

  // add field with existing columns should fail
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest with one column group for new field
    ManifestPtr new_field_manifest = CreateSampleManifest("/dummy_invalid_field.parquet", {"name", "value"});
    ASSERT_AND_ASSIGN(auto commit_result, transaction->commit(new_field_manifest, UpdateType::ADDFIELD,
                                                              TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_FALSE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_ABORTED);

    // the manifest should be unchanged
    VerifyLatestReadVersion(2);
  }
}

TEST_F(TransactionTest, ConflictResolveTest) {
  // initial commit with one column group
  {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest("/dummy_initial.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(1);
  }

  // simulate concurrent transactions
  auto transaction1 = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
  ASSERT_OK(transaction1->begin());

  auto transaction2 = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
  ASSERT_OK(transaction2->begin());

  // transaction1 appends files and commits
  {
    ManifestPtr manifest1 = CreateSampleManifest("/dummy_t1.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result1,
                      transaction1->commit(manifest1, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result1.success);
    ASSERT_EQ(transaction1->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(2);
  }

  // transaction2 tries to append files and resolve conflict by merging
  {
    ManifestPtr manifest2 = CreateSampleManifest("/dummy_t2.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result2,
                      transaction2->commit(manifest2, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_MERGE));
    ASSERT_TRUE(commit_result2.success);
    ASSERT_EQ(transaction2->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    {
      auto read_transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);

      ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->get_latest_manifest());
      ASSERT_EQ(read_transaction->read_version(), 3);
      ASSERT_NE(latest_manifest, nullptr);
      ASSERT_EQ(latest_manifest->size(), 1);
      ASSERT_EQ(latest_manifest->get_column_group(0)->files.size(), 3);  // initial + t1 + t2
    }
  }
}

TEST_F(TransactionTest, ConflictResolveOverwriteTest) {
  // initial 5 commit with one column group
  for (size_t i = 0; i < 5; ++i) {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest("/dummy_initial.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(commit_result.read_version, i);
    ASSERT_EQ(commit_result.committed_version, i + 1);
    ASSERT_TRUE(commit_result.failed_message.empty());
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(i + 1);
  }

  // overwrite from 3
  auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
  ASSERT_OK(transaction->begin(3));
  ASSERT_AND_ASSIGN(auto manifest, transaction->get_current_manifest());
  ASSERT_EQ(manifest->size(), 1);
  ASSERT_EQ(manifest->get_column_group(0)->files.size(), 3);

  ManifestPtr new_manifest = CreateSampleManifest("/dummy_initial.parquet", {"id", "name"});
  ASSERT_AND_ASSIGN(auto commit_result, transaction->commit(new_manifest, UpdateType::APPENDFILES,
                                                            TransResolveStrategy::RESOLVE_OVERWRITE));
  ASSERT_TRUE(commit_result.success);
  ASSERT_EQ(commit_result.read_version, 3);
  ASSERT_EQ(commit_result.committed_version, 6);
  ASSERT_TRUE(commit_result.failed_message.empty());
  ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

  // read back the latest manifest
  {
    auto read_transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);

    ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->get_latest_manifest());
    ASSERT_EQ(read_transaction->read_version(), 6);
    ASSERT_NE(latest_manifest, nullptr);
    ASSERT_EQ(latest_manifest->size(), 1);
    ASSERT_EQ(latest_manifest->get_column_group(0)->files.size(), 4);  // initial + 3 from before + 1 new
  }
}

TEST_F(TransactionTest, WriteReadByVersion) {
  // initial 5 commit with one column group
  int loop_times = 5;
  for (size_t i = 0; i < loop_times; ++i) {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_OK(transaction->begin());

    // create a new manifest
    ManifestPtr manifest = CreateSampleManifest("/dummy_initial.parquet", {"id", "name"});
    ASSERT_AND_ASSIGN(auto commit_result,
                      transaction->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL));
    ASSERT_TRUE(commit_result.success);
    ASSERT_EQ(commit_result.read_version, i);
    ASSERT_EQ(commit_result.committed_version, i + 1);
    ASSERT_TRUE(commit_result.failed_message.empty());
    ASSERT_EQ(transaction->status(), TransStatus::STATUS_COMMITTED);

    // read back the latest manifest
    VerifyLatestReadVersion(i + 1);
  }

  for (size_t i = 1; i < loop_times + 1; ++i) {
    auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
    ASSERT_STATUS_OK(transaction->begin(i));
    ASSERT_AND_ASSIGN(auto manifest, transaction->get_current_manifest());
    ASSERT_EQ(manifest->size(), 1);
    ASSERT_EQ(manifest->get_column_group(0)->files.size(), i);
  }

  auto transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
  ASSERT_STATUS_NOT_OK(transaction->begin(6));  // should failed as only 5 versions exist

  transaction = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path_);
  // valid, will use the lastest version
  ASSERT_STATUS_OK(transaction->begin(-1));
}

TEST_P(TransactionAtomicHandlerTest, testConcurrentCommits) {
  std::string handler_type = GetParam();
  std::string base_path = base_path_;

  if (handler_type == TRANSACTION_HANDLER_TYPE_CONDITIONAL) {
    // TODO: we need a global remote env for CI test
    // no need check the env in each test case
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type != "remote") {
      GTEST_SKIP() << "Conditional Atomic Handler only supported on S3";
    }
    auto bucket_name = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    if (bucket_name.empty()) {
      GTEST_SKIP() << "ENV_VAR_BUCKET_NAME is not set";
    }
    base_path = bucket_name;
  }

  ASSERT_EQ(api::SetValue(properties_, PROPERTY_TRANSACTION_HANDLER_TYPE, handler_type.c_str()), std::nullopt);

  size_t num_transactions = 5;
  std::vector<std::shared_ptr<TransactionImpl<Manifest>>> transactions;
  transactions.resize(num_transactions);
  for (size_t i = 0; i < num_transactions; ++i) {
    transactions[i] = std::make_shared<TransactionImpl<Manifest>>(properties_, base_path);
    ASSERT_OK(transactions[i]->begin());
  }

  // Use a shared start signal so all threads begin the commit at the same time.
  std::promise<void> start_promise;
  std::shared_future<void> start_signal(start_promise.get_future());

  std::vector<std::thread> threads;
  threads.reserve(num_transactions);
  std::vector<CommitResult> commit_results;

  for (size_t i = 0; i < num_transactions; ++i) {
    threads.emplace_back([&, i, start_signal]() {
      // wait for the common start signal
      start_signal.wait();
      ManifestPtr manifest = CreateSampleManifest(("/dummy_atomic_" + std::to_string(i) + ".parquet").c_str());
      auto arrow_commit_result =
          transactions[i]->commit(manifest, UpdateType::APPENDFILES, TransResolveStrategy::RESOLVE_FAIL);
      ASSERT_AND_ASSIGN(auto commit_result, arrow_commit_result);
      std::cout << "Transaction " << i << " commit result: success=" << commit_result.success
                << ", read_version=" << commit_result.read_version
                << ", committed_version=" << commit_result.committed_version
                << ", failed_message=" << commit_result.failed_message << std::endl;
      commit_results.emplace_back(commit_result);
    });
  }

  // Release all threads to run concurrently
  start_promise.set_value();

  // Join threads
  for (auto& t : threads) {
    if (t.joinable())
      t.join();
  }

  // Verify that only one transaction succeeded
  size_t success_count = std::count_if(commit_results.begin(), commit_results.end(),
                                       [](const CommitResult& result) { return result.success; });

  ASSERT_EQ(success_count, 1) << "Only one transaction should succeed in committing.";
}

INSTANTIATE_TEST_SUITE_P(TransactionAtomicHandlerTestP,
                         TransactionAtomicHandlerTest,
                         ::testing::Values(TRANSACTION_HANDLER_TYPE_UNSAFE, TRANSACTION_HANDLER_TYPE_CONDITIONAL));

}  // namespace milvus_storage::test