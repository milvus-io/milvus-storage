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
#include "milvus-storage/properties.h"
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

  arrow::Result<ManifestPtr> CreateSampleManifest(const std::string& dummy_name,
                                                  std::vector<std::string> cols = {"id", "name"}) {
    ManifestPtr manifest = std::make_shared<Manifest>();
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = cols;
    cg1->files = {
        {.path = base_path_ + dummy_name},
    };
    cg1->format = LOON_FORMAT_PARQUET;

    manifest->columnGroups().push_back(cg1);
    return manifest;
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

  arrow::Result<ManifestPtr> CreateSampleManifest(const std::string& dummy_name,
                                                  std::vector<std::string> cols = {"id", "name"}) {
    ManifestPtr manifest = std::make_shared<Manifest>();
    auto cg1 = std::make_shared<ColumnGroup>();
    cg1->columns = cols;
    cg1->files = {{.path = base_path_ + dummy_name}};
    cg1->format = LOON_FORMAT_PARQUET;

    manifest->columnGroups().push_back(cg1);
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
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    ASSERT_EQ(transaction->GetReadVersion(), 0);
    ASSERT_AND_ASSIGN(auto latest_manifest, transaction->GetManifest());
    ASSERT_NE(latest_manifest, nullptr);

    auto reader =
        Reader::create(std::make_shared<ColumnGroups>(latest_manifest->columnGroups()), schema_, nullptr, properties_);
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
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy1.parquet"));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, 1);
  }
}

TEST_F(TransactionTest, AppendFileTest) {
  size_t loop_times = 10;

  for (size_t i = 0; i <= loop_times; ++i) {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest(("/dummy" + std::to_string(i) + ".parquet").c_str()));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, i + 1);

    // read back the latest manifest
    ASSERT_AND_ASSIGN(auto read_transaction, Transaction::Open(fs_, base_path_));

    ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->GetManifest());
    ASSERT_NE(latest_manifest, nullptr);
    ASSERT_EQ(latest_manifest->columnGroups().size(), 1);
    ASSERT_EQ(latest_manifest->columnGroups()[0]->files.size(), i + 1);
  }

  // failed to commit with invalid append files, can't apply
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ManifestPtr manifest = std::make_shared<Manifest>();  // empty manifest
    // Empty manifest - no changes to apply
    ASSERT_STATUS_NOT_OK(transaction->Commit());

    // mismatch columns
    ASSERT_AND_ASSIGN(transaction, Transaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(manifest, CreateSampleManifest("/dummy_invalid.parquet", {"mismatched_col"}));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_STATUS_NOT_OK(transaction->Commit());
  }

  // duplicate paths now allowed in APPENDFILES
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest with a duplicate path
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy0.parquet"));  // duplicate path
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, loop_times + 2);
  }
}

TEST_F(TransactionTest, AddFieldTest) {
  // initial commit with one column group
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy_initial.parquet", {"id", "name"}));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, 1);
  }

  // add field with existing columns should fail
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest with one column group for new field
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy_invalid_field.parquet", {"name", "value"}));
    transaction->AddColumnGroup(manifest->columnGroups()[0]);
    ASSERT_STATUS_NOT_OK(transaction->Commit());

    // the manifest should be unchanged
    ASSERT_AND_ASSIGN(transaction, Transaction::Open(fs_, base_path_));
    ASSERT_AND_ASSIGN(manifest, transaction->GetManifest());
    ASSERT_EQ(manifest->columnGroups().size(), 1);
    ASSERT_EQ(manifest->columnGroups()[0]->files.size(), 1);
  }

  // add field commit
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest with one column group for new field
    ASSERT_AND_ASSIGN(auto new_field_manifest, CreateSampleManifest("/dummy_new_field.parquet", {"value", "vector"}));
    transaction->AddColumnGroup(new_field_manifest->columnGroups()[0]);
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, 2);

    // read back the latest manifest
    {
      ASSERT_AND_ASSIGN(auto read_transaction, Transaction::Open(fs_, base_path_));

      ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->GetManifest());
      ASSERT_NE(latest_manifest, nullptr);
      ASSERT_EQ(latest_manifest->columnGroups().size(), 2);
    }
  }
}

TEST_F(TransactionTest, ConflictResolveTest) {
  // initial commit with one column group
  {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy_initial.parquet", {"id", "name"}));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, 1);
  }

  // simulate concurrent transactions
  ASSERT_AND_ASSIGN(auto transaction1, Transaction::Open(fs_, base_path_));
  ASSERT_AND_ASSIGN(auto transaction2, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));

  // transaction1 appends files and commits
  {
    ASSERT_AND_ASSIGN(auto manifest1, CreateSampleManifest("/dummy_t1.parquet", {"id", "name"}));
    transaction1->AppendFiles(manifest1->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version1, transaction1->Commit());
    ASSERT_EQ(committed_version1, 2);
  }

  // transaction2 tries to append files and resolve conflict by merging
  {
    ASSERT_AND_ASSIGN(auto manifest2, CreateSampleManifest("/dummy_t2.parquet", {"id", "name"}));
    transaction2->AppendFiles(manifest2->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version2, transaction2->Commit());
    ASSERT_EQ(committed_version2, 3);

    // read back the latest manifest
    {
      ASSERT_AND_ASSIGN(auto read_transaction, Transaction::Open(fs_, base_path_));

      ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->GetManifest());
      ASSERT_NE(latest_manifest, nullptr);
      ASSERT_EQ(latest_manifest->columnGroups().size(), 1);
      ASSERT_EQ(latest_manifest->columnGroups()[0]->files.size(), 3);  // initial + t1 + t2
    }
  }
}

TEST_F(TransactionTest, ConflictResolveOverwriteTest) {
  // initial 5 commit with one column group
  for (size_t i = 0; i < 5; ++i) {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy_initial.parquet", {"id", "name"}));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, i + 1);
  }

  // overwrite from 3
  ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_, 3, OverwriteResolver));
  ASSERT_AND_ASSIGN(auto manifest, transaction->GetManifest());
  ASSERT_EQ(manifest->columnGroups().size(), 1);
  ASSERT_EQ(manifest->columnGroups()[0]->files.size(), 3);

  ASSERT_AND_ASSIGN(auto new_manifest, CreateSampleManifest("/dummy_initial.parquet", {"id", "name"}));
  transaction->AppendFiles(new_manifest->columnGroups());
  ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
  ASSERT_EQ(committed_version, 6);

  // read back the latest manifest
  {
    ASSERT_AND_ASSIGN(auto read_transaction, Transaction::Open(fs_, base_path_));

    ASSERT_AND_ASSIGN(auto latest_manifest, read_transaction->GetManifest());
    ASSERT_NE(latest_manifest, nullptr);
    ASSERT_EQ(latest_manifest->columnGroups().size(), 1);
    ASSERT_EQ(latest_manifest->columnGroups()[0]->files.size(), 4);  // initial + 3 from before + 1 new
  }
}

TEST_F(TransactionTest, WriteReadByVersion) {
  // initial 5 commit with one column group
  int loop_times = 5;
  for (size_t i = 0; i < loop_times; ++i) {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_));

    // create a new manifest
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy_initial.parquet", {"id", "name"}));
    transaction->AppendFiles(manifest->columnGroups());
    ASSERT_AND_ASSIGN(auto committed_version, transaction->Commit());
    ASSERT_EQ(committed_version, i + 1);
  }

  for (size_t i = 1; i < loop_times + 1; ++i) {
    ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_, i));
    ASSERT_AND_ASSIGN(auto manifest, transaction->GetManifest());
    ASSERT_EQ(manifest->columnGroups().size(), 1);
    ASSERT_EQ(manifest->columnGroups()[0]->files.size(), i);
  }

  // should fail as only 5 versions exist
  ASSERT_STATUS_NOT_OK(Transaction::Open(fs_, base_path_, 6));

  // valid, will use the latest version
  ASSERT_AND_ASSIGN(auto transaction, Transaction::Open(fs_, base_path_, LATEST));
  ASSERT_AND_ASSIGN(auto manifest, transaction->GetManifest());
  ASSERT_EQ(manifest->columnGroups().size(), 1);
  ASSERT_EQ(manifest->columnGroups()[0]->files.size(), loop_times);
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

  size_t num_transactions = 5;
  std::vector<std::unique_ptr<Transaction>> transactions;
  transactions.resize(num_transactions);
  for (size_t i = 0; i < num_transactions; ++i) {
    ASSERT_AND_ASSIGN(transactions[i], Transaction::Open(fs_, base_path));
  }

  // Use a shared start signal so all threads begin the commit at the same time.
  std::promise<void> start_promise;
  std::shared_future<void> start_signal(start_promise.get_future());

  std::vector<std::thread> threads;
  threads.reserve(num_transactions);
  std::vector<bool> commit_success;
  commit_success.resize(num_transactions);

  for (size_t i = 0; i < num_transactions; ++i) {
    threads.emplace_back([&, i, start_signal]() {
      // wait for the common start signal
      start_signal.wait();
      ASSERT_AND_ASSIGN(auto manifest,
                        CreateSampleManifest(("/dummy_atomic_" + std::to_string(i) + ".parquet").c_str()));
      transactions[i]->AppendFiles(manifest->columnGroups());
      auto arrow_commit_result = transactions[i]->Commit();
      if (arrow_commit_result.ok()) {
        commit_success[i] = true;
      } else {
        commit_success[i] = false;
      }
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
  size_t success_count =
      std::count_if(commit_success.begin(), commit_success.end(), [](bool success) { return success; });

  ASSERT_EQ(success_count, 1) << "Only one transaction should succeed in committing.";
}

INSTANTIATE_TEST_SUITE_P(TransactionAtomicHandlerTestP,
                         TransactionAtomicHandlerTest,
                         ::testing::Values(TRANSACTION_HANDLER_TYPE_UNSAFE, TRANSACTION_HANDLER_TYPE_CONDITIONAL));

// ==================== LOB Files Tests ====================

TEST_F(TransactionTest, AddLobFile) {
  // Create initial transaction to set up manifest
  ASSERT_AND_ASSIGN(auto txn1, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto manifest1, CreateSampleManifest("/dummy1.parquet"));
  txn1->AddColumnGroup(manifest1->columnGroups()[0]);

  // Add LOB file
  LobFileInfo lob1{"lob/field101_001.vortex", 101, 1000, 900, 1048576};
  txn1->AddLobFile(lob1);

  ASSERT_AND_ASSIGN(auto version1, txn1->Commit());
  ASSERT_EQ(version1, 1);

  // Read back and verify
  ASSERT_AND_ASSIGN(auto txn2, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto read_manifest, txn2->GetManifest());

  ASSERT_EQ(read_manifest->lobFiles().size(), 1);
  EXPECT_EQ(read_manifest->lobFiles()[0].path, base_path_ + "/_data/lob/field101_001.vortex");
  EXPECT_EQ(read_manifest->lobFiles()[0].field_id, 101);
  EXPECT_EQ(read_manifest->lobFiles()[0].total_rows, 1000);
  EXPECT_EQ(read_manifest->lobFiles()[0].valid_rows, 900);
  EXPECT_EQ(read_manifest->lobFiles()[0].file_size_bytes, 1048576);
}

TEST_F(TransactionTest, AddMultipleLobFiles) {
  // Create initial transaction
  ASSERT_AND_ASSIGN(auto txn1, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto manifest1, CreateSampleManifest("/dummy1.parquet"));
  txn1->AddColumnGroup(manifest1->columnGroups()[0]);

  // Add multiple LOB files
  txn1->AddLobFile({"lob/field101_001.vortex", 101, 1000, 900, 1048576});
  txn1->AddLobFile({"lob/field101_002.vortex", 101, 2000, 1800, 2097152});
  txn1->AddLobFile({"lob/field102_001.vortex", 102, 500, 450, 524288});

  ASSERT_AND_ASSIGN(auto version1, txn1->Commit());
  ASSERT_EQ(version1, 1);

  // Read back and verify
  ASSERT_AND_ASSIGN(auto txn2, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto read_manifest, txn2->GetManifest());

  ASSERT_EQ(read_manifest->lobFiles().size(), 3);

  // Test getLobFilesForField
  auto field101_files = read_manifest->getLobFilesForField(101);
  ASSERT_EQ(field101_files.size(), 2);

  auto field102_files = read_manifest->getLobFilesForField(102);
  ASSERT_EQ(field102_files.size(), 1);

  auto field999_files = read_manifest->getLobFilesForField(999);
  ASSERT_TRUE(field999_files.empty());
}

TEST_F(TransactionTest, LobFilesPreservedAcrossTransactions) {
  // Transaction 1: Add column groups and LOB files
  {
    ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy1.parquet"));
    txn->AddColumnGroup(manifest->columnGroups()[0]);
    txn->AddLobFile({"lob/field101_001.vortex", 101, 1000, 900, 1048576});
    ASSERT_AND_ASSIGN(auto version, txn->Commit());
    ASSERT_EQ(version, 1);
  }

  // Transaction 2: Add more LOB files
  {
    ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
    ASSERT_AND_ASSIGN(auto manifest, CreateSampleManifest("/dummy2.parquet"));
    txn->AppendFiles(manifest->columnGroups());
    txn->AddLobFile({"lob/field101_002.vortex", 101, 2000, 1800, 2097152});
    ASSERT_AND_ASSIGN(auto version, txn->Commit());
    ASSERT_EQ(version, 2);
  }

  // Verify both LOB files are present
  {
    ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
    ASSERT_AND_ASSIGN(auto manifest, txn->GetManifest());

    ASSERT_EQ(manifest->lobFiles().size(), 2);
    EXPECT_EQ(manifest->lobFiles()[0].total_rows, 1000);
    EXPECT_EQ(manifest->lobFiles()[1].total_rows, 2000);
  }
}

TEST_F(TransactionTest, EmptyLobFiles) {
  // Create transaction without LOB files
  ASSERT_AND_ASSIGN(auto txn1, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto manifest1, CreateSampleManifest("/dummy1.parquet"));
  txn1->AddColumnGroup(manifest1->columnGroups()[0]);

  ASSERT_AND_ASSIGN(auto version1, txn1->Commit());
  ASSERT_EQ(version1, 1);

  // Read back and verify empty LOB files
  ASSERT_AND_ASSIGN(auto txn2, Transaction::Open(fs_, base_path_, LATEST, MergeResolver));
  ASSERT_AND_ASSIGN(auto read_manifest, txn2->GetManifest());

  ASSERT_TRUE(read_manifest->lobFiles().empty());
}

TEST_F(TransactionTest, LobFileInfoEquality) {
  LobFileInfo file1{"path.vortex", 101, 1000, 900, 1048576};
  LobFileInfo file2{"path.vortex", 101, 1000, 900, 1048576};
  LobFileInfo file3{"different.vortex", 101, 1000, 900, 1048576};

  EXPECT_EQ(file1, file2);
  EXPECT_FALSE(file1 == file3);
}

}  // namespace milvus_storage::test