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

#include <arrow/array.h>
#include <arrow/testing/gtest_util.h>

#include "include/test_env.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage {
namespace {

class MaskedRecordBatchReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    api::Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("masked-record-batch-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  std::shared_ptr<api::Manifest> WriteManifest() {
    auto policy_result = CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_);
    if (!policy_result.ok()) {
      ADD_FAILURE() << policy_result.status().ToString();
      return nullptr;
    }

    auto writer = api::Writer::create(base_path_, schema_, std::move(policy_result).ValueOrDie(), properties_);
    if (!writer) {
      ADD_FAILURE() << "Writer::create returned nullptr";
      return nullptr;
    }

    auto status = writer->write(test_batch_);
    if (!status.ok()) {
      ADD_FAILURE() << status.ToString();
      return nullptr;
    }

    auto cgs_result = writer->close();
    if (!cgs_result.ok()) {
      ADD_FAILURE() << cgs_result.status().ToString();
      return nullptr;
    }
    return std::make_shared<api::Manifest>(*std::move(cgs_result).ValueOrDie());
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  std::string base_path_;
};

TEST_F(MaskedRecordBatchReaderTest, ColumnGroupsReaderRejectsMaskedRead) {
  auto manifest = WriteManifest();
  auto column_groups = std::make_shared<ColumnGroups>(manifest->columnGroups());
  auto reader = api::Reader::create(column_groups, schema_, nullptr, properties_);

  auto result = reader->get_masked_record_batch_reader(api::AliveReadOptions{});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("manifest-aware"), std::string::npos);
}

TEST_F(MaskedRecordBatchReaderTest, ManifestReaderReturnsAllTrueMaskAndEof) {
  auto manifest = WriteManifest();
  auto reader = api::Reader::create(manifest, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  ASSERT_AND_ASSIGN(auto masked_reader, reader->get_masked_record_batch_reader(api::AliveReadOptions{}));

  api::MaskedRecordBatch masked_batch;
  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  ASSERT_NE(masked_batch.batch, nullptr);
  ASSERT_NE(masked_batch.keep_mask, nullptr);
  EXPECT_EQ(masked_batch.keep_mask->length(), masked_batch.batch->num_rows());
  EXPECT_EQ(masked_batch.batch->num_rows(), test_batch_->num_rows());

  for (int64_t i = 0; i < masked_batch.keep_mask->length(); ++i) {
    EXPECT_TRUE(masked_batch.keep_mask->Value(i));
  }

  ASSERT_OK(masked_reader->ReadNext(&masked_batch));
  EXPECT_EQ(masked_batch.batch, nullptr);
  EXPECT_EQ(masked_batch.keep_mask, nullptr);
}

}  // namespace
}  // namespace milvus_storage
