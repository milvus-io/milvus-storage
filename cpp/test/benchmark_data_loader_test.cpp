// Copyright 2026 Zilliz
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

#include "benchmark_data_loader.h"
#include "test_env.h"

#include <gtest/gtest.h>

namespace milvus_storage::benchmark {

TEST(ReaderBenchmarkDataLoaderTest, DefinesApprovedDatasetMatrix) {
  struct Expected {
    ReaderBenchmarkDataset dataset;
    int64_t rows;
    std::vector<std::string> fields;
    std::string_view name;
  };
  const std::vector<Expected> expected = {
      {ReaderBenchmarkDataset::kSyntheticSmall, 4096, {"id", "name", "value", "vector"}, "SyntheticSmall"},
      {ReaderBenchmarkDataset::kSyntheticMedium, 40960, {"id", "name", "value", "vector"}, "SyntheticMedium"},
      {ReaderBenchmarkDataset::kSyntheticLarge, 409600, {"id", "name", "value", "vector"}, "SyntheticLarge"},
      {ReaderBenchmarkDataset::kScalarMedium, 40960, {"id", "name", "value"}, "ScalarMedium"},
      {ReaderBenchmarkDataset::kRandomVector64MiB, 65536, {"vector"}, "RandomVector64MiB"},
      {ReaderBenchmarkDataset::kLowEntropyVector256MiB, 262144, {"vector"}, "LowEntropyVector256MiB"},
      {ReaderBenchmarkDataset::kRandomVector2GiB, 2097152, {"vector"}, "RandomVector2GiB"},
  };
  for (const auto& item : expected) {
    ASSERT_AND_ASSIGN(auto loader, CreateReaderBenchmarkDataLoader(item.dataset));
    ASSERT_STATUS_OK(loader->Load());
    EXPECT_EQ(loader->NumRows(), item.rows);
    ASSERT_EQ(loader->GetSchema()->num_fields(), static_cast<int>(item.fields.size()));
    for (int i = 0; i < loader->GetSchema()->num_fields(); ++i) {
      EXPECT_EQ(loader->GetSchema()->field(i)->name(), item.fields[i]);
    }
    EXPECT_EQ(ReaderBenchmarkDatasetName(item.dataset), item.name);
  }
}

TEST(ReaderBenchmarkDataLoaderTest, StreamsTwoGiBInBoundedBatches) {
  ASSERT_AND_ASSIGN(auto loader, CreateReaderBenchmarkDataLoader(ReaderBenchmarkDataset::kRandomVector2GiB));
  ASSERT_STATUS_OK(loader->Load());
  EXPECT_EQ(loader->GetTable(), nullptr);
  ASSERT_AND_ASSIGN(auto reader, loader->GetRecordBatchReader());
  std::shared_ptr<arrow::RecordBatch> first;
  ASSERT_STATUS_OK(reader->ReadNext(&first));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->column(0)->type_id(), arrow::Type::FIXED_SIZE_BINARY);
  EXPECT_LT(first->num_rows(), loader->NumRows());
  EXPECT_LE(first->num_rows() * 256 * static_cast<int64_t>(sizeof(float)), 64LL * 1024 * 1024);
}

TEST(ReaderBenchmarkDataLoaderTest, FreshReadersRepeatMultipleBoundedBatches) {
  auto loader = CreateStreamingSyntheticDataLoader(
      {10, 8, 0, 4, true, {false, false, false, true}, VectorLayout::kFixedSizeBinary, "multi-batch"});
  ASSERT_STATUS_OK(loader->Load());
  EXPECT_EQ(loader->GetTable(), nullptr);
  ASSERT_AND_ASSIGN(auto left_reader, loader->GetRecordBatchReader());
  ASSERT_AND_ASSIGN(auto right_reader, loader->GetRecordBatchReader());

  std::shared_ptr<arrow::RecordBatch> previous_left_batch;
  for (int batch_index = 0; batch_index < 2; ++batch_index) {
    std::shared_ptr<arrow::RecordBatch> left_batch;
    std::shared_ptr<arrow::RecordBatch> right_batch;
    ASSERT_STATUS_OK(left_reader->ReadNext(&left_batch));
    ASSERT_STATUS_OK(right_reader->ReadNext(&right_batch));
    ASSERT_NE(left_batch, nullptr);
    ASSERT_NE(right_batch, nullptr);
    EXPECT_EQ(left_batch->num_rows(), 4);
    EXPECT_LE(left_batch->num_rows() * 8 * static_cast<int64_t>(sizeof(float)),
              4 * 8 * static_cast<int64_t>(sizeof(float)));
    EXPECT_TRUE(left_batch->Equals(*right_batch));
    if (previous_left_batch) {
      EXPECT_FALSE(left_batch->Equals(*previous_left_batch));
    }
    previous_left_batch = std::move(left_batch);
  }
}

TEST(ReaderBenchmarkDataLoaderTest, PreservesCrtStringPayloads) {
  auto random_loader = CreateStreamingSyntheticDataLoader({40960,
                                                           128,
                                                           128,
                                                           40960,
                                                           true,
                                                           {true, true, true, true},
                                                           VectorLayout::kFixedSizeBinary,
                                                           "synthetic/40960rows/128dim"});
  ASSERT_STATUS_OK(random_loader->Load());
  ASSERT_AND_ASSIGN(auto random_reader, random_loader->GetRecordBatchReader());
  std::shared_ptr<arrow::RecordBatch> random_batch;
  ASSERT_STATUS_OK(random_reader->ReadNext(&random_batch));
  ASSERT_NE(random_batch, nullptr);
  const auto random_names = std::static_pointer_cast<arrow::StringArray>(random_batch->GetColumnByName("name"));
  ASSERT_NE(random_names, nullptr);
  EXPECT_EQ(random_names->GetString(0),
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
            "abcdefghijklmnopqrstuvwx");
  EXPECT_EQ(random_names->GetString(1),
            "bcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
            "abcdefghijklmnopqrstuvwxy");

  auto sequential_loader = CreateStreamingSyntheticDataLoader(
      {2, 0, 128, 2, false, {false, true, false, false}, VectorLayout::kFixedSizeBinary, "sequential"});
  ASSERT_STATUS_OK(sequential_loader->Load());
  ASSERT_AND_ASSIGN(auto sequential_reader, sequential_loader->GetRecordBatchReader());
  std::shared_ptr<arrow::RecordBatch> sequential_batch;
  ASSERT_STATUS_OK(sequential_reader->ReadNext(&sequential_batch));
  ASSERT_NE(sequential_batch, nullptr);
  const auto sequential_names = std::static_pointer_cast<arrow::StringArray>(sequential_batch->GetColumnByName("name"));
  ASSERT_NE(sequential_names, nullptr);
  EXPECT_EQ(sequential_names->GetString(0), "name_0");
  EXPECT_EQ(sequential_names->GetString(1), "name_1");
}

}  // namespace milvus_storage::benchmark
