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

#include <filesystem>
#include <numeric>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <fmt/format.h>
#include <folly/json.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/paimon/paimon_common.h"
#include "milvus-storage/format/paimon/paimon_format_reader.h"
#include "milvus-storage/properties.h"
#include "paimon_bridge.h"
#include "test_env.h"

namespace milvus_storage::test {
namespace {

class PaimonIntegrationTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (IsCloudEnv()) {
      GTEST_SKIP() << "Paimon integration fixtures require a local filesystem";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_EQ(api::SetValue(properties_, PROPERTY_FS_ROOT_PATH, "/"), std::nullopt);
    ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "4"), std::nullopt);
    FilesystemCache::getInstance().clean();
    table_dir_ = "/tmp/milvus-storage-paimon-integration";
    std::filesystem::remove_all(table_dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(table_dir_);
    FilesystemCache::getInstance().clean();
  }

  arrow::Result<std::vector<api::ColumnGroupFile>> Explore(const std::string& mode) {
    if (auto error = api::SetValue(properties_, PROPERTY_PAIMON_SCAN_MODE, mode.c_str()); error) {
      return arrow::Status::Invalid(*error);
    }
    ARROW_ASSIGN_OR_RAISE(auto* format, Format::get(LOON_FORMAT_PAIMON_TABLE));
    return format->explore(table_dir_, properties_);
  }

  api::Properties properties_;
  std::string table_dir_;
};

std::string ReadPath(const api::ColumnGroupFile& file) {
  return folly::parseJson(file.Get<std::string>(api::kPropertyMetadata))["read_path"].asString();
}

int64_t ReadAllRows(const std::shared_ptr<FormatReader>& reader) {
  auto infos = reader->get_row_group_infos().ValueOrDie();
  int64_t rows = 0;
  for (size_t index = 0; index < infos.size(); ++index) {
    rows += reader->get_chunk(static_cast<int>(index)).ValueOrDie()->num_rows();
  }
  return rows;
}

TEST_F(PaimonIntegrationTest, AutoUsesDirectFileForAppendParquet) {
  constexpr uint64_t kRows = 12;
  paimon::CreateTestTable(table_dir_, kRows, "append");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  for (const auto& file : files) {
    EXPECT_EQ(ReadPath(file), "direct-file");
    EXPECT_GT(file.Get<uint64_t>(api::kPropertyFileSize), 0);
  }

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), files.front().end_index);
}

TEST_F(PaimonIntegrationTest, AutoUsesDirectFileForAppendVortex) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append", {}, {}, "vortex");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(ReadPath(files.front()), "direct-file");
  EXPECT_GT(files.front().Get<uint64_t>(api::kPropertyFileSize), 0);
  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(descriptor["data_format"].asString(), "vortex");

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));

  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 4, 16}));
  ASSERT_EQ(taken->num_rows(), 3);
  const std::vector<int32_t> expected = {0, 4, 16};
  for (int64_t row = 0; row < taken->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, taken->column(0)->GetScalar(row));
    ASSERT_NE(std::dynamic_pointer_cast<arrow::Int32Scalar>(scalar), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(scalar)->value, expected[row]);
  }

  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(3, 9));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range.get()));
  ASSERT_EQ(range_table->num_rows(), 6);
  ASSERT_AND_ASSIGN(auto first, range_table->column(0)->GetScalar(0));
  ASSERT_AND_ASSIGN(auto last, range_table->column(0)->GetScalar(5));
  EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(first)->value, 3);
  EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(last)->value, 8);

  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  EXPECT_EQ(ReadAllRows(clone), static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, VortexDeletionVectorUsesDirectFile) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9}, {}, "vortex");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(ReadPath(files.front()), "direct-file");
  EXPECT_EQ(files.front().end_index, 7);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 1, 4, 6}));
  ASSERT_EQ(taken->num_rows(), 4);
  const std::vector<int32_t> expected = {0, 2, 6, 8};
  for (int64_t row = 0; row < taken->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, taken->column(0)->GetScalar(row));
    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(scalar)->value, expected[row]);
  }
}

TEST_F(PaimonIntegrationTest, RelativeVortexPathUsesFilesystemRoot) {
  constexpr uint64_t kRows = 8;
  paimon::CreateTestTable(table_dir_, kRows, "append", {}, {}, "vortex");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(files.front().path.rfind("/", 0), 0);

  files.front().path.erase(0, 1);
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, ReadsSpecifiedSnapshot) {
  constexpr uint64_t kRows = 9;
  auto table = paimon::CreateTestTable(table_dir_, kRows, "append");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(table.snapshot_id).c_str()),
            std::nullopt);

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  EXPECT_EQ(files.front().end_index, static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, ForcedDataSplitStreamsSequentialChunksAndTake) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append");

  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));

  ASSERT_AND_ASSIGN(auto table, reader->take({0, 4, 16}));
  ASSERT_EQ(table->num_rows(), 3);
  const std::vector<int32_t> expected = {0, 4, 16};
  for (int64_t row = 0; row < table->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto value, table->column(0)->GetScalar(row));
    auto id = std::dynamic_pointer_cast<arrow::Int32Scalar>(value);
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->value, expected[row]);
  }

  ASSERT_AND_ASSIGN(auto projected, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                         {"name"}, nullptr));
  ASSERT_AND_ASSIGN(auto clone, projected->clone_reader());
  ASSERT_NE(std::dynamic_pointer_cast<paimon::PaimonFormatReader>(clone), nullptr);
  ASSERT_AND_ASSIGN(auto clone_batch, clone->get_chunk(0));
  ASSERT_EQ(clone_batch->num_columns(), 1);
  EXPECT_EQ(clone_batch->schema()->field(0)->name(), "name");
}

TEST_F(PaimonIntegrationTest, AutoUsesDataSplitForMergeOnRead) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "mor");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, AutoUsesDataSplitForVortexMergeOnRead) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "mor", {}, {}, "vortex");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, MalformedMetadataTypesFailClosed) {
  paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["record_count"] = "ten";
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid());
  EXPECT_NE(reader.status().ToString().find("field types"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, OversizedMetadataFailsBeforeJsonParsing) {
  paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  files.front().Set(api::kPropertyMetadata, std::string(12 * 1024 * 1024 + 1, 'x'));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid());
  EXPECT_NE(reader.status().ToString().find("metadata is too large"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, MalformedDeletionMetadataTypesFailClosed) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["deletion_file"]["offset"] = "zero";
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid());
  EXPECT_NE(reader.status().ToString().find("deletion_file has invalid field types"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, AutoUsesDirectFileAndAppliesDeletionVector) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9});

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");
  ASSERT_EQ(files.front().end_index, 7);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), 7);

  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 1, 4, 6}));
  ASSERT_EQ(taken->num_rows(), 4);
  auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(taken->column(0)->chunk(0));
  ASSERT_NE(ids, nullptr);
  EXPECT_EQ(ids->Value(0), 0);
  EXPECT_EQ(ids->Value(1), 2);
  EXPECT_EQ(ids->Value(2), 6);
  EXPECT_EQ(ids->Value(3), 8);

  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(1, 5));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range.get()));
  ASSERT_EQ(range_table->num_rows(), 4);
  auto range_ids = std::dynamic_pointer_cast<arrow::Int32Array>(range_table->column(0)->chunk(0));
  ASSERT_NE(range_ids, nullptr);
  EXPECT_EQ(range_ids->Value(0), 2);
  EXPECT_EQ(range_ids->Value(1), 3);
  EXPECT_EQ(range_ids->Value(2), 4);
  EXPECT_EQ(range_ids->Value(3), 6);

  auto tampered = files.front();
  auto descriptor = folly::parseJson(tampered.Get<std::string>(api::kPropertyMetadata));
  descriptor["deletion_file"]["cardinality"] = 99;
  tampered.Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto invalid = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, tampered, properties_, {"id"}, nullptr);
  ASSERT_FALSE(invalid.ok());
  EXPECT_TRUE(invalid.status().IsIOError());
}

TEST_F(PaimonIntegrationTest, DirectFileFragmentRangeUsesPostDeletionLogicalRows) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");

  auto fragment = files.front();
  fragment.start_index = 1;
  fragment.end_index = 5;
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = {std::move(fragment)};
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr));
  ASSERT_EQ(reader->total_rows(), 4);

  std::vector<int32_t> ids;
  for (size_t chunk = 0; chunk < reader->total_number_of_chunks(); ++chunk) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(static_cast<int64_t>(chunk)));
    auto values = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(0));
    ASSERT_NE(values, nullptr);
    for (int64_t row = 0; row < values->length(); ++row) {
      ids.push_back(values->Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3, 4, 6}));
}

TEST_F(PaimonIntegrationTest, DataSplitChunksSpanMergeReadBatches) {
  // A logical chunk spans many merge-read batches (1024 rows each).
  constexpr uint64_t kRows = 8192 + 1024;
  paimon::CreateTestTable(table_dir_, kRows, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "8192"), std::nullopt);

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = {files.front()};
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr));
  ASSERT_EQ(reader->total_rows(), static_cast<int64_t>(kRows));

  std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
  std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 2));
  ASSERT_EQ(batches.size(), chunk_indices.size());
  int64_t next_id = 0;
  for (const auto& batch : batches) {
    auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(0));
    ASSERT_NE(ids, nullptr);
    for (int64_t row = 0; row < ids->length(); ++row) {
      ASSERT_EQ(ids->Value(row), static_cast<int32_t>(next_id++));
    }
  }
  EXPECT_EQ(next_id, static_cast<int64_t>(kRows));

  auto tasks = api::ChunkTask::Build(chunk_indices, [&reader](int64_t chunk_index) -> const api::ChunkInfo& {
    return reader->get_chunk_info(chunk_index);
  });
  ASSERT_EQ(tasks.size(), 1);
  ASSERT_AND_ASSIGN(auto async_batches, std::move(reader->get_chunks_async(tasks.front())).get());
  ASSERT_EQ(async_batches.size(), chunk_indices.size());
  next_id = 0;
  for (const auto& batch : async_batches) {
    auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(0));
    ASSERT_NE(ids, nullptr);
    for (int64_t row = 0; row < ids->length(); ++row) {
      ASSERT_EQ(ids->Value(row), static_cast<int32_t>(next_id++));
    }
  }
  EXPECT_EQ(next_id, static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, ExplicitDirectFileRejectsMergeOnRead) {
  paimon::CreateTestTable(table_dir_, 10, "mor");
  auto files = Explore("direct-file");
  ASSERT_FALSE(files.ok());
  EXPECT_NE(files.status().ToString().find("cannot use direct-file"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, CorruptDataSplitDescriptorFailsClosed) {
  paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["payload_sha256"] = "bad";
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  // Descriptor corruption is terminal input state, not a retryable storage
  // failure; the Rust bridge marks it explicitly for the C++ boundary.
  EXPECT_TRUE(reader.status().IsInvalid());
  EXPECT_NE(reader.status().ToString().find("checksum"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, TruncatedDataSplitRangeFailsClosed) {
  paimon::CreateTestTable(table_dir_, 10, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["record_count"] = 12;
  files.front().end_index = 12;
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(0, 12));
  auto table = arrow::Table::FromRecordBatchReader(range.get());
  ASSERT_FALSE(table.ok());
  EXPECT_TRUE(table.status().IsInvalid());
  EXPECT_NE(table.status().ToString().find("requested range was complete"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, DataSplitWithExtraRowsFailsAtDeclaredEnd) {
  paimon::CreateTestTable(table_dir_, 10, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["record_count"] = 8;
  files.front().end_index = 8;
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto first, reader->get_chunk(0));
  ASSERT_EQ(first->num_rows(), 4);
  auto last = reader->get_chunk(1);
  ASSERT_FALSE(last.ok());
  EXPECT_TRUE(last.status().IsInvalid());
  EXPECT_NE(last.status().ToString().find("more rows than its declared row count"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, FullyDeletedTableProducesNoEntries) {
  constexpr uint64_t kRows = 6;
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {0, 1, 2, 3, 4, 5});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  EXPECT_TRUE(files.empty());
}

TEST_F(PaimonIntegrationTest, MissingTableFailsAndWriterIsReadOnly) {
  table_dir_ += "-missing";
  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());

  ASSERT_AND_ASSIGN(auto* format, Format::get(LOON_FORMAT_PAIMON_TABLE));
  auto writer = format->create_writer(nullptr, arrow::schema({arrow::field("id", arrow::int32())}), "unused", "unused",
                                      properties_);
  ASSERT_FALSE(writer.ok());
  EXPECT_TRUE(writer.status().IsNotImplemented());
}

TEST(PaimonErrorClassification, MarkersMapToTerminalStatuses) {
  const std::runtime_error expired(
      "[paimon:error=invalid] Paimon snapshot 3 no longer exists for table file:///t (earliest=5, latest=9); refresh "
      "the external collection");
  EXPECT_TRUE(paimon::ClassifyPaimonError("plan", expired).IsInvalid());

  const std::runtime_error corrupt(
      "[paimon:error=invalid] invalid Paimon split descriptor (missing field); refresh the external table");
  EXPECT_TRUE(paimon::ClassifyPaimonError("open", corrupt).IsInvalid());

  const std::runtime_error unsupported(
      "[paimon:error=not-implemented] Paimon bitmap64 deletion vectors are not supported yet");
  EXPECT_TRUE(paimon::ClassifyPaimonError("read", unsupported).IsNotImplemented());

  // Unmarked messages are never promoted to a terminal class.
  const std::runtime_error transient("connection reset by peer while reading snapshot-9");
  EXPECT_TRUE(paimon::ClassifyPaimonError("plan", transient).IsIOError());
}

TEST_F(PaimonIntegrationTest, MissingPinnedSnapshotFailsPlanAsInvalidWithBounds) {
  auto table = paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(table.snapshot_id + 1000).c_str()),
            std::nullopt);

  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsInvalid()) << files.status().ToString();
  const auto message = files.status().ToString();
  EXPECT_NE(message.find("no longer exists"), std::string::npos) << message;
  EXPECT_NE(message.find("earliest="), std::string::npos) << message;
  EXPECT_NE(message.find(fmt::format("latest={}", table.snapshot_id)), std::string::npos) << message;
  EXPECT_NE(message.find("refresh the external collection"), std::string::npos) << message;
}

TEST_F(PaimonIntegrationTest, ExpiredSnapshotOnDataSplitReadFailsAsInvalid) {
  auto table = paimon::CreateTestTable(table_dir_, 10, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);

  // Expire the pinned snapshot after planning: the descriptor still binds it.
  for (auto snapshot_id : table.snapshot_ids) {
    std::filesystem::remove(fmt::format("{}/snapshot/snapshot-{}", table_dir_, snapshot_id));
  }
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid()) << reader.status().ToString();
  EXPECT_NE(reader.status().ToString().find("refresh the external collection"), std::string::npos)
      << reader.status().ToString();
}

TEST_F(PaimonIntegrationTest, DataSplitRowGroupsUseSampledMemorySizes) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(descriptor.count("estimated_bytes"), 0);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  ASSERT_AND_ASSIGN(auto infos, reader->get_row_group_infos());
  ASSERT_FALSE(infos.empty());
  // The width comes from measuring the first merge-read batch, so it tracks
  // the real decoded row (short strings), not the conservative schema
  // estimate that charges every utf8 column 32 bytes.
  constexpr size_t kSchemaEstimatedRowWidth = 4 + 32 + 8;
  const size_t sampled_width = infos.front().memory_size / (infos.front().end_offset - infos.front().start_offset);
  EXPECT_GT(sampled_width, 0u);
  EXPECT_NE(sampled_width, kSchemaEstimatedRowWidth);
  EXPECT_LT(sampled_width, kSchemaEstimatedRowWidth);
  size_t total = 0;
  for (const auto& info : infos) {
    const auto rows = info.end_offset - info.start_offset;
    EXPECT_EQ(info.memory_size, rows * sampled_width) << info.ToString();
    total += info.memory_size;
  }
  EXPECT_EQ(total, kRows * sampled_width);
}

TEST_F(PaimonIntegrationTest, DataSplitChunksCarrySampledColumnMemorySizes) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "mor");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name", "value"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = {files.front()};
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8()),
                               arrow::field("value", arrow::float64())});
  ASSERT_AND_ASSIGN(
      auto reader, api::ColumnGroupReader::create(schema, column_group, {"id", "name", "value"}, properties_, nullptr));
  ASSERT_GT(reader->total_number_of_chunks(), 0u);

  // Per-column estimates used to be unavailable on the data-split path: the
  // sampled row width was a single scalar, so the public API returned
  // NotImplemented. They now come from the same sampled batch.
  ASSERT_AND_ASSIGN(auto chunk_size, reader->get_chunk_estimated_size(0));
  uint64_t column_total = 0;
  for (int col = 0; col < 3; ++col) {
    ASSERT_AND_ASSIGN(auto column_size, reader->get_chunk_column_estimated_size(0, col));
    EXPECT_GT(column_size, 0u) << "column " << col;
    column_total += column_size;
  }
  // Distribution keeps the parts summing to the chunk estimate.
  EXPECT_EQ(column_total, chunk_size);
  // A 4-byte int32 must not be charged more than the "row_N" string column.
  ASSERT_AND_ASSIGN(auto id_size, reader->get_chunk_column_estimated_size(0, 0));
  ASSERT_AND_ASSIGN(auto name_size, reader->get_chunk_column_estimated_size(0, 1));
  EXPECT_LT(id_size, name_size);
}

TEST_F(PaimonIntegrationTest, VortexRowGroupsUseDecodedSchemaMemorySizes) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append", {}, {}, "vortex");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");

  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(descriptor.count("estimated_bytes"), 0);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  ASSERT_AND_ASSIGN(auto infos, reader->get_row_group_infos());
  ASSERT_FALSE(infos.empty());
  size_t total_memory = 0;
  for (const auto& info : infos) {
    const auto rows = info.end_offset - info.start_offset;
    EXPECT_GT(info.memory_size, rows) << info.ToString();
    total_memory += info.memory_size;
  }
  EXPECT_GT(total_memory, kRows);
}

TEST_F(PaimonIntegrationTest, DataSplitSequentialReadsShareHandleAndCountRestarts) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append");
  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);

  const auto reader_opens = paimon::GetPaimonDataSplitReaderOpenCount();
  const auto stream_opens = paimon::GetPaimonDataSplitStreamOpenCount();

  // One create = one bridge handle (metadata load, which also opens one
  // short-lived stream to sample the row width) and one read stream from
  // open(); the reader construction reuses the handle instead of opening a
  // second one.
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(paimon::GetPaimonDataSplitReaderOpenCount() - reader_opens, 1);
  EXPECT_EQ(paimon::GetPaimonDataSplitStreamOpenCount() - stream_opens, 2);

  // Sequential consumption of every chunk keeps the same stream: no restarts.
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));
  EXPECT_EQ(paimon::GetPaimonDataSplitReaderOpenCount() - reader_opens, 1);
  EXPECT_EQ(paimon::GetPaimonDataSplitStreamOpenCount() - stream_opens, 2);

  // A backward seek is a counted stream restart, not a new handle.
  ASSERT_AND_ASSIGN(auto first, reader->get_chunk(0));
  EXPECT_EQ(first->num_rows(), 4);
  EXPECT_EQ(paimon::GetPaimonDataSplitReaderOpenCount() - reader_opens, 1);
  EXPECT_EQ(paimon::GetPaimonDataSplitStreamOpenCount() - stream_opens, 3);
}

}  // namespace
}  // namespace milvus_storage::test
