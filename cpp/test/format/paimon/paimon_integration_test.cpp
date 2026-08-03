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
#include <fstream>
#include <numeric>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <fmt/format.h>
#include <folly/json.h>
#include <parquet/arrow/writer.h>

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

std::string LocalFilePath(std::string path) {
  if (path.rfind("file://", 0) == 0) {
    return path.substr(7);
  }
  if (path.rfind("file:", 0) == 0) {
    return path.substr(5);
  }
  return path;
}

int64_t ReadAllRows(const std::shared_ptr<FormatReader>& reader) {
  auto infos = reader->get_row_group_infos().ValueOrDie();
  int64_t rows = 0;
  for (size_t index = 0; index < infos.size(); ++index) {
    rows += reader->get_chunk(static_cast<int>(index)).ValueOrDie()->num_rows();
  }
  return rows;
}

arrow::Status RewriteParquetWithRowGroups(const std::string& path, int32_t rows_per_group, int32_t group_count) {
  auto schema = arrow::schema({arrow::field("id", arrow::int32(), false)});
  ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::FileOutputStream::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
  for (int32_t group = 0; group < group_count; ++group) {
    arrow::Int32Builder builder;
    for (int32_t row = 0; row < rows_per_group; ++row) {
      ARROW_RETURN_NOT_OK(builder.Append(group * rows_per_group + row));
    }
    ARROW_ASSIGN_OR_RAISE(auto ids, builder.Finish());
    auto batch = arrow::RecordBatch::Make(schema, rows_per_group, {std::move(ids)});
    ARROW_RETURN_NOT_OK(writer->NewBufferedRowGroup());
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
  }
  ARROW_RETURN_NOT_OK(writer->Close());
  return sink->Close();
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

TEST_F(PaimonIntegrationTest, ReadsWithoutMetadataCache) {
  constexpr uint64_t kRows = 12;
  paimon::CreateTestTable(table_dir_, kRows, "append");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8())});
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "false"), std::nullopt);

  ASSERT_AND_ASSIGN(auto reader,
                    api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_, nullptr));
  EXPECT_EQ(reader->total_rows(), static_cast<int64_t>(kRows));
  std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
  std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 1));
  EXPECT_EQ(std::accumulate(batches.begin(), batches.end(), int64_t{0},
                            [](int64_t rows, const auto& batch) { return rows + batch->num_rows(); }),
            static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, DeletionVectorReadsBypassDisabledMetadataCache) {
  constexpr uint64_t kRows = 10;
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9});

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8())});
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "false"), std::nullopt);

  const auto key = paimon::PaimonFormatReader::MetaTrait::cache_key(files.front());
  const std::vector<int32_t> expected_ids = {0, 2, 3, 4, 6, 7, 8};
  MetadataCache cache;
  for (int round = 0; round < 2; ++round) {
    ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_,
                                                                  nullptr, "", cache));
    ASSERT_EQ(reader->total_rows(), static_cast<int64_t>(expected_ids.size()));
    std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
    std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 1));
    std::vector<int32_t> ids;
    for (const auto& batch : batches) {
      auto column = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(0));
      ASSERT_NE(column, nullptr);
      for (int64_t row = 0; row < column->length(); ++row) {
        ids.push_back(column->Value(row));
      }
    }
    EXPECT_EQ(ids, expected_ids);
    // Deletion positions must be reloaded per reader: the caller cache stays
    // empty, proving the disabled flag bypasses it instead of being ignored.
    EXPECT_FALSE(cache.get<paimon::PaimonFormatReader>()->get(key).has_value());
  }

  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "true"), std::nullopt);
  ASSERT_AND_ASSIGN(auto cached_reader, api::ColumnGroupReader::create(schema, column_group, {"id", "name"},
                                                                       properties_, nullptr, "", cache));
  ASSERT_EQ(cached_reader->total_rows(), static_cast<int64_t>(expected_ids.size()));
  EXPECT_TRUE(cache.get<paimon::PaimonFormatReader>()->get(key).has_value());
}

TEST_F(PaimonIntegrationTest, AutoUsesDirectFileForAppendVortex) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append", {}, "vortex");

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
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9}, "vortex");

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

TEST_F(PaimonIntegrationTest, VortexPredicateWithDeletionVectorFailsClosed) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}, "vortex");

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = {files.front()};
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8())});

  for (const auto* cache_enabled : {"true", "false"}) {
    SCOPED_TRACE(cache_enabled);
    ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, cache_enabled), std::nullopt);
    auto reader =
        api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_, nullptr, "name >= 'row_2'");
    ASSERT_FALSE(reader.ok());
    EXPECT_TRUE(reader.status().IsNotImplemented()) << reader.status().ToString();
  }
}

TEST_F(PaimonIntegrationTest, ReadsSpecifiedSnapshot) {
  constexpr uint64_t kRows = 9;
  auto snapshot_id = paimon::CreateTestTable(table_dir_, kRows, "append");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(snapshot_id).c_str()), std::nullopt);

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  EXPECT_EQ(files.front().end_index, static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, MergeOnReadTableFailsClosedAsNotImplemented) {
  paimon::CreateTestTable(table_dir_, 10, "mor");

  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsNotImplemented()) << files.status().ToString();
  const auto message = files.status().ToString();
  EXPECT_NE(message.find("data-split reading"), std::string::npos) << message;
}

TEST_F(PaimonIntegrationTest, InvalidScanModeFailsAsInvalid) {
  paimon::CreateTestTable(table_dir_, 10, "append");

  auto files = Explore("invalid-mode");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsInvalid()) << files.status().ToString();
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

TEST_F(PaimonIntegrationTest, JavaBitmap64DeletionVectorFailsClosedAsNotImplemented) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");

  // Serialized by Paimon Java 1.2.0 with deletion-vectors.bitmap64=true:
  // big-endian outer data length, little-endian bitmap64 magic (1681511377)
  // and RoaringTreemap payload; DeletionFile.length covers the complete
  // 64-byte serialized value rather than bitmap32's payload-only length.
  static constexpr unsigned char kJavaBitmap64[] = {
      0x00, 0x00, 0x00, 0x38, 0xd1, 0xd3, 0x39, 0x64, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x3a, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x3a, 0x30, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x89, 0xde, 0xf4, 0x64};

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  auto dv_path = LocalFilePath(descriptor["deletion_file"]["path"].asString());
  {
    std::ofstream out(dv_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(kJavaBitmap64), sizeof(kJavaBitmap64));
  }
  descriptor["deletion_file"]["offset"] = 0;
  descriptor["deletion_file"]["length"] = static_cast<int64_t>(sizeof(kJavaBitmap64));
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));

  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsNotImplemented()) << reader.status().ToString();
  const auto message = reader.status().ToString();
  EXPECT_NE(message.find("bitmap64"), std::string::npos) << message;
  EXPECT_NE(message.find("deletion-vectors.bitmap64=false"), std::string::npos) << message;
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

  auto negative_take = reader->take({-1});
  ASSERT_FALSE(negative_take.ok());
  EXPECT_TRUE(negative_take.status().IsInvalid());
  auto past_end_take = reader->take({7});
  ASSERT_FALSE(past_end_take.ok());
  EXPECT_TRUE(past_end_take.status().IsInvalid());

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
  EXPECT_TRUE(invalid.status().IsInvalid()) << invalid.status().ToString();
}

TEST_F(PaimonIntegrationTest, CorruptDeletionVectorCrcFailsAsInvalid) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  const auto& deletion = descriptor["deletion_file"];
  const auto path = LocalFilePath(deletion["path"].asString());
  const auto crc_offset = deletion["offset"].asInt() + 4 + deletion["length"].asInt();
  std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(stream.is_open());
  stream.seekg(crc_offset);
  char crc_byte = 0;
  stream.read(&crc_byte, 1);
  ASSERT_EQ(stream.gcount(), 1);
  stream.clear();
  stream.seekp(crc_offset);
  crc_byte ^= 1;
  stream.write(&crc_byte, 1);
  ASSERT_TRUE(stream.good());
  stream.close();

  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid()) << reader.status().ToString();
  EXPECT_NE(reader.status().ToString().find("CRC mismatch"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, MissingDeletionVectorFileRemainsIOError) {
  paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["deletion_file"]["path"] = fmt::format("file://{}/missing.dv", table_dir_);
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));

  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsIOError()) << reader.status().ToString();
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

  std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
  std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 1));
  std::vector<int32_t> ids;
  for (const auto& batch : batches) {
    auto values = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(0));
    ASSERT_NE(values, nullptr);
    for (int64_t row = 0; row < values->length(); ++row) {
      ids.push_back(values->Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3, 4, 6}));
}

TEST_F(PaimonIntegrationTest, ExplicitDirectFileRejectsMergeOnRead) {
  paimon::CreateTestTable(table_dir_, 10, "mor");
  auto files = Explore("direct-file");
  ASSERT_FALSE(files.ok());
  EXPECT_NE(files.status().ToString().find("cannot use direct-file"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, DataSplitDescriptorFailsClosedAsNotImplemented) {
  paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["read_path"] = "data-split";
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsNotImplemented()) << reader.status().ToString();
}

TEST_F(PaimonIntegrationTest, FullyDeletedTableProducesNoEntries) {
  constexpr uint64_t kRows = 6;
  paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {0, 1, 2, 3, 4, 5});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  EXPECT_TRUE(files.empty());
}

TEST_F(PaimonIntegrationTest, FullyDeletedTrailingRowGroupIsNotExposedAsChunk) {
  paimon::CreateTestTable(table_dir_, 12, "deletion-vector", {8, 9, 10, 11});
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_STATUS_OK(RewriteParquetWithRowGroups(LocalFilePath(files.front().path), 4, 3));

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr));
  ASSERT_EQ(reader->total_number_of_chunks(), 2);

  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks({0, 1}, 1));
  std::vector<int32_t> ids;
  for (const auto& batch : batches) {
    const auto& values = static_cast<const arrow::Int32Array&>(*batch->column(0));
    for (int64_t row = 0; row < values.length(); ++row) {
      ids.push_back(values.Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
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

  const std::runtime_error corrupt("[paimon:error=invalid] invalid Paimon metadata");
  EXPECT_TRUE(paimon::ClassifyPaimonError("open", corrupt).IsInvalid());

  const std::runtime_error unsupported(
      "[paimon:error=not-implemented] Paimon direct-file does not support format: orc");
  EXPECT_TRUE(paimon::ClassifyPaimonError("read", unsupported).IsNotImplemented());

  // Unmarked messages are never promoted to a terminal class.
  const std::runtime_error transient("connection reset by peer while reading snapshot-9");
  EXPECT_TRUE(paimon::ClassifyPaimonError("plan", transient).IsIOError());
}

TEST_F(PaimonIntegrationTest, MissingPinnedSnapshotFailsPlanAsInvalidWithBounds) {
  auto snapshot_id = paimon::CreateTestTable(table_dir_, 10, "append");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(snapshot_id + 1000).c_str()),
            std::nullopt);

  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsInvalid()) << files.status().ToString();
  const auto message = files.status().ToString();
  EXPECT_NE(message.find("no longer exists"), std::string::npos) << message;
  EXPECT_NE(message.find("earliest="), std::string::npos) << message;
  EXPECT_NE(message.find(fmt::format("latest={}", snapshot_id)), std::string::npos) << message;
  EXPECT_NE(message.find("refresh the external collection"), std::string::npos) << message;
}

TEST_F(PaimonIntegrationTest, VortexWithoutMemoryStatisticsReturnsNotImplemented) {
  constexpr uint64_t kRows = 17;
  paimon::CreateTestTable(table_dir_, kRows, "append", {}, "vortex");
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");

  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(descriptor.count("estimated_bytes"), 0);

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8())});
  ASSERT_AND_ASSIGN(auto reader,
                    api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_, nullptr));
  ASSERT_GT(reader->total_number_of_chunks(), 0);
  auto estimate = reader->get_chunk_estimated_size(0);
  EXPECT_TRUE(estimate.status().IsNotImplemented()) << estimate.status().ToString();
}

}  // namespace
}  // namespace milvus_storage::test
