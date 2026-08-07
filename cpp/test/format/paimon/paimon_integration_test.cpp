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

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/io_util.h>
#include <fmt/format.h>
#include <folly/json.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/paimon/paimon_common.h"
#include "milvus-storage/format/paimon/paimon_format_reader.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
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
  auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::FileOutputStream::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
  for (int32_t group = 0; group < group_count; ++group) {
    arrow::Int64Builder builder;
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "append").status());

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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "append").status());

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())});
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9}).status());

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())});
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "false"), std::nullopt);

  const auto key = paimon::PaimonFormatReader::MetaTrait::cache_key(files.front());
  const std::vector<int64_t> expected_ids = {0, 2, 3, 4, 6, 7, 8};
  MetadataCache cache;
  for (int round = 0; round < 2; ++round) {
    ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_,
                                                                  nullptr, "", cache));
    ASSERT_EQ(reader->total_rows(), static_cast<int64_t>(expected_ids.size()));
    std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
    std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 1));
    std::vector<int64_t> ids;
    for (const auto& batch : batches) {
      auto column = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "append", {}, "vortex").status());

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
  const std::vector<int64_t> expected = {0, 4, 16};
  for (int64_t row = 0; row < taken->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, taken->column(0)->GetScalar(row));
    ASSERT_NE(std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar)->value, expected[row]);
  }

  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(3, 9));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range.get()));
  ASSERT_EQ(range_table->num_rows(), 6);
  ASSERT_AND_ASSIGN(auto first, range_table->column(0)->GetScalar(0));
  ASSERT_AND_ASSIGN(auto last, range_table->column(0)->GetScalar(5));
  EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int64Scalar>(first)->value, 3);
  EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int64Scalar>(last)->value, 8);

  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  EXPECT_EQ(ReadAllRows(clone), static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, VortexDeletionVectorUsesDirectFile) {
  constexpr uint64_t kRows = 10;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9}, "vortex").status());

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(ReadPath(files.front()), "direct-file");
  EXPECT_EQ(files.front().end_index, 7);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 1, 4, 6}));
  ASSERT_EQ(taken->num_rows(), 4);
  const std::vector<int64_t> expected = {0, 2, 6, 8};
  for (int64_t row = 0; row < taken->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, taken->column(0)->GetScalar(row));
    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar)->value, expected[row]);
  }
}

TEST_F(PaimonIntegrationTest, ReadsSpecifiedSnapshot) {
  constexpr uint64_t kRows = 9;
  ASSERT_AND_ASSIGN(auto snapshot_id, paimon::CreateTestTable(table_dir_, kRows, "append"));
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(snapshot_id).c_str()), std::nullopt);

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_FALSE(files.empty());
  EXPECT_EQ(files.front().end_index, static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, ScanOptionsAreValidatedForLatestAndPinnedSnapshots) {
  ASSERT_AND_ASSIGN(auto snapshot_id, paimon::CreateTestTable(table_dir_, 1, "append"));

  std::ifstream input(fmt::format("{}/schema/schema-0", table_dir_));
  ASSERT_TRUE(input.is_open());
  auto latest_schema =
      folly::parseJson(std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
  latest_schema["id"] = 1;
  latest_schema["options"]["scan.watermark"] = "5";
  std::ofstream output(fmt::format("{}/schema/schema-1", table_dir_));
  ASSERT_TRUE(output.is_open());
  output << folly::toJson(latest_schema);
  ASSERT_TRUE(output.good());
  output.close();

  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsNotImplemented()) << files.status().ToString();
  EXPECT_NE(files.status().ToString().find("scan.watermark"), std::string::npos);

  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(snapshot_id).c_str()), std::nullopt);
  files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsNotImplemented()) << files.status().ToString();
  EXPECT_NE(files.status().ToString().find("scan.watermark"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, ExplicitDataSplitReadsAppendTable) {
  constexpr uint64_t kRows = 10;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "append").status());

  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(ReadPath(files.front()), "data-split");
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "false"), std::nullopt);
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto row_groups, reader->get_row_group_infos());
  ASSERT_FALSE(row_groups.empty());
  EXPECT_FALSE(row_groups.front().memory_size_available);
  auto column_sizes = reader->get_rg_column_memsz(0);
  ASSERT_FALSE(column_sizes.ok());
  EXPECT_TRUE(column_sizes.status().IsNotImplemented()) << column_sizes.status().ToString();
  EXPECT_EQ(ReadAllRows(reader), static_cast<int64_t>(kRows));

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  MetadataCache cache;
  ASSERT_AND_ASSIGN(auto column_group_reader,
                    api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr, "", cache));
  EXPECT_EQ(column_group_reader->total_rows(), static_cast<int64_t>(kRows));
  EXPECT_FALSE(cache.get<paimon::PaimonFormatReader>()
                   ->get(paimon::PaimonFormatReader::MetaTrait::cache_key(files.front()))
                   .has_value());
}

TEST_F(PaimonIntegrationTest, ExplicitDataSplitAppliesDeletionVector) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}).status());

  ASSERT_AND_ASSIGN(auto files, Explore("data-split"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");
  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));

  std::vector<int64_t> ids;
  ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
  for (size_t group = 0; group < groups.size(); ++group) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(static_cast<int>(group)));
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(values, nullptr);
    for (int64_t row = 0; row < values->length(); ++row) {
      ids.push_back(values->Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int64_t>{0, 2, 3, 4, 6, 7, 8}));
}

TEST_F(PaimonIntegrationTest, InvalidScanModeFailsAsInvalid) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "append").status());

  auto files = Explore("invalid-mode");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsInvalid()) << files.status().ToString();
}

TEST_F(PaimonIntegrationTest, MalformedMetadataTypesFailClosed) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "append").status());
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1}).status());
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1}).status());
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {1, 5, 9}).status());

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");
  ASSERT_EQ(files.front().end_index, 7);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  EXPECT_EQ(ReadAllRows(reader), 7);

  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 1, 4, 6}));
  ASSERT_EQ(taken->num_rows(), 4);
  auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(taken->column(0)->chunk(0));
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
  auto range_ids = std::dynamic_pointer_cast<arrow::Int64Array>(range_table->column(0)->chunk(0));
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}).status());
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
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}).status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["deletion_file"]["path"] = fmt::format("file://{}/missing.dv", table_dir_);
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));

  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsIOError()) << reader.status().ToString();
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(reader.status()), ENOENT);
}

TEST_F(PaimonIntegrationTest, DirectFileFragmentRangeUsesPostDeletionLogicalRows) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}).status());
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
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr));
  ASSERT_EQ(reader->total_rows(), 4);

  std::vector<int64_t> chunk_indices(reader->total_number_of_chunks());
  std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunk_indices, 1));
  std::vector<int64_t> ids;
  for (const auto& batch : batches) {
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(values, nullptr);
    for (int64_t row = 0; row < values->length(); ++row) {
      ids.push_back(values->Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int64_t>{2, 3, 4, 6}));
}

TEST_F(PaimonIntegrationTest, IgnoredPredicatePreservesDirectFileFragmentRange) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "deletion-vector", {1, 5, 9}).status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto fragment = files.front();
  fragment.start_index = 1;
  fragment.end_index = 5;
  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = {std::move(fragment)};
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  ASSERT_AND_ASSIGN(auto reader,
                    api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr, "id >= 0"));

  ASSERT_EQ(reader->total_number_of_chunks(), 1);
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  const auto& ids = static_cast<const arrow::Int64Array&>(*batch->column(0));
  ASSERT_EQ(ids.length(), 4);
  EXPECT_EQ(ids.Value(0), 2);
  EXPECT_EQ(ids.Value(1), 3);
  EXPECT_EQ(ids.Value(2), 4);
  EXPECT_EQ(ids.Value(3), 6);
}

TEST_F(PaimonIntegrationTest, ExplicitDirectFileRejectsMergeOnRead) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "mor").status());
  auto files = Explore("direct-file");
  ASSERT_FALSE(files.ok());
  EXPECT_NE(files.status().ToString().find("cannot use direct-file"), std::string::npos);
}

TEST_F(PaimonIntegrationTest, AutoUsesDataSplitForMergeOnRead) {
  constexpr uint64_t kRows = 17;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());

  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "data-split");

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id", "name"}, nullptr));
  int64_t rows = 0;
  ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
  for (size_t group = 0; group < groups.size(); ++group) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(static_cast<int>(group)));
    auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto names = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(1));
    ASSERT_NE(ids, nullptr);
    ASSERT_NE(names, nullptr);
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      const auto id = ids->Value(row);
      const auto multiplier = id < static_cast<int64_t>(kRows / 2) ? 1 : 10;
      EXPECT_EQ(names->GetString(row), fmt::format("row_{}", id * multiplier));
      ++rows;
    }
  }
  EXPECT_EQ(rows, static_cast<int64_t>(kRows));
}

TEST_F(PaimonIntegrationTest, DataSplitSupportsRangeAndCloneReads) {
  constexpr uint64_t kRows = 17;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(3, 9));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range.get()));
  EXPECT_EQ(range_table->num_rows(), 6);
  for (int64_t row = 0; row < range_table->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, range_table->column(0)->GetScalar(row));
    auto id = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->value, row + 3);
  }

  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  ASSERT_AND_ASSIGN(auto cloned_head, clone->get_chunk(0));
  EXPECT_EQ(cloned_head->num_rows(), 4);
}

TEST_F(PaimonIntegrationTest, DataSplitLogicalChunkSpansPaimonBatches) {
  constexpr uint64_t kRows = 10000;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "8192"), std::nullopt);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  ASSERT_EQ(batch->num_rows(), 8192);
  auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
  ASSERT_NE(ids, nullptr);
  EXPECT_EQ(ids->Value(0), 0);
  EXPECT_EQ(ids->Value(8191), 8191);
}

TEST_F(PaimonIntegrationTest, DataSplitReadsNonContiguousChunks) {
  constexpr uint64_t kRows = 4096;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "512"), std::nullopt);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  const std::vector<int> chunks = {0, 2, 4, 6};
  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(chunks));
  ASSERT_EQ(batches.size(), chunks.size());
  for (size_t index = 0; index < batches.size(); ++index) {
    auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(batches[index]->column(0));
    ASSERT_NE(ids, nullptr);
    ASSERT_EQ(ids->length(), 512);
    EXPECT_EQ(ids->Value(0), chunks[index] * 512);
    EXPECT_EQ(ids->Value(511), chunks[index] * 512 + 511);
  }
}

TEST_F(PaimonIntegrationTest, DataSplitTakeCompactsSparseBatches) {
  constexpr uint64_t kRows = 4096;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_,
                                                      {"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 2048, 4095}));
  ASSERT_EQ(taken->num_rows(), 3);

  int64_t value_buffer_bytes = 0;
  const auto& id_column = taken->column(0);
  for (const auto& chunk : id_column->chunks()) {
    ASSERT_GE(chunk->data()->buffers.size(), 2);
    ASSERT_NE(chunk->data()->buffers[1], nullptr);
    value_buffer_bytes += chunk->data()->buffers[1]->size();
  }
  EXPECT_LE(value_buffer_bytes, static_cast<int64_t>(3 * sizeof(int64_t)));

  const std::vector<int64_t> expected = {0, 2048, 4095};
  for (int64_t row = 0; row < taken->num_rows(); ++row) {
    ASSERT_AND_ASSIGN(auto scalar, id_column->GetScalar(row));
    ASSERT_NE(std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar)->value, expected[row]);
  }
}

TEST_F(PaimonIntegrationTest, AsyncDataSplitChunksSpanSourceBatches) {
  constexpr uint64_t kRows = 10000;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "mor").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "8192"), std::nullopt);

  auto column_groups = std::make_shared<api::ColumnGroups>();
  column_groups->push_back(std::make_shared<api::ColumnGroup>(
      api::ColumnGroup{.columns = {"id"}, .format = LOON_FORMAT_PAIMON_TABLE, .files = {files.front()}}));
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  auto reader = api::Reader::create(column_groups, schema, nullptr, properties_);
  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));

  const std::vector<int64_t> chunks = {0, 1};
  ASSERT_AND_ASSIGN(auto batches, std::move(chunk_reader->get_chunks_async(chunks, 8)).get());
  ASSERT_EQ(batches.size(), chunks.size());
  for (size_t index = 0; index < batches.size(); ++index) {
    auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(batches[index]->column(0));
    ASSERT_NE(ids, nullptr);
    const int64_t expected_rows = index == 0 ? 8192 : 1808;
    ASSERT_EQ(ids->length(), expected_rows);
    EXPECT_EQ(ids->Value(0), chunks[index] * 8192);
    EXPECT_EQ(ids->Value(expected_rows - 1), chunks[index] * 8192 + expected_rows - 1);
  }
}

TEST_F(PaimonIntegrationTest, MalformedDataSplitDescriptorFailsAsInvalid) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 10, "append").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);

  auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  descriptor["read_path"] = "data-split";
  files.front().Set(api::kPropertyMetadata, folly::toJson(descriptor));
  auto reader = FormatReader::create(nullptr, LOON_FORMAT_PAIMON_TABLE, files.front(), properties_, {"id"}, nullptr);
  ASSERT_FALSE(reader.ok());
  EXPECT_TRUE(reader.status().IsInvalid()) << reader.status().ToString();
}

TEST_F(PaimonIntegrationTest, FullyDeletedTableProducesNoEntries) {
  constexpr uint64_t kRows = 6;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "deletion-vector", {0, 1, 2, 3, 4, 5}).status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  EXPECT_TRUE(files.empty());
}

TEST_F(PaimonIntegrationTest, FullyDeletedTrailingRowGroupIsNotExposedAsChunk) {
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, 12, "deletion-vector", {8, 9, 10, 11}).status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_STATUS_OK(RewriteParquetWithRowGroups(LocalFilePath(files.front().path), 4, 3));

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, column_group, {"id"}, properties_, nullptr));
  ASSERT_EQ(reader->total_number_of_chunks(), 2);

  ASSERT_AND_ASSIGN(auto batches, reader->get_chunks({0, 1}, 1));
  std::vector<int64_t> ids;
  for (const auto& batch : batches) {
    const auto& values = static_cast<const arrow::Int64Array&>(*batch->column(0));
    for (int64_t row = 0; row < values.length(); ++row) {
      ids.push_back(values.Value(row));
    }
  }
  EXPECT_EQ(ids, (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(PaimonIntegrationTest, MissingTableFailsAndWriterIsReadOnly) {
  table_dir_ += "-missing";
  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());

  ASSERT_AND_ASSIGN(auto* format, Format::get(LOON_FORMAT_PAIMON_TABLE));
  auto writer = format->create_writer(nullptr, arrow::schema({arrow::field("id", arrow::int64())}), "unused", "unused",
                                      properties_);
  ASSERT_FALSE(writer.ok());
  EXPECT_TRUE(writer.status().IsNotImplemented());
}

TEST(PaimonBridgeErrorClassification, MarkersMapToArrowStatuses) {
  EXPECT_TRUE(paimon::MakePaimonBridgeErrorStatus("[paimon:error=invalid] invalid metadata").IsInvalid());
  EXPECT_TRUE(paimon::MakePaimonBridgeErrorStatus("[paimon:error=not-implemented] direct-file does not support orc")
                  .IsNotImplemented());

  auto not_found = paimon::MakePaimonBridgeErrorStatus("[paimon:error=not-found] missing object");
  EXPECT_TRUE(not_found.IsIOError());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(not_found), ENOENT);

  auto transient = paimon::MakePaimonBridgeErrorStatus("[paimon:error=transient-throttling] object store rate limit");
  auto detail = ExtendStatusDetail::UnwrapStatus(transient);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);

  transient = paimon::MakePaimonBridgeErrorStatus("[paimon:error=transient-service] object store unavailable");
  detail = ExtendStatusDetail::UnwrapStatus(transient);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientService);

  EXPECT_TRUE(paimon::MakePaimonBridgeErrorStatus("unclassified storage failure").IsIOError());
}

TEST_F(PaimonIntegrationTest, MissingPinnedSnapshotFailsPlanAsInvalidWithRefresh) {
  ASSERT_AND_ASSIGN(auto snapshot_id, paimon::CreateTestTable(table_dir_, 10, "append"));
  ASSERT_EQ(api::SetValue(properties_, PROPERTY_PAIMON_SNAPSHOT_ID, std::to_string(snapshot_id + 1000).c_str()),
            std::nullopt);

  auto files = Explore("auto");
  ASSERT_FALSE(files.ok());
  EXPECT_TRUE(files.status().IsInvalid()) << files.status().ToString();
  const auto message = files.status().ToString();
  EXPECT_NE(message.find("required metadata"), std::string::npos) << message;
  EXPECT_NE(message.find("was not found"), std::string::npos) << message;
  EXPECT_NE(message.find("refresh the external collection"), std::string::npos) << message;
}

TEST_F(PaimonIntegrationTest, VortexWithoutMemoryStatisticsReturnsNotImplemented) {
  constexpr uint64_t kRows = 17;
  ASSERT_STATUS_OK(paimon::CreateTestTable(table_dir_, kRows, "append", {}, "vortex").status());
  ASSERT_AND_ASSIGN(auto files, Explore("auto"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_EQ(ReadPath(files.front()), "direct-file");

  const auto descriptor = folly::parseJson(files.front().Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(descriptor.count("estimated_bytes"), 0);

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"id", "name"};
  column_group->format = LOON_FORMAT_PAIMON_TABLE;
  column_group->files = files;
  auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())});
  ASSERT_AND_ASSIGN(auto reader,
                    api::ColumnGroupReader::create(schema, column_group, {"id", "name"}, properties_, nullptr));
  ASSERT_GT(reader->total_number_of_chunks(), 0);
  auto estimate = reader->get_chunk_estimated_size(0);
  EXPECT_TRUE(estimate.status().IsNotImplemented()) << estimate.status().ToString();
}

}  // namespace
}  // namespace milvus_storage::test
