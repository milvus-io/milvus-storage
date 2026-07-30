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

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <parquet/arrow/writer.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/packed/writer.h"

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/format/parquet/file_reader.h"
#include "packed_test_base.h"

namespace milvus_storage {

namespace {

void ExpectPackedCode(const arrow::Status& status, ExtendStatusCode code) {
  ASSERT_FALSE(status.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), code);
}

void ExpectExceptionMessageContainsCode(const std::function<void()>& fn, const std::string& code_name) {
  try {
    fn();
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(code_name), std::string::npos) << e.what();
  }
}

}  // namespace

class PackedErrorStatusTest : public PackedTestBase {};

TEST_F(PackedErrorStatusTest, WriterPathGroupMismatchIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{0}, {1}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterColumnIndexOutOfRangeIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{schema_->num_fields()}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterRecordBatchColumnMismatchIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_));
  ASSERT_AND_ASSIGN(auto short_batch, record_batch_->SelectColumns({0, 1}));

  auto status = writer->Write(short_batch);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
  (void)writer->Close();
}

TEST_F(PackedErrorStatusTest, ColumnGroupNullBatchIsInvalidArgs) {
  ColumnGroup group(0, {0});

  auto status = group.AddRecordBatch(nullptr);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, ReaderMissingFileKeepsFilesystemError) {
  std::vector<std::string> paths = {path_ + "/missing.parquet"};

  try {
    PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    auto message = std::string(e.what());
    EXPECT_NE(message.find("missing.parquet"), std::string::npos) << message;
    EXPECT_EQ(message.find("PackedStorageIO"), std::string::npos) << message;
  }
}

TEST_F(PackedErrorStatusTest, ReaderNullOutputPointerIsInvalidArgs) {
  SetupOneFile();
  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_);

  auto status = reader.ReadNext(nullptr);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
  ASSERT_STATUS_OK(reader.Close());
}

TEST_F(PackedErrorStatusTest, ReaderMissingPackedMetadataIsMetadataCorrupted) {
  auto parquet_path = path_ + "/plain.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(parquet_path));
  ASSERT_STATUS_OK(::parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(), sink, 2));
  ASSERT_STATUS_OK(sink->Close());
  std::vector<std::string> paths = {parquet_path};

  ExpectExceptionMessageContainsCode([&]() { PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_); },
                                     "PackedMetadataCorrupted");
}

TEST_F(PackedErrorStatusTest, MakeReportsMissingFileAsStatusWithClassification) {
  std::vector<std::string> paths = {path_ + "/missing.parquet"};

  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);
  ASSERT_FALSE(result.ok());
  const auto& status = result.status();
  // The wrap preserves the filesystem's own not-found detail (WrapExtendError
  // keeps the cause's detail), so consumers get the fine-grained
  // classification the throwing constructor used to destroy.
  EXPECT_NE(status.ToString().find("missing.parquet"), std::string::npos) << status.ToString();
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist) << status.ToString();
}

TEST_F(PackedErrorStatusTest, MakeReportsEmptyPathsAsInvalidArgs) {
  std::vector<std::string> paths;

  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);
  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, MakeSucceedsOnValidFile) {
  SetupOneFile();
  std::vector<std::string> paths = {one_file_path_};

  ASSERT_AND_ASSIGN(auto reader, PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_));
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_STATUS_OK(reader->ReadNext(&batch));
  ASSERT_NE(batch, nullptr);
  ASSERT_STATUS_OK(reader->Close());
}

TEST_F(PackedErrorStatusTest, ColumnGroupTableSchemaMismatchIsNotCalledCorruption) {
  ColumnGroup group(0, {0});
  ASSERT_STATUS_OK(group.AddRecordBatch(record_batch_));
  // Second batch with a different schema. Table() must surface arrow's own
  // Invalid rather than wrapping it, so the message survives -- but the segcore
  // landing is StorageError, NOT DataFormatBroken: mismatched batches are the
  // caller's contract violation, not corrupt bytes on disk.
  auto other_schema = arrow::schema({arrow::field("other", arrow::int8())});
  arrow::Int8Builder builder;
  ASSERT_STATUS_OK(builder.AppendValues({1, 2, 3}));
  ASSERT_AND_ASSIGN(auto other_array, builder.Finish());
  auto other_batch = arrow::RecordBatch::Make(other_schema, 3, {other_array});
  ASSERT_STATUS_OK(group.AddRecordBatch(other_batch));

  auto table_result = group.Table();
  ASSERT_FALSE(table_result.ok());
  EXPECT_TRUE(table_result.status().IsInvalid()) << table_result.status().ToString();
  // Renamed from ...IsDataFormatBroken, because that was the wrong answer and
  // this case is the clearest argument for changing it. "Schema at index 1 was
  // different" is a caller handing us mismatched batches -- a contract
  // violation, not corrupt bytes on disk. Reporting it as DataFormatBroken sent
  // whoever read the alert to inspect a file that is perfectly fine.
  EXPECT_EQ(ToSegcoreError(table_result.status()).get_error_code(), milvus::StorageError)
      << table_result.status().ToString();
  EXPECT_NE(ToSegcoreError(table_result.status()).get_error_code(), milvus::DataFormatBroken)
      << table_result.status().ToString();
}

TEST_F(PackedErrorStatusTest, ColumnGroupNullBatchConstructorYieldsEmptyGroup) {
  // The batch-taking constructor has no status channel; a null batch must not
  // crash (previous behavior dereferenced it) and yields an empty group.
  ColumnGroup group(0, {0}, nullptr);
  EXPECT_EQ(group.size(), 0);
  EXPECT_EQ(group.GetTotalRows(), 0);
  EXPECT_EQ(group.Schema(), nullptr);
}

// Integration regression for the fifth unguarded ValueOrDie
// (file_reader.cpp: FileRowGroupReader::init, schema==nullptr branch): a
// parquet file that passes the packed key-value metadata checks but whose
// fields lack PARQUET:field_id must surface as a status, not abort.
TEST_F(PackedErrorStatusTest, FileRowGroupReaderMissingFieldIdsIsStatusNotAbort) {
  SetupOneFile();

  // Grab the packed key-value metadata from a genuine packed file.
  ASSERT_AND_ASSIGN(auto source, fs_->OpenInputFile(one_file_path_));
  std::unique_ptr<::parquet::arrow::FileReader> packed_reader;
  ASSERT_STATUS_OK(::parquet::arrow::OpenFile(source, arrow::default_memory_pool(), &packed_reader));
  auto packed_kv = packed_reader->parquet_reader()->metadata()->key_value_metadata();
  ASSERT_NE(packed_kv, nullptr);

  // Write a parquet file carrying the SAME packed key-value metadata but a
  // schema whose fields have no PARQUET:field_id.
  std::vector<std::shared_ptr<arrow::Field>> bare_fields;
  for (const auto& f : record_batch_->schema()->fields()) {
    bare_fields.push_back(f->WithMetadata(nullptr));
  }
  // Schema-level metadata is written into the parquet key-value metadata by
  // the arrow writer, so attaching the packed KV here reproduces "packed KV
  // present, field ids absent".
  auto bare_schema = arrow::schema(bare_fields);
  auto bare_batch = arrow::RecordBatch::Make(bare_schema, record_batch_->num_rows(), record_batch_->columns());
  ASSERT_AND_ASSIGN(auto bare_table, arrow::Table::FromRecordBatches({bare_batch}));
  auto no_fid_path = path_ + "/no_field_ids.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(no_fid_path));
  std::unique_ptr<::parquet::arrow::FileWriter> bare_writer;
  ASSERT_AND_ASSIGN(bare_writer, ::parquet::arrow::FileWriter::Open(*bare_schema, arrow::default_memory_pool(), sink,
                                                                    ::parquet::default_writer_properties(),
                                                                    ::parquet::default_arrow_writer_properties()));
  ASSERT_STATUS_OK(bare_writer->WriteTable(*bare_table, 100));
  ASSERT_STATUS_OK(bare_writer->AddKeyValueMetadata(packed_kv));
  ASSERT_STATUS_OK(bare_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  // schema==nullptr entry: the reader must derive the schema from the file,
  // hit the missing-field-id condition, and report it as a status.
  auto result = FileRowGroupReader::Make(fs_, no_fid_path);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
  EXPECT_NE(result.status().ToString().find("field"), std::string::npos) << result.status().ToString();
}

}  // namespace milvus_storage
