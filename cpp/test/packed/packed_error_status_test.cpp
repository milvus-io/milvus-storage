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

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <parquet/arrow/writer.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/packed/writer.h"

#include <parquet/arrow/reader.h>

#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
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

class FixedInputFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  FixedInputFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs,
                       std::shared_ptr<arrow::io::RandomAccessFile> file)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)), file_(std::move(file)) {}

  std::string type_name() const override { return "fixed-input"; }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string&) override {
    return file_;
  }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
};

class FailingParquetReadFile final : public arrow::io::RandomAccessFile {
  public:
  explicit FailingParquetReadFile(arrow::Status failure) : failure_(std::move(failure)) {}

  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }
  arrow::Status Abort() override { return Close(); }
  arrow::Result<int64_t> Tell() const override { return position_; }
  bool closed() const override { return closed_; }
  arrow::Status Seek(int64_t position) override {
    position_ = position;
    return arrow::Status::OK();
  }
  arrow::Result<int64_t> GetSize() override { return 8; }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    auto result = ReadAt(position_, nbytes, out);
    if (result.ok()) {
      position_ += result.ValueOrDie();
    }
    return result;
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    auto result = ReadAt(position_, nbytes);
    if (result.ok()) {
      position_ += result.ValueOrDie()->size();
    }
    return result;
  }
  arrow::Result<int64_t> ReadAt(int64_t, int64_t, void*) override { return failure_; }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t, int64_t) override { return failure_; }

  private:
  arrow::Status failure_;
  int64_t position_ = 0;
  bool closed_ = false;
};

struct PackedParquetSnapshot {
  std::shared_ptr<arrow::Table> table;
  std::shared_ptr<arrow::KeyValueMetadata> metadata;
};

arrow::Result<PackedParquetSnapshot> ReadPackedParquet(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                       const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto source, fs->OpenInputFile(path));
  std::unique_ptr<::parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(::parquet::arrow::OpenFile(source, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));
  auto metadata = reader->parquet_reader()->metadata()->key_value_metadata();
  if (!metadata) {
    return arrow::Status::Invalid("packed test input has no key-value metadata");
  }
  return PackedParquetSnapshot{std::move(table), metadata->Copy()};
}

arrow::Status WritePackedParquet(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                 const std::string& path,
                                 PackedParquetSnapshot snapshot,
                                 const std::string& field_mapping) {
  if (!snapshot.table || snapshot.table->num_rows() <= 0 || !snapshot.metadata) {
    return arrow::Status::Invalid("packed test snapshot is incomplete");
  }
  ARROW_RETURN_NOT_OK(snapshot.metadata->Set(GROUP_FIELD_ID_LIST_META_KEY, field_mapping));
  ARROW_RETURN_NOT_OK(
      snapshot.metadata->Set(ROW_GROUP_META_KEY, "1|" + std::to_string(snapshot.table->num_rows()) + "|0"));
  ARROW_ASSIGN_OR_RAISE(auto sink, fs->OpenOutputStream(path));
  std::unique_ptr<::parquet::arrow::FileWriter> writer;
  ARROW_ASSIGN_OR_RAISE(
      writer, ::parquet::arrow::FileWriter::Open(*snapshot.table->schema(), arrow::default_memory_pool(), sink,
                                                 ::parquet::default_writer_properties(),
                                                 ::parquet::default_arrow_writer_properties()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*snapshot.table, snapshot.table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->AddKeyValueMetadata(snapshot.metadata));
  ARROW_RETURN_NOT_OK(writer->Close());
  return sink->Close();
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

TEST_F(PackedErrorStatusTest, WriterDuplicateOutputPathIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/same.parquet", path_ + "/same.parquet"};
  std::vector<std::vector<int>> column_groups = {{0}, {1}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterDuplicateColumnIndexIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet", path_ + "/1.parquet"};
  std::vector<std::vector<int>> column_groups = {{0}, {0, 1}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterDuplicateFieldIdIsInvalidArgs) {
  auto duplicate_fields = schema_->fields();
  duplicate_fields[1] = duplicate_fields[1]->WithMetadata(duplicate_fields[0]->metadata());
  auto duplicate_schema = arrow::schema(std::move(duplicate_fields));
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};

  auto result =
      PackedRecordBatchWriter::Make(fs_, paths, duplicate_schema, storage_config_, column_groups, writer_memory_);

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
    EXPECT_EQ(message.find("PackedIO"), std::string::npos) << message;
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

TEST_F(PackedErrorStatusTest, MakeReportsShortParquetFooterAsFileCorrupted) {
  auto corrupt_path = path_ + "/short-footer.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(corrupt_path));
  ASSERT_STATUS_OK(sink->Write("PAR1"));
  ASSERT_STATUS_OK(sink->Close());
  std::vector<std::string> paths = {corrupt_path};

  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedFileCorrupted);
  EXPECT_EQ(ToSegcoreError(result.status()).get_error_code(), milvus::DataFormatBroken);
  EXPECT_NE(result.status().ToString().find(corrupt_path), std::string::npos) << result.status().ToString();
}

TEST_F(PackedErrorStatusTest, MakeReportsBadParquetMagicAsFileCorrupted) {
  auto corrupt_path = path_ + "/bad-magic.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(corrupt_path));
  ASSERT_STATUS_OK(sink->Write("12345678"));
  ASSERT_STATUS_OK(sink->Close());
  std::vector<std::string> paths = {corrupt_path};

  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedFileCorrupted);
  EXPECT_EQ(ToSegcoreError(result.status()).get_error_code(), milvus::DataFormatBroken);
}

TEST_F(PackedErrorStatusTest, MakePreservesTypedFilesystemFailureWhileReadingFooter) {
  auto typed_io = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "footer read timed out",
                                  "operation=ReadAt path=typed-io.parquet");
  auto failing_file = std::make_shared<FailingParquetReadFile>(typed_io);
  auto failing_fs = std::make_shared<FixedInputFileSystem>(fs_, std::move(failing_file));
  std::vector<std::string> paths = {"typed-io.parquet"};

  auto result = PackedRecordBatchReader::Make(failing_fs, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr) << result.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(detail->extra_info(), "operation=ReadAt path=typed-io.parquet");
  EXPECT_EQ(ToSegcoreError(result.status()).get_error_code(), milvus::StorageTransientError);
  EXPECT_EQ(result.status().ToString().find("PackedFileCorrupted"), std::string::npos) << result.status().ToString();
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
  // Second batch with a different schema. Table() must surface Arrow's own
  // Invalid rather than claim that persisted bytes are corrupt.
  auto other_schema = arrow::schema({arrow::field("other", arrow::int8())});
  arrow::Int8Builder builder;
  ASSERT_STATUS_OK(builder.AppendValues({1, 2, 3}));
  ASSERT_AND_ASSIGN(auto other_array, builder.Finish());
  auto other_batch = arrow::RecordBatch::Make(other_schema, 3, {other_array});
  ASSERT_STATUS_OK(group.AddRecordBatch(other_batch));

  auto table_result = group.Table();
  ASSERT_FALSE(table_result.ok());
  EXPECT_TRUE(table_result.status().IsInvalid()) << table_result.status().ToString();
  EXPECT_EQ(ToSegcoreError(table_result.status()).get_error_code(), milvus::StorageError)
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
  auto packed_kv = packed_reader->parquet_reader()->metadata()->key_value_metadata()->Copy();
  ASSERT_NE(packed_kv, nullptr);

  // Write a parquet file carrying genuine packed key-value metadata but a
  // schema whose fields have no PARQUET:field_id. The row-group entry is
  // adjusted below to match this smaller test file.
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
  ASSERT_STATUS_OK(packed_kv->Set(ROW_GROUP_META_KEY, "1|" + std::to_string(bare_table->num_rows()) + "|0"));
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
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(result.status().ToString().find("field"), std::string::npos) << result.status().ToString();
}

TEST_F(PackedErrorStatusTest, PrivateRowGroupMetadataMustMatchParquetFooter) {
  SetupOneFile();

  ASSERT_AND_ASSIGN(auto source, fs_->OpenInputFile(one_file_path_));
  std::unique_ptr<::parquet::arrow::FileReader> packed_reader;
  ASSERT_STATUS_OK(::parquet::arrow::OpenFile(source, arrow::default_memory_pool(), &packed_reader));
  auto corrupt_kv = packed_reader->parquet_reader()->metadata()->key_value_metadata()->Copy();
  ASSERT_STATUS_OK(corrupt_kv->Set(ROW_GROUP_META_KEY, "1|2|0"));

  const auto corrupt_path = path_ + "/wrong-row-count.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(corrupt_path));
  std::unique_ptr<::parquet::arrow::FileWriter> writer;
  ASSERT_AND_ASSIGN(writer, ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink,
                                                               ::parquet::default_writer_properties(),
                                                               ::parquet::default_arrow_writer_properties()));
  ASSERT_STATUS_OK(writer->WriteTable(*table_, table_->num_rows()));
  ASSERT_STATUS_OK(writer->AddKeyValueMetadata(corrupt_kv));
  ASSERT_STATUS_OK(writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  std::vector<std::string> corrupt_paths = {corrupt_path};
  auto packed_result = PackedRecordBatchReader::Make(fs_, corrupt_paths, schema_, reader_memory_);
  ASSERT_FALSE(packed_result.ok());
  ExpectPackedCode(packed_result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(packed_result.status().message().find("row count"), std::string::npos) << packed_result.status().ToString();

  api::Properties properties;
  parquet::ParquetFormatReader generic_reader(fs_, corrupt_path, properties, /*needed_columns=*/{}, nullptr);
  auto generic_status = generic_reader.open();
  ASSERT_FALSE(generic_status.ok());
  ExpectPackedCode(generic_status, ExtendStatusCode::PackedMetadataCorrupted);
}

TEST_F(PackedErrorStatusTest, DuplicatePersistedFieldIdIsMetadataCorrupted) {
  SetupOneFile();
  ASSERT_AND_ASSIGN(auto snapshot, ReadPackedParquet(fs_, one_file_path_));
  const auto corrupt_path = path_ + "/duplicate-field-id.parquet";
  ASSERT_STATUS_OK(WritePackedParquet(fs_, corrupt_path, std::move(snapshot), "100,100,300"));

  std::vector<std::string> paths = {corrupt_path};
  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(result.status().message().find("duplicate field id"), std::string::npos) << result.status().ToString();
}

TEST_F(PackedErrorStatusTest, PersistedFieldOrderMustMatchPhysicalSchema) {
  SetupOneFile();
  ASSERT_AND_ASSIGN(auto snapshot, ReadPackedParquet(fs_, one_file_path_));
  const auto corrupt_path = path_ + "/swapped-field-id.parquet";
  ASSERT_STATUS_OK(WritePackedParquet(fs_, corrupt_path, std::move(snapshot), "200,100,300"));

  std::vector<std::string> paths = {corrupt_path};
  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(result.status().message().find("physical field order"), std::string::npos) << result.status().ToString();
}

TEST_F(PackedErrorStatusTest, ReorderedPackedPathsAreMetadataCorrupted) {
  std::vector<std::string> paths = {path_ + "/group-0.parquet", path_ + "/group-1.parquet"};
  std::vector<std::vector<int>> column_groups = {{2}, {0, 1}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_));
  ASSERT_STATUS_OK(writer->Write(record_batch_));
  ASSERT_STATUS_OK(writer->Close());

  std::reverse(paths.begin(), paths.end());
  auto result = PackedRecordBatchReader::Make(fs_, paths, schema_, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(result.status().message().find("physical"), std::string::npos) << result.status().ToString();
}

TEST_F(PackedErrorStatusTest, EverySelectedFileMustUseTheSameFieldMapping) {
  std::vector<std::string> paths = {path_ + "/group-0.parquet", path_ + "/group-1.parquet"};
  std::vector<std::vector<int>> column_groups = {{2}, {0, 1}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_));
  ASSERT_STATUS_OK(writer->Write(record_batch_));
  ASSERT_STATUS_OK(writer->Close());

  ASSERT_AND_ASSIGN(auto snapshot, ReadPackedParquet(fs_, paths[1]));
  const auto corrupt_path = path_ + "/group-1-inconsistent.parquet";
  ASSERT_STATUS_OK(WritePackedParquet(fs_, corrupt_path, std::move(snapshot), "999;100,200"));
  paths[1] = corrupt_path;
  auto projected_schema = arrow::schema({schema_->field(0)});

  auto result = PackedRecordBatchReader::Make(fs_, paths, projected_schema, reader_memory_);

  ASSERT_FALSE(result.ok());
  ExpectPackedCode(result.status(), ExtendStatusCode::PackedMetadataCorrupted);
  EXPECT_NE(result.status().message().find("inconsistent field mappings"), std::string::npos)
      << result.status().ToString();
}

}  // namespace milvus_storage
