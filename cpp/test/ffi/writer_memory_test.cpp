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
#include <cstring>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/c/helpers.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/io_util.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/segment/segment_reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::test {
namespace {
using namespace milvus_storage::api;
constexpr int64_t kMiB = 1024 * 1024;

struct ReleaseState {
  void (*release)(ArrowArray*);
  void* private_data;
  std::shared_ptr<int> count;
};

std::shared_ptr<int> TrackRelease(ArrowArray* array) {
  auto count = std::make_shared<int>(0);
  array->private_data = new ReleaseState{array->release, array->private_data, count};
  array->release = [](ArrowArray* array) {
    std::unique_ptr<ReleaseState> state(static_cast<ReleaseState*>(array->private_data));
    ++*state->count;
    array->private_data = state->private_data;
    array->release = state->release;
    array->release(array);
  };
  return count;
}

struct ExportedBatch {
  ArrowArray array{};
  ~ExportedBatch() { ArrowArrayRelease(&array); }
};

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeBatch(int64_t rows, bool with_id = true) {
  ARROW_ASSIGN_OR_RAISE(auto data, arrow::AllocateBuffer(rows * 1024));
  std::memset(data->mutable_data(), 42, data->size());
  auto wide = std::make_shared<arrow::FixedSizeBinaryArray>(arrow::fixed_size_binary(1024), rows, std::move(data));
  std::vector<std::shared_ptr<arrow::Field>> fields{arrow::field("wide", wide->type(), false)};
  std::vector<std::shared_ptr<arrow::Array>> columns{wide};
  if (with_id) {
    arrow::Int64Builder builder;
    for (int64_t i = 0; i < rows; ++i) {
      ARROW_RETURN_NOT_OK(builder.Append(i));
    }
    ARROW_ASSIGN_OR_RAISE(auto id, builder.Finish());
    fields.push_back(arrow::field("id", arrow::int64(), false));
    columns.push_back(id);
  }
  return arrow::RecordBatch::Make(arrow::schema(fields), rows, columns);
}

arrow::Status CheckFFI(LoonFFIResult result) {
  auto status = loon_ffi_is_success(&result) ? arrow::Status::OK() : arrow::Status::Invalid(result.message);
  loon_ffi_free_result(&result);
  return status;
}

class WriterMemoryTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(directory_, arrow::internal::TemporaryDir::Make("writer-memory-"));
    root_ = directory_->path().ToString();
    const char* keys[] = {PROPERTY_FS_STORAGE_TYPE,         PROPERTY_FS_ROOT_PATH,
                          PROPERTY_WRITER_POLICY,           PROPERTY_WRITER_SCHEMA_BASE_PATTERNS,
                          PROPERTY_WRITER_FORMAT,           PROPERTY_WRITER_BUFFER_SIZE,
                          PROPERTY_WRITER_FILE_ROLLING_SIZE};
    const char* values[] = {"local", root_.c_str(), "schema_based", "wide,id", "parquet", "2097152", "0"};
    ASSERT_OK(CheckFFI(loon_properties_create(keys, values, 7, &properties_)));
  }

  void TearDown() override {
    loon_reader_destroy(reader_);
    loon_column_groups_destroy(groups_);
    loon_writer_destroy(writer_);
    loon_properties_free(&properties_);
  }

  void Open(const std::shared_ptr<arrow::Schema>& schema) {
    ArrowSchema exported{};
    ASSERT_OK(arrow::ExportSchema(*schema, &exported));
    auto status = CheckFFI(loon_writer_new("data", &exported, &properties_, &writer_));
    ArrowSchemaRelease(&exported);
    ASSERT_OK(status);
  }

  void CloseAndRead(const std::shared_ptr<arrow::Table>& expected) {
    ASSERT_OK(CheckFFI(loon_writer_close(writer_, nullptr, nullptr, 0, &groups_)));
    ArrowSchema schema{};
    ASSERT_OK(arrow::ExportSchema(*expected->schema(), &schema));
    auto status = CheckFFI(loon_reader_new(groups_, &schema, nullptr, 0, &properties_, &reader_));
    ArrowSchemaRelease(&schema);
    ASSERT_OK(status);
    ArrowArrayStream stream{};
    ASSERT_OK(CheckFFI(loon_get_record_batch_reader(reader_, nullptr, &stream)));
    auto imported = arrow::ImportRecordBatchReader(&stream);
    ASSERT_OK(imported.status());
    auto batch_reader = imported.ValueOrDie();
    ASSERT_OK_AND_ASSIGN(auto actual, batch_reader->ToTable());
    EXPECT_TRUE(actual->Equals(*expected));
    ASSERT_OK(batch_reader->Close());
  }

  std::unique_ptr<arrow::internal::TemporaryDir> directory_;
  std::string root_;
  LoonProperties properties_{};
  LoonWriterHandle writer_{};
  LoonReaderHandle reader_{};
  LoonColumnGroups* groups_{};
};

TEST_F(WriterMemoryTest, ReleasesRootButRetainsColumnsWithoutCopy) {
  ASSERT_OK_AND_ASSIGN(auto batch, MakeBatch(3));
  ASSERT_NO_FATAL_FAILURE(Open(batch->schema()));
  ExportedBatch exported;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
  auto root = TrackRelease(&exported.array);
  auto wide = TrackRelease(exported.array.children[0]);
  auto id = TrackRelease(exported.array.children[1]);
  ASSERT_OK(CheckFFI(loon_writer_write(writer_, &exported.array)));
  EXPECT_EQ(exported.array.release, nullptr);
  EXPECT_EQ(*root, 1);
  EXPECT_EQ(*wide, 0);
  EXPECT_EQ(*id, 0);
  ASSERT_OK_AND_ASSIGN(auto expected, arrow::Table::FromRecordBatches({batch}));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(expected));
  EXPECT_EQ(*wide, 1);
  EXPECT_EQ(*id, 1);
}

TEST_F(WriterMemoryTest, FlushesLargestGroupAndReleasesItIndependently) {
  ASSERT_OK_AND_ASSIGN(auto batch, MakeBatch(1536));
  ASSERT_NO_FATAL_FAILURE(Open(batch->schema()));
  ExportedBatch first;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &first.array));
  auto wide = TrackRelease(first.array.children[0]);
  auto id = TrackRelease(first.array.children[1]);
  ASSERT_OK(CheckFFI(loon_writer_write(writer_, &first.array)));
  ExportedBatch second;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &second.array));
  ASSERT_OK(CheckFFI(loon_writer_write(writer_, &second.array)));
  // Rolling drains the selected group completely. The larger, lower-index
  // group must be released without flushing or retaining the narrow group.
  EXPECT_EQ(*wide, 1);
  EXPECT_EQ(*id, 0);
  ASSERT_OK_AND_ASSIGN(auto expected, arrow::Table::FromRecordBatches({batch, batch}));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(expected));
  EXPECT_EQ(*id, 1);
}

TEST_F(WriterMemoryTest, ImportFailuresReleaseEveryChildExactlyOnce) {
  ASSERT_OK_AND_ASSIGN(auto batch, MakeBatch(3));
  ASSERT_NO_FATAL_FAILURE(Open(batch->schema()));
  for (int failure = 0; failure < 5; ++failure) {
    SCOPED_TRACE(failure);
    ExportedBatch exported;
    ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
    auto root = TrackRelease(&exported.array);
    auto wide = TrackRelease(exported.array.children[0]);
    auto id = TrackRelease(exported.array.children[1]);
    switch (failure) {
      case 0:
        exported.array.offset = 1;
        break;
      case 1:
        exported.array.null_count = 1;
        break;
      case 2:
        exported.array.children[1]->length = 2;
        break;
      case 3:
        exported.array.children[0]->n_buffers = 1;
        break;
      case 4:
        exported.array.children[1]->n_buffers = 1;
        break;
    }
    EXPECT_FALSE(CheckFFI(loon_writer_write(writer_, &exported.array)).ok());
    EXPECT_EQ(exported.array.release, nullptr);
    EXPECT_EQ(*root, 1);
    EXPECT_EQ(*wide, 1);
    EXPECT_EQ(*id, 1);
  }
  ExportedBatch released;
  EXPECT_FALSE(CheckFFI(loon_writer_write(writer_, &released.array)).ok());
}

TEST_F(WriterMemoryTest, SlicedColumnsAndUnknownRootNullCount) {
  ASSERT_OK_AND_ASSIGN(auto original, MakeBatch(16));
  auto batch = original->Slice(7, 3);
  ASSERT_NO_FATAL_FAILURE(Open(batch->schema()));
  ExportedBatch exported;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
  exported.array.null_count = -1;
  ASSERT_OK(CheckFFI(loon_writer_write(writer_, &exported.array)));
  ASSERT_OK_AND_ASSIGN(auto expected, arrow::Table::FromRecordBatches({batch}));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(expected));
}

TEST_F(WriterMemoryTest, NativeWriterAccountsForTailsAfterAutomaticAndExplicitFlush) {
  for (bool explicit_flush : {false, true}) {
    SCOPED_TRACE(explicit_flush);
    const auto properties =
        std::make_shared<const Properties>(Properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")},
                                                      {PROPERTY_FS_ROOT_PATH, root_},
                                                      {PROPERTY_WRITER_POLICY, std::string("single")},
                                                      {PROPERTY_WRITER_FORMAT, std::string("parquet")},
                                                      {PROPERTY_WRITER_BUFFER_SIZE, int32_t(kMiB + 1)}});
    ASSERT_OK_AND_ASSIGN(auto first, MakeBatch(768, false));
    ASSERT_OK_AND_ASSIGN(auto second, MakeBatch(768, false));
    ASSERT_OK_AND_ASSIGN(auto third, MakeBatch(256, false));
    ASSERT_OK_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(properties, first->schema()));
    auto writer = Writer::create("native", first->schema(), std::move(policy), properties);
    std::weak_ptr<arrow::Buffer> first_buffer = first->column(0)->data()->buffers[1];
    ASSERT_OK(writer->write(first));
    first.reset();
    if (explicit_flush) {
      ASSERT_OK(writer->flush());
    }
    ASSERT_OK(writer->write(second));
    EXPECT_FALSE(first_buffer.expired());
    // The first flush cannot emit a row group. Its tail must still contribute
    // to the next admission, which now has enough cached rows to release it.
    ASSERT_OK(writer->write(third));
    EXPECT_TRUE(first_buffer.expired());
    ASSERT_OK_AND_ASSIGN(auto groups, writer->close());
    auto reader = Reader::create(groups, third->schema(), nullptr, properties);
    ASSERT_OK_AND_ASSIGN(auto batches, reader->get_record_batch_reader());
    ASSERT_OK_AND_ASSIGN(auto actual, batches->ToTable());
    ASSERT_OK_AND_ASSIGN(auto all_rows, MakeBatch(1792, false));
    ASSERT_OK_AND_ASSIGN(auto expected, arrow::Table::FromRecordBatches({all_rows}));
    EXPECT_TRUE(actual->Equals(*expected));
    ASSERT_OK(batches->Close());
  }
}

TEST_F(WriterMemoryTest, ParquetCountsBackingBuffersOfRetainedSlices) {
  ASSERT_OK_AND_ASSIGN(auto batch, MakeBatch(2049, false));
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  ASSERT_OK_AND_ASSIGN(auto writer, parquet::ParquetFileWriter::Make(fs, batch->schema(), root_ + "/tail.parquet",
                                                                     std::make_shared<const Properties>()));
  ASSERT_OK(writer->Write(batch));
  ASSERT_OK(writer->Flush());
  EXPECT_EQ(writer->GetRetainedBufferSize(), GetRecordBatchMemorySize(batch));
  ASSERT_OK(writer->Close().status());
  EXPECT_EQ(writer->GetRetainedBufferSize(), 0);
}

class SegmentWriterMemoryTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(directory_, arrow::internal::TemporaryDir::Make("segment-writer-memory-"));
    root_ = directory_->path().ToString();
    segment_path_ = "segment";
    lob_path_ = "lobs/101";
    schema_ = arrow::schema({
        arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("content", arrow::utf8(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
    });
    const char* keys[] = {PROPERTY_FS_STORAGE_TYPE,         PROPERTY_FS_ROOT_PATH,
                          PROPERTY_WRITER_POLICY,           PROPERTY_WRITER_SCHEMA_BASE_PATTERNS,
                          PROPERTY_WRITER_FORMAT,           PROPERTY_WRITER_BUFFER_SIZE,
                          PROPERTY_WRITER_FILE_ROLLING_SIZE};
    // Neither the scalar tail nor the LOB references reach a row group or file
    // rolling threshold. Releasing raw TEXT must not depend on draining them.
    const char* values[] = {"local", root_.c_str(), "schema_based", "id,content", "parquet", "33554432", "2147483648"};
    ASSERT_OK(CheckFFI(loon_properties_create(keys, values, 7, &properties_)));
    LoonLobColumnConfig lob{};
    lob.lob_base_path = lob_path_.c_str();
    lob.field_id = 101;
    lob.inline_threshold = 256;
    lob.max_lob_file_bytes = 64 * kMiB;
    lob.flush_threshold_bytes = 16 * kMiB;
    LoonSegmentWriterConfig config{};
    config.segment_path = segment_path_.c_str();
    config.lob_columns = &lob;
    config.num_lob_columns = 1;
    ArrowSchema exported{};
    ASSERT_OK(arrow::ExportSchema(*schema_, &exported));
    auto status = CheckFFI(loon_segment_writer_new(&exported, &config, &properties_, &writer_));
    ArrowSchemaRelease(&exported);
    ASSERT_OK(status);
  }

  void TearDown() override {
    loon_segment_writer_destroy(writer_);
    loon_column_groups_destroy(output_.column_groups);
    loon_segment_write_output_free(&output_);
    loon_properties_free(&properties_);
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeTextBatch(int64_t rows,
                                                                   int64_t start_id = 0,
                                                                   bool mixed = false) {
    arrow::Int64Builder ids;
    arrow::StringBuilder texts;
    for (int64_t i = 0; i < rows; ++i) {
      auto id = start_id + i;
      ARROW_RETURN_NOT_OK(ids.Append(id));
      if (mixed && id % 3 == 0) {
        ARROW_RETURN_NOT_OK(texts.AppendNull());
      } else if (mixed && id % 3 == 1) {
        ARROW_RETURN_NOT_OK(texts.Append("inline-" + std::to_string(id)));
      } else {
        auto text = std::to_string(id) + std::string(128 * 1024, 'a' + id % 26);
        ARROW_RETURN_NOT_OK(texts.Append(text));
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto id_array, ids.Finish());
    ARROW_ASSIGN_OR_RAISE(auto text_array, texts.Finish());
    return arrow::RecordBatch::Make(schema_, rows, {id_array, text_array});
  }

  void CloseAndRead(const std::shared_ptr<arrow::RecordBatch>& expected_batch) {
    ASSERT_OK(CheckFFI(loon_segment_writer_close(writer_, &output_)));
    EXPECT_EQ(output_.rows_written, expected_batch->num_rows());
    EXPECT_GT(output_.num_lob_files, 0);
    auto groups = std::make_shared<ColumnGroups>();
    ASSERT_OK(column_groups_import(output_.column_groups, groups.get()));
    segment::SegmentReaderConfig config;
    auto error = ConvertFFIProperties(config.properties, &properties_);
    ASSERT_FALSE(error.has_value()) << error.value_or("");
    lob_column::LobColumnConfig lob;
    lob.field_id = 101;
    lob.lob_base_path = lob_path_;
    lob.properties = config.properties;
    config.lob_columns[101] = lob;
    ASSERT_OK_AND_ASSIGN(auto fs, FilesystemCache::getInstance().get(config.properties, segment_path_));
    ASSERT_OK_AND_ASSIGN(auto reader, segment::SegmentReader::Create(fs, groups, schema_, {}, config));
    ASSERT_OK_AND_ASSIGN(auto actual, reader->ToTable());
    ASSERT_OK_AND_ASSIGN(auto expected, arrow::Table::FromRecordBatches({expected_batch}));
    EXPECT_TRUE(actual->schema()->Equals(*expected->schema()));
    ASSERT_EQ(actual->num_rows(), expected->num_rows());
    ASSERT_EQ(actual->num_columns(), expected->num_columns());
    EXPECT_TRUE(actual->column(0)->Equals(expected->column(0)));
    // LobColumnReader returns BinaryArray for resolved text. Compare its full
    // contents and nulls through the equivalent UTF-8 view.
    ASSERT_OK_AND_ASSIGN(auto content, actual->column(1)->View(arrow::utf8()));
    EXPECT_TRUE(content->Equals(expected->column(1)));
    ASSERT_OK(reader->Close());
  }

  std::unique_ptr<arrow::internal::TemporaryDir> directory_;
  std::string root_;
  std::string segment_path_;
  std::string lob_path_;
  std::shared_ptr<arrow::Schema> schema_;
  LoonProperties properties_{};
  LoonSegmentWriterHandle writer_{};
  LoonSegmentWriteOutput output_{};
};

class SegmentWriterFlushMemoryTest : public SegmentWriterMemoryTest, public ::testing::WithParamInterface<bool> {};

TEST_P(SegmentWriterFlushMemoryTest, ReleasesRawTextWhileScalarTailIsCached) {
  constexpr int64_t rows = 8;
  constexpr int batches = 8;
  std::vector<std::shared_ptr<int>> id_releases;
  std::vector<std::weak_ptr<arrow::Buffer>> id_buffers;
  for (int batch_index = 0; batch_index < batches; ++batch_index) {
    SCOPED_TRACE(batch_index);
    ASSERT_OK_AND_ASSIGN(auto batch, MakeTextBatch(rows, batch_index * rows));
    std::weak_ptr<arrow::Buffer> text_buffer = batch->column(1)->data()->buffers[2];
    id_buffers.emplace_back(batch->column(0)->data()->buffers[1]);
    ExportedBatch exported;
    ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
    auto root = TrackRelease(&exported.array);
    auto id = TrackRelease(exported.array.children[0]);
    auto text = TrackRelease(exported.array.children[1]);
    ASSERT_OK(CheckFFI(loon_segment_writer_write(writer_, &exported.array)));
    batch.reset();
    EXPECT_EQ(exported.array.release, nullptr);
    EXPECT_EQ(*root, 1);
    EXPECT_EQ(*text, 1);
    EXPECT_TRUE(text_buffer.expired());
    EXPECT_EQ(*id, 0);
    EXPECT_FALSE(id_buffers.back().expired());
    if (GetParam()) {
      ASSERT_OK(CheckFFI(loon_segment_writer_flush(writer_)));
      EXPECT_TRUE(text_buffer.expired());
      EXPECT_EQ(*id, 0);
      EXPECT_FALSE(id_buffers.back().expired());
    }
    id_releases.emplace_back(std::move(id));
  }
  // Build expected values separately so the test itself does not retain input.
  ASSERT_OK_AND_ASSIGN(auto expected, MakeTextBatch(rows * batches));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(expected));
  for (size_t i = 0; i < id_releases.size(); ++i) {
    EXPECT_EQ(*id_releases[i], 1);
    EXPECT_TRUE(id_buffers[i].expired());
  }
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutFlush, SegmentWriterFlushMemoryTest, ::testing::Bool());

TEST_F(SegmentWriterMemoryTest, SlicedNullableAndInlineTextRoundTrip) {
  ASSERT_OK_AND_ASSIGN(auto original, MakeTextBatch(10, 0, true));
  auto batch = original->Slice(2, 6);
  std::weak_ptr<arrow::Buffer> text_buffer = batch->column(1)->data()->buffers[2];
  ExportedBatch exported;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
  exported.array.null_count = -1;
  auto root = TrackRelease(&exported.array);
  auto id = TrackRelease(exported.array.children[0]);
  auto text = TrackRelease(exported.array.children[1]);
  ASSERT_OK(CheckFFI(loon_segment_writer_write(writer_, &exported.array)));
  original.reset();
  batch.reset();
  EXPECT_EQ(*root, 1);
  EXPECT_EQ(*text, 1);
  EXPECT_EQ(*id, 0);
  EXPECT_TRUE(text_buffer.expired());
  ASSERT_OK(CheckFFI(loon_segment_writer_flush(writer_)));
  EXPECT_EQ(*id, 0);
  ASSERT_OK_AND_ASSIGN(auto expected, MakeTextBatch(6, 2, true));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(expected));
  EXPECT_EQ(*id, 1);
}

TEST_F(SegmentWriterMemoryTest, ImportFailuresReleaseEachColumnExactlyOnce) {
  ASSERT_OK_AND_ASSIGN(auto batch, MakeTextBatch(3));
  for (int column = 0; column < batch->num_columns(); ++column) {
    SCOPED_TRACE(column);
    ExportedBatch exported;
    ASSERT_OK(arrow::ExportRecordBatch(*batch, &exported.array));
    auto root = TrackRelease(&exported.array);
    auto id = TrackRelease(exported.array.children[0]);
    auto text = TrackRelease(exported.array.children[1]);
    // Exercise both failure before any column is imported and partial import.
    exported.array.children[column]->n_buffers = 1;
    EXPECT_FALSE(CheckFFI(loon_segment_writer_write(writer_, &exported.array)).ok());
    EXPECT_EQ(exported.array.release, nullptr);
    EXPECT_EQ(*root, 1);
    EXPECT_EQ(*id, 1);
    EXPECT_EQ(*text, 1);
  }
  ExportedBatch released;
  EXPECT_FALSE(CheckFFI(loon_segment_writer_write(writer_, &released.array)).ok());
  // A failed import must not poison subsequent writes.
  ExportedBatch valid;
  ASSERT_OK(arrow::ExportRecordBatch(*batch, &valid.array));
  ASSERT_OK(CheckFFI(loon_segment_writer_write(writer_, &valid.array)));
  ASSERT_NO_FATAL_FAILURE(CloseAndRead(batch));
}
}  // namespace
}  // namespace milvus_storage::test
