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

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/reader.h"
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
    Properties properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")},
                          {PROPERTY_FS_ROOT_PATH, root_},
                          {PROPERTY_WRITER_POLICY, std::string("single")},
                          {PROPERTY_WRITER_FORMAT, std::string("parquet")},
                          {PROPERTY_WRITER_BUFFER_SIZE, int32_t(kMiB + 1)}};
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
  ASSERT_OK_AND_ASSIGN(auto writer, parquet::ParquetFileWriter::Make(fs, batch->schema(), root_ + "/tail.parquet", {}));
  ASSERT_OK(writer->Write(batch));
  ASSERT_OK(writer->Flush());
  EXPECT_EQ(writer->GetRetainedBufferSize(), GetRecordBatchMemorySize(batch));
  ASSERT_OK(writer->Close().status());
  EXPECT_EQ(writer->GetRetainedBufferSize(), 0);
}
}  // namespace
}  // namespace milvus_storage::test
