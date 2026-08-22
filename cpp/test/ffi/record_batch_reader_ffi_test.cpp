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

#include <arrow/api.h>

#include <memory>
#include <string>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/record_batch_reader.h"
#include "milvus-storage/reader.h"

namespace milvus_storage::test {
namespace {

class FailingRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  explicit FailingRecordBatchReader(arrow::Status status) : status_(std::move(status)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return arrow::schema({}); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    *batch = nullptr;
    return status_;
  }

  private:
  arrow::Status status_;
};

class OneBatchRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  OneBatchRecordBatchReader(std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::RecordBatch> batch)
      : schema_(std::move(schema)), batch_(std::move(batch)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    *batch = std::move(batch_);
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> batch_;
};

class FailingReader final : public api::Reader {
  public:
  explicit FailingReader(arrow::Status read_next_status) : read_next_status_(std::move(read_next_status)) {}

  std::shared_ptr<api::ColumnGroups> get_column_groups() const override {
    return std::make_shared<api::ColumnGroups>();
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& /*predicate*/) const override {
    return std::shared_ptr<arrow::RecordBatchReader>(new FailingRecordBatchReader(read_next_status_));
  }

  arrow::Result<std::unique_ptr<api::ChunkReader>> get_chunk_reader(
      int64_t /*column_group_index*/,
      const std::shared_ptr<std::vector<std::string>>& /*needed_columns*/) const override {
    return arrow::Status::NotImplemented("unused");
  }

  arrow::Result<std::shared_ptr<arrow::Table>> take(
      const std::vector<int64_t>& /*row_indices*/,
      size_t /*parallelism*/,
      const std::shared_ptr<std::vector<std::string>>& /*needed_columns*/) override {
    return arrow::Status::NotImplemented("unused");
  }

  void set_keyretriever(const std::function<std::string(const std::string&)>& /*callback*/) override {}

  private:
  arrow::Status read_next_status_;
};

}  // namespace

TEST(RecordBatchReaderFFITest, ReadNextPreservesStructuredStatus) {
  auto typed_status = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "S3 read timed out",
                                      "operation=GetObject request_id=req-42");
  FailingReader reader(typed_status);

  LoonRecordBatchReaderHandle batch_reader = 0;
  ArrowSchema output_schema{};
  auto result =
      loon_record_batch_reader_new(reinterpret_cast<LoonReaderHandle>(&reader), nullptr, &batch_reader, &output_schema);
  ASSERT_TRUE(loon_ffi_is_success(&result)) << loon_ffi_get_errmsg(&result);
  ASSERT_NE(batch_reader, 0U);
  ASSERT_NE(output_schema.release, nullptr);
  auto schema_release = output_schema.release;

  ArrowArray array{};
  result = loon_record_batch_reader_read_next(batch_reader, &array);
  EXPECT_EQ(result.err_code, LOON_TRANSIENT_TIMEOUT);
  ASSERT_NE(result.message, nullptr);
  EXPECT_NE(std::string(result.message).find("S3 read timed out"), std::string::npos);
  EXPECT_NE(std::string(result.message).find("request_id=req-42"), std::string::npos);
  EXPECT_EQ(array.release, nullptr);
  EXPECT_EQ(output_schema.release, schema_release);

  loon_ffi_free_result(&result);
  // Reusing a handle after a FAILED read is a caller contract violation.
  result = loon_record_batch_reader_read_next(batch_reader, &array);
  EXPECT_EQ(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);
  output_schema.release(&output_schema);
  loon_record_batch_reader_destroy(batch_reader);
}

TEST(RecordBatchReaderFFITest, InvalidArgumentsAndEofAreDeterministic) {
  auto result = loon_record_batch_reader_new(0, nullptr, nullptr, nullptr);
  EXPECT_EQ(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);
  result = loon_record_batch_reader_read_next(0, nullptr);
  EXPECT_EQ(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);

  // The public contract allows a caller to omit the schema output. The handle
  // must still be usable and EOF must not fabricate an ArrowArray release.
  FailingReader reader(arrow::Status::OK());
  LoonRecordBatchReaderHandle handle = 99;
  result = loon_record_batch_reader_new(reinterpret_cast<LoonReaderHandle>(&reader), nullptr, &handle, nullptr);
  ASSERT_TRUE(loon_ffi_is_success(&result));
  ArrowArray array{};
  result = loon_record_batch_reader_read_next(handle, &array);
  EXPECT_TRUE(loon_ffi_is_success(&result));
  EXPECT_EQ(array.release, nullptr);
  // Normal EOF is idempotent, per the Arrow C stream convention: a defensive
  // second poll keeps answering EOF instead of erroring.
  result = loon_record_batch_reader_read_next(handle, &array);
  EXPECT_TRUE(loon_ffi_is_success(&result)) << loon_ffi_get_errmsg(&result);
  EXPECT_EQ(array.release, nullptr);
  loon_record_batch_reader_destroy(handle);
}

TEST(RecordBatchReaderFFITest, ExportFailureMakesConsumedReaderTerminal) {
  auto schema = arrow::schema({arrow::field("value", arrow::int32())});
  arrow::Int32Builder builder;
  ASSERT_TRUE(builder.Append(7).ok());
  auto array_result = builder.Finish();
  ASSERT_TRUE(array_result.ok()) << array_result.status().ToString();

  // Deliberately advertise two rows for a one-row column. ReadNext succeeds
  // and consumes the batch, but ExportRecordBatch rejects the inconsistent
  // shape while converting it to a struct array.
  auto invalid_batch = arrow::RecordBatch::Make(schema, 2, {array_result.ValueOrDie()});
  auto reader = std::make_shared<OneBatchRecordBatchReader>(schema, invalid_batch);
  auto* holder = new ffi_internal::RecordBatchReaderHolder{reader};
  auto handle = reinterpret_cast<LoonRecordBatchReaderHandle>(holder);

  ArrowArray array{};
  auto result = loon_record_batch_reader_read_next(handle, &array);
  EXPECT_EQ(result.err_code, LOON_ARROW_ERROR);
  EXPECT_EQ(array.release, nullptr);
  loon_ffi_free_result(&result);

  result = loon_record_batch_reader_read_next(handle, &array);
  EXPECT_EQ(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);
  loon_record_batch_reader_destroy(handle);
}

}  // namespace milvus_storage::test
