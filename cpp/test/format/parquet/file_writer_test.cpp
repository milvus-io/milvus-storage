// Copyright 2025 Zilliz
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
#include <cstring>
#include <mutex>
#include <vector>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/io/memory.h>
#include <arrow/io/file.h>
#include <arrow/memory_pool.h>
#include <arrow/filesystem/filesystem.h>
#include <parquet/column_page.h>
#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>

#include "test_env.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

namespace milvus_storage::test {
namespace {

struct ResizeEvent {
  int64_t old_size;
  int64_t new_size;
};

class TrackingMemoryPool final : public arrow::MemoryPool {
  public:
  explicit TrackingMemoryPool(arrow::MemoryPool* upstream) : upstream_(upstream) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    ARROW_RETURN_NOT_OK(upstream_->Allocate(size, alignment, out));
    Record(0, size);
    return arrow::Status::OK();
  }

  arrow::Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) override {
    ARROW_RETURN_NOT_OK(upstream_->Reallocate(old_size, new_size, alignment, ptr));
    Record(old_size, new_size);
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override { upstream_->Free(buffer, size, alignment); }

  int64_t bytes_allocated() const override { return upstream_->bytes_allocated(); }

  int64_t max_memory() const override { return upstream_->max_memory(); }

  int64_t total_bytes_allocated() const override { return upstream_->total_bytes_allocated(); }

  int64_t num_allocations() const override { return upstream_->num_allocations(); }

  std::string backend_name() const override { return upstream_->backend_name(); }

  void ClearEvents() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
  }

  std::vector<ResizeEvent> EventsSince(size_t offset) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (offset >= events_.size()) {
      return {};
    }
    return std::vector<ResizeEvent>(events_.begin() + static_cast<int64_t>(offset), events_.end());
  }

  size_t EventCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
  }

  private:
  void Record(int64_t old_size, int64_t new_size) {
    if (new_size <= old_size) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back({old_size, new_size});
  }

  arrow::MemoryPool* upstream_;
  mutable std::mutex mutex_;
  std::vector<ResizeEvent> events_;
};

class FailingOpenOutputFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  FailingOpenOutputFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs, arrow::Status failure)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)), failure_(std::move(failure)) {}

  std::string type_name() const override { return "failing-open-output"; }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    opened_path_ = path;
    return failure_;
  }

  const std::string& opened_path() const { return opened_path_; }

  private:
  arrow::Status failure_;
  std::string opened_path_;
};

class AbortTrackingOutputStream final : public arrow::io::OutputStream {
  public:
  explicit AbortTrackingOutputStream(bool close_on_abort = true) : close_on_abort_(close_on_abort) {}

  void ArmWriteFailure(arrow::Status failure) { write_failure_ = std::move(failure); }

  arrow::Status Close() override {
    ++close_count_;
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    ++abort_count_;
    closed_ = close_on_abort_;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("stream is closed");
    }
    return position_;
  }

  bool closed() const override { return closed_; }

  arrow::Status Write(const void*, int64_t nbytes) override {
    if (closed_) {
      return arrow::Status::Invalid("stream is closed");
    }
    ++write_count_;
    if (!write_failure_.ok()) {
      return write_failure_;
    }
    position_ += nbytes;
    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    if (closed_) {
      return arrow::Status::Invalid("stream is closed");
    }
    return arrow::Status::OK();
  }

  int abort_count() const { return abort_count_; }
  int close_count() const { return close_count_; }
  int write_count() const { return write_count_; }

  private:
  bool closed_ = false;
  bool close_on_abort_;
  int64_t position_ = 0;
  int abort_count_ = 0;
  int close_count_ = 0;
  int write_count_ = 0;
  arrow::Status write_failure_ = arrow::Status::OK();
};

class FailingWriteOutputStream final : public arrow::io::OutputStream {
  public:
  explicit FailingWriteOutputStream(arrow::Status failure) : failure_(std::move(failure)) {}

  arrow::Status Close() override {
    ++close_count_;
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    ++abort_count_;
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("stream is closed");
    }
    return 0;
  }

  bool closed() const override { return closed_; }

  arrow::Status Write(const void*, int64_t) override {
    ++write_count_;
    return failure_;
  }

  int abort_count() const { return abort_count_; }
  int close_count() const { return close_count_; }
  int write_count() const { return write_count_; }

  private:
  arrow::Status failure_;
  bool closed_ = false;
  int abort_count_ = 0;
  int close_count_ = 0;
  int write_count_ = 0;
};

// A stream whose Abort() throws something that is not std::exception.
//
// arrow::io::OutputStream::Abort() is allowed to throw -- it is a filesystem
// implementation we do not own -- and this repository already contains one that
// does (see c_abi_exception_boundary_test). Writers, by contrast, abandon
// through a void noexcept Abort(), so an escaping exception would call
// std::terminate and take the process down before the FFI boundary could report
// the caller's own failure.
class ThrowingAbortOutputStream final : public arrow::io::OutputStream {
  public:
  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    ++abort_count_;
    throw 42;
  }

  arrow::Result<int64_t> Tell() const override { return position_; }
  bool closed() const override { return closed_; }

  arrow::Status Write(const void*, int64_t nbytes) override {
    position_ += nbytes;
    return arrow::Status::OK();
  }

  arrow::Status Flush() override { return arrow::Status::OK(); }

  int abort_count() const { return abort_count_; }

  private:
  bool closed_ = false;
  int64_t position_ = 0;
  int abort_count_ = 0;
};

class TrackingOutputFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  TrackingOutputFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs,
                           std::shared_ptr<arrow::io::OutputStream> stream)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)), stream_(std::move(stream)) {}

  std::string type_name() const override { return "tracking-output"; }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string&, const std::shared_ptr<const arrow::KeyValueMetadata>&) override {
    return stream_;
  }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
};

class FailSecondOpenOutputFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  FailSecondOpenOutputFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs,
                                 std::shared_ptr<arrow::io::OutputStream> first_stream,
                                 arrow::Status second_failure)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)),
        first_stream_(std::move(first_stream)),
        second_failure_(std::move(second_failure)) {}

  std::string type_name() const override { return "fail-second-open-output"; }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string&, const std::shared_ptr<const arrow::KeyValueMetadata>&) override {
    ++open_count_;
    if (open_count_ == 2) {
      return second_failure_;
    }
    return first_stream_;
  }

  int open_count() const { return open_count_; }

  private:
  std::shared_ptr<arrow::io::OutputStream> first_stream_;
  arrow::Status second_failure_;
  int open_count_ = 0;
};

}  // namespace

class ParquetFileWriterTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("parquet-file-writer-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    // Create schema with mixed data types
    // Current test case exist some nullable columns
    // should set all field `nullable` to true.
    auto id_field =
        arrow::field("id", arrow::int64(), true /*nullable*/, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));
    auto text_field = arrow::field("text", arrow::utf8(), true /*nullable*/,
                                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}));
    auto vector_field = arrow::field("vector", arrow::fixed_size_binary(128), true /*nullable*/,
                                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"}));

    schema_ = arrow::schema({id_field, text_field, vector_field});
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  milvus_storage::api::Properties properties_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

TEST_F(ParquetFileWriterTest, OpenOutputStreamFailurePreservesExtendStatusDetail) {
  auto original = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "Azure request timed out",
                                  "operation=OpenOutputStream http_status=408");
  auto failing_fs = std::make_shared<FailingOpenOutputFileSystem>(fs_, original);
  const std::string file_path = "container/path/data.parquet";
  StorageConfig config;

  auto result = parquet::ParquetFileWriter::Make(schema_, failing_fs, file_path, config);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(failing_fs->opened_path(), file_path);
  EXPECT_EQ(result.status().code(), original.code());
  EXPECT_EQ(result.status().detail(), original.detail());
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr) << result.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(detail->extra_info(), "operation=OpenOutputStream http_status=408");
  EXPECT_TRUE(result.status().Equals(original));
}

TEST_F(ParquetFileWriterTest, HeaderWriteFailureAbortsWithoutFinalizingAndPreservesDetail) {
  auto original = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "Parquet header write timed out",
                                  "operation=WriteHeader http_status=408");
  auto stream = std::make_shared<FailingWriteOutputStream>(original);
  auto tracking_fs = std::make_shared<TrackingOutputFileSystem>(fs_, stream);
  const std::string file_path = base_path_ + "/header-write-failure.parquet";
  StorageConfig config;

  auto result = parquet::ParquetFileWriter::Make(schema_, tracking_fs, file_path, config);

  ASSERT_FALSE(result.ok());
  EXPECT_GT(stream->write_count(), 0);
  EXPECT_EQ(stream->abort_count(), 1);
  EXPECT_EQ(stream->close_count(), 0);
  EXPECT_TRUE(stream->closed());
  EXPECT_EQ(result.status().code(), original.code());
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr) << result.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(detail->extra_info(), "operation=WriteHeader http_status=408");
  EXPECT_NE(result.status().message().find("Parquet header write timed out"), std::string::npos);
}

TEST_F(ParquetFileWriterTest, FlushFailureAbortStillReachesDelegateWithoutFinalizing) {
  auto original = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "Parquet data write timed out",
                                  "operation=WriteRowGroup http_status=408");
  auto stream = std::make_shared<AbortTrackingOutputStream>();
  auto tracking_fs = std::make_shared<TrackingOutputFileSystem>(fs_, stream);
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema, 0, false, 512, 1024));
  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, parquet::ParquetFileWriter::Make(test_schema, tracking_fs,
                                                                  base_path_ + "/flush-failure.parquet", config));
  const int writes_after_init = stream->write_count();
  stream->ArmWriteFailure(original);

  ASSERT_STATUS_OK(writer->Write(record_batch));
  auto flush_status = writer->Flush();

  ASSERT_FALSE(flush_status.ok());
  EXPECT_GT(stream->write_count(), writes_after_init);
  auto detail = ExtendStatusDetail::UnwrapStatus(flush_status);
  ASSERT_NE(detail, nullptr) << flush_status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(detail->extra_info(), "operation=WriteRowGroup http_status=408");

  writer->Abort();
  EXPECT_EQ(stream->abort_count(), 1);
  EXPECT_EQ(stream->close_count(), 0);
  EXPECT_TRUE(writer->Flush().Equals(flush_status));
}

TEST_F(ParquetFileWriterTest, PackedMakeAbortsOpenedGroupsWhenLaterOpenFails) {
  auto original = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "Second output open timed out",
                                  "operation=OpenOutputStream group=1 http_status=408");
  auto first_stream = std::make_shared<AbortTrackingOutputStream>();
  auto failing_fs = std::make_shared<FailSecondOpenOutputFileSystem>(fs_, first_stream, original);
  StorageConfig config;
  std::vector<std::string> paths = {base_path_ + "/first.parquet", base_path_ + "/second.parquet"};
  std::vector<std::vector<int>> column_groups = {{0}, {1}};

  auto result = PackedRecordBatchWriter::Make(failing_fs, paths, schema_, config, column_groups, 1024 * 1024);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(failing_fs->open_count(), 2);
  EXPECT_EQ(first_stream->abort_count(), 1);
  EXPECT_EQ(first_stream->close_count(), 0);
  EXPECT_TRUE(first_stream->closed());
  EXPECT_TRUE(result.status().Equals(original)) << result.status().ToString();
}

TEST_F(ParquetFileWriterTest, DestroyDoesNotFinalizeOrAbort) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));
  auto stream = std::make_shared<AbortTrackingOutputStream>(/*close_on_abort=*/false);
  auto metrics_stream = std::make_shared<MetricsOutputStream>(stream, std::make_shared<FilesystemMetrics>());
  auto tracking_fs = std::make_shared<TrackingOutputFileSystem>(fs_, metrics_stream);

  int writes_before_destroy = 0;
  {
    StorageConfig config;
    ASSERT_AND_ASSIGN(auto writer, parquet::ParquetFileWriter::Make(test_schema, tracking_fs,
                                                                    base_path_ + "/discarded.parquet", config));
    ASSERT_STATUS_OK(writer->Write(record_batch));
    writes_before_destroy = stream->write_count();
  }

  EXPECT_EQ(stream->write_count(), writes_before_destroy);
  EXPECT_EQ(stream->abort_count(), 0);
  EXPECT_EQ(stream->close_count(), 0);
}

// The assertion here is that the test finishes at all. Abandoning reaches a
// stream that throws, and ParquetFileWriter::Abort() is noexcept, so without
// AbandonQuietly absorbing it this would std::terminate and take the whole test
// binary with it -- there would be no failure to report, just a dead process.
// Everything after the Abort() call is therefore the real check.
TEST_F(ParquetFileWriterTest, AbandonSurvivesAStreamThatThrows) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));
  auto stream = std::make_shared<ThrowingAbortOutputStream>();
  auto metrics_stream = std::make_shared<MetricsOutputStream>(stream, std::make_shared<FilesystemMetrics>());
  auto tracking_fs = std::make_shared<TrackingOutputFileSystem>(fs_, metrics_stream);

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, parquet::ParquetFileWriter::Make(test_schema, tracking_fs,
                                                                  base_path_ + "/throwing_abort.parquet", config));
  ASSERT_STATUS_OK(writer->Write(record_batch));

  writer->Abort();

  EXPECT_EQ(stream->abort_count(), 1) << "the abandonment must still reach the stream";
  // And the writer is spent, exactly as it would be after a clean abandonment.
  EXPECT_FALSE(writer->Close().ok());
  // Still idempotent: a second abandonment is a no-op and also must not throw.
  writer->Abort();
  EXPECT_EQ(stream->abort_count(), 1);
}

// Abort is the giving-up path, and it has to reach the stream: the stream is
// the only object that can release what the write allocated in the store (an S3
// multipart upload's parts, which no bucket listing can show). It must not
// close -- closing finalizes a file that was abandoned on purpose -- and it has
// to survive being called twice, because the caller is already handling a
// failure and should not have to track whether it aborted yet.
TEST_F(ParquetFileWriterTest, AbortReachesTheStreamWithoutFinalizing) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));
  auto stream = std::make_shared<AbortTrackingOutputStream>(/*close_on_abort=*/true);
  auto metrics_stream = std::make_shared<MetricsOutputStream>(stream, std::make_shared<FilesystemMetrics>());
  auto tracking_fs = std::make_shared<TrackingOutputFileSystem>(fs_, metrics_stream);

  StorageConfig config;
  ASSERT_AND_ASSIGN(
      auto writer, parquet::ParquetFileWriter::Make(test_schema, tracking_fs, base_path_ + "/aborted.parquet", config));
  ASSERT_STATUS_OK(writer->Write(record_batch));

  writer->Abort();
  EXPECT_EQ(stream->abort_count(), 1);
  EXPECT_EQ(stream->close_count(), 0);

  writer->Abort();
  EXPECT_EQ(stream->abort_count(), 1) << "abort must be idempotent";

  // The file was abandoned, so finishing it is not on offer any more.
  EXPECT_FALSE(writer->Close().ok());
  EXPECT_EQ(stream->close_count(), 0);
}

TEST_F(ParquetFileWriterTest, LargeRecordBatchSplitting) {
  // Create a large record batch with mixed data sizes
  const int64_t num_rows = 1000;

  // Create ID array (small, uniform size)
  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();

  // Create text array (mixed sizes - some very large)
  arrow::StringBuilder text_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    if (i % 20 == 0) {
      // Every 20th row has a very large text (simulating large text field)
      std::string large_text(50000, 'x');  // 50KB text
      ASSERT_TRUE(text_builder.Append(large_text).ok());
    } else {
      // Normal rows have small text
      std::string small_text = "row_" + std::to_string(i);
      ASSERT_TRUE(text_builder.Append(small_text).ok());
    }
  }
  auto text_array = text_builder.Finish().ValueOrDie();

  // Create vector array (uniform size)
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));
  std::vector<uint8_t> vector_data(128, 0);
  for (int64_t i = 0; i < num_rows; ++i) {
    // Fill with some pattern
    for (int j = 0; j < 128; ++j) {
      vector_data[j] = static_cast<uint8_t>((i + j) % 256);
    }
    ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
  }
  auto vector_array = vector_builder.Finish().ValueOrDie();

  // Create record batch
  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  // Create temporary file path
  std::string temp_file = base_path_ + "/data/test_large_batch.parquet";

  // Create packed writer and write record batch
  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 2 * 1024 * 1024));
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(writer->Write(record_batch).ok());
  }
  ASSERT_TRUE(writer->Close().ok());

  // Read back and verify
  ASSERT_AND_ASSIGN(auto reader, FileRowGroupReader::Make(fs_, temp_file, schema_));

  // Get metadata
  auto file_metadata = reader->file_metadata();
  auto row_group_metadata = file_metadata->GetRowGroupMetadataVector();
  int num_row_groups = row_group_metadata.size();

  // Verify each row group size
  for (int i = 0; i < num_row_groups; ++i) {
    const auto& metadata = row_group_metadata.Get(i);
    int64_t row_group_size = metadata.memory_size();

    // Verify that row group size is reasonable (should be around 1MB)
    EXPECT_LE(row_group_size, DEFAULT_MAX_ROW_GROUP_SIZE * 1.1);  // Allow some tolerance

    // only the last row group should be less than 1MB
    if (i < num_row_groups - 1) {
      EXPECT_GT(row_group_size, DEFAULT_MAX_ROW_GROUP_SIZE);
    }
  }
}

TEST_F(ParquetFileWriterTest, EmptyRecordBatch) {
  // Test writing empty record batch
  // Create empty arrays for each column in the schema
  auto id_array = arrow::MakeArrayOfNull(arrow::int64(), 0).ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), 0).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), 0).ValueOrDie();

  auto empty_batch = arrow::RecordBatch::Make(schema_, 0, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_empty_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(empty_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
}

TEST_F(ParquetFileWriterTest, CompressedStringPageReaderGrowsDecompressionBuffer) {
  auto str_schema = arrow::schema({arrow::field("text", arrow::utf8(), false)});

  arrow::StringBuilder builder;
  const std::vector<int64_t> string_sizes = {512, 8 * 1024, 32 * 1024, 128 * 1024};
  for (size_t group = 0; group < string_sizes.size(); ++group) {
    for (int row = 0; row < 8; ++row) {
      std::string value(static_cast<size_t>(string_sizes[group]), static_cast<char>('a' + group));
      ASSERT_STATUS_OK(builder.Append(value));
    }
  }

  ASSERT_AND_ASSIGN(auto text_array, builder.Finish());
  auto table = arrow::Table::Make(str_schema, {text_array});
  const std::string temp_file = get_data_filepath(base_path_, "test_compressed_string_pages.parquet");

  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(temp_file));
  ::parquet::WriterProperties::Builder props_builder;
  props_builder.compression(::parquet::Compression::SNAPPY);
  props_builder.disable_dictionary();
  props_builder.data_pagesize(4 * 1024);
  props_builder.write_batch_size(4);
  auto writer_props = props_builder.build();
  ASSERT_AND_ASSIGN(auto writer,
                    ::parquet::arrow::FileWriter::Open(*str_schema, arrow::default_memory_pool(), sink, writer_props));
  ASSERT_STATUS_OK(writer->WriteTable(*table, table->num_rows()));
  ASSERT_STATUS_OK(writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  TrackingMemoryPool tracking_pool(arrow::default_memory_pool());
  ::parquet::ReaderProperties reader_props(&tracking_pool);
  ASSERT_AND_ASSIGN(auto input, fs_->OpenInputFile(temp_file));
  auto file_reader = ::parquet::ParquetFileReader::Open(input, reader_props);
  auto file_metadata = file_reader->metadata();
  ASSERT_EQ(file_metadata->num_row_groups(), 1);

  auto column_metadata = file_metadata->RowGroup(0)->ColumnChunk(0);
  ASSERT_NE(column_metadata->compression(), ::parquet::Compression::UNCOMPRESSED);
  ASSERT_FALSE(column_metadata->has_dictionary_page());

  const int64_t start_offset = column_metadata->data_page_offset();
  const int64_t compressed_size = column_metadata->total_compressed_size();
  ASSERT_GT(start_offset, 0);
  ASSERT_GT(compressed_size, 0);

  tracking_pool.ClearEvents();
  auto stream = reader_props.GetStream(input, start_offset, compressed_size);
  auto page_reader =
      ::parquet::PageReader::Open(stream, column_metadata->num_values(), column_metadata->compression(), reader_props);

  std::vector<int64_t> data_page_sizes;
  int64_t largest_seen_page = 0;
  int growths_for_new_larger_pages = 0;
  while (true) {
    const size_t events_before = tracking_pool.EventCount();
    auto page = page_reader->NextPage();
    if (!page) {
      break;
    }

    if (page->type() != ::parquet::PageType::DATA_PAGE && page->type() != ::parquet::PageType::DATA_PAGE_V2) {
      continue;
    }

    const int64_t page_size = page->size();
    data_page_sizes.push_back(page_size);

    if (page_size > largest_seen_page) {
      const auto page_events = tracking_pool.EventsSince(events_before);
      const bool grew_to_this_page = std::any_of(page_events.begin(), page_events.end(), [&](const ResizeEvent& event) {
        return event.new_size >= page_size && event.new_size > event.old_size;
      });
      EXPECT_TRUE(grew_to_this_page) << "No tracked allocation/reallocation grew to data page size " << page_size;
      ++growths_for_new_larger_pages;
      largest_seen_page = page_size;
    }
  }

  ASSERT_GT(data_page_sizes.size(), 1u);
  ASSERT_GT(growths_for_new_larger_pages, 1);
  ASSERT_GT(largest_seen_page, data_page_sizes.front());
}

TEST_F(ParquetFileWriterTest, NullRecordBatch) {
  // Test writing null record batch
  std::string temp_file = base_path_ + "/data/test_null_batch.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  // Should handle null batch gracefully
  ASSERT_TRUE(writer->Write(nullptr).ok());
  ASSERT_TRUE(writer->Close().ok());
}

TEST_F(ParquetFileWriterTest, VerySmallBufferSize) {
  // Test with very small buffer size
  const int64_t num_rows = 100;

  // Create simple record batch
  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
    ASSERT_TRUE(text_builder.Append("row_" + std::to_string(i)).ok());

    std::vector<uint8_t> vector_data(128, static_cast<uint8_t>(i % 256));
    ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
  }

  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = text_builder.Finish().ValueOrDie();
  auto vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_small_buffer.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created and can be read
  ASSERT_AND_ASSIGN(auto reader, FileRowGroupReader::Make(fs_, temp_file, schema_));
  auto file_metadata = reader->file_metadata();
  ASSERT_GT(file_metadata->GetRowGroupMetadataVector().size(), 0);
}

TEST_F(ParquetFileWriterTest, LargeNumberOfSmallBatches) {
  // Test writing many small batches
  const int64_t batch_size = 10;
  const int num_batches = 100;

  std::string temp_file = base_path_ + "/data/test_many_small_batches.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  for (int batch = 0; batch < num_batches; ++batch) {
    arrow::Int64Builder id_builder;
    arrow::StringBuilder text_builder;
    arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

    for (int64_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(id_builder.Append(batch * batch_size + i).ok());
      ASSERT_TRUE(text_builder.Append("batch_" + std::to_string(batch) + "_row_" + std::to_string(i)).ok());

      std::vector<uint8_t> vector_data(128, static_cast<uint8_t>((batch + i) % 256));
      ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
    }

    auto id_array = id_builder.Finish().ValueOrDie();
    auto text_array = text_builder.Finish().ValueOrDie();
    auto vector_array = vector_builder.Finish().ValueOrDie();

    auto record_batch = arrow::RecordBatch::Make(schema_, batch_size, {id_array, text_array, vector_array});
    ASSERT_TRUE(writer->Write(record_batch).ok());
  }

  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
}

TEST_F(ParquetFileWriterTest, WriteWithNullArrays) {
  // Test writing record batch with null arrays
  const int64_t num_rows = 100;

  // Create null arrays using builders instead of MakeArrayOfNull
  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  // Append nulls for all rows
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.AppendNull().ok());
    ASSERT_TRUE(text_builder.AppendNull().ok());
    // For FixedSizeBinary, we append zero vectors instead of nulls
    std::vector<uint8_t> zero_vector(128, 0);
    ASSERT_TRUE(vector_builder.Append(zero_vector.data()).ok());
  }

  auto null_id_array = id_builder.Finish().ValueOrDie();
  auto null_text_array = text_builder.Finish().ValueOrDie();
  auto null_vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {null_id_array, null_text_array, null_vector_array});

  std::string temp_file = base_path_ + "/data/test_null_arrays.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
}

TEST_F(ParquetFileWriterTest, WriteWithMixedNullAndValidData) {
  // Test writing record batch with mixed null and valid data
  const int64_t num_rows = 100;

  arrow::Int64Builder id_builder;
  arrow::StringBuilder text_builder;
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(128));

  for (int64_t i = 0; i < num_rows; ++i) {
    if (i % 3 == 0) {
      ASSERT_TRUE(id_builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(id_builder.Append(i).ok());
    }

    if (i % 5 == 0) {
      ASSERT_TRUE(text_builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(text_builder.Append("row_" + std::to_string(i)).ok());
    }

    if (i % 7 == 0) {
      // FixedSizeBinaryBuilder doesn't support AppendNull, so we append a zero vector instead
      std::vector<uint8_t> zero_vector(128, 0);
      ASSERT_TRUE(vector_builder.Append(zero_vector.data()).ok());
    } else {
      std::vector<uint8_t> vector_data(128, static_cast<uint8_t>(i % 256));
      ASSERT_TRUE(vector_builder.Append(vector_data.data()).ok());
    }
  }

  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = text_builder.Finish().ValueOrDie();
  auto vector_array = vector_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_mixed_data.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024));

  ASSERT_TRUE(writer->Write(record_batch).ok());
  ASSERT_TRUE(writer->Close().ok());

  // Verify file was created
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidSchema) {
  // Test writing with invalid schema (null schema)
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, id_array, id_array});

  std::string temp_file = base_path_ + "/data/test_invalid_schema.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};

  // Should throw exception for null schema
  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, nullptr, config, column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidColumnGroups) {
  // Test writing with invalid column groups (out of range indices)
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_invalid_column_groups.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> invalid_column_groups = {{100, 200, 300}};  // Out of range

  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, schema_, config, invalid_column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, WriteWithNullFileSystem) {
  // Test writing with null filesystem
  const int64_t num_rows = 10;

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  auto id_array = id_builder.Finish().ValueOrDie();
  auto text_array = arrow::MakeArrayOfNull(arrow::utf8(), num_rows).ValueOrDie();
  auto vector_array = arrow::MakeArrayOfNull(arrow::fixed_size_binary(128), num_rows).ValueOrDie();

  auto record_batch = arrow::RecordBatch::Make(schema_, num_rows, {id_array, text_array, vector_array});

  std::string temp_file = base_path_ + "/data/test_null_filesystem.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should throw exception for null file system
  ASSERT_FALSE(PackedRecordBatchWriter::Make(nullptr, paths, schema_, config, column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, WriteWithInvalidFilePath) {
  // Test writing with invalid file path (empty path)
  StorageConfig config;
  std::vector<std::string> paths = {""};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  // Should fail for empty file path
  ASSERT_FALSE(PackedRecordBatchWriter::Make(fs_, paths, schema_, config, column_groups, 1024 * 1024).ok());
}

TEST_F(ParquetFileWriterTest, TellBeforeAndAfterClose) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_tell.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  // Write data and flush
  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_STATUS_OK(writer->Flush());

  // Tell after flush should be > 0
  ASSERT_AND_ASSIGN(auto tell_before_close, writer->Tell());
  ASSERT_GT(tell_before_close, 0);

  // Close
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  // Tell after close should return cached value >= tell before close
  ASSERT_AND_ASSIGN(auto tell_after_close, writer->Tell());
  ASSERT_GE(tell_after_close, tell_before_close);

  // Verify tell matches actual file size
  ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(temp_file));
  ASSERT_EQ(tell_after_close, static_cast<size_t>(file_info.size()));
}

TEST_F(ParquetFileWriterTest, PackedWriterTell) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file1 = base_path_ + "/data/test_packed_tell_1.parquet";
  std::string temp_file2 = base_path_ + "/data/test_packed_tell_2.parquet";

  StorageConfig config;
  std::vector<std::string> paths = {temp_file1, temp_file2};
  // Split: columns 0,1 in group 0, columns 2,3 in group 1
  std::vector<std::vector<int>> column_groups = {{0, 1}, {2, 3}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, test_schema, config, column_groups, 1024 * 1024));

  // Write data
  ASSERT_STATUS_OK(writer->Write(record_batch));

  // Close
  ASSERT_STATUS_OK(writer->Close());

  // Tell after close
  ASSERT_AND_ASSIGN(auto positions, writer->Tell());
  ASSERT_EQ(positions.size(), 2);
  ASSERT_GT(positions[0], 0);
  ASSERT_GT(positions[1], 0);

  // Verify tell matches actual file sizes
  ASSERT_AND_ASSIGN(auto file_info1, fs_->GetFileInfo(temp_file1));
  ASSERT_EQ(positions[0], static_cast<size_t>(file_info1.size()));

  ASSERT_AND_ASSIGN(auto file_info2, fs_->GetFileInfo(temp_file2));
  ASSERT_EQ(positions[1], static_cast<size_t>(file_info2.size()));
}

TEST_F(ParquetFileWriterTest, FooterSizeMatchesActualFile) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_footer_size.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  auto cached_footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(cached_footer_size, 0u);

  // Read actual footer size from the file:
  // Parquet tail: [Thrift metadata][4B footer_length LE][4B magic "PAR1"]
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(temp_file));
  ASSERT_AND_ASSIGN(auto file_size, file->GetSize());

  // Read last 8 bytes
  ASSERT_AND_ASSIGN(auto tail_buf, file->ReadAt(file_size - 8, 8));
  const uint8_t* tail = tail_buf->data();

  uint32_t footer_length = 0;
  std::memcpy(&footer_length, tail, 4);
  // Verify magic
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(tail + 4), 4), "PAR1");

  uint64_t actual_footer_size = static_cast<uint64_t>(footer_length) + 8;
  EXPECT_EQ(cached_footer_size, actual_footer_size)
      << "cached footer_size=" << cached_footer_size << " actual=" << actual_footer_size;

  // Also verify file_size
  EXPECT_EQ(close_result.Get<uint64_t>(api::kPropertyFileSize), static_cast<uint64_t>(file_size));
}

// Helper: write a small record batch through ParquetFileWriter::Make and
// return the parquet column-chunk compression codecs (one entry per column,
// in schema order) for the first row group.
namespace {
arrow::Result<std::vector<::parquet::Compression::type>> WriteAndReadColumnCompression(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::string& file_path,
    bool use_properties_make,
    const milvus_storage::api::Properties& properties) {
  if (use_properties_make) {
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          milvus_storage::parquet::ParquetFileWriter::Make(fs, schema, file_path, properties));
    ARROW_RETURN_NOT_OK(writer->Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto _close, writer->Close());
    (void)_close;
  } else {
    milvus_storage::StorageConfig config;
    ARROW_ASSIGN_OR_RAISE(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(schema, fs, file_path, config));
    ARROW_RETURN_NOT_OK(writer->Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto _close, writer->Close());
    (void)_close;
  }

  ARROW_ASSIGN_OR_RAISE(auto file, fs->OpenInputFile(file_path));
  auto reader = ::parquet::ParquetFileReader::Open(file);
  auto metadata = reader->metadata();
  std::vector<::parquet::Compression::type> codecs;
  codecs.reserve(schema->num_fields());
  auto rg = metadata->RowGroup(0);
  for (int i = 0; i < rg->num_columns(); ++i) {
    codecs.push_back(rg->ColumnChunk(i)->compression());
  }
  return codecs;
}
}  // namespace

// Dense vector columns (FIXED_SIZE_BINARY) should land UNCOMPRESSED in the
// file. BINARY columns may carry LOB / sparse-vector payloads where
// compression can still help, so they inherit the file-level codec. Other
// columns also follow the file-level codec. Verified through both
// ParquetFileWriter::Make overloads.
TEST_F(ParquetFileWriterTest, FixedSizeBinaryColumnsAreUncompressed) {
  const int64_t num_rows = 16;
  auto schema = arrow::schema({
      arrow::field("id", arrow::int64(), false /*nullable*/, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("dense_vec", arrow::fixed_size_binary(128), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
      arrow::field("blob", arrow::binary(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})),
  });

  arrow::Int64Builder id_builder;
  arrow::FixedSizeBinaryBuilder dense_builder(arrow::fixed_size_binary(128));
  arrow::BinaryBuilder blob_builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
    std::vector<uint8_t> v(128, static_cast<uint8_t>(i));
    ASSERT_TRUE(dense_builder.Append(v.data()).ok());
    std::string s(32, static_cast<char>('a' + (i % 26)));
    ASSERT_TRUE(blob_builder.Append(s).ok());
  }
  ASSERT_AND_ASSIGN(auto id_array, id_builder.Finish());
  ASSERT_AND_ASSIGN(auto dense_array, dense_builder.Finish());
  ASSERT_AND_ASSIGN(auto blob_array, blob_builder.Finish());
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {id_array, dense_array, blob_array});

  // Legacy Make (parquet::WriterProperties default => UNCOMPRESSED at file
  // level): constructor falls back to ZSTD-3 default. FIXED_SIZE_BINARY is
  // forced UNCOMPRESSED per-column; BINARY follows the file default (ZSTD).
  {
    auto file_path = base_path_ + "/data/vector_uncompressed_legacy.parquet";
    ASSERT_AND_ASSIGN(auto codecs, WriteAndReadColumnCompression(fs_, schema, batch, file_path,
                                                                 /*use_properties_make=*/false, properties_));
    ASSERT_EQ(codecs.size(), 3u);
    EXPECT_EQ(codecs[0], ::parquet::Compression::ZSTD) << "id should be ZSTD";
    EXPECT_EQ(codecs[1], ::parquet::Compression::UNCOMPRESSED) << "dense_vec should be UNCOMPRESSED";
    EXPECT_EQ(codecs[2], ::parquet::Compression::ZSTD) << "blob (BINARY) should follow file-level ZSTD";
  }

  // Properties-based Make (registry default zstd / level 3): same expectation.
  {
    auto file_path = base_path_ + "/data/vector_uncompressed_props.parquet";
    ASSERT_AND_ASSIGN(auto codecs, WriteAndReadColumnCompression(fs_, schema, batch, file_path,
                                                                 /*use_properties_make=*/true, properties_));
    ASSERT_EQ(codecs.size(), 3u);
    EXPECT_EQ(codecs[0], ::parquet::Compression::ZSTD) << "id should be ZSTD";
    EXPECT_EQ(codecs[1], ::parquet::Compression::UNCOMPRESSED) << "dense_vec should be UNCOMPRESSED";
    EXPECT_EQ(codecs[2], ::parquet::Compression::ZSTD) << "blob (BINARY) should follow file-level ZSTD";
  }
}

// File-level ZSTD setting does not leak into vector columns — they are
// always emitted UNCOMPRESSED regardless of the caller's WriterProperties.
TEST_F(ParquetFileWriterTest, FileLevelCompressionDoesNotPreventVectorUncompressed) {
  const int64_t num_rows = 8;
  auto schema = arrow::schema({
      arrow::field("dense_vec", arrow::fixed_size_binary(64), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
  });

  arrow::FixedSizeBinaryBuilder b(arrow::fixed_size_binary(64));
  for (int64_t i = 0; i < num_rows; ++i) {
    std::vector<uint8_t> v(64, static_cast<uint8_t>(i));
    ASSERT_TRUE(b.Append(v.data()).ok());
  }
  ASSERT_AND_ASSIGN(auto arr, b.Finish());
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {arr});

  // File-level ZSTD-7 — vector column still emitted UNCOMPRESSED.
  auto props =
      ::parquet::WriterProperties::Builder().compression(::parquet::Compression::ZSTD)->compression_level(7)->build();
  auto file_path = base_path_ + "/data/vector_filelevel_only.parquet";
  milvus_storage::StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer,
                    milvus_storage::parquet::ParquetFileWriter::Make(schema, fs_, file_path, config, props));
  ASSERT_TRUE(writer->Write(batch).ok());
  ASSERT_AND_ASSIGN(auto _close, writer->Close());
  (void)_close;

  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(file_path));
  auto reader = ::parquet::ParquetFileReader::Open(file);
  auto metadata = reader->metadata();
  EXPECT_EQ(metadata->RowGroup(0)->ColumnChunk(0)->compression(), ::parquet::Compression::UNCOMPRESSED);
}

TEST_F(ParquetFileWriterTest, SuppliedFooterSizeIsAuthoritative) {
  ASSERT_AND_ASSIGN(auto test_schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto record_batch, CreateTestData(test_schema));

  std::string temp_file = base_path_ + "/data/test_footer_size_mismatch.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(test_schema, fs_, temp_file, config));

  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());
  auto cached_footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  auto cached_file_size = close_result.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(cached_footer_size, 0u);
  ASSERT_GT(cached_file_size, cached_footer_size);

  // A supplied footer size drives one footer read. A stale value must fail
  // without falling back to another read path.
  auto verify_read = [&](uint64_t footer_size) {
    auto reader =
        milvus_storage::parquet::ParquetFormatReader(fs_, temp_file, properties_, /*needed_columns=*/{},
                                                     /*key_retriever=*/nullptr, cached_file_size, footer_size);
    ASSERT_STATUS_OK(reader.open());

    ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
    ASSERT_GT(row_group_infos.size(), 0u);

    // Read first row group to verify data integrity
    ASSERT_AND_ASSIGN(auto rb, reader.get_chunk(0));
    ASSERT_GT(rb->num_rows(), 0);
  };

  auto stale_reader = milvus_storage::parquet::ParquetFormatReader(fs_, temp_file, properties_, /*needed_columns=*/{},
                                                                   /*key_retriever=*/nullptr, cached_file_size, 1);
  auto stale_status = stale_reader.open();
  ASSERT_STATUS_NOT_OK(stale_status);
  auto stale_detail = ExtendStatusDetail::UnwrapStatus(stale_status);
  ASSERT_NE(stale_detail, nullptr) << stale_status.ToString();
  // The NEUTRAL data-format code, not a Packed* one: this reader also serves
  // iceberg and paimon files, so it is not entitled to claim which subsystem
  // wrote the bytes it failed to parse.
  EXPECT_EQ(stale_detail->code(), ExtendStatusCode::DataCorrupted);

  // Reading the whole file as a suffix remains valid because it contains the
  // complete footer and trailer.
  verify_read(cached_file_size);
}

TEST_F(ParquetFileWriterTest, StructRoundTripThroughParquetFormatReader) {
  constexpr int64_t kNumRows = 2;

  arrow::Int32Builder id_builder;
  ASSERT_STATUS_OK(id_builder.Append(10));
  ASSERT_STATUS_OK(id_builder.Append(20));
  ASSERT_AND_ASSIGN(auto ids, id_builder.Finish());

  arrow::StringBuilder label_builder;
  ASSERT_STATUS_OK(label_builder.Append("left"));
  ASSERT_STATUS_OK(label_builder.Append("right"));
  ASSERT_AND_ASSIGN(auto labels, label_builder.Finish());

  arrow::FieldVector struct_fields = {arrow::field("id", arrow::int32(), false),
                                      arrow::field("label", arrow::utf8(), false)};
  auto struct_type = arrow::struct_(struct_fields);
  auto struct_values = std::make_shared<arrow::StructArray>(struct_type, kNumRows,
                                                            std::vector<std::shared_ptr<arrow::Array>>{ids, labels});
  auto struct_field =
      arrow::field("payload", struct_type, false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));
  auto schema = arrow::schema({struct_field});
  auto record_batch = arrow::RecordBatch::Make(schema, kNumRows, {struct_values});

  std::string temp_file = base_path_ + "/data/test_struct.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(schema, fs_, temp_file, config));
  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  auto file_size = close_result.Get<uint64_t>(api::kPropertyFileSize);
  auto footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  auto reader = milvus_storage::parquet::ParquetFormatReader(fs_, temp_file, properties_, /*needed_columns=*/{},
                                                             /*key_retriever=*/nullptr, file_size, footer_size);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
  ASSERT_EQ(row_group_infos.size(), 1);

  ASSERT_AND_ASSIGN(auto read_batch, reader.get_chunk(0));
  ASSERT_TRUE(read_batch->Equals(*record_batch)) << "expected:\n"
                                                 << record_batch->ToString() << "\nactual:\n"
                                                 << read_batch->ToString() << "\nexpected schema:\n"
                                                 << record_batch->schema()->ToString(true) << "\nactual schema:\n"
                                                 << read_batch->schema()->ToString(true);
}

TEST_F(ParquetFileWriterTest, DISABLED_FixedSizeListRoundTripThroughParquetFormatReader) {
  constexpr int64_t kNumRows = 2;
  constexpr int32_t kListSize = 4;

  arrow::FloatBuilder values_builder;
  for (int64_t row = 0; row < kNumRows; ++row) {
    for (int32_t value = 0; value < kListSize; ++value) {
      ASSERT_STATUS_OK(values_builder.Append(static_cast<float>(row * kListSize + value)));
    }
  }
  ASSERT_AND_ASSIGN(auto values, values_builder.Finish());

  auto list_type = arrow::fixed_size_list(arrow::field("item", arrow::float32(), false), kListSize);
  ASSERT_AND_ASSIGN(auto vectors, arrow::FixedSizeListArray::FromArrays(values, list_type));
  auto list_field = arrow::field("embedding", list_type, false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"0"}));
  auto schema = arrow::schema({list_field});
  auto record_batch = arrow::RecordBatch::Make(schema, kNumRows, {vectors});

  std::string temp_file = base_path_ + "/data/test_fixed_size_list.parquet";

  StorageConfig config;
  ASSERT_AND_ASSIGN(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(schema, fs_, temp_file, config));
  ASSERT_STATUS_OK(writer->Write(record_batch));
  ASSERT_AND_ASSIGN(auto close_result, writer->Close());

  auto file_size = close_result.Get<uint64_t>(api::kPropertyFileSize);
  auto footer_size = close_result.Get<uint64_t>(api::kPropertyFooterSize);
  auto reader = milvus_storage::parquet::ParquetFormatReader(fs_, temp_file, properties_, /*needed_columns=*/{},
                                                             /*key_retriever=*/nullptr, file_size, footer_size);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
  ASSERT_EQ(row_group_infos.size(), 1);

  ASSERT_AND_ASSIGN(auto read_batch, reader.get_chunk(0));
  ASSERT_TRUE(read_batch->Equals(*record_batch)) << "expected:\n"
                                                 << record_batch->ToString() << "\nactual:\n"
                                                 << read_batch->ToString() << "\nexpected schema:\n"
                                                 << record_batch->schema()->ToString(true) << "\nactual schema:\n"
                                                 << read_batch->schema()->ToString(true);
}

}  // namespace milvus_storage::test
