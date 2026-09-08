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
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cstdint>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/io_util.h>
#include <arrow/table.h>
#include <arrow/array/concatenate.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader_cache.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/reader.h"
#include "test_env.h"

namespace milvus_storage {

using namespace lance;

namespace {

class FailingLanceRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  explicit FailingLanceRecordBatchReader(arrow::Status status) : status_(std::move(status)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return arrow::schema({}); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    *batch = nullptr;
    return status_;
  }

  arrow::Status Close() override { return status_; }

  private:
  arrow::Status status_;
};

class InjectedFailureRandomAccessFile final : public arrow::io::RandomAccessFile {
  public:
  InjectedFailureRandomAccessFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                  std::shared_ptr<std::atomic<int>> remaining_failures,
                                  arrow::Status failure_status,
                                  int64_t minimum_failure_size)
      : file_(std::move(file)),
        remaining_failures_(std::move(remaining_failures)),
        failure_status_(std::move(failure_status)),
        minimum_failure_size_(minimum_failure_size) {}

  arrow::Status Close() override { return file_->Close(); }
  arrow::Status Abort() override { return file_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override { return file_->Read(nbytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override { return file_->Read(nbytes); }
  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }
  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return file_->ReadMetadata();
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return file_->ReadMetadataAsync(io_context);
  }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    if (ShouldFail(nbytes)) {
      return failure_status_;
    }
    return file_->ReadAt(position, nbytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    if (ShouldFail(nbytes)) {
      return failure_status_;
    }
    return file_->ReadAt(position, nbytes);
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    return file_->ReadAsync(io_context, position, nbytes);
  }
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }

  private:
  bool ShouldFail(int64_t nbytes) {
    if (nbytes < minimum_failure_size_) {
      return false;
    }
    auto remaining = remaining_failures_->load(std::memory_order_relaxed);
    while (remaining > 0) {
      if (remaining_failures_->compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<std::atomic<int>> remaining_failures_;
  arrow::Status failure_status_;
  int64_t minimum_failure_size_;
};

class InjectedFailureFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  using arrow::fs::SubTreeFileSystem::OpenInputFile;

  InjectedFailureFileSystem(std::shared_ptr<arrow::fs::FileSystem> filesystem,
                            std::shared_ptr<std::atomic<int>> remaining_failures,
                            arrow::Status failure_status,
                            int64_t minimum_failure_size = 0)
      : arrow::fs::SubTreeFileSystem("", std::move(filesystem)),
        remaining_failures_(std::move(remaining_failures)),
        failure_status_(std::move(failure_status)),
        minimum_failure_size_(minimum_failure_size) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::SubTreeFileSystem::OpenInputFile(path));
    return std::make_shared<InjectedFailureRandomAccessFile>(std::move(file), remaining_failures_, failure_status_,
                                                             minimum_failure_size_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::SubTreeFileSystem::OpenInputFile(info));
    return std::make_shared<InjectedFailureRandomAccessFile>(std::move(file), remaining_failures_, failure_status_,
                                                             minimum_failure_size_);
  }

  private:
  std::shared_ptr<std::atomic<int>> remaining_failures_;
  arrow::Status failure_status_;
  int64_t minimum_failure_size_;
};

#ifdef WITH_CRT

thread_local bool async_completion_active = false;

class AsyncReaderLifecycleState {
  public:
  void ReaderOpened() {
    std::lock_guard lock(mutex_);
    ++opened_readers_;
    cv_.notify_all();
  }

  void ReadSubmitted() {
    std::lock_guard lock(mutex_);
    ++submitted_reads_;
    cv_.notify_all();
  }

  void ReadCompleted() {
    std::lock_guard lock(mutex_);
    ++completed_reads_;
    cv_.notify_all();
  }

  void ReaderClosed(bool in_completion) {
    std::lock_guard lock(mutex_);
    ++closed_readers_;
    close_in_completion_ |= in_completion;
    cv_.notify_all();
  }

  void ReaderDestroyed(bool in_completion) {
    std::lock_guard lock(mutex_);
    ++destroyed_readers_;
    destroy_in_completion_ |= in_completion;
    cv_.notify_all();
  }

  void AllowCompletions() {
    std::lock_guard lock(mutex_);
    allow_completions_ = true;
    cv_.notify_all();
  }

  void WaitForCompletionPermission() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return allow_completions_; });
  }

  bool WaitForSubmitted(std::chrono::seconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return submitted_reads_ != 0; });
  }

  bool WaitForAllCompletedAndDestroyed(std::chrono::seconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] {
      return submitted_reads_ != 0 && completed_reads_ == submitted_reads_ && opened_readers_ != 0 &&
             closed_readers_ == opened_readers_ && destroyed_readers_ == opened_readers_;
    });
  }

  bool close_in_completion() const {
    std::lock_guard lock(mutex_);
    return close_in_completion_;
  }

  bool destroy_in_completion() const {
    std::lock_guard lock(mutex_);
    return destroy_in_completion_;
  }

  void AddWorker(std::thread worker) {
    std::lock_guard lock(mutex_);
    workers_.emplace_back(std::move(worker));
  }

  void JoinWorkers() {
    std::vector<std::thread> workers;
    {
      std::lock_guard lock(mutex_);
      workers.swap(workers_);
    }
    for (auto& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
  size_t opened_readers_ = 0;
  size_t submitted_reads_ = 0;
  size_t completed_reads_ = 0;
  size_t closed_readers_ = 0;
  size_t destroyed_readers_ = 0;
  bool allow_completions_ = false;
  bool close_in_completion_ = false;
  bool destroy_in_completion_ = false;
};

class CallbackThreadRandomAccessFile final : public arrow::io::RandomAccessFile,
                                             public milvus_storage::NonBlockingRandomAccessFile {
  public:
  CallbackThreadRandomAccessFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                 std::shared_ptr<AsyncReaderLifecycleState> state)
      : file_(std::move(file)), state_(std::move(state)) {
    state_->ReaderOpened();
  }

  ~CallbackThreadRandomAccessFile() override { state_->ReaderDestroyed(async_completion_active); }

  arrow::Status Close() override {
    auto status = file_->Close();
    state_->ReaderClosed(async_completion_active);
    return status;
  }
  arrow::Status Abort() override { return file_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override { return file_->Read(nbytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override { return file_->Read(nbytes); }
  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }
  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return file_->ReadMetadata();
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return file_->ReadMetadataAsync(io_context);
  }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Future<int64_t> GetSizeAsync() override { return arrow::Future<int64_t>::MakeFinished(file_->GetSize()); }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return file_->ReadAt(position, nbytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    return file_->ReadAt(position, nbytes);
  }
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    auto future = arrow::Future<int64_t>::Make();
    auto file = file_;
    auto state = state_;
    auto worker =
        std::thread([file = std::move(file), state = std::move(state), future, position, nbytes, out]() mutable {
          state->ReadSubmitted();
          state->WaitForCompletionPermission();
          auto result = file->ReadAt(position, nbytes, out);
          async_completion_active = true;
          future.MarkFinished(std::move(result));
          async_completion_active = false;
          state->ReadCompleted();
        });
    state_->AddWorker(std::move(worker));
    return future;
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    return file_->ReadAsync(io_context, position, nbytes);
  }
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<AsyncReaderLifecycleState> state_;
};

class CallbackThreadFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  using arrow::fs::SubTreeFileSystem::OpenInputFile;

  CallbackThreadFileSystem(std::shared_ptr<arrow::fs::FileSystem> filesystem,
                           std::shared_ptr<AsyncReaderLifecycleState> state)
      : arrow::fs::SubTreeFileSystem("", std::move(filesystem)), state_(std::move(state)) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::SubTreeFileSystem::OpenInputFile(path));
    return std::make_shared<CallbackThreadRandomAccessFile>(std::move(file), state_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::SubTreeFileSystem::OpenInputFile(info));
    return std::make_shared<CallbackThreadRandomAccessFile>(std::move(file), state_);
  }

  private:
  std::shared_ptr<AsyncReaderLifecycleState> state_;
};

#endif  // WITH_CRT

}  // namespace

TEST(LanceBridgeErrorTest, OpenReturnsInvalidStatusForNullFilesystem) {
  auto result = BlockingDataset::Open("file:///does-not-matter", nullptr, {});

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid());
}

TEST(LanceBridgeErrorTest, TranslatesDeferredRecordBatchReaderReadNextError) {
  auto inner = std::make_shared<FailingLanceRecordBatchReader>(
      arrow::Status::IOError("__LOON_FFI_ERRCODE__=108; Injected fault"));
  auto reader = internal::WrapLanceRecordBatchReader(std::move(inner));

  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = reader->ReadNext(&batch);

  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(status.message().find("__LOON_FFI_ERRCODE__="), std::string::npos);
  EXPECT_NE(status.message().find("Injected fault"), std::string::npos);
}

TEST(LanceBridgeErrorTest, TranslatesDeferredRecordBatchReaderNotSupportedError) {
  auto inner = std::make_shared<FailingLanceRecordBatchReader>(
      arrow::Status::Invalid("__LOON_FFI_ERRCODE__=9; unsupported operation"));
  auto reader = internal::WrapLanceRecordBatchReader(std::move(inner));

  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = reader->ReadNext(&batch);

  EXPECT_TRUE(status.IsNotImplemented()) << status.ToString();
  EXPECT_EQ(status.message().find("__LOON_FFI_ERRCODE__="), std::string::npos);
  EXPECT_NE(status.message().find("unsupported operation"), std::string::npos);
}

TEST(LanceBridgeErrorTest, TranslatesDeferredRecordBatchReaderCloseError) {
  auto inner = std::make_shared<FailingLanceRecordBatchReader>(
      arrow::Status::IOError("__LOON_FFI_ERRCODE__=108; Injected fault"));
  auto reader = internal::WrapLanceRecordBatchReader(std::move(inner));

  auto status = reader->Close();

  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(status.message().find("__LOON_FFI_ERRCODE__="), std::string::npos);
  EXPECT_NE(status.message().find("Injected fault"), std::string::npos);
}

TEST(LanceWriterOwnershipTest, RejectedWriterOptionsReleaseBufferedBatches) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema, 0, false, 1));
  std::weak_ptr<arrow::RecordBatch> batch_weak = batch;

  {
    api::Properties properties = {
        {PROPERTY_FS_STORAGE_TYPE, std::string("remote")},
        {PROPERTY_FS_CLOUD_PROVIDER, std::string(kCloudProviderAWS)},
        {PROPERTY_FS_ROLE_ARN, std::string("arn:aws:iam::123456789012:role/test-role")},
    };
    LanceTableWriter writer("unused", schema, properties);
    ASSERT_STATUS_OK(writer.Write(batch));
    batch.reset();

    auto result = writer.Close();
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
  }

  EXPECT_TRUE(batch_weak.expired());
}

class LanceBasicTest : public ::testing::Test {
  protected:
  struct LanceDataFileVersion {
    uint16_t major;
    uint16_t minor;
  };

  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // For Arrow filesystem operations
    arrow_base_path_ = GetTestBasePath("lance-fragment-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, arrow_base_path_));

    // For Lance, use relative path - BuildLanceBaseUri will be called internally
    base_path_ = arrow_base_path_;

    // Create a simple test schema with field IDs required by packed writer
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());

    // Create test data
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override {
    if (!IsCloudEnv()) {
      ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
    }
  }

  arrow::Result<LanceDataFileVersion> ReadLanceDataFileVersion() const {
    auto* subtree = dynamic_cast<arrow::fs::SubTreeFileSystem*>(fs_.get());
    if (subtree == nullptr) {
      return arrow::Status::Invalid("Expected a subtree filesystem for local Lance test data");
    }

    const boost::filesystem::path root_path(subtree->base_path());
    const boost::filesystem::path dataset_path = root_path / arrow_base_path_;
    for (boost::filesystem::recursive_directory_iterator it(dataset_path), end; it != end; ++it) {
      if (!boost::filesystem::is_regular_file(it->path()) || it->path().extension() != ".lance") {
        continue;
      }

      const auto file_path = boost::filesystem::relative(it->path(), root_path).generic_string();
      ARROW_ASSIGN_OR_RAISE(auto input, fs_->OpenInputFile(file_path));
      ARROW_ASSIGN_OR_RAISE(auto size, input->GetSize());
      if (size < 8) {
        return arrow::Status::Invalid("Lance data file is too small to contain a version footer: ", file_path);
      }

      ARROW_ASSIGN_OR_RAISE(auto footer, input->ReadAt(size - 8, 8));
      const auto* bytes = footer->data();
      if (footer->size() != 8 || bytes[4] != 'L' || bytes[5] != 'A' || bytes[6] != 'N' || bytes[7] != 'C') {
        return arrow::Status::Invalid("Invalid Lance data file footer: ", file_path);
      }

      return LanceDataFileVersion{
          .major = static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8)),
          .minor = static_cast<uint16_t>(bytes[2] | (static_cast<uint16_t>(bytes[3]) << 8)),
      };
    }

    return arrow::Status::Invalid("No Lance data file found under ", arrow_base_path_);
  }

  struct CloudTakeIops {
    uint64_t total_iops = 0;
    uint64_t peak_one_second_iops = 0;
  };

  void RunCloudWideTableDuplicatedFragmentTake(int64_t column_count,
                                               int64_t row_count,
                                               size_t column_group_file_count,
                                               size_t reader_count,
                                               CloudTakeIops& result);

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;        // Lance URI (s3://bucket/path or /tmp/path)
  std::string arrow_base_path_;  // Path for Arrow filesystem operations
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

class LanceRleStorageVersionTest : public LanceBasicTest,
                                   public ::testing::WithParamInterface<LanceDataStorageFormat> {};

TEST_F(LanceBasicTest, AimdRetriesFfiThrottlingDuringRangeRead) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "This regression test uses an injected local filesystem.";
  }

  api::SetValue(properties_, PROPERTY_FS_LANCE_IO_PARALLELISM, "1");
  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto column_group_file, writer.Close());
  ASSERT_EQ(column_group_file.end_index, test_batch_->num_rows());

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));

  auto remaining_failures = std::make_shared<std::atomic<int>>(0);
  auto throttling_fs = std::make_shared<InjectedFailureFileSystem>(
      fs_, remaining_failures,
      MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "Injected throttling", "Injected throttling"));
  ASSERT_AND_ASSIGN(auto dataset, BlockingDataset::Open(lance_uri, throttling_fs, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto fragment_ids, dataset->GetAllFragmentIds());
  ASSERT_EQ(fragment_ids.size(), 1);

  LanceTableReader reader(dataset, throttling_fs, fragment_ids.front(), schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto row_groups, reader.get_row_group_infos());
  ASSERT_FALSE(row_groups.empty());

  remaining_failures->store(4, std::memory_order_relaxed);
  auto chunk = reader.get_chunk(0);

  ASSERT_TRUE(chunk.ok()) << chunk.status().ToString();
  EXPECT_EQ(chunk.ValueOrDie()->num_rows(), test_batch_->num_rows());
  EXPECT_EQ(remaining_failures->load(std::memory_order_relaxed), 0);
}

TEST_F(LanceBasicTest, MaterializedReadsTranslateDeferredErrors) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "This regression test uses an injected local filesystem.";
  }

  constexpr int64_t kRows = 32'768;
  api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "32768");
  ASSERT_AND_ASSIGN(auto large_batch, CreateTestData(schema_, 0, false, kRows));
  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(large_batch));
  ASSERT_AND_ASSIGN(auto column_group_file, writer.Close());
  ASSERT_EQ(column_group_file.end_index, large_batch->num_rows());

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));

  auto remaining_failures = std::make_shared<std::atomic<int>>(0);
  auto failing_fs = std::make_shared<InjectedFailureFileSystem>(
      fs_, remaining_failures,
      MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "Injected read failure", "Injected read failure"),
      4 * 1024);
  ASSERT_AND_ASSIGN(auto dataset, BlockingDataset::Open(lance_uri, failing_fs, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto fragment_ids, dataset->GetAllFragmentIds());
  ASSERT_EQ(fragment_ids.size(), 1);

  auto open_reader = [&]() -> arrow::Result<std::shared_ptr<LanceTableReader>> {
    remaining_failures->store(0, std::memory_order_relaxed);
    auto reader = std::make_shared<LanceTableReader>(dataset, failing_fs, fragment_ids.front(), schema_, properties_);
    ARROW_RETURN_NOT_OK(reader->open());
    return reader;
  };
  auto expect_timeout = [](const arrow::Status& status) {
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
    EXPECT_EQ(status.message().find("__LOON_FFI_ERRCODE__="), std::string::npos);
  };
  std::vector<int64_t> take_indices(kRows);
  std::iota(take_indices.begin(), take_indices.end(), 0);

  {
    SCOPED_TRACE("get_chunk");
    ASSERT_AND_ASSIGN(auto reader, open_reader());
    remaining_failures->store(1'000, std::memory_order_relaxed);
    expect_timeout(reader->get_chunk(0).status());
  }
  {
    SCOPED_TRACE("get_chunks");
    ASSERT_AND_ASSIGN(auto reader, open_reader());
    remaining_failures->store(1'000, std::memory_order_relaxed);
    expect_timeout(reader->get_chunks({0}).status());
  }
  {
    SCOPED_TRACE("take");
    ASSERT_AND_ASSIGN(auto reader, open_reader());
    remaining_failures->store(1'000, std::memory_order_relaxed);
    expect_timeout(reader->take(take_indices).status());
  }

  dataset.reset();
  failing_fs.reset();

  auto permission_failures = std::make_shared<std::atomic<int>>(0);
  auto permission_status =
      arrow::Status::IOError("Injected permission failure").WithDetail(arrow::internal::StatusDetailFromErrno(EACCES));
  auto permission_fs =
      std::make_shared<InjectedFailureFileSystem>(fs_, permission_failures, std::move(permission_status), 4 * 1024);
  ASSERT_AND_ASSIGN(auto permission_dataset,
                    BlockingDataset::Open(lance_uri, permission_fs, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto permission_fragment_ids, permission_dataset->GetAllFragmentIds());
  ASSERT_EQ(permission_fragment_ids.size(), 1);
  LanceTableReader permission_reader(permission_dataset, permission_fs, permission_fragment_ids.front(), schema_,
                                     properties_);
  ASSERT_STATUS_OK(permission_reader.open());

  permission_failures->store(1'000, std::memory_order_relaxed);
  auto permission_result = permission_reader.get_chunk(0);
  ASSERT_FALSE(permission_result.ok());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(permission_result.status()), EACCES);
  EXPECT_EQ(permission_result.status().message().find("__LOON_FFI_ERRCODE__="), std::string::npos);
  EXPECT_EQ(permission_result.status().message().find("Generic filesystem_c error"), std::string::npos)
      << permission_result.status().ToString();
}

#ifdef WITH_CRT

TEST_F(LanceBasicTest, AsyncReaderIsReleasedOutsideCrtCompletionCallback) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "This regression uses an injected local filesystem.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto column_group_file, writer.Close());
  ASSERT_EQ(column_group_file.end_index, test_batch_->num_rows());

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));

  auto lifecycle = std::make_shared<AsyncReaderLifecycleState>();
  auto callback_filesystem = std::make_shared<CallbackThreadFileSystem>(fs_, lifecycle);
  auto open_future = std::async(std::launch::async, [&]() {
    return BlockingDataset::Open(lance_uri, callback_filesystem, ToReaderOptions(fs_config));
  });

  if (!lifecycle->WaitForSubmitted(std::chrono::seconds(5))) {
    lifecycle->AllowCompletions();
    open_future.wait();
    lifecycle->JoinWorkers();
    FAIL() << "The asynchronous reader was not submitted";
  }
  lifecycle->AllowCompletions();
  auto dataset_result = open_future.get();
  lifecycle->JoinWorkers();
  ASSERT_TRUE(dataset_result.ok()) << dataset_result.status().ToString();
  ASSERT_TRUE(lifecycle->WaitForAllCompletedAndDestroyed(std::chrono::seconds(5)))
      << "The asynchronous readers were not all completed and destroyed";
  EXPECT_FALSE(lifecycle->close_in_completion())
      << "The CRT completion thread closed ReaderHandle inside the CRT callback; CRT resources must be released "
         "outside the callback to avoid destruction deadlock";
  EXPECT_FALSE(lifecycle->destroy_in_completion())
      << "The CRT completion thread destroyed RandomAccessFile inside the CRT callback; CRT resources must be "
         "released outside the callback to avoid destruction deadlock";
}

#endif  // WITH_CRT

TEST_P(LanceRleStorageVersionTest, WritesAndReadsCustomerShapedRleData) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 100'000;
  constexpr int64_t kRunLength = 4096;
  constexpr int64_t kSampleSize = 2 * kRunLength + 1;
  constexpr int32_t kEmbeddingDimension = 4;

  auto rle_metadata = arrow::key_value_metadata({"lance-encoding:rle-threshold", "lance-encoding:bss"}, {"1.0", "off"});
  auto curr_time_type = arrow::timestamp(arrow::TimeUnit::MILLI);
  auto rle_schema = arrow::schema({
      arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), kEmbeddingDimension), false),
      arrow::field("uuid", arrow::utf8(), false, rle_metadata),
      arrow::field("curr_time", curr_time_type, false, rle_metadata),
  });

  std::vector<float> embedding_values(kRows * kEmbeddingDimension);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int32_t dim = 0; dim < kEmbeddingDimension; ++dim) {
      embedding_values[row * kEmbeddingDimension + dim] = static_cast<float>(dim) + 0.25F;
    }
  }

  auto embedding_value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder embedding_builder(arrow::default_memory_pool(), embedding_value_builder,
                                                kEmbeddingDimension);
  ASSERT_STATUS_OK(embedding_builder.AppendValues(kRows));
  ASSERT_STATUS_OK(embedding_value_builder->AppendValues(embedding_values));

  constexpr std::string_view kUuidA = "00000000-0000-0000-0000-000000000001";
  constexpr std::string_view kUuidB = "00000000-0000-0000-0000-000000000002";
  arrow::StringBuilder uuid_builder;
  ASSERT_STATUS_OK(uuid_builder.Reserve(kRows));
  ASSERT_STATUS_OK(uuid_builder.ReserveData(kRows * kUuidA.size()));
  for (int64_t row = 0; row < kRows; ++row) {
    uuid_builder.UnsafeAppend((row / kRunLength) % 2 == 0 ? kUuidA : kUuidB);
  }

  constexpr int64_t kBaseTimestamp = 1'784'065'218'692;
  std::vector<int64_t> curr_time_values(kRows);
  for (int64_t row = 0; row < kRows; ++row) {
    curr_time_values[row] = kBaseTimestamp + row / kRunLength;
  }
  arrow::TimestampBuilder curr_time_builder(curr_time_type, arrow::default_memory_pool());
  ASSERT_STATUS_OK(curr_time_builder.AppendValues(curr_time_values));

  std::shared_ptr<arrow::Array> embedding_array;
  std::shared_ptr<arrow::Array> uuid_array;
  std::shared_ptr<arrow::Array> curr_time_array;
  ASSERT_STATUS_OK(embedding_builder.Finish(&embedding_array));
  ASSERT_STATUS_OK(uuid_builder.Finish(&uuid_array));
  ASSERT_STATUS_OK(curr_time_builder.Finish(&curr_time_array));
  auto batch = arrow::RecordBatch::Make(rle_schema, kRows, {embedding_array, uuid_array, curr_time_array});

  LanceTableWriter writer(base_path_, rle_schema, properties_, GetParam());
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);

  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));
  ASSERT_AND_ASSIGN(auto storage_version, ReadLanceDataFileVersion());
  ASSERT_EQ(storage_version.major, 2);
  ASSERT_EQ(storage_version.minor, static_cast<uint32_t>(GetParam()));

  std::vector<int64_t> row_indices(kSampleSize);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  const std::vector<std::vector<std::string>> projections = {
      {"embedding"},
      {"uuid", "curr_time"},
      {"embedding", "uuid", "curr_time"},
  };
  for (const auto& projection : projections) {
    LanceTableReader reader(fs_, parsed_uri.first, parsed_uri.second, nullptr, properties_, projection);
    ASSERT_STATUS_OK(reader.open());
    ASSERT_AND_ASSIGN(auto table, reader.take(row_indices));
    ASSERT_STATUS_OK(table->ValidateFull());
    ASSERT_EQ(table->num_rows(), kSampleSize);
    ASSERT_EQ(table->num_columns(), projection.size());
    for (size_t column = 0; column < projection.size(); ++column) {
      ASSERT_EQ(table->field(column)->name(), projection[column]);
    }

    ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());
    if (auto column = result_batch->GetColumnByName("embedding")) {
      auto embedding = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(column);
      ASSERT_NE(embedding, nullptr);
      auto values = std::dynamic_pointer_cast<arrow::FloatArray>(embedding->values());
      ASSERT_NE(values, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        for (int32_t dim = 0; dim < kEmbeddingDimension; ++dim) {
          ASSERT_FLOAT_EQ(values->Value(embedding->value_offset(row) + dim), static_cast<float>(dim) + 0.25F);
        }
      }
    }

    if (auto column = result_batch->GetColumnByName("uuid")) {
      auto uuid = std::dynamic_pointer_cast<arrow::StringArray>(column);
      ASSERT_NE(uuid, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        const auto expected = (row_indices[row] / kRunLength) % 2 == 0 ? kUuidA : kUuidB;
        ASSERT_EQ(uuid->GetString(row), std::string(expected));
      }
    }

    if (auto column = result_batch->GetColumnByName("curr_time")) {
      auto curr_time = std::dynamic_pointer_cast<arrow::TimestampArray>(column);
      ASSERT_NE(curr_time, nullptr);
      for (int64_t row = 0; row < kSampleSize; ++row) {
        ASSERT_EQ(curr_time->Value(row), kBaseTimestamp + row_indices[row] / kRunLength);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DataStorageVersions,
                         LanceRleStorageVersionTest,
                         ::testing::Values(LanceDataStorageFormat::V2_1,
                                           LanceDataStorageFormat::V2_2,
                                           LanceDataStorageFormat::V2_3),
                         [](const ::testing::TestParamInfo<LanceDataStorageFormat>& info) {
                           switch (info.param) {
                             case LanceDataStorageFormat::V2_1:
                               return "V2_1";
                             case LanceDataStorageFormat::V2_2:
                               return "V2_2";
                             case LanceDataStorageFormat::V2_3:
                               return "V2_3";
                             default:
                               return "Unknown";
                           }
                         });

TEST_F(LanceBasicTest, DefaultStorageVersionIsV2_1) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto storage_version, ReadLanceDataFileVersion());
  ASSERT_EQ(storage_version.major, 2);
  ASSERT_EQ(storage_version.minor, 1);
}

TEST_F(LanceBasicTest, WriteDatasetReturnsOnlyFragmentsCreatedByEachWrite) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  ASSERT_AND_ASSIGN(auto storage_options, ToWriterOptions(fs_config));

  auto write_once = [&]() -> arrow::Result<std::vector<uint64_t>> {
    ARROW_ASSIGN_OR_RAISE(
        auto batch_reader,
        arrow::RecordBatchReader::Make(std::vector<std::shared_ptr<arrow::RecordBatch>>{test_batch_}, schema_));
    ArrowArrayStream stream{};
    ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));
    return BlockingDataset::WriteDataset(lance_uri, &stream, storage_options);
  };

  ASSERT_AND_ASSIGN(auto first_write_fragment_ids, write_once());
  ASSERT_EQ(first_write_fragment_ids.size(), 1);

  ASSERT_AND_ASSIGN(auto second_write_fragment_ids, write_once());
  ASSERT_EQ(second_write_fragment_ids.size(), 1);
  EXPECT_NE(first_write_fragment_ids[0], second_write_fragment_ids[0]);

  ASSERT_AND_ASSIGN(auto read_fs, FilesystemCache::getInstance().get(properties_, lance_uri));
  ASSERT_AND_ASSIGN(auto dataset, BlockingDataset::Open(lance_uri, read_fs, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto all_fragment_ids, dataset->GetAllFragmentIds());
  std::sort(all_fragment_ids.begin(), all_fragment_ids.end());
  std::vector<uint64_t> expected_fragment_ids = {first_write_fragment_ids[0], second_write_fragment_ids[0]};
  std::sort(expected_fragment_ids.begin(), expected_fragment_ids.end());
  EXPECT_EQ(all_fragment_ids, expected_fragment_ids);
}

TEST_F(LanceBasicTest, InvalidUriCacheKeyDoesNotCollideWithBaseUri) {
  api::ColumnGroupFile invalid_file{.path = base_path_, .start_index = 0, .end_index = 1};
  auto valid_file = invalid_file;
  valid_file.path = MakeLanceUri(base_path_, 0);

  EXPECT_NE(LanceTableReader::MetaTrait::cache_key(invalid_file), LanceTableReader::MetaTrait::cache_key(valid_file));
}

TEST_F(LanceBasicTest, DifferentFragmentsShareDatasetMetadata) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused metadata-cache reproduction uses the local Lance fixture.";
  }

  LanceTableWriter first_writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(first_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto first_file, first_writer.Close());

  LanceTableWriter second_writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(second_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto second_file, second_writer.Close());
  ASSERT_NE(first_file.path, second_file.path);

  const std::vector<api::ColumnGroupFile> files = {std::move(first_file), std::move(second_file)};
  auto cache = FormatReaderMetadataCache<LanceTableReader>::Make();
  size_t metadata_load_count = 0;
  std::vector<LanceTableReader::MetaTrait::MetadataPtr> loaded_metadata;
  std::vector<std::shared_ptr<LanceTableReader>> readers;
  loaded_metadata.reserve(files.size());
  readers.reserve(files.size());

  ASSERT_EQ(LanceTableReader::MetaTrait::cache_key(files[0]), LanceTableReader::MetaTrait::cache_key(files[1]));

  for (size_t i = 0; i < files.size(); ++i) {
    const auto key = LanceTableReader::MetaTrait::cache_key(files[i]);
    std::cout << "[ LANCE_METADATA_CACHE_KEY ] key=" << key << std::endl;
    ASSERT_AND_ASSIGN(
        auto metadata, cache->get_or_open(key, [&]() -> arrow::Result<LanceTableReader::MetaTrait::MetadataPtr> {
          ++metadata_load_count;
          std::cout << "[ LANCE_METADATA_CACHE_MISS ] key=" << key << std::endl;
          return LanceTableReader::MetaTrait::load_metadata(files[i], properties_, nullptr /* key_retriever */);
        }));
    ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(files[i].path));
    std::cout << "[ LANCE_DATASET ] fragment_id=" << parsed_uri.second
              << ", dataset=" << metadata->payload.dataset.get() << std::endl;
    ASSERT_AND_ASSIGN(auto reader, LanceTableReader::MetaTrait::create_from_metadata(metadata, files[i], schema_,
                                                                                     std::vector<std::string>{}, ""));
    ASSERT_AND_ASSIGN(auto row_group_infos, reader->get_row_group_infos());
    ASSERT_FALSE(row_group_infos.empty());
    loaded_metadata.emplace_back(metadata);
    readers.emplace_back(std::move(reader));
  }

  ASSERT_EQ(loaded_metadata.size(), 2);
  EXPECT_EQ(metadata_load_count, 1);
  EXPECT_EQ(loaded_metadata[0].get(), loaded_metadata[1].get());
  EXPECT_EQ(loaded_metadata[0]->payload.dataset.get(), loaded_metadata[1]->payload.dataset.get());
  EXPECT_TRUE(loaded_metadata[0]->row_group_infos.empty());
  EXPECT_NE(readers[0].get(), readers[1].get());

  std::weak_ptr<BlockingDataset> weak_dataset = loaded_metadata[0]->payload.dataset;
  readers.clear();
  loaded_metadata.clear();
  cache.reset();
  EXPECT_TRUE(weak_dataset.expired());
}

TEST_F(LanceBasicTest, DifferentBaseUrisDoNotShareDataset) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused dataset-cache test uses the local Lance fixture.";
  }

  LanceTableWriter first_writer(base_path_ + "/first", schema_, properties_);
  ASSERT_STATUS_OK(first_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto first_file, first_writer.Close());

  LanceTableWriter second_writer(base_path_ + "/second", schema_, properties_);
  ASSERT_STATUS_OK(second_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto second_file, second_writer.Close());

  ASSERT_AND_ASSIGN(auto first_uri, ParseLanceUri(first_file.path));
  ASSERT_AND_ASSIGN(auto second_uri, ParseLanceUri(second_file.path));
  ASSERT_NE(first_uri.first, second_uri.first);

  ASSERT_AND_ASSIGN(auto first_metadata,
                    LanceTableReader::MetaTrait::load_metadata(first_file, properties_, nullptr /* key_retriever */));
  ASSERT_AND_ASSIGN(auto second_metadata,
                    LanceTableReader::MetaTrait::load_metadata(second_file, properties_, nullptr /* key_retriever */));
  EXPECT_NE(first_metadata->payload.dataset.get(), second_metadata->payload.dataset.get());
}

TEST_F(LanceBasicTest, SeparateTopLevelReaderCachesDoNotShareDataset) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused dataset-cache test uses the local Lance fixture.";
  }

  LanceTableWriter first_writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(first_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto first_file, first_writer.Close());
  const auto key = LanceTableReader::MetaTrait::cache_key(first_file);

  // ReaderImpl owns one MetadataCache. Independent handles here model two
  // top-level readers with independent Dataset metadata.
  MetadataCache first_reader_cache;
  ASSERT_AND_ASSIGN(auto first_metadata, first_reader_cache.get<LanceTableReader>()->get_or_open(key, [&]() {
    return LanceTableReader::MetaTrait::load_metadata(first_file, properties_, nullptr /* key_retriever */);
  }));

  LanceTableWriter second_writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(second_writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto second_file, second_writer.Close());
  ASSERT_EQ(key, LanceTableReader::MetaTrait::cache_key(second_file));

  ASSERT_AND_ASSIGN(auto first_metadata_again, first_reader_cache.get<LanceTableReader>()->get_or_open(key, [&]() {
    return LanceTableReader::MetaTrait::load_metadata(second_file, properties_, nullptr /* key_retriever */);
  }));
  ASSERT_EQ(first_metadata.get(), first_metadata_again.get());
  auto stale_fragment_result = LanceTableReader::MetaTrait::create_from_metadata(
      first_metadata_again, second_file, schema_, std::vector<std::string>{}, "");
  EXPECT_FALSE(stale_fragment_result.ok());

  MetadataCache second_reader_cache;
  ASSERT_AND_ASSIGN(auto second_metadata, second_reader_cache.get<LanceTableReader>()->get_or_open(key, [&]() {
    return LanceTableReader::MetaTrait::load_metadata(second_file, properties_, nullptr /* key_retriever */);
  }));
  ASSERT_AND_ASSIGN(auto second_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                            second_metadata, second_file, schema_, std::vector<std::string>{}, ""));

  EXPECT_NE(first_metadata->payload.dataset.get(), second_metadata->payload.dataset.get());
}

TEST_F(LanceBasicTest, SameFragmentMetadataIsReusedWithinReaderCache) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused metadata-cache test uses the local Lance fixture.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto file, writer.Close());

  MetadataCache reader_cache;
  auto cache = reader_cache.get<LanceTableReader>();
  const auto key = LanceTableReader::MetaTrait::cache_key(file);
  size_t metadata_load_count = 0;
  auto load_metadata = [&]() -> arrow::Result<LanceTableReader::MetaTrait::MetadataPtr> {
    ++metadata_load_count;
    return LanceTableReader::MetaTrait::load_metadata(file, properties_, nullptr /* key_retriever */);
  };

  ASSERT_AND_ASSIGN(auto first_metadata, cache->get_or_open(key, load_metadata));
  ASSERT_AND_ASSIGN(auto second_metadata, cache->get_or_open(key, load_metadata));
  EXPECT_EQ(metadata_load_count, 1);
  EXPECT_EQ(first_metadata.get(), second_metadata.get());
}

TEST_F(LanceBasicTest, TestBasic) {
  size_t num_of_batches = 10;
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Build Lance URI from relative path
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  ASSERT_AND_ASSIGN(auto storage_options, milvus_storage::lance::ToWriterOptions(fs_config));

  // write without flush, single fragment
  {
    LanceTableWriter writer(base_path_, schema_, properties_);
    for (int i = 0; i < num_of_batches; i++) {
      ASSERT_STATUS_OK(writer.Write(test_batch_));
    }
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(cgfile.end_index, test_batch_->num_rows() * num_of_batches);
  }

  auto verify_reader = [&]() {
    ASSERT_AND_ASSIGN(auto read_dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));
    ASSERT_AND_ASSIGN(auto fragment_ids, read_dataset->GetAllFragmentIds());

    uint64_t total_rows = 0;
    for (const auto& fragment_id : fragment_ids) {
      LanceTableReader reader(read_dataset, fs_, fragment_id, schema_, properties_);
      ASSERT_STATUS_OK(reader.open());
      ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
      ASSERT_FALSE(rgs.empty());
      total_rows += rgs.back().end_offset;
    }
    ASSERT_EQ(total_rows, test_batch_->num_rows() * num_of_batches);
  };

  verify_reader();
  ASSERT_STATUS_OK(DeleteTestDir(fs_, arrow_base_path_));
  ASSERT_STATUS_OK(CreateTestDir(fs_, arrow_base_path_));

  // write with flush, multiple fragments
  {
    LanceTableWriter writer(base_path_, schema_, properties_);
    for (int i = 0; i < num_of_batches; i++) {
      ASSERT_STATUS_OK(writer.Write(test_batch_));
      ASSERT_STATUS_OK(writer.Flush());
    }
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(cgfile.end_index, test_batch_->num_rows() * num_of_batches);
  }

  verify_reader();
}

TEST_F(LanceBasicTest, TestReaderHandlesFragmentMissingNullableDatasetColumn) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  ASSERT_AND_ASSIGN(auto original_schema, CreateTestSchema({true, true, false, false}));
  ASSERT_AND_ASSIGN(auto original_batch,
                    CreateTestData(original_schema, 0, false, 100, 4, 50, {true, true, false, false}));

  auto evolved_fields = original_schema->fields();
  evolved_fields.push_back(
      arrow::field("new_column", arrow::int32(), true, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"102"})));
  auto evolved_schema = arrow::schema(std::move(evolved_fields));
  ASSERT_AND_ASSIGN(auto new_column, arrow::MakeArrayOfNull(arrow::int32(), original_batch->num_rows()));
  auto evolved_columns = original_batch->columns();
  evolved_columns.push_back(std::move(new_column));
  auto evolved_batch = arrow::RecordBatch::Make(evolved_schema, original_batch->num_rows(), std::move(evolved_columns));

  LanceTableWriter initial_writer(base_path_, evolved_schema, properties_);
  ASSERT_STATUS_OK(initial_writer.Write(evolved_batch));
  ASSERT_AND_ASSIGN(auto initial_file, initial_writer.Close());

  // Lance permits an appended fragment to omit nullable dataset columns. This
  // has the same dataset-schema/physical-fragment shape as an old fragment
  // after a nullable column is added through schema evolution.
  LanceTableWriter append_writer(base_path_, original_schema, properties_);
  ASSERT_STATUS_OK(append_writer.Write(original_batch));
  ASSERT_AND_ASSIGN(auto appended_file, append_writer.Close());

  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(appended_file.path));
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto dataset,
                    BlockingDataset::Open(ToStandardLanceUri(parsed_uri.first), fs_, ToReaderOptions(fs_config)));

  LanceTableReader reader(dataset, fs_, parsed_uri.second, nullptr, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_EQ(reader.get_schema()->num_fields(), evolved_schema->num_fields());

  ASSERT_AND_ASSIGN(auto row_groups, reader.get_row_group_infos());
  ASSERT_FALSE(row_groups.empty());
  for (size_t row_group_index = 0; row_group_index < row_groups.size(); ++row_group_index) {
    ASSERT_AND_ASSIGN(auto memory_sizes, reader.get_rg_column_memsz(row_group_index));
    ASSERT_EQ(memory_sizes.size(), static_cast<size_t>(evolved_schema->num_fields()));
    EXPECT_EQ(memory_sizes.back(), 0);
  }

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1}));
  ASSERT_EQ(table->num_columns(), evolved_schema->num_fields());
  ASSERT_EQ(table->column(evolved_schema->num_fields() - 1)->null_count(), 2);
}

TEST_F(LanceBasicTest, TestRead) {
  ASSERT_AND_ASSIGN(auto large_batch, CreateTestData(schema_, 0, false, 200000));

  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  // Build Lance URI from relative path
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  ASSERT_AND_ASSIGN(auto storage_options, milvus_storage::lance::ToWriterOptions(fs_config));

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(large_batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, large_batch->num_rows());

  ASSERT_AND_ASSIGN(auto read_dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));

  ASSERT_AND_ASSIGN(auto fragment_ids, read_dataset->GetAllFragmentIds());
  // The splitting conditions(`WriteParams`) in lance are very strict.
  // So the default setting will only generate one fragment.
  ASSERT_EQ(fragment_ids.size(), 1);
  LanceTableReader reader(read_dataset, fs_, fragment_ids[0], schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_FALSE(rgs.empty());
  ASSERT_EQ(rgs.back().end_offset, large_batch->num_rows());
  ASSERT_AND_ASSIGN(auto fragment_column_memory_sizes, read_dataset->EstimateFragmentColumnMemory(fragment_ids[0]));
  ASSERT_EQ(fragment_column_memory_sizes.size(), schema_->num_fields());
  auto estimated_memory_size =
      std::accumulate(rgs.begin(), rgs.end(), uint64_t{0},
                      [](uint64_t total, const RowGroupInfo& rg) { return total + rg.memory_size; });
  ASSERT_AND_ASSIGN(auto fragment_memory_size, read_dataset->EstimateFragmentMemory(fragment_ids[0]));
  ASSERT_EQ(estimated_memory_size, fragment_memory_size);
  ASSERT_EQ(std::accumulate(fragment_column_memory_sizes.begin(), fragment_column_memory_sizes.end(), uint64_t{0}),
            estimated_memory_size);
  for (size_t row_group_index = 0; row_group_index < rgs.size(); ++row_group_index) {
    ASSERT_AND_ASSIGN(auto memory_sizes, reader.get_rg_column_memsz(row_group_index));
    ASSERT_EQ(memory_sizes.size(), schema_->num_fields());
    ASSERT_EQ(std::accumulate(memory_sizes.begin(), memory_sizes.end(), uint64_t{0}), rgs[row_group_index].memory_size);
  }

  auto verify_recordbatch = [&](const std::shared_ptr<arrow::RecordBatch>& batch, auto start_ridx, auto num_of_row) {
    ASSERT_EQ(batch->num_rows(), num_of_row);
    auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    for (int i = 0; i < num_of_row; i++) {
      ASSERT_EQ(id_column->Value(i), start_ridx + i);
    }
  };

  // get chunk && read range
  {
    for (size_t rg_idx = 0; rg_idx < rgs.size(); rg_idx++) {
      ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(rg_idx));
      verify_recordbatch(chunk, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);

      ASSERT_AND_ASSIGN(auto rbreader, reader.read_with_range(rgs[rg_idx].start_offset, rgs[rg_idx].end_offset));
      ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rbreader.get()));
      ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
      verify_recordbatch(result_batch, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);
    }
    ASSERT_GT(estimated_memory_size, 0);
  }

  // get chunks
  {
    std::vector<int> chunk_ids(rgs.size());
    std::iota(chunk_ids.begin(), chunk_ids.end(), 0);
    ASSERT_AND_ASSIGN(auto chunks, reader.get_chunks(chunk_ids));
    ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatches(chunks));
    ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
    verify_recordbatch(result_batch, 0, rgs.back().end_offset);
  }

  // test projection
  {
    ASSERT_AND_ASSIGN(auto projection_schema, CreateTestSchema({true, true, false, false}));

    LanceTableReader projection_reader(read_dataset, fs_, fragment_ids[0], projection_schema, properties_);
    ASSERT_STATUS_OK(projection_reader.open());
    ASSERT_AND_ASSIGN(auto projection_rgs, projection_reader.get_row_group_infos());
    ASSERT_EQ(projection_rgs.size(), rgs.size());
    for (size_t rg_idx = 0; rg_idx < rgs.size(); ++rg_idx) {
      ASSERT_EQ(projection_rgs[rg_idx].start_offset, rgs[rg_idx].start_offset);
      ASSERT_EQ(projection_rgs[rg_idx].end_offset, rgs[rg_idx].end_offset);
      ASSERT_EQ(projection_rgs[rg_idx].memory_size, rgs[rg_idx].memory_size);
      ASSERT_AND_ASSIGN(auto projection_memory_sizes, projection_reader.get_rg_column_memsz(rg_idx));
      ASSERT_AND_ASSIGN(auto memory_sizes, reader.get_rg_column_memsz(rg_idx));
      ASSERT_EQ(projection_memory_sizes, memory_sizes);
    }

    for (size_t rg_idx = 0; rg_idx < rgs.size(); rg_idx++) {
      ASSERT_AND_ASSIGN(auto chunk, projection_reader.get_chunk(rg_idx));
      verify_recordbatch(chunk, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);

      ASSERT_AND_ASSIGN(auto rbreader,
                        projection_reader.read_with_range(rgs[rg_idx].start_offset, rgs[rg_idx].end_offset));
      ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rbreader.get()));
      ASSERT_AND_ASSIGN(auto result_batch, table->CombineChunksToBatch());  // for test
      verify_recordbatch(result_batch, rgs[rg_idx].start_offset, rgs[rg_idx].end_offset - rgs[rg_idx].start_offset);
    }
  }
}

TEST_F(LanceBasicTest, ReadsThroughBoundFilesystemMetrics) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused filesystem metrics assertion uses the local Lance fixture.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto dataset,
                    BlockingDataset::Open(ToStandardLanceUri(parsed_uri.first), fs_, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto initial_io_stats, dataset->IOStatsIncremental());
  (void)initial_io_stats;

  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();

  LanceTableReader reader(dataset, fs_, parsed_uri.second, schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  ASSERT_GT(chunk->num_rows(), 0);
  EXPECT_GT(metrics->GetGetFileInfoCount(), 0);
  EXPECT_GT(metrics->GetReadCount(), 0);
  EXPECT_GT(metrics->GetReadBytes(), 0);
  ASSERT_AND_ASSIGN(auto lance_io_stats, dataset->IOStatsIncremental());
  EXPECT_EQ(lance_io_stats.read_bytes, metrics->GetReadBytes());
}

TEST_F(LanceBasicTest, CachedReaderReadsAfterMetadataRelease) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Focused filesystem lifetime assertion uses the local Lance fixture.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  ASSERT_AND_ASSIGN(auto metadata,
                    LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));
  ASSERT_AND_ASSIGN(auto reader, LanceTableReader::MetaTrait::create_from_metadata(metadata, cgfile, schema_,
                                                                                   std::vector<std::string>{}, ""));
  metadata.reset();
  FilesystemCache::getInstance().clean();

  ASSERT_AND_ASSIGN(auto chunk, reader->get_chunk(0));
  EXPECT_EQ(chunk->num_rows(), test_batch_->num_rows());
}

TEST_F(LanceBasicTest, SharedSchedulerReadsAfterOwnerDatasetDrop) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "This regression uses the local Lance fixture.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));

  api::SetValue(properties_, PROPERTY_FS_LANCE_IO_PARALLELISM, "0");
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_EQ(fs_config.lance_io_parallelism, 0);
  auto read_options = ToReaderOptions(fs_config);
  ASSERT_AND_ASSIGN(auto owner_dataset, BlockingDataset::Open(parsed_uri.first, fs_, read_options));
  ASSERT_AND_ASSIGN(auto surviving_dataset, BlockingDataset::Open(parsed_uri.first, fs_, read_options));
  ASSERT_AND_ASSIGN(auto owner_initial_stats, owner_dataset->IOStatsIncremental());
  ASSERT_AND_ASSIGN(auto surviving_initial_stats, surviving_dataset->IOStatsIncremental());
  (void)owner_initial_stats;
  (void)surviving_initial_stats;

  LanceTableReader reader(surviving_dataset, fs_, parsed_uri.second, schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  EXPECT_EQ(chunk->num_rows(), test_batch_->num_rows());
  ASSERT_AND_ASSIGN(auto owner_read_stats, owner_dataset->IOStatsIncremental());
  EXPECT_GT(owner_read_stats.read_bytes, 0);

  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();
  owner_dataset.reset();
  LanceTableReader reader_after_owner_drop(surviving_dataset, fs_, parsed_uri.second, schema_, properties_);
  ASSERT_STATUS_OK(reader_after_owner_drop.open());
  ASSERT_AND_ASSIGN(auto chunk_after_owner_drop, reader_after_owner_drop.get_chunk(0));
  EXPECT_EQ(chunk_after_owner_drop->num_rows(), test_batch_->num_rows());
  EXPECT_GT(metrics->GetReadCount(), 0);
  EXPECT_GT(metrics->GetReadBytes(), 0);
}

TEST_F(LanceBasicTest, EstimatedMemoryAccountsForDeletions) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 10'000;
  constexpr int64_t kDeletedRows = 2'000;
  ASSERT_AND_ASSIGN(auto id_schema, CreateTestSchema({true, false, false, false}));
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(id_schema, 0, false, kRows, 4, 50, {true, false, false, false}));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  ASSERT_AND_ASSIGN(auto storage_options, milvus_storage::lance::ToWriterOptions(fs_config));

  LanceTableWriter writer(base_path_, id_schema, properties_);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);
  ASSERT_STATUS_OK(BlockingDataset::DeleteRows(lance_uri, "id < 2000", storage_options));
  ASSERT_AND_ASSIGN(auto dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));

  ASSERT_AND_ASSIGN(auto fragment_ids, dataset->GetAllFragmentIds());
  ASSERT_EQ(fragment_ids.size(), 1);
  LanceTableReader reader(dataset, fs_, fragment_ids[0], id_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_EQ(rgs.size(), 1);
  ASSERT_EQ(rgs[0].end_offset, kRows - kDeletedRows);
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  ASSERT_EQ(rgs[0].memory_size, GetRecordBatchMemorySize(chunk));
}

TEST_F(LanceBasicTest, FixedSizeListUsesExactMemoryEstimate) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 10'000;
  constexpr int32_t kDimension = 16;
  auto vector_schema = arrow::schema({arrow::field(
      "embedding", arrow::fixed_size_list(arrow::float32(), kDimension), false,
      arrow::key_value_metadata({"lance-encoding:rle-threshold", "lance-encoding:bss"}, {"1.0", "off"}))});

  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder vector_builder(arrow::default_memory_pool(), value_builder, kDimension);
  ASSERT_STATUS_OK(vector_builder.AppendValues(kRows));
  ASSERT_STATUS_OK(value_builder->AppendValues(std::vector<float>(kRows * kDimension, 1.0F)));

  std::shared_ptr<arrow::Array> vector_array;
  ASSERT_STATUS_OK(vector_builder.Finish(&vector_array));
  auto batch = arrow::RecordBatch::Make(vector_schema, kRows, {vector_array});

  LanceTableWriter writer(base_path_, vector_schema, properties_);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));

  LanceTableReader reader(fs_, parsed_uri.first, parsed_uri.second, vector_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  auto estimated_memory_size =
      std::accumulate(rgs.begin(), rgs.end(), uint64_t{0},
                      [](uint64_t total, const RowGroupInfo& rg) { return total + rg.memory_size; });
  ASSERT_EQ(estimated_memory_size, GetRecordBatchMemorySize(batch));
}

#ifdef BUILD_WITH_FIU
TEST_F(LanceBasicTest, MemorySizeEstimationFailureDoesNotBlockOpen) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_AND_ASSIGN(auto parsed_uri, ParseLanceUri(cgfile.path));

  auto assert_memory_size_unavailable = [](const std::vector<RowGroupInfo>& row_group_infos) {
    ASSERT_FALSE(row_group_infos.empty());
    for (const auto& row_group_info : row_group_infos) {
      EXPECT_FALSE(row_group_info.memory_size_available);
      EXPECT_EQ(row_group_info.memory_size, 0u);
    }
  };

  {
    ScopedFiuFault fault(FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);

    LanceTableReader reader(fs_, parsed_uri.first, parsed_uri.second, schema_, properties_);
    ASSERT_STATUS_OK(reader.open());
    ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
    assert_memory_size_unavailable(row_group_infos);
    EXPECT_TRUE(reader.get_rg_column_memsz(0).status().IsNotImplemented());
    ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
    EXPECT_GT(chunk->num_rows(), 0);
  }

  LanceTableReader::MetaTrait::MetadataPtr metadata;
  {
    ScopedFiuFault fault(FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);
    ASSERT_AND_ASSIGN(metadata,
                      LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));
  }

  ASSERT_AND_ASSIGN(auto cached_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                            metadata, cgfile, schema_, std::vector<std::string>{}, ""));
  ASSERT_AND_ASSIGN(auto cached_row_group_infos, cached_reader->get_row_group_infos());
  assert_memory_size_unavailable(cached_row_group_infos);
  EXPECT_TRUE(cached_reader->get_rg_column_memsz(0).status().IsNotImplemented());
  ASSERT_AND_ASSIGN(auto chunk, cached_reader->get_chunk(0));
  EXPECT_GT(chunk->num_rows(), 0);
}
#endif

TEST_F(LanceBasicTest, LegacyFormatReadsWhenMemoryEstimateIsUnavailable) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  constexpr int64_t kRows = 1'024;
  ASSERT_AND_ASSIGN(auto vector_schema, CreateTestSchema({false, false, false, true}));
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(vector_schema, 0, false, kRows, 4, 50, {false, false, false, true}));

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));
  ASSERT_AND_ASSIGN(auto storage_options, milvus_storage::lance::ToWriterOptions(fs_config));

  LanceTableWriter writer(base_path_, vector_schema, properties_, LanceDataStorageFormat::Legacy);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, kRows);

  ASSERT_AND_ASSIGN(auto dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));
  ASSERT_AND_ASSIGN(auto fragment_ids, dataset->GetAllFragmentIds());
  ASSERT_EQ(fragment_ids.size(), 1);

  auto estimate_result = dataset->EstimateFragmentColumnMemory(fragment_ids[0]);
  ASSERT_TRUE(estimate_result.status().IsNotImplemented()) << estimate_result.status().ToString();

  LanceTableReader reader(dataset, fs_, fragment_ids[0], vector_schema, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto row_group_infos, reader.get_row_group_infos());
  ASSERT_FALSE(row_group_infos.empty());
  for (size_t row_group_index = 0; row_group_index < row_group_infos.size(); ++row_group_index) {
    EXPECT_FALSE(row_group_infos[row_group_index].memory_size_available);
    EXPECT_EQ(row_group_infos[row_group_index].memory_size, 0);
    EXPECT_TRUE(reader.get_rg_column_memsz(row_group_index).status().IsNotImplemented());
  }

  ASSERT_AND_ASSIGN(auto rb_reader, reader.read_with_range(0, kRows));
  ASSERT_AND_ASSIGN(auto table, arrow::Table::FromRecordBatchReader(rb_reader.get()));
  ASSERT_EQ(table->num_rows(), kRows);

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = {"vector"};
  column_group->format = LOON_FORMAT_LANCE_TABLE;
  column_group->files = {cgfile};
  auto column_groups = std::make_shared<api::ColumnGroups>();
  column_groups->emplace_back(std::move(column_group));

  auto api_reader = api::Reader::create(column_groups, vector_schema, nullptr, properties_);
  ASSERT_NE(api_reader, nullptr);
  ASSERT_AND_ASSIGN(auto chunk_reader, api_reader->get_chunk_reader(0));
  EXPECT_TRUE(chunk_reader->get_chunk_estimated_size().status().IsNotImplemented());
  EXPECT_TRUE(chunk_reader->get_chunk_column_estimated_size().status().IsNotImplemented());
  ASSERT_AND_ASSIGN(auto chunk, chunk_reader->get_chunk(0));
  ASSERT_GT(chunk->num_rows(), 0);

  ASSERT_AND_ASSIGN(auto packed_reader, api_reader->get_record_batch_reader());
  ASSERT_AND_ASSIGN(auto packed_table, packed_reader->ToTable());
  ASSERT_EQ(packed_table->num_rows(), kRows);
}

TEST_F(LanceBasicTest, TestCachedOpenRejectsMissingNeededColumnWithoutReadSchema) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  ASSERT_AND_ASSIGN(auto metadata,
                    LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));

  auto reader_result = LanceTableReader::MetaTrait::create_from_metadata(metadata, cgfile, nullptr /* read_schema */,
                                                                         {"id", "missing_column"}, "");
  ASSERT_FALSE(reader_result.ok());
  EXPECT_TRUE(reader_result.status().IsInvalid());
  EXPECT_NE(reader_result.status().ToString().find("missing_column"), std::string::npos);
}

TEST_F(LanceBasicTest, CachedCreateReaderReappliesProjection) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Lance fragment writer/reader not supported in cloud environment yet.";
  }

  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());

  ASSERT_AND_ASSIGN(auto metadata,
                    LanceTableReader::MetaTrait::load_metadata(cgfile, properties_, nullptr /* key_retriever */));
  auto id_metadata = metadata;
  auto value_metadata = metadata;
  ASSERT_EQ(id_metadata.get(), value_metadata.get());

  ASSERT_AND_ASSIGN(auto id_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                        id_metadata, cgfile, nullptr /* read_schema */, {"id"}, ""));
  ASSERT_AND_ASSIGN(auto id_rgs, id_reader->get_row_group_infos());
  ASSERT_FALSE(id_rgs.empty());
  for (size_t rg_idx = 0; rg_idx < id_rgs.size(); ++rg_idx) {
    ASSERT_AND_ASSIGN(auto memory_sizes, id_reader->get_rg_column_memsz(rg_idx));
    ASSERT_EQ(memory_sizes.size(), static_cast<size_t>(schema_->num_fields()));
    ASSERT_EQ(std::accumulate(memory_sizes.begin(), memory_sizes.end(), uint64_t{0}), id_rgs[rg_idx].memory_size);
  }
  ASSERT_AND_ASSIGN(auto id_chunk, id_reader->get_chunk(0));
  ASSERT_EQ(id_chunk->num_columns(), 1);
  ASSERT_EQ(id_chunk->schema()->field(0)->name(), "id");
  ASSERT_EQ(id_chunk->num_rows(), static_cast<int64_t>(id_rgs[0].end_offset - id_rgs[0].start_offset));
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(id_chunk->column(0));
  ASSERT_NE(id_array, nullptr);
  for (int64_t i = 0; i < id_chunk->num_rows(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(id_rgs[0].start_offset) + i);
  }

  ASSERT_AND_ASSIGN(auto value_reader, LanceTableReader::MetaTrait::create_from_metadata(
                                           value_metadata, cgfile, nullptr /* read_schema */, {"value"}, ""));
  ASSERT_AND_ASSIGN(auto value_rgs, value_reader->get_row_group_infos());
  ASSERT_FALSE(value_rgs.empty());
  ASSERT_AND_ASSIGN(auto value_chunk, value_reader->get_chunk(0));
  ASSERT_EQ(value_chunk->num_columns(), 1);
  ASSERT_EQ(value_chunk->schema()->field(0)->name(), "value");
  ASSERT_EQ(value_chunk->num_rows(), static_cast<int64_t>(value_rgs[0].end_offset - value_rgs[0].start_offset));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(value_chunk->column(0));
  ASSERT_NE(value_array, nullptr);
  for (int64_t i = 0; i < value_chunk->num_rows(); ++i) {
    const auto row = static_cast<int64_t>(value_rgs[0].start_offset) + i;
    ASSERT_DOUBLE_EQ(value_array->Value(i), row * 1.5);
  }
}

void LanceBasicTest::RunCloudWideTableDuplicatedFragmentTake(int64_t column_count,
                                                             int64_t row_count,
                                                             size_t column_group_file_count,
                                                             size_t reader_count,
                                                             CloudTakeIops& result) {
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  api::SetValue(properties_, "extfs.default.storage_type", fs_config.storage_type.c_str());
  api::SetValue(properties_, "extfs.default.cloud_provider", fs_config.cloud_provider.c_str());
  api::SetValue(properties_, "extfs.default.address", fs_config.address.c_str());
  api::SetValue(properties_, "extfs.default.bucket_name", fs_config.bucket_name.c_str());
  api::SetValue(properties_, "extfs.default.region", fs_config.region.c_str());
  api::SetValue(properties_, "extfs.default.access_key_id", fs_config.access_key_id.c_str());
  api::SetValue(properties_, "extfs.default.access_key_value", fs_config.access_key_value.c_str());
  api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "true");
  const auto lance_io_parallelism = std::to_string(fs_config.lance_io_parallelism);
  api::SetValue(properties_, "extfs.default.lance_io_parallelism", lance_io_parallelism.c_str());
  if (fs_config.use_ssl) {
    api::SetValue(properties_, "extfs.default.use_ssl", "true");
  }
  if (fs_config.use_iam) {
    api::SetValue(properties_, "extfs.default.use_iam", "true");
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::string> column_names;
  fields.reserve(column_count);
  column_names.reserve(column_count);
  for (int64_t column = 0; column < column_count; ++column) {
    column_names.emplace_back("column_" + std::to_string(column));
    fields.emplace_back(arrow::field(column_names.back(), arrow::int64(), false));
  }
  auto wide_schema = arrow::schema(std::move(fields));

  std::vector<int64_t> row_values(row_count);
  std::iota(row_values.begin(), row_values.end(), 0);
  arrow::Int64Builder value_builder;
  ASSERT_STATUS_OK(value_builder.AppendValues(row_values));
  std::shared_ptr<arrow::Array> values;
  ASSERT_STATUS_OK(value_builder.Finish(&values));
  std::vector<std::shared_ptr<arrow::Array>> columns(column_count, values);
  auto wide_batch = arrow::RecordBatch::Make(wide_schema, row_count, std::move(columns));

  LanceTableWriter writer(base_path_, wide_schema, properties_);
  ASSERT_STATUS_OK(writer.Write(wide_batch));
  ASSERT_AND_ASSIGN(auto written_file, writer.Close());
  ASSERT_EQ(written_file.end_index - written_file.start_index, row_count);

  ASSERT_AND_ASSIGN(auto parsed_lance_uri, ParseLanceUri(written_file.path));
  const auto lance_uri = ToStandardLanceUri(parsed_lance_uri.first);
  // IOStatsIncremental is test-only and reads a dataset-local ObjectStore tracker.
  // The shared scheduler retains the ObjectStore from its first dataset, so keep
  // that owner alive while later Reader datasets generate the measured I/O.
  ASSERT_AND_ASSIGN(auto io_stats_owner_dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));
  ASSERT_NE(io_stats_owner_dataset, nullptr);
  ASSERT_AND_ASSIGN(auto non_owner_dataset, BlockingDataset::Open(lance_uri, fs_, ToReaderOptions(fs_config)));
  ASSERT_NE(non_owner_dataset, nullptr);
  ASSERT_NE(io_stats_owner_dataset.get(), non_owner_dataset.get());

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = std::move(column_names);
  column_group->format = LOON_FORMAT_LANCE_TABLE;
  column_group->files.assign(column_group_file_count, written_file);
  auto column_groups = std::make_shared<api::ColumnGroups>();
  column_groups->emplace_back(std::move(column_group));

  std::vector<int64_t> first_row_indices;
  first_row_indices.reserve(column_group_file_count);
  for (size_t file = 0; file < column_group_file_count; ++file) {
    first_row_indices.emplace_back(static_cast<int64_t>(file) * row_count);
  }

  std::vector<std::unique_ptr<api::Reader>> readers;
  readers.reserve(reader_count);
  for (size_t reader_index = 0; reader_index < reader_count; ++reader_index) {
    auto reader = api::Reader::create(column_groups, wide_schema, nullptr, properties_);
    ASSERT_NE(reader, nullptr);
    readers.emplace_back(std::move(reader));
  }

  // Reader::take opens Lance metadata lazily, and IOStats counts those list
  // requests as read_iops. Warm each Reader's metadata cache before resetting
  // the tracker so the measured interval contains only fragment reads.
  for (auto& reader : readers) {
    ASSERT_AND_ASSIGN(auto metadata_reader, reader->get_chunk_reader(0));
    ASSERT_GT(metadata_reader->total_number_of_chunks(), 0);
  }

  std::barrier start_barrier(static_cast<std::ptrdiff_t>(reader_count + 1));
  std::vector<std::future<arrow::Result<std::shared_ptr<arrow::Table>>>> take_futures;
  take_futures.reserve(reader_count);
  for (size_t reader_index = 0; reader_index < readers.size(); ++reader_index) {
    take_futures.emplace_back(
        std::async(std::launch::async, [reader = readers[reader_index].get(), &first_row_indices, &start_barrier]() {
          start_barrier.arrive_and_wait();
          return reader->take(first_row_indices);
        }));
  }

  struct IopsSample {
    std::chrono::steady_clock::time_point timestamp;
    uint64_t iops;
  };
  ASSERT_AND_ASSIGN(auto initial_owner_stats, io_stats_owner_dataset->IOStatsIncremental());
  ASSERT_AND_ASSIGN(auto initial_non_owner_stats, non_owner_dataset->IOStatsIncremental());
  (void)initial_owner_stats;
  (void)initial_non_owner_stats;
  uint64_t total_iops = 0;
  uint64_t total_bytes_read = 0;
  auto collect_io_stats = [&]() -> arrow::Status {
    ARROW_ASSIGN_OR_RAISE(auto stats, io_stats_owner_dataset->IOStatsIncremental());
    total_iops += stats.read_iops;
    total_bytes_read += stats.read_bytes;
    return arrow::Status::OK();
  };
  const auto take_started_at = std::chrono::steady_clock::now();
  start_barrier.arrive_and_wait();

  std::vector<IopsSample> iops_samples;
  while (true) {
    ASSERT_STATUS_OK(collect_io_stats());
    iops_samples.emplace_back(IopsSample{
        .timestamp = std::chrono::steady_clock::now(),
        .iops = total_iops,
    });
    const bool all_ready = std::all_of(take_futures.begin(), take_futures.end(), [](auto& future) {
      return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    });
    if (all_ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const auto take_finished_at = std::chrono::steady_clock::now();
  ASSERT_STATUS_OK(collect_io_stats());
  iops_samples.emplace_back(IopsSample{.timestamp = take_finished_at, .iops = total_iops});

  uint64_t peak_one_second_iops = 0;
  size_t window_start = 0;
  for (size_t window_end = 0; window_end < iops_samples.size(); ++window_end) {
    while (window_start < window_end &&
           iops_samples[window_end].timestamp - iops_samples[window_start].timestamp > std::chrono::seconds(1)) {
      ++window_start;
    }
    peak_one_second_iops =
        std::max(peak_one_second_iops, iops_samples[window_end].iops - iops_samples[window_start].iops);
  }
  const auto take_seconds = std::chrono::duration<double>(take_finished_at - take_started_at).count();
  const auto average_iops = static_cast<double>(total_iops) / take_seconds;
  std::cout << "[ LANCE_IO_STATS ] parallelism=" << fs_config.lance_io_parallelism << ", read_iops=" << total_iops
            << ", read_bytes=" << total_bytes_read << ", duration_s=" << take_seconds
            << ", average_iops=" << average_iops << ", peak_1s_iops=" << peak_one_second_iops << std::endl;
  ASSERT_GT(total_iops, 0);
  ASSERT_GT(total_bytes_read, 0);
  result = CloudTakeIops{
      .total_iops = total_iops,
      .peak_one_second_iops = peak_one_second_iops,
  };

  // Although both handles share the scheduler, IOStatsIncremental is not a
  // scheduler/domain metric. Only the first owner's tracker receives these reads.
  ASSERT_AND_ASSIGN(auto non_owner_stats, non_owner_dataset->IOStatsIncremental());
  ASSERT_EQ(non_owner_stats.read_iops, 0);
  ASSERT_EQ(non_owner_stats.read_bytes, 0);

  for (size_t reader_index = 0; reader_index < take_futures.size(); ++reader_index) {
    SCOPED_TRACE("reader_index=" + std::to_string(reader_index));
    ASSERT_AND_ASSIGN(auto table, take_futures[reader_index].get());
    ASSERT_STATUS_OK(table->ValidateFull());
    ASSERT_EQ(table->num_rows(), static_cast<int64_t>(column_group_file_count));
    ASSERT_EQ(table->num_columns(), column_count);

    ASSERT_AND_ASSIGN(auto batch, table->CombineChunksToBatch());
    for (int column_index : {0, static_cast<int>(column_count - 1)}) {
      auto column = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(column_index));
      ASSERT_NE(column, nullptr);
      for (int64_t row = 0; row < column->length(); ++row) {
        EXPECT_EQ(column->Value(row), 0);
      }
    }
  }
}

TEST_F(LanceBasicTest, CloudWideTableDuplicatedFragmentTakeRateLimitRepro) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "This Lance rate-limit reproduction requires cloud storage.";
  }

  constexpr uint32_t kLanceIoParallelism = 64;
  constexpr uint64_t kMaxPeakIops = 5'500;
  api::SetValue(properties_, PROPERTY_FS_LANCE_IO_PARALLELISM, "64");

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_EQ(fs_config.lance_io_parallelism, kLanceIoParallelism);

  CloudTakeIops result;
  RunCloudWideTableDuplicatedFragmentTake(1'024, 16'384, 10, 10, result);
  ASSERT_LE(result.peak_one_second_iops, kMaxPeakIops)
      << "Shared Lance scheduler exceeded aggregate IOPS target across all Reader instances";
}

TEST_F(LanceBasicTest, CloudConfiguredAimdRateLimit) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "This Lance AIMD rate-limit test requires cloud storage.";
  }

  constexpr uint32_t kAimdRate = 1'500;
  constexpr uint64_t kMaxPeakIops = kAimdRate + 250;
  const auto aimd_rate = std::to_string(kAimdRate);
  api::SetValue(properties_, PROPERTY_FS_LANCE_IO_PARALLELISM, "64");
  api::SetValue(properties_, PROPERTY_FS_IOPS_INITIAL_RATE, aimd_rate.c_str());
  api::SetValue(properties_, PROPERTY_FS_IOPS_MAX_RATE, aimd_rate.c_str());
  // Reader creation resolves the external filesystem independently.
  api::SetValue(properties_, "extfs.default.iops_initial_rate", aimd_rate.c_str());
  api::SetValue(properties_, "extfs.default.iops_max_rate", aimd_rate.c_str());

  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ASSERT_EQ(fs_config.iops_initial_rate, kAimdRate);
  ASSERT_EQ(fs_config.iops_max_rate, kAimdRate);

  const char* previous_burst_capacity = std::getenv("LANCE_AIMD_BURST_CAPACITY");
  const bool had_previous_burst_capacity = previous_burst_capacity != nullptr;
  const std::string saved_burst_capacity = had_previous_burst_capacity ? previous_burst_capacity : "";
  ASSERT_EQ(setenv("LANCE_AIMD_BURST_CAPACITY", "0", 1), 0);
  CloudTakeIops result;
  RunCloudWideTableDuplicatedFragmentTake(256, 8'192, 8, 4, result);
  if (had_previous_burst_capacity) {
    EXPECT_EQ(setenv("LANCE_AIMD_BURST_CAPACITY", saved_burst_capacity.c_str(), 1), 0);
  } else {
    EXPECT_EQ(unsetenv("LANCE_AIMD_BURST_CAPACITY"), 0);
  }
  ASSERT_GT(result.total_iops, kAimdRate * 2);
  ASSERT_LE(result.peak_one_second_iops, kMaxPeakIops) << "Lance ObjectStore exceeded the configured AIMD IOPS target";
}

// Test that storage options are correctly passed through writer and reader
TEST_F(LanceBasicTest, TestStorageOptionsIntegration) {
  // Mirror fs.* into extfs.default.* so resolve_config can match by address+bucket
  ArrowFileSystemConfig fs_config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  if (fs_config.storage_type == "remote") {
    api::SetValue(properties_, "extfs.default.storage_type", "remote");
    api::SetValue(properties_, "extfs.default.cloud_provider", fs_config.cloud_provider.c_str());
    api::SetValue(properties_, "extfs.default.address", fs_config.address.c_str());
    api::SetValue(properties_, "extfs.default.bucket_name", fs_config.bucket_name.c_str());
    api::SetValue(properties_, "extfs.default.region", fs_config.region.c_str());
    api::SetValue(properties_, "extfs.default.access_key_id", fs_config.access_key_id.c_str());
    api::SetValue(properties_, "extfs.default.access_key_value", fs_config.access_key_value.c_str());
    if (fs_config.use_ssl) {
      api::SetValue(properties_, "extfs.default.use_ssl", "true");
    }
    if (fs_config.use_iam) {
      api::SetValue(properties_, "extfs.default.use_iam", "true");
    }
  }

  // Writer uses storage options from properties
  LanceTableWriter writer(base_path_, schema_, properties_);
  ASSERT_STATUS_OK(writer.Write(test_batch_));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  ASSERT_EQ(cgfile.end_index, test_batch_->num_rows());

  // Parse lance_uri and fragment_id from cgfile.path (format: {lance_uri}?fragment_id=X)
  ASSERT_AND_ASSIGN(auto parsed, ParseLanceUri(cgfile.path));
  auto lance_uri = parsed.first;
  auto fragment_id = parsed.second;

  // Reader opens dataset using full Lance URI and storage options from properties
  // Use the URI-based constructor to test the storage options path in open()
  LanceTableReader reader(fs_, lance_uri, fragment_id, schema_, properties_);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rgs, reader.get_row_group_infos());
  ASSERT_FALSE(rgs.empty());
  ASSERT_EQ(rgs.back().end_offset, test_batch_->num_rows());

  // Actually read the data and verify
  ASSERT_AND_ASSIGN(auto chunk, reader.get_chunk(0));
  ASSERT_EQ(chunk->num_rows(), test_batch_->num_rows());

  // Verify the data content
  auto expected_id_column = std::static_pointer_cast<arrow::Int64Array>(test_batch_->column(0));
  auto actual_id_column = std::static_pointer_cast<arrow::Int64Array>(chunk->column(0));
  for (int i = 0; i < chunk->num_rows(); i++) {
    ASSERT_EQ(actual_id_column->Value(i), expected_id_column->Value(i));
  }
}

}  // namespace milvus_storage
