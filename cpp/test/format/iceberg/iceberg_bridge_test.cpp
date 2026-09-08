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
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/filesystem/localfs.h>
#include <arrow/util/io_util.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "bridge_util.h"
#include "iceberg_bridge.h"
#include "rust/cxx.h"
#include "rust-bridge/lib.h"
#include "test_env.h"

namespace milvus_storage::iceberg {
namespace {

struct FilesystemCallCounters {
  std::atomic<int64_t> get_file_info{0};
  std::atomic<int64_t> open_input_file{0};
};

class CountingFileSystem final : public FileSystemProxy {
  public:
  explicit CountingFileSystem(std::shared_ptr<FilesystemCallCounters> counters)
      : FileSystemProxy("/", std::make_shared<arrow::fs::LocalFileSystem>()), counters_(std::move(counters)) {}

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    counters_->get_file_info.fetch_add(1, std::memory_order_relaxed);
    return FileSystemProxy::GetFileInfo(path);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    counters_->open_input_file.fetch_add(1, std::memory_order_relaxed);
    return FileSystemProxy::OpenInputFile(path);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    counters_->open_input_file.fetch_add(1, std::memory_order_relaxed);
    return arrow::fs::SubTreeFileSystem::OpenInputFile(info);
  }

  private:
  std::shared_ptr<FilesystemCallCounters> counters_;
};

class FailingInputFileSystem final : public FileSystemProxy {
  public:
  explicit FailingInputFileSystem(ExtendStatusCode code)
      : FileSystemProxy("/", std::make_shared<arrow::fs::LocalFileSystem>()), code_(code) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string&) override {
    return MakeExtendError(code_, "injected Iceberg filesystem read failure", "injected failure");
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo&) override {
    return MakeExtendError(code_, "injected Iceberg filesystem read failure", "injected failure");
  }

  private:
  ExtendStatusCode code_;
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

static std::filesystem::path NewIcebergBridgeTestDir(const std::string& suffix) {
  auto id = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / ("iceberg-bridge-" + suffix + "-" + std::to_string(id));
}

static ArrowFileSystemConfig LocalReaderConfig() {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "/";
  return config;
}

}  // namespace

class IcebergBridgeTest : public ::testing::Test {};

TEST_F(IcebergBridgeTest, PlanFilesNonexistentLocalMetadataReturnsNotFound) {
  auto filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "/";

  auto result = PlanFiles("/nonexistent/path/v1.metadata.json", 1, filesystem, ToReaderOptions(config));

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(result.status()), ENOENT);
}

TEST_F(IcebergBridgeTest, PlanFilesEmptyMetadataLocationReturnsError) {
  auto filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "/";

  auto result = PlanFiles("", 1, filesystem, ToReaderOptions(config));

  ASSERT_FALSE(result.ok());
  EXPECT_FALSE(result.status().message().empty());
}

TEST_F(IcebergBridgeTest, PlanFilesNegativeSnapshotOnMissingMetadataReturnsError) {
  auto filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "/";

  auto result = PlanFiles("file:///nonexistent/metadata.json", -999, filesystem, ToReaderOptions(config));

  EXPECT_FALSE(result.ok());
}

TEST_F(IcebergBridgeTest, PlanFilesErrorMessageIsDescriptive) {
  auto filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = "/";

  auto result = PlanFiles("/nonexistent/v1.metadata.json", 1, filesystem, ToReaderOptions(config));

  ASSERT_FALSE(result.ok());
  EXPECT_FALSE(result.status().message().empty());
}

TEST_F(IcebergBridgeTest, CreateTestTableErrorReturnsStatus) {
  auto result = CreateTestTable("unsupported://bucket/table", 1, false, {});

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("Failed to create Iceberg test table"), std::string::npos);
}

TEST_F(IcebergBridgeTest, PlanFilesUsesSuppliedFilesystem) {
  auto table_dir = NewIcebergBridgeTestDir("counting");
  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));
  auto counters = std::make_shared<FilesystemCallCounters>();
  auto filesystem = std::make_shared<CountingFileSystem>(counters);

  ASSERT_AND_ASSIGN(auto files, PlanFiles(table_info.metadata_location, table_info.snapshot_id, filesystem,
                                          ToReaderOptions(LocalReaderConfig())));

  ASSERT_FALSE(files.empty());
  EXPECT_GT(counters->get_file_info.load(std::memory_order_relaxed), 0);
  EXPECT_GT(counters->open_input_file.load(std::memory_order_relaxed), 0);
  std::filesystem::remove_all(table_dir);
}

TEST_F(IcebergBridgeTest, PlanFilesRejectsDataFileOutsideBoundLocalRoot) {
  auto test_dir = NewIcebergBridgeTestDir("data-binding");
  auto bound_root = test_dir / "source";
  auto outside_root = test_dir / "target";
  auto table_dir = bound_root / "table";
  ASSERT_EQ(bound_root.string().size(), outside_root.string().size());

  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));
  auto outside_data_uri = table_info.data_file_uri;
  auto root_pos = outside_data_uri.find(bound_root.string());
  ASSERT_NE(root_pos, std::string::npos);
  outside_data_uri.replace(root_pos, bound_root.string().size(), outside_root.string());
  ASSERT_EQ(outside_data_uri.size(), table_info.data_file_uri.size());

  auto manifest_path = table_dir / "metadata" / "manifest-data-0.avro";
  std::ifstream manifest_input(manifest_path, std::ios::binary);
  ASSERT_TRUE(manifest_input.is_open());
  std::string manifest_bytes((std::istreambuf_iterator<char>(manifest_input)), std::istreambuf_iterator<char>());
  manifest_input.close();

  auto uri_pos = manifest_bytes.find(table_info.data_file_uri);
  ASSERT_NE(uri_pos, std::string::npos);
  manifest_bytes.replace(uri_pos, table_info.data_file_uri.size(), outside_data_uri);
  ASSERT_EQ(manifest_bytes.find(table_info.data_file_uri, uri_pos + outside_data_uri.size()), std::string::npos);

  std::ofstream manifest_output(manifest_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(manifest_output.is_open());
  manifest_output.write(manifest_bytes.data(), static_cast<std::streamsize>(manifest_bytes.size()));
  manifest_output.close();
  ASSERT_TRUE(manifest_output.good());

  auto filesystem =
      std::make_shared<FileSystemProxy>(bound_root.string(), std::make_shared<arrow::fs::LocalFileSystem>());
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.root_path = bound_root.string();

  auto result = PlanFiles(table_info.metadata_location, table_info.snapshot_id, filesystem, ToReaderOptions(config));

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("outside the bound filesystem root"), std::string::npos)
      << result.status().ToString();
  std::filesystem::remove_all(test_dir);
}

TEST_F(IcebergBridgeTest, PlanFilesPreservesFilesystemErrorDetails) {
  auto table_dir = NewIcebergBridgeTestDir("errors");
  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));

  for (auto code : {ExtendStatusCode::AwsErrorAccessDenied, ExtendStatusCode::StorageTransientThrottling}) {
    auto filesystem = std::make_shared<FailingInputFileSystem>(code);
    auto result = PlanFiles(table_info.metadata_location, table_info.snapshot_id, filesystem,
                            ToReaderOptions(LocalReaderConfig()));

    ASSERT_FALSE(result.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), code);
  }
  std::filesystem::remove_all(table_dir);
}

TEST_F(IcebergBridgeTest, PlanFilesReleasesFilesystemLease) {
  auto table_dir = NewIcebergBridgeTestDir("lease");
  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));
  std::weak_ptr<arrow::fs::FileSystem> weak;
  {
    auto filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
    weak = filesystem;
    ASSERT_AND_ASSIGN(auto files, PlanFiles(table_info.metadata_location, table_info.snapshot_id, filesystem,
                                            ToReaderOptions(LocalReaderConfig())));
    ASSERT_FALSE(files.empty());
    filesystem.reset();
  }

  EXPECT_TRUE(weak.expired());
  std::filesystem::remove_all(table_dir);
}

#ifdef WITH_CRT

TEST_F(IcebergBridgeTest, PlanFilesReleasesAsyncReaderOutsideCrtCompletionCallback) {
  auto table_dir = NewIcebergBridgeTestDir("async-reader-lifecycle");
  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));

  auto lifecycle = std::make_shared<AsyncReaderLifecycleState>();
  auto local_filesystem = std::make_shared<FileSystemProxy>("/", std::make_shared<arrow::fs::LocalFileSystem>());
  auto callback_filesystem = std::make_shared<CallbackThreadFileSystem>(local_filesystem, lifecycle);
  auto plan_future = std::async(std::launch::async, [&]() {
    return PlanFiles(table_info.metadata_location, table_info.snapshot_id, callback_filesystem,
                     ToReaderOptions(LocalReaderConfig()));
  });

  if (!lifecycle->WaitForSubmitted(std::chrono::seconds(5))) {
    lifecycle->AllowCompletions();
    plan_future.wait();
    lifecycle->JoinWorkers();
    std::filesystem::remove_all(table_dir);
    FAIL() << "The asynchronous Iceberg reader was not submitted";
  }
  lifecycle->AllowCompletions();
  auto plan_result = plan_future.get();
  lifecycle->JoinWorkers();
  const bool all_released = lifecycle->WaitForAllCompletedAndDestroyed(std::chrono::seconds(5));
  std::filesystem::remove_all(table_dir);

  ASSERT_TRUE(plan_result.ok()) << plan_result.status().ToString();
  ASSERT_TRUE(all_released) << "The asynchronous Iceberg readers were not all completed and destroyed";
  EXPECT_FALSE(lifecycle->close_in_completion()) << "The CRT completion thread closed ReaderHandle inside the callback";
  EXPECT_FALSE(lifecycle->destroy_in_completion())
      << "The CRT completion thread destroyed RandomAccessFile inside the callback";
}

TEST_F(IcebergBridgeTest, PlanFilesSurvivesFilesystemCacheClearDuringInflightRead) {
  auto table_dir = NewIcebergBridgeTestDir("cache-clear");
  ASSERT_AND_ASSIGN(auto table_info, CreateTestTable(table_dir.string(), 10, false, {}));

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, "local");
  api::SetValue(properties, PROPERTY_FS_ROOT_PATH, "/");
  auto& cache = FilesystemCache::getInstance();
  cache.clean();
  ASSERT_AND_ASSIGN(auto cached_filesystem, GetFileSystem(properties));
  ASSERT_EQ(cache.size(), 1);

  auto lifecycle = std::make_shared<AsyncReaderLifecycleState>();
  auto callback_filesystem = std::make_shared<CallbackThreadFileSystem>(cached_filesystem, lifecycle);
  std::weak_ptr<arrow::fs::FileSystem> cached_weak = cached_filesystem;
  std::weak_ptr<arrow::fs::FileSystem> callback_weak = callback_filesystem;
  rust::Vec<rust::String> keys, values;
  ConvertStorageOptions(ToReaderOptions(LocalReaderConfig()), keys, values);
  auto lease = std::make_shared<FileSystemWrapper>(callback_filesystem);
  auto plan_future =
      std::async(std::launch::async, [metadata_location = table_info.metadata_location,
                                      snapshot_id = table_info.snapshot_id, filesystem = std::move(lease),
                                      keys = std::move(keys), values = std::move(values)]() mutable {
        return CatchRustResult<size_t>("Failed to plan Iceberg files", [&]() {
          auto files = ffi::iceberg_plan_files(std::move(filesystem),
                                               rust::Str(metadata_location.data(), metadata_location.length()),
                                               snapshot_id, std::move(keys), std::move(values));
          return files.size();
        });
      });

  if (!lifecycle->WaitForSubmitted(std::chrono::seconds(5))) {
    lifecycle->AllowCompletions();
    plan_future.wait();
    lifecycle->JoinWorkers();
    cache.clean();
    std::filesystem::remove_all(table_dir);
    FAIL() << "The asynchronous Iceberg reader was not submitted";
  }

  cache.clean();
  cached_filesystem.reset();
  callback_filesystem.reset();
  EXPECT_EQ(cache.size(), 0);
  EXPECT_FALSE(cached_weak.expired());
  EXPECT_FALSE(callback_weak.expired());

  lifecycle->AllowCompletions();
  auto plan_result = plan_future.get();
  lifecycle->JoinWorkers();
  const bool all_released = lifecycle->WaitForAllCompletedAndDestroyed(std::chrono::seconds(5));
  std::filesystem::remove_all(table_dir);

  ASSERT_TRUE(plan_result.ok()) << plan_result.status().ToString();
  ASSERT_GT(*plan_result, 0);
  ASSERT_TRUE(all_released) << "The asynchronous Iceberg readers were not all completed and destroyed";
  EXPECT_TRUE(callback_weak.expired());
  EXPECT_TRUE(cached_weak.expired());
}

#endif  // WITH_CRT

// IcebergFileInfo default construction
TEST_F(IcebergBridgeTest, FileInfoStructConstruction) {
  IcebergFileInfo info;
  info.data_file_path = "s3://bucket/table/data/file.parquet";
  info.record_count = 1000;
  info.delete_metadata_json = {'{', '}'};

  EXPECT_EQ(info.data_file_path, "s3://bucket/table/data/file.parquet");
  EXPECT_EQ(info.record_count, 1000);
  EXPECT_EQ(info.delete_metadata_json.size(), 2);
}

// Empty delete metadata
TEST_F(IcebergBridgeTest, FileInfoEmptyDeleteMetadata) {
  IcebergFileInfo info;
  info.data_file_path = "data.parquet";
  info.record_count = 100;

  EXPECT_TRUE(info.delete_metadata_json.empty());
}

}  // namespace milvus_storage::iceberg
