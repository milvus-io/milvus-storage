// Copyright 2024 Zilliz
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
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/filesystem/localfs.h>
#include <boost/filesystem.hpp>

#include <random>
#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <atomic>
#include <set>
#include <thread>
#include <latch>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <mutex>
#include <string>
#include <vector>

#include "test_env.h"
#include "milvus-storage/lob_column/lob_column_manager.h"
#include "milvus-storage/lob_column/lob_column_writer.h"
#include "milvus-storage/lob_column/lob_column_reader.h"
#include "milvus-storage/lob_column/lob_reference.h"

namespace milvus_storage::lob_column {

class LobColumnReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/lob_column_reader_test_" + std::to_string(std::random_device{}());
    boost::filesystem::create_directories(test_dir_);

    // create filesystem
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

    // create config with small thresholds for testing
    config_.lob_base_path = test_dir_ + "/lobs/100";
    config_.field_id = 100;
    config_.inline_threshold = 20;              // small threshold for testing inline vs LOB
    config_.max_lob_file_bytes = 1024 * 1024;   // 1MB for testing
    config_.flush_threshold_bytes = 64 * 1024;  // 64KB for testing

    api::Properties config_properties;
    ASSERT_STATUS_OK(InitTestProperties(config_properties));
    config_.properties = std::make_shared<const api::Properties>(std::move(config_properties));
  }

  void TearDown() override {
    // cleanup test directory
    boost::filesystem::remove_all(test_dir_);
  }

  std::string GenerateRandomString(size_t length) {
    static const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; i++) {
      result.push_back(charset[dis(gen)]);
    }
    return result;
  }

  // helper to write test data and return references
  std::vector<std::vector<uint8_t>> WriteTestData(const std::vector<std::string>& texts) {
    auto manager_result = LobColumnManager::Create(fs_, config_);
    EXPECT_TRUE(manager_result.ok());
    auto manager = std::move(manager_result).ValueOrDie();

    auto writer_result = manager->CreateWriter();
    EXPECT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto refs_result = writer->WriteBatch(texts);
    EXPECT_TRUE(refs_result.ok());

    auto close_result = writer->Close();
    EXPECT_TRUE(close_result.ok());

    return std::move(refs_result).ValueOrDie();
  }

  protected:
  std::string test_dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  LobColumnConfig config_;
  folly::CPUThreadPoolExecutor executor_{1};
};

TEST_F(LobColumnReaderTest, AsyncMultiFileOrderAndDuplicates) {
  std::vector<std::vector<uint8_t>> encoded;
  std::vector<std::string> values;
  for (int file = 0; file < 5; ++file) {
    std::vector<std::string> group{GenerateRandomString(65536), GenerateRandomString(65536), "inline"};
    auto refs = WriteTestData(group);
    encoded.insert(encoded.end(), refs.begin(), refs.end());
    values.insert(values.end(), group.begin(), group.end());
  }
  std::vector<EncodedRef> refs;
  std::vector<std::string> expected;
  for (size_t index : {12, 0, 7, 3, 1, 10, 4, 6, 9, 13, 2, 0}) {
    refs.push_back({encoded[index].data(), encoded[index].size()});
    expected.push_back(values[index]);
  }
  refs.push_back({nullptr, 0});
  expected.emplace_back();
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok()) << created.status();
  auto reader = std::move(created).ValueOrDie();
  auto default_result = reader->ReadBatchAsync(refs).via(&executor_).get();
  ASSERT_TRUE(default_result.ok()) << default_result.status();
  EXPECT_EQ(default_result.ValueOrDie(), expected);
  for (size_t threads : {1, 4}) {
    SCOPED_TRACE(threads);
    folly::CPUThreadPoolExecutor caller_executor(threads);
    auto result = reader->ReadBatchAsync(refs).via(&caller_executor).get();
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result.ValueOrDie(), expected);
  }
}

TEST_F(LobColumnReaderTest, InvalidRefDoesNotPoisonReader) {
  auto encoded = WriteTestData({GenerateRandomString(65536)});
  std::vector<EncodedRef> good{{encoded[0].data(), encoded[0].size()}};
  // Invalid input must not poison a later valid async read.
  auto bad = encoded[0];
  std::vector<EncodedRef> truncated{{bad.data(), bad.size() - 1}};
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  EXPECT_FALSE(reader->ReadBatchAsync(truncated).via(&executor_).get().ok());
  EXPECT_TRUE(reader->ReadBatchAsync(good).via(&executor_).get().ok());
  ASSERT_TRUE(reader->Close().ok());
  EXPECT_FALSE(reader->ReadBatchAsync(good).via(&executor_).get().ok());
}

namespace {
struct LobReadGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool hold = false;
  size_t active = 0;
  std::string fail_path;
  std::set<std::string> blocked_paths;
  bool failed = false;
  arrow::Status Enter(const std::string& path) {
    std::unique_lock lock(mutex);
    if (path == fail_path) {
      failed = true;
      cv.notify_all();
      return arrow::Status::IOError("injected async LOB read failure");
    }
    ++active;
    if (hold)
      blocked_paths.insert(path);
    cv.notify_all();
    cv.wait(lock, [&] { return !hold; });
    --active;
    return arrow::Status::OK();
  }
  bool WaitForFailure() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return failed; });
  }
  bool WaitForBlockedFiles(size_t count = 1) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return blocked_paths.size() >= count; });
  }
  bool WaitForActiveOperations(size_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return active >= count; });
  }
  void Release() {
    std::lock_guard lock(mutex);
    hold = false;
    cv.notify_all();
  }
};
class GatedLobFile : public arrow::io::RandomAccessFile {
  public:
  GatedLobFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::shared_ptr<LobReadGate> gate, std::string path)
      : file_(std::move(file)), gate_(std::move(gate)), path_(std::move(path)) {}
  arrow::Status Close() override { return file_->Close(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Status Seek(int64_t offset) override { return file_->Seek(offset); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Result<int64_t> Read(int64_t bytes, void* out) override { return file_->Read(bytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t bytes) override { return file_->Read(bytes); }
  arrow::Result<int64_t> ReadAt(int64_t offset, int64_t bytes, void* out) override {
    ARROW_RETURN_NOT_OK(gate_->Enter(path_));
    return file_->ReadAt(offset, bytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t offset, int64_t bytes) override {
    ARROW_RETURN_NOT_OK(gate_->Enter(path_));
    return file_->ReadAt(offset, bytes);
  }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<LobReadGate> gate_;
  std::string path_;
};
class GatedLobFileSystem : public arrow::fs::LocalFileSystem {
  public:
  std::atomic<size_t> opens{0};
  std::thread::id LastOpenThread() {
    std::lock_guard lock(open_mutex_);
    return last_open_thread_;
  }
  explicit GatedLobFileSystem(std::shared_ptr<LobReadGate> gate, bool gate_opens = false)
      : gate_(std::move(gate)), gate_opens_(gate_opens) {}
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ++opens;
    {
      std::lock_guard lock(open_mutex_);
      last_open_thread_ = std::this_thread::get_id();
    }
    if (gate_opens_)
      ARROW_RETURN_NOT_OK(gate_->Enter(path));
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::LocalFileSystem::OpenInputFile(path));
    return std::make_shared<GatedLobFile>(std::move(file), gate_, path);
  }
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenInputFile(info.path());
  }

  private:
  std::mutex open_mutex_;
  std::thread::id last_open_thread_;
  std::shared_ptr<LobReadGate> gate_;
  const bool gate_opens_;
};
}  // namespace

TEST_F(LobColumnReaderTest, AsyncReadFailureDrainsOtherFilesAndRecovers) {
  using namespace std::chrono_literals;
  std::vector<std::string> values;
  std::vector<std::vector<uint8_t>> encoded;
  for (int i = 0; i < 2; ++i) {
    values.push_back(GenerateRandomString(1024 * 1024));
    encoded.push_back(WriteTestData({values.back()}).front());
  }
  // Put the failure first in file-map order. Without draining, returning its
  // error would abandon the other file's still-blocked scan.
  const auto first_id = DecodeLOBReference(encoded[0].data()).file_id;
  const auto second_id = DecodeLOBReference(encoded[1].data()).file_id;
  const size_t bad_index = first_id < second_id ? 0 : 1;
  std::vector<EncodedRef> valid;
  for (auto& ref : encoded) valid.push_back({ref.data(), ref.size()});
  for (bool io_failure : {false, true}) {
    SCOPED_TRACE(io_failure ? "async I/O error" : "ready error from out-of-range row");
    auto gate = std::make_shared<LobReadGate>();
    auto fs = std::make_shared<GatedLobFileSystem>(gate);
    auto created = CreateLobColumnReader(fs, config_);
    ASSERT_TRUE(created.ok());
    auto reader = std::move(created).ValueOrDie();
    auto warmup = reader->ReadBatch(valid);
    ASSERT_TRUE(warmup.ok());
    ASSERT_EQ(warmup.ValueOrDie(), values);
    auto refs = valid;
    constexpr size_t file_id_offset = LOB_REFERENCE_SIZE - UUID_BINARY_SIZE - sizeof(int32_t);
    auto bad = EncodeLOBReference(encoded[bad_index].data() + file_id_offset, 1);
    ASSERT_EQ(bad.size(), LOB_REFERENCE_SIZE);
    if (io_failure) {
      gate->fail_path = BuildLOBFilePath(config_.lob_base_path, DecodeLOBReference(encoded[bad_index].data()).file_id);
    } else {
      refs[bad_index] = {bad.data(), bad.size()};  // Valid reference, but the file has only row 0.
    }
    gate->hold = true;
    auto worker = reader->ReadBatchAsync(refs).via(&executor_);
    const bool reading = gate->WaitForBlockedFiles();
    const bool failed = !io_failure || gate->WaitForFailure();
    const bool pending = !worker.isReady();
    gate->Release();
    auto result = std::move(worker).get();
    EXPECT_TRUE(reading);
    EXPECT_TRUE(failed);
    EXPECT_TRUE(pending);
    ASSERT_FALSE(result.ok());
    if (io_failure) {
      EXPECT_TRUE(result.status().IsIOError()) << result.status();
      EXPECT_NE(result.status().message().find("injected async LOB read failure"), std::string::npos);
    } else {
      EXPECT_TRUE(result.status().IsInvalid()) << result.status();
      EXPECT_NE(result.status().message().find("Row index out of range"), std::string::npos);
    }
    EXPECT_EQ(gate->active, 0);
    gate->fail_path.clear();
    auto recovered = reader->ReadBatchAsync(valid).via(&executor_).get();
    ASSERT_TRUE(recovered.ok()) << recovered.status();
    EXPECT_EQ(recovered.ValueOrDie(), values);
    ASSERT_TRUE(reader->Close().ok());
  }
}

TEST_F(LobColumnReaderTest, AsyncUsesCallerExecutorOwnsInputAndReusesCache) {
  const std::vector<std::string> values{GenerateRandomString(65536), "inline", ""};
  auto encoded = WriteTestData(values);
  std::vector<EncodedRef> refs;
  for (auto& ref : encoded) refs.push_back({ref.data(), ref.size()});
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  auto pending = reader->ReadBatchAsync(refs);
  EXPECT_EQ(fs->opens.load(), 0);  // Cold opens must wait for the caller executor.
  for (auto& ref : encoded) std::fill(ref.begin(), ref.end(), 0xff);
  encoded.clear();
  refs.clear();
  auto executor_thread = folly::via(&executor_, [] { return std::this_thread::get_id(); }).get();
  auto result = std::move(pending).via(&executor_).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), values);
  EXPECT_EQ(fs->LastOpenThread(), executor_thread);
  const auto opens = fs->opens.load();
  EXPECT_GT(opens, 0);

  // Reuse the same file across async and original synchronous entry points.
  auto encoded_again = WriteTestData({values[0]});
  std::vector<EncodedRef> again{{encoded_again[0].data(), encoded_again[0].size()}};
  ASSERT_TRUE(reader->ReadBatchAsync(again).via(&executor_).get().ok());
  const auto after_open = fs->opens.load();
  EXPECT_GT(after_open, opens);
  ASSERT_TRUE(reader->ReadBatch(again).ok());
  auto repeat = reader->ReadBatchAsync(again).via(&executor_).get();
  ASSERT_TRUE(repeat.ok());
  EXPECT_EQ(repeat.ValueOrDie(), std::vector<std::string>{values[0]});
  EXPECT_EQ(fs->opens.load(), after_open);
  reader->ClearCache();
  auto reopened = reader->ReadBatchAsync(again).via(&executor_).get();
  ASSERT_TRUE(reopened.ok()) << reopened.status();
  EXPECT_EQ(reopened.ValueOrDie(), std::vector<std::string>{values[0]});
  EXPECT_GT(fs->opens.load(), after_open);
}

TEST_F(LobColumnReaderTest, AsyncFansOutAllFilesWithoutBlockingExecutor) {
  std::vector<std::string> values;
  std::vector<std::vector<uint8_t>> encoded;
  for (size_t i = 0; i < 3; ++i) {
    values.push_back(GenerateRandomString(1024 * 1024));
    encoded.push_back(WriteTestData({values.back()}).front());
  }
  std::vector<EncodedRef> refs;
  for (auto& ref : encoded) refs.push_back({ref.data(), ref.size()});
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  ASSERT_TRUE(reader->ReadBatch(refs).ok());
  gate->hold = true;
  auto pending = reader->ReadBatchAsync(refs).via(&executor_);
  const bool reading = gate->WaitForBlockedFiles(3);
  size_t blocked_files;
  {
    std::lock_guard lock(gate->mutex);
    blocked_files = gate->blocked_paths.size();
  }
  // A one-thread caller executor must remain available while native I/O waits.
  std::promise<void> ping;
  auto ping_result = ping.get_future();
  executor_.add([&ping] { ping.set_value(); });
  const auto responsive = ping_result.wait_for(std::chrono::seconds(5));
  const bool returned_early = pending.isReady();
  gate->Release();  // Release before any fatal assertion.
  auto result = std::move(pending).get();
  ping_result.wait();
  EXPECT_TRUE(reading);
  EXPECT_EQ(blocked_files, 3);
  EXPECT_EQ(responsive, std::future_status::ready);
  EXPECT_FALSE(returned_early);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), values);
}

TEST_F(LobColumnReaderTest, AsyncEmptyInlineAndBinary) {
  config_.data_type = LobDataType::kBinary;
  std::string binary(65536, '\0');
  binary[100] = static_cast<char>(0xff);
  const std::vector<std::string> values{binary, std::string("a\0b", 3)};
  auto encoded = WriteTestData(values);
  std::vector<EncodedRef> refs;
  for (auto& ref : encoded) refs.push_back({ref.data(), ref.size()});
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  auto empty = reader->ReadBatchAsync({}).via(&executor_).get();
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty.ValueOrDie().empty());
  auto result = reader->ReadBatchAsync(refs).via(&executor_).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), values);
}

TEST_F(LobColumnReaderTest, ConcurrentAsyncCallsOverlapAndReuseWarmCache) {
  const std::vector<std::string> values{GenerateRandomString(1024 * 1024), GenerateRandomString(1024 * 1024)};
  auto first = WriteTestData({values[0]}).front();
  auto second = WriteTestData({values[1]}).front();
  std::vector<EncodedRef> a{{first.data(), first.size()}}, b{{second.data(), second.size()}};
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  folly::CPUThreadPoolExecutor pool(4);
  ASSERT_TRUE(reader->ReadBatchAsync(a).via(&pool).get().ok());
  ASSERT_TRUE(reader->ReadBatchAsync(b).via(&pool).get().ok());
  const auto opened = fs->opens.load();
  gate->hold = true;
  // Distinct files avoid mistaking backend same-file I/O coalescing for serialization.
  auto fa = reader->ReadBatchAsync(a).via(&pool);
  auto fb = reader->ReadBatchAsync(b).via(&pool);
  const bool overlapped = gate->WaitForBlockedFiles(2);
  const bool both_pending = !fa.isReady() && !fb.isReady();
  gate->Release();
  auto ra = std::move(fa).get();
  auto rb = std::move(fb).get();
  EXPECT_TRUE(overlapped);
  EXPECT_TRUE(both_pending);
  ASSERT_TRUE(ra.ok()) << ra.status();
  ASSERT_TRUE(rb.ok()) << rb.status();
  EXPECT_EQ(ra.ValueOrDie(), std::vector<std::string>{values[0]});
  EXPECT_EQ(rb.ValueOrDie(), std::vector<std::string>{values[1]});
  EXPECT_EQ(fs->opens.load(), opened);

  // Concurrent calls may also share the same cached Vortex reader.
  std::vector<folly::Future<arrow::Result<std::vector<std::string>>>> pending;
  for (size_t i = 0; i < 16; ++i) pending.push_back(reader->ReadBatchAsync(a).via(&pool));
  for (auto& future : pending) {
    auto result = std::move(future).get();
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{values[0]});
  }
  EXPECT_EQ(fs->opens.load(), opened);
}

TEST_F(LobColumnReaderTest, ClearCacheDoesNotWaitForColdOpenOrReuseItsResult) {
  const auto value = GenerateRandomString(65536);
  auto encoded = WriteTestData({value}).front();
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  gate->hold = true;
  auto pending = reader->ReadBatchAsync(refs).via(&executor_);
  const bool opening = gate->WaitForBlockedFiles();
  auto clear = std::async(std::launch::async, [&] { reader->ClearCache(); });
  const auto cleared = clear.wait_for(std::chrono::seconds(1));
  gate->Release();
  clear.get();
  auto result = std::move(pending).get();
  EXPECT_TRUE(opening);
  EXPECT_EQ(cleared, std::future_status::ready);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
  const auto old_opens = fs->opens.load();
  // The pre-clear open must not publish into the replacement cache.
  auto fresh = reader->ReadBatchAsync(refs).via(&executor_).get();
  ASSERT_TRUE(fresh.ok()) << fresh.status();
  EXPECT_EQ(fresh.ValueOrDie(), std::vector<std::string>{value});
  EXPECT_GT(fs->opens.load(), old_opens);
  const auto new_opens = fs->opens.load();
  ASSERT_TRUE(reader->ReadBatch(refs).ok());
  ASSERT_TRUE(reader->ReadBatchAsync(refs).via(&executor_).get().ok());
  EXPECT_EQ(fs->opens.load(), new_opens);
}

TEST_F(LobColumnReaderTest, CloseAndDestructionDoNotInterruptAdmittedAsyncReads) {
  const auto value = GenerateRandomString(1024 * 1024);
  auto encoded = WriteTestData({value}).front();
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
  for (bool warm : {false, true}) {
    SCOPED_TRACE(warm);
    auto gate = std::make_shared<LobReadGate>();
    auto fs = std::make_shared<GatedLobFileSystem>(gate);
    auto created = CreateLobColumnReader(fs, config_);
    ASSERT_TRUE(created.ok());
    auto reader = std::move(created).ValueOrDie();
    if (warm)
      ASSERT_TRUE(reader->ReadBatchAsync(refs).via(&executor_).get().ok());
    gate->hold = true;
    auto pending = reader->ReadBatchAsync(refs).via(&executor_);
    const bool reading = gate->WaitForBlockedFiles();
    auto close = std::async(std::launch::async, [&] { return reader->Close(); });
    const auto closed = close.wait_for(std::chrono::seconds(1));
    bool marked_closed = false;
    bool rejects_sync = false;
    bool rejects_async = false;
    bool stays_closed = false;
    arrow::Status close_status;
    if (closed == std::future_status::ready) {
      close_status = close.get();
      marked_closed = reader->IsClosed();
      rejects_sync = !reader->ReadBatch(refs).ok();
      rejects_async = !reader->ReadBatchAsync(refs).get().ok();
      reader->ClearCache();
      stays_closed = reader->IsClosed() && reader->Close().ok();
      // Destroy the public reader while the admitted operation is still blocked.
      reader.reset();
    }
    gate->Release();  // Always release before assertions or joining a regressed Close.
    if (closed != std::future_status::ready)
      close_status = close.get();
    EXPECT_TRUE(close_status.ok());
    EXPECT_TRUE(reading);
    EXPECT_EQ(closed, std::future_status::ready);
    EXPECT_TRUE(marked_closed);
    EXPECT_TRUE(rejects_sync);
    EXPECT_TRUE(rejects_async);
    EXPECT_TRUE(stays_closed);
    auto result = std::move(pending).get();
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
  }
}

TEST_F(LobColumnReaderTest, DeferredFutureOwnsResourcesAfterCloseAndDestruction) {
  const auto value = GenerateRandomString(65536);
  auto encoded = WriteTestData({value}).front();
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  auto pending = reader->ReadBatchAsync(refs);
  ASSERT_TRUE(reader->Close().ok());  // Must not wait for this unconsumed future.
  reader.reset();
  encoded.clear();
  refs.clear();
  auto result = std::move(pending).via(&executor_).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
}

TEST_F(LobColumnReaderTest, ConcurrentSyncAsyncReadsAndCacheClears) {
  const std::vector<std::string> values{GenerateRandomString(65536), GenerateRandomString(65536)};
  auto encoded = WriteTestData(values);
  std::vector<EncodedRef> refs;
  for (auto& ref : encoded) refs.push_back({ref.data(), ref.size()});
  arrow::BinaryBuilder builder;
  for (const auto& ref : encoded) ASSERT_STATUS_OK(builder.Append(ref.data(), static_cast<int32_t>(ref.size())));
  std::shared_ptr<arrow::BinaryArray> input;
  ASSERT_STATUS_OK(builder.Finish(&input));
  auto file_id = DecodeLOBReference(encoded[0].data()).file_id;
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  folly::CPUThreadPoolExecutor pool(4);
  std::latch start(1);
  std::vector<std::future<bool>> workers;
  for (size_t kind = 0; kind < 5; ++kind) {
    workers.push_back(std::async(std::launch::async, [&, kind] {
      start.wait();
      for (size_t i = 0; i < 20; ++i) {
        if (kind == 0) {
          auto r = reader->ReadText(refs[0].data, refs[0].size);
          if (!r.ok() || r.ValueOrDie() != values[0])
            return false;
        } else if (kind == 1) {
          auto r = reader->ReadBatch(refs);
          if (!r.ok() || r.ValueOrDie() != values)
            return false;
        } else if (kind == 2) {
          auto r = reader->Take(file_id, {0, 1});
          if (!r.ok() || r.ValueOrDie() != values)
            return false;
        } else if (kind == 3) {
          auto r = reader->ReadArrowArray(input);
          if (!r.ok() || r.ValueOrDie()->length() != 2 || r.ValueOrDie()->GetString(0) != values[0] ||
              r.ValueOrDie()->GetString(1) != values[1])
            return false;
        } else {
          auto r = reader->ReadBatchAsync(refs).via(&pool).get();
          if (!r.ok() || r.ValueOrDie() != values)
            return false;
        }
      }
      return true;
    }));
  }
  auto clearer = std::async(std::launch::async, [&] {
    start.wait();
    for (size_t i = 0; i < 20; ++i) {
      reader->ClearCache();
      std::this_thread::yield();
    }
  });
  start.count_down();
  clearer.get();
  for (auto& worker : workers) EXPECT_TRUE(worker.get());
}

TEST_F(LobColumnReaderTest, ConcurrentColdMissesPublishReusableReader) {
  const auto value = GenerateRandomString(65536);
  auto encoded = WriteTestData({value}).front();
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate, /*gate_opens=*/true);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  constexpr size_t callers = 4;
  folly::CPUThreadPoolExecutor pool(callers);
  gate->hold = true;
  std::vector<folly::Future<arrow::Result<std::vector<std::string>>>> pending;
  for (size_t i = 0; i < callers; ++i) pending.push_back(reader->ReadBatchAsync(refs).via(&pool));
  // All calls miss before any reader can be published. This also proves that
  // the cache mutex is not held while opening a file.
  const bool concurrent_opens = gate->WaitForActiveOperations(callers);
  const auto started = fs->opens.load();
  gate->Release();
  for (auto& future : pending) {
    auto result = std::move(future).get();
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
  }
  EXPECT_TRUE(concurrent_opens);
  EXPECT_EQ(started, callers);
  const auto opened = fs->opens.load();
  EXPECT_TRUE(reader->ReadBatch(refs).ok());
  EXPECT_TRUE(reader->ReadBatchAsync(refs).via(&pool).get().ok());
  EXPECT_EQ(fs->opens.load(), opened);
}

TEST_F(LobColumnReaderTest, ConcurrentCloseAndClearCacheDoNotResurrectReader) {
  const auto value = GenerateRandomString(65536);
  auto encoded = WriteTestData({value}).front();
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  gate->hold = true;
  auto pending = reader->ReadBatchAsync(refs).via(&executor_);
  const bool opening = gate->WaitForBlockedFiles();
  std::latch start(1);
  auto close_one = std::async(std::launch::async, [&] {
    start.wait();
    return reader->Close();
  });
  auto close_two = std::async(std::launch::async, [&] {
    start.wait();
    return reader->Close();
  });
  auto clearing = std::async(std::launch::async, [&] {
    start.wait();
    for (size_t i = 0; i < 32; ++i) reader->ClearCache();
  });
  start.count_down();
  const auto first_ready = close_one.wait_for(std::chrono::seconds(1));
  const auto second_ready = close_two.wait_for(std::chrono::seconds(1));
  const auto clear_ready = clearing.wait_for(std::chrono::seconds(1));
  gate->Release();
  EXPECT_TRUE(close_one.get().ok());
  EXPECT_TRUE(close_two.get().ok());
  clearing.get();
  auto result = std::move(pending).get();
  EXPECT_TRUE(opening);
  EXPECT_EQ(first_ready, std::future_status::ready);
  EXPECT_EQ(second_ready, std::future_status::ready);
  EXPECT_EQ(clear_ready, std::future_status::ready);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
  EXPECT_TRUE(reader->IsClosed());
  EXPECT_FALSE(reader->ReadBatchAsync(refs).get().ok());
  EXPECT_FALSE(reader->ReadData(refs[0].data, refs[0].size).ok());
  EXPECT_FALSE(reader->ReadBatchData(refs).ok());
  EXPECT_FALSE(reader->TakeData(DecodeLOBReference(encoded.data()).file_id, {0}).ok());
  arrow::BinaryBuilder builder;
  ASSERT_STATUS_OK(builder.Append(encoded.data(), static_cast<int32_t>(encoded.size())));
  std::shared_ptr<arrow::BinaryArray> input;
  ASSERT_STATUS_OK(builder.Finish(&input));
  EXPECT_FALSE(reader->ReadArrowArray(input).ok());
}

TEST_F(LobColumnReaderTest, ConcurrentAsyncFailureIsIsolatedToItsRequest) {
  const std::vector<std::string> values{GenerateRandomString(1024 * 1024), GenerateRandomString(1024 * 1024)};
  auto bad = WriteTestData({values[0]}).front();
  auto good = WriteTestData({values[1]}).front();
  std::vector<EncodedRef> bad_refs{{bad.data(), bad.size()}}, good_refs{{good.data(), good.size()}};
  auto gate = std::make_shared<LobReadGate>();
  auto fs = std::make_shared<GatedLobFileSystem>(gate);
  auto created = CreateLobColumnReader(fs, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  folly::CPUThreadPoolExecutor pool(2);
  ASSERT_TRUE(reader->ReadBatch(bad_refs).ok());
  ASSERT_TRUE(reader->ReadBatch(good_refs).ok());
  gate->fail_path = BuildLOBFilePath(config_.lob_base_path, DecodeLOBReference(bad.data()).file_id);
  gate->hold = true;
  auto failed = std::async(std::launch::async, [&] { return reader->ReadBatchAsync(bad_refs).via(&pool).get(); });
  auto successful = reader->ReadBatchAsync(good_refs).via(&pool);
  const bool reading = gate->WaitForBlockedFiles();
  const bool failed_io = gate->WaitForFailure();
  const auto failed_ready = failed.wait_for(std::chrono::seconds(1));
  const bool good_pending = !successful.isReady();
  gate->Release();
  auto error = failed.get();
  auto result = std::move(successful).get();
  EXPECT_TRUE(reading);
  EXPECT_TRUE(failed_io);
  EXPECT_EQ(failed_ready, std::future_status::ready);
  EXPECT_TRUE(good_pending);
  ASSERT_FALSE(error.ok());
  EXPECT_TRUE(error.status().IsIOError()) << error.status();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{values[1]});
  gate->fail_path.clear();
  auto recovered = reader->ReadBatchAsync(bad_refs).via(&pool).get();
  ASSERT_TRUE(recovered.ok()) << recovered.status();
  EXPECT_EQ(recovered.ValueOrDie(), std::vector<std::string>{values[0]});
}

TEST_F(LobColumnReaderTest, AsyncOpenFailureDoesNotPoisonReader) {
  const auto value = GenerateRandomString(65536);
  auto encoded = WriteTestData({value}).front();
  std::array<uint8_t, UUID_BINARY_SIZE> missing_id{};
  auto missing = EncodeLOBReference(missing_id.data(), 0);
  std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}, {missing.data(), missing.size()}};
  auto created = CreateLobColumnReader(fs_, config_);
  ASSERT_TRUE(created.ok());
  auto reader = std::move(created).ValueOrDie();
  EXPECT_FALSE(reader->ReadBatchAsync(refs).via(&executor_).get().ok());
  refs.pop_back();
  auto result = reader->ReadBatchAsync(refs).via(&executor_).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.ValueOrDie(), std::vector<std::string>{value});
}

// ==================== ReadText Tests ====================

// test reading single inline text
TEST_F(LobColumnReaderTest, ReadSingleInlineText) {
  std::string text = "short";
  auto refs = WriteTestData({text});
  ASSERT_EQ(refs.size(), 1);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), text);

  ASSERT_STATUS_OK(reader->Close());
}

// test reading single LOB text
TEST_F(LobColumnReaderTest, ReadSingleLOBText) {
  std::string text = GenerateRandomString(100);  // larger than inline threshold
  auto refs = WriteTestData({text});
  ASSERT_EQ(refs.size(), 1);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), text);

  ASSERT_STATUS_OK(reader->Close());
}

// test reading with invalid reference (null pointer)
TEST_F(LobColumnReaderTest, ReadTextInvalidNullPointer) {
  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(nullptr, 0);
  ASSERT_FALSE(read_result.ok());

  ASSERT_STATUS_OK(reader->Close());
}

// test reading with invalid LOB reference size
TEST_F(LobColumnReaderTest, ReadTextInvalidLOBReferenceSize) {
  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // create a fake LOB reference with wrong size
  std::vector<uint8_t> fake_ref(20, 0);
  fake_ref[0] = FLAG_LOB_REFERENCE;  // set LOB flag

  auto read_result = reader->ReadText(fake_ref.data(), fake_ref.size());
  ASSERT_FALSE(read_result.ok());  // should fail due to wrong size

  ASSERT_STATUS_OK(reader->Close());
}

// test reading after close
TEST_F(LobColumnReaderTest, ReadTextAfterClose) {
  std::string text = "test";
  auto refs = WriteTestData({text});

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());
  ASSERT_TRUE(reader->IsClosed());

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_FALSE(read_result.ok());  // should fail - reader is closed
}

// ==================== ReadBatch Tests ====================

// test batch read with mixed inline and LOB
TEST_F(LobColumnReaderTest, ReadBatchMixed) {
  std::vector<std::string> texts = {
      "short1",                   // inline
      GenerateRandomString(50),   // LOB
      "short2",                   // inline
      GenerateRandomString(100),  // LOB
      "short3",                   // inline
  };

  auto refs = WriteTestData(texts);
  ASSERT_EQ(refs.size(), 5);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // prepare encoded refs
  std::vector<EncodedRef> encoded_refs;
  for (const auto& ref : refs) {
    encoded_refs.push_back({ref.data(), ref.size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[i]) << "mismatch at index " << i;
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with empty refs
TEST_F(LobColumnReaderTest, ReadBatchEmpty) {
  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<EncodedRef> empty_refs;
  auto read_result = reader->ReadBatch(empty_refs);
  ASSERT_TRUE(read_result.ok());
  ASSERT_TRUE(read_result.ValueOrDie().empty());

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with null refs (empty data)
TEST_F(LobColumnReaderTest, ReadBatchWithNullRefs) {
  std::vector<std::string> texts = {"hello", GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // include null refs in the batch
  std::vector<EncodedRef> encoded_refs = {
      {refs[0].data(), refs[0].size()},
      {nullptr, 0},  // null ref
      {refs[1].data(), refs[1].size()},
  };

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), 3);
  ASSERT_EQ(read_texts[0], texts[0]);
  ASSERT_EQ(read_texts[1], "");  // null ref returns empty string
  ASSERT_EQ(read_texts[2], texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with only LOB texts (same file)
TEST_F(LobColumnReaderTest, ReadBatchAllLOBSameFile) {
  std::vector<std::string> texts;
  for (int i = 0; i < 10; i++) {
    texts.push_back(GenerateRandomString(50 + i * 10));
  }

  auto refs = WriteTestData(texts);
  ASSERT_EQ(refs.size(), 10);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<EncodedRef> encoded_refs;
  for (const auto& ref : refs) {
    encoded_refs.push_back({ref.data(), ref.size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read in reverse order
TEST_F(LobColumnReaderTest, ReadBatchReverseOrder) {
  std::vector<std::string> texts;
  for (int i = 0; i < 5; i++) {
    texts.push_back(GenerateRandomString(50));
  }

  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // reverse order
  std::vector<EncodedRef> encoded_refs;
  for (int i = refs.size() - 1; i >= 0; i--) {
    encoded_refs.push_back({refs[i].data(), refs[i].size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[texts.size() - 1 - i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Take API Tests ====================

// test Take with valid indices
TEST_F(LobColumnReaderTest, TakeValidIndices) {
  std::vector<std::string> texts;
  for (int i = 0; i < 10; i++) {
    texts.push_back(GenerateRandomString(50));
  }

  auto refs = WriteTestData(texts);

  // extract file_id from the first LOB reference
  ASSERT_TRUE(IsLOBReference(refs[0].data()));
  auto [file_id_str, row_offset] = DecodeLOBReference(refs[0].data());

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // take specific rows
  std::vector<int32_t> row_offsets = {0, 2, 5, 9};
  auto take_result = reader->Take(file_id_str, row_offsets);
  ASSERT_TRUE(take_result.ok()) << take_result.status().message();
  auto taken_texts = std::move(take_result).ValueOrDie();

  ASSERT_EQ(taken_texts.size(), 4);
  ASSERT_EQ(taken_texts[0], texts[0]);
  ASSERT_EQ(taken_texts[1], texts[2]);
  ASSERT_EQ(taken_texts[2], texts[5]);
  ASSERT_EQ(taken_texts[3], texts[9]);

  ASSERT_STATUS_OK(reader->Close());
}

// test Take with empty indices - Vortex Take API doesn't support empty indices
TEST_F(LobColumnReaderTest, TakeEmptyIndices) {
  std::vector<std::string> texts = {GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  auto [file_id_str, row_offset] = DecodeLOBReference(refs[0].data());

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // Note: Vortex Take API returns error for empty indices
  // This is expected behavior - caller should check for empty indices before calling Take
  std::vector<int32_t> empty_offsets;
  auto take_result = reader->Take(file_id_str, empty_offsets);
  ASSERT_FALSE(take_result.ok()) << "Take with empty indices should return error";

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== ReadArrowArray Tests ====================

// test ReadArrowArray with mixed data
TEST_F(LobColumnReaderTest, ReadArrowArrayMixed) {
  std::vector<std::string> texts = {
      "short1",
      GenerateRandomString(50),
      "short2",
  };

  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // build BinaryArray from refs
  arrow::BinaryBuilder builder;
  for (const auto& ref : refs) {
    ASSERT_STATUS_OK(builder.Append(ref.data(), ref.size()));
  }
  std::shared_ptr<arrow::BinaryArray> refs_array;
  ASSERT_STATUS_OK(builder.Finish(&refs_array));

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadArrowArray(refs_array);
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  auto output_array = std::move(read_result).ValueOrDie();

  ASSERT_EQ(output_array->length(), refs_array->length());
  for (int64_t i = 0; i < output_array->length(); i++) {
    ASSERT_EQ(output_array->GetString(i), texts[i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test ReadArrowArray with nulls
TEST_F(LobColumnReaderTest, ReadArrowArrayWithNulls) {
  std::vector<std::string> texts = {"hello", GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // build BinaryArray with nulls
  arrow::BinaryBuilder builder;
  ASSERT_STATUS_OK(builder.Append(refs[0].data(), refs[0].size()));
  ASSERT_STATUS_OK(builder.AppendNull());
  ASSERT_STATUS_OK(builder.Append(refs[1].data(), refs[1].size()));
  std::shared_ptr<arrow::BinaryArray> refs_array;
  ASSERT_STATUS_OK(builder.Finish(&refs_array));

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadArrowArray(refs_array);
  ASSERT_TRUE(read_result.ok());
  auto output_array = std::move(read_result).ValueOrDie();

  ASSERT_EQ(output_array->length(), 3);
  ASSERT_FALSE(output_array->IsNull(0));
  ASSERT_TRUE(output_array->IsNull(1));
  ASSERT_FALSE(output_array->IsNull(2));
  ASSERT_EQ(output_array->GetString(0), texts[0]);
  ASSERT_EQ(output_array->GetString(2), texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Cache Tests ====================

// test ClearCache
TEST_F(LobColumnReaderTest, ClearCache) {
  std::vector<std::string> texts = {GenerateRandomString(50), GenerateRandomString(60)};
  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // read to populate cache
  auto read1 = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read1.ok());

  // clear cache
  reader->ClearCache();

  // read again - should still work (re-opens file)
  auto read2 = reader->ReadText(refs[1].data(), refs[1].size());
  ASSERT_TRUE(read2.ok());
  ASSERT_EQ(read2.ValueOrDie(), texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// test multiple reads reuse cache
TEST_F(LobColumnReaderTest, MultipleReadsReuseCache) {
  std::vector<std::string> texts;
  for (int i = 0; i < 100; i++) {
    texts.push_back(GenerateRandomString(50));
  }
  auto refs = WriteTestData(texts);

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // read all texts multiple times
  for (int round = 0; round < 3; round++) {
    for (size_t i = 0; i < texts.size(); i++) {
      auto read_result = reader->ReadText(refs[i].data(), refs[i].size());
      ASSERT_TRUE(read_result.ok()) << "round " << round << ", index " << i;
      ASSERT_EQ(read_result.ValueOrDie(), texts[i]);
    }
  }

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Large Data Tests ====================

// test reading large text
TEST_F(LobColumnReaderTest, ReadLargeText) {
  // create a 1MB text
  std::string large_text = GenerateRandomString(1024 * 1024);
  auto refs = WriteTestData({large_text});

  auto manager_result = LobColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), large_text);

  ASSERT_STATUS_OK(reader->Close());
}

}  // namespace milvus_storage::lob_column
