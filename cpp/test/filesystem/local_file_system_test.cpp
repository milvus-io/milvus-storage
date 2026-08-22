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

#include <arrow/testing/gtest_util.h>
#include <arrow/testing/executor_util.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <unistd.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/filesystem/upload_sizable.h"
#include "milvus-storage/filesystem/util_internal.h"

#include "test_env.h"

namespace milvus_storage {
namespace {

void PrepareDeathTest() {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto prefix = "/tmp/milvus-storage-submit-io-death-test-gcov-" + std::to_string(getpid());
  (void)setenv("GCOV_PREFIX", prefix.c_str(), 1);
}

class RejectingExecutor final : public arrow::internal::Executor {
  public:
  int GetCapacity() override { return 1; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()>,
                          arrow::StopToken,
                          StopCallback&&) override {
    return arrow::Status::IOError("executor rejected background task");
  }
};

class ThrowingExecutor final : public arrow::internal::Executor {
  public:
  int GetCapacity() override { return 1; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()>,
                          arrow::StopToken,
                          StopCallback&&) override {
    throw std::runtime_error("executor submission exploded");
  }
};

class BadAllocThrowingExecutor final : public arrow::internal::Executor {
  public:
  int GetCapacity() override { return 1; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()>,
                          arrow::StopToken,
                          StopCallback&&) override {
    throw std::bad_alloc();
  }
};

class CancellingExecutor final : public arrow::internal::Executor {
  public:
  int GetCapacity() override { return 1; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()>,
                          arrow::StopToken,
                          StopCallback&& stop_callback) override {
    std::move(stop_callback)(arrow::Status::Cancelled("background I/O cancelled before execution"));
    return arrow::Status::OK();
  }
};

class BadAllocCopyCompletion {
  public:
  BadAllocCopyCompletion(int* completion_count, arrow::Status* completed_status)
      : completion_count_(completion_count), completed_status_(completed_status) {}

  BadAllocCopyCompletion(const BadAllocCopyCompletion&) { throw std::bad_alloc(); }
  BadAllocCopyCompletion(BadAllocCopyCompletion&&) noexcept = default;

  void operator()(const arrow::Status& status) {
    ++*completion_count_;
    *completed_status_ = status;
  }

  private:
  int* completion_count_;
  arrow::Status* completed_status_;
};

class BadAllocMoveTask {
  public:
  BadAllocMoveTask() = default;
  BadAllocMoveTask(const BadAllocMoveTask&) = delete;
  BadAllocMoveTask(BadAllocMoveTask&&) { throw std::bad_alloc(); }

  arrow::Status operator()() { return arrow::Status::OK(); }
};

class FailOnceOutputStream final : public arrow::io::OutputStream {
  public:
  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override { return position_; }

  bool closed() const override { return closed_; }

  arrow::Status Write(const void*, int64_t nbytes) override {
    if (fail_next_) {
      fail_next_ = false;
      return arrow::Status::IOError("injected output failure");
    }
    position_ += nbytes;
    return arrow::Status::OK();
  }

  arrow::Status Flush() override { return arrow::Status::OK(); }

  private:
  bool fail_next_ = true;
  bool closed_ = false;
  int64_t position_ = 0;
};

}  // namespace

TEST(SubmitIOWithCompletionTest, ExecutorRejectionStillCompletesExactlyOnce) {
  RejectingExecutor executor;
  arrow::io::IOContext io_context(&executor);
  int completion_count = 0;
  arrow::Status completed_status;

  auto status = arrow::io::internal::SubmitIOWithCompletion(
      io_context, [] { return arrow::Status::OK(); },
      [&](const arrow::Status& task_status) {
        ++completion_count;
        completed_status = task_status;
      });

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(completion_count, 1);
  EXPECT_TRUE(completed_status.Equals(status));
}

TEST(SubmitIOWithCompletionTest, ExecutorExceptionCompletesExactlyOnceWithInvariantDetail) {
  ThrowingExecutor executor;
  arrow::io::IOContext io_context(&executor);
  int completion_count = 0;
  arrow::Status completed_status;

  auto status = arrow::io::internal::SubmitIOWithCompletion(
      io_context, [] { return arrow::Status::OK(); },
      [&](const arrow::Status& task_status) {
        ++completion_count;
        completed_status = task_status;
      });

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(completion_count, 1);
  EXPECT_TRUE(completed_status.Equals(status));
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::InternalInvariantViolated);
  EXPECT_NE(status.message().find("executor submission exploded"), std::string::npos);
}

TEST(SubmitIOWithCompletionTest, ExecutorAllocationFailureIsFailStop) {
  PrepareDeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        BadAllocThrowingExecutor executor;
        arrow::io::IOContext io_context(&executor);
        (void)arrow::io::internal::SubmitIOWithCompletion(
            io_context, [] { return arrow::Status::OK(); }, [](const arrow::Status&) {});
      },
      "");
}

TEST(SubmitIOWithCompletionTest, CompletionConstructionAllocationFailureIsFailStop) {
  PrepareDeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        RejectingExecutor executor;
        arrow::io::IOContext io_context(&executor);
        int completion_count = 0;
        arrow::Status completed_status;
        BadAllocCopyCompletion completion(&completion_count, &completed_status);
        (void)arrow::io::internal::SubmitIOWithCompletion(io_context, [] { return arrow::Status::OK(); }, completion);
      },
      "");
}

TEST(SubmitIOWithCompletionTest, TaskConstructionAllocationFailureIsFailStop) {
  PrepareDeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        RejectingExecutor executor;
        arrow::io::IOContext io_context(&executor);
        BadAllocMoveTask task;
        (void)arrow::io::internal::SubmitIOWithCompletion(io_context, std::move(task), [](const arrow::Status&) {});
      },
      "");
}

TEST(SubmitIOWithCompletionTest, CancellationBeforeExecutionCompletesWithoutRunningTask) {
  CancellingExecutor executor;
  arrow::io::IOContext io_context(&executor);
  bool task_ran = false;
  int completion_count = 0;
  arrow::Status completed_status;

  ASSERT_STATUS_OK(arrow::io::internal::SubmitIOWithCompletion(
      io_context,
      [&] {
        task_ran = true;
        return arrow::Status::OK();
      },
      [&](const arrow::Status& task_status) {
        ++completion_count;
        completed_status = task_status;
      }));

  EXPECT_FALSE(task_ran);
  EXPECT_EQ(completion_count, 1);
  EXPECT_TRUE(completed_status.IsCancelled()) << completed_status.ToString();
}

TEST(SubmitIOWithCompletionTest, TaskFailureStillCompletesExactlyOnce) {
  std::promise<arrow::Status> completion;
  auto completed = completion.get_future();
  auto expected = arrow::Status::IOError("client acquisition failed");

  ASSERT_STATUS_OK(arrow::io::internal::SubmitIOWithCompletion(
      arrow::io::default_io_context(), [expected] { return expected; },
      [&completion](const arrow::Status& status) { completion.set_value(status); }));

  ASSERT_EQ(completed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(completed.get().Equals(expected));
}

TEST(SubmitIOWithCompletionTest, StandardTaskExceptionCompletesExactlyOnceWithInvariantDetail) {
  arrow::MockExecutor executor;
  arrow::io::IOContext io_context(&executor);
  int completion_count = 0;
  arrow::Status completed_status;

  ASSERT_STATUS_OK(arrow::io::internal::SubmitIOWithCompletion(
      io_context, []() -> arrow::Status { throw std::runtime_error("background task exploded"); },
      [&](const arrow::Status& task_status) {
        ++completion_count;
        completed_status = task_status;
      }));

  EXPECT_EQ(completion_count, 1);
  auto detail = ExtendStatusDetail::UnwrapStatus(completed_status);
  ASSERT_NE(detail, nullptr) << completed_status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::InternalInvariantViolated);
  EXPECT_NE(completed_status.message().find("background task exploded"), std::string::npos);
}

TEST(SubmitIOWithCompletionTest, TaskAllocationFailureIsFailStop) {
  PrepareDeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        arrow::MockExecutor executor;
        arrow::io::IOContext io_context(&executor);
        (void)arrow::io::internal::SubmitIOWithCompletion(
            io_context, []() -> arrow::Status { throw std::bad_alloc(); }, [](const arrow::Status&) {});
      },
      "");
}

TEST(SubmitIOWithCompletionTest, CompletionExceptionIsFailStop) {
  PrepareDeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        RejectingExecutor executor;
        arrow::io::IOContext io_context(&executor);
        (void)arrow::io::internal::SubmitIOWithCompletion(
            io_context, [] { return arrow::Status::OK(); },
            [](const arrow::Status&) { throw std::runtime_error("completion exploded"); });
      },
      "");
}

TEST(SubmitIOWithCompletionTest, NonStandardTaskExceptionCompletesExactlyOnceWithInvariantDetail) {
  arrow::MockExecutor executor;
  arrow::io::IOContext io_context(&executor);
  int completion_count = 0;
  arrow::Status completed_status;

  ASSERT_STATUS_OK(arrow::io::internal::SubmitIOWithCompletion(
      io_context, []() -> arrow::Status { throw 42; },
      [&](const arrow::Status& task_status) {
        ++completion_count;
        completed_status = task_status;
      }));

  EXPECT_EQ(completion_count, 1);
  auto detail = ExtendStatusDetail::UnwrapStatus(completed_status);
  ASSERT_NE(detail, nullptr) << completed_status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::InternalInvariantViolated);
  EXPECT_NE(completed_status.message().find("non-standard exception"), std::string::npos);
}

TEST(WriterStatusTest, ObservedAsyncFailureRemainsTheTerminalVerdict) {
  WriterStatus writer_status;
  auto first_child = MakeExtendError(ExtendStatusCode::StorageAccessDenied, "part 1 denied");
  auto later = MakeExtendError(ExtendStatusCode::StorageTransientService, "later failure");

  writer_status.ObserveFailure(first_child);

  EXPECT_TRUE(writer_status.Check().Equals(first_child));
  EXPECT_TRUE(writer_status.Fail(later).Equals(first_child));
  EXPECT_TRUE(writer_status.Check().Equals(first_child));
}

TEST(WriterStatusTest, LocallySuccessfulOperationStillReturnsConcurrentFailure) {
  WriterStatus writer_status;
  auto child_failure = MakeExtendError(ExtendStatusCode::StorageTransientService, "background upload failed");

  // Models a foreground operation that passed Check(), then lost the race to
  // an asynchronous completion before publishing its local OK.
  writer_status.ObserveFailure(child_failure);
  EXPECT_TRUE(writer_status.Fail(arrow::Status::OK()).Equals(child_failure));
  EXPECT_TRUE(writer_status.Fail(arrow::Status::OK()).Equals(child_failure));
}

TEST(WriterStatusTest, BackgroundCompletionPublishesTheObservedFirstFailure) {
  WriterStatus writer_status;
  auto first_child = MakeExtendError(ExtendStatusCode::StorageAccessDenied, "part 1 denied");
  auto later = MakeExtendError(ExtendStatusCode::StorageTransientService, "later failure");

  auto first_observed = arrow::io::internal::ObserveBackgroundIOStatus(&writer_status, first_child);
  auto later_observed = arrow::io::internal::ObserveBackgroundIOStatus(&writer_status, later);
  EXPECT_TRUE(first_observed.Equals(first_child));
  EXPECT_TRUE(later_observed.Equals(first_child));

  auto future = arrow::Future<>::Make();
  future.MarkFinished(std::move(later_observed));
  ASSERT_TRUE(future.Wait(1.0));
  EXPECT_TRUE(future.status().Equals(first_child));
}

// The failure travels with the move, and what is left behind does not read as
// a healthy writer: arrow::Status's own move would otherwise hand the source
// back an OK status, so a writer that had already failed a part would answer
// ok() and accept more data.
TEST(WriterStatusTest, MovingCarriesTheFailureAndDoesNotLeaveAHealthySource) {
  static_assert(std::is_nothrow_move_constructible_v<arrow::Status>);
  static_assert(std::is_nothrow_move_assignable_v<arrow::Status>);
  static_assert(std::is_nothrow_move_constructible_v<WriterStatus>);
  static_assert(std::is_nothrow_move_assignable_v<WriterStatus>);

  WriterStatus source;
  auto failure = MakeExtendError(ExtendStatusCode::StorageAccessDenied, "part 1 denied");
  source.ObserveFailure(failure);

  WriterStatus moved(std::move(source));
  EXPECT_TRUE(moved.Check().Equals(failure));
  EXPECT_FALSE(source.ok());  // NOLINT(bugprone-use-after-move) -- that is the point

  WriterStatus assigned;
  assigned = std::move(moved);
  EXPECT_TRUE(assigned.Check().Equals(failure));
  EXPECT_FALSE(moved.ok());  // NOLINT(bugprone-use-after-move)
}

TEST(WriterStatusTest, MovedHealthyStatusStillDiscards) {
  WriterStatus source;
  WriterStatus moved(std::move(source));
  moved.BeginDiscard();
  EXPECT_TRUE(moved.Check().IsCancelled());

  WriterStatus assigned_source;
  WriterStatus assigned;
  assigned = std::move(assigned_source);
  assigned.BeginDiscard();
  EXPECT_TRUE(assigned.Check().IsCancelled());
}

TEST(MetricsOutputStreamTest, FailureIsTerminal) {
  auto underlying = std::make_shared<FailOnceOutputStream>();
  auto metrics = std::make_shared<FilesystemMetrics>();
  auto writer = std::make_shared<MetricsOutputStream>(underlying, metrics);
  const uint8_t byte = 1;

  auto first_failure = writer->Write(&byte, 1);
  ASSERT_STATUS_NOT_OK(first_failure);

  EXPECT_TRUE(writer->Flush().Equals(first_failure));

  ASSERT_STATUS_OK(writer->Abort());
}

class LocalFsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // init local filesystem properties
    api::SetValue(properties_, PROPERTY_FS_STORAGE_TYPE, "local");
    api::SetValue(properties_, PROPERTY_FS_ROOT_PATH, "/tmp/milvus-storage-test");
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("local-fs-test");
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string base_path_;
};

TEST_F(LocalFsTest, UploadSizableNotImplemented) {
  auto sizable = std::dynamic_pointer_cast<UploadSizable>(fs_);
  ASSERT_NE(sizable, nullptr);

  auto result = sizable->OpenOutputStreamWithUploadSize("local-not-implemented.txt", nullptr, 5);
  ASSERT_TRUE(result.status().IsNotImplemented());
}

TEST_F(LocalFsTest, UploadConditional) {
  auto conditional = std::dynamic_pointer_cast<UploadConditional>(fs_);
  ASSERT_NE(conditional, nullptr);

  std::string file_to = base_path_ + "/local-conditional.txt";
  // Ensure file does not exist
  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  // First write should succeed
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());
    ASSERT_AND_ASSIGN(auto output_stream, conditional->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
    // Check file exists
    ASSIGN_OR_ABORT(auto file_info, fs_->GetFileInfo(file_to));
    ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
  }

  // Second write should fail because file already exists
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());
    auto result = conditional->OpenConditionalOutputStream(file_to, nullptr);
    ASSERT_STATUS_NOT_OK(result);
    ASSERT_TRUE(result.status().IsIOError());
  }

  (void)fs_->DeleteFile(file_to);
}

TEST_F(LocalFsTest, Observable) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);

  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  metrics->Reset();

  // Write a file
  std::string file_path = base_path_ + "/test-observable.txt";
  std::string content = "This is test content for observable tracking.";
  auto content_size = static_cast<int64_t>(content.size());

  ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_path));
  ASSERT_STATUS_OK(output_stream->Write(content.c_str(), content_size));
  ASSERT_STATUS_OK(output_stream->Close());

  metrics = observable->GetMetrics();

  ASSERT_EQ(metrics->GetReadCount(), 0);
  ASSERT_EQ(metrics->GetWriteCount(), 1);
  ASSERT_EQ(metrics->GetWriteBytes(), content_size);
  ASSERT_EQ(metrics->GetReadBytes(), 0);
  ASSERT_EQ(metrics->GetGetFileInfoCount(), 0);
  ASSERT_EQ(metrics->GetCreateDirCount(), 0);
  ASSERT_EQ(metrics->GetDeleteDirCount(), 0);
  ASSERT_EQ(metrics->GetDeleteFileCount(), 0);
  ASSERT_EQ(metrics->GetMoveCount(), 0);
  ASSERT_EQ(metrics->GetCopyFileCount(), 0);
}

TEST_F(LocalFsTest, TestRootPath) {
  auto boost_rmdir = [](const std::string& path) {
    boost::filesystem::path dir_path(path);
    if (boost::filesystem::exists(dir_path)) {
      boost::filesystem::remove_all(dir_path);
    }
  };

  auto boost_create_dir = [](const std::string& path) {
    boost::filesystem::path dir_path(path);
    if (!boost::filesystem::exists(dir_path)) {
      boost::filesystem::create_directories(dir_path);
    }
  };

  std::string abs_path = "/tmp/test-localfs/";
  std::string rel_path = "./test-localfs/";

  std::string abs_exist_path = "/tmp/test-exist-localfs/";
  std::string rel_exist_path = "./test-exist-localfs/";

  boost_rmdir(abs_path);
  boost_rmdir(rel_path);

  boost_rmdir(abs_exist_path);
  boost_rmdir(rel_exist_path);
  boost_create_dir(abs_exist_path);
  boost_create_dir(rel_exist_path);

  std::vector<std::string> paths = {
      abs_path,
      rel_path,
      abs_exist_path,
      rel_exist_path,
  };

  for (const auto& root_path : paths) {
    ArrowFileSystemConfig config;
    config.storage_type = "local";
    config.root_path = root_path;
    std::string write_content = "This is a test file.";

    ASSERT_AND_ASSIGN(auto local_fs, CreateArrowFileSystem(config));
    ASSERT_AND_ASSIGN(auto output_stream, local_fs->OpenOutputStream("test.txt"));
    ASSERT_STATUS_OK(output_stream->Write(write_content.c_str(), write_content.size()));
    ASSERT_STATUS_OK(output_stream->Close());

    ASSERT_TRUE(boost::filesystem::exists(root_path));
    ASSERT_TRUE(boost::filesystem::exists(root_path + "/test.txt"));

    ASSERT_AND_ASSIGN(auto input_stream, local_fs->OpenInputStream("test.txt"));
    ASSERT_AND_ASSIGN(auto read_buffer, input_stream->Read(write_content.size()));
    auto read_content = read_buffer->ToString();
    ASSERT_STATUS_OK(input_stream->Close());
    ASSERT_EQ(write_content, read_content);

    // Cleanup
    boost_rmdir(root_path);
  }
}

TEST_F(LocalFsTest, TestMetricsAfterFileOperations) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();
  EXPECT_EQ(metrics->GetWriteCount(), 0);
  EXPECT_EQ(metrics->GetReadCount(), 0);
  EXPECT_EQ(metrics->GetWriteBytes(), 0);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
  EXPECT_EQ(metrics->GetGetFileInfoCount(), 0);
  EXPECT_EQ(metrics->GetCreateDirCount(), 0);
  EXPECT_EQ(metrics->GetDeleteFileCount(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 0);

  // Write a file
  std::string file_path = base_path_ + "/test-metrics-write.txt";
  std::string content = "This is test content for metrics tracking.";
  auto content_size = static_cast<int64_t>(content.size());

  ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_path));
  ASSERT_STATUS_OK(output_stream->Write(content.c_str(), content_size));
  ASSERT_STATUS_OK(output_stream->Close());

  EXPECT_EQ(metrics->GetWriteCount(), 1);
  EXPECT_EQ(metrics->GetWriteBytes(), content_size);
  EXPECT_EQ(metrics->GetReadCount(), 0);
  EXPECT_EQ(metrics->GetReadBytes(), 0);

  // Read the file
  ASSERT_AND_ASSIGN(auto input_stream, fs_->OpenInputStream(file_path));
  ASSERT_AND_ASSIGN(auto read_buffer, input_stream->Read(content_size));
  ASSERT_STATUS_OK(input_stream->Close());

  EXPECT_EQ(metrics->GetReadCount(), 1);
  EXPECT_EQ(metrics->GetReadBytes(), content_size);
  EXPECT_EQ(metrics->GetWriteCount(), 1);  // Should remain unchanged

  // GetFileInfo operations
  ASSIGN_OR_ABORT(auto file_info, fs_->GetFileInfo(file_path));
  EXPECT_EQ(file_info.type(), arrow::fs::FileType::File);
  EXPECT_GE(metrics->GetGetFileInfoCount(), 1);

  // Delete file
  ASSERT_STATUS_OK(fs_->DeleteFile(file_path));
  EXPECT_EQ(metrics->GetDeleteFileCount(), 1);

  // Verify file doesn't exist
  ASSIGN_OR_ABORT(auto deleted_info, fs_->GetFileInfo(file_path));
  EXPECT_EQ(deleted_info.type(), arrow::fs::FileType::NotFound);
}

TEST_F(LocalFsTest, TestMetricsForDirectoryOperations) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();
  EXPECT_EQ(metrics->GetCreateDirCount(), 0);
  EXPECT_EQ(metrics->GetDeleteDirCount(), 0);

  // Create directory
  std::string dir_path = base_path_ + "/test-dir";
  ASSERT_STATUS_OK(fs_->CreateDir(dir_path, true));
  EXPECT_EQ(metrics->GetCreateDirCount(), 1);

  // GetFileInfo on directory
  ASSIGN_OR_ABORT(auto dir_info, fs_->GetFileInfo(dir_path));
  EXPECT_EQ(dir_info.type(), arrow::fs::FileType::Directory);

  // Delete directory
  ASSERT_STATUS_OK(fs_->DeleteDir(dir_path));
  EXPECT_EQ(metrics->GetDeleteDirCount(), 1);
}

TEST_F(LocalFsTest, TestMetricsForMoveAndCopy) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();
  EXPECT_EQ(metrics->GetMoveCount(), 0);
  EXPECT_EQ(metrics->GetCopyFileCount(), 0);

  // Create source file
  std::string src_path = base_path_ + "/source.txt";
  std::string content = "Test content for move and copy.";
  ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(src_path));
  ASSERT_STATUS_OK(output_stream->Write(content.c_str(), content.size()));
  ASSERT_STATUS_OK(output_stream->Close());

  // Copy file
  std::string copy_path = base_path_ + "/copy.txt";
  ASSERT_STATUS_OK(fs_->CopyFile(src_path, copy_path));
  EXPECT_EQ(metrics->GetCopyFileCount(), 1);

  // Verify copy exists
  ASSIGN_OR_ABORT(auto copy_info, fs_->GetFileInfo(copy_path));
  EXPECT_EQ(copy_info.type(), arrow::fs::FileType::File);

  // Move file
  std::string move_path = base_path_ + "/moved.txt";
  ASSERT_STATUS_OK(fs_->Move(src_path, move_path));
  EXPECT_EQ(metrics->GetMoveCount(), 1);

  // Verify source doesn't exist and destination exists
  ASSIGN_OR_ABORT(auto src_info, fs_->GetFileInfo(src_path));
  EXPECT_EQ(src_info.type(), arrow::fs::FileType::NotFound);
  ASSIGN_OR_ABORT(auto move_info, fs_->GetFileInfo(move_path));
  EXPECT_EQ(move_info.type(), arrow::fs::FileType::File);
}

TEST_F(LocalFsTest, TestMetricsForFailedOperations) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();
  EXPECT_EQ(metrics->GetFailedCount(), 0);

  // Try to delete non-existent file
  std::string non_existent = base_path_ + "/non-existent.txt";
  auto status = fs_->DeleteFile(non_existent);
  // This might succeed (no-op) or fail depending on implementation
  // But if it fails, it should increment failed count
  if (!status.ok()) {
    EXPECT_GE(metrics->GetFailedCount(), 1);
  }

  // Try to open non-existent file for reading
  auto read_result = fs_->OpenInputStream(non_existent);
  ASSERT_STATUS_NOT_OK(read_result);
  EXPECT_GE(metrics->GetFailedCount(), 1);
}

TEST_F(LocalFsTest, TestMetricsForMultipleReadsAndWrites) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();

  // Write multiple files
  std::vector<std::string> files;
  std::string content = "Test content";
  auto content_size = static_cast<int64_t>(content.size());

  for (int i = 0; i < 5; ++i) {
    std::string file_path = base_path_ + "/file" + std::to_string(i) + ".txt";
    ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_path));
    ASSERT_STATUS_OK(output_stream->Write(content.c_str(), content_size));
    ASSERT_STATUS_OK(output_stream->Close());
    files.push_back(file_path);
  }

  EXPECT_EQ(metrics->GetWriteCount(), 5);
  EXPECT_EQ(metrics->GetWriteBytes(), 5 * content_size);

  // Read all files
  for (const auto& file_path : files) {
    ASSERT_AND_ASSIGN(auto input_stream, fs_->OpenInputStream(file_path));
    ASSERT_AND_ASSIGN(auto buffer, input_stream->Read(content_size));
    ASSERT_STATUS_OK(input_stream->Close());
  }

  EXPECT_EQ(metrics->GetReadCount(), 5);
  EXPECT_EQ(metrics->GetReadBytes(), 5 * content_size);
}

TEST_F(LocalFsTest, TestMetricsForRandomAccessFile) {
  auto observable = std::dynamic_pointer_cast<Observable>(fs_);
  ASSERT_NE(observable, nullptr);
  auto metrics = observable->GetMetrics();
  ASSERT_NE(metrics, nullptr);

  metrics->Reset();

  // Create a file
  std::string file_path = base_path_ + "/random-access.txt";
  std::string content = "This is a test file for random access reading.";
  auto content_size = static_cast<int64_t>(content.size());

  ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_path));
  ASSERT_STATUS_OK(output_stream->Write(content.c_str(), content_size));
  ASSERT_STATUS_OK(output_stream->Close());

  // Read using RandomAccessFile
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(file_path));
  ASSERT_AND_ASSIGN(auto buffer1, file->ReadAt(0, 10));
  ASSERT_AND_ASSIGN(auto buffer2, file->ReadAt(10, 10));
  ASSERT_STATUS_OK(file->Close());

  EXPECT_EQ(metrics->GetReadCount(), 2);   // Two successful ReadAt calls
  EXPECT_EQ(metrics->GetReadBytes(), 20);  // 10 + 10 bytes read
}

}  // namespace milvus_storage
