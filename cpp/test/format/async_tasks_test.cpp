// Copyright 2023 Zilliz
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

#include <arrow/status.h>

#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <folly/futures/Promise.h>
#include <unistd.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

namespace {

void IsolateDeathTestCoverageFiles() {
  const auto prefix = "/tmp/milvus-storage-death-test-gcov-" + std::to_string(getpid());
  (void)setenv("GCOV_PREFIX", prefix.c_str(), 1);
}

arrow::Result<int> maybe_int(bool ok) {
  if (!ok) {
    return arrow::Status::Invalid("bad int");
  }
  return 41;
}

folly::SemiFuture<arrow::Result<int>> return_not_ok_future(bool ok) {
  FOLLY_ARROW_RETURN_NOT_OK(ok ? arrow::Status::OK() : arrow::Status::Invalid("bad status"));
  return folly::makeSemiFuture(arrow::Result<int>(42));
}

folly::SemiFuture<arrow::Result<int>> assign_or_raise_future(bool ok) {
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto value, maybe_int(ok));
  return folly::makeSemiFuture(arrow::Result<int>(value + 1));
}

}  // namespace

TEST(AsyncTasksTest, FollyArrowReturnNotOkPropagatesStatusToSemiFutureResult) {
  auto result = std::move(return_not_ok_future(false)).get();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("bad status") != std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaisePropagatesResultStatusToSemiFutureResult) {
  auto result = std::move(assign_or_raise_future(false)).get();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("bad int") != std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaiseExposesAssignedValueOnSuccess) {
  auto result = std::move(assign_or_raise_future(true)).get();

  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result.ValueUnsafe(), 42);
}

TEST(AsyncTasksTest, FailFastCollectorAcceptsEmptyInput) {
  std::vector<folly::SemiFuture<arrow::Result<int>>> futures;

  auto result = std::move(CollectAllResultsFailFast(std::move(futures))).get();

  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_TRUE(result.ValueUnsafe().empty());
}

TEST(AsyncTasksTest, FailFastCollectorPreservesInputOrderOnSuccess) {
  folly::Promise<arrow::Result<int>> first;
  folly::Promise<arrow::Result<int>> second;
  folly::Promise<arrow::Result<int>> third;
  std::vector<folly::SemiFuture<arrow::Result<int>>> futures;
  futures.push_back(first.getSemiFuture());
  futures.push_back(second.getSemiFuture());
  futures.push_back(third.getSemiFuture());
  auto collected = std::move(CollectAllResultsFailFast(std::move(futures))).toUnsafeFuture();

  third.setValue(30);
  first.setValue(10);
  EXPECT_FALSE(collected.isReady());
  second.setValue(20);

  ASSERT_TRUE(collected.isReady());
  auto result = std::move(collected).get();
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result.ValueUnsafe(), (std::vector<int>{10, 20, 30}));
}

TEST(AsyncTasksTest, FailFastCollectorReturnsTypedErrorBeforeBlockedSibling) {
  folly::Promise<arrow::Result<int>> blocked;
  auto lifetime = std::make_shared<int>(42);
  std::weak_ptr<int> weak_lifetime = lifetime;
  auto blocked_future = blocked.getSemiFuture().deferEnsure([lifetime] {});
  lifetime.reset();

  auto expected = MakeExtendErrorMsg(ExtendStatusCode::StorageTransientTimeout, "first reader failed");
  std::vector<folly::SemiFuture<arrow::Result<int>>> futures;
  futures.push_back(std::move(blocked_future));
  futures.push_back(folly::makeSemiFuture(arrow::Result<int>(expected)));
  auto collected = std::move(CollectAllResultsFailFast(std::move(futures))).toUnsafeFuture();

  ASSERT_TRUE(collected.isReady());
  auto result = std::move(collected).get();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().ToString(), expected.ToString());
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_FALSE(weak_lifetime.expired());

  blocked.setValue(7);
  EXPECT_TRUE(weak_lifetime.expired());
}

TEST(AsyncTasksTest, FailFastCollectorReturnsPlainUnknownForExceptionalChildBeforeBlockedSibling) {
  folly::Promise<arrow::Result<int>> blocked;
  folly::Promise<arrow::Result<int>> exceptional;
  std::vector<folly::SemiFuture<arrow::Result<int>>> futures;
  futures.push_back(blocked.getSemiFuture());
  futures.push_back(exceptional.getSemiFuture());
  auto collected = std::move(CollectAllResultsFailFast(std::move(futures), "test reader fan-in")).toUnsafeFuture();

  exceptional.setException(std::runtime_error("reader task escaped"));

  ASSERT_TRUE(collected.isReady());
  auto result = std::move(collected).get();
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsUnknownError()) << result.status().ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(result.status()), nullptr);
  EXPECT_NE(result.status().ToString().find("reader task escaped"), std::string::npos);

  blocked.setValue(7);
}

TEST(AsyncTasksTest, AllocationFailureIsFailStopBeforeWaitingForBlockedSibling) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // Set this in the parent before GoogleTest re-execs the death-test child.
  // libgcov reads GCOV_PREFIX during process startup, so setting it inside the
  // child statement is too late and makes parent/child corrupt the same .gcda.
  IsolateDeathTestCoverageFiles();
  EXPECT_DEATH_IF_SUPPORTED(
      {
        folly::Promise<arrow::Result<int>> blocked;
        std::vector<folly::SemiFuture<arrow::Result<int>>> futures;
        futures.push_back(blocked.getSemiFuture());
        futures.push_back(folly::makeSemiFuture<arrow::Result<int>>(folly::make_exception_wrapper<std::bad_alloc>()));
        (void)std::move(CollectAllResultsFailFast(std::move(futures), "test reader fan-in")).get();
      },
      "");
}

TEST(AsyncTasksTest, FailFastCollectorAllReadySuccessCannotLoseToBrokenFailurePromise) {
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::vector<folly::SemiFuture<arrow::Result<int>>> futures;
    futures.push_back(folly::makeSemiFuture(arrow::Result<int>(iteration)));
    futures.push_back(folly::makeSemiFuture(arrow::Result<int>(iteration + 1)));

    auto result = std::move(CollectAllResultsFailFast(std::move(futures))).get();

    ASSERT_TRUE(result.ok()) << "iteration=" << iteration << ": " << result.status().ToString();
    EXPECT_EQ(result.ValueUnsafe(), (std::vector<int>{iteration, iteration + 1}));
  }
}

TEST(AsyncTasksTest, ChunkTaskBuildGroupsByFileAndMergesContiguousRanges) {
  std::vector<ChunkInfo> chunk_infos = {
      {.file_index = 0, .row_offset_in_file = 0, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 10, .number_of_rows = 10},
      {.file_index = 1, .row_offset_in_file = 0, .number_of_rows = 7},
      {.file_index = 0, .row_offset_in_file = 30, .number_of_rows = 5},
  };
  auto get_chunk_info = [&chunk_infos](int64_t chunk_index) -> const ChunkInfo& { return chunk_infos[chunk_index]; };

  auto tasks = ChunkTask::Build({0, 1, 2, 3}, get_chunk_info);

  ASSERT_EQ(tasks.size(), 3);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[0].range_start, 0);
  EXPECT_EQ(tasks[0].range_end, 20);

  EXPECT_EQ(tasks[1].file_index, 0);
  EXPECT_EQ(tasks[1].chunk_indices, (std::vector<int64_t>{3}));
  EXPECT_EQ(tasks[1].range_start, 30);
  EXPECT_EQ(tasks[1].range_end, 35);

  EXPECT_EQ(tasks[2].file_index, 1);
  EXPECT_EQ(tasks[2].chunk_indices, (std::vector<int64_t>{2}));
  EXPECT_EQ(tasks[2].range_start, 0);
  EXPECT_EQ(tasks[2].range_end, 7);
}

TEST(AsyncTasksTest, TakeTaskBuildGroupsRowsByColumnGroupAndFile) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {
      {.path = "file0", .start_index = 10, .end_index = 13},
      {.path = "file1", .start_index = 20, .end_index = 25},
  };

  auto result = TakeTask::Build({group}, {0, 2, 3, 6});
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  const auto& tasks = result.ValueUnsafe();
  ASSERT_EQ(tasks.size(), 2);

  EXPECT_EQ(tasks[0].reader_index, 0);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{0, 1}));

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 1);
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{3, 6}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{2, 3}));
}

TEST(AsyncTasksTest, TakeTaskBuildRejectsInvalidRows) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {
      {.path = "file0", .start_index = 0, .end_index = 2},
  };

  EXPECT_FALSE(TakeTask::Build({group}, {-1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {1, 1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {2}).ok());
}

TEST(AsyncTasksTest, TakeTaskBuildAddsReaderIndex) {
  auto group0 = std::make_shared<ColumnGroup>();
  group0->files = {
      {.path = "group0-file0", .start_index = 0, .end_index = 2},
      {.path = "group0-file1", .start_index = 0, .end_index = 2},
  };

  auto group1 = std::make_shared<ColumnGroup>();
  group1->files = {
      {.path = "group1-file0", .start_index = 0, .end_index = 4},
  };

  auto result = TakeTask::Build({group0, group1}, {0, 2, 3});
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  const auto& tasks = result.ValueUnsafe();
  ASSERT_EQ(tasks.size(), 3);

  EXPECT_EQ(tasks[0].reader_index, 0);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{0}));

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 1);
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{1, 2}));

  EXPECT_EQ(tasks[2].reader_index, 1);
  EXPECT_EQ(tasks[2].file_index, 0);
  EXPECT_EQ(tasks[2].row_indices, (std::vector<int64_t>{0, 2, 3}));
  EXPECT_EQ(tasks[2].original_positions, (std::vector<size_t>{0, 1, 2}));
}

TEST(AsyncTasksTest, ChunkTaskNestedSplitTraitsSplitsChunkRange) {
  std::vector<ChunkInfo> chunk_infos = {
      {.file_index = 0, .row_offset_in_file = 0, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 10, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 20, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 30, .number_of_rows = 10},
  };
  auto get_chunk_info = [&chunk_infos](int64_t chunk_index) -> const ChunkInfo& { return chunk_infos[chunk_index]; };
  ChunkTask task{
      .file_index = 0,
      .chunk_indices = {0, 1, 2, 3},
      .range_start = 0,
      .range_end = 40,
  };

  ChunkTask::SplitTraits traits{get_chunk_info};
  auto right = traits.split(task);

  EXPECT_EQ(task.chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(task.range_start, 0);
  EXPECT_EQ(task.range_end, 20);
  EXPECT_EQ(right.chunk_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(right.range_start, 20);
  EXPECT_EQ(right.range_end, 40);
}

TEST(AsyncTasksTest, TakeTaskNestedSplitTraitsSplitsRowsAndPositionsTogether) {
  TakeTask task{
      .reader_index = 3,
      .file_index = 5,
      .row_indices = {0, 1, 2, 3},
      .original_positions = {10, 11, 12, 13},
  };

  TakeTask::SplitTraits traits;
  EXPECT_EQ(traits.size(task), 4);
  EXPECT_TRUE(traits.can_split(task));
  auto right = traits.split(task);

  EXPECT_EQ(task.reader_index, 3);
  EXPECT_EQ(task.file_index, 5);
  EXPECT_EQ(task.row_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(task.original_positions, (std::vector<size_t>{10, 11}));

  EXPECT_EQ(right.reader_index, 3);
  EXPECT_EQ(right.file_index, 5);
  EXPECT_EQ(right.row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(right.original_positions, (std::vector<size_t>{12, 13}));
}

TEST(AsyncTasksTest, SplitToParallelismPreservesLargestFirstOrder) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 0,
      .file_index = 0,
      .row_indices = {0, 1, 2, 3, 4, 5, 6, 7},
      .original_positions = {10, 11, 12, 13, 14, 15, 16, 17},
  }};

  SplitAsyncTasks(tasks, 3, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitToParallelism);

  ASSERT_EQ(tasks.size(), 3);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{4, 5, 6, 7}));
  EXPECT_EQ(tasks[2].row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{10, 11}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{14, 15, 16, 17}));
  EXPECT_EQ(tasks[2].original_positions, (std::vector<size_t>{12, 13}));
}

TEST(AsyncTasksTest, SplitAllCreatesOneTaskPerLogicalItem) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 3,
      .file_index = 5,
      .row_indices = {0, 1, 2, 3, 4, 5, 6, 7},
      .original_positions = {10, 11, 12, 13, 14, 15, 16, 17},
  }};

  SplitAsyncTasks(tasks, 1, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitAll);

  ASSERT_EQ(tasks.size(), 8);
  std::vector<bool> seen(8, false);
  for (const auto& task : tasks) {
    ASSERT_EQ(task.row_indices.size(), 1);
    ASSERT_EQ(task.original_positions.size(), 1);
    ASSERT_GE(task.row_indices[0], 0);
    ASSERT_LT(task.row_indices[0], static_cast<int64_t>(seen.size()));
    EXPECT_EQ(task.original_positions[0], static_cast<size_t>(task.row_indices[0] + 10));
    EXPECT_FALSE(seen[task.row_indices[0]]);
    seen[task.row_indices[0]] = true;
  }
  for (bool item_seen : seen) {
    EXPECT_TRUE(item_seen);
  }
}

}  // namespace milvus_storage::test
