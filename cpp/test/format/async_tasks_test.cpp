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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

namespace {

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

folly::SemiFuture<arrow::Status> return_not_ok_status_future(bool ok) {
  FOLLY_ARROW_RETURN_NOT_OK(ok ? arrow::Status::OK() : arrow::Status::Invalid("bad status future"));
  return folly::makeSemiFuture(arrow::Status::OK());
}

folly::SemiFuture<arrow::Status> assign_or_raise_status_future(bool ok) {
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto value, maybe_int(ok));
  return folly::makeSemiFuture(value == 41 ? arrow::Status::OK() : arrow::Status::UnknownError("bad value"));
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

TEST(AsyncTasksTest, FollyArrowReturnNotOkPropagatesStatusToStatusSemiFuture) {
  auto status = std::move(return_not_ok_status_future(false)).get();

  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.ToString().find("bad status future"), std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaisePropagatesResultStatusToStatusSemiFuture) {
  auto status = std::move(assign_or_raise_status_future(false)).get();

  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.ToString().find("bad int"), std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaiseSupportsSuccessfulStatusSemiFuture) {
  auto status = std::move(assign_or_raise_status_future(true)).get();

  EXPECT_TRUE(status.ok()) << status.ToString();
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
      {.path = "empty", .start_index = 5, .end_index = 5},
      {.path = "file1", .start_index = 20, .end_index = 25},
  };

  auto result = TakeTask::Build({group}, {0, 2, 3, 6});
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  const auto& tasks = result.ValueUnsafe();
  ASSERT_EQ(tasks.size(), 2);

  EXPECT_EQ(tasks[0].reader_index, 0);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].file_offsets, (std::vector<int64_t>{10, 12}));
  EXPECT_EQ(tasks[0].first_position, 0);

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 2);
  EXPECT_EQ(tasks[1].file_offsets, (std::vector<int64_t>{20, 23}));
  EXPECT_EQ(tasks[1].first_position, 2);
}

TEST(AsyncTasksTest, TakeTaskBuildMatchesExpandedFileRanges) {
  auto group = std::make_shared<ColumnGroup>();
  std::vector<std::pair<size_t, int64_t>> reference;
  for (size_t file = 0; file < 32; ++file) {
    const int64_t start = 100 - file;
    const int64_t end = start + file % 5;
    group->files.push_back({.start_index = start, .end_index = end});
    for (int64_t offset = start; offset < end; ++offset) {
      reference.emplace_back(file, offset);
    }
  }

  for (size_t stride : {1, 3, 7}) {
    std::vector<int64_t> rows;
    for (size_t row = 0; row < reference.size(); row += stride) {
      rows.push_back(row);
    }
    for (size_t batch_size : {0, 4}) {
      auto tasks = TakeTask::Build({group}, rows, batch_size);
      ASSERT_TRUE(tasks.ok()) << tasks.status();
      size_t position = 0;
      for (const auto& task : *tasks) {
        EXPECT_EQ(task.first_position, position);
        for (size_t i = 0; i < task.file_offsets.size(); ++i, ++position) {
          ASSERT_LT(position, rows.size());
          const auto& [file, offset] = reference[rows[position]];
          EXPECT_EQ(task.file_index, file);
          EXPECT_EQ(task.file_offsets[i], offset);
        }
      }
      EXPECT_EQ(position, rows.size());
    }
  }
}

TEST(AsyncTasksTest, TakeTaskSplitKeepsFileIdentityWhenOffsetRangesOverlap) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {{.path = "a", .start_index = 10, .end_index = 13}, {.path = "b", .start_index = 10, .end_index = 13}};
  auto result = TakeTask::Build({group}, {3, 5});
  ASSERT_TRUE(result.ok());
  auto& left = result->front();
  auto right = TakeTask::SplitTraits{}.split(left);
  EXPECT_TRUE(left.ValidateFile(*group).ok());
  EXPECT_TRUE(right.ValidateFile(*group).ok());
  EXPECT_EQ(right.file_offsets, (std::vector<int64_t>{12}));
  EXPECT_EQ(right.first_position, 1);
  right.file_index = 0;
  EXPECT_FALSE(right.ValidateFile(*group).ok());
  left.file_offsets[0] = 13;
  EXPECT_FALSE(left.ValidateFile(*group).ok());
}

TEST(AsyncTasksTest, TakeTaskBuildBalancesRowsAcrossFileBoundaries) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {{.start_index = 10, .end_index = 15}, {.start_index = 5, .end_index = 6}};
  auto result = TakeTask::Build({group}, {0, 1, 2, 3, 4, 5}, 2);
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 4);
  EXPECT_EQ((*result)[0].file_offsets, (std::vector<int64_t>{10, 11}));
  EXPECT_EQ((*result)[1].file_offsets, (std::vector<int64_t>{12, 13}));
  EXPECT_EQ((*result)[2].file_offsets, (std::vector<int64_t>{14}));
  EXPECT_EQ((*result)[3].file_offsets, (std::vector<int64_t>{5}));
  EXPECT_EQ((*result)[2].first_position / 2, (*result)[3].first_position / 2);
}

TEST(AsyncTasksTest, TakeTaskBuildRejectsInvalidRows) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {
      {.path = "file0", .start_index = 0, .end_index = 2},
  };

  EXPECT_FALSE(TakeTask::Build({group}, {-1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {1, 1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {1, 0}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {2}).ok());
}

TEST(AsyncTasksTest, TakeTaskBuildRejectsInvalidFileRanges) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {{.start_index = 4, .end_index = 2}, {.start_index = 0, .end_index = 3}};
  EXPECT_FALSE(TakeTask::Build({group}, {0}).ok());

  const auto max = std::numeric_limits<int64_t>::max();
  group->files = {{.start_index = 0, .end_index = max}, {.start_index = 0, .end_index = 1}};
  EXPECT_TRUE(TakeTask::Build({group}, {max - 1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {max}).ok());
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
  EXPECT_EQ(tasks[0].file_offsets, (std::vector<int64_t>{0}));
  EXPECT_EQ(tasks[0].first_position, 0);

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 1);
  EXPECT_EQ(tasks[1].file_offsets, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].first_position, 1);

  EXPECT_EQ(tasks[2].reader_index, 1);
  EXPECT_EQ(tasks[2].file_index, 0);
  EXPECT_EQ(tasks[2].file_offsets, (std::vector<int64_t>{0, 2, 3}));
  EXPECT_EQ(tasks[2].first_position, 0);
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
      .file_offsets = {0, 1, 2, 3},
      .first_position = 10,
  };

  TakeTask::SplitTraits traits;
  EXPECT_EQ(traits.size(task), 4);
  EXPECT_TRUE(traits.can_split(task));
  auto right = traits.split(task);

  EXPECT_EQ(task.reader_index, 3);
  EXPECT_EQ(task.file_index, 5);
  EXPECT_EQ(task.file_offsets, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(task.first_position, 10);

  EXPECT_EQ(right.reader_index, 3);
  EXPECT_EQ(right.file_index, 5);
  EXPECT_EQ(right.file_offsets, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(right.first_position, 12);
}

TEST(AsyncTasksTest, SplitToParallelismPreservesLargestFirstOrder) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 0,
      .file_index = 0,
      .file_offsets = {0, 1, 2, 3, 4, 5, 6, 7},
      .first_position = 10,
  }};

  SplitAsyncTasks(tasks, 3, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitToParallelism);

  ASSERT_EQ(tasks.size(), 3);
  EXPECT_EQ(tasks[0].file_offsets, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].file_offsets, (std::vector<int64_t>{4, 5, 6, 7}));
  EXPECT_EQ(tasks[2].file_offsets, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tasks[0].first_position, 10);
  EXPECT_EQ(tasks[1].first_position, 14);
  EXPECT_EQ(tasks[2].first_position, 12);
}

TEST(AsyncTasksTest, SplitAllCreatesOneTaskPerLogicalItem) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 3,
      .file_index = 5,
      .file_offsets = {0, 1, 2, 3, 4, 5, 6, 7},
      .first_position = 10,
  }};

  SplitAsyncTasks(tasks, 1, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitAll);

  ASSERT_EQ(tasks.size(), 8);
  std::vector<bool> seen(8, false);
  for (const auto& task : tasks) {
    ASSERT_EQ(task.file_offsets.size(), 1);
    ASSERT_GE(task.file_offsets[0], 0);
    ASSERT_LT(task.file_offsets[0], static_cast<int64_t>(seen.size()));
    EXPECT_EQ(task.first_position, static_cast<size_t>(task.file_offsets[0] + 10));
    EXPECT_FALSE(seen[task.file_offsets[0]]);
    seen[task.file_offsets[0]] = true;
  }
  for (bool item_seen : seen) {
    EXPECT_TRUE(item_seen);
  }
}

TEST(AsyncTasksTest, SplitByBytesPacksWholeChunksWithinNaturalRanges) {
  std::vector<ChunkInfo> infos = {
      {.file_index = 0, .row_offset_in_file = 0, .number_of_rows = 10, .read_size = 3},
      {.file_index = 0, .row_offset_in_file = 10, .number_of_rows = 10, .read_size = 5},
      {.file_index = 0, .row_offset_in_file = 20, .number_of_rows = 10, .read_size = 9},
      {.file_index = 0, .row_offset_in_file = 30, .number_of_rows = 10, .read_size = 2},
      {.file_index = 0, .row_offset_in_file = 40, .number_of_rows = 10, .read_size = 4},
      {.file_index = 0, .row_offset_in_file = 60, .number_of_rows = 10, .read_size = 1},
      {.file_index = 1, .row_offset_in_file = 0, .number_of_rows = 10, .read_size = 1},
  };
  auto lookup = [&infos](int64_t i) -> const ChunkInfo& { return infos[i]; };
  auto tasks = ChunkTask::Build({0, 1, 2, 3, 4, 5, 6}, lookup);
  SplitAsyncTasks(tasks, 1, ChunkTask::SplitTraits{lookup}, AsyncTaskSplitStrategy::kSplitByBytes, 8);
  ASSERT_EQ(tasks.size(), 5);
  EXPECT_EQ(tasks[0].chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[0].range_start, 0);
  EXPECT_EQ(tasks[0].range_end, 20);
  EXPECT_EQ(tasks[1].chunk_indices, (std::vector<int64_t>{2}));
  EXPECT_EQ(tasks[1].range_start, 20);
  EXPECT_EQ(tasks[1].range_end, 30);
  EXPECT_EQ(tasks[2].chunk_indices, (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(tasks[2].range_start, 30);
  EXPECT_EQ(tasks[2].range_end, 50);
  EXPECT_EQ(tasks[3].chunk_indices, (std::vector<int64_t>{5}));
  EXPECT_EQ(tasks[3].range_start, 60);
  EXPECT_EQ(tasks[4].file_index, 1);
}

TEST(AsyncTasksTest, SplitByBytesFallsBackWhenStoredSizesAreUnavailable) {
  std::vector<ChunkInfo> infos = {
      {.row_offset_in_file = 0, .number_of_rows = 10, .read_size = 4},
      {.row_offset_in_file = 10, .number_of_rows = 10},
      {.row_offset_in_file = 20, .number_of_rows = 10, .read_size = 4},
      {.row_offset_in_file = 30, .number_of_rows = 10, .read_size = 4},
  };
  auto lookup = [&infos](int64_t i) -> const ChunkInfo& { return infos[i]; };
  auto tasks = ChunkTask::Build({0, 1, 2, 3}, lookup);
  SplitAsyncTasks(tasks, 2, ChunkTask::SplitTraits{lookup}, AsyncTaskSplitStrategy::kSplitByBytes, 1);
  ASSERT_EQ(tasks.size(), 2);
  EXPECT_EQ(tasks[0].chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].chunk_indices, (std::vector<int64_t>{2, 3}));
}

TEST(AsyncTasksTest, SplitByBytesHandlesZeroAndMaximumStoredSizes) {
  std::vector<ChunkInfo> infos = {
      {.row_offset_in_file = 0, .number_of_rows = 1, .read_size = 0},
      {.row_offset_in_file = 1, .number_of_rows = 1, .read_size = UINT64_MAX},
      {.row_offset_in_file = 2, .number_of_rows = 1, .read_size = 1},
  };
  auto lookup = [&infos](int64_t i) -> const ChunkInfo& { return infos[i]; };
  auto tasks = ChunkTask::Build({0, 1, 2}, lookup);
  SplitAsyncTasks(tasks, 1, ChunkTask::SplitTraits{lookup}, AsyncTaskSplitStrategy::kSplitByBytes, UINT64_MAX);
  ASSERT_EQ(tasks.size(), 2);
  EXPECT_EQ(tasks[0].chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].chunk_indices, (std::vector<int64_t>{2}));
  tasks.clear();
  SplitAsyncTasks(tasks, 1, ChunkTask::SplitTraits{lookup}, AsyncTaskSplitStrategy::kSplitByBytes, 8);
  EXPECT_TRUE(tasks.empty());
}

TEST(AsyncTasksTest, SplitByBytesUsesParallelismForTakeTasks) {
  std::vector<TakeTask> tasks = {
      {.reader_index = 0, .file_index = 0, .file_offsets = {0, 1, 2, 3}, .first_position = 0}};
  SplitAsyncTasks(tasks, 2, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitByBytes, 1);
  ASSERT_EQ(tasks.size(), 2);
  EXPECT_EQ(tasks[0].file_offsets, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].first_position, 2);
}

TEST(AsyncTasksTest, BytesPropertiesKeepParallelismDefaultAndRejectZeroSize) {
  Properties properties;
  ASSERT_TRUE(GetValue<std::string>(std::make_shared<const api::Properties>(properties),
                                    PROPERTY_READER_ASYNC_TASK_SPLIT_STRATEGY)
                  .ok());
  EXPECT_EQ(GetAsyncTaskSplitStrategy(std::make_shared<const api::Properties>(properties)),
            AsyncTaskSplitStrategy::kSplitToParallelism);
  EXPECT_EQ(GetValueNoError<uint64_t>(std::make_shared<const api::Properties>(properties),
                                      PROPERTY_READER_ASYNC_TASK_SPLIT_SIZE_BYTES),
            8388608);
  EXPECT_EQ(SetValue(properties, PROPERTY_READER_ASYNC_TASK_SPLIT_STRATEGY, "bytes"), std::nullopt);
  EXPECT_EQ(GetAsyncTaskSplitStrategy(std::make_shared<const api::Properties>(properties)),
            AsyncTaskSplitStrategy::kSplitByBytes);
  EXPECT_NE(SetValue(properties, PROPERTY_READER_ASYNC_TASK_SPLIT_SIZE_BYTES, "0"), std::nullopt);
}

}  // namespace milvus_storage::test
