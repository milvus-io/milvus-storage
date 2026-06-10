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

#include "milvus-storage/format/async_tasks.h"

#include <array>
#include <concepts>
#include <map>
#include <queue>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::api {

ChunkTask::SplitTraits::SplitTraits(std::function<const ChunkInfo&(int64_t)> get_chunk_info)
    : get_chunk_info_(std::move(get_chunk_info)) {}

size_t ChunkTask::SplitTraits::size(const ChunkTask& task) const { return task.chunk_indices.size(); }

bool ChunkTask::SplitTraits::can_split(const ChunkTask& task) const { return size(task) > 1; }

ChunkTask ChunkTask::SplitTraits::split(ChunkTask& left) const {
  size_t mid = left.chunk_indices.size() / 2;
  // The first chunk in the right half defines both new half-open range boundaries.
  const auto& mid_info = get_chunk_info_(left.chunk_indices[mid]);

  ChunkTask right;
  right.file_index = left.file_index;
  right.chunk_indices.assign(left.chunk_indices.begin() + mid, left.chunk_indices.end());
  right.range_start = mid_info.row_offset_in_file;
  right.range_end = left.range_end;

  left.chunk_indices.resize(mid);
  left.range_end = mid_info.row_offset_in_file;
  return right;
}

std::vector<ChunkTask> ChunkTask::Build(const std::vector<int64_t>& chunk_indices,
                                        const std::function<const ChunkInfo&(int64_t)>& get_chunk_info) {
  // Build file-local natural tasks; only physically adjacent chunks are merged
  // into the same range read.
  std::map<size_t, std::vector<int64_t>> file_groups;
  for (auto idx : chunk_indices) {
    file_groups[get_chunk_info(idx).file_index].push_back(idx);
  }

  std::vector<ChunkTask> tasks;
  for (auto& [file_idx, chunks] : file_groups) {
    if (chunks.empty()) {
      continue;
    }

    const auto& first_info = get_chunk_info(chunks[0]);
    ChunkTask current;
    current.file_index = file_idx;
    current.chunk_indices.push_back(chunks[0]);
    current.range_start = first_info.row_offset_in_file;
    current.range_end = first_info.row_offset_in_file + first_info.number_of_rows;

    for (size_t i = 1; i < chunks.size(); ++i) {
      const auto& prev = get_chunk_info(chunks[i - 1]);
      const auto& curr = get_chunk_info(chunks[i]);

      if (curr.row_offset_in_file == prev.row_offset_in_file + prev.number_of_rows) {
        current.chunk_indices.push_back(chunks[i]);
        current.range_end = curr.row_offset_in_file + curr.number_of_rows;
      } else {
        tasks.push_back(std::move(current));
        current = ChunkTask{
            .file_index = file_idx,
            .chunk_indices = {chunks[i]},
            .range_start = curr.row_offset_in_file,
            .range_end = curr.row_offset_in_file + curr.number_of_rows,
        };
      }
    }
    tasks.push_back(std::move(current));
  }
  return tasks;
}

size_t TakeTask::SplitTraits::size(const TakeTask& task) const { return task.row_indices.size(); }

bool TakeTask::SplitTraits::can_split(const TakeTask& task) const { return size(task) > 1; }

TakeTask TakeTask::SplitTraits::split(TakeTask& left) const {
  size_t mid = left.row_indices.size() / 2;
  // Positions must be split with rows so fan-in can still restore input order.
  TakeTask right{
      left.reader_index,
      left.file_index,
      {left.row_indices.begin() + mid, left.row_indices.end()},
      {left.original_positions.begin() + mid, left.original_positions.end()},
  };
  left.row_indices.resize(mid);
  left.original_positions.resize(mid);
  return right;
}

arrow::Result<std::vector<TakeTask>> TakeTask::Build(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                     const std::vector<int64_t>& row_indices) {
  for (size_t i = 0; i < row_indices.size(); ++i) {
    if (row_indices[i] < 0) {
      return arrow::Status::Invalid(fmt::format("Row index is less than 0, [row_index={}]", row_indices[i]));
    }
    if (i > 0 && row_indices[i] <= row_indices[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Input row indices is not sorted or not unique,[index={}, row_index={}]", i, row_indices[i]));
    }
  }

  std::vector<TakeTask> all_tasks;
  // Plan every required column group into per-file tasks while carrying the
  // original row positions needed after asynchronous fan-in.
  for (size_t reader_index = 0; reader_index < column_groups.size(); ++reader_index) {
    if (!column_groups[reader_index]) {
      return arrow::Status::Invalid(
          fmt::format("Failed to build take task, column group at index {} is empty", reader_index));
    }

    const auto& files = column_groups[reader_index]->files;
    std::map<uint32_t, std::vector<std::pair<int64_t, size_t>>> file_groups;
    // Resolve each column-group logical row against manifest-visible file lengths.
    // Keep the logical value in the task; execution converts it to file-local.
    for (size_t pos = 0; pos < row_indices.size(); ++pos) {
      int64_t row_index_remain = row_indices[pos];
      bool found_file = false;
      for (uint32_t file_index = 0; file_index < files.size(); ++file_index) {
        if (files[file_index].start_index < 0 || files[file_index].end_index < 0) {
          return arrow::Status::Invalid(
              fmt::format("Invalid start/end index in [file_index={}, path={}]", file_index, files[file_index].path));
        }

        int64_t num_of_rows_in_file = files[file_index].end_index - files[file_index].start_index;
        if (row_index_remain < num_of_rows_in_file) {
          file_groups[file_index].push_back({row_indices[pos], pos});
          found_file = true;
          break;
        }

        row_index_remain -= num_of_rows_in_file;
      }
      if (!found_file) {
        return arrow::Status::Invalid(
            fmt::format("Row index is greater than the maximum range, [row_index={}]", row_indices[pos]));
      }
    }

    all_tasks.reserve(all_tasks.size() + file_groups.size());
    for (auto& [file_index, rows_and_positions] : file_groups) {
      TakeTask task;
      task.reader_index = reader_index;
      task.file_index = file_index;
      task.row_indices.reserve(rows_and_positions.size());
      task.original_positions.reserve(rows_and_positions.size());
      for (auto& [row, pos] : rows_and_positions) {
        task.row_indices.push_back(row);
        task.original_positions.push_back(pos);
      }
      all_tasks.push_back(std::move(task));
    }
  }
  return all_tasks;
}

template <typename Traits, typename Task>
concept AsyncTaskSplitTraits = requires(const Traits& traits, const Task& task, Task& mutable_task) {
  { traits.size(task) } -> std::convertible_to<size_t>;
  { traits.can_split(task) } -> std::convertible_to<bool>;
  { traits.split(mutable_task) } -> std::same_as<Task>;
};

// Max-heap of splittable task indices, ordered by the split weight reported by
// Traits. Indices keep heap entries stable when the task vector reallocates.
// After splitting, the mutated left task and appended right task are inserted
// again only if they remain splittable.
template <typename Task, typename Traits>
class SplittableTaskMaxHeap {
  public:
  SplittableTaskMaxHeap(std::vector<Task>& tasks, const Traits& traits) : tasks_(tasks), traits_(traits) {
    for (size_t index = 0; index < tasks_.size(); ++index) {
      push_if_splittable(index);
    }
  }

  // Split the largest candidate in place and append its right half. Return
  // false when no task can be split further.
  bool split_largest() {
    if (heap_.empty()) {
      return false;
    }

    auto candidate = heap_.top();
    heap_.pop();

    auto right = traits_.split(tasks_[candidate.index]);
    const auto right_index = tasks_.size();
    tasks_.push_back(std::move(right));

    push_if_splittable(candidate.index);
    push_if_splittable(right_index);
    return true;
  }

  private:
  struct Candidate {
    size_t size;
    size_t index;
  };

  struct CompareCandidate {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
      if (lhs.size != rhs.size) {
        return lhs.size < rhs.size;
      }
      // Match the previous linear scan: when sizes tie, split the task that
      // appears first in the task vector.
      return lhs.index > rhs.index;
    }
  };

  void push_if_splittable(size_t index) {
    if (traits_.can_split(tasks_[index])) {
      heap_.push(Candidate{traits_.size(tasks_[index]), index});
    }
  }

  std::vector<Task>& tasks_;
  const Traits& traits_;
  std::priority_queue<Candidate, std::vector<Candidate>, CompareCandidate> heap_;
};

template <typename Task, typename Traits>
class NoSplitPolicy {
  public:
  static void split(std::vector<Task>&, size_t, const Traits&) {}
};

template <typename Task, typename Traits>
class SplitToParallelismPolicy {
  public:
  static void split(std::vector<Task>& tasks, size_t target_count, const Traits& traits) {
    // Repeatedly bisect the largest task. Natural tasks are never merged when
    // their initial count already exceeds the requested target.
    if (tasks.size() >= target_count) {
      return;
    }

    SplittableTaskMaxHeap<Task, Traits> task_heap(tasks, traits);
    while (tasks.size() < target_count) {
      if (!task_heap.split_largest()) {
        return;
      }
    }
  }
};

template <typename Task, typename Traits>
class SplitAllPolicy {
  public:
  static void split(std::vector<Task>& tasks, size_t, const Traits& traits) {
    // A max-heap avoids rescanning the growing task vector after every split.
    // Splitting N logical items now takes O(N log N) planner work.
    SplittableTaskMaxHeap<Task, Traits> task_heap(tasks, traits);
    while (task_heap.split_largest()) {
    }
  }
};

template <typename Task, typename Traits>
void SplitAsyncTasks(std::vector<Task>& tasks,
                     size_t target_count,
                     const Traits& traits,
                     AsyncTaskSplitStrategy strategy) {
  static_assert(AsyncTaskSplitTraits<Traits, Task>);

  using SplitFn = void (*)(std::vector<Task>&, size_t, const Traits&);
  static constexpr std::array<SplitFn, 3> kSplitFns = {
      &NoSplitPolicy<Task, Traits>::split,
      &SplitToParallelismPolicy<Task, Traits>::split,
      &SplitAllPolicy<Task, Traits>::split,
  };

  auto index = static_cast<size_t>(strategy);
  if (index >= kSplitFns.size()) {
    index = static_cast<size_t>(AsyncTaskSplitStrategy::kSplitToParallelism);
  }
  kSplitFns[index](tasks, target_count, traits);
}

AsyncTaskSplitStrategy GetAsyncTaskSplitStrategy(const Properties& properties) {
  auto strategy = GetValueNoError<std::string>(properties, PROPERTY_READER_ASYNC_TASK_SPLIT_STRATEGY);
  if (strategy == "none") {
    return AsyncTaskSplitStrategy::kNoSplit;
  }
  if (strategy == "all") {
    return AsyncTaskSplitStrategy::kSplitAll;
  }
  // Missing and unknown values intentionally select the balanced default.
  return AsyncTaskSplitStrategy::kSplitToParallelism;
}

template void SplitAsyncTasks<ChunkTask, ChunkTask::SplitTraits>(std::vector<ChunkTask>& tasks,
                                                                 size_t target_count,
                                                                 const ChunkTask::SplitTraits& traits,
                                                                 AsyncTaskSplitStrategy strategy);

template void SplitAsyncTasks<TakeTask, TakeTask::SplitTraits>(std::vector<TakeTask>& tasks,
                                                               size_t target_count,
                                                               const TakeTask::SplitTraits& traits,
                                                               AsyncTaskSplitStrategy strategy);

}  // namespace milvus_storage::api
