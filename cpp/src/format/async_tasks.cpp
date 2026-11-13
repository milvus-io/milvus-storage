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

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
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

bool ChunkTask::SplitTraits::split_by_bytes(std::vector<ChunkTask>& tasks, uint64_t target_bytes) const {
  if (target_bytes == 0) {
    return false;
  }
  for (const auto& task : tasks) {
    for (auto index : task.chunk_indices) {
      if (!get_chunk_info_(index).read_size.has_value()) {
        return false;
      }
    }
  }

  std::vector<ChunkTask> result;
  for (const auto& task : tasks) {
    uint64_t bytes = 0;
    ChunkTask current{.file_index = task.file_index, .range_start = task.range_start};
    for (auto index : task.chunk_indices) {
      const auto& info = get_chunk_info_(index);
      const auto size = *info.read_size;
      if (!current.chunk_indices.empty() && (bytes >= target_bytes || size > target_bytes - bytes)) {
        result.push_back(std::move(current));
        current = ChunkTask{.file_index = task.file_index, .range_start = info.row_offset_in_file};
        bytes = 0;
      }
      current.chunk_indices.push_back(index);
      current.range_end = info.row_offset_in_file + info.number_of_rows;
      bytes += size;
    }
    if (!current.chunk_indices.empty()) {
      result.push_back(std::move(current));
    }
  }
  tasks = std::move(result);
  return true;
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

size_t TakeTask::SplitTraits::size(const TakeTask& task) const { return task.file_offsets.size(); }

bool TakeTask::SplitTraits::can_split(const TakeTask& task) const { return size(task) > 1; }

TakeTask TakeTask::SplitTraits::split(TakeTask& left) const {
  size_t mid = left.file_offsets.size() / 2;
  TakeTask right{
      left.reader_index,         left.file_index,   {left.file_offsets.begin() + mid, left.file_offsets.end()},
      left.first_position + mid, left.planned_file,
  };
  left.file_offsets.resize(mid);
  return right;
}

arrow::Result<std::vector<TakeTask>> TakeTask::Build(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                     const std::vector<int64_t>& row_indices,
                                                     size_t rows_per_batch) {
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
    size_t pos = 0;
    int64_t logical_end = 0;
    for (size_t file_index = 0; file_index < files.size() && pos < row_indices.size(); ++file_index) {
      const auto& file = files[file_index];
      if (file.start_index < 0 || file.end_index < file.start_index) {
        return arrow::Status::Invalid(
            fmt::format("Invalid start/end index in [file_index={}, path={}]", file_index, file.path));
      }
      const int64_t length = file.end_index - file.start_index;
      if (length > std::numeric_limits<int64_t>::max() - logical_end) {
        return arrow::Status::Invalid("Column group logical row range overflows at file ", file_index);
      }
      const int64_t logical_start = logical_end;
      logical_end += length;
      const size_t first = pos;
      while (pos < row_indices.size() && row_indices[pos] < logical_end) {
        ++pos;
      }
      if (pos == first) {
        continue;
      }

      if (file_index > std::numeric_limits<uint32_t>::max()) {
        return arrow::Status::Invalid("TakeTask file index out of range: ", file_index);
      }
      for (size_t begin = first; begin < pos;) {
        size_t count = pos - begin;
        if (rows_per_batch != 0) {
          count = std::min(count, rows_per_batch - begin % rows_per_batch);
        }
        auto& task = all_tasks.emplace_back();
        task.reader_index = reader_index;
        task.file_index = static_cast<uint32_t>(file_index);
        task.planned_file = std::shared_ptr<const ColumnGroupFile>(column_groups[reader_index], &file);
        task.file_offsets.reserve(count);
        for (size_t i = begin; i < begin + count; ++i) {
          task.file_offsets.push_back(file.start_index + (row_indices[i] - logical_start));
        }
        task.first_position = begin;
        begin += count;
      }
    }
    if (pos < row_indices.size()) {
      return arrow::Status::Invalid(
          fmt::format("Row index is greater than the maximum range, [row_index={}]", row_indices[pos]));
    }
  }
  return all_tasks;
}

arrow::Status TakeTask::ValidateFile(const ColumnGroup& group) const {
  if (file_index >= group.files.size() || planned_file.get() != &group.files[file_index]) {
    return arrow::Status::Invalid("TakeTask does not belong to the requested file: ", file_index);
  }
  for (const int64_t offset : file_offsets) {
    if (offset < planned_file->start_index || offset >= planned_file->end_index) {
      return arrow::Status::Invalid("TakeTask offset outside file range: ", offset);
    }
  }
  return arrow::Status::OK();
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
                     AsyncTaskSplitStrategy strategy,
                     uint64_t target_bytes) {
  static_assert(AsyncTaskSplitTraits<Traits, Task>);

  if (strategy == AsyncTaskSplitStrategy::kSplitByBytes) {
    if constexpr (requires { traits.split_by_bytes(tasks, target_bytes); }) {
      if (traits.split_by_bytes(tasks, target_bytes)) {
        return;
      }
    }
    SplitToParallelismPolicy<Task, Traits>::split(tasks, target_count, traits);
    return;
  }

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

AsyncTaskSplitStrategy GetAsyncTaskSplitStrategy(const std::shared_ptr<const Properties>& properties) {
  auto strategy = GetValueNoError<std::string>(properties, PROPERTY_READER_ASYNC_TASK_SPLIT_STRATEGY);
  if (strategy == "none") {
    return AsyncTaskSplitStrategy::kNoSplit;
  }
  if (strategy == "bytes") {
    return AsyncTaskSplitStrategy::kSplitByBytes;
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
                                                                 AsyncTaskSplitStrategy strategy,
                                                                 uint64_t target_bytes);

template void SplitAsyncTasks<TakeTask, TakeTask::SplitTraits>(std::vector<TakeTask>& tasks,
                                                               size_t target_count,
                                                               const TakeTask::SplitTraits& traits,
                                                               AsyncTaskSplitStrategy strategy,
                                                               uint64_t target_bytes);

}  // namespace milvus_storage::api
