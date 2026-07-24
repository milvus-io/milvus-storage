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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <arrow/result.h>

#include "milvus-storage/properties.h"

namespace milvus_storage::api {

struct ColumnGroup;
struct ChunkInfo;

enum class AsyncTaskSplitStrategy : size_t {
  kNoSplit = 0,
  kSplitToParallelism = 1,
  kSplitAll = 2,
};

// Resolve reader.async.task_split_strategy; unknown values use the default
// split-to-parallelism policy.
AsyncTaskSplitStrategy GetAsyncTaskSplitStrategy(const Properties& properties);

// One natural chunk task covers a contiguous half-open row range in one file.
// Splitting preserves the file and range-continuity invariant.
struct ChunkTask {
  size_t file_index;
  std::vector<int64_t> chunk_indices;
  uint64_t range_start;
  uint64_t range_end;

  class SplitTraits {
 public:
    // Bind the chunk metadata lookup used to place splits on chunk boundaries.
    explicit SplitTraits(std::function<const ChunkInfo&(int64_t)> get_chunk_info);

    // Return the number of logical chunks used as this task's split weight.
    [[nodiscard]] size_t size(const ChunkTask& task) const;
    // A chunk task is splittable only when each half can retain at least one chunk.
    [[nodiscard]] bool can_split(const ChunkTask& task) const;
    // Shrink left to its first half and return the contiguous right half.
    [[nodiscard]] ChunkTask split(ChunkTask& left) const;

 private:
    std::function<const ChunkInfo&(int64_t)> get_chunk_info_;
  };

  // Build one natural task per contiguous chunk range in each file. Callers
  // provide sorted, unique, validated chunk indices.
  static std::vector<ChunkTask> Build(const std::vector<int64_t>& chunk_indices,
                                      const std::function<const ChunkInfo&(int64_t)>& get_chunk_info);
};

// One take task belongs to one column-group reader and one file. Row indices
// remain global until execution; original_positions restores input order.
struct TakeTask {
  size_t reader_index;
  uint32_t file_index;
  std::vector<int64_t> row_indices;
  std::vector<size_t> original_positions;

  class SplitTraits {
 public:
    // Return the number of requested rows used as this task's split weight.
    [[nodiscard]] size_t size(const TakeTask& task) const;
    // A take task is splittable only when each half can retain at least one row.
    [[nodiscard]] bool can_split(const TakeTask& task) const;
    // Split rows and original_positions at the same boundary; mutate left and return right.
    [[nodiscard]] TakeTask split(TakeTask& left) const;
  };

  // Validate sorted unique logical row indices, then build one natural task for
  // every (column-group reader, file) pair touched by the request.
  static arrow::Result<std::vector<TakeTask>> Build(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                    const std::vector<int64_t>& row_indices);
};

// Split natural tasks toward target_count according to strategy. target_count
// controls task granularity, not an executor or concurrency limit; tasks are never merged.
template <typename Task, typename Traits>
void SplitAsyncTasks(std::vector<Task>& tasks,
                     size_t target_count,
                     const Traits& traits,
                     AsyncTaskSplitStrategy strategy);

}  // namespace milvus_storage::api
