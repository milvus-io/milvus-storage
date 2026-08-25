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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/properties.h"

namespace milvus_storage::api {

struct ColumnGroup;
struct ChunkInfo;

enum class AsyncTaskSplitStrategy : size_t {
  kNoSplit = 0,
  kSplitToParallelism = 1,
  kSplitAll = 2,
};

// Collect Arrow Result futures in input order. The returned future completes
// as soon as any child reports an error; already-started children remain owned
// by the drain branch and may finish in the background. Exceptional child
// futures are classified explicitly at this library boundary.
template <typename T>
folly::SemiFuture<arrow::Result<std::vector<T>>> CollectAllResultsFailFast(
    std::vector<folly::SemiFuture<arrow::Result<T>>> futures,
    std::string_view operation = "asynchronous reader fan-in") {
  using ItemResult = arrow::Result<T>;
  using OutputResult = arrow::Result<std::vector<T>>;

  if (futures.empty()) {
    return folly::makeSemiFuture(OutputResult(std::vector<T>{}));
  }

  struct FirstFailure {
    explicit FirstFailure(std::string operation) : operation(std::move(operation)) {}

    std::atomic<bool> fulfilled{false};
    folly::Promise<OutputResult> promise;
    std::string operation;

    // Once a child has failed, formatting and publishing that failure are part
    // of the fail-fast contract. They must not throw back into Folly before the
    // promise is fulfilled: Folly would store that exception on the observed
    // child and a blocked sibling could then keep both fan-in branches pending
    // forever. A real allocation failure in either operation is therefore
    // fail-stop rather than recoverable.
    arrow::Status from_exception(const folly::exception_wrapper& exception) const noexcept {
      return arrow::Status::UnknownError(operation,
                                         " child future raised a C++ exception: ", exception.what().toStdString());
    }

    void set(const arrow::Status& status) noexcept {
      if (!fulfilled.exchange(true, std::memory_order_relaxed)) {
        promise.setValue(OutputResult(status));
      }
    }
  };

  auto first_failure = std::make_shared<FirstFailure>(std::string(operation));
  auto first_failure_future = first_failure->promise.getSemiFuture();

  std::vector<folly::SemiFuture<ItemResult>> observed;
  observed.reserve(futures.size());
  for (auto& future : futures) {
    observed.push_back(std::move(future).defer([first_failure](folly::Try<ItemResult>&& child_try) -> ItemResult {
      if (child_try.hasException()) {
        // Keep our own exception handle. Publishing first_failure may run the
        // downstream collectAny continuation inline and release the observed
        // child's future state re-entrantly.
        auto exception = child_try.exception();
        if (exception.template is_compatible_with<std::bad_alloc>()) {
          // OOM cannot safely allocate either an Arrow status or the promise
          // state needed to publish one. Fail-stop is preferable to returning
          // into Folly with first_failure unresolved while another child can
          // remain pending forever.
          std::terminate();
        }

        ItemResult child_result(first_failure->from_exception(exception));
        first_failure->set(child_result.status());
        return child_result;
      }

      ItemResult child_result = std::move(child_try.value());
      if (!child_result.ok()) {
        first_failure->set(child_result.status());
      }
      return child_result;
    }));
  }

  auto all_success =
      folly::collectAll(std::move(observed))
          .deferValue([first_failure](std::vector<folly::Try<ItemResult>>&& child_tries) -> OutputResult {
            std::vector<T> values;
            values.reserve(child_tries.size());
            for (auto& child_try : child_tries) {
              if (child_try.hasException()) {
                if (child_try.exception().template is_compatible_with<std::bad_alloc>()) {
                  std::terminate();
                }
                return first_failure->from_exception(child_try.exception());
              }
              auto child_result = std::move(child_try.value());
              if (!child_result.ok()) {
                return child_result.status();
              }
              values.push_back(std::move(child_result).ValueUnsafe());
            }
            return values;
          });

  std::vector<folly::SemiFuture<OutputResult>> completion_futures;
  completion_futures.reserve(2);
  completion_futures.push_back(std::move(first_failure_future));
  completion_futures.push_back(std::move(all_success));
  return folly::collectAny(std::move(completion_futures))
      .deferValue([first_failure](std::pair<size_t, folly::Try<OutputResult>>&& first) -> OutputResult {
        if (first.second.hasException()) {
          if (first.second.exception().template is_compatible_with<std::bad_alloc>()) {
            std::terminate();
          }
          return first_failure->from_exception(first.second.exception());
        }
        return std::move(first.second.value());
      });
}

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
