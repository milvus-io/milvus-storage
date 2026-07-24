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

#include "milvus-storage/format/parquet/folly_arrow_executor.h"

#include <exception>
#include <utility>

#include <arrow/status.h>

namespace milvus_storage::parquet {
namespace {

// Arrow's generator expects its own Executor interface. This adapter forwards
// tasks to the Folly executor selected by the consuming future chain.
class FollyArrowExecutor final : public arrow::internal::Executor {
  public:
  FollyArrowExecutor(folly::Executor::KeepAlive<> executor, int capacity)
      : executor_(std::move(executor)), capacity_(capacity) {}

  // Capacity is an Arrow scheduling hint; it does not create additional workers.
  int GetCapacity() override { return capacity_; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()> task,
                          arrow::StopToken stop_token,
                          StopCallback&& stop_callback) override {
    // Honor cancellation both before enqueue and after the task reaches the
    // Folly executor, since cancellation may arrive while it is queued.
    if (stop_token.IsStopRequested()) {
      if (stop_callback) {
        std::move(stop_callback)(stop_token.Poll());
      }
      return arrow::Status::OK();
    }

    try {
      executor_->add([task = std::move(task), stop_token = std::move(stop_token),
                      stop_callback = std::move(stop_callback)]() mutable {
        if (!stop_token.IsStopRequested()) {
          std::move(task)();
        } else if (stop_callback) {
          std::move(stop_callback)(stop_token.Poll());
        }
      });
    } catch (const std::exception& e) {
      return arrow::Status::UnknownError("Failed to submit task to Folly executor: ", e.what());
    } catch (...) {
      return arrow::Status::UnknownError("Failed to submit task to Folly executor");
    }

    return arrow::Status::OK();
  }

  private:
  // Keep the caller-owned executor alive for every outstanding Arrow task.
  folly::Executor::KeepAlive<> executor_;
  int capacity_;
};

}  // namespace

std::shared_ptr<arrow::internal::Executor> MakeFollyArrowExecutor(folly::Executor::KeepAlive<> executor, int capacity) {
  return std::make_shared<FollyArrowExecutor>(std::move(executor), capacity);
}

}  // namespace milvus_storage::parquet
