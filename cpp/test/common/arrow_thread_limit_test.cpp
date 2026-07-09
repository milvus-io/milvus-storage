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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <arrow/io/interfaces.h>
#include <arrow/util/thread_pool.h>

#include "milvus-storage/thread_pool.h"
#include "test_env.h"

namespace milvus_storage::test {

namespace {

constexpr auto kProbeTimeout = std::chrono::seconds(5);
constexpr auto kSaturationHoldTime = std::chrono::milliseconds(200);

struct RuntimeProbeState {
  std::atomic<int> running{0};
  std::atomic<int> peak_running{0};
  std::atomic<int> started{0};
  std::mutex mutex;
  std::condition_variable cv;
  bool release = false;
};

void UpdatePeak(std::atomic<int>& peak, int value) {
  int current = peak.load();
  while (value > current && !peak.compare_exchange_weak(current, value)) {
  }
}

void ReleaseProbeTasks(const std::shared_ptr<RuntimeProbeState>& state) {
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->release = true;
  }
  state->cv.notify_all();
}

void RunBlockingProbeTask(const std::shared_ptr<RuntimeProbeState>& state) {
  auto running = state->running.fetch_add(1) + 1;
  UpdatePeak(state->peak_running, running);
  state->started.fetch_add(1);
  state->cv.notify_all();

  std::unique_lock<std::mutex> lock(state->mutex);
  state->cv.wait(lock, [&state] { return state->release; });

  state->running.fetch_sub(1);
}

arrow::Status SubmitBlockingProbeTasks(arrow::internal::Executor* executor,
                                       const std::shared_ptr<RuntimeProbeState>& state,
                                       int num_tasks,
                                       std::vector<arrow::Future<>>* futures) {
  futures->reserve(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    auto maybe_future = executor->Submit([state] { RunBlockingProbeTask(state); });
    if (!maybe_future.ok()) {
      return maybe_future.status();
    }
    futures->push_back(std::move(maybe_future).ValueOrDie());
  }
  return arrow::Status::OK();
}

arrow::Status WaitUntilExecutorIsSaturated(const std::shared_ptr<RuntimeProbeState>& state, int configured_threads) {
  std::unique_lock<std::mutex> lock(state->mutex);
  if (!state->cv.wait_for(lock, kProbeTimeout,
                          [&state, configured_threads] { return state->started.load() >= configured_threads; })) {
    return arrow::Status::Invalid("runtime did not start configured number of tasks");
  }
  return arrow::Status::OK();
}

arrow::Status WaitForProbeTasks(std::vector<arrow::Future<>>* futures) {
  for (auto& future : *futures) {
    if (!future.Wait(kProbeTimeout.count())) {
      return arrow::Status::Invalid("runtime task did not finish");
    }
    auto status = arrow::FutureToSync(future);
    if (!status.ok()) {
      return status;
    }
  }
  return arrow::Status::OK();
}

arrow::Status MeasureExecutorPeakParallelism(arrow::internal::Executor* executor,
                                             int configured_threads,
                                             int num_tasks,
                                             int* peak_running) {
  auto state = std::make_shared<RuntimeProbeState>();
  std::vector<arrow::Future<>> futures;

  auto status = SubmitBlockingProbeTasks(executor, state, num_tasks, &futures);
  if (!status.ok()) {
    ReleaseProbeTasks(state);
    auto wait_status = WaitForProbeTasks(&futures);
    return wait_status.ok() ? status : wait_status;
  }

  status = WaitUntilExecutorIsSaturated(state, configured_threads);
  if (!status.ok()) {
    ReleaseProbeTasks(state);
    auto wait_status = WaitForProbeTasks(&futures);
    return wait_status.ok() ? status : wait_status;
  }

  std::this_thread::sleep_for(kSaturationHoldTime);

  ReleaseProbeTasks(state);
  status = WaitForProbeTasks(&futures);
  if (!status.ok()) {
    return status;
  }

  *peak_running = state->peak_running.load();
  return arrow::Status::OK();
}

class StorageRuntimeTest : public ::testing::Test {
  protected:
  void SetUp() override {
    cpu_threads_ = arrow::GetCpuThreadPoolCapacity();
    io_threads_ = arrow::io::GetIOThreadPoolCapacity();
  }

  void TearDown() override {
    ASSERT_STATUS_OK(arrow::SetCpuThreadPoolCapacity(cpu_threads_));
    ASSERT_STATUS_OK(arrow::io::SetIOThreadPoolCapacity(io_threads_));
  }

  int cpu_threads_;
  int io_threads_;
};

}  // namespace

TEST_F(StorageRuntimeTest, ConfigureStorageRuntimeLimitsArrowRuntimeParallelism) {
  constexpr int kCpuThreads = 2;
  constexpr int kIoThreads = 3;
  constexpr int kTasksPerThread = 32;

  ASSERT_STATUS_OK(ConfigureStorageRuntime(kCpuThreads, kIoThreads));

  int cpu_peak_running = 0;
  ASSERT_STATUS_OK(MeasureExecutorPeakParallelism(arrow::internal::GetCpuThreadPool(), kCpuThreads,
                                                  kCpuThreads * kTasksPerThread, &cpu_peak_running));
  std::cout << "cpu_peak_running=" << cpu_peak_running << std::endl;
  EXPECT_LE(cpu_peak_running, kCpuThreads);

  int io_peak_running = 0;
  ASSERT_STATUS_OK(MeasureExecutorPeakParallelism(arrow::io::default_io_context().executor(), kIoThreads,
                                                  kIoThreads * kTasksPerThread, &io_peak_running));
  std::cout << "io_peak_running=" << io_peak_running << std::endl;
  EXPECT_LE(io_peak_running, kIoThreads);
}

}  // namespace milvus_storage::test
