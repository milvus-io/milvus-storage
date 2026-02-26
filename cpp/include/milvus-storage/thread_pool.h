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

#include <memory>
#include <mutex>
#include <thread>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/executors/ThreadPoolExecutor.h>

namespace milvus_storage {

class ThreadPoolHolder {
  public:
  ~ThreadPoolHolder() = default;

  // create or update the thread pool
  //
  // If the thread pool already exists, update the number of threads
  // Otherwise, create a new thread pool
  static void WithSingleton(size_t num_threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_pool_) {
      if (thread_pool_->numThreads() != num_threads) {
        thread_pool_->setNumThreads(num_threads);
      }
      return;
    }

    thread_pool_ = std::make_shared<folly::IOThreadPoolExecutor>(num_threads);
  }

  // get the thread pool
  //
  // If the thread pool does not exist, create a new thread pool
  // Otherwise, return the existing thread pool(with registered in `WithSingleton`)
  static std::shared_ptr<folly::ThreadPoolExecutor> GetThreadPool(size_t parallelism_hint) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_pool_) {
      return std::make_shared<folly::IOThreadPoolExecutor>(parallelism_hint);
    }
    return thread_pool_;
  }

  // get the parallelism degree
  //
  // Returns the thread pool size if a singleton exists, otherwise returns 1.
  static size_t GetParallelism() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_pool_) {
      return 1;
    }
    return thread_pool_->numThreads();
  }

  // release the thread pool
  //
  // If current thread pool still have active thread
  // then current function will be wait util all thread join
  static void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_pool_) {
      return;
    }
    thread_pool_->stop();
    thread_pool_ = nullptr;
  }

  private:
  ThreadPoolHolder() = default;
  ThreadPoolHolder(const ThreadPoolHolder&) = delete;
  ThreadPoolHolder& operator=(const ThreadPoolHolder&) = delete;

  static inline std::mutex mutex_{};
  static inline std::shared_ptr<folly::ThreadPoolExecutor> thread_pool_{nullptr};
};

}  // namespace milvus_storage