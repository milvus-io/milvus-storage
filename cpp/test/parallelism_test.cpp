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

#include <future>
#include <chrono>

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/CancellationToken.h>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <unordered_set>

namespace milvus_storage::test {

class ParallelismTest : public ::testing::TestWithParam<std::string> {};

size_t taskFunction(int x) {
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

struct PoolHolder {
  std::unique_ptr<folly::CPUThreadPoolExecutor> folly_pool;
  std::unique_ptr<boost::asio::thread_pool> boost_pool;
};

TEST_P(ParallelismTest, ThreadPoolTest) {
  PoolHolder pool_holder;
  std::string thread_pool_type = GetParam();
  int num_of_threads = 4;
  int num_of_tasks = 10;

  // create task
  std::vector<std::packaged_task<size_t()>> tasks;
  std::vector<std::future<size_t>> futures;
  for (int i = 0; i < num_of_tasks; ++i) {
    tasks.emplace_back(std::packaged_task<size_t()>(std::bind(taskFunction, i)));
    futures.emplace_back(tasks.back().get_future());
  }

  if (thread_pool_type == "folly") {
    pool_holder.folly_pool = std::make_unique<folly::CPUThreadPoolExecutor>(num_of_threads);
    for (auto& task : tasks) {
      pool_holder.folly_pool->add(std::move(task));
    }
  } else if (thread_pool_type == "boost") {
    pool_holder.boost_pool = std::make_unique<boost::asio::thread_pool>(num_of_threads);
    for (auto& task : tasks) {
      boost::asio::post(pool_holder.boost_pool->get_executor(), std::move(task));
    }
  }
  std::unordered_set<size_t> thread_ids;
  for (auto& future : futures) {
    thread_ids.insert(future.get());
  }
  EXPECT_EQ(thread_ids.size(), num_of_threads);

  // join anyway
  if (thread_pool_type == "folly") {
    pool_holder.folly_pool->join();
  } else if (thread_pool_type == "boost") {
    pool_holder.boost_pool->join();
  }
}

INSTANTIATE_TEST_SUITE_P(ParallelismTestP, ParallelismTest, ::testing::Values("folly", "boost"));
}  // namespace milvus_storage::test
