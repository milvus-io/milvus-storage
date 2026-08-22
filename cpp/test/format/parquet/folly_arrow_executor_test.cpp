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

#include <functional>
#include <new>
#include <string>
#include <thread>

#include <arrow/util/future.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Promise.h>

#include "milvus-storage/format/parquet/folly_arrow_executor.h"

namespace milvus_storage::parquet::test {
namespace {

size_t CurrentThreadId() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }

class BadAllocExecutor final : public folly::Executor {
  public:
  void add(folly::Func) override { throw std::bad_alloc(); }
};

TEST(FollyArrowExecutorTest, RoutesSubmittedTaskToFollyExecutor) {
  folly::CPUThreadPoolExecutor folly_executor(1);
  auto maybe_arrow_executor = MakeFollyArrowExecutor(folly::getKeepAliveToken(folly_executor), 1);
  ASSERT_TRUE(maybe_arrow_executor.ok()) << maybe_arrow_executor.status().ToString();
  auto arrow_executor = std::move(maybe_arrow_executor).ValueUnsafe();
  const auto caller_thread = CurrentThreadId();

  auto maybe_future = arrow_executor->Submit([] { return CurrentThreadId(); });
  ASSERT_TRUE(maybe_future.ok()) << maybe_future.status().ToString();
  auto result = maybe_future.ValueUnsafe().result();
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_NE(result.ValueUnsafe(), caller_thread);
}

TEST(FollyArrowExecutorTest, TransferAlwaysRunsContinuationOnFollyExecutor) {
  folly::CPUThreadPoolExecutor folly_executor(1);
  auto maybe_arrow_executor = MakeFollyArrowExecutor(folly::getKeepAliveToken(folly_executor), 1);
  ASSERT_TRUE(maybe_arrow_executor.ok()) << maybe_arrow_executor.status().ToString();
  auto arrow_executor = std::move(maybe_arrow_executor).ValueUnsafe();
  const auto caller_thread = CurrentThreadId();

  auto source = arrow::Future<>::Make();
  auto continuation = arrow_executor->TransferAlways(source).Then([] { return CurrentThreadId(); });
  source.MarkFinished();

  auto result = continuation.result();
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_NE(result.ValueUnsafe(), caller_thread);
}

TEST(FollyArrowExecutorTest, DeferredArrowTaskRunsOnWaitingThreadWithoutVia) {
  const auto caller_thread = CurrentThreadId();

  auto future = folly::makeSemiFuture().deferExValue(
      [](folly::Executor::KeepAlive<> executor, folly::Unit) -> folly::SemiFuture<arrow::Result<size_t>> {
        auto maybe_arrow_executor = MakeFollyArrowExecutor(std::move(executor), 1);
        if (!maybe_arrow_executor.ok()) {
          return folly::makeSemiFuture(arrow::Result<size_t>(maybe_arrow_executor.status()));
        }
        auto arrow_executor = std::move(maybe_arrow_executor).ValueUnsafe();
        auto maybe_arrow_future = arrow_executor->Submit([] { return CurrentThreadId(); });
        if (!maybe_arrow_future.ok()) {
          return folly::makeSemiFuture(arrow::Result<size_t>(maybe_arrow_future.status()));
        }

        auto promise = std::make_shared<folly::Promise<arrow::Result<size_t>>>();
        auto semi_future = promise->getSemiFuture();
        auto arrow_future = std::move(maybe_arrow_future).ValueUnsafe();
        arrow_future.AddCallback(
            [promise, arrow_executor = std::move(arrow_executor)](const arrow::Result<size_t>& result) {
              (void)arrow_executor;
              promise->setValue(result);
            });
        return semi_future;
      });

  auto result = std::move(future).get();
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result.ValueUnsafe(), caller_thread);
}

TEST(FollyArrowExecutorTest, RejectsEmptyExecutor) {
  auto result = MakeFollyArrowExecutor(folly::Executor::KeepAlive<>{}, 1);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
  EXPECT_NE(result.status().ToString().find("executor"), std::string::npos);
}

TEST(FollyArrowExecutorTest, RejectsInlineLikeExecutors) {
  auto expect_rejected = [](folly::Executor& executor) {
    auto result = MakeFollyArrowExecutor(folly::getKeepAliveToken(executor), 1);
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
    EXPECT_NE(result.status().ToString().find("non-inline"), std::string::npos);
  };

  expect_rejected(folly::InlineExecutor::instance());
  expect_rejected(folly::QueuedImmediateExecutor::instance());
}

TEST(FollyArrowExecutorTest, SubmissionAllocationFailureIsReported) {
  BadAllocExecutor folly_executor;
  auto arrow_executor = MakeFollyArrowExecutor(folly::getKeepAliveToken(folly_executor), 1);

  auto maybe_future = arrow_executor->Submit([] { return 42; });

  ASSERT_FALSE(maybe_future.ok());
  EXPECT_TRUE(maybe_future.status().IsUnknownError()) << maybe_future.status().ToString();
}

}  // namespace
}  // namespace milvus_storage::parquet::test
