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

#include "milvus-storage/ffi_async_context.h"
#include "milvus-storage/async.h"
#include <folly/Executor.h>
#include <folly/futures/Future.h>
#include <folly/executors/InlineExecutor.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

struct LoonAsyncOperation {
  std::shared_ptr<milvus_storage::AsyncOperation> state;
};
// Scheduling adapter only: all threads and queues belong to the caller.
class ExternalExecutor final : public folly::Executor {
  public:
  explicit ExternalExecutor(LoonAsyncExecutor descriptor) : descriptor_(descriptor) {}
  bool Submit(folly::Func task) { return TrySubmit(task); }
  void add(folly::Func task) override {
    // Initial admission uses Submit. Once accepted, Folly tasks cannot be
    // dropped: an exceptional queue failure completes them on this thread.
    try {
      if (TrySubmit(task))
        return;
    } catch (...) {
    }
    task();
  }

  private:
  bool TrySubmit(folly::Func& task) {
    auto* raw = new folly::Func(std::move(task));
    auto rejected = descriptor_.submit(
        descriptor_.context,
        [](void* data) {
          std::unique_ptr<folly::Func> owned(static_cast<folly::Func*>(data));
          (*owned)();
        },
        raw);
    if (rejected) {
      task = std::move(*raw);
      delete raw;
    }
    return !rejected;
  }
  LoonAsyncExecutor descriptor_;
};

// C callback admission/lifecycle belongs to the context passed by the caller.
// Native C++ operations inherit executors from their consuming SemiFuture.
struct LoonAsyncContext {
  explicit LoonAsyncContext(const LoonAsyncExecutor& executor)
      : executor_(std::make_shared<ExternalExecutor>(executor)) {}
  struct Admission {
    LoonAsyncContext* context = nullptr;
    ~Admission() {
      if (context) {
        std::lock_guard<std::mutex> lock(context->mutex_);
        --context->active_;
        context->drained_.notify_all();
      }
    }
  };
  template <typename T, typename Callback>
  milvus_storage::AsyncStatus Submit(folly::SemiFuture<T> future, Callback callback) {
    auto admission = std::make_shared<Admission>();
    std::shared_ptr<ExternalExecutor> executor;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_)
        return milvus_storage::AsyncStatus::Overloaded;
      executor = executor_;
      ++active_;
      admission->context = this;
    }
    struct Delivery {
      Callback callback;
      std::shared_ptr<Admission> admission;
      folly::Try<T> result;
      std::atomic<bool> delivered{false};
      Delivery(Callback cb, std::shared_ptr<Admission> lease) : callback(std::move(cb)), admission(std::move(lease)) {}
      void Run() {
        if (!delivered.exchange(true))
          callback(std::move(result));
      }
    };
    auto delivery = std::make_shared<Delivery>(std::move(callback), admission);
    // Initial enqueue rejection is synchronous and starts no native work.
    bool accepted = executor->Submit([future = std::move(future), executor, delivery]() mutable {
      try {
        // Already executing inside this exact caller executor. Bind deferred
        // work inline here to avoid a second, potentially rejected initial add.
        std::move(future)
            .viaInlineUnsafe(folly::getKeepAliveToken(executor.get()))
            .via(&folly::InlineExecutor::instance())
            .thenTry([executor, delivery](folly::Try<T>&& result) mutable {
              // Observe the result without an intermediate queue: a rejected
              // completion enqueue must not erase a confirmed write outcome.
              delivery->result = std::move(result);
              try {
                executor->add([delivery] { delivery->Run(); });
              } catch (...) {
                delivery->Run();
              }
            });
      } catch (...) {
        delivery->result = folly::Try<T>(folly::exception_wrapper(std::current_exception()));
        delivery->Run();
      }
    });
    return accepted ? milvus_storage::AsyncStatus{}
                    : milvus_storage::AsyncStatus{milvus_storage::AsyncStatus::Overloaded};
  }
  void Shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    stopping_ = true;
    drained_.wait(lock, [this] { return active_ == 0; });
  }
  ~LoonAsyncContext() { Shutdown(); }

  private:
  std::mutex mutex_;
  std::condition_variable drained_;
  std::shared_ptr<ExternalExecutor> executor_;
  size_t active_ = 0;  // Tracks accepted callbacks for shutdown; concurrency belongs to the caller executor.
  bool stopping_ = false;
};

namespace milvus_storage {
LoonFFIResult ValidateAsyncHandle(const LoonAsyncHandle* operation, uint64_t& timeout);
}  // namespace milvus_storage
