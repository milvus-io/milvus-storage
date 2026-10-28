// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <arrow/io/interfaces.h>
#include <arrow/util/future.h>

namespace milvus_storage {

// Shared request admission for one S3 filesystem, before payload allocation or dispatch.
class S3RequestLimiter {
  struct State;

  public:
  class Permit {
 public:
    ~Permit();
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;

 private:
    friend class S3RequestLimiter;
    friend struct State;
    explicit Permit(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
  };

  explicit S3RequestLimiter(size_t limit);
  arrow::Future<std::shared_ptr<Permit>> Acquire(const arrow::io::IOContext& io);

  private:
  std::shared_ptr<State> state_;
};

struct S3RequestLimiter::State : std::enable_shared_from_this<S3RequestLimiter::State> {
  explicit State(size_t limit) : limit(limit) {}

  std::mutex mutex;
  size_t inflight = 0;
  size_t pending_releases = 0;
  bool releasing = false;
  const size_t limit;
  std::deque<arrow::Future<std::shared_ptr<Permit>>> waiters;

  void Release() {
    std::unique_lock lock(mutex);
    ++pending_releases;
    if (releasing) {
      return;
    }
    releasing = true;
    // Immediate cancellation can return a permit inside MarkFinished. Record
    // those returns, including concurrent ones, and drain them without recursion.
    while (pending_releases != 0) {
      --pending_releases;
      if (waiters.empty()) {
        --inflight;
        continue;
      }
      {
        auto next = std::move(waiters.front());
        waiters.pop_front();
        lock.unlock();
        std::shared_ptr<Permit> permit;
        try {
          permit = std::shared_ptr<Permit>(new Permit(nullptr));
        } catch (const std::bad_alloc&) {
          // No permit owns this slot yet; keep it available for the next waiter.
          std::lock_guard guard(mutex);
          ++pending_releases;
        }
        if (permit) {
          permit->state_ = shared_from_this();
          next.MarkFinished(std::move(permit));
        } else {
          next.MarkFinished(arrow::Status::OutOfMemory("S3 request admission"));
        }
        // Destroy next outside the lock: its result may return the permit.
      }
      lock.lock();
    }
    releasing = false;
  }
};

inline S3RequestLimiter::S3RequestLimiter(size_t limit) : state_(std::make_shared<State>(limit)) {}

inline S3RequestLimiter::Permit::~Permit() {
  if (state_) {
    state_->Release();
  }
}

inline arrow::Future<std::shared_ptr<S3RequestLimiter::Permit>> S3RequestLimiter::Acquire(
    const arrow::io::IOContext& io) {
  ARROW_RETURN_NOT_OK(io.stop_token().Poll());
  auto future = arrow::Future<std::shared_ptr<Permit>>::Make();
  auto permit = std::shared_ptr<Permit>(new Permit(nullptr));
  std::lock_guard lock(state_->mutex);
  if (state_->inflight < state_->limit) {
    ++state_->inflight;
    permit->state_ = state_;
    future.MarkFinished(std::move(permit));
  } else {
    state_->waiters.push_back(future);
  }
  return future;
}

}  // namespace milvus_storage
