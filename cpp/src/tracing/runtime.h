// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "milvus-storage/tracing.h"
#include <atomic>
#include <mutex>
#include <optional>
#include <utility>
#include <folly/io/async/Request.h>
#include <folly/futures/Future.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <opentelemetry/trace/tracer.h>

namespace milvus_storage::tracing {
struct Context;
using ContextPtr = std::shared_ptr<const Context>;
ContextPtr Capture();
// Check the active flow without copying an owning context snapshot.
bool HasContext();
void StartCurrent();

class ContextScope {
  public:
  explicit ContextScope(ContextPtr context);

  private:
  std::optional<folly::ShallowCopyRequestContextScopeGuard> scope_;
};

class OperationTrace {
  public:
  OperationTrace() = default;
  // Lazy spans start on first actual work, not on construction of a SemiFuture.
  OperationTrace(const char* name,
                 bool lazy = false,
                 bool io = false,
                 opentelemetry::trace::SpanContext link = opentelemetry::trace::SpanContext::GetInvalid(),
                 const char* operation = nullptr,
                 const char* format = nullptr);
  ContextPtr context() const { return context_; }
  // A disabled snapshot still propagates, but needs no span completion or I/O accounting.
  bool IsEnabled() const;
  bool NeedsCompletion() const { return owns_state_; }
  void Start() const;
  void AccountRead(int64_t requested, int64_t returned) const;
  void Finish(const arrow::Status& status) const;
  void Attribute(const char* key, int64_t value) const;
  void Attribute(const char* key, const char* value) const;
  opentelemetry::trace::SpanContext span_context() const;

  private:
  ContextPtr context_;
  bool owns_state_ = false;
};

inline const arrow::Status& StatusOf(const arrow::Status& status) { return status; }
template <typename T>
const arrow::Status& StatusOf(const arrow::Result<T>& result) {
  return result.status();
}

// Restores only Storage-owned data, preserving other RequestContext keys.
template <typename F>
auto Bind(F&& fn) {
  return [context = Capture(), fn = std::forward<F>(fn)](auto&&... args) mutable -> decltype(auto) {
    ContextScope scope(context);
    StartCurrent();
    return fn(std::forward<decltype(args)>(args)...);
  };
}

template <typename F>
auto Run(const char* name, F&& fn, bool io = false, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn()) {
  if (!HasContext())
    return fn();
  OperationTrace trace(name, false, io, opentelemetry::trace::SpanContext::GetInvalid(), operation, format);
  ContextScope scope(trace.context());
  try {
    auto result = fn();
    trace.Finish(StatusOf(result));
    return result;
  } catch (...) {
    auto status = arrow::Status::UnknownError("exception");
    trace.Finish(status);
    return status;
  }
}

template <typename F>
auto RunAsync(const char* name, F&& fn, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn()) {
  if (!HasContext())
    return fn();
  OperationTrace trace(name, true, false, opentelemetry::trace::SpanContext::GetInvalid(), operation, format);
  ContextScope scope(trace.context());
  using Result = typename decltype(fn())::value_type;
  try {
    if (!trace.IsEnabled()) {
      auto future = fn();
      if (future.isReady()) {
        if (future.result().hasException())
          return folly::makeSemiFuture(Result(arrow::Status::UnknownError("exception")));
        return future;
      }
      // Keep exception-to-result conversion without capturing OperationTrace or
      // observing span completion. Folly still propagates its RequestContext.
      return std::move(future).defer([](folly::Try<Result>&& result) -> Result {
        if (result.hasException())
          return arrow::Status::UnknownError("exception");
        return std::move(result).value();
      });
    }
    return fn().defer([trace](auto&& result) -> Result {
      if (result.hasException()) {
        auto status = arrow::Status::UnknownError("exception");
        trace.Finish(status);
        return status;
      }
      trace.Finish(StatusOf(result.value()));
      return std::move(result).value();
    });
  } catch (...) {
    auto status = arrow::Status::UnknownError("exception");
    trace.Finish(status);
    return folly::makeSemiFuture(Result(status));
  }
}

// Native callbacks own completion. Return the original future so tracing does
// not introduce consumer-executor work or change eager native scheduling.
template <typename F>
auto RunNativeAsync(const char* name, F&& fn, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn(std::declval<OperationTrace>())) {
  if (!HasContext())
    return fn(OperationTrace{});
  OperationTrace trace(name, false, false, opentelemetry::trace::SpanContext::GetInvalid(), operation, format);
  ContextScope scope(trace.context());
  using Result = typename decltype(fn(trace))::value_type;
  try {
    auto future = fn(trace);
    if (future.isReady()) {
      const auto& result = future.result();
      if (result.hasException()) {
        auto status = arrow::Status::UnknownError("exception");
        trace.Finish(status);
        return folly::makeSemiFuture(Result(status));
      }
      trace.Finish(StatusOf(result.value()));
    }
    return future;
  } catch (...) {
    auto status = arrow::Status::UnknownError("exception");
    trace.Finish(status);
    return folly::makeSemiFuture(Result(status));
  }
}

// Observe the source Arrow future itself: completion does not depend on a
// consumer running a continuation and remains observed after a future is dropped.
template <typename T>
arrow::Future<T> Observe(arrow::Future<T> future, OperationTrace trace) {
  if (!trace.NeedsCompletion())
    return future;
  future.AddCallback([trace](const arrow::Result<T>& result) { trace.Finish(result.status()); });
  return future;
}
}  // namespace milvus_storage::tracing
