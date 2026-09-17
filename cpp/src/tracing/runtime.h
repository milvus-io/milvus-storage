// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "milvus-storage/tracing.h"
#include <atomic>
#include <mutex>
#include <optional>
#include <utility>
#include <type_traits>
#include <exception>
#include <folly/io/async/Request.h>
#include <folly/futures/Future.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <opentelemetry/trace/tracer.h>

namespace milvus_storage::tracing {
// Classify the caught exception without copying text or rethrowing it.
void RecordFailure(TraceFailure failure, const std::exception* error = nullptr) noexcept;
struct Context;
using ContextPtr = std::shared_ptr<const Context>;
ContextPtr Capture() noexcept;
// Check the active flow without copying an owning context snapshot.
bool HasContext() noexcept;
void StartCurrent() noexcept;

// Immutable request configuration for SDK spans that execute in another runtime.
// The snapshot retains the provider and shares the Storage operation's budget.
struct ExternalTrace;
std::shared_ptr<ExternalTrace> CaptureExternalTrace() noexcept;
opentelemetry::trace::SpanContext ExternalParent(const ExternalTrace& trace) noexcept;
opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> StartExternalSpan(
    const ExternalTrace& trace,
    opentelemetry::nostd::string_view name,
    const opentelemetry::common::KeyValueIterable& attributes,
    const opentelemetry::trace::StartSpanOptions& options) noexcept;

class ContextScope {
  public:
  explicit ContextScope(ContextPtr context) noexcept;
  ~ContextScope();

  private:
  std::optional<folly::ShallowCopyRequestContextScopeGuard> scope_;
  bool failed_ = false;
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
                 const char* format = nullptr,
                 bool native_completion = false) noexcept;
  // Native completion only enqueues; the caller's TraceCompletionQueue owns End.
  static OperationTrace Native(const char* name,
                               bool io = false,
                               const char* operation = nullptr,
                               const char* format = nullptr) noexcept {
    return OperationTrace(name, false, io, opentelemetry::trace::SpanContext::GetInvalid(), operation, format, true);
  }
  static OperationTrace WithAttributes(opentelemetry::nostd::string_view name,
                                       TraceScope::Attributes attributes,
                                       TraceScope::Links links = {},
                                       opentelemetry::trace::SpanKind kind = opentelemetry::trace::SpanKind::kInternal,
                                       bool lazy = false) noexcept;
  ContextPtr context() const { return context_; }
  // A disabled snapshot still propagates, but needs no span completion or I/O accounting.
  bool IsEnabled() const;
  bool NeedsCompletion() const { return owns_state_; }
  void Start() const noexcept;
  void AccountRead(int64_t requested, int64_t returned) const noexcept;
  void Finish(const arrow::Status& status) const noexcept;
  void FinishCode(arrow::StatusCode code, std::shared_ptr<arrow::StatusDetail> detail = nullptr) const noexcept;
  void FinishScope() const noexcept;
  void FinishException() const noexcept;
  void Attribute(const char* key, int64_t value) const noexcept;
  void Attribute(const char* key, const char* value) const noexcept;
  void Attribute(opentelemetry::nostd::string_view key,
                 const opentelemetry::common::AttributeValue& value) const noexcept;
  opentelemetry::trace::SpanContext span_context() const noexcept;

  private:
  ContextPtr context_;
  bool owns_state_ = false;
  void FinishImpl(arrow::StatusCode code,
                  std::shared_ptr<arrow::StatusDetail> detail,
                  bool exception = false) const noexcept;
};

// Observe unwinding without intercepting or replacing the business exception.
class ExceptionObserver {
  public:
  explicit ExceptionObserver(const OperationTrace& trace) : trace_(trace), exceptions_(std::uncaught_exceptions()) {}
  ~ExceptionObserver() {
    if (std::uncaught_exceptions() > exceptions_)
      trace_.FinishException();
  }

  private:
  const OperationTrace& trace_;
  int exceptions_;
};

inline const arrow::Status& StatusOf(const arrow::Status& status) { return status; }
template <typename T>
const arrow::Status& StatusOf(const arrow::Result<T>& result) {
  return result.status();
}

// Arrow 17 inspects operator()'s concrete argument types. Preserve them for
// ordinary lambdas; only genuinely generic callables need the variadic wrapper.
template <typename F, size_t... I>
auto BindTyped(F&& fn, std::index_sequence<I...>) {
  return [context = Capture(), fn = std::forward<F>(fn)](
             arrow::internal::call_traits::argument_type<I, F>... args) mutable -> decltype(auto) {
    ContextScope scope(context);
    StartCurrent();
    return fn(std::forward<arrow::internal::call_traits::argument_type<I, F>>(args)...);
  };
}
// Restores only Storage-owned data, preserving other RequestContext keys.
template <typename F>
auto Bind(F&& fn) {
  if constexpr (requires { &std::decay_t<F>::operator(); }) {
    return BindTyped(std::forward<F>(fn),
                     std::make_index_sequence<arrow::internal::call_traits::argument_count<F>::value>{});
  } else {
    return [context = Capture(), fn = std::forward<F>(fn)](
               auto&&... args) mutable -> std::invoke_result_t<std::decay_t<F>&, decltype(args)...> {
      ContextScope scope(context);
      StartCurrent();
      return fn(std::forward<decltype(args)>(args)...);
    };
  }
}

template <typename F>
auto Run(const char* name, F&& fn, bool io = false, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn()) {
  if (!HasContext())
    return fn();
  OperationTrace trace(name, false, io, opentelemetry::trace::SpanContext::GetInvalid(), operation, format);
  ContextScope scope(trace.context());
  ExceptionObserver observer(trace);
  auto result = fn();
  trace.Finish(StatusOf(result));
  return result;
}

template <typename F>
auto RunAsync(const char* name, F&& fn, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn()) {
  if (!HasContext())
    return fn();
  OperationTrace trace(name, true, false, opentelemetry::trace::SpanContext::GetInvalid(), operation, format);
  ContextScope scope(trace.context());
  ExceptionObserver observer(trace);
  auto future = fn();
  if (!trace.NeedsCompletion())
    return future;
  return std::move(future).defer([trace](auto&& result) {
    if (result.hasException()) {
      trace.FinishException();
    } else {
      trace.Finish(StatusOf(result.value()));
    }
    return std::move(result);
  });
}

// Native callbacks enqueue completion on the captured host queue. Drain owns
// span End/export. Returning the original future preserves eager scheduling.
template <typename F>
auto RunNativeAsync(const char* name, F&& fn, const char* operation = nullptr, const char* format = nullptr)
    -> decltype(fn(std::declval<OperationTrace>())) {
  if (!HasContext())
    return fn(OperationTrace{});
  auto trace = OperationTrace::Native(name, false, operation, format);
  ContextScope scope(trace.context());
  ExceptionObserver observer(trace);
  auto future = fn(trace);
  if (future.isReady()) {
    const auto& result = future.result();
    if (result.hasException()) {
      trace.FinishException();
    } else {
      trace.Finish(StatusOf(result.value()));
    }
  }
  return future;
}

// Observe the source Arrow future itself: completion does not depend on a
// consumer running a continuation and remains observed after a future is dropped.
// trace must be created with Native(); its host queue performs End/export.
template <typename T>
arrow::Future<T> Observe(arrow::Future<T> future, OperationTrace trace) {
  if (!trace.NeedsCompletion())
    return future;
  future.AddCallback([trace](const arrow::Result<T>& result) { trace.Finish(result.status()); });
  return future;
}
}  // namespace milvus_storage::tracing
