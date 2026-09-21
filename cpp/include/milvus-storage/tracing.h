// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <initializer_list>
#include <utility>
#include <arrow/status.h>
#include <opentelemetry/common/attribute_value.h>
#include <opentelemetry/trace/tracer_provider.h>
#include <opentelemetry/trace/span_context.h>
#include <opentelemetry/trace/span_startoptions.h>

namespace milvus_storage::tracing {

struct TraceParent {
  TraceParent() = default;
  explicit TraceParent(const opentelemetry::trace::SpanContext& upstream);
  std::array<uint8_t, 16> trace_id{};
  std::array<uint8_t, 8> span_id{};
  uint8_t trace_flags = 0;
  std::string tracestate;
  bool is_remote = false;
};

using ProviderPtr = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>;

// Storage never installs a global provider, creates an exporter, or shuts down
// an injected provider. The host must use a matching OTel C++ ABI.
arrow::Status SetTracerProvider(ProviderPtr provider) noexcept;

struct TraceOptions {
  bool io_spans = true;
  // Includes the operation span. Excess children are aggregated on the operation.
  uint32_t max_spans_per_operation = 256;
};
// Changes apply to subsequent operations; active operations retain their snapshot.
arrow::Status SetTraceOptions(const TraceOptions& options) noexcept;

// Monotonic, allocation-free diagnostics. Snapshot fields independently; callers
// may export deltas through their own metrics system. No logging/export callbacks
// run on the failing thread, and business results are never replaced.
enum class TraceFailure : uint8_t {
  Create,
  Attach,
  Attributes,
  Complete,
  Metadata,
  RustCapture,
  RustAttach,
  MissingCompletionQueue,
  Count
};
struct TraceFailures {
  std::array<uint64_t, static_cast<size_t>(TraceFailure::Count)> counts{};
  std::array<uint64_t, static_cast<size_t>(TraceFailure::Count)> allocation_failures{};
  std::array<uint64_t, static_cast<size_t>(TraceFailure::Count)> standard_exceptions{};
};
TraceFailures GetTraceFailures() noexcept;

struct TraceCompletionNode;
// Native callbacks only enqueue preallocated completion nodes. The host must
// drain on its execution thread, periodically and after all outstanding work
// has finished, before shutting down the provider. Drain never waits for I/O.
// Pending nodes retain this queue until drained, including dropped futures.
class TraceCompletionQueue {
  public:
  TraceCompletionQueue() = default;
  TraceCompletionQueue(const TraceCompletionQueue&) = delete;
  TraceCompletionQueue& operator=(const TraceCompletionQueue&) = delete;
  void Drain() noexcept;
  // Internal producer entry point; nodes are owned by the tracing runtime.
  void Enqueue(TraceCompletionNode* node) noexcept;

  private:
  std::atomic<TraceCompletionNode*> pending_{nullptr};
};

class TraceScope {
  public:
  using Attributes =
      std::initializer_list<std::pair<opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue>>;
  using Links = std::initializer_list<std::pair<opentelemetry::trace::SpanContext, Attributes>>;
  explicit TraceScope(opentelemetry::nostd::string_view name,
                      Attributes attributes = {},
                      Links links = {},
                      opentelemetry::trace::SpanKind kind = opentelemetry::trace::SpanKind::kInternal) noexcept;
  ~TraceScope() noexcept;
  // Explicit completion records the result and restores the preceding context.
  // Normal destruction leaves status unset; unwinding records only an exception classification.
  void Finish(const arrow::Status& status) noexcept;
  void SetAttribute(opentelemetry::nostd::string_view key, const opentelemetry::common::AttributeValue& value) noexcept;
  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;
  TraceScope(TraceScope&&) = delete;
  TraceScope& operator=(TraceScope&&) = delete;

  private:
  struct Impl;
  explicit TraceScope(const TraceParent& parent, std::shared_ptr<TraceCompletionQueue> completions) noexcept;
  std::unique_ptr<Impl> impl_;
  bool failed_ = false;
  friend TraceScope AttachParent(const TraceParent& parent, std::shared_ptr<TraceCompletionQueue> completions) noexcept;
};

// Only changes the Storage RequestContext. Does not create/end a parent span
// or modify OTel TLS. Invalid parents mask enclosing scopes. Destroy on the
// same execution flow in stack order; Folly fibers may suspend with this guard.
// Native async spans require a completion queue. Without one those spans are
// suppressed and MissingCompletionQueue is incremented; synchronous/Folly spans
// and the business operation remain available.
[[nodiscard]] TraceScope AttachParent(const TraceParent& parent,
                                      std::shared_ptr<TraceCompletionQueue> completions = nullptr) noexcept;

}  // namespace milvus_storage::tracing
