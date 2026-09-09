// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <opentelemetry/trace/tracer_provider.h>

namespace milvus_storage::tracing {

struct TraceParent {
  std::array<uint8_t, 16> trace_id{};
  std::array<uint8_t, 8> span_id{};
  uint8_t trace_flags = 0;
  std::string tracestate;
  bool is_remote = false;
};

using ProviderPtr = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>;

// Storage never installs a global provider, creates an exporter, or shuts down
// an injected provider. The host must use a matching OTel C++ ABI.
void SetTracerProvider(ProviderPtr provider);

struct TraceOptions {
  bool io_spans = true;
  // Includes the operation span. Excess children are aggregated on the operation.
  uint32_t max_spans_per_operation = 256;
};
// Changes apply to subsequent operations; active operations retain their snapshot.
void SetTraceOptions(const TraceOptions& options);

class TraceScope {
  public:
  ~TraceScope();
  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;
  TraceScope(TraceScope&&) = delete;
  TraceScope& operator=(TraceScope&&) = delete;

  private:
  struct Impl;
  explicit TraceScope(const TraceParent& parent);
  std::unique_ptr<Impl> impl_;
  friend TraceScope AttachParent(const TraceParent& parent);
};

// Only changes the Storage RequestContext. Does not create/end a parent span
// or modify OTel TLS. Invalid parents mask enclosing scopes. Destroy on the
// same execution flow in stack order; Folly fibers may suspend with this guard.
[[nodiscard]] TraceScope AttachParent(const TraceParent& parent);

}  // namespace milvus_storage::tracing
