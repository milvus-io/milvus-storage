// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#ifdef WITH_TALON
#include "talon/tracing_bridge.h"
#include "rust-bridge/talon/talon_bridge.h"
#include "tracing/runtime.h"
#include <algorithm>
#include <chrono>
#include <opentelemetry/trace/trace_state.h>

namespace milvus_storage::talon::ffi {
namespace ot = opentelemetry::trace;
namespace {
using Clock = std::chrono::steady_clock;
opentelemetry::nostd::string_view View(rust::Str value) { return {value.data(), value.size()}; }
opentelemetry::nostd::string_view View(const rust::String& value) { return {value.data(), value.size()}; }
TraceCarrier Carrier(const ot::SpanContext& context) {
  TraceCarrier carrier;
  const auto trace = context.trace_id();
  const auto span = context.span_id();
  std::copy(trace.Id().begin(), trace.Id().end(), carrier.trace_id.begin());
  std::copy(span.Id().begin(), span.Id().end(), carrier.span_id.begin());
  carrier.flags = context.trace_flags().flags();
  carrier.remote = context.IsRemote();
  carrier.state = context.trace_state()->ToHeader();
  return carrier;
}
ot::SpanContext Context(const TraceCarrier& carrier, bool remote) {
  return ot::SpanContext(ot::TraceId(carrier.trace_id), ot::SpanId(carrier.span_id), ot::TraceFlags(carrier.flags),
                         remote, ot::TraceState::FromHeader(View(carrier.state)));
}
opentelemetry::common::AttributeValue Value(const SpanAttribute& value) {
  switch (value.kind) {
    case 0:
      return value.boolean;
    case 1:
      return value.integer;
    case 2:
      return value.floating;
    default:
      return View(value.text);
  }
}
class Attributes final : public opentelemetry::common::KeyValueIterable {
  public:
  explicit Attributes(rust::Slice<const SpanAttribute> values) : values_(values) {}
  size_t size() const noexcept override { return values_.size(); }
  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue)>
          callback) const noexcept override {
    for (const auto& value : values_) {
      if (!callback(View(value.key), Value(value)))
        return false;
    }
    return true;
  }

  private:
  rust::Slice<const SpanAttribute> values_;
};
}  // namespace
struct TalonSpan {
  std::shared_ptr<TalonTrace> trace;  // Retain this request's provider through completion.
  opentelemetry::nostd::shared_ptr<ot::Span> span;
  uint64_t start_ns;
  Clock::time_point start_steady;
};
std::shared_ptr<TalonTrace> capture_talon_trace() noexcept { return tracing::CaptureExternalTrace(); }
TraceCarrier talon_trace_parent(const TalonTrace& trace) { return Carrier(tracing::ExternalParent(trace)); }
std::shared_ptr<TalonSpan> start_talon_span(const std::shared_ptr<TalonTrace>& trace,
                                            rust::Str name,
                                            uint8_t kind,
                                            const TraceCarrier& parent,
                                            uint64_t start_ns,
                                            rust::Slice<const SpanAttribute> attributes) {
  if (!trace)
    return nullptr;
  // Talon's W3C parser marks every carrier remote. These SDK spans are in the
  // same process: preserve the original parent's bit, and keep SDK children local.
  const auto original = tracing::ExternalParent(*trace);
  const bool remote = original.IsRemote() && original.trace_id() == ot::TraceId(parent.trace_id) &&
                      original.span_id() == ot::SpanId(parent.span_id);
  auto parent_context = Context(parent, remote);
  if (!parent_context.IsValid())
    return nullptr;
  ot::StartSpanOptions options;
  options.parent = parent_context;
  options.kind = kind == 1 ? ot::SpanKind::kClient : ot::SpanKind::kInternal;
  const auto start_system = std::chrono::system_clock::time_point(std::chrono::nanoseconds(start_ns));
  const auto start_steady = Clock::now() - (std::chrono::system_clock::now() - start_system);
  options.start_system_time = start_system;
  options.start_steady_time = start_steady;
  auto span = tracing::StartExternalSpan(*trace, View(name), Attributes(attributes), options);
  if (!span)
    return nullptr;
  return std::make_shared<TalonSpan>(TalonSpan{trace, std::move(span), start_ns, start_steady});
}
TraceCarrier talon_span_context(const TalonSpan& span) { return Carrier(span.span->GetContext()); }
void finish_talon_span(const TalonSpan& span,
                       rust::Str name,
                       uint8_t status,
                       uint64_t end_ns,
                       rust::Slice<const SpanAttribute> attributes) {
  span.span->UpdateName(View(name));
  for (const auto& value : attributes) span.span->SetAttribute(View(value.key), Value(value));
  // Talon reports sanitized outcome labels; do not export business error text.
  if (status != 0)
    span.span->SetStatus(status == 1 ? ot::StatusCode::kOk : ot::StatusCode::kError);
  ot::EndSpanOptions options;
  options.end_steady_time =
      span.start_steady + std::chrono::nanoseconds(end_ns > span.start_ns ? end_ns - span.start_ns : 0);
  span.span->End(options);
}
}  // namespace milvus_storage::talon::ffi
#endif
