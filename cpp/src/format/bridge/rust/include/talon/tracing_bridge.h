// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include "rust/cxx.h"
namespace milvus_storage::tracing {
struct ExternalTrace;
}
namespace milvus_storage::talon::ffi {
using TalonTrace = tracing::ExternalTrace;
struct TalonSpan;
struct TraceCarrier;
struct SpanAttribute;
std::shared_ptr<TalonTrace> capture_talon_trace() noexcept;
TraceCarrier talon_trace_parent(const TalonTrace& trace);
std::shared_ptr<TalonSpan> start_talon_span(const std::shared_ptr<TalonTrace>& trace,
                                            rust::Str name,
                                            uint8_t kind,
                                            const TraceCarrier& parent,
                                            uint64_t start_ns,
                                            rust::Slice<const SpanAttribute> attributes);
TraceCarrier talon_span_context(const TalonSpan& span);
void finish_talon_span(const TalonSpan& span,
                       rust::Str name,
                       uint8_t status,
                       uint64_t end_ns,
                       rust::Slice<const SpanAttribute> attributes);
}  // namespace milvus_storage::talon::ffi
