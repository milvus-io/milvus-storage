// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "tracing_bridge.h"
#include "tracing/runtime.h"
namespace milvus_storage::rust_bridge::ffi {
namespace {
struct ScopedAttachment final : TraceAttachment {
  explicit ScopedAttachment(tracing::ContextPtr context) : scope(std::move(context)) {}
  tracing::ContextScope scope;
};
}  // namespace
TraceAttachment::~TraceAttachment() = default;
std::shared_ptr<TraceContext> capture_trace_context() {
  auto snapshot = tracing::Capture();
  if (!snapshot)
    return nullptr;
  auto context = std::make_shared<TraceContext>();
  context->context = std::move(snapshot);
  return context;
}
std::unique_ptr<TraceAttachment> attach_trace_context(const std::shared_ptr<TraceContext>& context) {
  // No allocation for the usual disabled path. A captured empty context still
  // masks any unrelated Storage parent on the worker executing this poll.
  if (!context && !tracing::HasContext())
    return nullptr;
  auto attachment = std::make_unique<ScopedAttachment>(context ? context->context : nullptr);
  tracing::StartCurrent();
  return attachment;
}
}  // namespace milvus_storage::rust_bridge::ffi
