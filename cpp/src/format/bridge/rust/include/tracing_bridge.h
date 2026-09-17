// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
namespace milvus_storage::tracing {
struct Context;
}
namespace milvus_storage::rust_bridge::ffi {
struct TraceContext;
struct TraceAttachment;
std::shared_ptr<TraceContext> capture_trace_context();
std::unique_ptr<TraceAttachment> attach_trace_context(const std::shared_ptr<TraceContext>& context);
// These remain opaque to Rust. Keep the shared snapshot directly in its handle
// rather than allocating a second shared Impl for every capture.
struct TraceContext {
  std::shared_ptr<const tracing::Context> context;
};
// The private derived object owns its scope in the same allocation. Virtual
// destruction lets CXX drop the opaque token without seeing the runtime types.
struct TraceAttachment {
  virtual ~TraceAttachment();
};
}  // namespace milvus_storage::rust_bridge::ffi
