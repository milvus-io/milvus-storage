// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
namespace milvus_storage::rust_bridge::ffi {
struct TraceContext;
struct TraceAttachment;
std::shared_ptr<TraceContext> capture_trace_context();
std::unique_ptr<TraceAttachment> attach_trace_context(const std::shared_ptr<TraceContext>& context);
// Out-of-line destructors keep the private Storage runtime out of the CXX ABI.
struct TraceContext {
  struct Impl;
  std::shared_ptr<Impl> impl;
};
struct TraceAttachment {
  struct Impl;
  std::unique_ptr<Impl> impl;
  TraceAttachment();
  ~TraceAttachment();
};
}  // namespace milvus_storage::rust_bridge::ffi
