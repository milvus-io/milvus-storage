// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "tracing/runtime.h"
#include "milvus-storage/common/extend_status.h"
#include <opentelemetry/trace/trace_state.h>
#include <string_view>

namespace milvus_storage::tracing {
namespace ot = opentelemetry::trace;
namespace {
struct Configuration {
  ProviderPtr provider;
  opentelemetry::nostd::shared_ptr<ot::Tracer> tracer;
  TraceOptions options;
};
std::mutex configuration_mutex;
std::shared_ptr<const Configuration> configuration = std::make_shared<Configuration>();
const folly::RequestToken storage_key("milvus-storage.tracing.v1");
// Contexts can only originate from AttachParent. Until the first attachment,
// avoid touching Folly's request-context TLS on the default disabled path.
// Never reset: old operations must retain context after provider replacement.
std::atomic<bool> contexts_seen{false};
struct Budget {
  std::atomic<uint64_t> used{0};
  std::atomic<int64_t> dropped{0};
  std::atomic<int64_t> reads{0}, requested_bytes{0}, returned_bytes{0};
};
struct SpanState {
  mutable std::mutex mutex;
  opentelemetry::nostd::shared_ptr<ot::Span> span;
  ContextPtr parent;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  const char* name;
  const char* operation = nullptr;
  const char* format = nullptr;
  ot::SpanContext link = ot::SpanContext::GetInvalid();
  bool finished = false;
  bool root = false;
  // Publishes the immutable span pointer (or a disabled decision) once Start
  // completes. Keep this beside the other flags to reuse their padding.
  std::atomic<bool> started{false};
  void Start();
  ~SpanState() {
    if (span && !finished) {
      span->SetAttribute("storage.completion.unobserved", true);
      span->End();
    }
  }
};
}  // namespace
struct Context {
  ot::SpanContext parent = ot::SpanContext::GetInvalid();
  std::shared_ptr<SpanState> operation;
  // Freeze a disabled operation without allocating SpanState, Budget or a mutex.
  bool disabled = false;
};
namespace {
ot::SpanContext Parent(const ContextPtr& context) {
  if (!context)
    return ot::SpanContext::GetInvalid();
  if (!context->operation)
    return context->parent;
  auto& op = context->operation;
  op->Start();
  return op->span ? op->span->GetContext() : Parent(op->parent);
}
void SpanState::Start() {
  if (started.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(mutex);
  if (span || finished) {
    started.store(true, std::memory_order_release);
    return;
  }
  auto parent_context = Parent(parent);
  if (!parent_context.IsValid() || !config->tracer) {
    started.store(true, std::memory_order_release);
    return;
  }
  ot::StartSpanOptions options;
  options.parent = parent_context;
  if (link.IsValid()) {
    span = config->tracer->StartSpan(name, {}, {{link, {}}}, options);
  } else {
    span = config->tracer->StartSpan(name, options);
  }
  if (span && span->IsRecording()) {
    if (operation)
      span->SetAttribute("storage.operation", operation);
    if (format)
      span->SetAttribute("storage.format", format);
  }
  if (span)
    started.store(true, std::memory_order_release);
}
struct Data final : folly::RequestData {
  explicit Data(ContextPtr value) : context(std::move(value)) {}
  bool hasCallback() override { return false; }
  const ContextPtr context;
};
}  // namespace
ContextPtr Capture() {
  if (!contexts_seen.load(std::memory_order_relaxed))
    return nullptr;
  auto* data = static_cast<Data*>(folly::RequestContext::get()->getContextData(storage_key));
  return data ? data->context : nullptr;
}
bool HasContext() {
  if (!contexts_seen.load(std::memory_order_relaxed))
    return false;
  auto* data = static_cast<Data*>(folly::RequestContext::get()->getContextData(storage_key));
  return data && data->context;
}
void StartCurrent() {
  auto context = Capture();
  if (context && context->operation)
    context->operation->Start();
}
ContextScope::ContextScope(ContextPtr context) {
  // Common disabled path does not allocate a RequestContext. A captured empty
  // context must still mask unrelated context on a foreign completion thread.
  if (context || HasContext())
    scope_.emplace(storage_key, std::make_unique<Data>(std::move(context)));
}

struct TraceScope::Impl {
  explicit Impl(const TraceParent& parent)
      : scope(std::make_shared<Context>(Context{ot::SpanContext(ot::TraceId(parent.trace_id),
                                                                ot::SpanId(parent.span_id),
                                                                ot::TraceFlags(parent.trace_flags),
                                                                parent.is_remote,
                                                                ot::TraceState::FromHeader(parent.tracestate)),
                                                nullptr})) {}
  ContextScope scope;
};
TraceScope::TraceScope(const TraceParent& parent) {
  contexts_seen.store(true, std::memory_order_relaxed);
  impl_ = std::make_unique<Impl>(parent);
}
TraceScope::~TraceScope() = default;
TraceScope AttachParent(const TraceParent& parent) { return TraceScope(parent); }
void SetTracerProvider(ProviderPtr provider) {
  auto tracer = provider ? provider->GetTracer("milvus-storage", MILVUS_STORAGE_VERSION) : nullptr;
  std::lock_guard<std::mutex> lock(configuration_mutex);
  auto next = std::make_shared<Configuration>(*configuration);
  next->provider = std::move(provider);
  next->tracer = std::move(tracer);
  configuration = std::move(next);
}
void SetTraceOptions(const TraceOptions& options) {
  std::lock_guard<std::mutex> lock(configuration_mutex);
  auto next = std::make_shared<Configuration>(*configuration);
  next->options = options;
  configuration = std::move(next);
}
OperationTrace::OperationTrace(
    const char* name, bool lazy, bool io, ot::SpanContext link, const char* operation, const char* format) {
  context_ = Capture();
  if (!context_ || context_->disabled)
    return;
  // The cache leader already owns the physical metadata load. A format open
  // below that leader must not create another span for the same work.
  if (context_->operation && std::string_view(name) == "storage.metadata.load" &&
      std::string_view(context_->operation->name) == "storage.metadata.load")
    return;
  const bool root = !context_->operation;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  if (!root) {
    config = context_->operation->config;
    // Children retain their parent's fixed configuration even if the host
    // injects a provider later. No child state is needed for suppressed spans.
    if (!config->tracer || (io && !config->options.io_spans))
      return;
    budget = context_->operation->budget;
    if (budget->used.fetch_add(1, std::memory_order_relaxed) >=
        std::max<uint32_t>(1, config->options.max_spans_per_operation)) {
      budget->dropped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  } else {
    if (!context_->parent.IsValid())
      return;
    {
      std::lock_guard<std::mutex> lock(configuration_mutex);
      config = configuration;
    }
    if (!config->tracer || (io && !config->options.io_spans)) {
      context_ = std::make_shared<Context>(Context{context_->parent, nullptr, true});
      return;
    }
    budget = std::make_shared<Budget>();
  }
  auto state = std::make_shared<SpanState>();
  state->parent = context_;
  state->name = name;
  state->operation = operation;
  state->format = format;
  state->link = std::move(link);
  state->config = std::move(config);
  state->budget = std::move(budget);
  state->root = root;
  if (state->root)
    state->budget->used.store(1, std::memory_order_relaxed);
  context_ = std::make_shared<Context>(Context{ot::SpanContext::GetInvalid(), std::move(state)});
  owns_state_ = true;
  if (!lazy)
    Start();
}
bool OperationTrace::IsEnabled() const { return context_ && context_->operation; }
void OperationTrace::AccountRead(int64_t requested, int64_t returned) const {
  if (!context_ || !context_->operation)
    return;
  auto budget = context_->operation->budget;
  budget->reads.fetch_add(1, std::memory_order_relaxed);
  budget->requested_bytes.fetch_add(std::max<int64_t>(0, requested), std::memory_order_relaxed);
  budget->returned_bytes.fetch_add(std::max<int64_t>(0, returned), std::memory_order_relaxed);
}
void OperationTrace::Start() const {
  if (owns_state_)
    context_->operation->Start();
}
void OperationTrace::Finish(const arrow::Status& status) const {
  if (!owns_state_)
    return;
  auto op = context_->operation;
  op->Start();
  {
    std::lock_guard<std::mutex> lock(op->mutex);
    if (op->finished)
      return;
    op->finished = true;
  }
  if (!op->span)
    return;
  if (!status.ok()) {
    op->span->SetStatus(ot::StatusCode::kError);
    if (op->span->IsRecording()) {
      if (auto detail = ExtendStatusDetail::UnwrapStatus(status)) {
        op->span->SetAttribute("error.type", detail->CodeAsString());
        op->span->SetAttribute("error.retryable", detail->retryable());
      } else {
        op->span->SetAttribute("error.type", status.CodeAsString());
      }
    }
  }
  if (op->root) {
    op->span->SetAttribute("storage.spans.dropped", op->budget->dropped.load());
    op->span->SetAttribute("storage.io.reads", op->budget->reads.load());
    op->span->SetAttribute("storage.io.requested_bytes", op->budget->requested_bytes.load());
    op->span->SetAttribute("storage.io.returned_bytes", op->budget->returned_bytes.load());
  }
  op->span->End();
}
void OperationTrace::Attribute(const char* key, int64_t value) const {
  Start();
  if (owns_state_ && context_->operation->span)
    context_->operation->span->SetAttribute(key, value);
}
void OperationTrace::Attribute(const char* key, const char* value) const {
  Start();
  if (owns_state_ && context_->operation->span)
    context_->operation->span->SetAttribute(key, value);
}
ot::SpanContext OperationTrace::span_context() const { return Parent(context_); }
}  // namespace milvus_storage::tracing
