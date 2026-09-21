// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "tracing/runtime.h"
#include "common/exception.h"
#include "milvus-storage/common/extend_status.h"
#include <opentelemetry/trace/trace_state.h>
#include <string_view>
#include <algorithm>
#include <vector>
#include <opentelemetry/sdk/common/attribute_utils.h>
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage::tracing {
namespace ot = opentelemetry::trace;
struct TraceCompletionNode {
  TraceCompletionNode* next = nullptr;
  virtual void CompletePending() noexcept = 0;
  virtual ~TraceCompletionNode() = default;
};
void TraceCompletionQueue::Enqueue(TraceCompletionNode* node) noexcept {
  node->next = pending_.load(std::memory_order_relaxed);
  while (!pending_.compare_exchange_weak(node->next, node, std::memory_order_release, std::memory_order_relaxed)) {
  }
}
void TraceCompletionQueue::Drain() noexcept {
  // Releasing a completed child can abandon its last native parent. Drain
  // those newly queued nodes too, so a final drain after producer shutdown is
  // sufficient to release the entire retained chain.
  for (;;) {
    auto* pending = pending_.exchange(nullptr, std::memory_order_acquire);
    if (!pending)
      return;
    while (pending) {
      auto* next = pending->next;
      pending->CompletePending();
      pending = next;
    }
  }
}
namespace {
std::array<std::atomic<uint64_t>, static_cast<size_t>(TraceFailure::Count)> failures{};
std::array<std::atomic<uint64_t>, static_cast<size_t>(TraceFailure::Count)> allocation_failures{};
std::array<std::atomic<uint64_t>, static_cast<size_t>(TraceFailure::Count)> standard_exceptions{};
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
// An attachment failure must not let unrelated flows inherit a foreign parent.
std::atomic<uint32_t> failed_scopes{0};
struct Budget {
  std::atomic<uint64_t> used{0};
  std::atomic<int64_t> dropped{0};
  std::atomic<int64_t> reads{0}, requested_bytes{0}, returned_bytes{0};
};
class OwnedAttributes final : public opentelemetry::common::KeyValueIterable {
  public:
  explicit OwnedAttributes(TraceScope::Attributes attributes) {
    // AttributeMap::SetAttribute is noexcept despite allocating. Use the SDK's
    // owning conversion directly so our allocation failures reach the caller's
    // tracing-only catch boundary instead of terminating inside that setter.
    for (const auto& [key, value] : attributes) {
      attributes_.insert_or_assign(
          std::string(key), opentelemetry::nostd::visit(opentelemetry::sdk::common::AttributeConverter{}, value));
    }
  }

  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue)>
          callback) const noexcept override {
    try {
      // Clang 14 cannot capture structured bindings in the visitor below.
      for (const auto& entry : attributes_) {
        const auto& key = entry.first;
        const auto& attribute = entry.second;
        const bool keep_going = opentelemetry::nostd::visit(
            [&](const auto& value) {
              using Value = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<Value, std::vector<bool>>) {
                // vector<bool> has no contiguous bool storage to expose as a span.
                auto values = std::make_unique<bool[]>(value.size());
                std::copy(value.begin(), value.end(), values.get());
                return callback(key, opentelemetry::nostd::span<const bool>(values.get(), value.size()));
              } else if constexpr (std::is_same_v<Value, std::vector<std::string>>) {
                std::vector<opentelemetry::nostd::string_view> values(value.begin(), value.end());
                return callback(key, opentelemetry::nostd::span<const opentelemetry::nostd::string_view>(values));
              } else if constexpr (std::is_arithmetic_v<Value>) {
                return callback(key, value);
              } else if constexpr (std::is_same_v<Value, std::string>) {
                return callback(key, opentelemetry::nostd::string_view(value));
              } else {
                return callback(key, opentelemetry::nostd::span<const typename Value::value_type>(value));
              }
            },
            attribute);
        if (!keep_going)
          return false;
      }
      return true;
    } catch (const std::exception& error) {
      RecordFailure(TraceFailure::Attributes, &error);
      // SDK iteration callbacks are noexcept; our temporary array views must
      // not let allocation failures escape through that contract.
      return false;
    } catch (...) {
      RecordFailure(TraceFailure::Attributes);
      // SDK iteration callbacks are noexcept; our temporary array views must
      // not let allocation failures escape through that contract.
      return false;
    }
  }

  size_t size() const noexcept override { return attributes_.size(); }

  private:
  opentelemetry::sdk::common::AttributeMap attributes_;
};
struct DeferredSpan final : ot::SpanContextKeyValueIterable {
  DeferredSpan(opentelemetry::nostd::string_view span_name,
               TraceScope::Attributes span_attributes,
               TraceScope::Links span_links)
      : name(span_name), attributes(span_attributes) {
    links.reserve(span_links.size());
    for (const auto& [context, values] : span_links) links.emplace_back(context, values);
  }

  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(ot::SpanContext, const opentelemetry::common::KeyValueIterable&)>
          callback) const noexcept override {
    try {
      for (const auto& [context, values] : links) {
        if (!callback(context, values))
          return false;
      }
      return true;
    } catch (const std::exception& error) {
      RecordFailure(TraceFailure::Attributes, &error);
      return false;
    } catch (...) {
      RecordFailure(TraceFailure::Attributes);
      return false;
    }
  }

  size_t size() const noexcept override { return links.size(); }

  const std::string name;
  const OwnedAttributes attributes;
  std::vector<std::pair<ot::SpanContext, OwnedAttributes>> links;
};
struct SpanState : TraceCompletionNode {
  std::shared_ptr<TraceCompletionQueue> completions;
  std::shared_ptr<SpanState> pending_owner;
  arrow::StatusCode pending_code = arrow::StatusCode::OK;
  std::shared_ptr<ExtendStatusDetail> pending_detail;
  bool pending_exception = false;
  ot::EndSpanOptions end_options;
  void Complete() noexcept;
  void CompletePending() noexcept override {
    auto owner = std::move(pending_owner);
    Complete();
    if (!owner)
      delete this;
  }
  bool unobserved = false;
  // Even an abandoned native span must be ended only by the host's drain.
  void Release() noexcept {
    if (completions && span && !finished) {
      finished = true;
      unobserved = true;
      end_options.end_steady_time = std::chrono::steady_clock::now();
      completions->Enqueue(this);
    } else {
      delete this;
    }
  }
  mutable std::mutex mutex;
  opentelemetry::nostd::shared_ptr<ot::Span> span;
  ContextPtr parent;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  const char* name;
  std::unique_ptr<DeferredSpan> attributes;
  ot::SpanKind kind = ot::SpanKind::kInternal;
  const char* operation = nullptr;
  const char* format = nullptr;
  ot::SpanContext link = ot::SpanContext::GetInvalid();
  bool finished = false;
  bool root = false;
  // Publishes the immutable span pointer (or a disabled decision) once Start
  // completes. Keep this beside the other flags to reuse their padding.
  std::atomic<bool> started{false};
  void Start() noexcept;
  ~SpanState() override {
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
  std::shared_ptr<TraceCompletionQueue> completions;
};
namespace {
ot::SpanContext Parent(const ContextPtr& context) noexcept {
  if (!context)
    return ot::SpanContext::GetInvalid();
  if (!context->operation)
    return context->parent;
  auto& op = context->operation;
  op->Start();
  if (!op->started.load(std::memory_order_acquire))
    return ot::SpanContext::GetInvalid();
  return op->span ? op->span->GetContext() : Parent(op->parent);
}
void SpanState::Start() noexcept {
  try {
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
    options.kind = kind;
    if (attributes) {
      span = config->tracer->StartSpan(attributes->name, attributes->attributes, *attributes, options);
    } else {
      std::array<std::pair<opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue>, 2> values;
      size_t count = 0;
      if (operation)
        values[count++] = {"storage.operation", operation};
      if (format)
        values[count++] = {"storage.format", format};
      auto initial = opentelemetry::nostd::span<const decltype(values)::value_type>(values.data(), count);
      if (link.IsValid()) {
        span = config->tracer->StartSpan(name, initial, {{link, {}}}, options);
      } else {
        span = config->tracer->StartSpan(name, initial, options);
      }
    }
    started.store(true, std::memory_order_release);
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Create, &error);
    // No partially initialized span is published by this path.
  } catch (...) {
    RecordFailure(TraceFailure::Create);
    // No partially initialized span is published by this path.
  }
}

struct Data final : folly::RequestData {
  explicit Data(ContextPtr value) : context(std::move(value)) {}
  bool hasCallback() override { return false; }
  const ContextPtr context;
};
}  // namespace
void RecordFailure(TraceFailure failure, const std::exception* error) noexcept {
  const auto index = static_cast<size_t>(failure);
  failures[index].fetch_add(1, std::memory_order_relaxed);
  if (error) {
    if (dynamic_cast<const std::bad_alloc*>(error))
      allocation_failures[index].fetch_add(1, std::memory_order_relaxed);
    else
      standard_exceptions[index].fetch_add(1, std::memory_order_relaxed);
  }
}
TraceFailures GetTraceFailures() noexcept {
  TraceFailures result;
  for (size_t i = 0; i < result.counts.size(); ++i) {
    result.counts[i] = failures[i].load(std::memory_order_relaxed);
    result.allocation_failures[i] = allocation_failures[i].load(std::memory_order_relaxed);
    result.standard_exceptions[i] = standard_exceptions[i].load(std::memory_order_relaxed);
  }
  return result;
}
ContextPtr Capture() noexcept {
  if (!contexts_seen.load(std::memory_order_relaxed) || failed_scopes.load(std::memory_order_acquire))
    return nullptr;
  const auto* request = folly::RequestContext::try_get();
  const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
  return data ? data->context : nullptr;
}
bool HasContext() noexcept {
  if (!contexts_seen.load(std::memory_order_relaxed) || failed_scopes.load(std::memory_order_acquire))
    return false;
  const auto* request = folly::RequestContext::try_get();
  const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
  return data && data->context;
}
void StartCurrent() noexcept {
  auto context = Capture();
  if (context && context->operation)
    context->operation->Start();
}
ContextScope::ContextScope(ContextPtr context) noexcept {
  if (!context && !contexts_seen.load(std::memory_order_relaxed))
    return;
  try {
    FIU_DO_ON(FIUKEY_TRACING_CONTEXT_ATTACH_FAIL, {
      RecordFailure(TraceFailure::Attach);
      failed_scopes.fetch_add(1, std::memory_order_acq_rel);
      failed_ = true;
      return;
    });
    const auto* request = folly::RequestContext::try_get();
    const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
    if (context || (data && data->context))
      scope_.emplace(storage_key, std::make_unique<Data>(std::move(context)));
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Attach, &error);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  } catch (...) {
    RecordFailure(TraceFailure::Attach);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  }
}
ContextScope::~ContextScope() {
  scope_.reset();
  if (failed_)
    failed_scopes.fetch_sub(1, std::memory_order_acq_rel);
}

struct TraceScope::Impl {
  explicit Impl(ContextPtr context) : scope(std::move(context)) {}
  Impl(opentelemetry::nostd::string_view name, Attributes attributes, Links links, ot::SpanKind kind)
      : trace(OperationTrace::WithAttributes(name, attributes, links, kind)), scope(trace.context()) {}
  OperationTrace trace;
  ContextScope scope;
  int exceptions = std::uncaught_exceptions();
};
TraceParent::TraceParent(const ot::SpanContext& upstream)
    : trace_flags(upstream.trace_flags().flags()),
      tracestate(upstream.trace_state()->ToHeader()),
      is_remote(upstream.IsRemote()) {
  const auto trace = upstream.trace_id();
  const auto span = upstream.span_id();
  std::copy(trace.Id().begin(), trace.Id().end(), trace_id.begin());
  std::copy(span.Id().begin(), span.Id().end(), span_id.begin());
}
TraceScope::TraceScope(const TraceParent& parent, std::shared_ptr<TraceCompletionQueue> completions) noexcept {
  contexts_seen.store(true, std::memory_order_relaxed);
  try {
    impl_ = std::make_unique<Impl>(std::make_shared<Context>(Context{
        ot::SpanContext(ot::TraceId(parent.trace_id), ot::SpanId(parent.span_id), ot::TraceFlags(parent.trace_flags),
                        parent.is_remote, ot::TraceState::FromHeader(parent.tracestate)),
        nullptr, false, std::move(completions)}));
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Attach, &error);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  } catch (...) {
    RecordFailure(TraceFailure::Attach);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  }
}
TraceScope::TraceScope(opentelemetry::nostd::string_view name,
                       Attributes attributes,
                       Links links,
                       ot::SpanKind kind) noexcept {
  if (!HasContext())
    return;
  try {
    impl_ = std::make_unique<Impl>(name, attributes, links, kind);
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Attach, &error);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  } catch (...) {
    RecordFailure(TraceFailure::Attach);
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  }
}
TraceScope::~TraceScope() noexcept {
  if (impl_) {
    if (std::uncaught_exceptions() > impl_->exceptions)
      impl_->trace.FinishException();
    else
      impl_->trace.FinishScope();
  }
  impl_.reset();
  if (failed_)
    failed_scopes.fetch_sub(1, std::memory_order_acq_rel);
}
void TraceScope::Finish(const arrow::Status& status) noexcept {
  if (impl_)
    impl_->trace.Finish(status);
  impl_.reset();
  if (failed_) {
    failed_ = false;
    failed_scopes.fetch_sub(1, std::memory_order_acq_rel);
  }
}
void TraceScope::SetAttribute(opentelemetry::nostd::string_view key,
                              const opentelemetry::common::AttributeValue& value) noexcept {
  if (impl_)
    impl_->trace.Attribute(key, value);
}
TraceScope AttachParent(const TraceParent& parent, std::shared_ptr<TraceCompletionQueue> completions) noexcept {
  return TraceScope(parent, std::move(completions));
}
arrow::Status SetTracerProvider(ProviderPtr provider) noexcept {
  try {
    FIU_RETURN_ON(FIUKEY_TRACING_CONFIGURATION_FAIL, arrow::Status::UnknownError("tracing configuration failure"));
    auto tracer = provider ? provider->GetTracer("milvus-storage", MILVUS_STORAGE_VERSION) : nullptr;
    std::lock_guard<std::mutex> lock(configuration_mutex);
    auto next = std::make_shared<Configuration>(*configuration);
    next->provider = std::move(provider);
    next->tracer = std::move(tracer);
    configuration = std::move(next);
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    return detail::ExceptionStatus("Failed to configure Storage tracer provider", &e);
  } catch (...) {
    return detail::ExceptionStatus("Failed to configure Storage tracer provider");
  }
}
arrow::Status SetTraceOptions(const TraceOptions& options) noexcept {
  try {
    FIU_RETURN_ON(FIUKEY_TRACING_CONFIGURATION_FAIL, arrow::Status::UnknownError("tracing configuration failure"));
    std::lock_guard<std::mutex> lock(configuration_mutex);
    auto next = std::make_shared<Configuration>(*configuration);
    next->options = options;
    configuration = std::move(next);
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    return detail::ExceptionStatus("Failed to configure Storage trace options", &e);
  } catch (...) {
    return detail::ExceptionStatus("Failed to configure Storage trace options");
  }
}
OperationTrace::OperationTrace(const char* name,
                               bool lazy,
                               bool io,
                               ot::SpanContext link,
                               const char* operation,
                               const char* format,
                               bool native_completion) noexcept {
  try {
    FIU_DO_ON(FIUKEY_TRACING_SCOPE_FAIL, {
      RecordFailure(TraceFailure::Create);
      return;
    });
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
      if (native_completion && !context_->completions) {
        RecordFailure(TraceFailure::MissingCompletionQueue);
        return;
      }
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
      if (native_completion && !context_->completions) {
        RecordFailure(TraceFailure::MissingCompletionQueue);
        return;
      }
      budget = std::make_shared<Budget>();
    }
    auto state = std::shared_ptr<SpanState>(new SpanState, [](SpanState* state) { state->Release(); });
    if (native_completion)
      state->completions = context_->completions;
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
    context_ = std::make_shared<Context>(
        Context{ot::SpanContext::GetInvalid(), std::move(state), false, context_->completions});
    owns_state_ = true;
    if (!lazy)
      Start();
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Create, &error);
    context_.reset();
    owns_state_ = false;
  } catch (...) {
    RecordFailure(TraceFailure::Create);
    context_.reset();
    owns_state_ = false;
  }
}
OperationTrace OperationTrace::WithAttributes(opentelemetry::nostd::string_view name,
                                              TraceScope::Attributes attributes,
                                              TraceScope::Links links,
                                              ot::SpanKind kind,
                                              bool lazy) noexcept {
  OperationTrace trace("storage.custom", true);
  if (!trace.owns_state_)
    return trace;
  try {
    auto& state = trace.context_->operation;
    state->attributes = std::make_unique<DeferredSpan>(name, attributes, links);
    state->name = state->attributes->name.c_str();
    state->kind = kind;
    if (!lazy)
      trace.Start();
    return trace;
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Attributes, &error);
    return OperationTrace{};
  } catch (...) {
    RecordFailure(TraceFailure::Attributes);
    return OperationTrace{};
  }
}
bool OperationTrace::IsEnabled() const { return context_ && context_->operation; }
void OperationTrace::AccountRead(int64_t requested, int64_t returned) const noexcept {
  if (!context_ || !context_->operation)
    return;
  auto budget = context_->operation->budget;
  budget->reads.fetch_add(1, std::memory_order_relaxed);
  budget->requested_bytes.fetch_add(std::max<int64_t>(0, requested), std::memory_order_relaxed);
  budget->returned_bytes.fetch_add(std::max<int64_t>(0, returned), std::memory_order_relaxed);
}
void OperationTrace::Start() const noexcept {
  if (owns_state_)
    context_->operation->Start();
}
void OperationTrace::Finish(const arrow::Status& status) const noexcept { FinishCode(status.code(), status.detail()); }
void OperationTrace::FinishCode(arrow::StatusCode code, std::shared_ptr<arrow::StatusDetail> detail) const noexcept {
  FinishImpl(code, std::move(detail));
}
void OperationTrace::FinishException() const noexcept { FinishImpl(arrow::StatusCode::UnknownError, nullptr, true); }
void OperationTrace::FinishScope() const noexcept {
  if (owns_state_ && context_->operation->started.load(std::memory_order_acquire))
    FinishImpl(arrow::StatusCode::OK, nullptr);
}
void OperationTrace::FinishImpl(arrow::StatusCode code,
                                std::shared_ptr<arrow::StatusDetail> detail,
                                bool exception) const noexcept {
  try {
    if (!owns_state_)
      return;
    auto op = context_->operation;
    op->Start();
    if (!op->started.load(std::memory_order_acquire))
      return;
    {
      std::lock_guard<std::mutex> lock(op->mutex);
      if (op->finished)
        return;
      op->finished = true;
    }
    if (!op->span)
      return;
    op->end_options.end_steady_time = std::chrono::steady_clock::now();
    // Arrow Status copies allocate. Snapshot only the code and shared detail;
    // classification strings are built later on the host's drain thread.
    op->pending_code = code;
    op->pending_detail = std::dynamic_pointer_cast<ExtendStatusDetail>(std::move(detail));
    op->pending_exception = exception;
    if (op->completions) {
      op->pending_owner = op;
      op->completions->Enqueue(op.get());
    } else {
      op->Complete();
    }
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Complete, &error);
  } catch (...) {
    RecordFailure(TraceFailure::Complete);
  }
}
void SpanState::Complete() noexcept {
  try {
    try {
      if (unobserved)
        span->SetAttribute("storage.completion.unobserved", true);
      if (pending_exception || pending_code != arrow::StatusCode::OK) {
        span->SetStatus(ot::StatusCode::kError);
        if (span->IsRecording()) {
          if (pending_exception) {
            span->SetAttribute("error.type", "UnknownError");
          } else if (pending_detail) {
            span->SetAttribute("error.type", pending_detail->CodeAsString());
            span->SetAttribute("error.retryable", pending_detail->retryable());
          } else {
            span->SetAttribute("error.type", arrow::Status::CodeAsString(pending_code));
          }
        }
      }
      if (root) {
        span->SetAttribute("storage.spans.dropped", budget->dropped.load());
        span->SetAttribute("storage.io.reads", budget->reads.load());
        span->SetAttribute("storage.io.requested_bytes", budget->requested_bytes.load());
        span->SetAttribute("storage.io.returned_bytes", budget->returned_bytes.load());
      }
    } catch (const std::exception& error) {
      RecordFailure(TraceFailure::Attributes, &error);
      // Annotation failure must not prevent span completion.
    } catch (...) {
      RecordFailure(TraceFailure::Attributes);
      // Annotation failure must not prevent span completion.
    }
    span->End(end_options);
  } catch (const std::exception& error) {
    RecordFailure(TraceFailure::Complete, &error);
  } catch (...) {
    RecordFailure(TraceFailure::Complete);
  }
}

void OperationTrace::Attribute(const char* key, int64_t value) const noexcept {
  Attribute(opentelemetry::nostd::string_view(key), opentelemetry::common::AttributeValue(value));
}
void OperationTrace::Attribute(const char* key, const char* value) const noexcept {
  Attribute(opentelemetry::nostd::string_view(key), opentelemetry::common::AttributeValue(value));
}
void OperationTrace::Attribute(opentelemetry::nostd::string_view key,
                               const opentelemetry::common::AttributeValue& value) const noexcept {
  Start();
  if (owns_state_ && context_->operation->started.load(std::memory_order_acquire) && context_->operation->span)
    context_->operation->span->SetAttribute(key, value);
}
ot::SpanContext OperationTrace::span_context() const noexcept { return Parent(context_); }
}  // namespace milvus_storage::tracing
