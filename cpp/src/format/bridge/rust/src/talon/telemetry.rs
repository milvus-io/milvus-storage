// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
//! Talon SDK spans use the Storage request's C++ provider. There is no Rust
//! exporter, global subscriber, or second host initialization API.
use super::ffi;
use opentelemetry::trace::{
    SpanBuilder, SpanContext, SpanId, SpanKind, Status, TraceContextExt, TraceFlags, TraceId,
    Tracer, noop::NoopSpan,
};
use opentelemetry::{Context, KeyValue, Value};
use std::{
    future::Future,
    sync::{LazyLock, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};
use talon::{RequestOptions, TraceContext, TraceParent};
use tracing::instrument::WithSubscriber;
use tracing_opentelemetry::{OtelData, PreSampledTracer};
use tracing_subscriber::{Layer, layer::SubscriberExt};

// These C++ snapshots are immutable, own their provider, and expose only
// thread-safe OTel span operations. No thread-local attachment crosses Rust.
unsafe impl Send for ffi::TalonTrace {}
unsafe impl Sync for ffi::TalonTrace {}
unsafe impl Send for ffi::TalonSpan {}
unsafe impl Sync for ffi::TalonSpan {}

pub(super) fn initialize_storage_talon_tracing() -> anyhow::Result<()> {
    static INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| {
            // Protocol capability is a deployment setting, never learned by retrying
            // a failed v2 RPC as v1. SDK recording itself follows Storage's provider.
            let v2_endpoints = std::env::var("TALON_TELEMETRY_V2_ENDPOINTS")
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            talon_telemetry::configure(talon_telemetry::Config {
                mode: talon_telemetry::Mode::Standard,
                root_sample_ratio: 0.0,
                max_spans_per_request: 4096,
                v2_endpoints,
                ..Default::default()
            })
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(super) struct RequestTrace {
    pub parent: Option<TraceContext>,
    dispatch: Option<tracing::Dispatch>,
}
impl RequestTrace {
    pub fn capture() -> Self {
        let trace = ffi::capture_talon_trace();
        let parent = trace
            .as_ref()
            .and_then(|trace| ffi::talon_trace_parent(trace).ok())
            .and_then(|carrier| {
                TraceContext::from_otel(&Context::new().with_remote_span_context(context(&carrier)))
            });
        let dispatch = parent.as_ref().map(|_| {
            tracing::Dispatch::new(
                tracing_subscriber::registry().with(
                    tracing_opentelemetry::layer()
                        .with_tracer(StorageTracer(trace))
                        .with_location(false)
                        .with_threads(false)
                        .with_tracked_inactivity(false)
                        .with_filter(tracing_subscriber::filter::filter_fn(|m| {
                            m.target() == "talon_telemetry"
                        })),
                ),
            )
        });
        Self { parent, dispatch }
    }
    pub async fn run<F: Future>(self, future: F) -> F::Output {
        static DISABLED: LazyLock<talon_telemetry::Operation> =
            LazyLock::new(talon_telemetry::Operation::disabled);
        match self.dispatch {
            Some(dispatch) => future.with_subscriber(dispatch).await,
            None => DISABLED.scope(future).await,
        }
    }
}
pub(super) fn options(parent: Option<&TraceContext>) -> RequestOptions<'_> {
    RequestOptions {
        parent: parent
            .map(TraceParent::Explicit)
            .unwrap_or(TraceParent::Root),
    }
}

fn nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(i64::MAX as u128) as u64
}
fn context(c: &ffi::TraceCarrier) -> SpanContext {
    SpanContext::new(
        TraceId::from_bytes(c.trace_id),
        SpanId::from_bytes(c.span_id),
        TraceFlags::new(c.flags),
        c.remote,
        c.state.parse().unwrap_or_default(),
    )
}
fn carrier(c: &SpanContext) -> ffi::TraceCarrier {
    ffi::TraceCarrier {
        trace_id: c.trace_id().to_bytes(),
        span_id: c.span_id().to_bytes(),
        flags: c.trace_flags().to_u8(),
        remote: c.is_remote(),
        state: c.trace_state().header(),
    }
}
fn attributes(values: &[KeyValue]) -> Vec<ffi::SpanAttribute> {
    values
        .iter()
        .filter_map(|kv| {
            let mut value = ffi::SpanAttribute {
                key: kv.key.to_string(),
                kind: 0,
                boolean: false,
                integer: 0,
                floating: 0.0,
                text: String::new(),
            };
            match &kv.value {
                Value::Bool(v) => value.boolean = *v,
                Value::I64(v) => {
                    value.kind = 1;
                    value.integer = *v;
                }
                Value::F64(v) => {
                    value.kind = 2;
                    value.floating = *v;
                }
                Value::String(v) => {
                    value.kind = 3;
                    value.text = v.to_string();
                }
                // Talon's bounded recording API emits scalar attributes only.
                _ => return None,
            }
            Some(value)
        })
        .collect()
}

struct StorageTracer(cxx::SharedPtr<ffi::TalonTrace>);
#[derive(Clone)]
struct StartedSpan {
    span: cxx::SharedPtr<ffi::TalonSpan>,
    context: SpanContext,
}
impl StorageTracer {
    fn start(&self, builder: &SpanBuilder, parent: &Context) -> StartedSpan {
        let native = ffi::start_talon_span(
            &self.0,
            &builder.name,
            u8::from(builder.span_kind == Some(SpanKind::Client)),
            &carrier(parent.span().span_context()),
            nanos(builder.start_time.unwrap_or_else(SystemTime::now)),
            &attributes(builder.attributes.as_deref().unwrap_or_default()),
        )
        .unwrap_or_else(|_| cxx::SharedPtr::null());
        let sc = native
            .as_ref()
            .and_then(|span| ffi::talon_span_context(span).ok())
            .map(|c| context(&c))
            .unwrap_or_else(|| parent.span().span_context().clone());
        StartedSpan {
            span: native,
            context: sc,
        }
    }
    fn cached<'a>(&self, builder: &SpanBuilder, parent: &'a Context) -> Option<&'a StartedSpan> {
        parent
            .get::<StartedSpan>()
            .filter(|s| Some(s.context.span_id()) == builder.span_id)
    }
}
impl PreSampledTracer for StorageTracer {
    fn sampled_context(&self, data: &mut OtelData) -> Context {
        let started = self
            .cached(&data.builder, &data.parent_cx)
            .cloned()
            .unwrap_or_else(|| self.start(&data.builder, &data.parent_cx));
        data.builder.span_id = Some(started.context.span_id());
        data.builder.trace_id = Some(started.context.trace_id());
        let sc = started.context.clone();
        data.parent_cx = data.parent_cx.with_value(started);
        data.parent_cx.with_remote_span_context(sc)
    }
    // C++ assigns IDs and samples when sampled_context starts the actual span.
    fn new_trace_id(&self) -> TraceId {
        TraceId::INVALID
    }
    fn new_span_id(&self) -> SpanId {
        SpanId::INVALID
    }
}
impl Tracer for StorageTracer {
    type Span = NoopSpan;
    fn build_with_context(&self, builder: SpanBuilder, parent: &Context) -> Self::Span {
        let started = self
            .cached(&builder, parent)
            .cloned()
            .unwrap_or_else(|| self.start(&builder, parent));
        if let Some(span) = started.span.as_ref() {
            let status = match builder.status {
                Status::Unset => 0,
                Status::Ok => 1,
                Status::Error { .. } => 2,
            };
            let _ = ffi::finish_talon_span(
                span,
                &builder.name,
                status,
                nanos(builder.end_time.unwrap_or_else(SystemTime::now)),
                &attributes(builder.attributes.as_deref().unwrap_or_default()),
            );
        }
        // The layer drops this return value immediately; the C++ span has
        // already ended, and sampled_context supplied its identity earlier.
        NoopSpan::DEFAULT
    }
}
