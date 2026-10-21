//! Request-local tracing with allocation-free propagation and optional export.
mod context;
pub use context::{RequestOptions, TraceContext, TraceParent};
#[cfg(feature = "export")]
pub mod export;
#[cfg(feature = "export")]
mod export_health;

/// Export health is additive to existing registries; no request IDs as labels.
pub fn metrics() -> String {
    #[cfg(feature = "export")]
    {
        export_health::render()
    }
    #[cfg(not(feature = "export"))]
    {
        String::new()
    }
}
#[cfg(feature = "recording")]
mod recording;

use std::future::Future;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Initial releases are off until deployment-specific performance acceptance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Propagate,
    Standard,
    Diagnostic,
}

/// Process policy, installed explicitly by a binary or host before serving.
#[derive(Debug, Clone)]
pub struct Config {
    pub mode: Mode,
    pub root_sample_ratio: f64,
    pub max_spans_per_request: usize,
    pub diagnostic_duration: Duration,
    pub diagnostic_requests: usize,
    /// Explicit endpoint capability allowlist; never learned from timeouts.
    pub v2_endpoints: Vec<String>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Off,
            root_sample_ratio: 0.01,
            max_spans_per_request: 256,
            diagnostic_duration: Duration::ZERO,
            diagnostic_requests: 0,
            v2_endpoints: Vec::new(),
        }
    }
}
impl Config {
    pub fn from_env() -> Result<Self, String> {
        let mode = match std::env::var("TALON_TELEMETRY_MODE")
            .as_deref()
            .unwrap_or("off")
        {
            "off" => Mode::Off,
            "propagate" => Mode::Propagate,
            "standard" => Mode::Standard,
            "diagnostic" => Mode::Diagnostic,
            _ => return Err("invalid TALON_TELEMETRY_MODE".into()),
        };
        let mut c = Self {
            mode,
            ..Self::default()
        };
        if let Ok(v) = std::env::var("TALON_TELEMETRY_SAMPLE_RATIO") {
            c.root_sample_ratio = v.parse().map_err(|_| "invalid sample ratio")?;
        }
        if let Ok(v) = std::env::var("TALON_TELEMETRY_MAX_SPANS") {
            c.max_spans_per_request = v.parse().map_err(|_| "invalid span budget")?;
        }
        if let Ok(v) = std::env::var("TALON_TELEMETRY_DIAGNOSTIC_SECONDS") {
            c.diagnostic_duration =
                Duration::from_secs(v.parse().map_err(|_| "invalid diagnostic duration")?);
        }
        if let Ok(v) = std::env::var("TALON_TELEMETRY_DIAGNOSTIC_REQUESTS") {
            c.diagnostic_requests = v.parse().map_err(|_| "invalid diagnostic budget")?;
        }
        c.v2_endpoints = std::env::var("TALON_TELEMETRY_V2_ENDPOINTS")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.root_sample_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.root_sample_ratio)
            || self.max_spans_per_request == 0
            || self.max_spans_per_request > 4096
        {
            return Err("invalid telemetry sampling or span budget".into());
        }
        if self.mode == Mode::Diagnostic
            && (self.diagnostic_duration.is_zero()
                || self.diagnostic_duration > Duration::from_secs(3600)
                || self.diagnostic_requests == 0)
        {
            return Err(
                "diagnostic requires a duration (at most 3600 seconds) and request budget".into(),
            );
        }
        #[cfg(not(feature = "recording"))]
        if matches!(self.mode, Mode::Standard | Mode::Diagnostic) {
            return Err("telemetry recording feature is not compiled".into());
        }
        Ok(())
    }
}
static CONFIG: OnceLock<Config> = OnceLock::new();
static STARTED: OnceLock<Instant> = OnceLock::new();
static DIAGNOSTIC_REMAINING: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Does not install or replace any subscriber/provider.
pub fn configure(config: Config) -> Result<(), String> {
    config.validate()?;
    let requests = config.diagnostic_requests;
    CONFIG
        .set(config)
        .map_err(|_| "telemetry already configured")?;
    STARTED.get_or_init(Instant::now);
    DIAGNOSTIC_REMAINING.store(requests, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub fn enabled() -> bool {
    CONFIG.get().is_some_and(|c| c.mode != Mode::Off)
        && (!CURRENT.is_set() || CURRENT.with(|op| op.active))
}
pub fn v2_enabled(endpoint: &str) -> bool {
    enabled()
        && CONFIG
            .get()
            .is_some_and(|c| c.v2_endpoints.iter().any(|e| e == endpoint || e == "*"))
}

scoped_tls::scoped_thread_local!(static CURRENT: Operation);

/// Owns only telemetry state. Scope is entered separately on each future poll.
pub struct Operation {
    active: bool,
    carrier: Option<TraceContext>,
    read_id: Option<[u8; 16]>,
    #[cfg(feature = "recording")]
    recording: Option<recording::Recording>,
}
impl Operation {
    pub fn new(name: &'static str, kind: &'static str, parent: TraceParent<'_>) -> Self {
        if !enabled() {
            return Self::disabled();
        }
        let carrier = match parent {
            TraceParent::Explicit(c) => Some(c.clone()),
            TraceParent::Root => None,
            TraceParent::Inherit => current_carrier(),
        };
        let read_id = if matches!(parent, TraceParent::Inherit) && CURRENT.is_set() {
            CURRENT.with(|o| o.read_id)
        } else {
            None
        };
        let mut op = Self {
            active: true,
            carrier,
            read_id,
            #[cfg(feature = "recording")]
            recording: None,
        };
        #[cfg(feature = "recording")]
        recording::start(&mut op, name, kind, matches!(parent, TraceParent::Inherit));
        let _ = (name, kind, &mut op);
        op
    }
    /// A request-local opt-out, even when process telemetry is enabled.
    /// Enter with `scope` or `in_scope` to suppress recording, ambient context
    /// inheritance and v2 propagation for nested operations. The scope is
    /// restored on each future poll, including Pending and unwinding.
    pub fn disabled() -> Self {
        Self {
            active: false,
            carrier: None,
            read_id: None,
            #[cfg(feature = "recording")]
            recording: None,
        }
    }
    pub fn server(carrier: Option<&TraceContext>, read_id: Option<[u8; 16]>) -> Self {
        let mut op = Self::new(
            "talon.rpc",
            "server",
            carrier
                .map(TraceParent::Explicit)
                .unwrap_or(TraceParent::Root),
        );
        if read_id.is_some() {
            op.read_id = read_id;
        }
        op.record_read_id();
        op
    }
    pub fn carrier(&self) -> Option<&TraceContext> {
        self.carrier.as_ref()
    }
    pub fn read_id(&self) -> Option<[u8; 16]> {
        self.read_id
    }
    pub fn is_recording(&self) -> bool {
        #[cfg(feature = "recording")]
        {
            self.recording.is_some()
        }
        #[cfg(not(feature = "recording"))]
        {
            false
        }
    }
    pub fn diagnostic(&self) -> bool {
        #[cfg(feature = "recording")]
        {
            self.recording.as_ref().is_some_and(|r| r.budget.diagnostic)
        }
        #[cfg(not(feature = "recording"))]
        {
            false
        }
    }
    pub fn record(&self, key: &'static str, value: u64) {
        #[cfg(feature = "recording")]
        if let Some(r) = &self.recording {
            r.attribute(key, value);
        }
        let _ = (key, value);
    }
    pub fn text(&self, key: &'static str, value: &str) {
        #[cfg(feature = "recording")]
        if let Some(r) = &self.recording {
            r.text(key, value);
        }
        let _ = (key, value);
    }
    pub fn outcome(&self, value: &'static str) {
        #[cfg(feature = "recording")]
        if let Some(r) = &self.recording {
            if r.finished.swap(true, std::sync::atomic::Ordering::Relaxed) {
                return;
            }
        }
        #[cfg(feature = "recording")]
        if let Some(r) = &self.recording {
            r.terminal.store(
                if value == "timeout" { 2 } else { 1 },
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.text("talon.outcome", value);
        #[cfg(feature = "recording")]
        if matches!(value, "error" | "timeout" | "http_error") {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            if let Some(r) = &self.recording {
                r.span
                    .set_status(opentelemetry::trace::Status::error(value));
            }
        }
    }
    fn record_read_id(&self) {
        if self.is_recording() {
            if let Some(id) = self.read_id {
                self.text("talon.read.id", std::str::from_utf8(&hex_id(id)).unwrap());
            }
        }
    }
    pub fn scope<F: Future>(&self, future: F) -> Scoped<'_, F> {
        Scoped {
            operation: self,
            future,
        }
    }
    pub fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        if !self.active {
            // Keep the never-enabled fast path, but retain an explicit disabled
            // request snapshot after process telemetry has been configured.
            return if CONFIG.get().is_some_and(|c| c.mode != Mode::Off) {
                CURRENT.set(self, f)
            } else {
                f()
            };
        }
        #[cfg(feature = "recording")]
        if let Some(r) = &self.recording {
            return tracing::dispatcher::with_default(&r.dispatch, || {
                let _guard = r.span.enter();
                CURRENT.set(self, f)
            });
        }
        CURRENT.set(self, f)
    }
}
pub fn current_carrier() -> Option<TraceContext> {
    if !enabled() {
        return None;
    }
    if CURRENT.is_set() {
        return CURRENT.with(|o| o.carrier.clone());
    }
    #[cfg(feature = "recording")]
    {
        recording::ambient()
    }
    #[cfg(not(feature = "recording"))]
    {
        None
    }
}
pub fn current_read_id() -> Option<[u8; 16]> {
    if CURRENT.is_set() {
        CURRENT.with(|o| o.read_id)
    } else {
        None
    }
}
/// Size of the scoped tracestate, without copying the carrier. None means a
/// caller outside a Talon scope may inherit an as-yet-unknown host context.
pub fn current_tracestate_len() -> Option<usize> {
    CURRENT
        .is_set()
        .then(|| CURRENT.with(|o| o.carrier.as_ref().map_or(0, |c| c.tracestate().len())))
}
pub fn record(key: &'static str, value: u64) {
    if CURRENT.is_set() {
        CURRENT.with(|o| o.record(key, value));
    }
}
pub fn text(key: &'static str, value: &str) {
    if CURRENT.is_set() {
        CURRENT.with(|o| o.text(key, value));
    }
}
/// Merge observed serving tiers on the request summary without per-page spans.
pub fn cache_tier(tier: &'static str) {
    #[cfg(feature = "recording")]
    if CURRENT.is_set() {
        CURRENT.with(|o| {
            if let Some(r) = &o.recording {
                let bit = match tier {
                    "l1" => 1,
                    "l2" => 2,
                    "origin" => 4,
                    _ => 0,
                };
                r.budget
                    .cache_tiers
                    .fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
            }
        });
    }
    let _ = tier;
}
pub fn diagnostic() -> bool {
    CURRENT.is_set() && CURRENT.with(Operation::diagnostic)
}
pub fn is_recording() -> bool {
    CURRENT.is_set() && CURRENT.with(Operation::is_recording)
}

/// Stable flight reference. Stores context only, never a live span or business guard.
#[derive(Clone, Default)]
pub struct Flight(Option<std::sync::Arc<std::sync::Mutex<Option<TraceContext>>>>);
impl Flight {
    pub fn new() -> Self {
        Self(is_recording().then(|| std::sync::Arc::new(std::sync::Mutex::new(None))))
    }
    pub fn bind_current(&self) {
        if let Some(state) = &self.0 {
            *state.lock().unwrap() = current_carrier();
        }
    }
    pub fn link(&self, operation: &Operation) {
        #[cfg(feature = "recording")]
        {
            use opentelemetry::trace::TraceContextExt;
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            if let Some(r) = &operation.recording {
                let context = self.0.as_ref().and_then(|s| s.lock().unwrap().clone());
                r.budget
                    .dependencies_complete
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                if let Some(context) = context {
                    let otel = context.to_otel();
                    let span_context = otel.span().span_context().clone();
                    r.span.add_link(span_context.clone());
                    let mut id = [0; 24];
                    id[..16].copy_from_slice(&span_context.trace_id().to_bytes());
                    id[16..].copy_from_slice(&span_context.span_id().to_bytes());
                    let mut dependencies = r.budget.dependencies.lock().unwrap();
                    if !dependencies.contains(&id) && dependencies.len() < 32 {
                        dependencies.push(id);
                    }
                    r.text("talon.refill.dependency", &context.traceparent()[3..52]);
                    // One known link cannot certify all generations observed by wait().
                    r.span.set_attribute("talon.dependencies.complete", false);
                } else {
                    r.span.set_attribute("talon.dependencies.complete", false);
                }
            }
        }
        let _ = operation;
    }
}

/// A follower owns its wait; a link points to the original leader's context.
pub async fn wait_for(flight: &Flight, future: impl Future<Output = ()>) {
    if !enabled() {
        return future.await;
    }
    let operation = Operation::new("talon.refill.wait", "internal", TraceParent::Inherit);
    #[cfg(feature = "recording")]
    if let Some(r) = &operation.recording {
        r.budget
            .dependencies_complete
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
    operation.scope(future).await;
    flight.link(&operation);
    operation.outcome("success");
}
fn hex_id(id: [u8; 16]) -> [u8; 32] {
    let mut result = [0; 32];
    let hex = b"0123456789abcdef";
    for (i, byte) in id.into_iter().enumerate() {
        result[2 * i] = hex[(byte >> 4) as usize];
        result[2 * i + 1] = hex[(byte & 15) as usize];
    }
    result
}

pin_project_lite::pin_project! {
    pub struct Scoped<'a, F> { operation: &'a Operation, #[pin] future: F }
}
impl<F: Future> Future for Scoped<'_, F> {
    type Output = F::Output;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        this.operation.in_scope(|| this.future.poll(cx))
    }
}

/// Instrument one existing operation without changing its result/cancellation.
pub async fn observe<T, E>(
    name: &'static str,
    kind: &'static str,
    future: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    if !enabled() {
        return future.await;
    }
    let op = Operation::new(name, kind, TraceParent::Inherit);
    let result = op.scope(future).await;
    op.outcome(if result.is_ok() { "success" } else { "error" });
    result
}

pub fn outcome(value: &'static str) {
    if CURRENT.is_set() {
        CURRENT.with(|o| o.outcome(value));
    }
}

/// Diagnostic application-level filesystem timing, never claimed as device time.
pub fn sync_io<T, E>(name: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    if !diagnostic() {
        return f();
    }
    let operation = Operation::new(name, "internal", TraceParent::Inherit);
    let result = operation.in_scope(f);
    operation.outcome(if result.is_ok() { "success" } else { "error" });
    result
}
