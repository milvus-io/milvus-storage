use super::*;
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use std::sync::{
    atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicU8, AtomicUsize, Ordering},
    Arc,
};
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub(crate) struct Budget {
    remaining: AtomicUsize,
    omitted: AtomicUsize,
    pub diagnostic: bool,
    pub cache_tiers: AtomicU8,
    totals: std::sync::Mutex<[u64; 9]>,
    pub dependencies: std::sync::Mutex<Vec<[u8; 24]>>,
    pub dependencies_complete: AtomicBool,
}
pub(crate) struct Recording {
    pub span: tracing::Span,
    pub dispatch: tracing::Dispatch,
    pub budget: Arc<Budget>,
    pub finished: AtomicBool,
    pub terminal: Arc<AtomicU8>,
    parent_terminal: Option<Arc<AtomicU8>>,
    name: &'static str,
    summary: bool,
    local: [AtomicU64; 10],
    recorded: AtomicU16,
    attributes: std::sync::Mutex<Vec<opentelemetry::KeyValue>>,
}
// Leave room in the production 48-attribute limit for tracing metadata,
// counters, outcome and completeness. Never buffer one entry per chunk/page.
const MAX_DETAIL_ATTRIBUTES: usize = 23;
impl Recording {
    pub fn attribute(&self, key: &'static str, value: u64) {
        if let Some(i) = LOCAL_KEYS.iter().position(|k| *k == key) {
            self.local[i].store(value, Ordering::Relaxed);
            self.recorded.fetch_or(1 << i, Ordering::Relaxed);
            return;
        }
        self.detail(key, (value.min(i64::MAX as u64) as i64).into());
    }
    fn detail(&self, key: &'static str, value: opentelemetry::Value) {
        let mut attributes = self.attributes.lock().unwrap();
        if let Some(attribute) = attributes.iter_mut().find(|a| a.key.as_str() == key) {
            attribute.value = value;
        } else if attributes.len() < MAX_DETAIL_ATTRIBUTES {
            attributes.push(opentelemetry::KeyValue::new(key, value));
        } else {
            self.budget.omitted.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn text(&self, key: &'static str, value: &str) {
        if self.name == "HTTP attempt" && key == "http.request.method" {
            if value == "GET" {
                self.local[4].store(1, Ordering::Relaxed);
            }
            if value == "HEAD" {
                self.local[5].store(1, Ordering::Relaxed);
            }
        }
        if self.name == "talon.refill" && key == "talon.outcome" {
            let i = match value {
                "success" => 1,
                "cancelled" => 3,
                _ => 2,
            };
            self.local[i].store(1, Ordering::Relaxed);
        }
        let mut end = value.len().min(256);
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        if key == "talon.outcome" {
            // Operation::outcome (or cancellation in Drop) writes this once.
            // Keep the terminal state outside the optional detail budget.
            self.span.set_attribute(key, value[..end].to_owned());
        } else {
            self.detail(key, value[..end].to_owned().into());
        }
    }
}
const STAT_KEYS: [&str; 9] = [
    "talon.refill.started",
    "talon.refill.completed",
    "talon.refill.failed",
    "talon.refill.cancelled",
    "talon.origin.get.attempts",
    "talon.origin.head.attempts",
    "talon.origin.body_bytes",
    "talon.origin.validated_bytes",
    "talon.cache.committed_bytes",
];
const LOCAL_KEYS: [&str; 10] = [
    STAT_KEYS[0],
    STAT_KEYS[1],
    STAT_KEYS[2],
    STAT_KEYS[3],
    STAT_KEYS[4],
    STAT_KEYS[5],
    STAT_KEYS[6],
    STAT_KEYS[7],
    STAT_KEYS[8],
    "talon.response.bytes",
];
impl Drop for Recording {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Relaxed) {
            let timeout = self
                .parent_terminal
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Relaxed) == 2);
            self.text(
                "talon.outcome",
                if timeout { "timeout" } else { "cancelled" },
            );
            if timeout {
                self.span
                    .set_status(opentelemetry::trace::Status::error("timeout"));
            }
        }
        if self.name == "talon.refill" {
            self.local[0].store(1, Ordering::Relaxed);
        }
        let omitted = self.budget.omitted.load(Ordering::Relaxed);
        self.span
            .set_attribute("talon.details.omitted", omitted as i64);
        self.span.set_attribute(
            "talon.details.complete",
            omitted == 0
                && self.finished.load(Ordering::Relaxed)
                && (!self.summary || Arc::strong_count(&self.budget) == 1),
        );
        let mut totals = self.budget.totals.lock().unwrap();
        for (i, total) in totals.iter_mut().enumerate() {
            *total = total.saturating_add(self.local[i].load(Ordering::Relaxed));
        }
        if self.summary {
            let tier = match self.budget.cache_tiers.load(Ordering::Relaxed) {
                0 => "unknown",
                1 => "l1",
                2 => "l2",
                4 => "origin",
                _ => "mixed",
            };
            self.span.set_attribute("talon.cache.tier", tier);
            for (i, key) in STAT_KEYS.iter().enumerate() {
                self.span
                    .set_attribute(*key, totals[i].min(i64::MAX as u64) as i64);
            }
            self.span.set_attribute(
                "talon.refill.shared_dependencies",
                self.budget.dependencies.lock().unwrap().len() as i64,
            );
            self.span.set_attribute(
                "talon.dependencies.complete",
                self.budget.dependencies_complete.load(Ordering::Relaxed) && omitted == 0,
            );
        }
        let recorded = self.recorded.load(Ordering::Relaxed);
        for (i, key) in LOCAL_KEYS.iter().enumerate() {
            if recorded & (1 << i) != 0 && (!self.summary || i >= STAT_KEYS.len()) {
                self.span.set_attribute(
                    *key,
                    self.local[i].load(Ordering::Relaxed).min(i64::MAX as u64) as i64,
                );
            }
        }
        for attribute in self.attributes.get_mut().unwrap().drain(..) {
            self.span.set_attribute(attribute.key, attribute.value);
        }
    }
}
impl TraceContext {
    pub fn from_otel(context: &opentelemetry::Context) -> Option<Self> {
        let span = context.span();
        let c = span.span_context();
        if !c.is_valid() {
            return None;
        }
        Self::from_w3c(
            &format!(
                "00-{}-{}-{:02x}",
                c.trace_id(),
                c.span_id(),
                c.trace_flags().to_u8()
            ),
            Some(&c.trace_state().header()),
        )
    }
    pub fn to_otel(&self) -> opentelemetry::Context {
        let p = self.traceparent();
        opentelemetry::Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex(&p[3..35]).unwrap(),
            SpanId::from_hex(&p[36..52]).unwrap(),
            TraceFlags::new(u8::from_str_radix(&p[53..55], 16).unwrap()),
            true,
            self.tracestate().parse::<TraceState>().unwrap_or_default(),
        ))
    }
}
pub fn ambient() -> Option<TraceContext> {
    TraceContext::from_otel(&tracing::Span::current().context())
        .or_else(|| TraceContext::from_otel(&opentelemetry::Context::current()))
}
pub fn start(op: &mut Operation, name: &'static str, kind: &'static str, inherit: bool) {
    let config = CONFIG.get().unwrap();
    if !matches!(config.mode, Mode::Standard | Mode::Diagnostic)
        || op.carrier.as_ref().is_some_and(|c| !c.sampled())
    {
        return;
    }
    let parent_budget = if inherit && CURRENT.is_set() {
        CURRENT.with(|o| o.recording.as_ref().map(|r| r.budget.clone()))
    } else {
        None
    };
    // Never resample an inherited non-recording operation.
    if inherit && CURRENT.is_set() && parent_budget.is_none() {
        return;
    }
    // Decide root sampling before allocating recording state or creating a span.
    if op.carrier.is_none() {
        use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator};
        let generator = RandomIdGenerator::default();
        let id = generator.new_trace_id().to_bytes();
        let fraction = u64::from_be_bytes(id[..8].try_into().unwrap()) as f64 / u64::MAX as f64;
        if config.root_sample_ratio == 0.0 || fraction >= config.root_sample_ratio {
            op.carrier = Some(TraceContext::from_ids(
                id,
                generator.new_span_id().to_bytes(),
                0,
            ));
            return;
        }
    }
    let summary = parent_budget.is_none();
    let budget = match parent_budget {
        Some(b) => {
            if b.remaining
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
                .is_err()
            {
                b.omitted.fetch_add(1, Ordering::Relaxed);
                return;
            }
            b
        }
        None => Arc::new(Budget {
            cache_tiers: AtomicU8::new(0),
            totals: std::sync::Mutex::new([0; 9]),
            dependencies: std::sync::Mutex::new(Vec::new()),
            dependencies_complete: AtomicBool::new(true),
            remaining: AtomicUsize::new(config.max_spans_per_request - 1),
            omitted: AtomicUsize::new(0),
            diagnostic: config.mode == Mode::Diagnostic
                && STARTED
                    .get()
                    .is_some_and(|s| s.elapsed() < config.diagnostic_duration)
                && DIAGNOSTIC_REMAINING
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
                    .is_ok(),
        }),
    };
    let span = tracing::info_span!(target: "talon_telemetry", parent: None, "talon.operation", otel.name = name, otel.kind = kind, trace_id = tracing::field::Empty, span_id = tracing::field::Empty);
    span.set_parent(
        op.carrier
            .as_ref()
            .map(TraceContext::to_otel)
            .unwrap_or_default(),
    );
    let carrier = TraceContext::from_otel(&span.context());
    if carrier.is_none() {
        return;
    }
    op.carrier = carrier;
    if !op.carrier.as_ref().is_some_and(TraceContext::sampled) {
        return;
    }
    if op.read_id.is_none() {
        use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator};
        op.read_id = Some(RandomIdGenerator::default().new_trace_id().to_bytes());
    }
    if let Some(c) = op.carrier() {
        span.record("trace_id", &c.traceparent()[3..35]);
        span.record("span_id", &c.traceparent()[36..52]);
    }
    op.recording = Some(Recording {
        span,
        dispatch: tracing::dispatcher::get_default(Clone::clone),
        budget,
        finished: AtomicBool::new(false),
        terminal: Arc::new(AtomicU8::new(0)),
        parent_terminal: if inherit && CURRENT.is_set() {
            CURRENT.with(|o| o.recording.as_ref().map(|r| r.terminal.clone()))
        } else {
            None
        },
        name,
        summary,
        local: std::array::from_fn(|_| AtomicU64::new(0)),
        recorded: AtomicU16::new(0),
        attributes: std::sync::Mutex::new(Vec::new()),
    });
    op.record_read_id();
    if name == "talon.refill" {
        if let Some(c) = op.carrier() {
            op.text("talon.refill.id", &c.traceparent()[3..52]);
        }
    }
}
