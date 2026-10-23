//! Admission bounds include queued AND exporting spans. This makes drop counts
//! observable without replacing the SDK batching runtime or accessing its internals.
use opentelemetry_sdk::trace::{BatchSpanProcessor, Span, SpanData, SpanExporter, SpanProcessor};
use opentelemetry_sdk::{error::OTelSdkResult, Resource};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static PENDING: AtomicUsize = AtomicUsize::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
pub const CAPACITY: usize = 4096;

pub fn render() -> String {
    format!("# TYPE talon_telemetry_queue_size gauge\ntalon_telemetry_queue_size{{exporter=\"otlp\"}} {}\n# TYPE talon_telemetry_dropped_total counter\ntalon_telemetry_dropped_total{{reason=\"capacity_or_shutdown\"}} {}\n# TYPE talon_telemetry_export_failures_total counter\ntalon_telemetry_export_failures_total{{exporter=\"otlp\",error_class=\"export\"}} {}\n",
        PENDING.load(Ordering::Relaxed), DROPPED.load(Ordering::Relaxed), FAILURES.load(Ordering::Relaxed))
}

#[derive(Debug)]
pub struct BoundedProcessor {
    inner: BatchSpanProcessor,
    closed: std::sync::RwLock<bool>,
}
impl BoundedProcessor {
    pub fn new(inner: BatchSpanProcessor) -> Self {
        Self {
            inner,
            closed: std::sync::RwLock::new(false),
        }
    }
}
impl SpanProcessor for BoundedProcessor {
    fn on_start(&self, span: &mut Span, context: &opentelemetry::Context) {
        self.inner.on_start(span, context);
    }
    fn on_end(&self, span: SpanData) {
        if !span.span_context.is_sampled() {
            return;
        }
        // Producers never wait, including when shutdown closes admission.
        let Ok(closed) = self.closed.try_read() else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if *closed
            || PENDING
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    (n < CAPACITY).then_some(n + 1)
                })
                .is_err()
        {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.inner.on_end(span);
    }
    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }
    fn shutdown(&self) -> OTelSdkResult {
        *self.closed.write().unwrap() = true;
        self.inner.shutdown()
    }
    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

#[derive(Debug)]
pub struct HealthExporter<T>(pub T);
// SDK 0.28 drains only one batch on shutdown. Account for the remainder
// when its worker exits, after every in-progress export has completed.
impl<T> Drop for HealthExporter<T> {
    fn drop(&mut self) {
        DROPPED.fetch_add(PENDING.swap(0, Ordering::Relaxed) as u64, Ordering::Relaxed);
    }
}
impl<T: SpanExporter> SpanExporter for HealthExporter<T> {
    fn export(
        &mut self,
        spans: Vec<SpanData>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = OTelSdkResult> + Send + 'static>> {
        let count = spans.len();
        let result = self.0.export(spans);
        Box::pin(async move {
            let result = result.await;
            PENDING.fetch_sub(count, Ordering::Relaxed);
            if result.is_err() {
                FAILURES.fetch_add(1, Ordering::Relaxed);
            }
            result
        })
    }
    fn shutdown(&mut self) -> OTelSdkResult {
        self.0.shutdown()
    }
    fn set_resource(&mut self, resource: &Resource) {
        self.0.set_resource(resource);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{Tracer, TracerProvider};
    use opentelemetry_sdk::trace::{BatchConfigBuilder, SdkTracerProvider};
    use std::sync::{Arc, Condvar, Mutex};

    #[derive(Debug)]
    struct Slow(Arc<(Mutex<bool>, Condvar)>);
    impl SpanExporter for Slow {
        fn export(
            &mut self,
            _: Vec<SpanData>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = OTelSdkResult> + Send>> {
            let gate = self.0.clone();
            Box::pin(async move {
                let (lock, changed) = &*gate;
                let mut ready = lock.lock().unwrap();
                while !*ready {
                    ready = changed.wait(ready).unwrap();
                }
                Err(opentelemetry_sdk::error::OTelSdkError::InternalFailure(
                    "collector unavailable".into(),
                ))
            })
        }
    }

    #[test]
    fn slow_or_failed_exporter_cannot_block_producers_or_grow_queue() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let batch = BatchSpanProcessor::builder(HealthExporter(Slow(gate.clone())))
            .with_batch_config(
                BatchConfigBuilder::default()
                    .with_max_queue_size(CAPACITY)
                    .with_max_export_batch_size(1)
                    .build(),
            )
            .build();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(BoundedProcessor::new(batch))
            .build();
        let tracer = provider.tracer("bounded-test");
        for _ in 0..CAPACITY + 10 {
            drop(tracer.start("finished"));
        }
        assert!(PENDING.load(Ordering::Relaxed) <= CAPACITY);
        assert!(DROPPED.load(Ordering::Relaxed) >= 10);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let _ = provider.force_flush();
        let _ = provider.shutdown();
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        assert!(FAILURES.load(Ordering::Relaxed) > 0);
    }
}
