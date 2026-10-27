#![cfg(feature = "recording")]
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
use std::sync::{Arc, Mutex};
use talon_telemetry::*;
use tracing::instrument::WithSubscriber;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Debug, Clone, Default)]
struct Capture(Arc<Mutex<Vec<SpanData>>>);
impl SpanExporter for Capture {
    fn export(
        &mut self,
        batch: Vec<SpanData>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send>,
    > {
        self.0.lock().unwrap().extend(batch);
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn parents_concurrent_polls_sampling_cancellation_and_budget() {
    configure(Config {
        mode: Mode::Standard,
        root_sample_ratio: 1.0,
        max_spans_per_request: 8,
        ..Config::default()
    })
    .unwrap();
    let exporter = Capture::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .with_max_attributes_per_span(48)
        .build();
    let dispatch = tracing::Dispatch::new(
        tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test"))),
    );
    let a = TraceContext::from_w3c(
        "00-11111111111111111111111111111111-1111111111111111-01",
        Some("a=1"),
    )
    .unwrap();
    let b = TraceContext::from_w3c(
        "00-22222222222222222222222222222222-2222222222222222-01",
        None,
    )
    .unwrap();
    async {
        let ambient = Operation::new("ambient", "internal", TraceParent::Explicit(&b));
        ambient
            .scope(async {
                let explicit = Operation::new("explicit", "internal", TraceParent::Explicit(&a));
                let inherited = Operation::new("inherited", "internal", TraceParent::Inherit);
                let root = Operation::new("root", "internal", TraceParent::Root);
                assert_eq!(
                    &explicit.carrier().unwrap().traceparent()[3..35],
                    &a.traceparent()[3..35]
                );
                assert_eq!(
                    &inherited.carrier().unwrap().traceparent()[3..35],
                    &b.traceparent()[3..35]
                );
                assert_ne!(
                    &root.carrier().unwrap().traceparent()[3..35],
                    &b.traceparent()[3..35]
                );
                let read_a = explicit.scope(async {
                    for _ in 0..3 {
                        tokio::task::yield_now().await;
                        assert_eq!(
                            &current_carrier().unwrap().traceparent()[3..35],
                            &a.traceparent()[3..35]
                        );
                    }
                });
                let read_b = inherited.scope(async {
                    for _ in 0..3 {
                        tokio::task::yield_now().await;
                        assert_eq!(
                            &current_carrier().unwrap().traceparent()[3..35],
                            &b.traceparent()[3..35]
                        );
                    }
                });
                tokio::join!(read_a, read_b);
                explicit.outcome("success");
                inherited.outcome("success");
                root.outcome("success");
                let unsampled = TraceContext::from_w3c(
                    "00-33333333333333333333333333333333-3333333333333333-00",
                    Some("vendor=state"),
                )
                .unwrap();
                let operation =
                    Operation::new("unsampled", "internal", TraceParent::Explicit(&unsampled));
                assert!(!operation.is_recording());
                operation
                    .scope(async {
                        let child =
                            Operation::new("unsampled-child", "client", TraceParent::Inherit);
                        assert!(!child.is_recording());
                        assert_eq!(child.carrier(), Some(&unsampled));
                    })
                    .await;
            })
            .await;
        ambient.outcome("success");
        let bounded = Operation::new("bounded", "internal", TraceParent::Root);
        bounded
            .scope(async {
                for _ in 0..32 {
                    let child = Operation::new("child", "internal", TraceParent::Inherit);
                    child.outcome("success");
                }
            })
            .await;
        bounded.outcome("success");
        let timeout = Operation::new("deadline", "internal", TraceParent::Root);
        let attempt = timeout
            .in_scope(|| Operation::new("timed-out-attempt", "client", TraceParent::Inherit));
        timeout.outcome("timeout");
        drop(attempt);
        drop(Operation::new("cancelled", "internal", TraceParent::Root));
        let body = Operation::new("long-body", "internal", TraceParent::Root);
        for n in 1..=100_000 {
            body.record("talon.origin.body_bytes", n);
            body.record("talon.cache.committed_bytes", n);
            body.record("talon.response.bytes", n);
            body.record("custom.counter", n);
        }
        body.text("custom.label", "old");
        body.text("custom.label", "new");
        body.outcome("success");
        let full = Operation::new("full-details", "internal", TraceParent::Root);
        let truncated = Operation::new("truncated-details", "internal", TraceParent::Root);
        const KEYS: [&str; 23] = [
            "detail.00",
            "detail.01",
            "detail.02",
            "detail.03",
            "detail.04",
            "detail.05",
            "detail.06",
            "detail.07",
            "detail.08",
            "detail.09",
            "detail.10",
            "detail.11",
            "detail.12",
            "detail.13",
            "detail.14",
            "detail.15",
            "detail.16",
            "detail.17",
            "detail.18",
            "detail.19",
            "detail.20",
            "detail.21",
            "detail.22",
        ];
        for key in &KEYS[..22] {
            full.record(key, 1);
        }
        full.record("talon.response.bytes", 100);
        full.outcome("success");
        for key in KEYS {
            truncated.record(key, 1);
        }
        truncated.outcome("success");
    }
    .with_subscriber(dispatch)
    .await;
    provider.force_flush().unwrap();
    let spans = exporter.0.lock().unwrap();
    let explicit = spans.iter().find(|s| s.name == "explicit").unwrap();
    for (name, complete) in [("full-details", "true"), ("truncated-details", "false")] {
        let span = spans.iter().find(|s| s.name == name).unwrap();
        assert_eq!(span.dropped_attributes_count, 0, "{name}");
        assert!(
            span.attributes
                .iter()
                .any(|a| a.key.as_str() == "talon.details.complete"
                    && a.value.to_string() == complete)
        );
        assert!(span
            .attributes
            .iter()
            .any(|a| a.key.as_str() == "talon.outcome" && a.value.to_string() == "success"));
    }

    let body = spans.iter().find(|s| s.name == "long-body").unwrap();
    assert_eq!(body.dropped_attributes_count, 0);
    let keys: std::collections::HashSet<_> =
        body.attributes.iter().map(|a| a.key.as_str()).collect();
    assert_eq!(keys.len(), body.attributes.len());
    for (key, expected) in [
        ("talon.origin.body_bytes", "100000"),
        ("talon.cache.committed_bytes", "100000"),
        ("talon.response.bytes", "100000"),
        ("custom.counter", "100000"),
        ("custom.label", "new"),
        ("talon.outcome", "success"),
        ("talon.details.complete", "true"),
    ] {
        assert!(
            body.attributes
                .iter()
                .any(|a| a.key.as_str() == key && a.value.to_string() == expected),
            "{key}"
        );
    }
    assert_eq!(explicit.parent_span_id.to_string(), "1111111111111111");
    assert_eq!(spans.iter().filter(|s| s.name == "child").count(), 7);
    assert!(!spans.iter().any(|s| s.name.starts_with("unsampled")));
    let cancelled = spans.iter().find(|s| s.name == "cancelled").unwrap();
    assert!(cancelled
        .attributes
        .iter()
        .any(|a| a.key.as_str() == "talon.outcome" && a.value.to_string() == "cancelled"));
    let timed_out = spans
        .iter()
        .find(|s| s.name == "timed-out-attempt")
        .unwrap();
    assert!(timed_out
        .attributes
        .iter()
        .any(|a| a.key.as_str() == "talon.outcome" && a.value.to_string() == "timeout"));
    let bounded = spans.iter().find(|s| s.name == "bounded").unwrap();
    assert!(bounded
        .attributes
        .iter()
        .any(|a| a.key.as_str() == "talon.details.complete" && a.value.to_string() == "false"));
}
