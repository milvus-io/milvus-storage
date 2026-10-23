//! Explicit provider ownership. The SDK batch processor uses a dedicated
//! thread and a bounded try_send queue; no Tokio reactor is needed on rings.
use super::*;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider,
};
use tracing_subscriber::{layer::SubscriberExt, Layer};

pub struct ExportOwner {
    provider: Option<SdkTracerProvider>,
}
impl Drop for ExportOwner {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            // Never wait on a ring or a single-thread reactor during Drop.
            let _ = std::thread::Builder::new()
                .name("talon-otel-shutdown".into())
                .spawn(move || {
                    let _ = provider.shutdown();
                });
        }
    }
}
impl ExportOwner {
    /// Call after business drain, from a blocking thread. SDK shutdown is bounded.
    pub fn shutdown(mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.shutdown();
        }
    }
}

/// Binary-only global initialization. Libraries must use configure + host layer.
pub fn init_scoped(
    service: &'static str,
) -> Result<(ExportOwner, tracing::Dispatch), Box<dyn std::error::Error + Send + Sync>> {
    // Serialize explicit initialization before constructing any exporter. A
    // second host must not disturb the active provider's health accounting.
    static INITIALIZING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _initializing = INITIALIZING.lock().unwrap();
    if CONFIG.get().is_some() {
        return Err("telemetry already configured".into());
    }
    let config = Config::from_env()?;
    config.validate()?;
    let filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let provider = if matches!(config.mode, Mode::Standard | Mode::Diagnostic) {
        let endpoint = std::env::var("TALON_TELEMETRY_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4318/v1/traces".into());
        // reqwest's blocking client must be constructed outside an async runtime.
        Some(
            std::thread::spawn(move || -> Result<SdkTracerProvider, String> {
                let exporter = opentelemetry_otlp::SpanExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint)
                    .with_timeout(Duration::from_secs(3))
                    .build()
                    .map_err(|e| e.to_string())?;
                let processor =
                    BatchSpanProcessor::builder(super::export_health::HealthExporter(exporter))
                        .with_batch_config(
                            BatchConfigBuilder::default()
                                .with_max_queue_size(4096)
                                .with_max_export_batch_size(256)
                                .with_scheduled_delay(Duration::from_secs(1))
                                .build(),
                        )
                        .build();
                Ok(SdkTracerProvider::builder()
                    .with_span_processor(super::export_health::BoundedProcessor::new(processor))
                    .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                    .with_max_attributes_per_span(48)
                    .with_max_events_per_span(32)
                    .with_max_links_per_span(32)
                    .with_max_attributes_per_event(8)
                    .with_max_attributes_per_link(4)
                    .with_resource(
                        opentelemetry_sdk::Resource::builder_empty()
                            .with_attributes([
                                opentelemetry::KeyValue::new("service.name", service),
                                opentelemetry::KeyValue::new(
                                    "service.version",
                                    env!("CARGO_PKG_VERSION"),
                                ),
                            ])
                            .build(),
                    )
                    .build())
            })
            .join()
            .map_err(|_| "telemetry initialization thread panicked")??,
        )
    } else {
        None
    };
    let layer = provider.as_ref().map(|p| {
        tracing_opentelemetry::layer()
            .with_tracer(p.tracer("talon"))
            .with_filter(tracing_subscriber::filter::filter_fn(|m| {
                m.target() == "talon_telemetry"
            }))
    });
    let dispatch = tracing::Dispatch::new(
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_filter(filter))
            .with(layer),
    );
    configure(config)?;
    Ok((ExportOwner { provider }, dispatch))
}

/// Install the binary subscriber explicitly; never called by client constructors.
pub fn init(
    service: &'static str,
) -> Result<ExportOwner, Box<dyn std::error::Error + Send + Sync>> {
    let (owner, dispatch) = init_scoped(service)?;
    tracing::dispatcher::set_global_default(dispatch)?;
    Ok(owner)
}
