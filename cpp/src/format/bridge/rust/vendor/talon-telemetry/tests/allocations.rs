//! Narrow allocation regression check; not a throughput/latency acceptance test.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use talon_telemetry::*;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn exercise(parent: &TraceContext) {
    for _ in 0..100 {
        let parent = std::hint::black_box(parent.clone());
        let operation = Operation::new("read", "internal", TraceParent::Explicit(&parent));
        operation.in_scope(|| {
            let child = Operation::new("rpc", "client", TraceParent::Inherit);
            std::hint::black_box(child.carrier());
            assert!(!child.is_recording());
            child.outcome("success");
        });
        operation.outcome("success");
    }
}
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn off_and_explicit_unsampled_scopes_do_not_allocate() {
    let parent = TraceContext::from_w3c(
        "00-11111111111111111111111111111111-2222222222222222-00",
        Some("test=state"),
    )
    .unwrap();
    exercise(&parent);
    assert_eq!(allocations(|| exercise(&parent)), 0);
    configure(Config {
        #[cfg(feature = "recording")]
        mode: Mode::Standard,
        #[cfg(not(feature = "recording"))]
        mode: Mode::Propagate,
        ..Default::default()
    })
    .unwrap();
    exercise(&parent);
    assert_eq!(allocations(|| exercise(&parent)), 0);
    #[cfg(feature = "recording")]
    {
        use opentelemetry::trace::TracerProvider;
        use tracing_subscriber::layer::SubscriberExt;
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
        let dispatch =
            tracing::Dispatch::new(tracing_subscriber::registry().with(
                tracing_opentelemetry::layer().with_tracer(provider.tracer("allocation-test")),
            ));
        let sampled = TraceContext::from_w3c(
            "00-11111111111111111111111111111111-2222222222222222-01",
            None,
        )
        .unwrap();
        tracing::dispatcher::with_default(&dispatch, || {
            let operation = Operation::new("body", "internal", TraceParent::Explicit(&sampled));
            assert!(operation.is_recording());
            operation.record("custom.counter", 0);
            assert_eq!(
                allocations(|| {
                    for n in 0..100_000 {
                        operation.record("talon.origin.body_bytes", n);
                        operation.record("talon.cache.committed_bytes", n);
                        operation.record("talon.response.bytes", n);
                        operation.record("custom.counter", n);
                    }
                }),
                0
            );
            operation.outcome("success");
        });
    }
}
