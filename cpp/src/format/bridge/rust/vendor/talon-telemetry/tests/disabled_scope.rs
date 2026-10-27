use talon_telemetry::{Config, Mode, Operation, TraceContext, TraceParent};

#[tokio::test]
async fn disabled_request_masks_parent_and_restores_each_poll() {
    talon_telemetry::configure(Config {
        mode: Mode::Propagate,
        v2_endpoints: vec!["*".into()],
        ..Default::default()
    })
    .unwrap();
    let parent = TraceContext::from_w3c(
        "00-11111111111111111111111111111111-2222222222222222-01",
        Some("vendor=value"),
    )
    .unwrap();
    let outer = Operation::new("outer", "internal", TraceParent::Explicit(&parent));
    let disabled = Operation::disabled();
    outer
        .scope(async {
            let suppressed = disabled.scope(async {
                for _ in 0..3 {
                    assert!(!talon_telemetry::enabled());
                    assert!(!talon_telemetry::v2_enabled("127.0.0.1:1"));
                    assert!(talon_telemetry::current_carrier().is_none());
                    for policy in [
                        TraceParent::Inherit,
                        TraceParent::Explicit(&parent),
                        TraceParent::Root,
                    ] {
                        let child = Operation::new("child", "internal", policy);
                        assert!(child.carrier().is_none());
                        assert!(!child.is_recording());
                        child.in_scope(|| assert!(!talon_telemetry::enabled()));
                    }
                    tokio::task::yield_now().await;
                }
            });
            let concurrent = async {
                for _ in 0..3 {
                    assert!(talon_telemetry::enabled());
                    assert_eq!(talon_telemetry::current_carrier(), Some(parent.clone()));
                    tokio::task::yield_now().await;
                }
            };
            tokio::join!(suppressed, concurrent);
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                disabled.in_scope(|| panic!("test unwind"));
            }));
            assert!(panic.is_err());
            assert!(talon_telemetry::enabled());
            assert_eq!(talon_telemetry::current_carrier(), Some(parent.clone()));

            // An operation created inside the opt-out remains disabled after it
            // moves to another task with a different ambient parent.
            let child =
                disabled.in_scope(|| Operation::new("later", "internal", TraceParent::Inherit));
            tokio::spawn(async move {
                child
                    .scope(async {
                        assert!(!talon_telemetry::enabled());
                        assert!(talon_telemetry::current_carrier().is_none());
                    })
                    .await;
            })
            .await
            .unwrap();
        })
        .await;
    assert!(talon_telemetry::enabled());
    assert!(talon_telemetry::current_carrier().is_none());
}
