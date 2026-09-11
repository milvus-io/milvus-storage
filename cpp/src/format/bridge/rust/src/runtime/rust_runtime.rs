use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RustRuntimeConfig {
    worker_threads: usize,
    max_blocking_threads: usize,
}

#[derive(Default)]
struct RustRuntimeState {
    config: Option<RustRuntimeConfig>,
    initialized: bool,
}

struct RustRuntime {
    state: Mutex<RustRuntimeState>,
}

impl RustRuntime {
    const fn new() -> Self {
        Self {
            state: Mutex::new(RustRuntimeState {
                config: None,
                initialized: false,
            }),
        }
    }

    fn configure(&self, worker_threads: usize, max_blocking_threads: usize) -> anyhow::Result<()> {
        if worker_threads == 0 {
            anyhow::bail!("worker_threads must be greater than 0");
        }
        if max_blocking_threads == 0 {
            anyhow::bail!("max_blocking_threads must be greater than 0");
        }

        let mut state = self.state.lock().unwrap();
        if state.initialized {
            anyhow::bail!(
                "Rust runtime has already been initialized; call ConfigureRustRuntime before first Rust bridge use"
            );
        }
        if state.config.is_some() {
            anyhow::bail!("Rust runtime has already been configured");
        }

        state.config = Some(RustRuntimeConfig {
            worker_threads,
            max_blocking_threads,
        });
        Ok(())
    }
}

pub(crate) fn configure_rust_runtime(
    worker_threads: u32,
    max_blocking_threads: u32,
) -> anyhow::Result<()> {
    RUST_RUNTIME.configure(worker_threads as usize, max_blocking_threads as usize)
}

static RUST_RUNTIME: RustRuntime = RustRuntime::new();

fn default_runtime_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(32)
}

/// Shared Tokio runtime for async work in the Rust bridge.
///
/// Lance, Iceberg, and Vortex all run through this runtime so the bridge does
/// not create separate Tokio worker and blocking thread pools per format.
pub(crate) static TOKIO_RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    let config = {
        let mut state = RUST_RUNTIME.state.lock().unwrap();
        state.initialized = true;
        state.config
    };

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if let Some(config) = config {
        builder.worker_threads(config.worker_threads);
        builder.max_blocking_threads(config.max_blocking_threads);
    } else {
        let thread_count = default_runtime_thread_count();
        builder.worker_threads(thread_count);
        builder.max_blocking_threads(thread_count);
    }
    builder.build().expect("Failed to create tokio runtime")
});

#[cfg(test)]
mod runtime_config_tests {
    use super::*;
    use std::sync::{
        Arc, Condvar,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    #[derive(Default)]
    struct RuntimeProbe {
        running: AtomicUsize,
        peak_running: AtomicUsize,
        started: AtomicUsize,
        release: Mutex<bool>,
        cv: Condvar,
    }

    impl RuntimeProbe {
        fn run(&self) {
            let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.update_peak(running);
            self.started.fetch_add(1, Ordering::SeqCst);
            self.cv.notify_all();

            let mut release = self.release.lock().unwrap();
            while !*release {
                release = self.cv.wait(release).unwrap();
            }

            self.running.fetch_sub(1, Ordering::SeqCst);
        }

        fn update_peak(&self, value: usize) {
            let mut current = self.peak_running.load(Ordering::SeqCst);
            while value > current {
                match self.peak_running.compare_exchange_weak(
                    current,
                    value,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(next) => current = next,
                }
            }
        }

        fn wait_until_started(&self, configured_threads: usize) {
            let release = self.release.lock().unwrap();
            let result = self
                .cv
                .wait_timeout_while(release, Duration::from_secs(5), |_| {
                    self.started.load(Ordering::SeqCst) < configured_threads
                })
                .unwrap();
            assert!(
                !result.1.timed_out(),
                "runtime did not start configured number of tasks"
            );
        }

        fn release(&self) {
            *self.release.lock().unwrap() = true;
            self.cv.notify_all();
        }

        fn peak_running(&self) -> usize {
            self.peak_running.load(Ordering::SeqCst)
        }
    }

    fn probe_worker_parallelism(configured_threads: usize, num_tasks: usize) -> usize {
        let probe = Arc::new(RuntimeProbe::default());
        let handles = (0..num_tasks)
            .map(|_| {
                let probe = Arc::clone(&probe);
                TOKIO_RT.spawn(async move {
                    probe.run();
                })
            })
            .collect::<Vec<_>>();

        probe.wait_until_started(configured_threads);
        std::thread::sleep(Duration::from_millis(200));
        probe.release();

        TOKIO_RT.block_on(async {
            for handle in handles {
                handle.await.unwrap();
            }
        });

        probe.peak_running()
    }

    fn probe_blocking_parallelism(configured_threads: usize, num_tasks: usize) -> usize {
        let probe = Arc::new(RuntimeProbe::default());
        let handles = (0..num_tasks)
            .map(|_| {
                let probe = Arc::clone(&probe);
                TOKIO_RT.spawn_blocking(move || {
                    probe.run();
                })
            })
            .collect::<Vec<_>>();

        probe.wait_until_started(configured_threads);
        std::thread::sleep(Duration::from_millis(200));
        probe.release();

        TOKIO_RT.block_on(async {
            for handle in handles {
                handle.await.unwrap();
            }
        });

        probe.peak_running()
    }

    fn runtime_config(runtime: &RustRuntime) -> Option<RustRuntimeConfig> {
        runtime.state.lock().unwrap().config
    }

    fn runtime_initialized(runtime: &RustRuntime) -> bool {
        runtime.state.lock().unwrap().initialized
    }

    #[test]
    fn configure_runtime_before_initialization_records_config() {
        let runtime = RustRuntime::new();

        runtime.configure(2, 3).unwrap();

        assert_eq!(
            runtime_config(&runtime),
            Some(RustRuntimeConfig {
                worker_threads: 2,
                max_blocking_threads: 3,
            })
        );
        assert!(!runtime_initialized(&runtime));
    }

    #[test]
    fn configure_runtime_rejects_zero_thread_counts() {
        let runtime = RustRuntime::new();

        assert!(runtime.configure(0, 3).is_err());
        assert!(runtime.configure(2, 0).is_err());
    }

    #[test]
    fn configure_runtime_fails_after_initialization() {
        let runtime = RustRuntime::new();

        runtime.state.lock().unwrap().initialized = true;

        let error = runtime.configure(2, 3).unwrap_err().to_string();
        assert!(error.contains("already been initialized"));
    }

    #[test]
    fn configure_runtime_fails_when_called_twice() {
        let runtime = RustRuntime::new();

        runtime.configure(2, 3).unwrap();

        let error = runtime.configure(2, 3).unwrap_err().to_string();
        assert!(error.contains("already been configured"));
    }

    #[test]
    fn default_runtime_thread_count_uses_available_parallelism() {
        let expected = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(32);

        assert_eq!(default_runtime_thread_count(), expected);
    }

    // This test mutates process-global runtime state and must be run alone.
    #[test]
    #[ignore]
    fn global_tokio_runtime_limits_worker_and_blocking_parallelism() {
        const WORKER_THREADS: usize = 2;
        const MAX_BLOCKING_THREADS: usize = 3;
        const TASKS_PER_THREAD: usize = 32;

        configure_rust_runtime(WORKER_THREADS as u32, MAX_BLOCKING_THREADS as u32).unwrap();

        let worker_peak =
            probe_worker_parallelism(WORKER_THREADS, WORKER_THREADS * TASKS_PER_THREAD);
        println!("worker_peak={worker_peak}");
        assert!(
            worker_peak <= WORKER_THREADS,
            "worker peak {} exceeded configured worker threads {}",
            worker_peak,
            WORKER_THREADS
        );

        let blocking_peak = probe_blocking_parallelism(
            MAX_BLOCKING_THREADS,
            MAX_BLOCKING_THREADS * TASKS_PER_THREAD,
        );
        println!("blocking_peak={blocking_peak}");
        assert!(
            blocking_peak <= MAX_BLOCKING_THREADS,
            "blocking peak {} exceeded configured max blocking threads {}",
            blocking_peak,
            MAX_BLOCKING_THREADS
        );
    }
}
