// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0

use crate::rust_runtime_ffi::{self as ffi, TraceContext};
use futures::{Stream, future::BoxFuture};
use std::future::Future;
use std::sync::Arc;
use vortex::io::runtime::tokio::{TokioBlockingIterator, TokioRuntime};
use vortex::io::runtime::{AbortHandleRef, BlockingRuntime, Executor, Handle};

// C++ owns immutable context snapshots. shared_ptr Clone/Drop performs the
// retain/release; no OTel provider or ABI value crosses into Rust.
unsafe impl Send for TraceContext {}
unsafe impl Sync for TraceContext {}

pub(crate) fn capture() -> cxx::SharedPtr<TraceContext> {
    ffi::capture_trace_context().unwrap_or_else(|_| {
        ffi::record_trace_bridge_failure(false);
        cxx::SharedPtr::null()
    })
}
fn attach(context: &cxx::SharedPtr<TraceContext>) -> Option<cxx::UniquePtr<ffi::TraceAttachment>> {
    match ffi::attach_trace_context(context) {
        Ok(scope) => Some(scope),
        Err(_) => {
            ffi::record_trace_bridge_failure(true);
            None
        }
    }
}

pub(crate) fn instrument<F: Future>(future: F) -> impl Future<Output = F::Output> {
    instrument_with_context(future, capture())
}
fn instrument_with_context<F: Future>(
    future: F,
    context: cxx::SharedPtr<TraceContext>,
) -> impl Future<Output = F::Output> {
    async move {
        // Pin inside the wrapper's state rather than adding a heap allocation
        // for every future, including futures with tracing disabled.
        futures::pin_mut!(future);
        futures::future::poll_fn(move |cx| {
            // The guard is local to each poll, including Pending/unwinding.
            // Empty snapshots still mask an unrelated worker-thread parent.
            let _scope = attach(&context);
            future.as_mut().poll(cx)
        })
        .await
    }
}
pub(crate) fn bind<F, R>(task: F) -> impl FnOnce() -> R + Send
where
    F: FnOnce() -> R + Send,
{
    let context = capture();
    move || {
        let _scope = attach(&context);
        task()
    }
}

// Match Vortex's Tokio scheduling and profiling labels, submitting the tracing
// wrapper directly to Tokio so an already boxed Vortex task is not boxed again.
// Empty snapshots still attach on every poll/task to mask foreign parents.
struct TracedExecutor(tokio::runtime::Handle);
impl Executor for TracedExecutor {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        let future = instrument(future);
        #[cfg(unix)]
        let future = {
            use custom_labels::asynchronous::Label;
            future.with_current_labels()
        };
        Box::new(self.0.spawn(future).abort_handle())
    }
    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let task = bind(task);
        let future = async move { task() };
        #[cfg(unix)]
        let future = {
            use custom_labels::asynchronous::Label;
            future.with_current_labels()
        };
        Box::new(self.0.spawn(future).abort_handle())
    }
    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let task = bind(task);
        #[cfg(unix)]
        let task = {
            let mut labels = custom_labels::Labelset::clone_from_current();
            move || labels.enter(task)
        };
        Box::new(self.0.spawn_blocking(task).abort_handle())
    }
}
pub(crate) struct TracedRuntime {
    inner: TokioRuntime,
    executor: Arc<dyn Executor>,
}
impl TracedRuntime {
    pub(crate) fn new(handle: tokio::runtime::Handle) -> Self {
        Self {
            inner: TokioRuntime::new(handle.clone()),
            executor: Arc::new(TracedExecutor(handle)),
        }
    }
}
impl BlockingRuntime for TracedRuntime {
    type BlockingIterator<'a, R: 'a> = TokioBlockingIterator<'a, R>;
    fn handle(&self) -> Handle {
        Handle::new(Arc::downgrade(&self.executor))
    }
    fn block_on<F, R>(&self, future: F) -> R
    where
        F: Future<Output = R>,
    {
        self.inner.block_on(instrument(future))
    }
    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        // Each synchronous next() polls on the calling execution flow, where
        // Storage's ReadNext scope is active. Child spawns capture at that poll.
        self.inner.block_on_stream(stream)
    }
}

// Captured for each submitted request, never for a shared file/scheduler.
struct RequestReader {
    reader: Arc<dyn lance_io::traits::Reader>,
    context: cxx::SharedPtr<TraceContext>,
}
impl std::fmt::Debug for RequestReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RequestReader").field(&self.reader).finish()
    }
}
impl deepsize::DeepSizeOf for RequestReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.reader.deep_size_of_children(context)
    }
}
impl lance_io::traits::Reader for RequestReader {
    fn path(&self) -> &object_store::path::Path {
        self.reader.path()
    }
    fn block_size(&self) -> usize {
        self.reader.block_size()
    }
    fn io_parallelism(&self) -> usize {
        self.reader.io_parallelism()
    }
    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(instrument_with_context(
            async { self.reader.size().await },
            self.context.clone(),
        ))
    }
    fn get_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> BoxFuture<'static, object_store::Result<bytes::Bytes>> {
        let reader = self.reader.clone();
        Box::pin(instrument_with_context(
            async move { reader.get_range(range).await },
            self.context.clone(),
        ))
    }
    fn get_all(&self) -> BoxFuture<'_, object_store::Result<bytes::Bytes>> {
        Box::pin(instrument_with_context(
            async { self.reader.get_all().await },
            self.context.clone(),
        ))
    }
}
pub(crate) fn wrap_lance_request(
    reader: Arc<dyn lance_io::traits::Reader>,
) -> Arc<dyn lance_io::traits::Reader> {
    // Preserve empty snapshots as well: a foreign worker parent must be masked.
    Arc::new(RequestReader {
        reader,
        context: capture(),
    })
}
