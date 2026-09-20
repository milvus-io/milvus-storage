# Asynchronous manifest transaction scheduling

This patch adds caller-executor scheduling and C callbacks around the existing
synchronous transaction API. It does not change the filesystem, S3 client,
credentials, network transport, retries, or manifest storage protocol.
Network waits occupy a caller worker; this is not native nonblocking S3 I/O.

## Execution boundary

```text
C callback bridge / C++ SemiFuture
  -> caller-owned blocking-work executor
  -> existing Transaction::Open / Transaction::Commit
  -> existing Arrow filesystem and backend
```

Only transaction scheduling and callback/handle lifecycle are new. LIST,
manifest reads, resolver application, conditional writes and conflict retries
use the synchronous transaction implementation and may block executor workers.
The patch adds no filesystem interface, S3 client, CRT request engine,
dependency, credential provider, retry policy or SDK shutdown mechanism.

## C++ API

`BeginAsync` returns a lazy `folly::SemiFuture<BeginResult>`. Consuming it with
`.via(&executor)` runs filesystem initialization and `Transaction::Open` on that
executor. Supply a pool suitable for blocking work, not an event-loop thread.
Inline executors are rejected. A continuation may select a different executor.
The library creates no pool. Each operation can use a different caller pool.

The existing filesystem cache, version lookup, manifest read/cache and resolver
semantics are reused. All backends supported by synchronous transactions remain
available; there is no new AWS-only capability check. An injected ordinary Arrow
filesystem can be used in C++ tests. Returned transactions retain the ordinary
filesystem and can use existing synchronous methods.

## C ABI and ownership

Configure `loon_async_configure_executor` once with a `LoonAsyncExecutor`.
Its `submit` function must enqueue without waiting or running inline, return zero
only when it will run the task exactly once, and neither retain nor run a rejected
task. It must be thread-safe and must not throw. Storage copies the descriptor;
the caller owns its context and workers.

Submission copies inputs. An initial enqueue rejection returns an error and
invokes no callback. Accepted operations invoke exactly one callback, normally
on the supplied executor and possibly before submission returns. Exceptional
completion-enqueue failure delivers the result on the completing worker.
Callbacks must not throw, block, or call shutdown. The callback owns its result
and successful transaction and must free them using the existing APIs.

Cancel and release are separate: cancel requests skipping work that has not
started; release only drops the handle. Neither waits. Releasing a handle early
does not cancel accepted work. Dropping an unconsumed C++ future starts no work.
The resolver must outlive any transaction retaining it.

## Cancellation, timeout and admission

Cancellation and deadlines are checked before executing the synchronous
transaction operation (and after filesystem initialization for begin). Once
`Transaction::Open` is running, it completes normally, even if the operation is
cancelled or its deadline expires. There is no interruption of in-flight network
I/O and no end-to-end deadline guarantee. Input and callback ownership lasts
until completion. Options default to 30 seconds; the maximum is one day.

`LOON_ASYNC_MAX_OPERATIONS` defaults to 256 and limits C operations through
callback return. The caller executor controls native C++ concurrency. Existing
filesystem memory policies apply; this patch introduces no response-byte limit
or buffer-budget setting. Invalid admission configuration returns an error.

`loon_async_shutdown` rejects new admission and waits for accepted callbacks.
Call it on an application thread, then drain/join the caller pool before freeing
its context. It does not stop that pool and cannot interrupt synchronous I/O;
shutdown may wait for the underlying backend's timeout. C++ callers must drain
their own futures before destroying executors or shutting down storage.

## Scope and verification

This single PR includes begin and commit scheduling, lifecycle tests and C/Go
callers. Native S3 work is outside its scope.

It depends directly on [PR #693](https://github.com/milvus-io/milvus-storage/pull/693)
(`s3-async/metadata`). The filesystem cache's concrete `FileSystemPtr` is retained
through the transaction's Arrow pointer. This layer still schedules synchronous
transaction operations; it does not switch them to the base PR's native async
metadata methods or depend on its subsequent write/mutation layers.

C++ tests cover lazy execution, caller executor selection, queued cancellation,
queue deadlines, blocking I/O ownership and synchronous interoperability. C FFI
tests exercise one caller worker, admission, callback ownership and local storage.
Set `LOON_ASYNC_TEST_ENDPOINT` to opt into the existing MinIO round trips; the
bucket is `manifest-async-test`, with test credentials `manifesttest` and
`manifesttestsecret`. Run builds/tests through the repository's `wt-build`
development-container launcher with all outputs and caches on the data disk.

## Commit API

`CommitAsync` schedules the existing `Transaction::Commit`, including resolver
processing and conflict retries. Its lazy future reserves the transaction;
discarding the future releases that reservation. Once execution starts, repeated
submission returns Busy. Keep the transaction alive and unmodified until the
result/callback; destruction inside the terminal C callback is supported.
Initial C admission rejection leaves the transaction available for resubmission.

The callback reports COMMITTED and the version on synchronous success, even if
cancellation/deadline occurred during execution. Cancellation/deadline before
execution reports NOT_COMMITTED. Any error from the synchronous Commit call
conservatively reports UNKNOWN and version -1: the existing API does not expose
write-dispatch or durability details. This includes errors that in fact happened
before a write. The original error remains available; do not blindly replay an
uncertain commit. Backend retries and conditional-write behavior are unchanged.
This wrapper adds no stronger durability or atomicity guarantee to any backend.

C FFI commit round trips run on local storage by default and optionally on MinIO.
C++ tests cover reservation release, queued cancellation, executor switching,
confirmed success after cancellation and custom resolver latest-manifest merging.
