# Native asynchronous S3 metadata operations

The existing filesystem instance supports native nonblocking HEAD/stat and
paginated listing through `NativeS3Operations`. Input handles reuse CRT range GET
and the existing metadata caches. Arrow's batch `GetFileInfoAsync` and
`GetFileInfoGenerator` expose these operations; `FileSystemProxy` forwards batch
stat through subtree prefixes. Opening an input handle performs no network I/O.

The transport borrows the SDK-owned CRT client through `S3CrtClientHolder` leases.
It creates no separate client or connection pool. Native completions mark Arrow
futures on CRT callback threads; consumers select continuation executors. Leases
cover request shutdown, and blocking SDK destruction runs off callback threads.
Global finalization waits for client destruction.

## Output compatibility

`OpenOutputStream`, `OpenConditionalOutputStream`, and
`OpenOutputStreamWithUploadSize` use the existing ordinary S3 SDK upload path,
regardless of whether CRT reads/native metadata are enabled. This includes PUT,
multipart creation, part upload, completion and abort. Conditional writes keep
the SDK provider-specific signing path. Native metadata transport only accepts
HEAD and GET requests; it does not submit mutations or manage upload buffers.

`background_writes` retains its existing meaning: enabled uploads run synchronous
SDK requests on the configured I/O executor; disabled uploads run on the caller.
`CloseAsync` waits for background work, but multipart control operations can still
block. `AsyncOutputStream::FlushAsync` remains available; `AbortAsync` returns
`NotImplemented` on the SDK stream, and callers use `Flush` then `Abort`.
There is no native part-count admission limit imposed by `max_connections`.

## Parallel upload streams

UploadParallel::OpenParallelUploadOutputStream(path, metadata) opens a stream
without a declared total length. UploadSizable remains the separate capability for
configuring ordinary multipart part sizes. FileSystemProxy applies its prefix and
Talon forwards to its origin; unsupported capabilities return NotImplemented.

The parallel stream requires a CRT build and an enabled CRT client. It opens one
PUT_OBJECT meta request on the existing client with send_using_async_writes. CRT
owns multipart creation, part sizes, scheduling, retries and cleanup. Unknown-length
uploads use multipart, including small and empty objects; use the ordinary stream
when its small-object single PUT behavior is preferred. Write supplies
a chunk using aws_s3_meta_request_write; Close supplies EOF and waits for publication.
It does not call NativeS3Transport or accumulate a complete file before uploading.
No SDK fallback or additional Storage admission queue is used.

CRT allows one pending write per upload. Before supplying the next chunk, Write
waits on the calling thread for CRT to consume the previous one. This does not
serialize network parts: CRT schedules consumed parts independently. Shared buffers
are retained until consumption; raw-pointer Write waits for consumption before
returning, so Storage need not copy that input. Flush waits for consumption, not
publication. CRT's internal memory budget does not include callers' retained input;
callers must still budget concurrent upload tasks and their data.

An idle producer does not hold a network or data-stream worker waiting for input.
Abort cancels the meta request, including after CloseAsync starts; destruction also
cancels an unclosed stream. Completion waits for native shutdown before releasing
request resources. Stop tokens are checked on input and network progress; synchronous
write waits also poll cancellation. Calls on a stream are serialized by the caller.

Public async completions use IOContext::executor(), which must stay alive and
schedule work off CRT threads. Internal completion and synchronous Close do not
need that executor, so a caller continuation can synchronously close another upload.
If dispatch is rejected, the returned future fails on the completion thread; callbacks
on this failure path must not block it. Callbacks registered after a future completes
may run on the registering thread, as with other Arrow futures.

Milvus explicitly selects the parallel capability for whole-buffer artifact uploads;
it no longer passes the artifact size to open the stream. Ordinary Parquet,
compaction and other output callers retain their existing interfaces.

## Lifecycle

Setup and global SDK shutdown are synchronous lifecycle boundaries. Request bodies
and client leases remain alive until backend completion. Callers finalize S3 after
clients and I/O have stopped. Shared bootstrap configuration remains independent of
the upload stream selection.
When Linux confirms SMT is inactive, CRT uses the Arrow CPU pool capacity set by
ConfigureStorageRuntime before SDK initialization. Milvus already supplies that
budget through InitStorageRuntime. If SMT is active or cannot be identified, CRT
retains the SDK default. Later pool resizing does not resize the bootstrap.
Downloads use the same shared runtime.

## Retry and provider behavior

Metadata and CRT range reads retain the SDK CRT client's normal retry policy.
There is no client-wide single-attempt override for uploads. Custom retry
strategies and explicit proxies continue to use SDK metadata fallbacks.
Supported providers reuse the SDK credentials provider, including session tokens
and refresh. Credential retrieval and lifecycle setup can still block.

The transport preserves SDK response parsing and error classification, limits
metadata responses to 16 MiB, and rejects incomplete responses. GCP conditional
uploads remain on `GoogleHttpClientDelegator`; the native metadata adapter does
not perform conditional-write signing. GCS XML multipart does not support
generation preconditions. Such writes return `NotImplemented` before multipart
creation on both CRT settings, preserving the rejection of unsafe publication.
Directory mutation, copy, move and cleanup retain their existing SDK behavior.

## Verification

Run `cpp/scripts/test_native_s3.py` inside the storage development container for
local HTTP-fixture coverage. Tests cover nonblocking metadata with a stopped
executor, credentials, pagination, errors, client lifetime, transient metadata
and range-GET retries, and SDK output routing with CRT disabled/enabled. Upload
coverage includes ordinary, conditional and sized factories, background executor
requirements, multi-part writes exceeding the former native admission limit,
flush/close/abort, payload readback and metrics.

Cloud tests prepare objects through an SDK filesystem with a running executor;
metadata/read assertions use a filesystem with a stopped executor. The synchronous
compatibility suite exercises both CRT settings. Local fixture coverage does not
establish real-cloud authentication/TLS, load, or production compatibility.
