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

This change does not introduce a new asynchronous data upload implementation.
CRT itself supports native asynchronous PUT with automatic multipart handling;
adapting that capability to Storage output semantics is separate work.

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
