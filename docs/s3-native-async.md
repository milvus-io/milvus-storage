# Native asynchronous S3 filesystem

Scope: Storage's filesystem and S3 transport only. Existing synchronous APIs keep
their behavior. Callers obtain the new capability with `MakeAsyncFileSystem(fs,
io_context)`; unsupported backends/configurations fail explicitly instead of
running synchronous network requests in a worker pool.

## Planned local stack

1. `s3-async/metadata`: native CRT request transport and lifetime barrier, subtree
   routing, HEAD/stat, paginated LIST, and focused transport tests. Existing CRT range
   GET is reused; file metadata/size reuse the existing RandomAccessFile API.
2. `s3-async/write`: ordinary and conditional PUT, multipart creation, upload,
   completion and abort, with owned buffers and failure/lifecycle tests.
3. `s3-async/mutations`: delete/copy/move and directory lifecycle operations,
   bounded enumeration, compatibility checks and integration validation.

Each layer contains implementation, tests and documentation for its operations.
No remote publication is part of this task.

## Execution model

`aws-c-s3` owns DNS, connections, TLS and request progression. The C++ SDK is used
for endpoint resolution and request/response models, not its generated `Async`
methods: in the pinned SDK, many of those submit synchronous HTTP to an executor.
Network response completion crosses the CRT request shutdown boundary before
being dispatched to the caller's Arrow executor. Rejected completion dispatch
must preserve the actual I/O result, particularly successful writes.

Transport setup and global SDK shutdown are explicit synchronous lifecycle
boundaries. Request methods never wait for network completion. Callers must keep
their executor running until all futures complete and must not block inside an
inline continuation. Global S3 shutdown drains native clients and completions
before releasing the AWS SDK.

The initial transport supports AWS/MinIO with explicit, anonymous or native CRT
default-chain credentials. Explicit AssumeRole/WebIdentity options, custom C++
credential/retry adapters and explicit proxies require additional configuration
adapters and are rejected rather than silently ignored. Credential configuration
files may be read during setup. Responses to metadata requests are limited to
16 MiB per request, and concurrent network requests to `max_connections`.

Requests make one attempt. A transport failure after dispatch can leave the
outcome of a mutation unknown; the transport never automatically replays a
conditional write. `io_context.stop_token()` is checked before dispatch, and
does not currently cancel an already dispatched request. Keep the transport and
executor lifecycle separate from per-request cancellation.

The synchronous `S3CrtClient::HeadObjectAsync` wrapper in the pinned C++ SDK is
not used by this capability. The existing CRT RandomAccessFile uses native HEAD for ReadMetadataAsync and
GetSizeAsync. Its range GET and ReadAsync implementations are reused.
AsyncFileSystem adds no second read API.

Validation uses `wt-build`, data-backed temporary/cache paths and an isolated
S3 fixture. It must test delayed metadata requests with one caller worker,
pagination, HTTP errors, special-character keys, buffer lifetime and shutdown.

Layer 1 validation: CRT-enabled Release library and test binary built in the
storage development container. Eight isolated HTTP-fixture tests passed, including
one-worker delayed HEAD/LIST/GET, pagination, HTTP failures, encoded object names,
request lifetime and rejected completion dispatch. This is protocol/lifecycle
coverage, not AWS authentication, production throughput or TLS certification.
