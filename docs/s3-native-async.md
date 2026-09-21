# Native asynchronous S3 filesystem operations

Sync and async operations use the same filesystem instance returned by Storage's
factory and cache. Arrow's existing virtual methods remain available through a
`shared_ptr<arrow::fs::FileSystem>`. Opening an output stream initializes local
state and returns synchronously; network operations use the stream's async methods.

```cpp
ARROW_ASSIGN_OR_RAISE(auto fs, CreateArrowFileSystem(config));
auto info = fs->GetFileInfo("key");
auto pending = fs->GetFileInfoAsync(std::vector<std::string>{"key"});
auto reader = fs->OpenInputFileAsync("key");
ARROW_ASSIGN_OR_RAISE(auto output, fs->OpenOutputStream("key"));
ARROW_RETURN_NOT_OK(output->Write(arrow::Buffer::FromString("payload")));
auto closed = output->CloseAsync();
```

S3FileSystem owns native request execution internally and shares its transport and
IOContext across calls. FileSystemProxy applies its existing subtree prefix. Batch
stat overrides Arrow's existing virtual API; single-path queries pass a one-element
vector. Listing reuses GetFileInfoGenerator.
Existing CRT reads, metadata caches and file objects are reused. CRT input files
require native transport at initialization; unsupported configurations fail to open
with NotImplemented, rather than falling back to executor-backed metadata reads.
In CRT builds, unsupported async metadata, listing and input-opening
operations return NotImplemented instead of scheduling synchronous SDK calls. The existing
SDK input path remains available when CRT reads are disabled. The executor must
outlive operations; request completions retain their state and preserve the actual result if completion dispatch is rejected.

The transport borrows the **same** `aws_s3_client` owned by the filesystem's
`Aws::S3Crt::S3CrtClient`, using its existing `S3CrtClientHolder` and operation
leases. It creates no client, credentials provider, TLS context or connection
pool of its own. `ExtendS3CrtClient.cmake` adds a header-only borrowed-handle
accessor to a build-local copy of the pinned SDK header; it does not modify the
shared dependency cache or the SDK class layout. The accessor can be removed
when an equivalent SDK API is available.

Request leases live through native request shutdown and completion dispatch.
If an inline completion drops the last holder on a CRT callback, the SDK's
blocking destructor runs on a dedicated cleanup worker. This worker performs
only client teardown, not S3 operations. The existing finalizer's live-client
barrier waits for that destructor before `Aws::ShutdownAPI()`.

CRT 0.12.6 configures retries per client, not per request. The shared CRT client
uses the transport's single-attempt policy to prevent replaying mutations after
an ambiguous response. This also disables automatic retries for SDK CRT reads;
the ordinary synchronous SDK client keeps its existing retry policy. Supporting
different read/write retry policies on the same native client requires a CRT
request-level retry extension.

## Existing output streams and transport

The existing `OpenOutputStream` returns an Arrow `OutputStream` whose
`CustomOutputStream` also implements `AsyncOutputStream`. When the filesystem has
a native CRT transport, this factory attaches it to the stream before local
initialization. `OpenConditionalOutputStream` and `OpenOutputStreamWithUploadSize`
use the same creation path. Opening performs no network I/O and needs no Future.
Without a supported native transport, output streams retain the existing SDK path;
that path does not guarantee nonblocking I/O.

The stream retains the existing buffers, part numbering, metadata,
conditional-write headers and completed-part state. The native path replaces
network submission for PUT, multipart create/upload/complete/abort. `Write` copies
or retains memory and submits work; `CloseAsync` waits through continuations and
publishes the object. The existing synchronous `Close`/`Abort` remain explicit
blocking wrappers. An `AsyncOutputStream` extension adds only `FlushAsync` and
`AbortAsync`, which Arrow does not expose. It does not redefine Write or Close.

Callers serialize calls on a stream. Shared input buffers must remain immutable
until FlushAsync or CloseAsync completes. The pending part count is bounded by
max_connections; an oversized Write returns CapacityError before consuming bytes.
Await FlushAsync and submit smaller chunks. Closing drains pending parts before
submitting the last buffer. Failed close attempts abort known multipart uploads;
cleanup failure is reported with the original error. No destructor starts I/O.

## Transport and lifetime

The pinned C++ SDK's generated HEAD/LIST/multipart Async methods can run synchronous
HTTP on an executor. These operations use aws-c-s3 DEFAULT meta requests instead;
the SDK supplies request/response models and endpoint resolution. Existing native
CRT GET is reused. All native request bodies are memory-backed.

Setup and global SDK shutdown are synchronous lifecycle boundaries. The supplied
executor must outlive all pending operations. Completion normally dispatches there;
if dispatch is rejected it completes inline with the original I/O result. Inline
continuations must not block. Global shutdown drains native clients and completion
callbacks before releasing the AWS SDK.

The transport supports AWS/MinIO with explicit, anonymous or native default-chain
credentials. Explicit AssumeRole/WebIdentity settings, custom C++ credential/retry
providers and explicit proxies need adapters and are rejected. Credentials may
read configuration during setup. Metadata responses are capped at 16 MiB.

Requests make one attempt. A lost mutation response can follow a successful write;
it is reported as an error with an unknown possible outcome, never automatically
replayed. HTTP 200 with an embedded completion Error is not success. Cancellation
is checked before dispatch; an already dispatched request is not cancelled.

## Validation

Validation runs through wt-build in the storage development container, with all
outputs, caches and test-service data on /data/yuruiz. The HTTP fixture tests delayed
requests with one caller worker, pagination, encoded paths, HTTP failures, owned
buffers, completion rejection and shutdown. Isolated MinIO validates signing,
metadata, checksums and service semantics. No throughput or AWS/TLS production
certification is claimed.

## Scope

Native operations cover metadata/stat, listing, input reads and output writes.
Stat uses Arrow's existing vector interface for both single-path and batch calls;
there is no added public single-path overload.

Directory cleanup keeps its pre-existing SDK implementation, including
`DeleteDirContentsAsync`; this PR makes no nonblocking guarantee for it. CreateDir,
DeleteDir, DeleteFile, CopyFile and Move keep their existing interfaces. Append
remains unsupported by S3. Changes are confined to FileSystem/S3 and do not modify
Manifest, Transaction or FFI.

## Verified locally (2026-09-21)

The CRT-enabled Release library and C++ tests build in the storage development
container (`WITH_UT=ON`, `WITH_ASAN=OFF`). The output factory now uses the
existing `OpenOutputStream` entry point:

- Native HTTP fixture: 24 passed. Coverage
  includes opening without network access, ordinary and conditional native writes,
  multipart completion/abort, stat/list/read and original directory cleanup.
- CRT read, metadata and lifetime regressions: 23 passed, one cloud-only test skipped.
- Filesystem cache regressions: 20 passed in a separate process.
- Four affected translation units pass syntax compilation with `WITH_CRT`
  undefined; this is not a full no-CRT build or runtime test.
- Clang-format 18 and the error-handling ratchet pass (throw baseline: 21).

This run does not claim ASan, MinIO, production AWS/TLS or load validation.
