# Native asynchronous S3 filesystem operations

Sync and async operations use the same filesystem instance. Storage's factory and
cache return `FileSystemPtr` (`shared_ptr<FileSystemProxy>`), retaining the concrete
methods instead of erasing them to Arrow's smaller interface. The handle remains
implicitly convertible to `shared_ptr<arrow::fs::FileSystem>` for existing consumers.
No separate async factory, cast, capability object or lifecycle is needed.

```cpp
ARROW_ASSIGN_OR_RAISE(auto fs, CreateArrowFileSystem(config));
auto info = fs->GetFileInfo("key");
auto pending = fs->GetFileInfoAsync("key");
auto reader = fs->OpenInputFileAsync("key");
```

S3FileSystem owns native request execution internally and shares its transport and
IOContext across calls. FileSystemProxy applies its existing subtree prefix. Batch
stat overrides Arrow's virtual API too, and listing reuses GetFileInfoGenerator.
Existing CRT reads, metadata caches and file objects are reused. CRT input files
require native transport at initialization; unsupported configurations fail to open
with NotImplemented, rather than falling back to executor-backed metadata reads.
Other unsupported new async operations also return NotImplemented. The existing
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

Local stack: metadata and same-instance API, output-stream native submission,
then directory/delete/copy/move. Validation uses the storage development container
through wt-build, with all outputs and fixtures on /data/yuruiz.
