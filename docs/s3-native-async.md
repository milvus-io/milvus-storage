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

SDK CRT clients and native transports use the same `S3CrtClientHolder`, operation
leases and `S3CrtClientFinalizer` registry. Each holder owns either an SDK client or
a native C client. Native holders close admission and release their client after
the last lease; the finalizer waits for the native shutdown callback as well as
SDK destructors before `Aws::ShutdownAPI()`. There is no separate transport
registry or global shutdown barrier. This preserves callback-safe transport
destruction without moving the SDK's blocking destructor onto a CRT thread.

Local stack: metadata and same-instance API, output-stream native submission,
then directory/delete/copy/move. Validation uses the storage development container
through wt-build, with all outputs and fixtures on /data/yuruiz.
