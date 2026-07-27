# Filesystem Metrics Observability Redesign

## Status

Design. Supersedes the flat counter-based `FilesystemMetrics`
(`cpp/include/milvus-storage/filesystem/observable.h`) and its FFI
snapshot (`cpp/include/milvus-storage/ffi_filesystem_metrics_c.h`).

## Motivation

The current `FilesystemMetrics` is 13 monotonic `atomic<int64_t>`
counters exposed as a flat snapshot over FFI. It answers "how much
happened" but not the questions a monitoring system actually needs:

- No timing. Not a single operation duration is measured, so p50/p95/
  p99 latency — the primary SLI for an object store — cannot be built.
- No distributions. `read_bytes`/`write_bytes` are running sums; the
  per-request size distribution (average object size, tail sizes) is
  unrecoverable.
- No dimensions. Every field is a flat global, so failures cannot be
  attributed to an operation, and there is no per-op error rate. A
  single `failed_count_` collapses throttling, timeouts, auth failures,
  and not-found into one number.
- No saturation signal. There is no point-in-time gauge (in-flight
  requests, connections) to tell whether the client is bottlenecked.
- Retries are invisible. A request that succeeds after five retries
  looks identical to one that succeeded immediately.

This redesign replaces the flat counters with a labeled registry keyed
by operation type and status, adds latency and size histograms, adds
saturation gauges, and makes error classification and retries
first-class. The FFI exports sum + count + fixed buckets so a consumer
(milvus) can reconstruct Prometheus histograms.

## Goals

- Per-operation latency histograms.
- Per-operation error classification (throttling, timeout, auth,
  network, not-found, server, client, unknown) and per-op error rate.
- Retry counts per operation.
- In-flight and connection-pool gauges (saturation).
- Per-request size distributions for data-transfer operations.
- First-class coverage of `List`, `OpenInput`, `OpenOutput`, and the
  full multipart lifecycle including `MultipartAbort` (leaked-part
  detection).
- A self-contained FFI snapshot with no dynamic allocation.

## Non-goals

- Higher-layer node metrics (compaction, index, cache) — those live in
  milvus, not the storage library.
- Changing the milvus consumer in this repository. The milvus bridge is
  a separate repo; it must migrate to the labeled snapshot (see
  Migration), but that change does not land here.
- Distributed tracing / span propagation.

## Design

### Core model

Every signal is an observation against two enums, so any metric can be
sliced by `op_type` and (for errors) `status`.

`OpType` (15) — real backend operations, not stream-level reads:

```
Read, Write, List, Head, OpenInput, OpenOutput, CreateDir, DeleteDir,
DeleteFile, Move, Copy, MultipartCreate, MultipartUploadPart,
MultipartComplete, MultipartAbort
```

`OpStatus` (9) — success plus the error taxonomy:

```
Ok, NotFound, Throttled, Auth, Timeout, Network, ServerError,
ClientError, Unknown
```

Metric primitives, all `std::atomic<int64_t>` with `memory_order_relaxed`:

- `Counter` — monotonic.
- `Gauge` — up/down; used for in-flight, open/idle connections, pending
  multipart.
- `Histogram` — fixed static bucket bounds plus `sum` and `count`;
  per-bucket atomic counts. Two bucket families: latency (microseconds,
  exponential, 16 buckets ~100us..~130s) and size (bytes, exponential,
  14 buckets ~256B..~4GB). Bounds are compile-time constants shared by
  both sides of the FFI.

### One `ScopedOp` = one backend request

Hard rule: a `ScopedOp` measures exactly one physical request/RPC. This
removes all aggregate-versus-per-request ambiguity.

- A multipart upload is not one `ScopedOp`. It is `1x MultipartCreate` +
  `N x MultipartUploadPart` + `1x MultipartComplete`, each its own
  scope.
- `size` on a `MultipartUploadPart` is that part's bytes; `retry_count`
  is the retries of that one part's request.
- "Total upload size" is never stored; it is `sum(MultipartUploadPart
  .size)`, recovered downstream. Same for total retries.
- `MultipartCreate`, `MultipartComplete`, `MultipartAbort` are control
  requests with no payload.

### Size applies only to transfer operations

Only three operations move a payload: `Read`, `Write`,
`MultipartUploadPart`. Rather than carry a dead size histogram on all
15 op types, the record is split into two shapes, and the C++ type
encodes whether size applies:

```cpp
// Control + metadata ops. Latency + status + retries. No size.
[[nodiscard]] ScopedOp   StartOp(OpType op);

// Data-transfer ops only: Read, Write, MultipartUploadPart.
// A ScopedOp plus RecordBytes().
[[nodiscard]] ScopedXfer StartTransfer(OpType op);
```

`StartTransfer` must only be called with `Read`, `Write`, or
`MultipartUploadPart` (debug-asserted). Choosing the factory answers
"does this op have a size?" and there is no `SetBytes` sitting uselessly
on a `DeleteFile`.

### API shape

```cpp
class FilesystemMetrics {
 public:
  // Starts the timer, increments in_flight[op]. Records latency,
  // status, retries, and decrements in_flight on destruction of the
  // returned scope.
  [[nodiscard]] ScopedOp   StartOp(OpType op);
  [[nodiscard]] ScopedXfer StartTransfer(OpType op);

  // Best-effort connection-pool gauges, set by the backend client.
  void SetConnectionStats(int64_t open, int64_t idle);

  Snapshot GetSnapshot() const;   // structured, in-process view
  void Reset();
};

class ScopedOp {
 public:
  void RecordRetry(int n = 1);    // retries of THIS request -> retry_count[op]
  void Fail(OpStatus s);          // final status; default Ok
  ~ScopedOp();                    // latency_hist[op] << elapsed;
                                  // count_by_status[op][status]++;
                                  // in_flight[op]--
 protected:
  // ... op, start time, status, metrics back-pointer
};

class ScopedXfer : public ScopedOp {
 public:
  void RecordBytes(int64_t n);    // this request's payload ->
                                  // size_hist[xfer] << n; bytes_total[xfer] += n
};
```

Call site (S3 read):

```cpp
auto op = metrics_->StartTransfer(OpType::Read);
auto outcome = client_->GetObject(request);
if (!outcome.IsSuccess()) {
  op.Fail(ClassifyS3Error(outcome.GetError()));
  return TranslateError(outcome);
}
op.RecordBytes(outcome.GetResult().GetContentLength());
```

Call site (control op):

```cpp
auto op = metrics_->StartOp(OpType::DeleteFile);
auto st = arrow::fs::LocalFileSystem::DeleteFile(path);
if (!st.ok()) op.Fail(ClassifyArrowStatus(st));
return st;
```

### Error classification

Per-backend free functions map a native error to `OpStatus`:

```cpp
OpStatus ClassifyS3Error(const Aws::S3::S3Error& e);      // s3_client.cpp
OpStatus ClassifyAzureError(const Azure::Core::...& e);   // azurefs.cc
OpStatus ClassifyArrowStatus(const arrow::Status& s);     // local_fs_producer.cpp
```

Mapping intent (S3 example): HTTP 503 / `SlowDown` / 429 -> `Throttled`;
403 / signature errors -> `Auth`; 404 / `NoSuchKey` -> `NotFound`;
request timeout / connection timeout -> `Timeout`; connection reset /
DNS / TLS -> `Network`; other 5xx -> `ServerError`; other 4xx ->
`ClientError`; anything unmapped -> `Unknown`.

### Retries

The existing s3_client retry loop calls `op.RecordRetry()` on each retry
attempt for the request it is driving. Retries of a part upload count
against that part's `MultipartUploadPart` scope.

### FFI shape

Both enums and both bucket families are compile-time constant on each
side, so the entire snapshot is a fixed-size, caller-allocated struct —
no malloc/free and no length negotiation.

```c
#define LOON_OP_TYPE_COUNT    15
#define LOON_STATUS_COUNT      9
#define LOON_LATENCY_BUCKETS  16   /* exp, ~100us .. ~130s */
#define LOON_SIZE_BUCKETS     14   /* exp, ~256B .. ~4GB   */
#define LOON_TRANSFER_COUNT    3   /* Read=0, Write=1, MultipartUploadPart=2 */

typedef struct LoonOpStats {          /* every op */
  int64_t count_by_status[LOON_STATUS_COUNT];
  int64_t retry_count;
  int64_t latency_sum_us;
  int64_t latency_count;
  int64_t latency_buckets[LOON_LATENCY_BUCKETS];
} LoonOpStats;

typedef struct LoonTransferStats {    /* Read, Write, MultipartUploadPart */
  int64_t bytes_total;
  int64_t size_sum_bytes;
  int64_t size_count;
  int64_t size_buckets[LOON_SIZE_BUCKETS];
} LoonTransferStats;

typedef struct LoonFilesystemMetricsSnapshot {
  LoonOpStats       ops[LOON_OP_TYPE_COUNT];        /* indexed by OpType */
  LoonTransferStats transfers[LOON_TRANSFER_COUNT]; /* indexed by transfer enum */
  int64_t in_flight;
  int64_t open_connections;
  int64_t idle_connections;
  int64_t pending_multipart_created;
  int64_t pending_multipart_finished;
} LoonFilesystemMetricsSnapshot;

/* Static bucket bounds, queried once at startup. */
FFI_EXPORT const int64_t* loon_fs_latency_bucket_bounds_us(int32_t* out_len);
FFI_EXPORT const int64_t* loon_fs_size_bucket_bounds_bytes(int32_t* out_len);

FFI_EXPORT LoonFFIResult loon_filesystem_get_metrics(
    FileSystemHandle handle, LoonFilesystemMetricsSnapshot* out);
FFI_EXPORT LoonFFIResult loon_filesystem_reset_metrics(FileSystemHandle handle);
```

Snapshot size is ~476 `int64_t` (~3.8 KB). The consumer reconstructs
Prometheus histograms from `sum` / `count` / `buckets`, and reads the
static bounds once.

### Bucket semantics

`latency_buckets[i]` and `size_buckets[i]` hold the count of
observations whose value is less than or equal to
`bounds[i]` and greater than `bounds[i-1]` (non-cumulative, one bucket
per interval, with an implicit `+Inf` overflow captured by
`count - sum(buckets)`). `latency_sum_us` / `size_sum_bytes` carry the
exact sum so means are exact regardless of bucketing.

### Instrumentation points

- `s3_client.cpp`: wrap Get/Put/Multipart-part with `StartTransfer`;
  wrap CreateMultipartUpload/Complete/Abort and metadata calls with
  `StartOp`; classify via `ClassifyS3Error`; hook `RecordRetry` in the
  retry loop; feed `SetConnectionStats` from the AWS HTTP client where
  available (best-effort).
- `local_fs_producer.cpp`: replace the `TRACK_METRICS` macros with
  `StartOp` / `StartTransfer` scopes; classify via `ClassifyArrowStatus`.
- `azurefs.cc`: same treatment; classify via `ClassifyAzureError`;
  connection stats best-effort.
- `observable.h` stream wrappers (`MetricsInputStream`,
  `MetricsRandomAccessFile`, `MetricsOutputStream`): each `Read`/`Write`
  records a `Read`/`Write` transfer scope. This raises the volume of
  small-latency observations; acceptable and expected.

### Query API for in-repo consumers

The labeled registry is the single source of truth: there are no flat
counter members. In-repo consumers (the S3 CRT async read path, the
`benchmark_predicate` benchmark, and the filesystem tests) read the
registry through a small set of live, labeled query methods:

- `OpCount(op, status)` / `OpCount(op)` — per-cell and per-op counts.
- `TransferBytes(op)` — bytes moved by a transfer op.
- `FailedCount()` — sum of all non-Ok statuses across every op.
- `InFlight()`, `MultipartCreated()`, `MultipartFinished()` — gauges.

`GetSnapshot()` returns the full labeled snapshot (ops, transfers,
gauges, histograms) for consumers that need histograms or a consistent
copy. Stream wrappers only record a transfer op when it moved bytes; a
zero-byte (EOF) success calls `ScopedOp::Cancel()` (no op recorded,
in-flight balanced), and failures are recorded with a classified non-Ok
status.

### Migration

This removes the old flat `LoonFilesystemMetricsSnapshot`. The milvus
bridge (a separate repo) must migrate `PublishFilesystemMetrics` from the
old flat struct to the labeled `loon_filesystem_get_metrics` snapshot,
indexing `ops[]` / `transfers[]` by the exported `LOON_OP_*` /
`LOON_STATUS_*` / `LOON_XFER_*` enums and reconstructing histograms from
the sum/count/bucket data. No compatibility shim is kept.

## Threading

All primitives are `std::atomic<int64_t>` updated with
`memory_order_relaxed`. Histogram observation is a bucket-index compute
(branchless / binary search over static bounds) followed by three
relaxed increments (bucket, sum, count). `GetSnapshot` and the FFI
export do relaxed loads; the snapshot is not a consistent instant across
fields, which is acceptable for monitoring.

## Testing

- `cloud_fs_metrics_test.cpp` (extended): latency histograms populate;
  `in_flight` returns to zero after each op completes; retries counted;
  status attributed to the correct op; multipart part sizes recorded per
  part and totals recoverable by summation.
- Error-classification unit tests: table-driven, mapping representative
  S3 / Azure / arrow errors to the expected `OpStatus`.
- FFI test (`ffi/...`): assert the snapshot struct is zeroable, that
  `loon_fs_*_bucket_bounds_*` return monotonically increasing bounds of
  the declared length, and that `loon_filesystem_get_metrics` rejects a
  null handle.

## Risks and open questions

- Connection-pool gauges depend on what the AWS and Azure SDKs actually
  expose; may be partial or unavailable. Treated as best-effort — the
  gauges exist and read zero when the backend cannot supply them.
- Stream-level reads produce one observation per `Read` call, increasing
  observation volume for small reads. Mitigated by relaxed atomics and
  fixed-bucket histograms (O(1) per observation).
- Bucket boundaries are fixed at compile time. Chosen ranges
  (latency ~100us..~130s, size ~256B..~4GB) are expected to cover object
  storage workloads; revisiting them is a boundary-table change plus a
  consumer re-read of `loon_fs_*_bucket_bounds_*`.
```
