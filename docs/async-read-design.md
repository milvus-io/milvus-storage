# Async Read Design (Current Implementation)

## Status and Scope

This document preserves the original problem and proposal context, then describes
the current async read implementation and its remaining limitations.

Reference points:

- Original proposal: `d6e6c8a0281674238ce71b7065043f9eb48bc14f`
- Implementation baseline reviewed here: `cefa09c215212eb1fed02e2c2ce0cd66973a1200`
  plus the current working tree

The async surface covers random reads:

- `ChunkReader::get_chunks_async()`
- `Reader::get_chunk_reader_async()`
- `Reader::take_async()`
- `ColumnGroupReader::create_async()` and `get_chunks_async(ChunkTask)`
- `ColumnGroupLazyReader::take_async(TakeTask)`
- `FormatReader` async-shaped methods

The following paths remain synchronous:

- `Reader::create()`
- `Reader::get_record_batch_reader()`
- `PackedRecordBatchReader` full scans

There is no `AsyncReadOptions`, no executor argument in the storage API, and no
storage-owned `folly::via()` call. The caller and the format backend determine
where work actually runs.

## Problem

At proposal time, the design started with a direct observation:

> The current read path serializes I/O across column groups (CGs):

```text
load_internal() / ReaderImpl::take()
  └─ for each CG (serial, sync barrier between CGs):
       └─ get_chunks() / take()
            └─ split into N tasks → thread pool → wait all
                 └─ read_chunks_from_files() / take_rows_from_files()
                      └─ FormatReader::read_with_range() / take() (blocking I/O)
```

Each column group made its own task split, submitted work through a
`ThreadPoolHolder`-managed Folly executor, and waited for all of that work before
the next column group could start. The executor provided parallelism inside one
column group, but the outer column-group loop remained serial.

This exposed two core problems:

1. **Column groups could not overlap.** A take spanning multiple column groups
   introduced a synchronous barrier after every group, even when later groups
   could have started independent I/O.
2. **Task planning lacked the full read context.** `split_chunks()` and
   `split_row_indices()` divided item arrays locally. They could split work from
   the same file without first establishing file-aware natural task boundaries,
   increasing repeated opens, row-group reads, and decode work.

In the current implementation, `PackedRecordBatchReader` full scans still process
column groups in a loop. The centralized task model is used by the async random
chunk and take overrides. Synchronous random reads retain their legacy paths, and
converting full scans to the same async model is out of scope.

## Solution: Centralized Task Planning + Async Execution

The original proposal established the core direction: plan work centrally, pass
async operations through the column-group and format layers, and aggregate the
results after fan-out. Several mechanisms changed while the implementation was
refined:

| Original proposal mechanism | Current implementation |
|---|---|
| Column-group `get_natural_tasks()` APIs | `ChunkTask::Build()` and `TakeTask::Build()` in `async_tasks.*` |
| Split toward `ThreadPoolHolder::numThreads()` | Split toward the API `parallelism` target using a configurable policy |
| Storage submits work with `folly::via()` | Storage does not choose a Folly executor |
| Clone an already-open `FormatReader` per task | Open an independent task reader, reusing cached immutable metadata when enabled |

The executor change is a later design decision, not part of the original Problem
statement. Parquet generator work inherits the executor from the consumed Folly
chain. Vortex scan orchestration and collection run on its shared Tokio runtime,
while physical I/O follows the filesystem backend.

The implemented solution has four parts.

1. **Build tasks before format execution.** The async overrides in `reader.cpp` use
   `ChunkTask::Build()` or `TakeTask::Build()` to create file-aware natural tasks.
   Take planning sees all required column groups and produces one flat task list.
2. **Separate natural task construction from optional splitting.** Nested
   `SplitTraits` describe how each task type is bisected, while
   `SplitAsyncTasks()` applies the configured `none`, `parallelism`, or `all`
   policy.
3. **Pass one planned task through each lower layer.** Column-group readers execute
   `ChunkTask` or `TakeTask`; they no longer expose a separate
   `get_natural_tasks()` interface or make async-path parallelism decisions.
4. **Let the format own native execution.** Storage constructs and combines
   `SemiFuture`s without calling `folly::via()`. Parquet generator work inherits
   the executor from the consumed Folly chain, Vortex scan orchestration and
   collection run on the shared Tokio runtime, and formats without native async
   support use the synchronous ready-future fallback.

### Architecture

```text
ChunkReaderImpl::get_chunks_async()
  requested chunks
      |
      v
  sort, deduplicate, validate
      |
      v
  ChunkTask::Build()
  one column group -> file x contiguous-range tasks
      |
      v
  SplitAsyncTasks(strategy, target = parallelism)
      |
      v
  ColumnGroupReader::get_chunks_async(task) for every task
      |
      v
  collectAll -> map by chunk index -> restore request order and duplicates
```

```text
ReaderImpl::take_async()
  requested rows + required column groups
      |
      v
  TakeTask::Build()
  +-------------------+-------------------+-------------------+
  | CG0 / file 0      | CG0 / file 2      | CG1 / file 0      | ...
  | rows + positions  | rows + positions  | rows + positions  |
  +-------------------+-------------------+-------------------+
      |
      v
  SplitAsyncTasks(strategy, target = parallelism)
      |
      v
  for each task
      |
  format-specific reader acquisition
      +-- metadata cache hit: synchronous create_from_metadata()
      +-- Parquet cold/no-cache: deferred blocking open on consumed Folly chain
      +-- Vortex cold/no-cache: deferred loader wrapping a native Tokio open
      `-- Lance/Iceberg: synchronous fallback
      |
  format-specific read
      +-- Parquet: Arrow generator on caller Folly executor
      +-- Vortex: Tokio scan/collection; filesystem-owned physical I/O
      `-- Lance/Iceberg: inline synchronous fallback
      |
  folly::collectAll()
      |
  group by CG -> reorder by original_positions
      |
  build final table
```

There is deliberately no storage-level executor box in this diagram. On the
task-based async path, `parallelism` is the task-splitting target, not a
thread-pool size. Creating all task futures also does not mean that every backend
submitted work to Folly; the backend-specific execution model determines where
the task runs.

### Why It Works

- **Cross-column-group fan-out:** take tasks from all required column groups are
  represented in the same future set before `collectAll()`. On metadata-cache
  hits, a stateful reader is reconstructed synchronously from cached metadata.
  On cold or cache-disabled paths, Parquet defers blocking open work to the
  consumed Folly chain, Vortex wraps a native Tokio open, and Lance/Iceberg may
  still block while task futures are constructed. Native async data reads can
  overlap after each task reader is ready.
- **No nested async storage scheduling:** a planned async storage task is executed
  directly by one column-group reader and one format reader. Lower async layers do
  not create another storage task split.
- **File-aware locality:** chunk natural tasks merge only physically contiguous
  ranges in one file. Take natural tasks keep all requested rows for one
  column-group file together before optional splitting.
- **Explicit granularity trade-off:** `none` keeps natural locality, `all` fully
  fragments work, and the default policy splits only until the requested task
  target is reached when possible.
- **Executor ownership stays outside storage:** callers can select the Folly
  executor used by Parquet generator work. Vortex scan orchestration and
  collection use Tokio, while its physical reads use the filesystem backend.
- **Stable result semantics:** chunk indices and take rows are reconstructed in the
  caller-requested order after asynchronous fan-in.

## Public API Behavior

The base public interfaces provide compatibility defaults:

```cpp
ChunkReader::get_chunks_async(...)
  -> folly::makeSemiFuture(get_chunks(...))

Reader::get_chunk_reader_async(...)
  -> folly::makeSemiFuture(get_chunk_reader(...))

Reader::take_async(...)
  -> folly::makeSemiFuture(take(...))
```

These defaults execute the synchronous call immediately and return a ready
`SemiFuture`. They preserve compatibility for other implementations; they do not
make a synchronous implementation non-blocking.

`ReaderImpl` and `ChunkReaderImpl` provide concrete async overrides. Their async
chunk and take reads use the task-based implementation described below;
chunk-reader creation uses the async-shaped open chain.

The synchronous APIs remain separate and do not call an async override or wait on
one with `.get()`:

- `ChunkReaderImpl::get_chunks()` always calls its synchronous helper, which then
  calls `ColumnGroupReader::get_chunks()`. With `parallelism <= 1` it reads
  directly; with `parallelism > 1` the legacy column-group implementation uses
  `ThreadPoolHolder` and `split_chunks()`.
- `ReaderImpl::take()` always calls `take_tables_sync()` and visits the required
  column groups serially. Within each column group,
  `ColumnGroupLazyReader::take()` reads directly for `parallelism <= 1` or uses
  `ThreadPoolHolder` and `split_row_indices()` for `parallelism > 1`.
- For non-empty input, `ChunkReaderImpl::get_chunks_async()` and
  `ReaderImpl::take_async()` use the centralized task path even when
  `parallelism == 1`. Empty-input and validation failures can return a ready
  future before task construction.

## Async Call Paths

### Chunk Reader Creation

```text
ReaderImpl::get_chunk_reader_async()
  -> construct ChunkReaderImpl
  -> ChunkReaderImpl::open_async()
  -> ColumnGroupReader::create_async()
  -> ColumnGroupReaderImpl::open_async()
  -> open_reader_for_file_async() for each file
       -> cache enabled:
            Parquet/Vortex -> deferred get_or_open_async()
                              -> hit: cached metadata
                              -> miss leader: load_metadata_async()
                              -> synchronous create_from_metadata()
            Lance/Iceberg  -> synchronous metadata-cache path
       -> cache disabled:
            FormatReader::create_async()
```

This chain has mixed execution semantics. On cache-disabled or cache-miss paths,
Parquet defers a blocking open to the inherited Folly context, while Vortex wraps
a native Tokio open. On a cache hit, neither format runs its metadata loader or
calls `FormatReader::open_async()`; `create_from_metadata()` reconstructs reader
state synchronously and may still open backend state. Lance and Iceberg retain
the synchronous compatibility fallback. See "Open and Metadata Loading" below.

### Chunk Reads

```text
ChunkReaderImpl::get_chunks_async(chunk_indices, parallelism)
  -> sort and deduplicate indices for execution
  -> validate chunk bounds
  -> ChunkTask::Build(...)
  -> SplitAsyncTasks(...)
  -> ColumnGroupReader::get_chunks_async(task) for every task
  -> folly::collectAll(...)
  -> map results by chunk index
  -> restore the caller's original order and duplicates
```

`ChunkReader` is already bound to one column group, so `ChunkTask` does not need a
column-group or reader index.

### Take Reads

```text
ReaderImpl::take_async(row_indices, parallelism, needed_columns)
  -> resolve projected columns and required column groups
  -> create one ColumnGroupLazyReader per required column group
  -> TakeTask::Build(column_groups, row_indices)
  -> SplitAsyncTasks(...)
  -> ColumnGroupLazyReader::take_async(task) for every task
  -> folly::collectAll(...)
  -> concatenate results per column group
  -> reorder each column group by original_positions
  -> align projected columns and build the final table
```

Tasks from all required column groups are placed in one list before execution.
This allows file reads from different column groups to overlap when the backend
and the selected executor support it.

## Task Model

Task definitions and builders live in:

- `cpp/include/milvus-storage/format/async_tasks.h`
- `cpp/src/format/async_tasks.cpp`

There are two task types. There is no separate `ColumnGroupTakeTask`.

### ChunkTask

```cpp
struct ChunkTask {
  size_t file_index;
  std::vector<int64_t> chunk_indices;
  uint64_t range_start;
  uint64_t range_end;

  class SplitTraits;

  static std::vector<ChunkTask> Build(...);
};
```

`range_start` and `range_end` form a half-open file row range `[start, end)`.

`ChunkTask::Build()`:

1. Groups requested chunks by file.
2. Preserves the sorted chunk order within each file.
3. Merges adjacent chunks only when their physical row ranges are exactly
   contiguous.
4. Starts a new task when there is a gap, even if both ranges are in the same
   file.

The high-level caller sorts, deduplicates, and validates chunk indices before
calling `Build()`. The final aggregation restores the original order and duplicate
requests.

`ChunkTask::SplitTraits` bisects a task by chunk count. It preserves `file_index`
and updates the left and right range boundary from `ChunkInfo`. Both resulting
tasks therefore remain continuous file ranges.

### TakeTask

```cpp
struct TakeTask {
  size_t reader_index;
  uint32_t file_index;
  std::vector<int64_t> row_indices;
  std::vector<size_t> original_positions;

  class SplitTraits;

  static arrow::Result<std::vector<TakeTask>> Build(...);
};
```

`reader_index` identifies the selected column-group lazy reader. `row_indices`
remain global logical row indices until the task is executed. `original_positions`
records each row's position in the input array.

`TakeTask::Build()` performs planning in this order:

1. Validate that row indices are non-negative, strictly increasing, and unique.
2. Iterate required column groups (`reader_index`).
3. Within each column group, map rows to manifest files.
4. Create one natural task for each `(reader_index, file_index)` pair.

This means take planning first separates column groups, then files, and only then
splits rows inside a file task.

`TakeTask::SplitTraits` bisects `row_indices` by item count and splits
`original_positions` at the same boundary. Both halves retain the same
`reader_index` and `file_index`.

A take task guarantees same-file membership, not contiguous I/O. Sparse rows can
touch multiple row groups, and splitting a file task can make two tasks read the
same row group or open the same file independently.

## Split Policies

The property `reader.async.task_split_strategy` selects one of three strategies:

| Value | Behavior |
|---|---|
| `parallelism` | Default. Split the largest splittable task until task count reaches the requested `parallelism`. |
| `none` | Keep the natural tasks unchanged. |
| `all` | Keep splitting until every task contains one chunk or one row. |

Missing or unrecognized property values select the default `parallelism` policy.
The async callers normalize `parallelism` to at least 1 before splitting.

Important semantics:

- Natural tasks are never merged. The final task count can already be greater
  than `parallelism`.
- The default policy can stop below `parallelism` when no task can be split.
- `all` ignores the requested target count.
- All resulting task futures are created before `collectAll()` waits for them.
  There is no scheduler-side concurrency cap or backpressure queue here.
- `SplitAsyncTasks()` mutates its input vector in place. Current callers pass a
  newly built local vector, so the mutation is contained within planning.

The splitter uses task-specific nested `SplitTraits` and compile-time policy
classes. An `AsyncTaskSplitTraits` concept checks the required `size`, `can_split`,
and `split` operations. Runtime strategy selection dispatches to the instantiated
template policies through a fixed function table.

### Locality Trade-off

For chunk reads, natural tasks maximize continuous range I/O, and every split
preserves continuity.

For take reads, `none` maximizes same-file grouping, while `all` maximizes task
count and can increase repeated file opens, row-group reads, and decode work. The
default `parallelism` policy is the middle ground: retain natural file grouping
unless more task granularity was explicitly requested.

## Column-Group Execution Boundary

The task-based async column-group interfaces execute tasks but do not plan them:

```cpp
ColumnGroupReader::get_chunks_async(const ChunkTask& task)
ColumnGroupLazyReader::take_async(const TakeTask& task)
```

This boundary is specific to the async task path. The legacy synchronous
`ColumnGroupReader::get_chunks()` and `ColumnGroupLazyReader::take()` methods
still split their own inputs and schedule work through `ThreadPoolHolder` when
the passed `parallelism` is greater than 1. A preconfigured singleton can then
determine the actual pool size.

For each task, the implementation opens an independent format reader, validates
that the task belongs to the expected file, converts any global row positions to
file-local positions, and calls the corresponding `FormatReader` async method.

This independent-reader model avoids concurrent mutation of a shared
`FormatReader`. The async task path does not currently call `clone_reader()`.

When metadata caching is enabled, independent readers reuse immutable metadata.
Parquet and Vortex use deferred `get_or_open_async()` lookup and singleflight: a
cache hit returns the existing metadata, while only the miss leader runs the
async loader. Lance and Iceberg use the synchronous cache loader. Every task
still constructs an independent stateful reader from that metadata.

When caching is disabled, each task uses `FormatReader::create_async()`. Parquet
and Vortex follow their format-specific deferred open paths, while Lance and
Iceberg may complete reader creation synchronously.

The public `ChunkReader` API has no predicate argument and constructs its
column-group reader with an empty predicate, so its async chunk path does not
encounter filtered row counts. Predicate handling belongs to the full-scan
`PackedRecordBatchReader` path and has narrower semantics:

- Predicates are ignored when a scan requires more than one column group because
  cross-group row alignment is not implemented.
- Predicate support is format-specific. The base implementation is a no-op;
  Vortex implements predicate parsing and pushdown, while Parquet currently does
  not.
- For a single-group predicate scan, `PackedRecordBatchReader` calls
  `ColumnGroupReader::get_chunk()` one chunk at a time and avoids the batched
  slicing path. Both lower-level batched methods,
  `ColumnGroupReader::get_chunks()` and `get_chunks_async(ChunkTask)`, still slice
  using pre-filter chunk sizes and are not safe with a non-empty predicate.

## FormatReader Contract

`FormatReader` defines:

```cpp
open_async()
create_async(...)
read_with_range_async(...)
take_async(...)
```

The compatibility defaults for `open_async()`, `read_with_range_async()`, and
`take_async()` execute the synchronous operation immediately and return a ready
`SemiFuture`. `FormatReader::create_async()` is different: it dispatches to
`Format::create_reader_async()`. The base `Format` implementation is synchronous,
while `PlainFormat` defers filesystem resolution, reader construction, and
`open_async()` until the future is consumed.

The storage wrappers do not select `ThreadPoolHolder`, call `folly::via()`, or
choose a storage-owned Folly executor. A backend override may still submit work
to its own runtime, as Vortex does with Tokio. Execution follows the selected
format implementation and the consumed Folly chain.

For cache-disabled reader creation, current `Format::create_reader_async()`
behavior is:

| Format | Cache-disabled factory/open | Range/take |
|---|---|---|
| Parquet | Deferred `PlainFormat` factory; `open_async()` defers blocking `open()` to the consumed Folly context | Native async Arrow generator path |
| Vortex | Deferred `PlainFormat` factory; `open_async()` submits the native file open to Tokio | Native async Rust/Tokio scan and collection path |
| Lance | Synchronous `Format` fallback | Synchronous compatibility default |
| Iceberg | Synchronous `Format` fallback | Synchronous compatibility default |

When metadata caching is enabled, both hits and misses bypass the format factory
shown in the table. A miss runs the format's metadata loader: Parquet and Vortex
use deferred `load_metadata_async()`, while Lance and Iceberg use synchronous
`load_metadata()`. A hit skips that metadata load. Both outcomes then call
`create_from_metadata()` synchronously to construct stateful reader state.

Skipping the metadata load and `FormatReader::open_async()` does not mean that no
backend object is opened. Vortex reuses the cached `VortexFile` directly. Parquet
opens a file handle and constructs a new Arrow `FileReader`; that work can block,
and readers configured with a key retriever may need to load their footer again
because Parquet metadata is not cached in that configuration. Lance and Iceberg
can also synchronously open their stateful backend readers from cached metadata.

For Lance and Iceberg, calling an async-shaped method can block before it returns
the `SemiFuture`. Splitting tasks does not make these format calls concurrent,
because each synchronous fallback completes while the task-future list is being
constructed. Synchronous work performed by a deferred Parquet cache-hit or cold
path instead runs when the Folly chain is consumed.

## Parquet Execution Model

Parquet defers generator startup until its `SemiFuture` is consumed:

```text
ParquetFormatReader::{read_with_range_async,take_async}
  -> folly::makeSemiFuture().deferExValue(...)
  -> receive folly::Executor::KeepAlive from the future chain
  -> MakeFollyArrowExecutor(keep_alive)
  -> FileReader::GetRecordBatchGenerator(..., arrow_executor)
  -> arrow::CollectAsyncGenerator(...)
  -> bridge Arrow Future to Folly Promise
```

`FollyArrowExecutor` adapts `folly::Executor` to
`arrow::internal::Executor`:

- Arrow task submission calls `executor_->add(...)`.
- The adapter reports capacity 1 by default. Parallelism primarily comes from
  multiple storage tasks and the caller's executor, rather than unrestricted
  per-task Arrow fan-out.
- Arrow `StopToken` is checked before enqueue and before execution.
- Submission exceptions are converted to Arrow errors.
- The stored `folly::Executor::KeepAlive<>` retains the underlying executor while
  Arrow work is outstanding when that executor supports Folly keep-alive
  ref-counting. Executors without that support provide a dummy keep-alive token
  and do not gain the stronger lifetime guarantee.
- `bridge_arrow_future()` retains the adapter until the Arrow future completes.

Storage never chooses the Folly executor:

- If the consumer attaches `.via(pool)` before consuming the returned future,
  Parquet decode/materialization tasks use that executor. Folly preserves the
  nested deferred executors through `collectAll()`.
- If the consumer calls `.get()` directly, Folly's wait executor drives the
  deferred work on the waiting thread.

Therefore, `parallelism=N` alone does not create N Folly worker threads. Under the
default policy it asks the splitter to reach N task units when possible; the final
count can still be below N when tasks are indivisible or above N when the natural
task count already exceeds N. CPU parallelism requires a multi-thread executor
supplied by the caller. Consuming the async future directly with `.get()` does not
provide that multi-thread executor.

Parquet enables Arrow pre-buffering. For CRT-backed S3 files, network I/O is driven
by CRT async requests; the Folly adapter is used for Arrow generator work such as
decode/materialization, not for the CRT network event loop. Other filesystems use
their own Arrow/file implementation behavior.

The async generator path does not use `arrow::internal::GetCpuThreadPool()`.

## Vortex Execution Model

Vortex uses a Folly promise only as the C++ result bridge:

```text
VortexFormatReader::{read_with_range_async,take_async}
  -> build Vortex scan and C callback context
  -> vortex_scan_collect_async(...)
  -> Rust takes ownership of the scan builder
  -> TOKIO_RT.spawn(async scan and collection)
  -> Rust invokes the C callback
  -> callback imports Arrow data and fulfills the Folly promise
```

`TOKIO_RT` is the shared process-wide Tokio runtime used by the Rust bridge.
`VORTEX_RT` is only a Vortex runtime adapter backed by the same Tokio handle; it is
not a second runtime.

Vortex scan and materialization do not inherit a caller-supplied Folly executor.
Attaching `.via()` can control later Folly continuations, but it does not move the
Rust scan or collection onto that Folly executor. A successful scan callback, and
failures produced after the Tokio task has been spawned, are invoked from that
Tokio task. Setup failures detected before `TOKIO_RT.spawn()` may invoke the C
callback synchronously on the calling thread, so the callback itself does not
have an unconditional Tokio-thread guarantee.

The range-read builder uses the same 8 MiB maximum coalescing window as the
synchronous path. The first range read can synchronously create and cache an
alternate Vortex view because the file was initially opened with the default
1 MiB window. This opens another filesystem reader handle but reuses the loaded
footer without issuing footer or data reads while the view is created. Scan
orchestration and collection then run on Tokio. Physical reads follow the
filesystem backend: a CRT-capable async reader uses CRT requests and callbacks;
otherwise Vortex offloads synchronous `ReadAt()` calls to the Tokio blocking
pool.

The current Vortex async range implementation collects all batches before
fulfilling the future, then exposes them through an Arrow reader. It is async with
respect to completion, but it is not a progressively available streaming result.

## Runtime Configuration

`ConfigureStorageRuntime(cpu_threads, io_threads)` configures process-wide runtime
resources before first Rust runtime use:

- Arrow CPU pool capacity: `cpu_threads`
- Arrow I/O pool capacity: `io_threads`
- Tokio worker threads: `cpu_threads`
- Tokio maximum blocking-thread limit: `io_threads`

If the Rust runtime is not explicitly configured, both Tokio limits default to
`std::thread::available_parallelism()`, falling back to 32 if that query fails.

This configuration does not create or select a Folly executor for async reads.
Parquet's explicit `FollyArrowExecutor` bypasses the global Arrow CPU pool for its
async generator tasks. Vortex scan orchestration and collection run on Tokio;
physical I/O remains owned by the selected filesystem backend.

## Open and Metadata Loading

`FormatReader::create_async()` dispatches to `Format::create_reader_async()`; it
does not call `FormatReader::create()` directly. `PlainFormat` defers filesystem
resolution, reader construction, and `open_async()` until future consumption.
This is the cache-disabled path used to create a fresh format reader.

With metadata caching enabled, reader acquisition takes a different path:

- Parquet and Vortex call deferred `get_or_open_async()`. Lookup occurs when the
  returned Folly chain is consumed. A hit returns cached immutable metadata; a
  miss leader runs `load_metadata_async()`, and same-key followers wait on the
  shared result without blocking a thread.
- After either a hit or a successful miss, `create_from_metadata()` runs
  synchronously to create independent stateful reader state.
- Lance and Iceberg use their synchronous metadata-cache loaders and synchronous
  `create_from_metadata()` implementations.

On a Parquet cache miss or cache-disabled path, `open_async()` is lazy but
internally calls blocking `open()`. Its one-read footer fast path uses
`ReadAtAsyncInto()` only when the build enables CRT, the opened file implements
`NonBlockingReadAtFile`, no encryption key retriever is active, and valid manifest
`file_size` and `footer_size` values are available. `open()` then waits on
`.result()`: CRT drives the network request, but the Folly executor or waiting
thread consuming the deferred operation remains blocked. Other cases use
synchronous `ReadAt()` or Arrow's normal footer path.

On a Vortex cache miss or cache-disabled path, filesystem lookup and URI parsing
run in the consumed Folly continuation. `open_async()` then submits the native
file open to the shared Tokio runtime, and immutable metadata is snapshotted after
that open succeeds. On a cache hit, `create_from_metadata()` reuses the cached
`VortexFile` and does not submit another Tokio open.

Parquet cache hits also bypass `open_async()`, but synchronous
`create_from_metadata()` constructs a new Arrow `FileReader`. When no key
retriever is configured, cached Parquet footer metadata is reused. When a key
retriever is configured, footer metadata is intentionally not cached, so reader
construction may perform footer I/O again even if the underlying file is
plaintext.

Consequently, `Reader::get_chunk_reader_async()` has cache- and format-dependent
behavior. Cold/no-cache Vortex opens natively on Tokio; cold/no-cache Parquet
defers a blocking open to the consumed Folly context. Warm cache hits skip
metadata loading and `FormatReader::open_async()` but still reconstruct reader
state synchronously, which may include opening a file handle or backend reader.
Lance and Iceberg may block before the composed future is returned.

## ThreadPoolHolder and Full Scans

The task-based async path does not use `ThreadPoolHolder` and does not call
`folly::via()`.

`ThreadPoolHolder` still exists in legacy synchronous paths:

- `ColumnGroupReader::get_chunks(..., parallelism)`
- `ColumnGroupLazyReader::take(..., parallelism)`
- `PackedRecordBatchReader`, which reads its configured parallelism from the
  singleton pool

`PackedRecordBatchReader::load_internal()` still processes column groups in a
loop. A column group can perform internal parallel reads, but the full-scan path
does not use the centralized cross-column-group async task list described for
`Reader::take_async()`.

## Result Ordering

Chunk reads execute sorted unique chunk indices, then rebuild the output from a
chunk-index map. The caller receives the same order and duplicate positions as in
the original request.

Take planning groups rows by column group and file, and splitting appends new tasks
to the task list. Completion order therefore cannot be used as output order.
`original_positions` is carried with every task and split. After `collectAll()`,
each column group's tables are concatenated and reordered with
`CopySelectedRows()` before columns are combined into the final table.

## Error Handling and Lifetime

`FOLLY_ARROW_RETURN_NOT_OK` and `FOLLY_ARROW_ASSIGN_OR_RAISE` in
`common/arrow_util.h` adapt Arrow errors to functions returning
`folly::SemiFuture<arrow::Result<T>>`. `FollyArrowErrorFuture` performs the
result-type conversion without repeating the concrete `T` at every return site.

High-level fan-in uses `folly::collectAll()`:

- It waits for every task, even when one task has already failed.
- Folly exceptions are converted to Arrow `IOError`.
- Arrow `Status` errors propagate through `arrow::Result`.
- Reader and format-reader lifetimes are extended by captures in the relevant
  continuations or callback contexts.

There is no end-to-end cancellation bridge from the public Folly future:

- Parquet honors Arrow stop tokens for individual executor tasks, but public
  Folly cancellation is not connected to those tokens.
- Vortex discards the Tokio `JoinHandle`; dropping the Folly future does not
  cancel the Rust scan.

## Current Limitations

1. On cache-miss or cache-disabled paths, only Vortex has a native non-blocking
   metadata/file open. Parquet defers a blocking open, while Lance and Iceberg
   retain synchronous create/open and metadata loading. Cache hits skip metadata
   loading and `FormatReader::open_async()` but may still synchronously open a
   file handle or backend reader while reconstructing stateful reader state.
2. Lance and Iceberg do not have native async range/take implementations.
3. On the task-based async APIs, `parallelism` is only a task-count target; it
   does not guarantee CPU concurrency or cap the number of in-flight backend
   operations. The synchronous APIs retain their separate thread-pool semantics.
4. `SplitAll` can create large future lists and repeat file or row-group work.
5. Vortex async range reads collect the complete result before completing.
6. The public async path has no end-to-end cancellation propagation.
7. Full scans remain on the synchronous `PackedRecordBatchReader` path.
8. The public random-read APIs do not expose predicates. Full-scan predicate
   pushdown is limited to one column group and formats that implement it
   (currently Vortex). The lower-level batched synchronous and async chunk methods
   are not safe with a non-empty predicate because they slice using pre-filter
   chunk sizes.

## Test Coverage

Relevant tests include:

- `cpp/test/format/async_tasks_test.cpp`: task builders, validation, nested split
  traits, and async Arrow error macros
- `cpp/test/format/column_groups_wr_test.cpp`: task-level chunk and take execution
- `cpp/test/api_writer_reader_test.cpp`: public async APIs, Parquet caller-executor
  propagation, and deferred reader creation from cached metadata
- `cpp/test/format/parquet/folly_arrow_executor_test.cpp`: Folly-to-Arrow executor
  routing, `TransferAlways`, and direct `.get()` behavior
- `cpp/test/format/format_reader_test.cpp`: Parquet/Vortex async open behavior,
  deferred `PlainFormat` creation, async metadata loading, Parquet executor
  routing, and synchronous creation of the Vortex large-window cached view
- `cpp/test/format/format_reader_cache_test.cpp`: deferred cache loading,
  same-key singleflight, sync/async interaction, lifetime, and failure retry
- `cpp/test/format/vortex/vortex_basic_test.cpp`: Tokio scan panic propagated
  through the async callback
- `cpp/test/format/vortex/vortex_v2_test.cpp`: async row-index validation before
  the Vortex FFI handoff

The current tests do not directly cover every split policy's final task count,
backpressure/concurrency limits, successful Vortex scan callback thread behavior,
the physical request shape produced by the 8 MiB Vortex coalescing window, a real
CRT event-loop footer read, or end-to-end cancellation.
