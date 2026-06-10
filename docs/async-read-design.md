# Async Read Design (Current Implementation)

## Status and Scope

This document preserves the original problem and proposal context, then describes
the current async read implementation and its remaining limitations.

Reference points:

- Original proposal: `d6e6c8a0281674238ce71b7065043f9eb48bc14f`
- Implementation baseline reviewed here: `9023e2684c047aa747baf12aa33fe78ed05aa73f`
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
column groups in a loop. This design addresses random chunk and take reads;
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
statement. Parquet now inherits the executor from the consumed Folly chain, while
Vortex keeps execution on its native Tokio runtime.

The implemented solution has four parts.

1. **Build tasks before format execution.** `reader.cpp` uses
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
   `SemiFuture`s without calling `folly::via()`. Parquet inherits the executor from
   the consumed Folly chain, Vortex runs on the shared Tokio runtime, and formats
   without native async support use the synchronous ready-future fallback.

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
  synchronous reader create/open
  cached metadata may be reused
      |
      +--------------------+--------------------+
      |                    |                    |
      v                    v                    v
  Parquet native        Vortex native    default sync fallback
  Arrow generator       TOKIO_RT         inline synchronous call
  caller Folly exec     Rust callback    ready future
      |                    |                    |
      +--------------------+--------------------+
                           |
                           v
                   folly::collectAll()
                           |
                           v
              group by CG -> reorder by original_positions
                           |
                           v
                    build final table
```

There is deliberately no storage-level executor box in this diagram.
`parallelism` is the task-splitting target, not a thread-pool size. Creating all
task futures also does not mean that every backend submitted work to Folly; the
backend-specific execution model determines where the task runs.

### Why It Works

- **Cross-column-group fan-out:** take tasks from all required column groups are
  represented in the same future set before `collectAll()`. Native async backends
  can overlap data reads after each task's synchronous reader creation completes.
- **No nested storage scheduling:** a planned storage task is executed directly by
  one column-group reader and one format reader. Lower layers do not create another
  storage task split.
- **File-aware locality:** chunk natural tasks merge only physically contiguous
  ranges in one file. Take natural tasks keep all requested rows for one
  column-group file together before optional splitting.
- **Explicit granularity trade-off:** `none` keeps natural locality, `all` fully
  fragments work, and the default policy splits only until the requested task
  target is reached when possible.
- **Executor ownership stays outside storage:** callers can select the Folly
  executor used by Parquet, while Vortex continues to use its native Tokio
  runtime.
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

`ReaderImpl` and `ChunkReaderImpl` provide concrete overrides. Chunk and take
reads use the task-based implementation described below; chunk-reader creation
uses the async-shaped open chain.

The synchronous APIs remain separate:

- `ChunkReaderImpl::get_chunks(..., parallelism <= 1)` uses the synchronous path.
- `ChunkReaderImpl::get_chunks(..., parallelism > 1)` calls the async task path and
  waits with `.get()`.
- `ReaderImpl::take(..., parallelism <= 1)` reads each required column group
  synchronously.
- `ReaderImpl::take(..., parallelism > 1)` calls the async task path and waits with
  `.get()`.
- For non-empty input with available column groups, `ReaderImpl::take_async()`
  uses the async task path even when `parallelism == 1`. Empty-input and validation
  failures can return a ready future before task construction.

## Async Call Paths

### Chunk Reader Creation

```text
ReaderImpl::get_chunk_reader_async()
  -> construct ChunkReaderImpl
  -> ChunkReaderImpl::open_async()
  -> ColumnGroupReader::create_async()
  -> ColumnGroupReaderImpl::open_async()
  -> FormatReader::create_async() for each file
```

This path is async-shaped, but format-reader creation and open are currently
synchronous. See "Open and Metadata Loading" below.

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

`parallelism` is normalized to at least 1 before splitting.

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

The column-group interfaces execute tasks but do not plan them:

```cpp
ColumnGroupReader::get_chunks_async(const ChunkTask& task)
ColumnGroupLazyReader::take_async(const TakeTask& task)
```

For each task, the implementation opens an independent format reader, validates
that the task belongs to the expected file, converts any global row positions to
file-local positions, and calls the corresponding `FormatReader` async method.

This independent-reader model avoids concurrent mutation of a shared
`FormatReader`. The async task path does not currently call `clone_reader()`.

When metadata caching is enabled, independent readers can reuse immutable cached
metadata. When it is disabled, multiple tasks for the same file can repeat reader
creation and footer loading.

## FormatReader Contract

`FormatReader` defines:

```cpp
open_async()
create_async(...)
read_with_range_async(...)
take_async(...)
```

The default implementations call the synchronous operation immediately and wrap
its result with `folly::makeSemiFuture(...)`. The wrapper itself does not select
`ThreadPoolHolder`, call `folly::via()`, or choose another executor. Any runtime
used inside the synchronous operation belongs to that backend's sync path.

Current format behavior is:

| Format | Open/create | Range/take |
|---|---|---|
| Parquet | Synchronous ready-future wrapper | Native async Arrow generator path |
| Vortex | Synchronous ready-future wrapper | Native async Rust/Tokio scan path |
| Lance | Synchronous ready-future wrapper | Synchronous compatibility default |
| Iceberg | Synchronous ready-future wrapper | Synchronous compatibility default |

For Lance and Iceberg, calling an async-shaped method can block before it returns
the `SemiFuture`. Splitting tasks does not make these format calls concurrent,
because each synchronous fallback completes while the task-future list is being
constructed.

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
- The stored `folly::Executor::KeepAlive<>` prevents the underlying executor from
  being destroyed while Arrow work is outstanding.
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
supplied by the caller. A synchronous wrapper that calls `.get()` directly does
not provide that executor.

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
Rust scan or its callback away from Tokio.

The current Vortex async range implementation collects all batches before
fulfilling the future, then exposes them through an Arrow reader. It is async with
respect to completion, but it is not a progressively available streaming result.

## Runtime Configuration

`ConfigureStorageRuntime(cpu_threads, io_threads)` configures process-wide runtime
resources before first Rust runtime use:

- Arrow CPU pool capacity: `cpu_threads`
- Arrow I/O pool capacity: `io_threads`
- Tokio worker threads: `cpu_threads`
- Tokio blocking threads: `io_threads`

If the Rust runtime is not explicitly configured, both Tokio limits default to
`std::thread::available_parallelism()`.

This configuration does not create or select a Folly executor for async reads.
Parquet's explicit `FollyArrowExecutor` bypasses the global Arrow CPU pool for its
async generator tasks, and Vortex runs on Tokio.

## Open and Metadata Loading

Open and metadata loading are not truly asynchronous in the current
implementation:

- `FormatReader::open_async()` calls `open()` before returning a ready future.
- `FormatReader::create_async()` calls `FormatReader::create()` before returning a
  ready future.
- `Format::create_reader()` creates and opens the format reader synchronously.
- `ColumnGroupReaderImpl::open_async()` composes these ready futures, so file opens
  are not moved to an executor or made concurrent by `collectAll()`.
- `ColumnGroupLazyReader::take_async()` opens its independent task reader
  synchronously before calling the backend's async take method.

For Parquet with CRT, the footer fast path may use `ReadAtAsyncInto()`, but it waits
on `.result()` during open. The network operation uses CRT, while the public open
call remains blocking.

Consequently, `Reader::get_chunk_reader_async()` does not defer footer I/O. In the
current implementation, format-reader create/open work runs synchronously while
the method constructs its composed future.

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
`arrow::compute::Take()` before columns are combined into the final table.

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

1. Open/footer loading is synchronous despite async-shaped APIs.
2. Lance and Iceberg do not have native async range/take implementations.
3. `parallelism` is only a task-count target; it does not guarantee CPU
   concurrency or cap the number of in-flight backend operations.
4. `SplitAll` can create large future lists and repeat file or row-group work.
5. Vortex async range reads collect the complete result before completing.
6. The public async path has no end-to-end cancellation propagation.
7. Full scans remain on the synchronous `PackedRecordBatchReader` path.

## Test Coverage

Relevant tests include:

- `cpp/test/format/async_tasks_test.cpp`: task builders, validation, nested split
  traits, and async Arrow error macros
- `cpp/test/format/column_groups_wr_test.cpp`: task-level chunk and take execution
- `cpp/test/api_writer_reader_test.cpp`: public async API smoke coverage
- `cpp/test/format/parquet/folly_arrow_executor_test.cpp`: Folly-to-Arrow executor
  routing, `TransferAlways`, and direct `.get()` behavior
- `cpp/test/format/format_reader_test.cpp`: Parquet decode submission to a
  caller-supplied executor
- `cpp/test/format/vortex/vortex_v2_test.cpp`: Vortex async error propagation

The current tests do not directly cover every split policy's final task count,
backpressure/concurrency limits, successful Vortex callback thread behavior, or
end-to-end cancellation.
