# Asynchronous V2 packed Parquet reads

## Problem and scope

V2 callers currently open `PreparedParquetFile`, create a `FileRowGroupReader`,
and call `ReadNextRowGroup()`. Opening the footer and reading row groups can wait
for storage on the calling thread. Moving that entire operation to a shared CPU
executor would also move the blocking wait there.

This change adds two lazy C++ APIs on the existing prepared-file context:

- `PreparedParquetFile::OpenAsync(fs, path, reader_properties)`
- `prepared->ReadRowGroupsAsync(offset, count, schema, arrow_properties)`

Both return `folly::SemiFuture<arrow::Result<...>>`. The second returns one table
per requested row group in file order. It opens independent reader state per
operation rather than sharing the synchronous reader's cursor. Existing `Open`,
`MakeReader`, `SetRowGroupOffsetAndCount` and `ReadNextRowGroup` remain available.
C++ consumers must rebuild against the updated class definition.
No persisted format, filesystem selection, default pool or Milvus configuration
changes in this PR.

## Execution

The consumer supplies a non-inline Folly executor through `.via()` or its
coroutine context. Storage does not create a CPU pool or select the global Arrow
CPU executor.

```text
consumer executor
  open file off the consumer executor
    -> filesystem IOContext executor runs OpenInputFile
  obtain size asynchronously
    -> NonBlockingRandomAccessFile::GetSizeAsync, or file IOContext executor
  inspect trailer and open Parquet footer asynchronously
    -> existing async Parquet open machinery
  validate and publish the prepared context

consumer executor, for each admitted read range
  open independent reader, reuse trusted plaintext metadata when possible
  enable pre-buffering, select Parquet leaf columns for requested field IDs
    -> GetRecordBatchGenerator -> WhenBuffered
    -> filesystem ReadAsync/ReadManyAsync
  decode on the consumer executor via MakeFollyArrowExecutor
  collect batches, restore requested field order/nulls, slice per row group
```

Arrow allows `OpenInputFileAsync` to execute synchronously, so handle opening is
explicitly submitted to the filesystem I/O executor.

In Arrow 17, `RandomAccessFile::ReadAsync` defaults to submitting synchronous
`ReadAt` to its I/O executor. Thus the SDK backend still occupies I/O threads, but
not the consumer's CPU worker while waiting for network data. Native async
backends use their own implementation. This change does not enable CRT or alter
TLS requirements.

Pre-buffering is mandatory for the new range-read API, even when the supplied
Arrow properties disable it. Otherwise Arrow's generator can submit blocking
reads to the decode executor. Coalescing/cache options and other caller-supplied
Arrow properties are retained. The existing V3 open helper gains an optional
trusted metadata input; its existing callers keep the same behavior.

The private helpers live with the existing Parquet implementation so footer
opening, future transfer and generator executor adaptation are reused rather
than reimplemented as a separate storage scheduler.

## Compatibility and ownership

- Files are immutable for the lifetime of a prepared context. A rewritten file
  requires a new context.
- Async and sync open share packed metadata and field-ID validation. A normal
  first open always runs Parquet's encryption/plaintext policy checks.
- Plaintext footer metadata is shared read-only. Encrypted footers (`PARE`) and
  plaintext footers with encrypted columns are not cached; each operation opens
  with an independent clone of the decryption properties. User key callbacks
  must support concurrent invocation, as with synchronous prepared readers.
- Schema projection uses field IDs. Present fields retain the stored field
  definition; missing fields become nullable null columns. Nested fields expand
  to their Parquet leaf columns. Missing-only projections need no data-page I/O.
- Offset/count bounds are checked without integer overflow. Packed row counts
  must agree with the selected Parquet row groups before materialization.
- Operations retain the prepared context, filesystem, reader and executor while
  needed. A caller may release its context after creating the lazy future.
- The executor must continue accepting work until operations complete. Inline
  executors are rejected. Do not block a worker waiting for work queued to that
  same executor. Direct `get()` belongs outside the consumer pool.
- There is no new cancellation protocol. Dropping a consumed future does not
  cancel already issued I/O. Each range operation tracks issued reads and drains
  them before reporting a read/decode failure, because Arrow's `AllComplete` can
  fail before sibling reads finish. Callers must keep their admission reservation
  until completion; abandoning a result is not an early resource-release API.

## Memory and Milvus integration

The range API materializes all selected row groups; it is not a progressive
stream and has no independent memory-limit parameter. Callers split ranges and
reserve memory before invoking it. Returned tables may share Arrow backing
buffers, so they must remain accounted for until consumed.

A follow-up Milvus change can bind these futures to `MILVUS_ASYNC_LOAD`, retain
HIGH/LOW task priorities, await admission and read completion, and finalize
chunks. Local mmap writes remain a separate local-I/O concern. Updating Milvus's
V2 translators, `LoadWithStrategy` and legacy thread-count estimates is not part
of this storage-library PR.

## Validation

Prepared-file tests cover lazy execution, ownership across deferred execution,
plaintext footer reuse, independent ranges, field-ID projection, missing fields
(including a name collision), nested leaf-column projection, invalid ranges,
I/O failure and retry, plaintext decryption policy, both encrypted footer modes,
wrong keys, corrupt metadata and rejection of inline executors.

The scheduling tests deliberately hold a synchronous filesystem read pending
and require unrelated work to complete on the same one-worker CPU executor,
then release I/O and check the result. They cover handle opening, footer opening and data reads;
the data test explicitly disables pre-buffering in the supplied properties.
A separate test fails one range while holding sibling I/O pending and requires
error completion to wait for those reads. These validate the SDK-style Arrow fallback without needing cloud credentials.
Provider-specific network behavior and Milvus end-to-end loading remain separate
integration checks.
