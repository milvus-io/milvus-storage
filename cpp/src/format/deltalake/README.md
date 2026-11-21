# Delta Lake format

`deltalake-table` reads Parquet-backed Delta Lake tables as external collections. It resolves
snapshot membership, column mappings, partition values, schema evolution, and deletion vectors
from the Delta log. It supports inline, UUID-relative, and absolute-URL deletion vectors,
including multiple vectors at distinct offsets in one file.

## Snapshot and reader contract

- Set `reader.exttable.snapshot_id` to `-1` to resolve the latest snapshot during exploration,
  or to a nonnegative version, including `0`, to select an exact snapshot.
- Exploration records a versioned descriptor with the table URI and snapshot version alongside
  each file's path, size, and live row count. Readers always reopen the pinned version. Source
  owners must retain its immutable logs, checkpoints, data files, and deletion vectors.
- Select top-level logical fields with `read_schema` or `needed_columns`. An explicit
  `read_schema` controls field order and must use compatible types. An empty `read_schema`
  requests zero columns. An empty `needed_columns` with no `read_schema` requests all fields.
  `get_schema()` returns the full snapshot schema.
- Each file descriptor covers the whole file's live rows starting at zero. Use the reader's
  range API to read a subset. For `take`, supply sorted, unique live-row indices after deletion
  vector filtering.
- Async entry points use synchronous fallbacks. Apply predicates to the returned rows in the caller.
  Writes, change data feeds, incremental reads, timestamp-based time travel, and catalog discovery
  are outside the read API.

## Schema evolution

Exploration validates logical types and recorded `delta.typeChanges` across the snapshot
schema. File open validates compatibility with the full visible schema, including fields outside
the requested projection. The reader handles `name` and `id` column mappings using the pinned
Kernel's ID-first matching with physical-name fallback. Files with missing or conflicting field
IDs can therefore resolve differently from strict Delta column-mapping mode rules.

When an existing struct has no matching child fields in the requested schema, the reader
preserves its validity. For example, `{old: 7}` becomes `{new: null}`, and a null struct stays
null. Enclosing lists and maps keep their offsets and the distinction between empty and null.
Map keys stay associated with their values. A nullable container absent from the file yields a
typed null. Rust aligns the physical arrays before Kernel fixup to work around the nullable
struct behavior tracked by [Kernel issue #3961](https://github.com/delta-io/delta-kernel-rs/issues/3961).

| Delta type | Supported Arrow representation |
| --- | --- |
| Boolean | Boolean |
| Byte, short, integer, long | Signed 8-, 16-, 32-, and 64-bit integers |
| Float, double | 32- and 64-bit floats |
| String, binary | Ordinary Utf8 and Binary |
| Decimal | Decimal128, precision 1 to 38 and scale 0 to precision |
| Date | Date32 |
| Timestamp, timestamp_ntz | Microseconds with UTC or without a timezone, respectively |
| Struct, array, map | Recursive Struct, List, and Map with non-null map keys |
| Void, missing nullable fields | Null and typed-null arrays |

Variant and shredding, file-reference types, geospatial types, UDTs, intervals, unsigned integers,
Decimal256, fixed-size arrays, and unions are outside the supported subset. Callers can select and
order fields. Partial struct projections and caller-defined casts are unsupported.

These lossless widening rules apply to Parquet data fields, including nested fields.

| Source | Supported targets |
| --- | --- |
| Byte | Short, integer, long, double, or sufficiently large decimal |
| Short | Integer, long, double, or sufficiently large decimal |
| Integer | Long, double, or sufficiently large decimal |
| Long | Sufficiently large decimal |
| Float | Double |
| Decimal | Greater precision or scale without reducing scale or integer digits |
| Date | Timestamp_ntz |

### Partition conversion

Primitive partition values come from snapshot metadata using mapped physical names.

- Missing values become typed nulls. Empty strings stay empty for string and binary partitions
  and become null for other primitive types. Binary values use the UTF-8 bytes of the JSON string.
- Decimal text must match the declared scale. Nonempty `timestamp` values require an explicit
  offset or UTC suffix, such as `2024-01-01T00:00:00Z`. Use timezone-free text for `timestamp_ntz`.
- Partition widening supports integer promotions, including byte, short, and integer to double,
  decimal precision growth at the same scale, and integer-to-decimal at scale zero.
- The reader rejects recorded float-to-double partition widening. Parsing an old float literal
  such as `0.1` as double can change its value because the text omits the original float rounding.
  Decimal scale increases, integer-to-decimal with nonzero scale, and date-to-`timestamp_ntz`
  partition changes also require conversions outside the supported subset.

## Filesystem requirements

- Logs, checkpoints, sidecars, data files, and deletion vectors must share one configured
  filesystem identity. Source URLs must omit credentials, queries, and fragments. Local file
  URLs must omit a host.
- Local URLs use the configured root and path spelling after absolute lexical normalization.
  Keep roots and symlink aliases stable while cached filesystems or readers are in use. Relative
  roots also require a stable working directory. Use relative source entries or absolute URLs
  under the configured logical root.
- When endpoints share a bucket name, give their destination collections distinct manifest
  paths. The shared manifest cache keys on the filesystem root or bucket and manifest path.
  Milvus supplies this uniqueness through collection, partition, and segment directories.
  [Manifest cache discussion](https://github.com/milvus-io/milvus-storage/pull/800#issuecomment-6936593703)

## I/O and caching

- Exploration uses snapshot row counts when available, including parsed checkpoint counts, and
  falls back to Parquet footers when needed. Files with encrypted footers require snapshot row
  counts because exploration has no key callback.
- On a per-file cache miss, file open validates the footer and decodes deletion vectors. C++
  reads selected physical top-level columns using the ordinary Parquet reader and handles
  decryption and live-row filtering. Rust aligns those arrays to Kernel's physical schema and
  fills missing nullable fields before Kernel casts and logical transforms.
- Recovering an existing container's shape requires reading its top-level column. A selected
  root can also contain dropped children, so scheduling counts the full compressed bytes of all
  selected roots. Projections that need only constants or typed nulls can skip payload reads.
  Fully deleted groups also report zero payload bytes. Output-memory estimates can be
  unavailable after schema adaptation.
- A reader with a key callback initializes its own decryptors, including for plaintext files.
  It can reread a footer after validation, and clones repeat this setup. The ordinary Parquet
  metadata cache retains reusable footers for readers without a callback.
- The metadata cache holds the file schema, row-group metadata, decoded deletion positions, and
  a shared snapshot reference per file. Projections and clones reuse file metadata. Independent
  readers can repeat footer and deletion-vector reads. Outstanding chunk, range, and clone
  readers can extend metadata lifetimes.
- A cold snapshot load indexes all active files. File transforms are built lazily from indexed
  metadata rows. Once the last snapshot owner releases it, a later reader can need to replay
  the pinned logs. Per-file `cache_size` estimates exclude shared snapshot state. The metadata
  cache has no byte-bounded eviction.

## Validation

Generated native and Rust tests cover nested schema evolution, mapping and widening, encryption,
deletion vectors, projected read sizes, and reader and cache lifetimes. Shared metadata-cache
tests check footer and DV reuse and run concurrent reads across the supported formats.

Local Python tests cover manifest round trips and reopening pinned snapshots in fresh processes.
Both MinIO and RustFS pass the 15 Delta integration cases and related regressions, including
endpoint and credential isolation, encoded paths, shared DV containers, and missing-object recovery.
These tests use generated fixtures. Independent Spark or delta-rs producers and live cloud services
remain unvalidated.
