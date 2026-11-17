# Delta Lake format

`deltalake-table` is a read-only external collection format.

## Known limitations

### Existing containers with no matching physical leaves

The reader currently retains the behavior of the pinned
[`delta-kernel-rs` revision](https://github.com/delta-io/delta-kernel-rs/commit/913389cbf94528225c70b2a396a9408f41dd9954).
If an existing container has no matching requested physical leaves in a data
file, Kernel can reconstruct that container as null. For nullable structs this
loses the existing parent's validity. For example, reading
`profile: struct<new: string>` from a file containing
`profile: struct<old: long>` produces:

| Stored value | Required value-preserving result | Current Kernel result |
| --- | --- | --- |
| `{old: 7}` | `{new: null}` | `null` |
| `{old: null}` | `{new: null}` | `null` |
| `null` | `null` | `null` |

The same zero-leaf planning affects enclosing lists. Reading
`items: array<struct<new: string>>` from
`items: array<struct<old: long>>` can produce:

| Stored items | Required value-preserving result | Current Kernel result |
| --- | --- | --- |
| `[{old: 7}]` | `[{new: null}]` | `null` |
| `[]` | `[]` | `null` |
| `null` | `null` | `null` |

This loses list lengths and the distinction between empty and null lists. The
behavior also applies recursively to affected structs inside lists or map
values. Maps whose key leaves remain selected retain their map boundaries;
this does not mean every affected map loses its offsets or becomes null.

The nullable-struct case is tracked in
[Kernel issue #3961](https://github.com/delta-io/delta-kernel-rs/issues/3961).
The [in-memory Kernel tests](../bridge/rust/src/deltalake/deltalake_reader.rs)
record the pinned behavior; the desired-semantics struct reproduction is ignored.
The reader retains this upstream container behavior without a local workaround
or special rejection. Adding nullable children alongside retained physical
children does not trigger the zero-leaf case. The missing-container plan originates
from
[Kernel PR #2170](https://github.com/delta-io/delta-kernel-rs/pull/2170).

### Feature and API scope

- Only Parquet-backed Delta tables are supported. Writes, CDF/incremental read
  APIs, timestamp-based time travel, and catalog discovery are not implemented.
  Column mapping in both `name` and `id` modes and inline, UUID-relative, and
  absolute-URL deletion vectors are supported.
- Predicate strings passed to the format reader are not evaluated or pushed
  down. Callers must apply their own filtering to the returned live rows.
- Column selection uses top-level logical names. A supplied `read_schema`
  selects/orders fields but cannot change their types or request only part of a
  struct. Otherwise, nonempty `needed_columns` selects fields; an empty list
  selects the full schema. An explicit empty `read_schema` selects zero columns.
  `get_schema()` always returns the full pinned snapshot schema.
- Async entry points use synchronous fallbacks and may block their executing
  thread. Native asynchronous reading is deferred.

### Types and schema evolution

The initial logical type and Arrow layout subset is:

| Delta type | Supported representation |
| --- | --- |
| Boolean; byte/short/integer/long; float/double | Boolean; signed 8/16/32/64-bit integers; 32/64-bit floats |
| String; binary | Ordinary Utf8/Binary layouts |
| Decimal | Decimal128, precision 1..38 and scale 0..precision |
| Date | Date32 |
| Timestamp; timestamp_ntz | Microseconds with UTC; microseconds without timezone |
| Struct; array; map | Recursive Struct/List/Map, with non-null map keys |
| Void; missing nullable fields | Null/typed-null arrays |

Variant/shredding, file-reference types, geospatial/UDT types, and intervals are
outside this subset. Unsigned integers, Decimal256, fixed-size arrays, and unions
are outside the supported layout contract. Caller-defined casts are not applied.
Nested adaptation follows the pinned Kernel's behavior, including the limitation
above.

Supported widening is limited to these lossless families:

| Source | Target |
| --- | --- |
| Byte | Short, integer, long, double, or a sufficiently large decimal |
| Short | Integer, long, double, or a sufficiently large decimal |
| Integer | Long, double, or a sufficiently large decimal |
| Long | A sufficiently large decimal |
| Float | Double |
| Decimal | Greater precision/scale without losing integer digits |
| Date | Timestamp_ntz |

These rules also apply to nested data fields. Exploration validates all logical
types and recorded `delta.typeChanges`, including unprojected fields. File open
validates compatibility with the full visible snapshot schema, so selecting fewer
columns does not bypass incompatible types or missing required fields. Dropped
physical fields are ignored, and missing nullable fields are synthesized.

### Partition conversion

Partition values come from snapshot metadata using mapped physical names, never
from directory names. Only primitive partition types are supported.

- Missing/null values become typed nulls. Empty strings remain empty for
  string/binary partitions and become null for other primitive types.
- Binary values use the JSON string's UTF-8 bytes; no Base64 decoding is applied.
- Decimal text must match the declared scale exactly; values are not rounded or
  rescaled automatically.
- Nonempty `timestamp` values require an explicit offset/UTC suffix, for example
  `2024-01-01T00:00:00Z`. Offset-less timestamps are rejected even though Kernel
  can assume UTC. `timestamp_ntz` uses timezone-free text. There is no reader
  timezone override.
- Partition widening supports numeric promotions, decimal precision growth at
  unchanged scale, and integer-to-decimal conversion at scale zero. Decimal
  scale increases, integer-to-decimal with nonzero scale, and
  date-to-`timestamp_ntz` partition changes are not supported.

### Snapshot, filesystem, and file-reference constraints

- Exploration resolves latest afresh when `reader.exttable.snapshot_id=-1`, or
  pins a nonnegative version, including `0`. File readers always use the recorded
  exact version; they do not refresh to latest or support timestamp selectors.
- Source owners must retain the logs/checkpoints and data/DV files needed for
  that version. Missing historical sources cause errors; the reader does not
  retain source files or substitute a newer snapshot.
- Source URLs must continue identifying the same immutable planned files.
  Table-path aliases within a configured filesystem root must remain stable
  while their snapshot descriptors or readers are in use. Retargeting those
  aliases is unsupported: keeping the old physical files elsewhere does not
  prevent cached snapshot/footer state from being combined with different
  payloads opened through a changed alias. Source owners provide this stability.
- All log, checkpoint/sidecar, data, and DV references must use the configured
  filesystem identity. Cross-bucket/cross-filesystem references and local paths
  outside the configured filesystem root are unsupported. Source URLs cannot
  embed credentials, queries, or fragments; local file URLs cannot include a host.
- Local table/file URLs preserve the configured root and table-path spelling
  after absolute lexical normalization, matching the other table formats. Only
  the underlying filesystem/cache identity uses the resolved physical root.
  Local source entries must be relative or use absolute URLs under the configured
  logical root; other spellings outside that prefix are unsupported, even if a
  symlink makes them physically equivalent.
- Reader creation requires a valid snapshot descriptor, recorded file size, and
  live row count, normally supplied by exploration. A path alone is insufficient.
  The descriptor covers the whole file's live rows starting at zero; use the
  reader's range API for partial reads instead of slicing the descriptor.
- Row indices refer to live rows after DV filtering. `take` requires sorted,
  unique indices within those live-row bounds.
- Callers must keep collection-manifest cache keys unique across distinct
  destinations within one process. The shared manifest cache uses filesystem
  root/bucket and manifest path, without endpoint or credential identity. Use
  distinct destination collection paths when endpoints share a bucket name.
  Milvus supplies uniqueness through collection/partition/segment directories;
  this existing contract is retained following the [PR #800 discussion](https://github.com/milvus-io/milvus-storage/pull/800#issuecomment-6936593703).
  Delta source snapshot/file caches isolate filesystem identities separately.

### I/O, caching, and memory estimates

- Exploration uses snapshot row counts when available, but may read Parquet
  footers for files without usable counts. On a per-file cache miss, reader
  creation loads/validates the footer and decodes DV positions even for
  partition-only/null-only projections; those projections can still avoid
  payload-column reads. When no physical columns are needed, chunk/range/take
  output is materialized only for the live/requested row count.
- Encrypted data reads use the existing C++ key callback, including distinct
  column keys. Exploration has no key callback for its count fallback, so files
  with encrypted footers require usable row counts in the snapshot metadata.
- A stateful Parquet payload reader created with a non-null key callback reads
  its footer to initialize its own Arrow 17 decryptors. A cold read can therefore
  load the footer for validation and again for payload-reader setup; clones also
  repeat payload-reader footer setup. This applies even to plaintext files when
  a callback is supplied. Cached decoded metadata does not eliminate these reads.
- Snapshot caching is in memory and tied to reader-metadata ownership. Releasing
  the last owner can require a later reader to replay the pinned snapshot's logs.
  Per-file descriptors do not contain enough table metadata for a cold open
  without access to those logs/checkpoints.
- Snapshot/schema/active-file/projection state can be shared across top-level
  readers. The general reader metadata cache holds one Delta entry per table,
  snapshot and data-file identity (path, size and live-row count). Each entry
  directly retains that file's footer, decoded DV, logical row groups and a shared
  snapshot reference, including for plaintext files. Projections and clones reuse
  it; independent readers load and retain their own entries. Outstanding
  chunk/range readers or clones can extend metadata lifetimes beyond the parent
  reader. A snapshot reference alone does not retain other file entries.
- This ownership can release file metadata earlier, but overlapping readers may
  repeat footer/DV reads and retain duplicate decoded positions. Disabling the
  metadata cache also forgoes reuse across independent file-reader opens.
  Underlying filesystem caches may reduce actual remote traffic. The general
  metadata cache has no byte-bounded eviction. Per-file `cache_size` estimates
  cover file metadata; they exclude shared snapshot state to avoid counting it
  once per file and do not represent total retained memory.
- Snapshot loading still resolves/indexes all active files. New projections cache
  shared schema/scan state; each file plan builds its transform from one indexed
  cached metadata row. Separate reader opens can rebuild that transform, adding
  per-file setup overhead when reading an entire table. Live plans and clones
  reuse their transforms. Loaded footers and decoded DV positions also consume
  memory; DV positions are retained as 64-bit row indices.
- Row-group and per-column memory estimates are unavailable when schema
  adaptation prevents reliable reuse of physical Parquet estimates, including
  mapping, widening, partition constants (even when also stored in Parquet), and
  missing columns. This does not prevent data reads.
- Row-group read-byte estimates describe projected compressed physical payload
  bytes for scheduling, independently of output-memory estimates. No-column or
  fully deleted groups report zero payload bytes; partially deleted groups keep
  their full projected physical size. Zero I/O bytes do not imply zero CPU or
  output allocation and can leave constant-only groups in one byte-based task.

### Validation status

Native/Rust compilation, generated local reader/exploration tests, and existing
Parquet/metadata-cache regressions pass. Reader tests cover Arrow 17/59 batch
adaptation, encrypted/plaintext footers, distinct column keys, and public
synchronous/async-fallback reads with caching enabled and disabled. Remote
filesystem identity and encoded paths are covered through a local-backed proxy
and actual MinIO/RustFS services. Live cloud providers have not been validated.

Generated DV tests cover inline, UUID-relative (with and without a prefix), and
absolute references; multiple vectors at distinct offsets in one container;
empty/fully deleted files; bitmap cardinality and row bounds; malformed payloads,
CRC/size/offset errors; and failed-read recovery with a retained snapshot cache.
Read coverage includes chunks, ranges, take, clones and cached readers.

Shared I/O/cache tests verify footer and DV reuse across projections, chunk-reader
lifetime beyond its parent, and correct projected live values under the existing
eight-thread, 30-second stress harness. Exact I/O assertions use small plaintext
files with recorded counts and no key callback; log replay is excluded from the
measurement.

Local Python integration tests cover exploration, file-property and manifest
round trips, scan/chunk/take reads with caching enabled and disabled, snapshot
pinning after source replacement, and reopening old/latest imports in fresh
processes. All 15 remote Delta cases pass against the repository-pinned MinIO and
RustFS services, covering persisted imports, projected live values, snapshot
pinning, inline/UUID-relative/absolute DVs, shared-container offsets, encoded keys,
invalid credentials, missing-object recovery and source isolation across endpoints.
Relevant external-format, credential, manifest, chunk-reader and lifecycle
regressions also pass on both services.
The two-endpoint Delta fixture uses identical source-table/data/DV keys and
distinct destination collection paths, preserving the shared manifest namespace
contract while validating Delta source-cache isolation.

The remote fixtures generate small sparse DVs directly in Python, including
inline encodings and multiple offsets in one external container. A unit test
checks their bytes against reference output from the pinned Kernel writer.
