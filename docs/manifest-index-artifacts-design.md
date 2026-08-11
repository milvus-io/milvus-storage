# Manifest index-info publication design

**Status:** Proposed
**Owners:** Milvus Storage and Milvus engine
**Scope:** StorageV3 / LOON manifests and Milvus DataCoord, IndexNode, QueryCoord, and QueryNode integration

## Summary

`milvus-storage` already defines `Index` in `cpp/include/milvus-storage/manifest.h`:

```cpp
struct Index {
    std::string column_name;
    std::string index_name;
    std::string index_type;
    std::string path;
    int64_t field_id;
    int64_t index_id;
    int64_t build_id;
    int64_t index_version;
    int64_t num_rows;
    int64_t serialized_size;
    int64_t mem_size;
    int32_t current_index_version;
    int32_t current_scalar_index_version;
    int32_t index_store_path_version;
    std::vector<std::string> index_file_keys;
    std::map<std::string, std::string> properties;
};
```

This design reuses that definition without adding an index-artifact, index-file, or writer-handle type.

No index-related API is added to `Manifest`. In particular, `Manifest` does
not open or read index files. Consumers continue to obtain the metadata they
need through the existing `getIndex(column_name, index_type)` lookup (or the
existing index collection) and give the typed `Index` fields and
`Index::properties` to the engine's established file-manager/index-loader
path.

The index-building flow writes its index file before it interacts with the transaction. The transaction's responsibility is only to atomically record or replace that completed index's `Index` information in a new manifest revision. Index presence is determined from the manifest's existing `indexes` payload; no separate marker is persisted.

```text
IndexNode writes an index file
  -> Transaction::AddIndexInfo(index metadata)
  -> transaction commit writes a new manifest revision
  -> DataCoord publishes only the current manifest path in SegmentInfo etcd metadata
  -> QueryNode reads Index from that manifest and loads it through its cache layer
```

## Goals

1. Record completed index files in the same immutable manifest revision model used by data files, delta logs, and stats.
2. Make the manifest the sole persistent source for completed-index load metadata, minimizing etcd index state.
3. Keep index-byte ownership in the existing Milvus QueryNode cache layer; do not add a second index-byte cache in `milvus-storage`.
4. Reuse the existing `Manifest::getIndex` lookup and `Index` serialization; use `Transaction::AddIndexInfo` for publication.

## Non-goals

- Writing index bytes through `Transaction`.
- Adding `IndexArtifact`, `IndexFile`, `IndexWriteHandle`, or `BeginIndexWrite` APIs.
- Returning a write-side file handle from `Transaction`.
- Adding a new index-file read API to `Manifest`.
- Changing `Manifest::getIndex` or making `Manifest` own index-file I/O or caching.
- Making the pre-existing file write and the object-store manifest write a distributed atomic transaction.
- Replacing QueryNode's `CacheSlot` ownership with a manifest-level byte-buffer cache.

## Existing model

`Manifest::getIndex` already provides the necessary metadata lookup. Its
return value exposes the existing `Index::path` and `Index::properties`; the
manifest layer deliberately stops there. This proposal adds no index-specific
`Manifest` method such as `ReadIndexFile`.

The existing transaction API already stores an `Index` in its update set and replaces an index with the same `(column_name, index_type)` during manifest resolution. The proposed API makes that intent explicit for engine callers:

```cpp
Transaction& Transaction::AddIndexInfo(const Index& index);
```

`AddIndexInfo` is metadata-only. It does not create, write, delete, rename, checksum, or otherwise manage `index.path`. The caller must write the completed index file before calling it.

The typed fields in `Index` carry the completed artifact identity, name, file
list, sizes, row count, and engine/path versions. `Index::properties` is reserved for
open-ended algorithm parameters, such as metric type, dimensions, M, and
efConstruction. This avoids string parsing for required load metadata while
retaining an evolution path for index-type-specific options.

## Manifest format version

### Manifest version and index metadata

`MANIFEST_VERSION` remains a serialization-format constant. It is bumped to 6 for this transaction index-publication addition, but it is not a persistent per-revision routing signal: an Avro read reports the version compiled into the reader.

The existing `Index` entries are the only persistent index-presence signal. Readers must inspect `Manifest::indexes()` directly. No `ManifestMinorVersion`, `minor_version` Avro field, or C-FFI minor-version field is added.

Existing manifests remain compatible because the pre-existing `indexes` Avro field has an empty-array default. No index-descriptor conversion is required.

## Transaction and commit API

### Registering completed index information

The engine sequence is:

1. Build and write the index file using the existing index-builder/file-manager path.
2. Fill an existing `Index` with the completed artifact's typed metadata and algorithm properties.
3. Open a transaction at the manifest revision used by the index build.
4. Call `AddIndexInfo(index)`.
5. Commit the manifest.

```cpp
Index index{
    .column_name = "100",
    .index_name = "vector_hnsw",
    .index_type = "HNSW",
    .path = "<segment-base>/_index/<artifact-prefix>",
    .field_id = 100,
    .index_id = 101,
    .build_id = 10001,
    .index_version = 4,
    .num_rows = 100000,
    .serialized_size = 1048576,
    .mem_size = 2097152,
    .current_index_version = 15,
    .current_scalar_index_version = 7,
    .index_store_path_version = 1,
    .index_file_keys = {"index.bin", "raw_data.bin"},
    .properties = {
        {"metric_type", "COSINE"},
        {"M", "16"},
    },
};

ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs, base_path, 42, IndexResolver));
txn->AddIndexInfo(index);
ARROW_ASSIGN_OR_RAISE(auto committed, txn->Commit());
```

The source manifest revision is transaction state: DataCoord opens the transaction at the revision used by the index build. It is not duplicated in `Index` or `properties`.

### Conflict rules

`milvus-storage` retains its existing generic resolvers. The Milvus engine integration must not use `OverwriteResolver` to publish an index built from an older manifest: a data append can invalidate it. That integration needs an index-aware resolver which:

1. Open the transaction at the source manifest revision.
2. Commit only if the latest manifest revision is still that source revision.
3. Otherwise fail transaction resolution and rebuild the index against the newer data.

This initial engine rule is deliberately conservative: a concurrent stats or delta-only change may require a rebuild. A later optimization may compare the affected column-group files and safely rebase only when their source data is unchanged. It is not part of the storage-only API change in this document.

`AddIndexInfo` replaces the matching `(column_name, index_type)` metadata entry. The pre-existing file lifecycle is outside this transaction; when a new path replaces an old path, DataCoord garbage collection determines when the no-longer-referenced old file is safe to delete.

## DataCoord and etcd publication

### Minimal etcd state

`datapb.SegmentInfo.manifest_path` is the only durable StorageV3 reference needed by the load path. The manifest `Index` also carries `index_name`, so it can form the complete QueryNode `FieldIndexInfo`. Every completed-index loading field currently held by etcd `indexpb.SegmentIndex` is persisted in the matching manifest `Index` instead: `field_id`, `index_id`, `build_id`, `index_version`, `num_rows`, `serialized_size`, `mem_size`, current vector/scalar engine versions, path-layout version, and `index_file_keys`.

`path` is the artifact-directory prefix and `index_file_keys` identifies each file beneath it. QueryCoord constructs the complete file paths from these two fields. `SegmentIndex` may retain transient task state while a build is queued or in progress, but a finished artifact's paths, sizes, identity, and load versions must not be read from etcd.

## QueryCoord and QueryNode load path

1. QueryCoord obtains the segment's current `SegmentInfo.manifest_path` from etcd and opens that immutable manifest revision.
2. It obtains the matching `Index` metadata with `getIndex` (or scans `indexes()` by `field_id` and `index_id`).
3. It builds `FieldIndexInfo` entirely from the manifest Index: identity, file paths, sizes, row count, and engine versions are copied directly; index-type parameters come from `properties`.
4. QueryNode continues to use its existing `SealedIndexTranslator` and `CacheSlot<IndexBase>` for loading, warmup, pinning, eviction, and memory/disk accounting.

A missing or corrupt matching index file remains an index-load error. A missing matching Index entry in the current manifest is a stale-index condition, not a legacy fallback.

### Cache and mmap identity

No storage-layer byte cache is introduced. Cache lifecycle remains owned by QueryNode.

When the engine replaces an index path, its QueryNode cache key and mmap local path must distinguish the old and new index. The key uses typed Index fields:

```text
segment_id + field_id + index_id + build_id + index_version
```

For new engine-written index information, `field_id`, `index_id`, and `build_id` are required. Replacing an index must retire or cancel the old cache slot before the old local mmap files are reclaimed; existing readers retain their old slot until release.

## Index-file lifecycle and garbage collection

Writing index bytes is intentionally outside `Transaction::AddIndexInfo`. The index-builder path is responsible for creating the file before metadata registration, and DataCoord remains responsible for deleting unreferenced files.

For StorageV3 index paths, DataCoord garbage collection must inspect `Index` entries in every live and protected manifest. It may delete a file only after it is absent from:

1. the live SegmentInfo manifest reference;
2. protected snapshots and compaction fallback manifests; and
3. any existing engine metadata that retains the same file reference.

If manifest metadata cannot be read, garbage collection must fail closed for that reference set.

## FFI and protobuf changes

The required additive interfaces are:

- C/C++ FFI exposure for `AddIndexInfo`;
- manifest metadata export support for existing `Index` entries;
- Go bindings that forward the completed `Index` information, rather than index bytes, from the build/publish path and use it to form QueryNode load requests.

There is no transaction streaming-write FFI and no index-artifact ABI.

## Failure handling

| Failure | Required behaviour |
| --- | --- |
| Index file write fails before `AddIndexInfo` | Do not start index metadata publication. |
| `AddIndexInfo` input lacks path | The storage FFI rejects it; native callers must provide a valid completed-file path. |
| Engine index publication lacks source revision | DataCoord must reject it before publication; source-version enforcement is an engine integration responsibility. |
| Manifest conflict after index file exists | Do not publish its metadata; rebuild or later GC the unreferenced file. |
| Manifest committed but DataCoord publication fails | File and manifest remain invisible to QueryNode until an idempotent publication retry succeeds. |
| DataCoord receives stale base revision | Reject the update and keep the current manifest reference unchanged. |
| QueryNode cannot read the matching manifest index file | Fail index load; never silently load a different path or a cached older index. |

## Compatibility and rollout

- Existing `Index` records stay valid.
- Older manifests with an empty index list continue to represent no manifest-published index metadata.
- During rollout, completed legacy indexes may use the existing `SegmentIndex` path construction until they are rebuilt and republished into a manifest.
- Deploy manifest-index readers before enabling IndexNode publication through `AddIndexInfo`.
- During mixed-version rollout, DataCoord must not publish manifest-index references to QueryNodes that cannot resolve them.

## Test matrix

| Area | Required tests |
| --- | --- |
| Manifest | Version-6 Avro round-trip preserves every typed Index load field, file key, and property; existing manifests without index entries remain readable. No index-file read API is added. |
| Transaction | `AddIndexInfo` adds or replaces index metadata for the matching `(column_name, index_type)`. |
| Conflict | Milvus integration test: an index built from a stale source manifest is rejected after a data append; its old file stays unreferenced and is eligible for GC. |
| DataCoord | Form QueryNode load metadata entirely from the manifest Index; do not read completed-artifact paths, sizes, or identity from SegmentIndex. |
| QueryNode | A manifest Index loads through the existing file-manager/index-loader path; replacement produces a different cache/mmap identity. |
| GC | Live, snapshot, and fallback manifests protect their recorded `Index::path`; unreferenced index files are reclaimed only after protection ends. |
| End-to-end | Write index file, register via `AddIndexInfo`, publish through DataCoord, load on QueryNode, replace it, and verify old-cache retirement. |
