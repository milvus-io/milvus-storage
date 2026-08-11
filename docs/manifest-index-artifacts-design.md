# Manifest index-info publication design

**Status:** Proposed
**Owners:** Milvus Storage and Milvus engine
**Scope:** StorageV3 / LOON manifests and Milvus DataCoord, IndexNode, QueryCoord, and QueryNode integration

## Summary

`milvus-storage` already defines `Index` in `cpp/include/milvus-storage/manifest.h`:

```cpp
struct Index {
  std::string column_name;
  std::string index_type;
  std::string path;
  std::map<std::string, std::string> properties;
};
```

This design reuses that definition without adding an index-artifact, index-file, or writer-handle type.

No index-related API is added to `Manifest`. In particular, `Manifest` does
not open or read index files. Consumers continue to obtain the metadata they
need through the existing `getIndex(column_name, index_type)` lookup (or the
existing index collection) and give `Index::path` and `Index::properties` to
the engine's established file-manager/index-loader path.

The index-building flow writes its index file before it interacts with the transaction. The transaction's responsibility is only to atomically record or replace that completed index's `Index` information in a new manifest revision. A persistent manifest minor version lets DataCoord determine whether a manifest has index information and persist that fact alongside the manifest reference in etcd.

```text
IndexNode writes an index file
  -> Transaction::AddIndexInfo(index metadata)
  -> transaction commit writes a new manifest revision
  -> DataCoord atomically persists manifest revision + minor version in etcd
  -> QueryNode sees the minor version, reads Index from the manifest, and loads it through its cache layer
```

## Goals

1. Record completed index files in the same immutable manifest revision model used by data files, delta logs, and stats.
2. Let DataCoord and QueryCoord identify index-bearing StorageV3 manifests without eagerly reading every manifest.
3. Keep index-byte ownership in the existing Milvus QueryNode cache layer; do not add a second index-byte cache in `milvus-storage`.
4. Reuse the existing `Manifest::getIndex` lookup, existing `Index` serialization, and existing `Transaction::AddIndex` callers.

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

`AddIndex` remains source-compatible. It may be retained as an alias of `AddIndexInfo`, with `AddIndexInfo` used by the Milvus engine path to distinguish index-metadata publication from index construction.

`Index::properties` remains the place for engine metadata already needed by the load path, such as `index_id`, `build_id`, `index_version`, metric type, dimensions, and algorithm-specific settings. This proposal adds no new manifest index structure.

## Manifest minor version

### Problem

`Manifest::version()` and `MANIFEST_VERSION` describe the manifest serialization format. In the current Avro object-container-file implementation, a manifest read sets this value to the compiled format version. It is not a persistent per-manifest marker that DataCoord can use to decide whether index information exists.

### Persistent field

Add a separate persistent field to `Manifest` and its Avro record:

```cpp
enum class ManifestMinorVersion : int32_t {
  NONE = 0,
  INDEX_INFO = 1,
};

[[nodiscard]] int32_t minorVersion() const;
[[nodiscard]] bool hasIndexInfo() const {
  return minorVersion() >= static_cast<int32_t>(ManifestMinorVersion::INDEX_INFO);
}
```

The Avro schema adds:

```json
{
  "name": "minor_version",
  "type": "int",
  "default": 0
}
```

`MANIFEST_VERSION` keeps its format-compatibility meaning. Version 6 adds the persistent `minor_version` field, which supplements existing `Index` metadata with a marker that index information is present.

### Invariant

The minor version is derived from the final manifest state:

| Final manifest state | `minor_version` |
| --- | --- |
| `indexes()` is empty | `NONE` (`0`) |
| `indexes()` is non-empty | `INDEX_INFO` (`1`) |

Thus a stats-only transaction preserves `1` when it inherits index information, and dropping the last index writes `0`. A failed transaction does not affect the minor version.

`hasIndexInfo()` is only a derived helper. It is not separately serialized, returned in a commit response, or persisted in etcd, because that would duplicate `minor_version`.

### Compatibility

Old manifests do not contain `minor_version`; Avro schema resolution supplies its default value `0`. Existing index records retain their current schema. No conversion to a new index descriptor is required.

## Transaction and commit API

### Registering completed index information

The engine sequence is:

1. Build and write the index file using the existing index-builder/file-manager path.
2. Fill an existing `Index` with column name, index type, completed path, and properties.
3. Open a transaction at the manifest revision used by the index build.
4. Call `AddIndexInfo(index)`.
5. Commit the manifest.

```cpp
Index index{
    .column_name = "100",
    .index_type = "HNSW",
    .path = "<segment-base>/_index/<written-file>",
    .properties = {
        {"index_id", "101"},
        {"build_id", "10001"},
        {"source_manifest_version", "42"},
    },
};

ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs, base_path, 42, IndexResolver));
txn->AddIndexInfo(index);
ARROW_ASSIGN_OR_RAISE(auto committed, txn->Commit());
```

`source_manifest_version` is an engine convention stored in `properties`, not a new `Index` field. It allows DataCoord and the resolver to reject publication of an index built from stale source data.

### Conflict rules

`milvus-storage` retains its existing generic resolvers. The Milvus engine integration must not use `OverwriteResolver` to publish an index built from an older manifest: a data append can invalidate it. That integration needs an index-aware resolver which:

1. Read `source_manifest_version` from `Index::properties`.
2. Require it to equal the transaction's read version.
3. Commit if the latest manifest revision is the same as the source revision.
4. Otherwise fail transaction resolution and rebuild the index against the newer data.

This initial engine rule is deliberately conservative: a concurrent stats or delta-only change may require a rebuild. A later optimization may compare the affected column-group files and safely rebase only when their source data is unchanged. It is not part of the storage-only API change in this document.

`AddIndexInfo` still replaces the matching `(column_name, index_type)` metadata entry, exactly as `AddIndex` does today. The pre-existing file lifecycle is outside this transaction; when a new path replaces an old path, DataCoord garbage collection determines when the no-longer-referenced old file is safe to delete.

## DataCoord and etcd publication

### Atomic manifest reference

For StorageV3 segments, the following fields form one logical manifest reference:

```text
manifest_path
manifest_version
manifest_minor_version
```

DataCoord adds `manifest_minor_version` as an additive field on `datapb.SegmentInfo` and persists all three values in the same SegmentInfo etcd write. They must not be written in independent keys or independent updates.

The existing manifest-update payload (`BatchUpdateManifestItem`) and `UpdateManifestVersion` operation must carry the minor version together with the manifest revision and retain base-version validation. A stale index-build publication must not move the manifest pointer backwards or attach a minor version to a different manifest revision.

Old SegmentInfo records decode the new protobuf field as `0`.

### What the minor version means

`manifest_minor_version == 0` means DataCoord and QueryCoord do not need to resolve StorageV3 index information from that manifest.

`manifest_minor_version >= 1` means the referenced immutable manifest contains `Index` metadata. It is a routing signal, not a replacement for the metadata itself. The load or garbage-collection path must still read `Index::path` and `Index::properties` from that manifest.

## QueryCoord and QueryNode load path

1. QueryCoord propagates the manifest reference and minor version from DataCoord.
2. For minor version `0`, it skips StorageV3 manifest-index resolution.
3. For minor version `1`, it opens the exact manifest revision and obtains the existing `Index` metadata with `getIndex`.
4. QueryNode passes `Index::path` and `Index::properties` to its existing file-manager/index-loader path.
5. QueryNode continues to use its existing `SealedIndexTranslator` and `CacheSlot<IndexBase>` for loading, warmup, pinning, eviction, and memory/disk accounting.

Minor version avoids needless manifest index parsing; it does not allow a missing or corrupt index file to be ignored. Such a file remains an index-load error.

### Cache and mmap identity

No storage-layer byte cache is introduced. Cache lifecycle remains owned by QueryNode.

When the engine replaces an index path, its QueryNode cache key and mmap local path must distinguish the old and new index. The key should use existing `Index::properties` values where available:

```text
segment_id + field_id + index_id + build_id + index_version
```

If a property is absent for legacy metadata, QueryNode follows its existing cache-key behaviour. For new engine-written index information, `index_id` and `build_id` are required. Replacing an index must retire or cancel the old cache slot before the old local mmap files are reclaimed; existing readers retain their old slot until release.

## Index-file lifecycle and garbage collection

Writing index bytes is intentionally outside `Transaction::AddIndexInfo`. The index-builder path is responsible for creating the file before metadata registration, and DataCoord remains responsible for deleting unreferenced files.

For StorageV3 index paths, DataCoord garbage collection must inspect `Index` entries in every live and protected manifest with `minor_version >= 1`. It may delete a file only after it is absent from:

1. the live SegmentInfo manifest reference;
2. protected snapshots and compaction fallback manifests; and
3. any existing engine metadata that retains the same file reference.

If manifest metadata cannot be read, garbage collection must fail closed for that reference set.

## FFI and protobuf changes

The required additive interfaces are:

- C/C++ FFI exposure for `AddIndexInfo`;
- manifest metadata export support for existing `Index` entries and `minor_version`;
- `datapb.SegmentInfo`, manifest-update messages, and load metadata carrying `manifest_minor_version`;
- Go bindings that forward the completed `Index` information, rather than index bytes, from the build/publish path.

There is no transaction streaming-write FFI and no index-artifact ABI.

## Failure handling

| Failure | Required behaviour |
| --- | --- |
| Index file write fails before `AddIndexInfo` | Do not start index metadata publication. |
| `AddIndexInfo` input lacks path | The storage FFI rejects it; native callers must provide a valid completed-file path. |
| Engine index publication lacks source revision | DataCoord must reject it before publication; source-version enforcement is an engine integration responsibility. |
| Manifest conflict after index file exists | Do not publish its metadata; rebuild or later GC the unreferenced file. |
| Manifest committed but DataCoord publication fails | File and manifest remain invisible to QueryNode until an idempotent publication retry succeeds. |
| DataCoord receives stale base revision | Reject the update and keep the current manifest reference/minor pair unchanged. |
| QueryNode sees minor `1` but cannot read file | Fail index load; never silently load a different path or a cached older index. |

## Compatibility and rollout

- Older manifests read with `minor_version = 0`.
- Existing `Index` records and `Transaction::AddIndex` stay valid.
- Older etcd/protobuf records default the new minor field to `0`.
- Deploy readers of `minor_version` and `Index` metadata before enabling IndexNode publication through `AddIndexInfo`.
- During mixed-version rollout, DataCoord must not publish manifest-index references to QueryNodes that cannot resolve them.

## Test matrix

| Area | Required tests |
| --- | --- |
| Manifest | Avro round-trip for minor `0`/`1`; old manifest defaults to `0`; existing `getIndex` returns the recorded path and properties. No index-file read API is added. |
| Transaction | `AddIndexInfo` adds/replaces index metadata, preserves/reset minor version as indexes appear/disappear, and leaves `AddIndex` compatible. |
| Conflict | Milvus integration test: an index built from a stale source manifest is rejected after a data append; its old file stays unreferenced and is eligible for GC. |
| DataCoord | Manifest revision and minor version update atomically; stale update rejection; old etcd record defaults to `0`. |
| QueryNode | Minor `0` skips index resolution; minor `1` loads through the existing file-manager/index-loader path; replacement produces a different cache/mmap identity. |
| GC | Live, snapshot, and fallback manifests protect their recorded `Index::path`; unreferenced index files are reclaimed only after protection ends. |
| End-to-end | Write index file, register via `AddIndexInfo`, publish through DataCoord, load on QueryNode, replace it, and verify old-cache retirement. |
