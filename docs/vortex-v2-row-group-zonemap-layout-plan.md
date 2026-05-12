# Vortex V2 Row Group ZoneMap Layout Plan

**Status**: implementation-aligned design note
**Date**: 2026-05-12
**Baseline commit**: `cee4a7f feat: add Vortex V2 format reader and writer with row-group layout support`
**Primary code scope**: `cpp/src/format/bridge/rust/src`
**Test scope**: `cpp/test/format/vortex/vortex_v2_test.cpp`
**Out of scope**: do not patch `cpp/src/format/bridge/rust/_vortex_patched/vortex-*`

---

## 1. Background

The Vortex V2 writer already writes byte-size row groups. Its data layout is built from row-group chunks:

```text
Root: ChunkedLayout
  RG0: StructLayout
    col_a: ChunkedLayout -> BtrBlocks -> Flat(inline_array_node=true)
    col_b: ChunkedLayout -> BtrBlocks -> Flat(inline_array_node=true)
    ...
  RG1: StructLayout
  ...
```

This layout keeps row-group data physically grouped, which is good for sequential scans. The missing piece was row-group-level pruning. Without row-group ZoneMaps, selective scans still need to read row groups that cannot match the filter.

The default Vortex strategy has a different shape. It builds a top-level `StructStrategy`, then each column uses `RepartitionStrategy -> ZonedStrategy -> data/zones`. Those ZoneMaps are part of each column layout and are based on fixed row counts, not on the V2 byte-size row groups. Reusing the default layout directly would not preserve the desired V2 physical data order.

The core design is to reuse Vortex per-column `ZoneMap` semantics and change only the root layout plus the physical placement of zone data. The stats table dtype, fields, truncation markers, and pruning predicate semantics stay Vortex-compatible. The Milvus-specific code is responsible for V2 row-group granularity, moving all zones after all data segments, and expanding a row-group mask into a row mask.

## 2. Target Layout

Logical layout:

```text
Root: milvus.v2_zoned_row_group
  data: ChunkedLayout<RG StructLayout>
    RG0 data
    RG1 data
    RG2 data
    ...
  zones: StructLayout / container
    col_a: Vortex-compatible ZoneMap table, one row per RG
    col_b: Vortex-compatible ZoneMap table, one row per RG
    ...
    __rg_boundaries:
      __rg_start: u64
      __rg_len:   u64
```

Physical segment order:

```text
RG0 data segments
RG1 data segments
RG2 data segments
...
all per-column ZoneMap segments
RG boundary segments
footer
```

The important properties are:

- All row-group data segments are written before any zone or boundary segment.
- ZoneMap data is inside the Vortex file and participates in the Vortex layout tree.
- Per-column stats tables use `ZoneMap::dtype_for_stats_table(column_dtype, present_stats)`.
- Per-column stats tables can be validated with `ZoneMap::try_new`.
- Pruning uses Vortex `checked_pruning_expr` and evaluates the resulting predicate on a Vortex-compatible stats view.
- The custom reader only selects the relevant column ZoneMap, applies conservative expression mapping, combines masks, reads `__rg_boundaries`, and expands row-group results to row results.
- Zone granularity is one row per V2 row group, not a fixed row count such as 8192 rows.

## 3. Current Implementation

The implementation lives in:

```text
cpp/src/format/bridge/rust/src/vortex_layout_strategy_v2.rs
```

The custom layout pieces are named:

```text
RowGroupZoneMapLayoutEncoding
RowGroupZoneMapLayout
RowGroupZoneMapVTable
RowGroupZoneMapReader
RowGroupZoneMapStrategy
```

The encoding id remains:

```text
milvus.v2_zoned_row_group
```

This id is the Milvus custom Vortex format boundary. A generic Vortex reader that has not registered this encoding should fail to open the file. The Milvus bridge registers the encoding in `VORTEX_SESSION`.

V2 strategy selection is unified through:

```rust
build_row_group_strategy(row_group_max_size, enable_zone_map, stats)
```

The bridge caller passes `enable_zone_map = enable_stats` for V2. There is no extra environment variable, feature flag, or `v2_zonemap` switch. The enable condition is exactly:

```text
enable_stats && format_version == VORTEX_FORMAT_V2
```

When `enable_stats=false`, the writer keeps the plain V2 row-group layout and does not emit the custom root layout. If stats are enabled but no column can produce usable ZoneMap stats, the writer also falls back to the plain data layout.

## 4. Layout Contract

### 4.1 Children

```text
child[0] = data
  type = Transparent("data")
  dtype = original struct dtype
  row_count = file row count

child[1] = zones
  type = Auxiliary("zones")
  dtype = Struct {
    col_a: ZoneMap::dtype_for_stats_table(dtype(col_a), present_stats(col_a)),
    col_b: ZoneMap::dtype_for_stats_table(dtype(col_b), present_stats(col_b)),
    ...
    __rg_boundaries: Struct {
      __rg_start: u64,
      __rg_len:   u64,
    }
  }
  row_count = row group count
```

The zones child is not an external sidecar and is not a flattened wide stats table. Each real column with usable stats owns a complete Vortex-compatible `ZoneMap` stats table. The synthetic `__rg_boundaries` table is separate from all per-column stats tables.

### 4.2 Metadata

Current metadata contains:

- `version: u16`
- `stats_version: u16`
- `rg_count: u64`
- `columns: Vec<ColumnZoneMetadata>`
- For each column: `field_name` and `present_stats`

The current bridge implementation maps top-level struct fields by name. If nested field paths are added later, that should be a metadata versioned change.

The metadata does not store all row-group start and length values. Those live in the `__rg_boundaries` table to avoid footer growth proportional to the row-group count.

### 4.3 Reserved Fields

The input schema must not contain a top-level field named:

```text
__rg_boundaries
```

That name is reserved for the synthetic boundary table under the zones child. A conflict is rejected by the writer.

## 5. ZoneMap Table Design

For each column with usable stats:

```text
zones.<column>:
  min
  min_is_truncated
  max
  max_is_truncated
  null_count
  nan_count
  ...
```

The exact field set is determined by Vortex:

```rust
ZoneMap::dtype_for_stats_table(column_dtype, present_stats)
```

Stats are accumulated with Vortex `StatsAccumulator`. This preserves Vortex field names and string/binary truncation semantics, including `min_is_truncated` and `max_is_truncated`.

The boundary table is independent:

```text
zones.__rg_boundaries:
  __rg_start: u64
  __rg_len:   u64
```

The boundary table is used only to expand a row-group-level pruning result into a row-level keep mask. It is never mixed into the per-column `ZoneMap` table shape.

## 6. Writer Flow

The V2 writer uses two internal strategies:

```text
RowGroupSplitStrategy
  split stream into byte-size row groups
  write only the normal V2 data layout

RowGroupZoneMapStrategy
  split stream into byte-size row groups
  write the normal V2 data layout as child[0]
  write per-column ZoneMaps and boundaries as child[1]
  return RowGroupZoneMapLayout
```

`RowGroupZoneMapStrategy` flow:

```text
input stream
  -> split by uncompressed byte size
  -> for each row group in order:
       compute requested stats for each top-level field
       append one stats row to that column's StatsAccumulator
       append one row to __rg_boundaries
       yield the row group to the data child

data_layout = write data child first with data_eof = eof.split_off()
zones_layout = write zones child after the data child
return RowGroupZoneMapLayout(data_layout, zones_layout, metadata)
```

The physical order is enforced with `SequencePointer::split_off()`:

```text
data child consumes the data_eof sequence
zones child consumes the later eof sequence
```

This is what guarantees:

```text
all data segments < all zone/boundary segments
```

## 7. Reader Flow

`RowGroupZoneMapReader` delegates non-pruning behavior to the data child:

```text
projection_evaluation(row_range, expr, mask)
  -> data_child.projection_evaluation(row_range, expr, mask)

filter_evaluation(row_range, expr, mask)
  -> data_child.filter_evaluation(row_range, expr, mask)

register_splits(field_mask, row_range, splits)
  -> data_child.register_splits(field_mask, row_range, splits)
```

Pruning flow:

```text
1. Ask data_child.pruning_evaluation(...) for the base pruning result.
2. Build root-scope available_stats from metadata, for example col_a.min and col_a.max.
3. Call checked_pruning_expr(expr, available_stats).
4. If the expression is not safely stats-prunable, return the data child result.
5. Require all stats used by the pruning expression to belong to one column.
6. Build a temporary root-scope stats view over the column-local ZoneMap arrays, for example `col_a_max -> zones.col_a.max`.
7. Load zones.<column> and validate it with ZoneMap::try_new.
8. Evaluate the pruning predicate on that temporary stats view to get a row-group prune mask.
9. Load __rg_boundaries and expand the row-group prune mask into a row keep mask.
10. Intersect the input mask, the ZoneMap-derived row keep mask, and the data child result.
```

The stats predicate returns `true` for a row group that can be skipped. `LayoutReader::pruning_evaluation` returns a row keep mask. Expansion therefore inverts the row-group prune bit:

```text
rg_keep = !rg_prune
row_keep = expand_by(__rg_start, __rg_len, rg_keep, row_range)
```

Partial row ranges are handled by intersecting each row-group boundary with the requested `row_range`. The expanded mask length must exactly match `row_range.end - row_range.start`.

## 8. Expression Support

The first implementation is intentionally conservative:

- Single-column predicates can use the matching column ZoneMap.
- Multiple conjuncts already split by Vortex scan can be handled independently and intersected by the scan pipeline.
- A predicate is pruned only when `checked_pruning_expr` and the required stats reduce it to one column scope.
- Cross-column `AND` can still benefit when Vortex scan has already split it into independent conjuncts that are evaluated separately.
- Cross-column comparisons, `OR`, unsupported functions, missing stats, unsupported types, and unsafe rewrites return the data child pruning result.

The implementation does not introduce a custom wide-table expression evaluator and does not redefine Vortex stats scope.

## 9. Why Not a Wide Stats Table

A rejected alternative was a single wide stats table:

```text
col_a_min
col_a_max
col_b_min
col_b_max
...
```

That shape can be made semantically equivalent, but it would redefine Vortex's stats table layout. It would need more glue code for dtype construction, field naming, truncation markers, and validation. It also prevents directly validating each column with `ZoneMap::try_new`.

The per-column table design is closer to Vortex `ZonedLayout`:

```text
one column -> one Vortex-compatible ZoneMap stats table
```

The only differences are row-group granularity and physical placement.

## 10. Other Rejected Alternatives

### External Sidecar ZoneMap

Rejected because Vortex would not see the sidecar through the layout tree, so `LayoutReader::pruning_evaluation` would not use it in the native scan path.

### ZoneMap Inside Each Row Group

Rejected because it would produce interleaved physical layout:

```text
RG0 data
RG0 zones
RG1 data
RG1 zones
...
```

That breaks the desired continuous data region for sequential scans.

### Wrapping V2 Root With Built-In ZonedLayout

Rejected because built-in `ZonedLayout` uses fixed `zone_len` semantics, while V2 row groups are byte-size and have variable row counts. Wrapping the struct root also does not naturally produce one Vortex-compatible ZoneMap table per real column.

### Patching Vortex

Not needed. The required extension points are public: layout encoding, layout reader, session registration, `ZoneMap`, and `StatsAccumulator`.

## 11. Tests

### Implemented C++ gtests

The main behavioral coverage is in `cpp/test/format/vortex/vortex_v2_test.cpp`:

```text
VortexV2Test.TestV2StatsEnabledUsesRowGroupZoneMapLayout
VortexV2Test.TestV2StatsDisabledUsesPlainRowGroupLayout
VortexV2Test.TestV2RowGroupZoneMapFilterScan
```

These tests cover:

- `enable_stats=true` with V2 writes root encoding `milvus.v2_zoned_row_group`.
- `enable_stats=false` with V2 keeps the plain row-group layout.
- The row-group count matches scan splits.
- Data segments are physically before zone and boundary segments.
- C++ filtered scan over V2 RowGroupZoneMap data returns correct rows.

Bridge inspection helpers used by the tests:

```text
VortexFile::RootLayoutEncoding()
VortexFile::RowGroupZoneMapCount()
VortexFile::RowGroupZoneMapDataBeforeZones()
```

### Implemented Rust Unit Coverage

The Rust module also keeps focused unit coverage for:

- metadata serialization roundtrip
- partial `row_range` row-group mask expansion
- scan filter path using row-group ZoneMap pruning

The C++ gtests remain the required integration coverage because the user-facing writer and reader path is C++.

### Additional Coverage To Preserve Or Extend

ZoneMap compatibility:

- Numeric, string, and nullable columns produce per-column zones tables whose dtype equals `ZoneMap::dtype_for_stats_table`.
- `min_is_truncated` and `max_is_truncated` match Vortex builder semantics.
- `ZoneMap::try_new` validates each per-column zones table.

Physical layout:

- Write at least three row groups and verify every data segment offset is less than every zone or boundary segment offset.
- Verify `__rg_boundaries` covers the full file without gaps or overlaps.

Pruning behavior:

- Single-column range filters skip impossible row groups.
- Unsupported expressions, missing stats, unsupported types, and complex cross-column expressions are conservative and do not prune.
- Partial `row_range` expansion has correct length and boundaries.

Compatibility:

- Registered `milvus.v2_zoned_row_group` encoding can read the file.
- Missing encoding fails with a diagnostic error.
- `enable_stats=false` keeps the current V2 layout unchanged.

## 12. Implementation Checklist

Implemented:

- `vortex_layout_strategy_v2.rs` defines the custom layout, metadata, reader, and writer strategy.
- `VORTEX_SESSION` registers `RowGroupZoneMapLayoutEncoding`.
- `vortex_bridgeimpl.rs` uses the unified `build_row_group_strategy`.
- V2 ZoneMap enablement is controlled only by `enable_stats && format_version == VORTEX_FORMAT_V2`.
- V2 stats-disabled writes keep the plain row-group layout.
- Per-column Vortex-compatible ZoneMap stats tables are generated with `StatsAccumulator`.
- `__rg_boundaries` is stored as a separate zones child field.
- Data child is written before zones child with `SequencePointer::split_off()`.
- The reader delegates projection/filter/splits to the data child.
- The reader uses `checked_pruning_expr` plus a temporary root-scope stats view for safe single-column pruning.
- C++ gtests cover enabled layout, disabled fallback, physical ordering, and filtered scan correctness.

Not in scope for the first implementation:

- Custom wide-table stats evaluation.
- Aggressive cross-column pruning.
- Patching Vortex internals.
- A separate V2 ZoneMap environment variable or feature flag.

## 13. Final Recommendation

The implemented design is the recommended design:

```text
milvus.v2_zoned_row_group
  data: current V2 ChunkedLayout<RG StructLayout>
  zones: Struct/container
    col_a: Vortex-compatible ZoneMap table, one row per RG
    col_b: Vortex-compatible ZoneMap table, one row per RG
    ...
    __rg_boundaries: __rg_start/__rg_len table, one row per RG
```

This satisfies the original requirements without patching Vortex:

- The data region is physically continuous.
- ZoneMap data stays inside the Vortex file.
- Stats dtype, fields, truncation markers, and pruning semantics match Vortex per-column `ZoneMap`.
- Pruning reuses `checked_pruning_expr` and evaluates its predicate on Vortex-compatible ZoneMap stats arrays.
- Zone granularity matches V2 byte-size row groups.
- The Milvus bridge can evolve this custom Vortex format independently.
