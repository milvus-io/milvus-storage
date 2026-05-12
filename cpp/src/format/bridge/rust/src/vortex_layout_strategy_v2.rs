use std::ops::{BitAnd, Range};
use std::sync::{Arc, Mutex};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::{StreamExt as _, pin_mut};
use vortex::ArrayContext;
use vortex::arrays::{PrimitiveArray, StructArray};
use vortex::buffer::Buffer;
use vortex::dtype::{
    DType, Field, FieldName, FieldNames, FieldPath, FieldPathSet, Nullability, PType, StructFields,
};
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure, vortex_err};
use vortex::expr::pruning::{checked_pruning_expr, field_path_stat_field_name};
use vortex::expr::{Expression, get_item, root};
use vortex::io::runtime::Handle;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::collect::CollectStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::struct_::writer::StructStrategy;
use vortex::layout::layouts::zoned::zone_map::{StatsAccumulator, ZoneMap};
use vortex::layout::segments::{SegmentId, SegmentSinkRef, SegmentSource};
use vortex::layout::sequence::{
    SendableSequentialStream, SequencePointer, SequentialArrayStreamExt, SequentialStreamAdapter,
    SequentialStreamExt,
};
use vortex::layout::{
    IntoLayout, LayoutChildType, LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReader,
    LayoutReaderRef, LayoutRef, LayoutStrategy, LazyReaderChildren, VTable, vtable,
};
use vortex::mask::{Mask, MaskMut};
use vortex::stats::{Stat, as_stat_bitset_bytes, stats_from_bitset_bytes};
use vortex::validity::Validity;
use vortex::{
    Array, ArrayRef, DeserializeMetadata, IntoArray, MaskFuture, SerializeMetadata, ToCanonical,
};

const LAYOUT_ID: &str = "milvus.v2_zoned_row_group";
const METADATA_VERSION: u16 = 1;
const STATS_VERSION: u16 = 1;
const RG_BOUNDARIES_FIELD: &str = "__rg_boundaries";
const RG_START_FIELD: &str = "__rg_start";
const RG_LEN_FIELD: &str = "__rg_len";

static ROW_GROUP_ZONE_MAP_PRUNE_EVAL_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static ROW_GROUP_ZONE_MAP_PRUNED_ROW_GROUP_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) fn reset_row_group_zone_map_pruning_stats() {
    ROW_GROUP_ZONE_MAP_PRUNE_EVAL_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ROW_GROUP_ZONE_MAP_PRUNED_ROW_GROUP_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn row_group_zone_map_pruning_stats() -> (u64, u64) {
    (
        ROW_GROUP_ZONE_MAP_PRUNE_EVAL_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ROW_GROUP_ZONE_MAP_PRUNED_ROW_GROUP_COUNT.load(std::sync::atomic::Ordering::Relaxed),
    )
}

vtable!(RowGroupZoneMap);

#[derive(Debug)]
pub struct RowGroupZoneMapLayoutEncoding;

#[derive(Clone, Debug)]
pub struct RowGroupZoneMapLayout {
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
    metadata: RowGroupZoneMapMetadata,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ColumnZoneMetadata {
    pub field_name: FieldName,
    pub present_stats: Arc<[Stat]>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RowGroupZoneMapMetadata {
    pub version: u16,
    pub stats_version: u16,
    pub rg_count: u64,
    pub columns: Vec<ColumnZoneMetadata>,
}

impl RowGroupZoneMapMetadata {
    fn new(rg_count: u64, columns: Vec<ColumnZoneMetadata>) -> Self {
        Self {
            version: METADATA_VERSION,
            stats_version: STATS_VERSION,
            rg_count,
            columns,
        }
    }

    fn column(&self, field_name: &FieldName) -> Option<&ColumnZoneMetadata> {
        self.columns
            .iter()
            .find(|column| &column.field_name == field_name)
    }
}

impl DeserializeMetadata for RowGroupZoneMapMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let mut reader = MetadataReader::new(metadata);
        let version = reader.read_u16()?;
        let stats_version = reader.read_u16()?;
        let rg_count = reader.read_u64()?;
        let column_count = reader.read_u32()? as usize;
        let mut columns = Vec::with_capacity(column_count);

        for _ in 0..column_count {
            let field_name = FieldName::from(reader.read_string()?);
            let stat_bytes = reader.read_bytes()?;
            let present_stats = Arc::<[Stat]>::from(stats_from_bitset_bytes(stat_bytes));
            columns.push(ColumnZoneMetadata {
                field_name,
                present_stats,
            });
        }

        vortex_ensure!(
            reader.is_done(),
            "Trailing bytes in {LAYOUT_ID} metadata: {}",
            reader.remaining()
        );

        Ok(Self {
            version,
            stats_version,
            rg_count,
            columns,
        })
    }
}

impl SerializeMetadata for RowGroupZoneMapMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.stats_version.to_le_bytes());
        bytes.extend_from_slice(&self.rg_count.to_le_bytes());
        bytes.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        for column in self.columns {
            write_string(&mut bytes, column.field_name.as_ref());
            let stat_bytes = as_stat_bitset_bytes(&column.present_stats);
            bytes.extend_from_slice(&(stat_bytes.len() as u16).to_le_bytes());
            bytes.extend_from_slice(&stat_bytes);
        }
        bytes
    }
}

struct MetadataReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> MetadataReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_u16(&mut self) -> VortexResult<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> VortexResult<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> VortexResult<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> VortexResult<String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| vortex_err!("Invalid UTF-8 in {LAYOUT_ID} metadata: {e}"))
    }

    fn read_bytes(&mut self) -> VortexResult<&'a [u8]> {
        let len = self.read_u16()? as usize;
        self.read_exact(len)
    }

    fn read_exact(&mut self, len: usize) -> VortexResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| vortex_err!("Invalid {LAYOUT_ID} metadata length"))?;
        if end > self.bytes.len() {
            vortex_bail!("Truncated {LAYOUT_ID} metadata");
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn is_done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    let len = u16::try_from(value.len()).vortex_expect("field name too long");
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

impl VTable for RowGroupZoneMapVTable {
    type Layout = RowGroupZoneMapLayout;
    type Encoding = RowGroupZoneMapLayoutEncoding;
    type Metadata = RowGroupZoneMapMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref(LAYOUT_ID)
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(RowGroupZoneMapLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.children.child_row_count(0)
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        layout.metadata.clone()
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => layout.children.child(0, &layout.dtype),
            1 => layout
                .children
                .child(1, &zones_dtype(&layout.dtype, &layout.metadata)?),
            _ => vortex_bail!("Invalid {LAYOUT_ID} child index: {idx}"),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("zones".into()),
            _ => vortex::error::vortex_panic!("Invalid {LAYOUT_ID} child index: {idx}"),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(RowGroupZoneMapReader::try_new(
            layout.clone(),
            name,
            segment_source,
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &RowGroupZoneMapMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure!(
            metadata.version == METADATA_VERSION,
            "Unsupported {LAYOUT_ID} metadata version: {}",
            metadata.version
        );
        vortex_ensure!(
            metadata.stats_version == STATS_VERSION,
            "Unsupported {LAYOUT_ID} stats version: {}",
            metadata.stats_version
        );
        vortex_ensure!(
            children.nchildren() == 2,
            "{LAYOUT_ID} expected 2 children, got {}",
            children.nchildren()
        );

        Ok(RowGroupZoneMapLayout {
            dtype: dtype.clone(),
            children: children.to_arc(),
            metadata: metadata.clone(),
        })
    }
}

impl RowGroupZoneMapLayout {
    fn new(
        data: LayoutRef,
        zones: LayoutRef,
        metadata: RowGroupZoneMapMetadata,
    ) -> VortexResult<Self> {
        let dtype = data.dtype().clone();
        let expected_zones_dtype = zones_dtype(&dtype, &metadata)?;
        if zones.dtype() != &expected_zones_dtype {
            vortex_bail!(
                "Invalid {LAYOUT_ID} zones dtype: expected {}, got {}",
                expected_zones_dtype,
                zones.dtype()
            );
        }

        Ok(Self {
            dtype,
            children: Arc::new(MilvusLayoutChildren(vec![data, zones])),
            metadata,
        })
    }
}

#[derive(Clone)]
struct MilvusLayoutChildren(Vec<LayoutRef>);

impl LayoutChildren for MilvusLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        let Some(child) = self.0.get(idx) else {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.0.len());
        };
        if child.dtype() != dtype {
            vortex_bail!("Child dtype mismatch: {} != {}", child.dtype(), dtype);
        }
        Ok(child.clone())
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.0[idx].row_count()
    }

    fn nchildren(&self) -> usize {
        self.0.len()
    }
}

fn zones_dtype(dtype: &DType, metadata: &RowGroupZoneMapMetadata) -> VortexResult<DType> {
    let struct_fields = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("{LAYOUT_ID} data dtype must be a struct"))?;

    let mut names: Vec<FieldName> = Vec::with_capacity(metadata.columns.len() + 1);
    let mut dtypes: Vec<DType> = Vec::with_capacity(metadata.columns.len() + 1);

    for column in &metadata.columns {
        let column_dtype = struct_fields
            .field(&column.field_name)
            .ok_or_else(|| vortex_err!("Missing zonemap column {}", column.field_name))?;
        names.push(column.field_name.clone());
        dtypes.push(ZoneMap::dtype_for_stats_table(
            &column_dtype,
            &column.present_stats,
        ));
    }

    names.push(RG_BOUNDARIES_FIELD.into());
    dtypes.push(boundary_dtype());

    Ok(DType::Struct(
        StructFields::new(FieldNames::from(names), dtypes),
        Nullability::NonNullable,
    ))
}

fn boundary_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from([RG_START_FIELD, RG_LEN_FIELD]),
            vec![
                DType::Primitive(PType::U64, Nullability::NonNullable),
                DType::Primitive(PType::U64, Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    )
}

struct RowGroupZoneMapReader {
    layout: RowGroupZoneMapLayout,
    name: Arc<str>,
    // child[0] is the real data layout; child[1] is the auxiliary zones layout.
    lazy_children: LazyReaderChildren,
}

impl RowGroupZoneMapReader {
    fn try_new(
        layout: RowGroupZoneMapLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<Self> {
        let dtypes = vec![
            layout.dtype.clone(),
            zones_dtype(&layout.dtype, &layout.metadata)?,
        ];
        let names = vec![name.clone(), format!("{}.zones", name).into()];
        let lazy_children =
            LazyReaderChildren::new(layout.children.clone(), dtypes, names, segment_source);

        Ok(Self {
            layout,
            name,
            lazy_children,
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    fn zones_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(1)
    }

    fn available_stats(&self) -> FieldPathSet {
        // checked_pruning_expr works in root scope, so expose stats as `col.min`,
        // `col.max`, etc. even though the stored ZoneMap child uses local names.
        FieldPathSet::from_iter(self.layout.metadata.columns.iter().flat_map(|column| {
            column
                .present_stats
                .iter()
                .map(|stat| FieldPath::from_name(column.field_name.clone()).push(stat.name()))
        }))
    }
}

impl LayoutReader for RowGroupZoneMapReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.layout.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[vortex::dtype::FieldMask],
        row_range: &Range<u64>,
        splits: &mut std::collections::BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_child()?
            .register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Always preserve pruning behavior from the data child. RowGroupZoneMap pruning
        // can only further narrow the mask when the filter is safely mappable to one column.
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        let Some((predicate, column_name, present_stats, required_stats)) =
            self.pruning_predicate_for_single_column(expr)?
        else {
            return Ok(data_eval);
        };

        let column_dtype = self
            .layout
            .dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.field(&column_name))
            .ok_or_else(|| vortex_err!("Missing zonemap column {column_name}"))?;

        let rg_count = usize::try_from(self.layout.metadata.rg_count)
            .map_err(|_| vortex_err!("Too many row groups for {LAYOUT_ID}"))?;
        let zones_child = self.zones_child()?.clone();
        let zone_expr = get_item(column_name.clone(), root());
        let boundary_expr = get_item(RG_BOUNDARIES_FIELD, root());
        // This first version reads the full selected column ZoneMap and full boundary table.
        // They are small side data compared with row-group payloads, and keeping them whole
        // avoids a second index just to map row ranges back to row-group ids.
        let zone_eval = zones_child.projection_evaluation(
            &(0..self.layout.metadata.rg_count),
            &zone_expr,
            MaskFuture::new_true(rg_count),
        )?;
        let boundary_eval = zones_child.projection_evaluation(
            &(0..self.layout.metadata.rg_count),
            &boundary_expr,
            MaskFuture::new_true(rg_count),
        )?;

        let input_mask = mask.clone();
        let row_range = row_range.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let zone_array = zone_eval.await?.to_struct();
            // Validate that the stored per-column stats table still matches Vortex ZoneMap
            // shape before evaluating the root-scope pruning predicate against it.
            let zone_map = ZoneMap::try_new(column_dtype, zone_array, present_stats)?;
            let stats_view = Self::build_root_stats_view_for_column(&zone_map, &required_stats)?;
            let rg_prune_mask = predicate
                .evaluate(&stats_view)?
                .try_to_mask_fill_null_false()?;
            ROW_GROUP_ZONE_MAP_PRUNE_EVAL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ROW_GROUP_ZONE_MAP_PRUNED_ROW_GROUP_COUNT.fetch_add(
                rg_prune_mask.true_count() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            let boundaries = boundary_eval.await?.to_struct();
            let starts = boundaries
                .field_by_name(RG_START_FIELD)?
                .to_primitive()
                .as_slice::<u64>()
                .to_vec();
            let lens = boundaries
                .field_by_name(RG_LEN_FIELD)?
                .to_primitive()
                .as_slice::<u64>()
                .to_vec();

            // ZoneMap pruning returns one bit per row group, while LayoutReader must return
            // one keep bit per requested row. Boundaries bridge those two granularities.
            let row_keep = Self::expand_rg_prune_mask(&starts, &lens, &rg_prune_mask, &row_range)?;
            let mut stats_mask = input_mask.bitand(&row_keep);
            if !stats_mask.all_false() {
                stats_mask = stats_mask.bitand(&data_eval.await?);
            }
            Ok(stats_mask)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_child()?.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<vortex::layout::ArrayFuture> {
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}

impl RowGroupZoneMapReader {
    /// Build a conservative pruning plan for one column.
    ///
    /// Vortex gives us a root-scope stats predicate. We only use it when all
    /// required stats belong to one top-level field, because the zones layout stores
    /// one independent ZoneMap table per column.
    fn pruning_predicate_for_single_column(
        &self,
        expr: &Expression,
    ) -> VortexResult<
        Option<(
            Expression,
            FieldName,
            Arc<[Stat]>,
            vortex::expr::pruning::RequiredStats,
        )>,
    > {
        let Some((predicate, required_stats)) = checked_pruning_expr(expr, &self.available_stats())
        else {
            return Ok(None);
        };

        let Some(column_name) = Self::pruning_expr_selected_field(&required_stats) else {
            return Ok(None);
        };

        let Some(column) = self.layout.metadata.column(&column_name) else {
            return Ok(None);
        };
        Ok(Some((
            predicate,
            column_name,
            column.present_stats.clone(),
            required_stats,
        )))
    }

    fn pruning_expr_selected_field(
        required_stats: &vortex::expr::pruning::RequiredStats,
    ) -> Option<FieldName> {
        let mut selected: Option<FieldName> = None;
        for field_path in required_stats.map().keys() {
            // The initial implementation supports top-level struct fields only.
            // Anything nested or synthetic stays on the conservative path.
            let [Field::Name(field_name)] = field_path.parts() else {
                return None;
            };
            match &selected {
                Some(selected) if selected != field_name => return None,
                Some(_) => {}
                None => selected = Some(field_name.clone()),
            }
        }
        selected
    }

    fn build_root_stats_view_for_column(
        zone_map: &ZoneMap,
        required_stats: &vortex::expr::pruning::RequiredStats,
    ) -> VortexResult<ArrayRef> {
        let zone_array = zone_map.array();
        let mut names = Vec::new();
        let mut arrays = Vec::new();

        // checked_pruning_expr builds predicates in root stats scope, e.g. `x_max`.
        // The stored per-column ZoneMap keeps Vortex's local field names, e.g. `max`.
        // Build a cheap in-memory StructArray that renames ArrayRefs without copying data.
        for (field_path, stats) in required_stats.map() {
            for stat in stats {
                names.push(field_path_stat_field_name(field_path, *stat));
                arrays.push(zone_array.field_by_name(stat.name())?.clone());
            }
        }

        vortex_ensure!(
            !arrays.is_empty(),
            "No required stats available for {LAYOUT_ID} pruning"
        );

        Ok(StructArray::try_new(
            FieldNames::from(names),
            arrays,
            zone_array.len(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn expand_rg_prune_mask(
        starts: &[u64],
        lens: &[u64],
        rg_prune_mask: &Mask,
        row_range: &Range<u64>,
    ) -> VortexResult<Mask> {
        // rg_prune_mask uses ZoneMap semantics: true means "skip this row group".
        // The returned mask uses scan semantics: true means "keep this row".
        vortex_ensure!(
            starts.len() == lens.len(),
            "RG boundary start/len length mismatch: {} != {}",
            starts.len(),
            lens.len()
        );
        vortex_ensure!(
            starts.len() == rg_prune_mask.len(),
            "RG prune mask length mismatch: {} != {}",
            starts.len(),
            rg_prune_mask.len()
        );

        let mut out = MaskMut::new_true(0);
        for (rg_idx, (&rg_start, &rg_len)) in starts.iter().zip(lens).enumerate() {
            let rg_end = rg_start
                .checked_add(rg_len)
                .ok_or_else(|| vortex_err!("RG boundary overflow"))?;
            let start = rg_start.max(row_range.start);
            let end = rg_end.min(row_range.end);
            if start >= end {
                continue;
            }
            out.append_n(!rg_prune_mask.value(rg_idx), usize::try_from(end - start)?);
        }

        let expected_len = usize::try_from(row_range.end - row_range.start)?;
        vortex_ensure!(
            out.len() == expected_len,
            "Expanded RG mask length mismatch: {} != {}",
            out.len(),
            expected_len
        );
        Ok(out.freeze())
    }
}

#[derive(Clone)]
struct RowGroupSplitOptions {
    block_size_minimum: u64,
    canonicalize: bool,
}

struct RowGroupBuffer {
    data: std::collections::VecDeque<(ArrayRef, u64)>,
    nbytes: u64,
    block_size_minimum: u64,
}

impl RowGroupBuffer {
    fn new(block_size_minimum: u64) -> Self {
        Self {
            data: std::collections::VecDeque::new(),
            nbytes: 0,
            block_size_minimum,
        }
    }

    fn push(&mut self, chunk: ArrayRef) {
        let nbytes = chunk.nbytes() as u64;
        self.nbytes += nbytes;
        self.data.push_back((chunk, nbytes));
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.block_size_minimum
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn drain_one_group(&mut self, dtype: &DType) -> Option<ArrayRef> {
        if self.data.is_empty() {
            return None;
        }

        let mut group = Vec::new();
        let mut group_bytes = 0u64;

        while let Some((chunk, est_bytes)) = self.data.pop_front() {
            let chunk_len = chunk.len();
            self.nbytes -= est_bytes;

            if group_bytes + est_bytes <= self.block_size_minimum {
                group_bytes += est_bytes;
                group.push(chunk);
                if group_bytes >= self.block_size_minimum {
                    break;
                }
            } else {
                let space_left = self.block_size_minimum - group_bytes;
                let rows_to_take = if est_bytes > 0 {
                    space_left
                        .saturating_mul(chunk_len as u64)
                        .div_ceil(est_bytes) as usize
                } else {
                    chunk_len
                }
                .max(1)
                .min(chunk_len);

                group.push(chunk.slice(0..rows_to_take));

                if rows_to_take < chunk_len {
                    let right = chunk.slice(rows_to_take..chunk_len);
                    let right_est =
                        est_bytes * (chunk_len - rows_to_take) as u64 / chunk_len as u64;
                    self.nbytes += right_est;
                    self.data.push_front((right, right_est));
                }
                break;
            }
        }

        let chunked = vortex::arrays::ChunkedArray::try_new(group, dtype.clone()).ok()?;
        Some(chunked.to_canonical().into_array())
    }
}

fn split_stream_into_row_groups<F>(
    stream: SendableSequentialStream,
    options: RowGroupSplitOptions,
    on_row_group_emitted: F,
) -> SendableSequentialStream
where
    F: Fn(&ArrayRef) -> VortexResult<()> + Send + 'static,
{
    let dtype = stream.dtype().clone();
    let stream = if options.canonicalize {
        SequentialStreamAdapter::new(
            dtype.clone(),
            stream.map(|chunk| {
                let (sequence_id, chunk) = chunk?;
                VortexResult::Ok((sequence_id, chunk.to_canonical().into_array()))
            }),
        )
        .sendable()
    } else {
        stream
    };

    let dtype_clone = dtype.clone();
    let block_size_minimum = options.block_size_minimum;
    let row_group_stream = try_stream! {
        let stream = stream.peekable();
        pin_mut!(stream);
        let mut buffer = RowGroupBuffer::new(block_size_minimum);

        while let Some(chunk) = stream.as_mut().next().await {
            let (sequence_id, chunk) = chunk?;
            let mut sp = sequence_id.descend();

            if !chunk.is_empty() {
                buffer.push(chunk);
            }

            let is_eof = stream.as_mut().peek().await.is_none();
            while buffer.have_enough() || (is_eof && !buffer.is_empty()) {
                if let Some(row_group) = buffer.drain_one_group(&dtype_clone) {
                    on_row_group_emitted(&row_group)?;
                    yield (sp.advance(), row_group)
                } else {
                    break;
                }
            }
        }
    };

    SequentialStreamAdapter::new(dtype, row_group_stream).sendable()
}

/// Splits a stream of arrays into byte-size row groups without building zonemap side data.
///
/// This strategy is a writer-side stream adapter only: it does not serialize a custom layout
/// node into the Vortex footer. The emitted row-group arrays are passed to `child`, which
/// builds the normal V2 data layout tree:
///
/// ```text
/// ChunkedLayout                         // one chunk per row group
///   StructLayout                        // row group payload
///     field layouts: Chunked -> Compressed -> Flat
///     validity layouts: Collect -> Compressed -> Flat
/// ```
struct RowGroupSplitStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RowGroupSplitOptions,
}

impl RowGroupSplitStrategy {
    fn new(child: Arc<dyn LayoutStrategy>, options: RowGroupSplitOptions) -> Self {
        Self { child, options }
    }
}

#[async_trait]
impl LayoutStrategy for RowGroupSplitStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let row_group_stream =
            split_stream_into_row_groups(stream, self.options.clone(), |_: &ArrayRef| Ok(()));

        self.child
            .write_stream(ctx, segment_sink, row_group_stream, eof, handle)
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}

/// Builds the row-group data layout plus a sidecar zonemap layout used for pruning.
///
/// Unlike `RowGroupSplitStrategy`, this strategy does serialize a custom root layout. The
/// root's data child is the same row-grouped V2 data tree produced by `RowGroupSplitStrategy`.
/// The zones child is a struct with one row per row group: each real column with usable stats
/// stores a Vortex `ZoneMap` stats table, and `__rg_boundaries` maps row-group pruning results
/// back to row ranges.
///
/// ```text
/// RowGroupZoneMapLayout                 // encoding: milvus.v2_zoned_row_group
///   data: ChunkedLayout                 // one chunk per row group
///     StructLayout
///       field layouts: Chunked -> Compressed -> Flat
///       validity layouts: Collect -> Compressed -> Flat
///   zones: StructLayout                 // one row per row group
///     <column>: Compressed -> Flat      // per-column ZoneMap stats table
///     __rg_boundaries: Compressed -> Flat
/// ```
pub struct RowGroupZoneMapStrategy {
    data_child: Arc<dyn LayoutStrategy>,
    zones_child: Arc<dyn LayoutStrategy>,
    row_group_options: RowGroupSplitOptions,
    stats: Arc<[Stat]>,
    max_variable_length_statistics_size: usize,
}

impl RowGroupZoneMapStrategy {
    fn new(
        data_child: Arc<dyn LayoutStrategy>,
        zones_child: Arc<dyn LayoutStrategy>,
        row_group_options: RowGroupSplitOptions,
        stats: Arc<[Stat]>,
        max_variable_length_statistics_size: usize,
    ) -> Self {
        Self {
            data_child,
            zones_child,
            row_group_options,
            stats,
            max_variable_length_statistics_size,
        }
    }
}

struct StatsBuildState {
    column_names: Vec<FieldName>,
    accumulators: Vec<StatsAccumulator>,
    rg_starts: Vec<u64>,
    rg_lens: Vec<u64>,
    next_row: u64,
    stats: Arc<[Stat]>,
}

impl StatsBuildState {
    fn new(
        dtype: &DType,
        stats: Arc<[Stat]>,
        max_variable_length_statistics_size: usize,
    ) -> VortexResult<Self> {
        let struct_fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("{LAYOUT_ID} writer requires struct input"))?;
        // The boundaries table is a synthetic child under the zones struct.
        // Reject an input column with the same name so the serialized layout
        // has an unambiguous column-to-zone-map mapping.
        if struct_fields.find(RG_BOUNDARIES_FIELD).is_some() {
            vortex_bail!("Input schema field {RG_BOUNDARIES_FIELD} is reserved by {LAYOUT_ID}");
        }

        // Keep one Vortex StatsAccumulator per input column. Each accumulator
        // produces a standalone Vortex-compatible ZoneMap table whose row
        // count is the number of row groups, rather than building a custom
        // wide stats table across all columns.
        let mut column_names = Vec::with_capacity(struct_fields.nfields());
        let mut accumulators = Vec::with_capacity(struct_fields.nfields());
        for idx in 0..struct_fields.nfields() {
            let field_name = struct_fields
                .field_name(idx)
                .vortex_expect("field index checked")
                .clone();
            let field_dtype = struct_fields
                .field_by_index(idx)
                .vortex_expect("field index checked");
            column_names.push(field_name);
            accumulators.push(StatsAccumulator::new(
                &field_dtype,
                &stats,
                max_variable_length_statistics_size,
            ));
        }

        Ok(Self {
            column_names,
            accumulators,
            rg_starts: Vec::new(),
            rg_lens: Vec::new(),
            next_row: 0,
            stats,
        })
    }

    fn push_row_group(&mut self, row_group: &ArrayRef) -> VortexResult<()> {
        let rg_start = self.next_row;
        let rg_len = row_group.len() as u64;
        self.next_row = self
            .next_row
            .checked_add(rg_len)
            .ok_or_else(|| vortex_err!("row count overflow"))?;

        // This is called by split_stream_into_row_groups before the row group
        // is yielded to the data child strategy. The StructArray clone shares
        // the same field ArrayRefs as row_group, so computing statistics here
        // stores them in each field array's statistics cache.
        //
        // The zone-map accumulator reads those cached values via
        // push_chunk_without_compute, matching Vortex ZonedStrategy's
        // "compute before child write" pattern. Downstream data layout
        // strategies also receive the same field ArrayRefs and can observe the
        // cached field-level stats if they consult array.statistics().
        let struct_row_group = row_group.to_struct();
        for (field, accumulator) in struct_row_group
            .fields()
            .iter()
            .zip(self.accumulators.iter_mut())
        {
            field.statistics().compute_all(&self.stats)?;
            accumulator.push_chunk_without_compute(field.as_ref())?;
        }

        self.rg_starts.push(rg_start);
        self.rg_lens.push(rg_len);
        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Option<(StructArray, RowGroupZoneMapMetadata)>> {
        let rg_count = self.rg_starts.len();
        let mut zone_names = Vec::new();
        let mut zone_arrays = Vec::new();
        let mut metadata_columns = Vec::new();

        // Finalize each per-column accumulator independently. Columns whose
        // dtype cannot produce any requested stats simply do not get a ZoneMap
        // child, and are omitted from metadata so pruning treats them as
        // unavailable.
        for (column_name, accumulator) in self.column_names.iter().zip(self.accumulators.iter_mut())
        {
            let Some(zone_map) = accumulator.as_stats_table() else {
                continue;
            };
            if zone_map.present_stats().is_empty() {
                continue;
            }
            zone_names.push(column_name.clone());
            zone_arrays.push(zone_map.array().clone().into_array());
            metadata_columns.push(ColumnZoneMetadata {
                field_name: column_name.clone(),
                present_stats: zone_map.present_stats().clone(),
            });
        }

        if metadata_columns.is_empty() {
            return Ok(None);
        }

        // Keep row-group boundaries separate from per-column ZoneMap tables.
        // ZoneMap children preserve Vortex's dtype/field semantics, while this
        // synthetic table only maps RG-level pruning masks back to row ranges.
        let boundaries = boundary_array(&self.rg_starts, &self.rg_lens)?;
        zone_names.push(RG_BOUNDARIES_FIELD.into());
        zone_arrays.push(boundaries.into_array());

        // The zones struct has one row per row group. Metadata records only
        // the columns that actually have ZoneMap children plus their
        // present_stats, matching the validation needed by ZoneMap::try_new.
        let zones = StructArray::try_new(
            FieldNames::from(zone_names),
            zone_arrays,
            rg_count,
            Validity::NonNullable,
        )?;
        let metadata = RowGroupZoneMapMetadata::new(rg_count as u64, metadata_columns);
        Ok(Some((zones, metadata)))
    }
}

fn push_stats_state(
    stats_state: &Arc<Mutex<StatsBuildState>>,
    row_group: &ArrayRef,
) -> VortexResult<()> {
    stats_state
        .lock()
        .map_err(|_| vortex_err!("StatsBuildState mutex poisoned"))?
        .push_row_group(row_group)
}

fn boundary_array(starts: &[u64], lens: &[u64]) -> VortexResult<StructArray> {
    vortex_ensure!(
        starts.len() == lens.len(),
        "RG boundary start/len length mismatch: {} != {}",
        starts.len(),
        lens.len()
    );
    let starts = PrimitiveArray::new(Buffer::copy_from(starts), Validity::NonNullable).into_array();
    let lens = PrimitiveArray::new(Buffer::copy_from(lens), Validity::NonNullable).into_array();
    let len = starts.len();
    StructArray::try_new(
        FieldNames::from([RG_START_FIELD, RG_LEN_FIELD]),
        vec![starts, lens],
        len,
        Validity::NonNullable,
    )
}

#[async_trait]
impl LayoutStrategy for RowGroupZoneMapStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let stats_state = Arc::new(Mutex::new(StatsBuildState::new(
            &dtype,
            self.stats.clone(),
            self.max_variable_length_statistics_size,
        )?));

        let stats_state_for_stream = stats_state.clone();
        let row_group_stream = split_stream_into_row_groups(
            stream,
            self.row_group_options.clone(),
            move |row_group| push_stats_state(&stats_state_for_stream, row_group),
        );

        let data_eof = eof.split_off();
        let data_layout = self
            .data_child
            .write_stream(
                ctx.clone(),
                segment_sink.clone(),
                row_group_stream,
                data_eof,
                handle.clone(),
            )
            .await?;

        let Some((zones_array, metadata)) = stats_state
            .lock()
            .map_err(|_| vortex_err!("StatsBuildState mutex poisoned"))?
            .finish()?
        else {
            return Ok(data_layout);
        };

        let zones_stream = zones_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .zones_child
            .write_stream(ctx, segment_sink, zones_stream, eof, handle)
            .await?;

        Ok(RowGroupZoneMapLayout::new(data_layout, zones_layout, metadata)?.into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.data_child.buffered_bytes() + self.zones_child.buffered_bytes()
    }
}

pub fn build_row_group_strategy(
    row_group_max_size: u64,
    enable_zone_map: bool,
    stats: Arc<[Stat]>,
) -> Arc<dyn LayoutStrategy> {
    let data_strategy = build_data_strategy();
    let row_group_options = RowGroupSplitOptions {
        block_size_minimum: row_group_max_size,
        canonicalize: false,
    };

    if enable_zone_map {
        Arc::new(RowGroupZoneMapStrategy::new(
            data_strategy,
            build_zones_strategy(),
            row_group_options,
            stats,
            64,
        ))
    } else {
        Arc::new(RowGroupSplitStrategy::new(data_strategy, row_group_options))
    }
}

fn build_data_strategy() -> Arc<dyn LayoutStrategy> {
    let flat = FlatLayoutStrategy {
        inline_array_node: true,
        ..Default::default()
    };
    let compress_flat = CompressingStrategy::new_btrblocks(flat, false);
    let chunked_inner = ChunkedLayoutStrategy::new(compress_flat.clone());
    let validity = CollectStrategy::new(compress_flat);
    let struct_inner = StructStrategy::new(chunked_inner, validity);
    Arc::new(ChunkedLayoutStrategy::new(struct_inner))
}

fn build_zones_strategy() -> Arc<dyn LayoutStrategy> {
    let flat = FlatLayoutStrategy {
        inline_array_node: false,
        ..Default::default()
    };
    let compress_flat = CompressingStrategy::new_btrblocks(flat, false);
    let validity = CollectStrategy::new(compress_flat.clone());
    Arc::new(StructStrategy::new(compress_flat, validity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex::arrays::ChunkedArray;
    use vortex::buffer::{BitBufferMut, ByteBufferMut};
    use vortex::expr::{and, gt_eq, lit, lt};
    use vortex::file::{OpenOptionsSessionExt, VortexWriteOptions};
    use vortex::stream::ArrayStreamExt;

    #[test]
    fn metadata_roundtrip() {
        let metadata = RowGroupZoneMapMetadata::new(
            3,
            vec![ColumnZoneMetadata {
                field_name: "a".into(),
                present_stats: Arc::from([Stat::Max, Stat::Min]),
            }],
        );

        let roundtrip =
            RowGroupZoneMapMetadata::deserialize(&metadata.clone().serialize()).unwrap();
        assert_eq!(roundtrip, metadata);
    }

    #[test]
    fn expand_partial_row_range() {
        let starts = vec![0, 10, 20];
        let lens = vec![10, 10, 10];
        let rg_prune = Mask::from_buffer(BitBufferMut::from_iter([true, false, true]).freeze());
        let expanded =
            RowGroupZoneMapReader::expand_rg_prune_mask(&starts, &lens, &rg_prune, &(5..25))
                .unwrap();

        assert_eq!(expanded.len(), 20);
        for idx in 0..5 {
            assert!(!expanded.value(idx));
        }
        for idx in 5..15 {
            assert!(expanded.value(idx));
        }
        for idx in 15..20 {
            assert!(!expanded.value(idx));
        }
    }

    fn i32_struct_chunk(values: &[i32]) -> VortexResult<ArrayRef> {
        let len = values.len();
        let values =
            PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable).into_array();
        Ok(StructArray::try_new(
            FieldNames::from(["x"]),
            vec![values],
            len,
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[tokio::test]
    async fn scan_filter_uses_row_group_zone_map_pruning() -> VortexResult<()> {
        reset_row_group_zone_map_pruning_stats();

        let chunks = vec![
            i32_struct_chunk(&(0..10).collect::<Vec<_>>())?,
            i32_struct_chunk(&(100..110).collect::<Vec<_>>())?,
            i32_struct_chunk(&(200..210).collect::<Vec<_>>())?,
        ];
        let dtype = chunks[0].dtype().clone();
        let row_group_max_size = chunks[0].nbytes() as u64;
        let input = ChunkedArray::try_new(chunks, dtype.clone())?.into_array();

        let stats = Arc::<[Stat]>::from([Stat::Min, Stat::Max]);
        let mut buffer = ByteBufferMut::empty();
        VortexWriteOptions::new(crate::VORTEX_SESSION.clone())
            .with_file_statistics(stats.to_vec())
            .with_strategy(build_row_group_strategy(
                row_group_max_size,
                true,
                stats.clone(),
            ))
            .write(&mut buffer, input.to_array_stream())
            .await?;

        let file = crate::VORTEX_SESSION.open_options().open_buffer(buffer)?;
        let layout = file.footer().layout();
        assert_eq!(layout.encoding_id().as_ref(), LAYOUT_ID);
        assert_eq!(layout.child(1)?.row_count(), 3);

        let filter = and(
            gt_eq(get_item("x", root()), lit(100i32)),
            lt(get_item("x", root()), lit(110i32)),
        );
        let result = file
            .scan()?
            .with_filter(filter)
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(result.len(), 10);
        let x = result
            .to_struct()
            .field_by_name("x")?
            .clone()
            .to_primitive();
        assert_eq!(x.as_slice::<i32>(), &(100..110).collect::<Vec<_>>());

        let (prune_evals, pruned_row_groups) = row_group_zone_map_pruning_stats();
        assert!(
            prune_evals > 0,
            "expected scan filter to evaluate row-group zonemap pruning"
        );
        assert!(
            pruned_row_groups > 0,
            "expected row-group zonemap pruning to drop at least one row group"
        );
        Ok(())
    }
}
