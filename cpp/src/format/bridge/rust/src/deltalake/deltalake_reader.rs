// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Kernel owns schema adaptation and DV decoding; C++ owns Parquet I/O.

use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use delta_kernel::EngineData;
use delta_kernel::arrow::array::{Array, StructArray};
use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
use delta_kernel::arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use delta_kernel::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::engine::arrow_utils::fixup_parquet_read;
use delta_kernel::expressions::ExpressionRef;
use delta_kernel::scan::state::transform_to_logical;
use delta_kernel::scan::{Scan, get_transform_for_row};
use delta_kernel::schema::StructType;
use lru::LruCache;

use crate::deltalake_bridgeimpl::{DeltaSnapshotContext, delta_file_key, live_rows};
use crate::deltalake_engine::DeltaEngine;
use crate::deltalake_projection::FileProjection;

const INVALID: &str = "[delta:error=invalid]";

pub(crate) type ProjectionCache = LruCache<Vec<String>, Arc<SnapshotProjection>>;

pub(crate) struct SnapshotProjection {
    scan: Scan,
    physical_schema: ArrowSchema,
    output_schema: Arc<ArrowSchema>,
}

impl SnapshotProjection {
    fn new(scan: Scan) -> Result<Self> {
        let output_schema = Arc::new(scan.logical_schema().as_ref().try_into_arrow()?);
        let physical_schema = scan.physical_schema().as_ref().try_into_arrow()?;
        Ok(Self {
            scan,
            physical_schema,
            output_schema,
        })
    }

    fn file_transform(
        &self,
        engine: &DeltaEngine,
        metadata: RecordBatch,
    ) -> Result<Option<ExpressionRef>> {
        ensure!(
            metadata.num_rows() == 1,
            "{INVALID} expected one Delta file metadata row"
        );
        let input = vec![Box::new(ArrowEngineData::new(metadata)) as Box<dyn EngineData>];
        // At the same version Kernel processes only the supplied row, with no
        // log replay. This is per-file planning, not a complete table scan.
        let mut batches =
            self.scan
                .scan_metadata_from(engine, self.scan.snapshot().version(), input, None)?;
        let batch = batches
            .next()
            .transpose()?
            .ok_or_else(|| anyhow!("{INVALID} Delta projection lost an active file"))?;
        ensure!(
            batch.scan_files.data().len() == 1
                && batch
                    .scan_files
                    .selection_vector()
                    .first()
                    .copied()
                    .unwrap_or(true)
                && batches.next().transpose()?.is_none(),
            "{INVALID} Delta projection must return exactly one selected file"
        );
        // Read the expression directly: rebuilding a ScanFile would reparse
        // statistics and copy partition maps already held by the snapshot.
        // An absent expression is a valid identity transform.
        Ok(get_transform_for_row(0, &batch.scan_file_transforms))
    }
}

impl DeltaSnapshotContext {
    fn projection(&self, names: Vec<String>) -> Result<Arc<SnapshotProjection>> {
        // Only schema/scan construction runs under this lock. Per-file transforms
        // are built outside it, without retaining expressions for unrelated files.
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| anyhow!("projection cache poisoned"))?;
        if let Some(plan) = cache.get(&names) {
            return Ok(plan.clone());
        }
        let fields = names
            .iter()
            .map(|name| {
                self.snapshot
                    .schema()
                    .field(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("{INVALID} unknown Delta field {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let logical =
            Arc::new(StructType::try_new(fields).map_err(|error| anyhow!("{INVALID} {error}"))?);
        let scan = self
            .snapshot
            .clone()
            .scan_builder()
            .with_schema(logical)
            .build()?;
        let plan = Arc::new(SnapshotProjection::new(scan)?);
        cache.put(names, plan.clone());
        Ok(plan)
    }

    pub unsafe fn delta_snapshot_export_schema(&self, output: usize) -> Result<()> {
        let schema: ArrowSchema = self.snapshot.schema().as_ref().try_into_arrow()?;
        unsafe { export_schema(&schema, output) }
    }

    pub fn delta_snapshot_has_partition_columns(&self) -> bool {
        !self
            .snapshot
            .table_configuration()
            .logical_partition_columns()
            .is_empty()
    }

    pub fn delta_snapshot_memory_size(&self) -> u64 {
        let batches: usize = self
            .scan_metadata
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum();
        let index: usize = self
            .active_files
            .iter()
            .map(|(path, file)| std::mem::size_of_val(file) + path.len())
            .sum();
        batches
            .saturating_add(index)
            .saturating_add(self.dv_string_bytes) as u64
    }

    pub unsafe fn delta_load_file(
        &self,
        location: &str,
        file_schema: usize,
        rows: u64,
        expected_live: u64,
        expected_size: u64,
    ) -> Result<Box<DeltaFileContext>> {
        let result = || -> Result<Box<DeltaFileContext>> {
            let url = self.table_url.join(location)?;
            self.binding.validate_url(&url)?;
            let location = delta_file_key(&url)?;
            let file = self.active_files.get(&location).ok_or_else(|| {
                anyhow!("{INVALID} file is not active in the pinned Delta snapshot: {location}")
            })?;
            ensure!(
                file.size == expected_size,
                "{INVALID} Delta file size does not match snapshot"
            );
            ensure!(file_schema != 0, "{INVALID} null Parquet file schema");
            let schema = unsafe { FFI_ArrowSchema::from_raw(file_schema as *mut _) };
            let schema = ArrowSchema::try_from(&schema)?;
            if let Some(num_records) = file.num_records {
                ensure!(
                    num_records == rows,
                    "{INVALID} Parquet count differs from Delta snapshot statistics"
                );
            }
            let deleted = file.dv_info.cardinality()?.unwrap_or(0);
            ensure!(
                live_rows(rows, deleted)? == expected_live,
                "{INVALID} Delta live count does not match descriptor"
            );
            // Validate all visible data fields even for a narrower read projection.
            // The baseline scan already supplied the full physical schema; no
            // full-table transform projection is needed to validate a file schema.
            let input = FileProjection::new(&schema, &self.physical_arrow_schema)?;
            // Fixup validates the original types and field IDs of matched fields
            // without reading payloads. The projection already aligns field order.
            fixup_parquet_read(
                input.empty_batch()?,
                &[],
                None,
                None,
                Some(&self.physical_schema),
            )?;
            let deletions = file
                .dv_info
                .get_row_indexes(self.engine.as_ref(), &self.table_url)?
                .unwrap_or_default();
            ensure!(
                deletions.len() as u64 == deleted,
                "{INVALID} DV bitmap cardinality differs from descriptor"
            );
            ensure!(
                deletions.last().is_none_or(|position| *position < rows),
                "{INVALID} DV position exceeds Parquet row count"
            );
            ensure!(
                deletions.windows(2).all(|p| p[0] < p[1]),
                "{INVALID} DV positions are not sorted and unique"
            );
            Ok(Box::new(DeltaFileContext {
                location,
                schema,
                deletions,
                rows,
                live_rows: expected_live,
            }))
        };
        result().map_err(|error| anyhow!("{error:#}"))
    }
}

pub struct DeltaFileContext {
    location: String,
    schema: ArrowSchema,
    deletions: Vec<u64>,
    rows: u64,
    live_rows: u64,
}

impl DeltaFileContext {
    pub fn delta_file_deletions(&self) -> &[u64] {
        &self.deletions
    }
    pub fn delta_file_rows(&self) -> u64 {
        self.rows
    }
    pub fn delta_file_live_rows(&self) -> u64 {
        self.live_rows
    }

    pub fn delta_file_memory_size(&self) -> u64 {
        let fields: usize = self.schema.fields().iter().map(|field| field.size()).sum();
        let metadata: usize = self
            .schema
            .metadata()
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum();
        // Approximate file schema and DV state. The shared snapshot is accounted
        // separately from individual data files.
        (std::mem::size_of::<Self>()
            + self.location.capacity()
            + self.deletions.capacity() * std::mem::size_of::<u64>()
            + fields
            + metadata) as u64
    }

    pub fn delta_file_plan(
        &self,
        snapshot: &DeltaSnapshotContext,
        names: Vec<String>,
    ) -> Result<Box<DeltaReadPlan>> {
        let result = || -> Result<Box<DeltaReadPlan>> {
            let projection = snapshot.projection(names)?;
            let file = snapshot.active_files.get(&self.location).ok_or_else(|| {
                anyhow!("{INVALID} file is not active in the pinned Delta snapshot")
            })?;
            let metadata = snapshot.scan_metadata[file.batch_index].slice(file.row_index, 1);
            let transform = projection.file_transform(snapshot.engine.as_ref(), metadata)?;
            let input = FileProjection::new(&self.schema, &projection.physical_schema)?;
            Ok(Box::new(DeltaReadPlan {
                projection,
                transform,
                input,
                engine: snapshot.engine.clone(),
            }))
        };
        result().map_err(|error| anyhow!("{error:#}"))
    }
}

pub struct DeltaReadPlan {
    projection: Arc<SnapshotProjection>,
    transform: Option<ExpressionRef>,
    input: FileProjection,
    engine: Arc<DeltaEngine>,
}

impl DeltaReadPlan {
    pub fn delta_plan_columns(&self) -> &[String] {
        self.input.columns()
    }

    pub unsafe fn delta_plan_export_schema(&self, output: usize) -> Result<()> {
        unsafe { export_schema(self.projection.output_schema.as_ref(), output) }
    }

    pub unsafe fn delta_plan_apply(
        &self,
        input_array: usize,
        input_schema: usize,
        output_array: usize,
        output_schema: usize,
    ) -> Result<()> {
        let result = || -> Result<()> {
            ensure!(
                input_array != 0 && input_schema != 0 && output_array != 0 && output_schema != 0,
                "{INVALID} null Arrow C data pointer"
            );
            // Move both C objects immediately. C++ release guards see cleared
            // callbacks; Rust releases imported buffers on every error path.
            let array = unsafe { FFI_ArrowArray::from_raw(input_array as *mut _) };
            let schema = unsafe { FFI_ArrowSchema::from_raw(input_schema as *mut _) };
            let data = unsafe { from_ffi(array, &schema)? };
            let batch = self
                .input
                .apply(RecordBatch::from(StructArray::from(data)))?;
            let rows = batch.num_rows();
            // The projection preserves container validity and offsets and puts
            // every field in physical schema order. Kernel handles casts and
            // schema normalization without any further reordering.
            let physical = fixup_parquet_read(
                batch,
                &[],
                None,
                None,
                Some(self.projection.scan.physical_schema()),
            )?;
            let logical = transform_to_logical(
                self.engine.as_ref(),
                Box::new(physical),
                self.projection.scan.physical_schema(),
                self.projection.scan.logical_schema(),
                self.transform.clone(),
            )?
            .try_into_record_batch()?;
            // Kernel preserves some producer layouts. Normalize representation
            // only (e.g. view strings / timestamp units) before Arrow 17 imports.
            let expected = &self.projection.output_schema;
            ensure!(
                logical.num_rows() == rows && logical.num_columns() == expected.fields().len(),
                "{INVALID} Kernel transform changed batch row count or output shape"
            );
            let columns = logical
                .columns()
                .iter()
                .zip(expected.fields())
                .map(|(column, field)| {
                    if column.data_type() == field.data_type() {
                        Ok(column.clone())
                    } else {
                        delta_kernel::arrow::compute::cast(column, field.data_type())
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let batch = RecordBatch::try_new_with_options(
                expected.clone(),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(logical.num_rows())),
            )?;
            let array = StructArray::from(batch);
            let (array, schema) = to_ffi(&array.to_data())?;
            unsafe {
                std::ptr::write(output_array as *mut FFI_ArrowArray, array);
                std::ptr::write(output_schema as *mut FFI_ArrowSchema, schema);
            }
            Ok(())
        };
        result().map_err(|error| anyhow!("{error:#}"))
    }
}

unsafe fn export_schema(schema: &ArrowSchema, output: usize) -> Result<()> {
    ensure!(output != 0, "{INVALID} null Arrow C schema");
    let schema = FFI_ArrowSchema::try_from(schema)?;
    unsafe { std::ptr::write(output as *mut FFI_ArrowSchema, schema) };
    Ok(())
}

pub(crate) fn new_projection_cache() -> ProjectionCache {
    // Active readers retain plans strongly after eviction. Bound inactive
    // schema/scan projections; per-file transforms belong only to live read plans.
    LruCache::new(NonZeroUsize::new(8).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::arrow::array::{Int64Array, StringArray};
    use delta_kernel::arrow::datatypes::{DataType as ArrowType, Field as ArrowField};
    use delta_kernel::engine::arrow_utils::parquet_read_plan;
    use delta_kernel::engine_data::FilteredEngineData;
    use delta_kernel::parquet::arrow::ProjectionMask;
    use delta_kernel::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use delta_kernel::scan::{ScanMetadata, StatsOptions};
    use delta_kernel::{Snapshot, SnapshotRef};
    use delta_kernel_default_engine::DefaultEngineBuilder;
    use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use serde_json::json;
    use url::Url;

    use crate::deltalake_bridgeimpl::select_scan_metadata;
    use crate::deltalake_ffi::{DeltaFilesystemBinding, DeltaStorageKind};
    use crate::deltalake_object_store::DeltaPathBinding;

    struct ProjectionFixture {
        engine: Arc<DeltaEngine>,
        snapshot: SnapshotRef,
        metadata: Vec<RecordBatch>,
    }

    impl ProjectionFixture {
        fn new(mapping: &str) -> Self {
            let field = |name, data_type, id| {
                json!({
                    "name": name, "type": data_type, "nullable": true,
                    "metadata": if mapping == "none" { json!({}) } else { json!({
                        "delta.columnMapping.id": id,
                        "delta.columnMapping.physicalName": format!("col-{id}")
                    }) }
                })
            };
            let schema = json!({"type": "struct", "fields": [
                field("id", "long", 1), field("region", "string", 2)
            ]});
            let mut actions = vec![
                if mapping == "none" {
                    json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}})
                } else {
                    json!({"protocol": {
                        "minReaderVersion": 3, "minWriterVersion": 7,
                        "readerFeatures": ["columnMapping"], "writerFeatures": ["columnMapping"]
                    }})
                },
                json!({"metaData": {
                    "id": "projection-unit-test", "format": {"provider": "parquet", "options": {}},
                    "schemaString": schema.to_string(), "partitionColumns": ["region"],
                    "configuration": {
                        "delta.columnMapping.mode": mapping, "delta.columnMapping.maxColumnId": "2"
                    }
                }}),
            ];
            let partition = if mapping == "none" { "region" } else { "col-2" };
            for region in ["US", "EU", "CN"] {
                actions.push(json!({"add": {
                    "path": format!("{region}.parquet"), "size": 100,
                    "modificationTime": 0, "dataChange": true,
                    "partitionValues": {partition: region}, "stats": "{\"numRecords\":2}"
                }}));
            }
            let store = Arc::new(InMemory::new());
            let log_path = Path::from("table/_delta_log/00000000000000000000.json");
            let log = actions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            crate::TOKIO_RT
                .block_on(store.put(&log_path, log.into()))
                .unwrap();
            let table_url = Url::parse("s3://bucket/table/").unwrap();
            let binding = DeltaPathBinding::new(
                &DeltaFilesystemBinding {
                    kind: DeltaStorageKind::Remote,
                    scheme: "s3".into(),
                    authority: "bucket".into(),
                    local_root: String::new(),
                },
                &table_url,
            )
            .unwrap();
            let executor = Arc::new(TokioMultiThreadExecutor::new(
                crate::TOKIO_RT.handle().clone(),
            ));
            let engine = Arc::new(DeltaEngine::new(
                DefaultEngineBuilder::new(store.clone())
                    .with_task_executor(executor)
                    .build(),
                binding,
            ));
            let snapshot = Snapshot::builder_for(table_url.as_str())
                .at_version(0)
                .build(engine.as_ref())
                .unwrap();
            let scan = snapshot
                .clone()
                .scan_builder()
                .with_stats(StatsOptions::json_only())
                .without_row_transforms()
                .build()
                .unwrap();
            let metadata = scan
                .scan_metadata(engine.as_ref())
                .unwrap()
                .map(|batch| select_scan_metadata(batch.unwrap()).unwrap().0)
                .collect::<Vec<_>>();
            // All later planning must use cached metadata; no log or data file
            // remains available, even in the in-memory store.
            crate::TOKIO_RT.block_on(store.delete(&log_path)).unwrap();
            Self {
                engine,
                snapshot,
                metadata,
            }
        }
    }

    #[test]
    fn deltalake_cached_metadata_rows_follow_selection() {
        let fixture = ProjectionFixture::new("none");
        assert_eq!(fixture.metadata.len(), 1);
        let source = &fixture.metadata[0];
        assert_eq!(source.num_rows(), 3);
        let paths = source
            .column_by_name("path")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for selection in [
            vec![],
            vec![false],
            vec![false, true, false],
            vec![true, false, true],
            vec![false; 3],
        ] {
            let expected = (0..source.num_rows())
                .filter(|row| selection.get(*row).copied().unwrap_or(true))
                .map(|row| paths.value(row))
                .collect::<Vec<_>>();
            let batch = ScanMetadata {
                scan_files: FilteredEngineData::try_new(
                    Box::new(ArrowEngineData::new(source.clone())),
                    selection,
                )
                .unwrap(),
                scan_file_transforms: Vec::new(),
            };
            let (metadata, files) = select_scan_metadata(batch).unwrap();
            assert_eq!(files.len(), expected.len());
            for (row, file) in files.iter().enumerate() {
                let slice = metadata.slice(row, 1);
                let path = slice
                    .column_by_name("path")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                assert_eq!(file.path, expected[row]);
                assert_eq!(path.value(0), file.path);
            }
        }
    }

    #[test]
    fn deltalake_lazy_file_transforms_match_full_batch_planning() {
        for mapping in ["none", "name", "id"] {
            let fixture = ProjectionFixture::new(mapping);
            for names in [
                vec!["region", "id"],
                vec!["id", "region"],
                vec!["region"],
                vec!["id"],
                vec![],
            ] {
                let schema = fixture.snapshot.schema();
                let logical = Arc::new(
                    StructType::try_new(
                        names.iter().map(|name| schema.field(name).unwrap().clone()),
                    )
                    .unwrap(),
                );
                let scan = fixture
                    .snapshot
                    .clone()
                    .scan_builder()
                    .with_schema(logical)
                    .build()
                    .unwrap();
                let projection = SnapshotProjection::new(scan).unwrap();
                let input = fixture
                    .metadata
                    .iter()
                    .cloned()
                    .map(|batch| Box::new(ArrowEngineData::new(batch)) as Box<dyn EngineData>)
                    .collect::<Vec<_>>();
                let mut eager_files = Vec::new();
                for batch in projection
                    .scan
                    .scan_metadata_from(
                        fixture.engine.as_ref(),
                        fixture.snapshot.version(),
                        input,
                        None,
                    )
                    .unwrap()
                {
                    eager_files = batch
                        .unwrap()
                        .visit_scan_files(eager_files, |files, file| files.push(file))
                        .unwrap();
                }
                let physical_schema: ArrowSchema = projection
                    .scan
                    .physical_schema()
                    .as_ref()
                    .try_into_arrow()
                    .unwrap();
                let columns = physical_schema
                    .fields()
                    .iter()
                    .map(|_| Arc::new(Int64Array::from(vec![17, 23])) as _)
                    .collect();
                let physical = RecordBatch::try_new_with_options(
                    Arc::new(physical_schema),
                    columns,
                    &RecordBatchOptions::new().with_row_count(Some(2)),
                )
                .unwrap();
                for metadata in &fixture.metadata {
                    for row in 0..metadata.num_rows() {
                        let slice = metadata.slice(row, 1);
                        let path = slice
                            .column_by_name("path")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .value(0);
                        let eager = eager_files.iter().find(|file| file.path == path).unwrap();
                        let transform = projection
                            .file_transform(fixture.engine.as_ref(), slice.clone())
                            .unwrap();
                        assert_eq!(transform, eager.transform, "{mapping}: {names:?}: {path}");
                        let apply = |transform| {
                            transform_to_logical(
                                fixture.engine.as_ref(),
                                Box::new(ArrowEngineData::new(physical.clone())),
                                projection.scan.physical_schema(),
                                projection.scan.logical_schema(),
                                transform,
                            )
                            .unwrap()
                            .try_into_record_batch()
                            .unwrap()
                        };
                        let output = apply(transform);
                        assert_eq!(output, apply(eager.transform.clone()));
                        assert_eq!(output.num_rows(), 2);
                        assert_eq!(
                            output
                                .schema()
                                .fields()
                                .iter()
                                .map(|field| field.name().as_str())
                                .collect::<Vec<_>>(),
                            names
                        );
                        for (column, name) in names.iter().enumerate() {
                            if *name == "region" {
                                let values = output
                                    .column(column)
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .unwrap();
                                let partition = if mapping == "none" { "region" } else { "col-2" };
                                assert_eq!(values.value(0), eager.partition_values[partition]);
                                assert_eq!(values.value(1), values.value(0));
                            } else {
                                let values = output
                                    .column(column)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                assert_eq!(values.values().as_ref(), &[17, 23]);
                            }
                        }
                    }
                }
            }
        }
    }

    fn kernel_projected_parquet_batch(
        input: &RecordBatch,
        requested: Arc<StructType>,
    ) -> (RecordBatch, RecordBatch) {
        use delta_kernel::parquet::arrow::ArrowWriter;
        use delta_kernel::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, input.schema(), None).unwrap();
        writer.write(input).unwrap();
        writer.close().unwrap();
        let bytes = bytes::Bytes::from(buffer);
        let metadata = ArrowReaderMetadata::load(&bytes, ArrowReaderOptions::default()).unwrap();
        let read = |mask| {
            ParquetRecordBatchReaderBuilder::new_with_metadata(bytes.clone(), metadata.clone())
                .with_projection(mask)
                .build()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
        };
        let control = read(ProjectionMask::all());
        let (ordering, mask) = parquet_read_plan(&requested, &metadata).unwrap();
        let projected = read(mask.unwrap_or_else(ProjectionMask::all));
        let fixed = fixup_parquet_read(projected, &ordering, None, None, Some(&requested)).unwrap();
        (control, fixed.record_batch().clone())
    }

    #[test]
    fn deltalake_kernel_zero_leaf_lists_keep_pinned_behavior() {
        use delta_kernel::arrow::array::ListArray;
        use delta_kernel::arrow::buffer::{NullBuffer, OffsetBuffer};
        use delta_kernel::schema::{ArrayType, DataType, StructField};

        let values = StructArray::new(
            vec![Arc::new(ArrowField::new("old", ArrowType::Int64, true))].into(),
            vec![Arc::new(Int64Array::from(vec![Some(7)]))],
            None,
        );
        let items = ListArray::new(
            Arc::new(ArrowField::new("element", values.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 1, 1, 1].into()),
            Arc::new(values),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let input = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", ArrowType::Int64, false),
                ArrowField::new("items", items.data_type().clone(), true),
            ])),
            vec![Arc::new(Int64Array::from(vec![0, 1, 2])), Arc::new(items)],
        )
        .unwrap();
        let requested = Arc::new(
            StructType::try_new([
                StructField::not_null("id", DataType::LONG),
                StructField::nullable(
                    "items",
                    ArrayType::new(
                        StructType::try_new([StructField::nullable("new", DataType::STRING)])
                            .unwrap(),
                        true,
                    ),
                ),
            ])
            .unwrap(),
        );
        let (control, actual) = kernel_projected_parquet_batch(&input, requested);
        assert_eq!(control, input);
        assert_eq!(actual.num_rows(), 3);
        let items = actual
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // Characterize the raw Kernel behavior: it replaces both
        // an existing nonempty list and an empty list with null, losing shape.
        assert_eq!(items.null_count(), 3);
        assert_eq!(items.value_offsets(), &[0, 0, 0, 0]);
    }

    // Run explicitly with --ignored. This asserts the desired read semantics,
    // not the incorrect output, and calls the pinned Kernel helpers directly.
    // Upstream: https://github.com/delta-io/delta-kernel-rs/issues/3961
    #[test]
    #[ignore = "Kernel 913389c loses existing struct validity when all requested children are missing"]
    fn deltalake_kernel_preserves_existing_struct_with_only_missing_children() {
        use delta_kernel::arrow::array::Int64Array;
        use delta_kernel::arrow::buffer::NullBuffer;
        use delta_kernel::schema::{DataType, StructField};

        // Stored profile values: {old: 7}, {old: null}, null.
        let profile = StructArray::new(
            vec![Arc::new(ArrowField::new("old", ArrowType::Int64, true))].into(),
            vec![Arc::new(Int64Array::from(vec![Some(7), None, None]))],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowType::Int64, false),
            ArrowField::new("profile", profile.data_type().clone(), true),
        ]));
        let input = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![0, 1, 2])), Arc::new(profile)],
        )
        .unwrap();
        let validity = |batch: &RecordBatch| {
            (0..batch.num_rows())
                .map(|row| batch.column(1).is_valid(row))
                .collect::<Vec<_>>()
        };
        let requested = Arc::new(
            StructType::try_new([
                StructField::not_null("id", DataType::LONG),
                StructField::nullable(
                    "profile",
                    StructType::try_new([StructField::nullable("new", DataType::STRING)]).unwrap(),
                ),
            ])
            .unwrap(),
        );
        let (control, actual) = kernel_projected_parquet_batch(&input, requested);
        // Parquet contains enough information to distinguish a null child from
        // a null parent, independently of the Kernel projection/fixup helpers.
        assert_eq!(validity(&control), vec![true, true, false]);
        assert_eq!(actual.num_rows(), 3);
        let profile = actual
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(profile.column(0).null_count(), 3);
        assert_eq!(
            validity(&actual),
            vec![true, true, false],
            "synthesizing profile.new must preserve the existing profile parent validity"
        );
    }
}
