// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Snapshot resolution, exploration and shared reader metadata. Retain Kernel
//! state in memory instead of serializing it into per-file descriptors.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use cxx::SharedPtr;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::engine::ensure_data_types::{ValidationMode, ensure_data_types};
#[allow(deprecated)]
use delta_kernel::parquet::arrow::async_reader::ParquetObjectReader;
use delta_kernel::parquet::file::metadata::ParquetMetaDataReader;
use delta_kernel::scan::state::{DvInfo, ScanFile};
use delta_kernel::scan::{ScanMetadata, StatsOptions};
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, PrimitiveType, StructField, StructType,
};
use delta_kernel::{Snapshot, SnapshotRef};
use delta_kernel_default_engine::DefaultEngineBuilder;
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use object_store::ObjectStore;
use object_store::path::Path;
use serde::Deserialize;
use url::Url;

use crate::TOKIO_RT;
use crate::deltalake_engine::DeltaEngine;
use crate::deltalake_ffi::{DeltaFileInfo, DeltaFilesystemBinding};
use crate::deltalake_object_store::{DeltaObjectStore, DeltaPathBinding};
use crate::deltalake_reader::ProjectionCache;
use crate::lance_ffi::FileSystemWrapper;

const INVALID: &str = "[delta:error=invalid]";
const UNSUPPORTED: &str = "[delta:error=unsupported]";

fn invalid(message: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{INVALID} {message}")
}

fn unsupported(message: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{UNSUPPORTED} {message}")
}

pub(crate) struct CachedScanFile {
    pub(crate) size: u64,
    pub(crate) num_records: Option<u64>,
    pub(crate) dv_info: DvInfo,
    pub(crate) batch_index: usize,
    pub(crate) row_index: usize,
}

// The schema, cached Arrow batches and indexed file facts stay Rust-owned.
// File plans slice one indexed row without copying the cached Arrow buffers.
pub struct DeltaSnapshotContext {
    pub(crate) engine: Arc<DeltaEngine>,
    pub(crate) snapshot: SnapshotRef,
    pub(crate) physical_schema: delta_kernel::schema::SchemaRef,
    pub(crate) scan_metadata: Vec<RecordBatch>,
    pub(crate) active_files: HashMap<String, CachedScanFile>,
    pub(crate) dv_string_bytes: usize,
    pub(crate) table_url: Url,
    pub(crate) binding: DeltaPathBinding,
    pub(crate) projections: std::sync::Mutex<ProjectionCache>,
    files: Vec<DeltaFileInfo>,
}

impl DeltaSnapshotContext {
    pub fn delta_snapshot_version(&self) -> u64 {
        self.snapshot.version()
    }

    pub fn delta_snapshot_files(&self) -> &[DeltaFileInfo] {
        &self.files
    }
}

pub fn delta_open_snapshot(
    filesystem: SharedPtr<FileSystemWrapper>,
    table_url: &str,
    requested_version: i64,
    options: DeltaFilesystemBinding,
) -> Result<Box<DeltaSnapshotContext>> {
    open_snapshot_impl(filesystem, table_url, requested_version, options, true)
        // CXX displays only the outer anyhow context. Flatten the chain so
        // filesystem status markers survive snapshot/footer context messages.
        .map_err(|error| anyhow!("{error:#}"))
}

pub fn delta_load_snapshot(
    filesystem: SharedPtr<FileSystemWrapper>,
    table_url: &str,
    version: i64,
    options: DeltaFilesystemBinding,
) -> Result<Box<DeltaSnapshotContext>> {
    ensure!(
        version >= 0,
        "{INVALID} reader requires an exact Delta version"
    );
    open_snapshot_impl(filesystem, table_url, version, options, false)
        .map_err(|error| anyhow!("{error:#}"))
}

fn open_snapshot_impl(
    filesystem: SharedPtr<FileSystemWrapper>,
    table_url: &str,
    requested_version: i64,
    options: DeltaFilesystemBinding,
    explore: bool,
) -> Result<Box<DeltaSnapshotContext>> {
    ensure!(
        !filesystem.is_null(),
        "{INVALID} Delta exploration requires a filesystem"
    );
    ensure!(
        requested_version >= -1,
        "{INVALID} invalid Delta snapshot version"
    );
    let table_url = Url::parse(table_url).map_err(|error| invalid(error))?;
    ensure!(
        table_url.path().ends_with('/'),
        "{INVALID} Delta table URL must end in '/'"
    );
    let binding = DeltaPathBinding::new(&options, &table_url).map_err(invalid)?;
    let store: Arc<dyn ObjectStore> = Arc::new(DeltaObjectStore::new(filesystem, binding.clone()));
    let executor = Arc::new(TokioMultiThreadExecutor::new(TOKIO_RT.handle().clone()));
    let engine = Arc::new(DeltaEngine::new(
        DefaultEngineBuilder::new(store.clone())
            .with_task_executor(executor)
            .build(),
        binding.clone(),
    ));
    let kernel_table_url = binding.kernel_table_url(&table_url).map_err(invalid)?;
    let builder = Snapshot::builder_for(kernel_table_url.as_str());
    let builder = if requested_version >= 0 {
        builder.at_version(requested_version as u64)
    } else {
        builder
    };
    let snapshot = builder.build(engine.as_ref()).with_context(|| {
        format!("Cannot resolve Delta table {table_url}, requested version {requested_version}")
    })?;
    let configuration = snapshot.table_configuration();
    ensure!(
        configuration.metadata().format_provider() == "parquet",
        "{UNSUPPORTED} Delta data-file format must be parquet"
    );
    validate_schema(
        snapshot.schema().as_ref(),
        configuration.logical_partition_columns(),
    )?;
    let mapping = configuration.column_mapping_mode();
    let partition_fields = configuration
        .logical_partition_columns()
        .iter()
        .map(|name| {
            let field = snapshot
                .schema()
                .field(name)
                .cloned()
                .ok_or_else(|| invalid(format!("missing partition field {name}")))?;
            if !matches!(field.data_type(), DataType::Primitive(_)) {
                return Err(unsupported(format!("non-scalar partition column {name}")));
            }
            Ok(field)
        })
        .collect::<Result<Vec<_>>>()?;

    // Keep the cache in scan_row_schema's raw-map/JSON-stats shape, which
    // scan_metadata_from can restore. Partition validation below still runs
    // before returning any files, including our stricter timestamp rule.
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_stats(StatsOptions::json_only())
        .without_row_transforms()
        .build()?;
    let mut context = DeltaSnapshotContext {
        engine: engine.clone(),
        snapshot,
        physical_schema: scan.physical_schema().clone(),
        scan_metadata: Vec::new(),
        active_files: HashMap::new(),
        dv_string_bytes: 0,
        table_url: table_url.clone(),
        binding: binding.clone(),
        projections: std::sync::Mutex::new(crate::deltalake_reader::new_projection_cache()),
        files: Vec::new(),
    };
    for batch in scan.scan_metadata(engine.as_ref())? {
        let (metadata, candidates) = select_scan_metadata(batch?)?;
        let batch_index = context.scan_metadata.len();
        for (row_index, candidate) in candidates.into_iter().enumerate() {
            let location = table_url
                .join(&candidate.path)
                .map_err(|error| invalid(format!("invalid Delta data path: {error}")))?;
            binding
                .validate_url(&location)
                .map_err(|error| unsupported(error))?;
            validate_partitions(&candidate, &partition_fields, mapping)?;
            let dv_string_bytes =
                validate_deletion_location(&candidate.dv_info, &table_url, &binding)?;
            context.dv_string_bytes = context.dv_string_bytes.saturating_add(dv_string_bytes);
            ensure!(
                candidate.size >= 0,
                "{INVALID} negative Delta file size: {location}"
            );

            let deleted = candidate
                .dv_info
                .cardinality()
                .map_err(invalid)?
                .unwrap_or(0);
            // Snapshot statistics describe physical rows, including rows later
            // deleted by a DV. Trust them here so normal exploration never opens
            // data files. Reader open checks actual schemas/counts.
            let physical_rows = match &candidate.stats {
                Some(stats) => stats.num_records,
                None if explore => TOKIO_RT
                    .block_on(load_footer_row_count(
                        store.clone(),
                        &location,
                        candidate.size as u64,
                    ))
                    .with_context(|| format!("Cannot count Delta file {location}"))?,
                None => 0,
            };
            let rows = if explore {
                live_rows(physical_rows, deleted)?
            } else {
                0
            };
            let path = location.to_string();
            let key = delta_file_key(&location)?;
            ensure!(
                !context.active_files.contains_key(&key),
                "{INVALID} duplicate active Delta data path: {location}"
            );
            if rows != 0 {
                context.files.push(DeltaFileInfo {
                    path: path.clone(),
                    file_size: candidate.size as u64,
                    row_count: rows,
                });
            }
            context.active_files.insert(
                key,
                CachedScanFile {
                    size: candidate.size as u64,
                    num_records: candidate.stats.map(|stats| stats.num_records),
                    dv_info: candidate.dv_info,
                    batch_index,
                    row_index,
                },
            );
        }
        context.scan_metadata.push(metadata);
    }
    context
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    Ok(Box::new(context))
}

pub(crate) fn select_scan_metadata(batch: ScanMetadata) -> Result<(RecordBatch, Vec<ScanFile>)> {
    let files = batch.visit_scan_files(Vec::new(), |files, file| files.push(file))?;
    let metadata = batch
        .scan_files
        .apply_selection_vector()?
        .try_into_record_batch()?;
    // Both operations retain selected rows in input order. Only active Add rows
    // may remain, so their compacted positions are stable file-index locators.
    ensure!(
        metadata.num_rows() == files.len(),
        "{INVALID} selected Delta metadata row does not identify an active file"
    );
    Ok((metadata, files))
}

// URL spelling can change at the external-URI boundary (percent case and
// escaped unreserved characters). Index by decoded ObjectStore identity.
pub(crate) fn delta_file_key(location: &Url) -> Result<String> {
    Ok(Path::from_url_path(location.path())?.to_string())
}

pub(crate) fn live_rows(physical: u64, deleted: u64) -> Result<u64> {
    // Kernel exposes u64 statistics, but Delta counts are signed 64-bit. Check
    // before subtracting so an excessive count cannot be hidden by a large DV.
    ensure!(
        physical <= i64::MAX as u64,
        "{INVALID} Delta physical row count exceeds signed 64-bit range"
    );
    physical
        .checked_sub(deleted)
        .ok_or_else(|| invalid("DV cardinality exceeds file row count"))
}

#[allow(deprecated)] // Upstream Kernel 0.30 uses this same Parquet metadata adapter.
async fn load_footer_row_count(
    store: Arc<dyn ObjectStore>,
    location: &Url,
    size: u64,
) -> Result<u64> {
    let path = if location.scheme() == "file" {
        Path::from_absolute_path(
            location
                .to_file_path()
                .map_err(|_| invalid("invalid local Delta URL"))?,
        )?
    } else {
        Path::from_url_path(location.path())?
    };
    let mut reader = ParquetObjectReader::new(store, path).with_file_size(size);
    let metadata = ParquetMetaDataReader::new()
        .load_and_finish(&mut reader, size)
        .await?;
    u64::try_from(metadata.file_metadata().num_rows())
        .map_err(|_| invalid("negative Parquet row count"))
}

fn validate_schema(schema: &StructType, partition_columns: &[String]) -> Result<()> {
    for field in schema.fields() {
        validate_type(field.data_type())?;
        validate_type_changes(field, partition_columns.contains(field.name()))?;
    }
    Ok(())
}

fn validate_type(data_type: &DataType) -> Result<()> {
    match data_type {
        DataType::Primitive(p) => match p {
            PrimitiveType::String
            | PrimitiveType::Long
            | PrimitiveType::Integer
            | PrimitiveType::Short
            | PrimitiveType::Byte
            | PrimitiveType::Float
            | PrimitiveType::Double
            | PrimitiveType::Boolean
            | PrimitiveType::Binary
            | PrimitiveType::Date
            | PrimitiveType::Timestamp
            | PrimitiveType::TimestampNtz
            | PrimitiveType::Void
            | PrimitiveType::Decimal(_) => Ok(()),
            _ => Err(unsupported(format!(
                "Delta type {data_type} is outside the initial reader subset"
            ))),
        },
        DataType::Struct(s) => validate_schema(s, &[]),
        DataType::Array(a) => validate_type(a.element_type()),
        DataType::Map(m) => {
            validate_type(m.key_type())?;
            validate_type(m.value_type())
        }
        _ => Err(unsupported(format!(
            "Delta type {data_type} is outside the initial reader subset"
        ))),
    }
}

fn validate_partitions(
    file: &ScanFile,
    fields: &[StructField],
    mapping: delta_kernel::table_features::ColumnMappingMode,
) -> Result<()> {
    for field in fields {
        let Some(raw) = file.partition_values.get(field.physical_name(mapping)) else {
            continue;
        };
        let DataType::Primitive(primitive) = field.data_type() else {
            unreachable!()
        };
        if raw.is_empty() {
            continue;
        } // Kernel's scan path handles typed empty/null values.
        if *primitive == PrimitiveType::Timestamp {
            // The protocol does not retain the writer's timezone. Require an
            // explicit offset instead of accepting Kernel's UTC assumption.
            chrono::DateTime::parse_from_str(raw, "%+").map_err(|_| {
                unsupported(format!(
                    "offset-less or invalid timestamp partition {}",
                    field.name()
                ))
            })?;
        }
        primitive
            .parse_scalar(raw)
            .map_err(|error| invalid(error))?;
    }
    Ok(())
}

// DvInfo intentionally hides its descriptor, but supports serde transport
// within the same Kernel version. Inspect that typed envelope only in memory;
// never make Kernel's serde representation part of our persisted descriptor.
#[derive(Deserialize)]
struct DvEnvelope {
    deletion_vector: Option<delta_kernel::actions::deletion_vector::DeletionVectorDescriptor>,
}

fn validate_deletion_location(
    info: &DvInfo,
    root: &Url,
    binding: &DeltaPathBinding,
) -> Result<usize> {
    if !info.has_vector() {
        return Ok(0);
    }
    let envelope: DvEnvelope =
        serde_json::from_value(serde_json::to_value(info)?).map_err(invalid)?;
    if let Some(descriptor) = envelope.deletion_vector {
        if let Some(location) = descriptor.absolute_path(root)? {
            binding.validate_url(&location).map_err(unsupported)?;
        }
        // Account for the descriptor's retained path/inline bytes during this
        // existing inspection pass instead of re-serializing it for estimates.
        return Ok(descriptor.path_or_inline_dv.len());
    }
    Ok(0)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TypeChange {
    from_type: DataType,
    to_type: DataType,
    #[serde(default)]
    field_path: Option<String>,
}

// Validate all history, not only the current projection. Kernel's physical-read
// casts also allow timestamp/checkpoint accommodations that are not Delta type
// widening, so restrict the families before asking Kernel about losslessness.
fn supports_widening(from: &DataType, to: &DataType) -> bool {
    use PrimitiveType::*;
    let (DataType::Primitive(source), DataType::Primitive(target)) = (from, to) else {
        return false;
    };
    let allowed_family = matches!(
        (source, target),
        (Byte, Short | Integer | Long | Double | Decimal(_))
            | (Short, Integer | Long | Double | Decimal(_))
            | (Integer, Long | Double | Decimal(_))
            | (Long, Decimal(_))
            | (Float, Double)
            | (Decimal(_), Decimal(_))
            | (Date, TimestampNtz)
    );
    if !allowed_family || from == to {
        return false;
    }
    let Ok(arrow_source) = from.try_into_arrow() else {
        return false;
    };
    ensure_data_types(to, &arrow_source, ValidationMode::TypesOnly).is_ok()
}

fn supports_partition_widening(change: &TypeChange) -> bool {
    match (&change.from_type, &change.to_type) {
        (
            DataType::Primitive(PrimitiveType::Date),
            DataType::Primitive(PrimitiveType::TimestampNtz),
        ) => false,
        (source, DataType::Primitive(PrimitiveType::Decimal(target))) => match source {
            DataType::Primitive(PrimitiveType::Decimal(source)) => source.scale() == target.scale(),
            _ => target.scale() == 0,
        },
        _ => true,
    }
}

fn type_change_target<'a>(field: &'a StructField, path: Option<&str>) -> Result<&'a DataType> {
    let mut target = field.data_type();
    for segment in path
        .filter(|path| !path.is_empty())
        .into_iter()
        .flat_map(|path| path.split('.'))
    {
        target = match (target, segment) {
            (DataType::Array(array), "element") => array.element_type(),
            (DataType::Map(map), "key") => map.key_type(),
            (DataType::Map(map), "value") => map.value_type(),
            _ => {
                return Err(invalid(format!(
                    "invalid type-change fieldPath for {}",
                    field.name()
                )));
            }
        };
    }
    Ok(target)
}

fn validate_type_changes(field: &StructField, partition: bool) -> Result<()> {
    let Some(metadata) = field.get_config_value(&ColumnMetadataKey::TypeChanges) else {
        return Ok(());
    };
    let MetadataValue::Other(value) = metadata else {
        return Err(invalid(format!(
            "delta.typeChanges for {} must be an array",
            field.name()
        )));
    };
    let changes: Vec<TypeChange> = serde_json::from_value(value.clone()).map_err(|error| {
        invalid(format!(
            "invalid delta.typeChanges for {}: {error}",
            field.name()
        ))
    })?;
    for change in changes {
        if !supports_widening(&change.from_type, &change.to_type) {
            return Err(unsupported(format!(
                "Delta type change {} -> {} on {} is outside the initial reader subset",
                change.from_type,
                change.to_type,
                field.name()
            )));
        }
        let target = type_change_target(field, change.field_path.as_deref())?;
        ensure!(
            change.to_type == *target || supports_widening(&change.to_type, target),
            "{INVALID} type-change history does not match current field {}",
            field.name()
        );
        if partition && !supports_partition_widening(&change) {
            return Err(unsupported(format!(
                "Delta partition type change {} -> {} on {} requires additional string conversion",
                change.from_type,
                change.to_type,
                field.name()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::schema::{ArrayType, MapType};
    use serde_json::json;

    fn field_with_changes(
        data_type: impl Into<DataType>,
        changes: serde_json::Value,
    ) -> StructField {
        StructField::nullable("value", data_type).with_metadata([(
            ColumnMetadataKey::TypeChanges.as_ref(),
            MetadataValue::Other(changes),
        )])
    }

    #[test]
    fn live_row_count_is_checked() {
        assert_eq!(live_rows(10, 3).unwrap(), 7);
        assert_eq!(live_rows(10, 10).unwrap(), 0);
        assert!(live_rows(3, 4).is_err());
        assert!(live_rows(u64::MAX, 0).is_err());
        assert!(live_rows(i64::MAX as u64 + 1, 1).is_err());
    }

    #[test]
    fn file_identity_normalizes_url_spelling_without_decoding_twice() {
        let key = |path: &str| delta_file_key(&Url::parse(path).unwrap()).unwrap();
        assert_eq!(
            key("s3://bucket/table/%70art.parquet"),
            key("s3://bucket/table/part.parquet")
        );
        assert_eq!(
            key("s3://bucket/table/a%2fb.parquet"),
            key("s3://bucket/table/a/b.parquet")
        );
        assert_eq!(
            key("s3://bucket/table/%e9%9b%aa.parquet"),
            key("s3://bucket/table/%E9%9B%AA.parquet")
        );
        assert_ne!(
            key("s3://bucket/table/a%252Fb.parquet"),
            key("s3://bucket/table/a%2Fb.parquet")
        );
    }

    #[test]
    fn widening_only_accepts_lossless_supported_families() {
        assert!(supports_widening(&DataType::INTEGER, &DataType::LONG));
        assert!(supports_widening(&DataType::INTEGER, &DataType::DOUBLE));
        assert!(supports_widening(&DataType::FLOAT, &DataType::DOUBLE));
        assert!(supports_widening(&DataType::DATE, &DataType::TIMESTAMP_NTZ));
        assert!(!supports_widening(&DataType::LONG, &DataType::INTEGER));
        assert!(!supports_widening(&DataType::LONG, &DataType::DOUBLE));
        assert!(!supports_widening(&DataType::INTEGER, &DataType::DATE));
        assert!(!supports_widening(
            &DataType::TIMESTAMP,
            &DataType::TIMESTAMP_NTZ
        ));
        let decimal = |p, s| DataType::decimal(p, s).unwrap();
        assert!(supports_widening(&decimal(10, 2), &decimal(12, 4)));
        assert!(!supports_widening(&decimal(10, 2), &decimal(11, 4)));
        assert!(supports_widening(&DataType::LONG, &decimal(22, 2)));
        assert!(!supports_widening(&DataType::LONG, &decimal(21, 2)));
    }

    #[test]
    fn history_validates_container_paths_and_multiple_changes() {
        let field = field_with_changes(
            ArrayType::new(MapType::new(DataType::STRING, DataType::LONG, true), true),
            json!([
                {"fromType": "short", "toType": "integer", "fieldPath": "element.value"},
                {"fromType": "integer", "toType": "long", "fieldPath": "element.value"}
            ]),
        );
        validate_type_changes(&field, false).unwrap();
        for path in ["element.element", "element.value.missing", "missing", ""] {
            let invalid = field_with_changes(
                field.data_type().clone(),
                json!([{"fromType": "integer", "toType": "long", "fieldPath": path}]),
            );
            assert!(validate_type_changes(&invalid, false).is_err(), "{path}");
        }
        let inconsistent = field_with_changes(
            DataType::STRING,
            json!([{"fromType": "integer", "toType": "long"}]),
        );
        assert!(validate_type_changes(&inconsistent, false).is_err());
    }

    #[test]
    fn unsupported_history_in_nested_fields_is_rejected_without_files() {
        let bad = field_with_changes(
            DataType::INTEGER,
            json!([{"fromType": "long", "toType": "integer"}]),
        );
        let nested = StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable(
                "unprojected",
                ArrayType::new(StructType::try_new([bad]).unwrap(), true),
            ),
        ])
        .unwrap();
        assert!(validate_schema(&nested, &[]).is_err());
        for malformed in [json!(null), json!({}), json!([{"fromType": "integer"}])] {
            let field = field_with_changes(DataType::LONG, malformed);
            assert!(validate_type_changes(&field, false).is_err());
        }
    }

    #[test]
    fn partition_changes_preserve_the_existing_string_parser() {
        let cases = [
            ("integer", "long", true),
            ("float", "double", true),
            ("decimal(10,2)", "decimal(12,2)", true),
            ("integer", "decimal(10,0)", true),
            ("decimal(10,2)", "decimal(12,3)", false),
            ("integer", "decimal(12,2)", false),
            ("date", "timestamp_ntz", false),
        ];
        for (from, to, partition_supported) in cases {
            let data_type: DataType = serde_json::from_value(json!(to)).unwrap();
            let field = field_with_changes(data_type, json!([{"fromType": from, "toType": to}]));
            validate_type_changes(&field, false).unwrap();
            assert_eq!(
                validate_type_changes(&field, true).is_ok(),
                partition_supported,
                "{from} -> {to}"
            );
        }
    }
}
