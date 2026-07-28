// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Paimon planning and streaming bridge.
//!
//! The descriptor owns a small versioned envelope around Paimon's
//! Java-compatible `SplitSerializer` payload. The envelope carries integrity
//! and identity metadata while the codec dispatcher keeps old manifests
//! readable across upstream wire-format upgrades.

use anyhow::{Context, Result, anyhow, bail, ensure};
use arrow_schema58::SchemaRef;
use arrow58::error::ArrowError;
use arrow58::ffi::FFI_ArrowSchema;
use arrow58::ffi_stream::FFI_ArrowArrayStream;
use arrow58::record_batch::{RecordBatch, RecordBatchReader};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use futures::{StreamExt, stream::BoxStream};
use paimon::catalog::Identifier;
use paimon::io::{FileIO, FileRead};
use paimon::spec::MergeEngine;
use paimon::table::{SchemaManager, SnapshotManager};
use paimon::{DataSplit, DeletionFile, Table};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Cursor;

use crate::TOKIO_RT;
use crate::paimon_ffi::PaimonFileInfo;

const METADATA_VERSION: u32 = 1;
const ERROR_INVALID_PREFIX: &str = "[paimon:error=invalid]";
const ERROR_NOT_IMPLEMENTED_PREFIX: &str = "[paimon:error=not-implemented]";
const DELETION_VECTOR_MAGIC: u32 = 1_581_511_376;
/// Magic of Paimon's 64-bit deletion vectors, from Java
/// `org.apache.paimon.deletionvectors.Bitmap64DeletionVector.MAGIC_NUMBER`.
/// Java writes this magic little-endian and records the complete serialized
/// size in `DeletionFile.length`, unlike the bitmap32 envelope.
const DELETION_VECTOR_BITMAP64_MAGIC: u32 = 1_681_511_377;
const MAX_DATA_SPLIT_METADATA_BYTES: usize = 12 * 1024 * 1024;
/// Soft warning threshold for one atomic DataSplit. A split is the smallest
/// unit Milvus can place into a segment, so an oversized one defeats segment
/// balancing; planning is the only layer that still sees per-file physical
/// sizes (descriptors deliberately carry no byte hints).
const OVERSIZED_SPLIT_WARN_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    Auto,
    DirectFile,
    DataSplit,
}

impl ScanMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "direct-file" => Ok(Self::DirectFile),
            "data-split" => Ok(Self::DataSplit),
            other => bail!(
                "invalid Paimon scan mode '{other}'; expected auto, direct-file, or data-split"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadPath {
    DirectFile,
    DataSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteReason {
    AutoDirectFile,
    ForcedDirectFile,
    ForcedDataSplit,
    MergeSemantics,
    RowRange,
    SchemaEvolution,
    DataEvolution,
    DedicatedVectorFile,
    UnsupportedFormat,
}

impl RouteReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::AutoDirectFile => "auto-direct-file",
            Self::ForcedDirectFile => "forced-direct-file",
            Self::ForcedDataSplit => "forced-data-split",
            Self::MergeSemantics => "merge-semantics",
            Self::RowRange => "row-range",
            Self::SchemaEvolution => "schema-evolution",
            Self::DataEvolution => "data-evolution",
            Self::DedicatedVectorFile => "dedicated-vector-file",
            Self::UnsupportedFormat => "unsupported-format",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RouteDecision {
    read_path: ReadPath,
    reason: RouteReason,
}

#[derive(Debug, Clone, Copy, Default)]
struct TableReadSemantics {
    merge_engine: Option<MergeEngine>,
    deletion_vectors_enabled: bool,
    deletion_vectors_merge_on_read: bool,
    has_blob_fields: bool,
}

impl TableReadSemantics {
    fn from_table(table: &Table) -> Result<Self> {
        let schema = table.schema();
        let options = schema.core_options();
        Ok(Self {
            merge_engine: (!schema.primary_keys().is_empty())
                .then(|| options.merge_engine())
                .transpose()?,
            deletion_vectors_enabled: options.deletion_vectors_enabled(),
            deletion_vectors_merge_on_read: options.deletion_vectors_merge_on_read(),
            has_blob_fields: !options.blob_fields().is_empty(),
        })
    }
}

impl ReadPath {
    fn as_str(self) -> &'static str {
        match self {
            Self::DirectFile => "direct-file",
            Self::DataSplit => "data-split",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaimonMetadata {
    version: u32,
    read_path: String,
    data_format: String,
    snapshot_id: i64,
    bucket: i32,
    record_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    table_location: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    codec: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deletion_file: Option<DeletionFileDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeletionFileDescriptor {
    path: String,
    offset: u64,
    length: u64,
    cardinality: i64,
}

pub(crate) fn options_from_vecs(
    keys: Vec<String>,
    values: Vec<String>,
) -> Result<HashMap<String, String>> {
    ensure!(
        keys.len() == values.len(),
        "storage option key/value count mismatch: {} keys, {} values",
        keys.len(),
        values.len()
    );
    Ok(keys.into_iter().zip(values).collect())
}

fn table_name(location: &str) -> String {
    location
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("table")
        .to_string()
}

/// Prove that the pinned snapshot exists before time travel pins it.
///
/// `Table::copy_with_time_travel` mirrors Java's `tryTimeTravel` and swallows
/// EVERY resolution failure into a silent fallback, so a transient storage
/// error while loading `snapshot-N` would be indistinguishable from the
/// snapshot having expired. This probe separates the three states:
///
/// - confirmed present -> continue into time travel;
/// - confirmed absent (`exists()` returned `Ok(false)`) -> a terminal
///   input-state error carrying the earliest/latest bounds and the
///   "refresh the external collection" advice (the C++ boundary maps this
///   marker to `Status::Invalid`);
/// - storage error -> propagated as-is WITHOUT the refresh advice so it keeps
///   retryable IO semantics.
async fn ensure_pinned_snapshot_exists(
    file_io: &FileIO,
    table_location: &str,
    snapshot_id: i64,
) -> Result<()> {
    let manager = SnapshotManager::new(file_io.clone(), table_location.to_string());
    let snapshot_path = manager.snapshot_path(snapshot_id);
    let input = file_io
        .new_input(&snapshot_path)
        .with_context(|| format!("cannot address Paimon snapshot file '{snapshot_path}'"))?;
    let exists = input.exists().await.with_context(|| {
        format!("cannot check Paimon snapshot {snapshot_id} for table {table_location}")
    })?;
    if exists {
        return Ok(());
    }
    // The snapshot file is confirmed missing. Resolve the live bounds so the
    // error is actionable. A failed bound lookup keeps IO semantics: without
    // bounds the non-existence story is no longer complete, and the caller
    // may retry.
    let earliest = manager.earliest_snapshot_id().await.with_context(|| {
        format!("cannot resolve the earliest Paimon snapshot for table {table_location}")
    })?;
    let latest = manager.get_latest_snapshot_id().await.with_context(|| {
        format!("cannot resolve the latest Paimon snapshot for table {table_location}")
    })?;
    let bound = |value: Option<i64>| value.map_or_else(|| "none".to_string(), |id| id.to_string());
    bail!(
        "{ERROR_INVALID_PREFIX} Paimon snapshot {snapshot_id} no longer exists for table {table_location} \
         (earliest={}, latest={}); refresh the external collection",
        bound(earliest),
        bound(latest)
    )
}

async fn load_table(
    table_location: &str,
    storage_options: &HashMap<String, String>,
    snapshot_id: Option<i64>,
) -> Result<Table> {
    let file_io = FileIO::from_path(table_location)
        .with_context(|| format!("cannot infer Paimon storage from '{table_location}'"))?
        .with_props(storage_options)
        .build()
        .with_context(|| format!("cannot build Paimon FileIO for '{table_location}'"))?;
    let schema = SchemaManager::new(file_io.clone(), table_location.to_string())
        .latest()
        .await?
        .ok_or_else(|| anyhow!("Paimon table has no schema: {table_location}"))?;
    if let Some(snapshot_id) = snapshot_id {
        ensure_pinned_snapshot_exists(&file_io, table_location, snapshot_id).await?;
    }
    let table = Table::new(
        file_io,
        Identifier::new("unknown", table_name(table_location)),
        table_location.to_string(),
        (*schema).clone(),
        None,
    );
    if let Some(snapshot_id) = snapshot_id {
        let table = table
            .copy_with_time_travel(HashMap::from([(
                "scan.snapshot-id".to_string(),
                snapshot_id.to_string(),
            )]))
            .await?;
        // The probe above already proved existence; reaching this branch means
        // the snapshot disappeared in between (expiration race) or its content
        // is unreadable. Both are terminal states for this descriptor.
        ensure!(
            table.has_resolved_travel_snapshot(),
            "{ERROR_INVALID_PREFIX} Paimon snapshot {snapshot_id} no longer exists for table {table_location}; refresh the external collection"
        );
        Ok(table)
    } else {
        Ok(table)
    }
}

fn file_format(path: &str) -> &'static str {
    let name = path.split('?').next().unwrap_or(path).to_ascii_lowercase();
    if name.ends_with(".parquet") {
        "parquet"
    } else if name.ends_with(".vortex") {
        "vortex"
    } else if name.ends_with(".orc") {
        "orc"
    } else {
        "unknown"
    }
}

fn direct_file_ineligibility(
    split: &DataSplit,
    table_schema_id: i64,
    table: TableReadSemantics,
) -> Option<(RouteReason, String)> {
    if table.has_blob_fields {
        return Some((
            RouteReason::DataEvolution,
            "table contains Paimon BLOB fields that require Paimon's blob resolver".to_string(),
        ));
    }
    if matches!(
        table.merge_engine,
        Some(MergeEngine::PartialUpdate | MergeEngine::Aggregation)
    ) {
        if !table.deletion_vectors_enabled {
            return Some((
                RouteReason::MergeSemantics,
                "partial-update and aggregation tables require Paimon merge-on-read".to_string(),
            ));
        }
        if table.deletion_vectors_merge_on_read {
            return Some((
                RouteReason::MergeSemantics,
                "partial-update and aggregation tables with deletion-vectors.merge-on-read=true require Paimon's reader"
                    .to_string(),
            ));
        }
        let fully_materialized = split.raw_convertible()
            && split
                .data_files()
                .iter()
                .all(|file| file.level != 0 && file.delete_row_count == Some(0));
        if !fully_materialized {
            return Some((
                RouteReason::MergeSemantics,
                "partial-update and aggregation deletion-vector splits must be fully materialized"
                    .to_string(),
            ));
        }
    }
    if !split.raw_convertible() {
        return Some((
            RouteReason::MergeSemantics,
            "split requires Paimon merge-on-read".to_string(),
        ));
    }
    if split.row_ranges().is_some() {
        return Some((
            RouteReason::RowRange,
            "split carries row ranges".to_string(),
        ));
    }
    for file in split.data_files() {
        // raw_convertible only describes merge semantics. Paimon's normal
        // reader also reconciles an older physical file schema with the
        // snapshot schema (added/defaulted/renamed fields), which a bare
        // Parquet or Vortex reader cannot reproduce.
        if file.schema_id != table_schema_id {
            return Some((
                RouteReason::SchemaEvolution,
                format!(
                    "data file '{}' uses schema {} while the snapshot uses schema {}",
                    file.file_name, file.schema_id, table_schema_id
                ),
            ));
        }
        if file.file_name.to_ascii_lowercase().contains(".vector.") {
            return Some((
                RouteReason::DedicatedVectorFile,
                format!(
                    "data file '{}' is a dedicated vector-store file that requires Paimon column merge",
                    file.file_name
                ),
            ));
        }
        if file.delete_row_count.is_some_and(|count| count > 0) {
            return Some((
                RouteReason::MergeSemantics,
                format!("data file '{}' records deleted rows", file.file_name),
            ));
        }
        if file.write_cols.is_some() || !file.extra_files.is_empty() {
            return Some((
                RouteReason::DataEvolution,
                format!(
                    "data file '{}' uses data-evolution or sidecar semantics",
                    file.file_name
                ),
            ));
        }
    }
    let files = split.data_files();
    if files.len() > 1 && files.iter().any(|file| file.first_row_id.is_some()) {
        if files.iter().any(|file| file.first_row_id.is_none()) {
            return Some((
                RouteReason::DataEvolution,
                "split mixes row-tracking and non-row-tracking data files".to_string(),
            ));
        }
        let mut ranges = files
            .iter()
            .filter_map(|file| file.row_id_range())
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 >= pair[1].0) {
            return Some((
                RouteReason::DataEvolution,
                "split contains overlapping row-id ranges that require Paimon column merge"
                    .to_string(),
            ));
        }
    }
    if split.data_files().iter().any(|file| {
        !matches!(
            file_format(&split.data_file_path(file)),
            "parquet" | "vortex"
        )
    }) {
        return Some((
            RouteReason::UnsupportedFormat,
            "direct-file currently supports Parquet and Vortex data files only".to_string(),
        ));
    }
    None
}

fn decide_route(
    mode: ScanMode,
    split: &DataSplit,
    table_schema_id: i64,
    table: TableReadSemantics,
) -> Result<RouteDecision> {
    match mode {
        ScanMode::DataSplit => Ok(RouteDecision {
            read_path: ReadPath::DataSplit,
            reason: RouteReason::ForcedDataSplit,
        }),
        ScanMode::Auto => match direct_file_ineligibility(split, table_schema_id, table) {
            None => Ok(RouteDecision {
                read_path: ReadPath::DirectFile,
                reason: RouteReason::AutoDirectFile,
            }),
            Some((reason, _)) => Ok(RouteDecision {
                read_path: ReadPath::DataSplit,
                reason,
            }),
        },
        ScanMode::DirectFile => match direct_file_ineligibility(split, table_schema_id, table) {
            None => Ok(RouteDecision {
                read_path: ReadPath::DirectFile,
                reason: RouteReason::ForcedDirectFile,
            }),
            Some((_, detail)) => bail!("Paimon split cannot use direct-file: {detail}"),
        },
    }
}

fn checked_non_negative(value: i64, name: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("Paimon {name} is negative: {value}"))
}

fn metadata_split_row_count(split: &DataSplit) -> Result<Option<u64>> {
    if split.row_ranges().is_some() {
        return Ok(None);
    }
    split
        .merged_row_count()
        .map(|count| checked_non_negative(count, "merged row count"))
        .transpose()
}

/// Build the plan-time advisory for an oversized atomic DataSplit, or `None`
/// when the split is within bounds. Pure so the check is unit-testable; the
/// planner logs the returned message. Advisory only — the split still plans
/// and reads, because rejecting it would turn a sizing concern into an
/// availability failure.
fn oversized_split_warning(split: &DataSplit, table_location: &str) -> Option<String> {
    let total_bytes: u64 = split
        .data_files()
        .iter()
        .map(|file| u64::try_from(file.file_size).unwrap_or(0))
        .fold(0u64, u64::saturating_add);
    (total_bytes > OVERSIZED_SPLIT_WARN_BYTES).then(|| {
        format!(
            "Paimon DataSplit for table {table_location} (snapshot {}, bucket {}) spans {} bytes \
             across {} data files, above the {} byte advisory threshold; consider lowering \
             source.split.target-size or reviewing bucket sizing so one atomic split stays \
             segment-sized",
            split.snapshot_id(),
            split.bucket(),
            total_bytes,
            split.data_files().len(),
            OVERSIZED_SPLIT_WARN_BYTES
        )
    })
}

async fn count_split_rows(table: &Table, split: &DataSplit) -> Result<u64> {
    // `DataSplit::merged_row_count` describes the full file group and does
    // not account for an attached row-range selection. Such splits must be
    // counted through the actual Paimon reader.
    if let Some(count) = metadata_split_row_count(split)? {
        return Ok(count);
    }

    // Count through Paimon's merge reader, but request a zero-column Arrow
    // projection so the fallback scans row cardinality rather than payload.
    let mut builder = table.new_read_builder();
    builder.with_projection(&[])?;
    let read = builder.new_read()?;
    let mut stream = read.to_arrow(std::slice::from_ref(split))?;
    let mut rows = 0u64;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        rows = rows
            .checked_add(u64::try_from(batch.num_rows())?)
            .ok_or_else(|| anyhow!("Paimon split row count overflow"))?;
    }
    Ok(rows)
}

fn encode_split_metadata(
    table_location: &str,
    split: &DataSplit,
    record_count: u64,
) -> Result<String> {
    let encoded = crate::paimon_split_serde::serialize(split)?;
    let checksum = hex::encode(Sha256::digest(&encoded.payload));
    serde_json::to_string(&PaimonMetadata {
        version: METADATA_VERSION,
        read_path: ReadPath::DataSplit.as_str().to_string(),
        data_format: "paimon".to_string(),
        snapshot_id: split.snapshot_id(),
        bucket: split.bucket(),
        record_count,
        table_location: Some(table_location.to_string()),
        codec: Some(encoded.codec.to_string()),
        payload_sha256: Some(checksum),
        payload: Some(BASE64_STANDARD.encode(encoded.payload)),
        deletion_file: None,
    })
    .map_err(Into::into)
}

fn decode_split_metadata(metadata_json: &str) -> Result<(PaimonMetadata, DataSplit)> {
    ensure!(
        metadata_json.len() <= MAX_DATA_SPLIT_METADATA_BYTES,
        "Paimon metadata descriptor is {} bytes, above the {} byte limit",
        metadata_json.len(),
        MAX_DATA_SPLIT_METADATA_BYTES
    );
    let metadata: PaimonMetadata =
        serde_json::from_str(metadata_json).context("cannot parse Paimon metadata descriptor")?;
    ensure!(
        metadata.version == METADATA_VERSION,
        "unsupported Paimon metadata version {}; expected {}",
        metadata.version,
        METADATA_VERSION
    );
    ensure!(
        metadata.read_path == ReadPath::DataSplit.as_str(),
        "Paimon descriptor read_path is '{}', expected data-split",
        metadata.read_path
    );
    let codec = metadata.codec.as_deref().unwrap_or_default();
    let encoded = metadata
        .payload
        .as_deref()
        .ok_or_else(|| anyhow!("Paimon data-split descriptor has no payload"))?;
    let max_base64_bytes = crate::paimon_split_serde::MAX_DESCRIPTOR_BYTES
        .div_ceil(3)
        .checked_mul(4)
        .ok_or_else(|| anyhow!("Paimon descriptor base64 limit overflow"))?;
    ensure!(
        encoded.len() <= max_base64_bytes,
        "Paimon split payload is too large before base64 decoding"
    );
    let payload = BASE64_STANDARD
        .decode(encoded)
        .context("cannot decode Paimon split payload as base64")?;
    let actual_checksum = hex::encode(Sha256::digest(&payload));
    ensure!(
        metadata.payload_sha256.as_deref() == Some(actual_checksum.as_str()),
        "Paimon split descriptor checksum mismatch"
    );
    let split = crate::paimon_split_serde::deserialize(codec, &payload)?;
    Ok((metadata, split))
}

fn validate_data_split_binding(
    metadata: &PaimonMetadata,
    split: &DataSplit,
    expected_table_location: &str,
) -> Result<()> {
    let table_location = metadata
        .table_location
        .as_deref()
        .ok_or_else(|| anyhow!("Paimon data-split descriptor has no table_location"))?;
    ensure!(
        !expected_table_location.is_empty(),
        "Paimon data-split reader requires an expected table location"
    );
    ensure!(
        table_location == expected_table_location,
        "Paimon descriptor table location '{}' disagrees with manifest table location '{}'",
        table_location,
        expected_table_location
    );
    ensure!(
        metadata.snapshot_id == split.snapshot_id(),
        "Paimon descriptor snapshot {} disagrees with split snapshot {}",
        metadata.snapshot_id,
        split.snapshot_id()
    );
    ensure!(
        metadata.bucket == split.bucket(),
        "Paimon descriptor bucket {} disagrees with split bucket {}",
        metadata.bucket,
        split.bucket()
    );
    Ok(())
}

fn deletion_descriptor(file: &DeletionFile) -> Result<DeletionFileDescriptor> {
    Ok(DeletionFileDescriptor {
        path: file.path().to_string(),
        offset: checked_non_negative(file.offset(), "deletion vector offset")?,
        length: checked_non_negative(file.length(), "deletion vector length")?,
        cardinality: file.cardinality().unwrap_or(-1),
    })
}

fn encode_direct_metadata(
    split: &DataSplit,
    data_format: &str,
    record_count: u64,
    deletion_file: Option<&DeletionFile>,
) -> Result<String> {
    serde_json::to_string(&PaimonMetadata {
        version: METADATA_VERSION,
        read_path: ReadPath::DirectFile.as_str().to_string(),
        data_format: data_format.to_string(),
        snapshot_id: split.snapshot_id(),
        bucket: split.bucket(),
        record_count,
        table_location: None,
        codec: None,
        payload_sha256: None,
        payload: None,
        deletion_file: deletion_file.map(deletion_descriptor).transpose()?,
    })
    .map_err(Into::into)
}

pub fn paimon_plan_files(
    table_location: &str,
    snapshot_id: i64,
    scan_mode: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Vec<PaimonFileInfo>> {
    let mode = ScanMode::parse(scan_mode)?;
    let options = options_from_vecs(storage_options_keys, storage_options_values)?;
    TOKIO_RT.block_on(async move {
        let snapshot_id = (snapshot_id >= 0).then_some(snapshot_id);
        let table = load_table(table_location, &options, snapshot_id).await?;
        let table_read_semantics = TableReadSemantics::from_table(&table)?;
        let plan = table.new_read_builder().new_scan().plan().await?;
        let mut result = Vec::new();
        for split in plan.splits() {
            // Direct-file descriptors do not pass through the DataSplit codec,
            // so validate the planner output before either route can consume
            // positional deletion-file slots or row metadata.
            crate::paimon_split_serde::validate_split(split)?;
            let route = decide_route(
                mode,
                split,
                table.schema().id(),
                table_read_semantics,
            )?;
            match route.read_path {
                ReadPath::DataSplit => {
                    let record_count = count_split_rows(&table, split).await?;
                    if record_count == 0 {
                        continue;
                    }
                    if let Some(warning) = oversized_split_warning(split, table_location) {
                        eprintln!("{warning}");
                    }
                    result.push(PaimonFileInfo {
                        path: table_location.to_string(),
                        read_path: route.read_path.as_str().to_string(),
                        route_reason: route.reason.as_str().to_string(),
                        file_size: 0,
                        metadata_json: encode_split_metadata(table_location, split, record_count)?,
                    });
                }
                ReadPath::DirectFile => {
                    for (index, file) in split.data_files().iter().enumerate() {
                        let deletion = split.deletion_file_for_data_file_index(index);
                        let physical_rows = checked_non_negative(file.row_count, "data file row count")?;
                        let deleted_rows = match deletion {
                            None => 0,
                            Some(deletion) => match deletion.cardinality() {
                                Some(cardinality) => checked_non_negative(
                                    cardinality,
                                    "deletion vector cardinality",
                                )?,
                                None => {
                                    read_deletion_vector(&options, deletion).await?.len() as u64
                                }
                            },
                        };
                        ensure!(
                            deleted_rows <= physical_rows,
                            "Paimon deletion cardinality {deleted_rows} exceeds physical row count {physical_rows} for {}",
                            file.file_name
                        );
                        let record_count = physical_rows - deleted_rows;
                        // Empty data files do not become zero-row column groups.
                        if record_count == 0 {
                            continue;
                        }
                        let path = split.data_file_path(file);
                        let data_format = file_format(&path);
                        result.push(PaimonFileInfo {
                            path,
                            read_path: route.read_path.as_str().to_string(),
                            route_reason: route.reason.as_str().to_string(),
                            file_size: checked_non_negative(file.file_size, "data file size")?,
                            metadata_json: encode_direct_metadata(
                                split,
                                data_format,
                                record_count,
                                deletion,
                            )?,
                        });
                    }
                }
            }
        }
        Ok(result)
    })
}

async fn read_deletion_vector(
    storage_options: &HashMap<String, String>,
    deletion_file: &DeletionFile,
) -> Result<Vec<u64>> {
    let offset = checked_non_negative(deletion_file.offset(), "deletion vector offset")?;
    let length = checked_non_negative(deletion_file.length(), "deletion vector length")?;
    read_deletion_vector_at(
        deletion_file.path(),
        offset,
        length,
        deletion_file.cardinality().unwrap_or(-1),
        storage_options,
    )
    .await
}

async fn read_deletion_vector_at(
    path: &str,
    offset: u64,
    length: u64,
    expected_cardinality: i64,
    storage_options: &HashMap<String, String>,
) -> Result<Vec<u64>> {
    ensure!(
        length >= 4,
        "Paimon deletion vector length is too small: {length}"
    );
    let file_io = FileIO::from_path(path)?
        .with_props(storage_options)
        .build()?;
    let input = file_io.new_input(path)?;
    let file_size = input.metadata().await?.size;
    let total_length = length
        .checked_add(8)
        .ok_or_else(|| anyhow!("Paimon deletion vector range length overflow"))?;
    let requested_end = offset
        .checked_add(total_length)
        .ok_or_else(|| anyhow!("Paimon deletion vector range end overflow"))?;
    let header_end = offset
        .checked_add(8)
        .ok_or_else(|| anyhow!("Paimon deletion vector header range overflow"))?;
    ensure!(
        header_end <= file_size,
        "Paimon deletion vector header range [{offset}, {header_end}) exceeds file size \
         {file_size}: {path}"
    );
    let reader = input.reader().await?;
    let bytes = reader.read(offset..requested_end.min(file_size)).await?;
    ensure!(
        bytes.len() >= 8,
        "Paimon deletion vector short header: expected 8 bytes, got {}",
        bytes.len()
    );

    let declared_length = u32::from_be_bytes(bytes[0..4].try_into()?) as u64;
    let big_endian_magic = u32::from_be_bytes(bytes[4..8].try_into()?);
    if big_endian_magic != DELETION_VECTOR_MAGIC
        && u32::from_le_bytes(bytes[4..8].try_into()?) == DELETION_VECTOR_BITMAP64_MAGIC
    {
        ensure!(
            declared_length.checked_add(8) == Some(length),
            "Paimon bitmap64 deletion vector length mismatch: descriptor {length}, payload \
             {declared_length} plus 8-byte envelope"
        );
        let end = offset
            .checked_add(length)
            .ok_or_else(|| anyhow!("Paimon bitmap64 deletion vector range end overflow"))?;
        ensure!(
            end <= file_size && bytes.len() >= usize::try_from(length)?,
            "Paimon bitmap64 deletion vector range [{offset}, {end}) exceeds file size \
             {file_size}: {path}"
        );
        bail!(
            "{ERROR_NOT_IMPLEMENTED_PREFIX} Paimon bitmap64 deletion vectors \
             (deletion-vectors.bitmap64=true) are not supported yet; rewrite the affected \
             snapshot with deletion-vectors.bitmap64=false: {path}"
        );
    }
    ensure!(
        big_endian_magic == DELETION_VECTOR_MAGIC,
        "invalid Paimon deletion vector magic: expected {DELETION_VECTOR_MAGIC}, got \
         {big_endian_magic}"
    );
    ensure!(
        declared_length == length,
        "Paimon deletion vector length mismatch: descriptor {length}, payload {declared_length}"
    );

    ensure!(
        requested_end <= file_size,
        "Paimon deletion vector range [{offset}, {requested_end}) exceeds file size \
         {file_size}: {path}"
    );
    ensure!(
        bytes.len() == usize::try_from(total_length)?,
        "Paimon deletion vector short read: expected {total_length} bytes, got {}",
        bytes.len()
    );

    let payload_end = usize::try_from(4 + length)?;
    let stored_crc = u32::from_be_bytes(bytes[payload_end..payload_end + 4].try_into()?);
    let mut crc = crc32fast::Hasher::new();
    crc.update(&bytes[4..payload_end]);
    let actual_crc = crc.finalize();
    ensure!(
        stored_crc == actual_crc,
        "Paimon deletion vector CRC mismatch: expected {stored_crc}, got {actual_crc}"
    );
    let mut bitmap_input = Cursor::new(&bytes[8..payload_end]);
    let bitmap = RoaringBitmap::deserialize_from(&mut bitmap_input)
        .context("cannot deserialize Paimon roaring deletion vector")?;
    if expected_cardinality >= 0 {
        ensure!(
            bitmap.len() == u64::try_from(expected_cardinality)?,
            "Paimon deletion vector cardinality mismatch: expected {expected_cardinality}, got {}",
            bitmap.len()
        );
    }
    Ok(bitmap.iter().map(u64::from).collect())
}

pub fn paimon_read_deletion_vector(
    path: &str,
    offset: u64,
    length: u64,
    expected_cardinality: i64,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Vec<u64>> {
    let options = options_from_vecs(storage_options_keys, storage_options_values)?;
    TOKIO_RT.block_on(read_deletion_vector_at(
        path,
        offset,
        length,
        expected_cardinality,
        &options,
    ))
}

type PaimonBatchStream = BoxStream<'static, paimon::Result<RecordBatch>>;

struct PaimonStreamReader {
    stream: PaimonBatchStream,
    schema: SchemaRef,
}

fn classify_stream_error(error: paimon::Error) -> ArrowError {
    let message = error.to_string();
    if matches!(
        &error,
        paimon::Error::IoUnexpected { source, .. } if source.is_temporary()
    ) {
        return ArrowError::IoError(
            message,
            std::io::Error::new(std::io::ErrorKind::Other, error),
        );
    }
    if matches!(
        &error,
        paimon::Error::Unsupported { .. } | paimon::Error::IoUnsupported { .. }
    ) {
        return ArrowError::NotYetImplemented(message);
    }
    ArrowError::InvalidArgumentError(message)
}

impl Iterator for PaimonStreamReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        TOKIO_RT
            .block_on(self.stream.next())
            .map(|result| result.map_err(classify_stream_error))
    }
}

impl RecordBatchReader for PaimonStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Projection-agnostic handle over one decoded DataSplit.
///
/// Opening this handle performs the expensive descriptor work exactly once:
/// payload decode + identity binding, `SchemaManager` schema resolution, and
/// the pinned-snapshot time travel. Streams are then opened per call with a
/// caller-supplied projection, so one cached handle serves every reader
/// instance created from the same metadata (see
/// `PaimonFormatReader::MetaTrait`).
pub struct BlockingPaimonDataSplitReader {
    table: Table,
    split: DataSplit,
    schema: SchemaRef,
}

pub fn paimon_open_data_split_reader(
    metadata_json: &str,
    expected_table_location: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Box<BlockingPaimonDataSplitReader>> {
    let options = options_from_vecs(storage_options_keys, storage_options_values)?;
    let (metadata, split) = decode_split_metadata(metadata_json).map_err(|error| {
        anyhow!("{ERROR_INVALID_PREFIX} invalid Paimon data-split descriptor: {error:#}")
    })?;
    validate_data_split_binding(&metadata, &split, expected_table_location).map_err(|error| {
        anyhow!("{ERROR_INVALID_PREFIX} invalid Paimon data-split binding: {error:#}")
    })?;
    let table_location = metadata
        .table_location
        .as_deref()
        .ok_or_else(|| anyhow!("Paimon data-split descriptor has no table_location"))?;
    let table = TOKIO_RT.block_on(load_table(
        table_location,
        &options,
        Some(metadata.snapshot_id),
    ))?;
    let read = table.new_read_builder().new_read()?;
    let schema = paimon::arrow::build_target_arrow_schema(read.read_type())?;
    Ok(Box::new(BlockingPaimonDataSplitReader {
        table,
        split,
        schema,
    }))
}

impl BlockingPaimonDataSplitReader {
    /// Export the full (unprojected) read schema of the split.
    pub unsafe fn export_schema(&self, out_schema_ptr: *mut u8) -> Result<()> {
        let ffi_schema = FFI_ArrowSchema::try_from(self.schema.as_ref())?;
        unsafe { std::ptr::write(out_schema_ptr.cast::<FFI_ArrowSchema>(), ffi_schema) };
        Ok(())
    }

    /// Open a new merge-read stream with the given projection (empty selects
    /// every column). Only `&self` state is touched, so concurrent streams
    /// from one shared handle are safe.
    pub unsafe fn open_stream(
        &self,
        projected_columns: Vec<String>,
        out_stream_ptr: *mut u8,
    ) -> Result<()> {
        let mut builder = self.table.new_read_builder();
        if !projected_columns.is_empty() {
            let projected_refs: Vec<&str> = projected_columns.iter().map(String::as_str).collect();
            builder.with_projection(&projected_refs)?;
        }
        let read = builder.new_read()?;
        let schema = paimon::arrow::build_target_arrow_schema(read.read_type())?;
        let stream = read.to_arrow(std::slice::from_ref(&self.split))?;
        let reader = PaimonStreamReader { stream, schema };
        let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        unsafe { std::ptr::write(out_stream_ptr.cast::<FFI_ArrowArrayStream>(), ffi_stream) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow58::ffi::FFI_ArrowArray;
    use paimon::spec::{BinaryRow, DataFileMeta};
    use paimon::{DeletionFile, RowRange};

    fn test_file(path: &str) -> DataFileMeta {
        let empty_stats = serde_json::json!({
            "_MIN_VALUES": [],
            "_MAX_VALUES": [],
            "_NULL_COUNTS": []
        });
        serde_json::from_value(serde_json::json!({
            "_FILE_NAME": path,
            "_FILE_SIZE": 128,
            "_ROW_COUNT": 10,
            "_MIN_KEY": [],
            "_MAX_KEY": [],
            "_KEY_STATS": empty_stats.clone(),
            "_VALUE_STATS": empty_stats,
            "_MIN_SEQUENCE_NUMBER": 0,
            "_MAX_SEQUENCE_NUMBER": 0,
            "_SCHEMA_ID": 0,
            "_LEVEL": 0,
            "_EXTRA_FILES": [],
            "_CREATION_TIME": null,
            "_DELETE_ROW_COUNT": null,
            "_EMBEDDED_FILE_INDEX": null
        }))
        .unwrap()
    }

    fn test_split(raw_convertible: bool, path: &str) -> DataSplit {
        DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![test_file(path)])
            .with_raw_convertible(raw_convertible)
            .build()
            .unwrap()
    }

    fn stream_error_code(error: paimon::Error) -> i32 {
        let stream = futures::stream::iter([Err(error)]).boxed();
        let reader = PaimonStreamReader {
            stream,
            schema: std::sync::Arc::new(arrow_schema58::Schema::empty()),
        };
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let mut array = FFI_ArrowArray::empty();
        unsafe { ffi_stream.get_next.unwrap()(&mut ffi_stream, &mut array) }
    }

    #[test]
    fn stream_errors_keep_retryability_across_arrow_ffi() {
        let temporary = opendal58::Error::new(
            opendal58::ErrorKind::RateLimited,
            "temporary object-store failure",
        )
        .set_temporary();
        let temporary = paimon::Error::IoUnexpected {
            message: "stream read failed".to_string(),
            source: Box::new(temporary),
        };
        assert_eq!(stream_error_code(temporary), 5); // EIO

        let permanent = paimon::Error::IoUnexpected {
            message: "stream read failed".to_string(),
            source: Box::new(opendal58::Error::new(
                opendal58::ErrorKind::PermissionDenied,
                "permanent object-store failure",
            )),
        };
        assert_eq!(stream_error_code(permanent), 22); // EINVAL

        let invalid = paimon::Error::DataInvalid {
            message: "corrupt record batch".to_string(),
            source: None,
        };
        assert_eq!(stream_error_code(invalid), 22); // EINVAL
    }

    #[test]
    fn route_matrix_is_explicit() {
        let parquet = test_split(true, "data.parquet");
        let mor = test_split(false, "data.parquet");
        let vortex = test_split(true, "data.vortex");
        let mut deletion_vector_file = test_file("data.parquet");
        // Row tracking alone does not require a data-evolution merge.  The
        // deletion bitmap remains file-local and is safe for direct-file.
        deletion_vector_file.first_row_id = Some(100);
        let with_dv = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![deletion_vector_file])
            .with_data_deletion_files(vec![Some(DeletionFile::new(
                "file:///tmp/table/index/dv".to_string(),
                0,
                12,
                Some(1),
            ))])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let with_range = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![test_file("data.parquet")])
            .with_row_ranges(vec![RowRange::new(1, 3)])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        assert_eq!(with_range.merged_row_count(), Some(10));
        assert_eq!(metadata_split_row_count(&with_range).unwrap(), None);
        let mut evolved_file = test_file("data.parquet");
        evolved_file.write_cols = Some(vec!["id".to_string()]);
        let with_write_cols = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![evolved_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let mut sidecar_file = test_file("data.parquet");
        sidecar_file.extra_files = vec!["vector.idx".to_string()];
        let with_sidecar = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![sidecar_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let mut legacy_level_file = test_file("legacy.parquet");
        legacy_level_file.level = 1;
        legacy_level_file.delete_row_count = None;
        let legacy_level = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![legacy_level_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let dedicated_vector = test_split(true, "data.vector.parquet");
        let mut delete_count_file = test_file("delete-count.parquet");
        delete_count_file.delete_row_count = Some(1);
        let with_delete_count = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![delete_count_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let mut old_schema_file = test_file("old-schema.parquet");
        old_schema_file.schema_id = 1;
        let old_schema = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![old_schema_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let mut tracked = test_file("tracked.parquet");
        tracked.first_row_id = Some(100);
        let mixed_row_tracking = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![tracked.clone(), test_file("untracked.parquet")])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let mut overlapping = test_file("overlapping.parquet");
        overlapping.first_row_id = Some(105);
        let overlapping_row_ids = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![tracked, overlapping])
            .with_raw_convertible(true)
            .build()
            .unwrap();

        let cases = vec![
            (
                "raw append",
                ScanMode::Auto,
                parquet.clone(),
                Some((ReadPath::DirectFile, RouteReason::AutoDirectFile)),
            ),
            (
                "forced data-split",
                ScanMode::DataSplit,
                parquet,
                Some((ReadPath::DataSplit, RouteReason::ForcedDataSplit)),
            ),
            (
                "raw with deletion vector",
                ScanMode::Auto,
                with_dv,
                Some((ReadPath::DirectFile, RouteReason::AutoDirectFile)),
            ),
            (
                "merge-on-read",
                ScanMode::Auto,
                mor.clone(),
                Some((ReadPath::DataSplit, RouteReason::MergeSemantics)),
            ),
            (
                "row range",
                ScanMode::Auto,
                with_range,
                Some((ReadPath::DataSplit, RouteReason::RowRange)),
            ),
            (
                "data evolution",
                ScanMode::Auto,
                with_write_cols,
                Some((ReadPath::DataSplit, RouteReason::DataEvolution)),
            ),
            (
                "sidecar file",
                ScanMode::Auto,
                with_sidecar,
                Some((ReadPath::DataSplit, RouteReason::DataEvolution)),
            ),
            (
                "vortex",
                ScanMode::Auto,
                vortex.clone(),
                Some((ReadPath::DirectFile, RouteReason::AutoDirectFile)),
            ),
            (
                "other non-Parquet",
                ScanMode::Auto,
                test_split(true, "data.orc"),
                Some((ReadPath::DataSplit, RouteReason::UnsupportedFormat)),
            ),
            (
                "legacy compacted file without delete count",
                ScanMode::Auto,
                legacy_level,
                Some((ReadPath::DirectFile, RouteReason::AutoDirectFile)),
            ),
            (
                "dedicated vector file",
                ScanMode::Auto,
                dedicated_vector,
                Some((ReadPath::DataSplit, RouteReason::DedicatedVectorFile)),
            ),
            (
                "data file delete count",
                ScanMode::Auto,
                with_delete_count,
                Some((ReadPath::DataSplit, RouteReason::MergeSemantics)),
            ),
            (
                "schema evolution",
                ScanMode::Auto,
                old_schema,
                Some((ReadPath::DataSplit, RouteReason::SchemaEvolution)),
            ),
            (
                "mixed row tracking",
                ScanMode::Auto,
                mixed_row_tracking,
                Some((ReadPath::DataSplit, RouteReason::DataEvolution)),
            ),
            (
                "overlapping row ids",
                ScanMode::Auto,
                overlapping_row_ids,
                Some((ReadPath::DataSplit, RouteReason::DataEvolution)),
            ),
            (
                "forced direct merge-on-read",
                ScanMode::DirectFile,
                mor,
                None,
            ),
            (
                "forced direct vortex",
                ScanMode::DirectFile,
                vortex,
                Some((ReadPath::DirectFile, RouteReason::ForcedDirectFile)),
            ),
        ];
        for (name, mode, split, expected) in cases {
            let actual = decide_route(mode, &split, 0, TableReadSemantics::default());
            match expected {
                Some((path, reason)) => {
                    let actual = actual.unwrap();
                    assert_eq!((actual.read_path, actual.reason), (path, reason), "{name}");
                }
                None => assert!(actual.is_err(), "{name}"),
            }
        }
    }

    #[test]
    fn oversized_split_warning_fires_only_above_threshold() {
        let split = test_split(false, "data.parquet");
        assert!(oversized_split_warning(&split, "file:///tmp/table").is_none());

        // Two files just above half the threshold each: the advisory must sum
        // the whole split, not look at files individually.
        let mut big_file = test_file("big.parquet");
        big_file.file_size = i64::try_from(OVERSIZED_SPLIT_WARN_BYTES / 2 + 1).unwrap();
        let big = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![big_file.clone(), big_file])
            .with_raw_convertible(false)
            .build()
            .unwrap();
        let warning = oversized_split_warning(&big, "file:///tmp/table")
            .expect("two half-threshold files must trip the advisory");
        assert!(warning.contains("bucket 3"), "{warning}");
        assert!(warning.contains("2 data files"), "{warning}");
        assert!(warning.contains("source.split.target-size"), "{warning}");
    }

    #[test]
    fn table_semantics_keep_direct_file_fail_closed() {
        let raw = test_split(true, "data.parquet");
        let partial_update = TableReadSemantics {
            merge_engine: Some(MergeEngine::PartialUpdate),
            ..Default::default()
        };
        let route = decide_route(ScanMode::Auto, &raw, 0, partial_update).unwrap();
        assert_eq!(
            (route.read_path, route.reason),
            (ReadPath::DataSplit, RouteReason::MergeSemantics)
        );

        let aggregation_mor_dv = TableReadSemantics {
            merge_engine: Some(MergeEngine::Aggregation),
            deletion_vectors_enabled: true,
            deletion_vectors_merge_on_read: true,
            has_blob_fields: false,
        };
        let route = decide_route(ScanMode::Auto, &raw, 0, aggregation_mor_dv).unwrap();
        assert_eq!(
            (route.read_path, route.reason),
            (ReadPath::DataSplit, RouteReason::MergeSemantics)
        );

        let mut compacted_file = test_file("compacted.parquet");
        compacted_file.level = 1;
        compacted_file.delete_row_count = Some(0);
        let compacted = DataSplit::builder()
            .with_snapshot(7)
            .with_partition(BinaryRow::new(0))
            .with_bucket(3)
            .with_bucket_path("file:///tmp/table/bucket-0".to_string())
            .with_total_buckets(4)
            .with_data_files(vec![compacted_file])
            .with_raw_convertible(true)
            .build()
            .unwrap();
        let materialized_dv = TableReadSemantics {
            merge_engine: Some(MergeEngine::Aggregation),
            deletion_vectors_enabled: true,
            deletion_vectors_merge_on_read: false,
            has_blob_fields: false,
        };
        let route = decide_route(ScanMode::Auto, &compacted, 0, materialized_dv).unwrap();
        assert_eq!(
            (route.read_path, route.reason),
            (ReadPath::DirectFile, RouteReason::AutoDirectFile)
        );

        let blob_table = TableReadSemantics {
            has_blob_fields: true,
            ..Default::default()
        };
        let route = decide_route(ScanMode::Auto, &raw, 0, blob_table).unwrap();
        assert_eq!(
            (route.read_path, route.reason),
            (ReadPath::DataSplit, RouteReason::DataEvolution)
        );
    }

    #[test]
    fn descriptor_round_trip_and_corruption_detection() {
        let split = test_split(false, "data.parquet");
        let encoded = encode_split_metadata("file:///tmp/table", &split, 10).unwrap();
        let (metadata, decoded) = decode_split_metadata(&encoded).unwrap();
        assert_eq!(metadata.version, 1);
        assert_eq!(decoded.snapshot_id(), 7);
        validate_data_split_binding(&metadata, &decoded, "file:///tmp/table").unwrap();

        let mut parsed: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(parsed["codec"], "paimon-split");
        assert!(parsed.get("codec_version").is_none());
        assert!(parsed.get("producer_revision").is_none());
        assert!(parsed.get("descriptor_identity").is_none());
        parsed["payload_sha256"] = serde_json::Value::String("bad".to_string());
        assert!(decode_split_metadata(&parsed.to_string()).is_err());

        let mut unsupported: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        unsupported["codec"] = serde_json::Value::String("paimon-unknown".to_string());
        let error = decode_split_metadata(&unsupported.to_string()).unwrap_err();
        assert!(error.to_string().contains("unsupported Paimon"));

        let mut malformed: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        malformed["payload"] = serde_json::Value::String("not-base64".to_string());
        assert!(decode_split_metadata(&malformed.to_string()).is_err());

        let mut wrong_version: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        wrong_version["version"] = serde_json::Value::from(2);
        assert!(decode_split_metadata(&wrong_version.to_string()).is_err());
    }

    #[test]
    fn descriptor_binding_rejects_outer_payload_mismatches() {
        let split = test_split(false, "data.parquet");
        let encoded = encode_split_metadata("file:///tmp/table", &split, 10).unwrap();
        let (metadata, decoded) = decode_split_metadata(&encoded).unwrap();

        for (name, altered, expected) in [
            (
                "table",
                PaimonMetadata {
                    table_location: Some("file:///tmp/other".to_string()),
                    ..metadata.clone()
                },
                "table location",
            ),
            (
                "snapshot",
                PaimonMetadata {
                    snapshot_id: metadata.snapshot_id + 1,
                    ..metadata.clone()
                },
                "snapshot",
            ),
            (
                "bucket",
                PaimonMetadata {
                    bucket: metadata.bucket + 1,
                    ..metadata.clone()
                },
                "bucket",
            ),
        ] {
            let error = validate_data_split_binding(&altered, &decoded, "file:///tmp/table")
                .expect_err(name);
            assert!(error.to_string().contains(expected), "{name}: {error}");
        }

        let error = validate_data_split_binding(&metadata, &decoded, "file:///tmp/manifest-table")
            .unwrap_err();
        assert!(error.to_string().contains("manifest table location"));
    }

    #[test]
    fn bitmap64_deletion_vector_is_rejected_with_actionable_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dv64-index");
        let bytes = include_bytes!("../testdata/paimon-java-bitmap64.dv");
        let declared_length = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as u64;
        assert_eq!(declared_length + 8, bytes.len() as u64);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            DELETION_VECTOR_BITMAP64_MAGIC
        );
        std::fs::write(&path, bytes).unwrap();

        let error = TOKIO_RT
            .block_on(read_deletion_vector_at(
                path.to_str().unwrap(),
                0,
                bytes.len() as u64,
                -1,
                &HashMap::new(),
            ))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("bitmap64"), "{message}");
        assert!(message.contains("are not supported yet"), "{message}");
        assert!(
            message.contains("rewrite the affected snapshot"),
            "{message}"
        );
        assert!(!message.contains("data-split"), "{message}");
    }

    #[test]
    fn missing_pinned_snapshot_reports_bounds_and_refresh_advice() {
        let directory = tempfile::tempdir().unwrap();
        let table_location = directory.path().join("snap-table");
        let table_location = table_location.to_str().unwrap();
        let info = crate::paimon_testutil::paimon_create_test_table(
            table_location,
            10,
            "append",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "parquet",
            0,
        )
        .unwrap();

        // A pinned snapshot that exists must load.
        TOKIO_RT
            .block_on(load_table(
                table_location,
                &HashMap::new(),
                Some(info.snapshot_id),
            ))
            .unwrap();

        // A confirmed-missing snapshot is a terminal input-state error with
        // the live bounds and the refresh advice (the C++ boundary keys the
        // Invalid classification off that marker).
        let error = TOKIO_RT
            .block_on(load_table(
                table_location,
                &HashMap::new(),
                Some(info.snapshot_id + 1000),
            ))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("no longer exists"), "{message}");
        assert!(message.contains("earliest=1"), "{message}");
        assert!(
            message.contains(&format!("latest={}", info.snapshot_id)),
            "{message}"
        );
        assert!(
            message.contains("refresh the external collection"),
            "{message}"
        );
    }

    #[test]
    fn corrupt_snapshot_content_does_not_claim_nonexistence_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let table_location = directory.path().join("snap-table");
        let table_location = table_location.to_str().unwrap();
        let info = crate::paimon_testutil::paimon_create_test_table(
            table_location,
            10,
            "append",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "parquet",
            0,
        )
        .unwrap();
        // Corrupt the pinned snapshot file: the existence probe passes, time
        // travel fails to resolve, and the error is the terminal fallback
        // message without fabricated earliest/latest bounds.
        let snapshot_file = format!("{table_location}/snapshot/snapshot-{}", info.snapshot_id);
        std::fs::write(&snapshot_file, b"{not json").unwrap();
        let error = TOKIO_RT
            .block_on(load_table(
                table_location,
                &HashMap::new(),
                Some(info.snapshot_id),
            ))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("refresh the external collection"),
            "{message}"
        );
        assert!(!message.contains("earliest="), "{message}");
    }

    #[test]
    fn deletion_vector_exact_eof_and_crc_are_validated() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dv-index");
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(9);
        let mut bitmap_bytes = Vec::new();
        bitmap.serialize_into(&mut bitmap_bytes).unwrap();
        let length = 4u64 + bitmap_bytes.len() as u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(length as u32).to_be_bytes());
        bytes.extend_from_slice(&DELETION_VECTOR_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&bitmap_bytes);
        let mut crc = crc32fast::Hasher::new();
        crc.update(&bytes[4..]);
        bytes.extend_from_slice(&crc.finalize().to_be_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let positions = TOKIO_RT
            .block_on(read_deletion_vector_at(
                path.to_str().unwrap(),
                0,
                length,
                2,
                &HashMap::new(),
            ))
            .unwrap();
        assert_eq!(positions, vec![1, 9]);

        assert!(
            TOKIO_RT
                .block_on(read_deletion_vector_at(
                    path.to_str().unwrap(),
                    0,
                    length,
                    3,
                    &HashMap::new(),
                ))
                .is_err()
        );

        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(
            TOKIO_RT
                .block_on(read_deletion_vector_at(
                    path.to_str().unwrap(),
                    0,
                    length,
                    2,
                    &HashMap::new(),
                ))
                .is_err()
        );

        std::fs::write(&path, &bytes).unwrap();

        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(
            TOKIO_RT
                .block_on(read_deletion_vector_at(
                    path.to_str().unwrap(),
                    0,
                    length,
                    2,
                    &HashMap::new(),
                ))
                .is_err()
        );
    }
}
