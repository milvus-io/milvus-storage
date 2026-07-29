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

//! Paimon direct-file planning and deletion-vector bridge.

use anyhow::{Context, Result, anyhow, bail, ensure};
use paimon::catalog::Identifier;
use paimon::io::{FileIO, FileRead};
use paimon::spec::MergeEngine;
use paimon::table::{SchemaManager, SnapshotManager};
use paimon::{DataSplit, DeletionFile, Table};
use roaring::RoaringBitmap;
use serde::Serialize;
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

// These stable markers are consumed by ClassifyPaimonError at the C++ boundary.
// Keep classification at the point where an error is known to be terminal;
// unclassified storage and network errors retain retryable IOError semantics.
fn invalid_message(message: impl std::fmt::Display) -> String {
    format!("{ERROR_INVALID_PREFIX} {message}")
}

fn not_implemented_message(message: impl std::fmt::Display) -> String {
    format!("{ERROR_NOT_IMPLEMENTED_PREFIX} {message}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    Auto,
    DirectFile,
}

impl ScanMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "direct-file" => Ok(Self::DirectFile),
            "data-split" => bail!(not_implemented_message(
                "Paimon data-split reads are not supported"
            )),
            other => bail!(invalid_message(format_args!(
                "invalid Paimon scan mode '{other}'; expected auto or direct-file"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteReason {
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
            Self::MergeSemantics => "merge-semantics",
            Self::RowRange => "row-range",
            Self::SchemaEvolution => "schema-evolution",
            Self::DataEvolution => "data-evolution",
            Self::DedicatedVectorFile => "dedicated-vector-file",
            Self::UnsupportedFormat => "unsupported-format",
        }
    }
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
                .transpose()
                .with_context(|| invalid_message("invalid Paimon merge-engine table option"))?,
            deletion_vectors_enabled: options.deletion_vectors_enabled(),
            deletion_vectors_merge_on_read: options.deletion_vectors_merge_on_read(),
            has_blob_fields: !options.blob_fields().is_empty(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct PaimonMetadata {
    version: u32,
    read_path: String,
    data_format: String,
    record_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deletion_file: Option<DeletionFileDescriptor>,
}

#[derive(Debug, Clone, Serialize)]
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
        invalid_message(format_args!(
            "storage option key/value count mismatch: {} keys, {} values",
            keys.len(),
            values.len()
        ))
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
        .with_context(|| {
            invalid_message(format_args!(
                "cannot address Paimon snapshot file '{snapshot_path}'"
            ))
        })?;
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
    bail!(invalid_message(format_args!(
        "Paimon snapshot {snapshot_id} no longer exists for table {table_location} \
         (earliest={}, latest={}); refresh the external collection",
        bound(earliest),
        bound(latest)
    )))
}

async fn load_table(
    table_location: &str,
    storage_options: &HashMap<String, String>,
    snapshot_id: Option<i64>,
) -> Result<Table> {
    let file_io = FileIO::from_path(table_location)
        .with_context(|| {
            invalid_message(format_args!(
                "cannot infer Paimon storage from '{table_location}'"
            ))
        })?
        .with_props(storage_options)
        .build()
        .with_context(|| {
            invalid_message(format_args!(
                "cannot build Paimon FileIO for '{table_location}'"
            ))
        })?;
    let schema = SchemaManager::new(file_io.clone(), table_location.to_string())
        .latest()
        .await?
        .ok_or_else(|| {
            anyhow!(invalid_message(format_args!(
                "Paimon table has no schema: {table_location}"
            )))
        })?;
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
            invalid_message(format_args!(
                "Paimon snapshot {snapshot_id} no longer exists for table {table_location}; \
                 refresh the external collection"
            ))
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
) -> Result<()> {
    match mode {
        ScanMode::Auto => match direct_file_ineligibility(split, table_schema_id, table) {
            None => Ok(()),
            Some((reason, detail)) => bail!(not_implemented_message(format_args!(
                "Paimon split requires data-split reading ({}): {detail}",
                reason.as_str()
            ))),
        },
        ScanMode::DirectFile => match direct_file_ineligibility(split, table_schema_id, table) {
            None => Ok(()),
            Some((_, detail)) => bail!(not_implemented_message(format_args!(
                "Paimon split cannot use direct-file: {detail}"
            ))),
        },
    }
}

fn checked_non_negative(value: i64, name: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| {
        invalid_message(format_args!("Paimon {name} is negative: {value}"))
    })
}

fn deletion_descriptor(file: &DeletionFile) -> Result<DeletionFileDescriptor> {
    ensure!(
        !file.path().is_empty(),
        invalid_message("Paimon deletion vector path is empty")
    );
    Ok(DeletionFileDescriptor {
        path: file.path().to_string(),
        offset: checked_non_negative(file.offset(), "deletion vector offset")?,
        length: {
            let length = checked_non_negative(file.length(), "deletion vector length")?;
            ensure!(
                length > 0,
                invalid_message("Paimon deletion vector length is zero")
            );
            length
        },
        cardinality: file.cardinality().unwrap_or(-1),
    })
}

fn encode_direct_metadata(
    data_format: &str,
    record_count: u64,
    deletion_file: Option<&DeletionFile>,
) -> Result<String> {
    serde_json::to_string(&PaimonMetadata {
        version: METADATA_VERSION,
        read_path: "direct-file".to_string(),
        data_format: data_format.to_string(),
        record_count,
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
            decide_route(mode, split, table.schema().id(), table_read_semantics)?;
            for (index, file) in split.data_files().iter().enumerate() {
                let deletion = split.deletion_file_for_data_file_index(index);
                let physical_rows = checked_non_negative(file.row_count, "data file row count")?;
                let deleted_rows = match deletion {
                    None => 0,
                    Some(deletion) => match deletion.cardinality() {
                        Some(cardinality) => {
                            checked_non_negative(cardinality, "deletion vector cardinality")?
                        }
                        None => read_deletion_vector(&options, deletion).await?.len() as u64,
                    },
                };
                ensure!(
                    deleted_rows <= physical_rows,
                    invalid_message(format_args!(
                        "Paimon deletion cardinality {deleted_rows} exceeds physical row count \
                         {physical_rows} for {}",
                        file.file_name
                    ))
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
                    file_size: checked_non_negative(file.file_size, "data file size")?,
                    metadata_json: encode_direct_metadata(
                        data_format,
                        record_count,
                        deletion,
                    )?,
                });
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
        expected_cardinality >= -1,
        invalid_message(format_args!(
            "Paimon deletion vector cardinality is invalid: {expected_cardinality}"
        ))
    );
    ensure!(
        length >= 4,
        invalid_message(format_args!(
            "Paimon deletion vector length is too small: {length}"
        ))
    );
    let file_io = FileIO::from_path(path)
        .with_context(|| {
            invalid_message(format_args!(
                "cannot infer Paimon deletion vector storage from '{path}'"
            ))
        })?
        .with_props(storage_options)
        .build()
        .with_context(|| {
            invalid_message(format_args!(
                "cannot build Paimon deletion vector FileIO for '{path}'"
            ))
        })?;
    let input = file_io.new_input(path).with_context(|| {
        invalid_message(format_args!(
            "cannot address Paimon deletion vector file '{path}'"
        ))
    })?;
    let file_size = input.metadata().await?.size;
    let total_length = length
        .checked_add(8)
        .ok_or_else(|| {
            anyhow!(invalid_message(
                "Paimon deletion vector range length overflow"
            ))
        })?;
    let requested_end = offset
        .checked_add(total_length)
        .ok_or_else(|| {
            anyhow!(invalid_message(
                "Paimon deletion vector range end overflow"
            ))
        })?;
    let header_end = offset
        .checked_add(8)
        .ok_or_else(|| {
            anyhow!(invalid_message(
                "Paimon deletion vector header range overflow"
            ))
        })?;
    ensure!(
        header_end <= file_size,
        invalid_message(format_args!(
            "Paimon deletion vector header range [{offset}, {header_end}) exceeds file size \
             {file_size}: {path}"
        ))
    );
    let reader = input.reader().await?;
    let bytes = reader.read(offset..requested_end.min(file_size)).await?;
    ensure!(
        bytes.len() >= 8,
        invalid_message(format_args!(
            "Paimon deletion vector short header: expected 8 bytes, got {}",
            bytes.len()
        ))
    );

    let declared_length = u32::from_be_bytes(bytes[0..4].try_into()?) as u64;
    let big_endian_magic = u32::from_be_bytes(bytes[4..8].try_into()?);
    // Java bitmap64 uses a little-endian magic, and DeletionFile.length
    // covers the complete serialized value rather than bitmap32's payload
    // length. Recognize that envelope before applying bitmap32 validation.
    if big_endian_magic != DELETION_VECTOR_MAGIC
        && u32::from_le_bytes(bytes[4..8].try_into()?) == DELETION_VECTOR_BITMAP64_MAGIC
    {
        ensure!(
            declared_length.checked_add(8) == Some(length),
            invalid_message(format_args!(
                "Paimon bitmap64 deletion vector length mismatch: descriptor {length}, payload \
                 {declared_length} plus 8-byte envelope"
            ))
        );
        let end = offset
            .checked_add(length)
            .ok_or_else(|| {
                anyhow!(invalid_message(
                    "Paimon bitmap64 deletion vector range end overflow"
                ))
            })?;
        let bitmap64_length = usize::try_from(length).with_context(|| {
            invalid_message("Paimon bitmap64 deletion vector length exceeds addressable memory")
        })?;
        ensure!(
            end <= file_size && bytes.len() >= bitmap64_length,
            invalid_message(format_args!(
                "Paimon bitmap64 deletion vector range [{offset}, {end}) exceeds file size \
                 {file_size}: {path}"
            ))
        );
        bail!(not_implemented_message(format_args!(
            "Paimon bitmap64 deletion vectors (deletion-vectors.bitmap64=true) are not \
             supported yet; rewrite the affected snapshot with \
             deletion-vectors.bitmap64=false: {path}"
        )));
    }
    ensure!(
        big_endian_magic == DELETION_VECTOR_MAGIC,
        invalid_message(format_args!(
            "invalid Paimon deletion vector magic: expected {DELETION_VECTOR_MAGIC}, got \
             {big_endian_magic}"
        ))
    );
    ensure!(
        declared_length == length,
        invalid_message(format_args!(
            "Paimon deletion vector length mismatch: descriptor {length}, payload \
             {declared_length}"
        ))
    );

    ensure!(
        requested_end <= file_size,
        invalid_message(format_args!(
            "Paimon deletion vector range [{offset}, {requested_end}) exceeds file size \
             {file_size}: {path}"
        ))
    );
    let expected_total_length = usize::try_from(total_length).with_context(|| {
        invalid_message("Paimon deletion vector range length exceeds addressable memory")
    })?;
    ensure!(
        bytes.len() == expected_total_length,
        invalid_message(format_args!(
            "Paimon deletion vector short read: expected {total_length} bytes, got {}",
            bytes.len()
        ))
    );

    let payload_end = usize::try_from(4 + length).with_context(|| {
        invalid_message("Paimon deletion vector payload length exceeds addressable memory")
    })?;
    let stored_crc = u32::from_be_bytes(bytes[payload_end..payload_end + 4].try_into()?);
    let mut crc = crc32fast::Hasher::new();
    crc.update(&bytes[4..payload_end]);
    let actual_crc = crc.finalize();
    ensure!(
        stored_crc == actual_crc,
        invalid_message(format_args!(
            "Paimon deletion vector CRC mismatch: expected {stored_crc}, got {actual_crc}"
        ))
    );
    let mut bitmap_input = Cursor::new(&bytes[8..payload_end]);
    let bitmap = RoaringBitmap::deserialize_from(&mut bitmap_input)
        .with_context(|| invalid_message("cannot deserialize Paimon roaring deletion vector"))?;
    if expected_cardinality >= 0 {
        let expected_cardinality = u64::try_from(expected_cardinality).with_context(|| {
            invalid_message("Paimon deletion vector cardinality exceeds the supported range")
        })?;
        ensure!(
            bitmap.len() == expected_cardinality,
            invalid_message(format_args!(
                "Paimon deletion vector cardinality mismatch: expected {expected_cardinality}, \
                 got {}",
                bitmap.len()
            ))
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn scan_mode_errors_are_classified() {
        let invalid = ScanMode::parse("invalid-mode").unwrap_err();
        assert!(
            invalid.to_string().contains(ERROR_INVALID_PREFIX),
            "{invalid}"
        );

        let unsupported = ScanMode::parse("data-split").unwrap_err();
        assert!(
            unsupported
                .to_string()
                .contains(ERROR_NOT_IMPLEMENTED_PREFIX),
            "{unsupported}"
        );
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
            ("raw append", ScanMode::Auto, parquet, true),
            ("raw with deletion vector", ScanMode::Auto, with_dv, true),
            ("merge-on-read", ScanMode::Auto, mor.clone(), false),
            ("row range", ScanMode::Auto, with_range, false),
            ("data evolution", ScanMode::Auto, with_write_cols, false),
            ("sidecar file", ScanMode::Auto, with_sidecar, false),
            ("vortex", ScanMode::Auto, vortex.clone(), true),
            (
                "other non-Parquet",
                ScanMode::Auto,
                test_split(true, "data.orc"),
                false,
            ),
            (
                "legacy compacted file without delete count",
                ScanMode::Auto,
                legacy_level,
                true,
            ),
            (
                "dedicated vector file",
                ScanMode::Auto,
                dedicated_vector,
                false,
            ),
            (
                "data file delete count",
                ScanMode::Auto,
                with_delete_count,
                false,
            ),
            ("schema evolution", ScanMode::Auto, old_schema, false),
            (
                "mixed row tracking",
                ScanMode::Auto,
                mixed_row_tracking,
                false,
            ),
            (
                "overlapping row ids",
                ScanMode::Auto,
                overlapping_row_ids,
                false,
            ),
            (
                "forced direct merge-on-read",
                ScanMode::DirectFile,
                mor,
                false,
            ),
            ("forced direct vortex", ScanMode::DirectFile, vortex, true),
        ];
        for (name, mode, split, supported) in cases {
            assert_eq!(
                decide_route(mode, &split, 0, TableReadSemantics::default()).is_ok(),
                supported,
                "{name}"
            );
        }
    }

    #[test]
    fn table_semantics_keep_direct_file_fail_closed() {
        let raw = test_split(true, "data.parquet");
        let partial_update = TableReadSemantics {
            merge_engine: Some(MergeEngine::PartialUpdate),
            ..Default::default()
        };
        assert!(decide_route(ScanMode::Auto, &raw, 0, partial_update).is_err());

        let aggregation_mor_dv = TableReadSemantics {
            merge_engine: Some(MergeEngine::Aggregation),
            deletion_vectors_enabled: true,
            deletion_vectors_merge_on_read: true,
            has_blob_fields: false,
        };
        assert!(decide_route(ScanMode::Auto, &raw, 0, aggregation_mor_dv).is_err());

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
        assert!(decide_route(ScanMode::Auto, &compacted, 0, materialized_dv).is_ok());

        let blob_table = TableReadSemantics {
            has_blob_fields: true,
            ..Default::default()
        };
        assert!(decide_route(ScanMode::Auto, &raw, 0, blob_table).is_err());
    }

    #[test]
    fn missing_pinned_snapshot_reports_bounds_and_refresh_advice() {
        let directory = tempfile::tempdir().unwrap();
        let table_location = directory.path().join("snap-table");
        let table_location = table_location.to_str().unwrap();
        let snapshot_id = crate::paimon_testutil::paimon_create_test_table(
            table_location,
            10,
            "append",
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
                Some(snapshot_id),
            ))
            .unwrap();

        // A confirmed-missing snapshot is a terminal input-state error with
        // the live bounds and the refresh advice (the C++ boundary keys the
        // Invalid classification off that marker).
        let error = TOKIO_RT
            .block_on(load_table(
                table_location,
                &HashMap::new(),
                Some(snapshot_id + 1000),
            ))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("no longer exists"), "{message}");
        assert!(message.contains("earliest=1"), "{message}");
        assert!(
            message.contains(&format!("latest={snapshot_id}")),
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
        let snapshot_id = crate::paimon_testutil::paimon_create_test_table(
            table_location,
            10,
            "append",
            Vec::new(),
            "parquet",
            0,
        )
        .unwrap();
        // Corrupt the pinned snapshot file: the existence probe passes, time
        // travel fails to resolve, and the error is the terminal fallback
        // message without fabricated earliest/latest bounds.
        let snapshot_file = format!("{table_location}/snapshot/snapshot-{snapshot_id}");
        std::fs::write(&snapshot_file, b"{not json").unwrap();
        let error = TOKIO_RT
            .block_on(load_table(
                table_location,
                &HashMap::new(),
                Some(snapshot_id),
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

        let cardinality_error = TOKIO_RT
            .block_on(read_deletion_vector_at(
                path.to_str().unwrap(),
                0,
                length,
                3,
                &HashMap::new(),
            ))
            .unwrap_err();
        assert!(
            cardinality_error.to_string().contains(ERROR_INVALID_PREFIX),
            "{cardinality_error}"
        );

        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        let short_read_error = TOKIO_RT
            .block_on(read_deletion_vector_at(
                path.to_str().unwrap(),
                0,
                length,
                2,
                &HashMap::new(),
            ))
            .unwrap_err();
        assert!(
            short_read_error.to_string().contains(ERROR_INVALID_PREFIX),
            "{short_read_error}"
        );

        std::fs::write(&path, &bytes).unwrap();

        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        let crc_error = TOKIO_RT
            .block_on(read_deletion_vector_at(
                path.to_str().unwrap(),
                0,
                length,
                2,
                &HashMap::new(),
            ))
            .unwrap_err();
        assert!(
            crc_error.to_string().contains(ERROR_INVALID_PREFIX),
            "{crc_error}"
        );
    }
}
