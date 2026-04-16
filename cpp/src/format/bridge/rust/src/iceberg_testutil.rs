// Copyright 2025 Zilliz
//
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

//! Test utility for creating Iceberg tables.
//! Supports both local filesystem and cloud storage (S3, GCS, Azure).
//! Used by C++ integration tests and the loon CLI tool via CXX bridge.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use iceberg::io::{FileIO, FileIOBuilder};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestListWriter,
    ManifestWriterBuilder, NestedField, Operation, PartitionSpec, PrimitiveType, Schema, Snapshot,
    SortOrder, Summary, TableMetadataBuilder, Type, UnboundPartitionSpec,
};

use crate::iceberg_bridgeimpl::{denormalize_uri, normalize_uri, vec_to_hashmap};
use crate::iceberg_test_ffi::IcebergTestTableInfo;
use crate::TOKIO_RT;

/// Write a Parquet record batch to bytes in memory.
fn write_parquet_to_bytes(
    batch: &RecordBatch,
    schema: Arc<ArrowSchema>,
) -> Result<Vec<u8>, anyhow::Error> {
    let mut buf = Vec::new();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(buf)
}

/// Create an Iceberg table with test data.
///
/// Supports local filesystem (table_dir is a local path) and cloud storage
/// (table_dir is a URI like s3://bucket/path). When storage_options are
/// provided, they are passed to the FileIO builder for cloud authentication.
///
/// The table has schema: id (int64), name (string), value (float64)
/// with `num_rows` rows where id=0..N-1, name="row_0".."row_{N-1}", value=0.0, 1.5, 3.0, ...
///
/// Optionally writes positional delete files for specific row positions.
pub fn iceberg_create_test_table(
    table_dir: &str,
    num_rows: u64,
    with_positional_deletes: bool,
    deleted_positions: Vec<i64>,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    // Empty string disables the swap (cxx can't express `Option` across the
    // FFI boundary directly, so the empty-string convention stands in for
    // `None` here). When non-empty, after all metadata is written this
    // function byte-rewrites every embedded URI from `<write_scheme>://` to
    // `<record_scheme_override>://` across the metadata tree. See
    // `rewrite_iceberg_scheme` for the full motivation.
    record_scheme_override: &str,
) -> Result<IcebergTestTableInfo, anyhow::Error> {
    // Lift the FFI empty-string sentinel into an `Option` immediately so the
    // rest of the body reads naturally.
    let record_scheme_override: Option<&str> =
        (!record_scheme_override.is_empty()).then_some(record_scheme_override);

    TOKIO_RT.block_on(async {
        let props = vec_to_hashmap(storage_options_keys, storage_options_values);

        // Normalize URI for opendal and detect FileIO scheme in one pass.
        let (resolved_dir, scheme) = normalize_uri(table_dir, &props);
        let data_dir = format!("{}/data", resolved_dir);
        let metadata_dir = format!("{}/metadata", resolved_dir);

        let is_local = scheme == "file";

        // Build FileIO with storage options
        let mut file_io_builder = FileIOBuilder::new(&scheme);
        for (k, v) in &props {
            file_io_builder = file_io_builder.with_prop(k, v);
        }
        let file_io = file_io_builder.build()?;

        // Create directories for local filesystem only (S3 has no directories)
        if is_local {
            std::fs::create_dir_all(&data_dir)?;
            std::fs::create_dir_all(&metadata_dir)?;
        }

        // 1. Create Arrow schema and data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids: Vec<i64> = (0..num_rows as i64).collect();
        let names: Vec<String> = (0..num_rows).map(|i| format!("row_{}", i)).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let values: Vec<f64> = (0..num_rows).map(|i| i as f64 * 1.5).collect();

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(name_refs)),
                Arc::new(Float64Array::from(values)),
            ],
        )?;

        // 2. Write data file as Parquet via FileIO
        let data_file_path = format!("{}/00000-0-data.parquet", data_dir);
        let data_bytes = write_parquet_to_bytes(&batch, arrow_schema.clone())?;
        let data_file_size = data_bytes.len() as u64;
        let output = file_io.new_output(&data_file_path)?;
        output.write(Bytes::from(data_bytes)).await?;

        let data_file_uri = if is_local {
            std::fs::canonicalize(&data_file_path)?
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path"))?
                .to_string()
        } else {
            data_file_path.clone()
        };

        // 3. Optionally write positional delete file
        let mut delete_file_uri = String::new();
        let mut delete_file_size: u64 = 0;
        if with_positional_deletes && !deleted_positions.is_empty() {
            let delete_schema = Arc::new(ArrowSchema::new(vec![
                Field::new("file_path", DataType::Utf8, false),
                Field::new("pos", DataType::Int64, false),
            ]));

            let paths: Vec<&str> = deleted_positions
                .iter()
                .map(|_| data_file_uri.as_str())
                .collect();
            let delete_batch = RecordBatch::try_new(
                delete_schema.clone(),
                vec![
                    Arc::new(StringArray::from(paths)),
                    Arc::new(Int64Array::from(deleted_positions.clone())),
                ],
            )?;

            let delete_path = format!("{}/00000-0-pos-delete.parquet", data_dir);
            let delete_bytes = write_parquet_to_bytes(&delete_batch, delete_schema)?;
            delete_file_size = delete_bytes.len() as u64;
            let output = file_io.new_output(&delete_path)?;
            output.write(Bytes::from(delete_bytes)).await?;

            delete_file_uri = if is_local {
                std::fs::canonicalize(&delete_path)?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path"))?
                    .to_string()
            } else {
                delete_path
            };
        }

        // 4. Build Iceberg schema
        let iceberg_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "value", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()?;
        let schema_ref = Arc::new(iceberg_schema.clone());

        let partition_spec = PartitionSpec::unpartition_spec();
        let sort_order = SortOrder::unsorted_order();

        // 5. Write data manifest
        let manifest_path = format!("{}/manifest-data-0.avro", metadata_dir);
        let manifest_output = file_io.new_output(&manifest_path)?;

        let snapshot_id: i64 = 1;
        let sequence_number: i64 = 1;

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(data_file_uri.clone())
            .file_format(DataFileFormat::Parquet)
            .record_count(num_rows)
            .file_size_in_bytes(data_file_size)
            .partition_spec_id(0)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build DataFile: {}", e))?;

        let mut manifest_writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(snapshot_id),
            None,
            schema_ref.clone(),
            partition_spec.clone(),
        )
        .build_v2_data();

        manifest_writer.add_file(data_file, sequence_number)?;
        let data_manifest_file = manifest_writer.write_manifest_file().await?;

        // 6. Optionally write delete manifest
        let mut all_manifest_files = vec![data_manifest_file];
        if with_positional_deletes && !delete_file_uri.is_empty() {
            let delete_manifest_path = format!("{}/manifest-deletes-0.avro", metadata_dir);
            let delete_manifest_output = file_io.new_output(&delete_manifest_path)?;

            let delete_data_file = DataFileBuilder::default()
                .content(DataContentType::PositionDeletes)
                .file_path(delete_file_uri)
                .file_format(DataFileFormat::Parquet)
                .record_count(deleted_positions.len() as u64)
                .file_size_in_bytes(delete_file_size)
                .partition_spec_id(0)
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build delete DataFile: {}", e))?;

            let mut delete_manifest_writer = ManifestWriterBuilder::new(
                delete_manifest_output,
                Some(snapshot_id),
                None,
                schema_ref.clone(),
                partition_spec.clone(),
            )
            .build_v2_deletes();

            delete_manifest_writer.add_file(delete_data_file, sequence_number)?;
            let delete_manifest_file = delete_manifest_writer.write_manifest_file().await?;
            all_manifest_files.push(delete_manifest_file);
        }

        // 7. Write manifest list
        let manifest_list_path =
            format!("{}/snap-{}-manifest-list.avro", metadata_dir, snapshot_id);
        let manifest_list_output = file_io.new_output(&manifest_list_path)?;

        let mut manifest_list_writer =
            ManifestListWriter::v2(manifest_list_output, snapshot_id, None, sequence_number);
        manifest_list_writer.add_manifests(all_manifest_files.into_iter())?;
        manifest_list_writer.close().await?;

        // 8. Build table metadata and serialize
        let snapshot = Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_sequence_number(sequence_number)
            .with_timestamp_ms(chrono::Utc::now().timestamp_millis())
            .with_manifest_list(manifest_list_path.clone())
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .with_schema_id(0)
            .build();

        let table_location = if is_local {
            std::fs::canonicalize(&resolved_dir)?
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path"))?
                .to_string()
        } else {
            resolved_dir.clone()
        };
        let builder = TableMetadataBuilder::new(
            iceberg_schema,
            UnboundPartitionSpec::builder().build(),
            sort_order,
            table_location,
            FormatVersion::V2,
            HashMap::new(),
        )?;

        let builder = builder
            .add_snapshot(snapshot)?
            .set_ref(
                "main",
                iceberg::spec::SnapshotReference {
                    snapshot_id,
                    retention: iceberg::spec::SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            )?;

        let build_result = builder.build()?;
        let metadata = build_result.metadata;

        // 9. Write metadata JSON via FileIO
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        let metadata_file_path = format!("{}/v1.metadata.json", metadata_dir);
        let output = file_io.new_output(&metadata_file_path)?;
        output.write(Bytes::from(metadata_json)).await?;

        // 10. Optional cross-scheme rewrite. Separate function so the
        //     motivation (cross-tenant GCP tests that must write with HMAC
        //     over S3-compat yet read via native `gs://`) is documented in
        //     one place rather than scattered at the call site.
        let (final_metadata_path, final_data_file_uri) = match record_scheme_override {
            Some(to_scheme) if to_scheme != scheme => {
                rewrite_iceberg_scheme(
                    &file_io,
                    &metadata_dir,
                    snapshot_id,
                    with_positional_deletes,
                    &scheme,
                    to_scheme,
                )
                .await?;
                let from = format!("{}://", scheme);
                let to = format!("{}://", to_scheme);
                (
                    metadata_file_path.replacen(&from, &to, 1),
                    data_file_uri.replacen(&from, &to, 1),
                )
            }
            _ => (metadata_file_path, data_file_uri),
        };

        // Denormalize returned paths: strip Azure container@endpoint back to
        // scheme://container/path so C++ sees a uniform format across providers.
        Ok(IcebergTestTableInfo {
            metadata_location: denormalize_uri(&final_metadata_path),
            snapshot_id,
            data_file_uri: denormalize_uri(&final_data_file_uri),
        })
    })
}

/// Byte-rewrite the URI scheme prefix across every metadata file that
/// `iceberg_create_test_table` just wrote (`v1.metadata.json`, the manifest
/// list AVRO, the data manifest AVRO, and the delete manifest AVRO when
/// present).
///
/// ## Why this exists
///
/// iceberg-rust bakes absolute, write-time URIs into every level of a table's
/// metadata tree and never rewrites them on read:
///
/// - `v1.metadata.json` records each snapshot's `manifest_list` location
/// - the manifest list AVRO records each manifest's file path
/// - each manifest AVRO records its data (and delete) file paths
///
/// A table physically written under one scheme therefore reads as a table of
/// that scheme forever, regardless of which FileIO scheme the reader picks.
/// A top-level scheme flip on `metadata_location` alone is not enough —
/// iceberg-rust's `plan_files` follows the chain and will reject any embedded
/// reference whose scheme disagrees with the FileIO it's using.
///
/// ## When this applies (and when it doesn't)
///
/// This is purely a test-side affordance for cross-tenant GCP: the test has
/// to write via S3-compatibility (`s3://` endpoint at `storage.googleapis.com`
/// with HMAC AK/SK — the only way, because opendal's native GCS backend
/// rejects HMAC) but must then read via native `gs://` with SA impersonation,
/// which is the feature under test. Production writers pick one scheme up
/// front and stick with it, so they never need this path.
///
/// ## Why byte-level replace is safe
///
/// `s3` and `gs` are both two bytes, so `s3://` and `gs://` are the same byte
/// count. JSON stays valid after the swap, and AVRO string length varints
/// (which encode a byte count, not a codepoint count) don't need recomputing.
/// The caller must ensure `from_scheme` and `to_scheme` have equal length; we
/// bail otherwise.
async fn rewrite_iceberg_scheme(
    file_io: &FileIO,
    metadata_dir: &str,
    snapshot_id: i64,
    with_positional_deletes: bool,
    from_scheme: &str,
    to_scheme: &str,
) -> Result<(), anyhow::Error> {
    if from_scheme.len() != to_scheme.len() {
        anyhow::bail!(
            "rewrite_iceberg_scheme requires equal-length schemes (got `{}` -> `{}`); \
             byte-level replace preserves AVRO length varints only when byte counts match",
            from_scheme,
            to_scheme
        );
    }
    let from = format!("{}://", from_scheme);
    let to = format!("{}://", to_scheme);
    let from_b = from.as_bytes();
    let to_b = to.as_bytes();

    // File names are fully determined by `iceberg_create_test_table`'s own
    // write calls (see steps 5, 6, 7, 9 in that function) — we don't need to
    // list the directory, and doing so would require an opendal Operator
    // separate from the iceberg FileIO abstraction.
    let mut paths = vec![
        format!("{}/v1.metadata.json", metadata_dir),
        format!("{}/snap-{}-manifest-list.avro", metadata_dir, snapshot_id),
        format!("{}/manifest-data-0.avro", metadata_dir),
    ];
    if with_positional_deletes {
        paths.push(format!("{}/manifest-deletes-0.avro", metadata_dir));
    }

    for path in &paths {
        let bytes = file_io.new_input(path)?.read().await?;
        let mut buf = bytes.to_vec();
        let n = from_b.len();
        let mut i = 0;
        while i + n <= buf.len() {
            if &buf[i..i + n] == from_b {
                buf[i..i + n].copy_from_slice(to_b);
                i += n;
            } else {
                i += 1;
            }
        }
        // opendal-backed FileIO writes replace on PUT — no explicit delete needed.
        file_io.new_output(path)?.write(Bytes::from(buf)).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_basic_table() {
        let dir = tempfile::tempdir().unwrap();
        let table_dir = dir.path().to_str().unwrap();

        let info =
            iceberg_create_test_table(table_dir, 10, false, vec![], vec![], vec![], "").unwrap();
        assert_eq!(info.snapshot_id, 1);
        assert!(!info.metadata_location.is_empty());
        assert!(!info.data_file_uri.is_empty());

        // Verify we can plan files from this table
        let result = crate::iceberg_bridgeimpl::iceberg_plan_files(
            &info.metadata_location,
            info.snapshot_id,
            vec![],
            vec![],
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].record_count, 10);
        assert_eq!(result[0].data_file_path, info.data_file_uri);
    }

    #[test]
    fn test_create_table_with_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let table_dir = dir.path().to_str().unwrap();

        let info =
            iceberg_create_test_table(table_dir, 20, true, vec![2, 5, 10], vec![], vec![], "")
                .unwrap();
        assert_eq!(info.snapshot_id, 1);

        let result = crate::iceberg_bridgeimpl::iceberg_plan_files(
            &info.metadata_location,
            info.snapshot_id,
            vec![],
            vec![],
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].record_count, 20);
        // Should have delete metadata
        assert!(!result[0].delete_metadata_json.is_empty());
    }
}
