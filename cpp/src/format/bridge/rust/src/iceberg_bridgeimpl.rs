// Copyright 2024 Zilliz
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

use crate::TOKIO_RT;

use arrow_array::Array;
use futures::TryStreamExt;
use std::collections::HashMap;

use iceberg::TableIdent;
use iceberg::scan::FileScanTask;
use iceberg::table::StaticTable;
use crate::iceberg_ffi::IcebergFileInfo;
use crate::iceberg_opendal::build_filesystem_file_io;

/// Internal representation for a delete file reference, serialized to JSON.
#[derive(serde::Serialize)]
struct DeleteFileRef {
    path: String,
    file_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    equality_ids: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_size: Option<i64>,
}

pub(crate) fn vec_to_hashmap(keys: Vec<String>, values: Vec<String>) -> HashMap<String, String> {
    keys.into_iter().zip(values.into_iter()).collect()
}

/// Convert a provider-specific URI back to the uniform `scheme://bucket/path` format.
///
/// - S3/GCS: returned unchanged
/// - Azure aliases: `scheme://container@endpoint/path` → `scheme://container/path`
pub(crate) fn denormalize_uri(uri: &str) -> String {
    let scheme_end = match uri.find("://") {
        Some(pos) => pos,
        None => return uri.to_string(),
    };
    let authority_start = scheme_end + 3;
    let rest = &uri[authority_start..];
    match &uri[..scheme_end] {
        "azure" | "abfs" | "abfss" | "wasb" | "wasbs" => {
            // Only look for '@' in the authority (before the first '/').
            let first_slash = rest.find('/');
            let authority = match first_slash {
                Some(pos) => &rest[..pos],
                None => rest,
            };
            let at_pos = match authority.find('@') {
                Some(p) => p,
                None => return uri.to_string(), // no @ in authority, already simple
            };
            let container = &rest[..at_pos];
            let scheme = &uri[..authority_start];
            let path = match first_slash {
                Some(pos) => &rest[pos..],
                None => "",
            };
            format!("{}{}{}", scheme, container, path)
        }
        _ => uri.to_string(), // s3, gs, file, etc. — no transform needed
    }
}

/// Count positional delete rows matching a specific data file.
/// Reads each positional delete Parquet file and counts rows where file_path matches.
async fn count_positional_deletes(
    file_io: &iceberg::io::FileIO,
    data_file_path: &str,
    delete_refs: &[DeleteFileRef],
) -> Result<u64, anyhow::Error> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut total = 0u64;
    for del_ref in delete_refs {
        if del_ref.file_type != "position" {
            continue;
        }

        // Read the delete file via FileIO
        let input = file_io.new_input(&del_ref.path)?;
        let bytes = input.read().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;

        for batch in reader {
            let batch = batch?;
            let schema = batch.schema();
            let file_path_idx = schema.index_of("file_path").unwrap_or(0);

            let file_path_col = batch
                .column(file_path_idx)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>();

            if let Some(file_path_array) = file_path_col {
                for i in 0..file_path_array.len() {
                    if !file_path_array.is_null(i) && file_path_array.value(i) == data_file_path {
                        total += 1;
                    }
                }
            }
        }
    }
    Ok(total)
}

fn build_delete_metadata(task: &FileScanTask) -> Vec<DeleteFileRef> {
    task.deletes
        .iter()
        .map(|d| {
            let file_type = match d.file_type {
                iceberg::spec::DataContentType::PositionDeletes => "position".to_string(),
                iceberg::spec::DataContentType::EqualityDeletes => "equality".to_string(),
                _ => "unknown".to_string(),
            };
            DeleteFileRef {
                path: d.file_path.clone(),
                file_type,
                equality_ids: d.equality_ids.clone(),
                content_offset: None,
                content_size: None,
            }
        })
        .collect()
}

pub fn iceberg_plan_files(
    filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
    metadata_location: &str,
    snapshot_id: i64,
    read_option_keys: Vec<String>,
    read_option_values: Vec<String>,
) -> Result<Vec<IcebergFileInfo>, anyhow::Error> {
    if filesystem.is_null() {
        anyhow::bail!("iceberg_plan_files requires a non-null filesystem");
    }
    if metadata_location.is_empty() {
        anyhow::bail!("metadata_location must not be empty");
    }

    TOKIO_RT.block_on(async {
        let read_options = vec_to_hashmap(read_option_keys, read_option_values);
        let (file_io, uri_binding) =
            build_filesystem_file_io(filesystem, metadata_location, read_options)?;

        // Load table metadata directly from location (no catalog needed)
        let table_ident = TableIdent::from_strs(["default", "table"])?;
        let table = StaticTable::from_metadata_file(metadata_location, table_ident, file_io.clone())
            .await?;
        let table = table.into_table();

        // A negative snapshot ID selects the current snapshot.
        let scan_builder = table.scan();
        let scan_builder = if snapshot_id < 0 {
            scan_builder
        } else {
            scan_builder.snapshot_id(snapshot_id)
        };
        let scan = scan_builder.build()?;

        // Plan files — returns one FileScanTask per data file
        let tasks: Vec<FileScanTask> = scan.plan_files().await?.try_collect().await?;

        let mut result = Vec::with_capacity(tasks.len());
        for task in &tasks {
            // Planning reads metadata and manifests, but not the data file itself.
            // Validate the manifest-provided URI before exposing it to C++.
            uri_binding.relative_path(&task.data_file_path)?;

            // Build delete metadata JSON
            let delete_refs = build_delete_metadata(task);

            // Reject equality deletes — they must be pre-converted to
            // positional deletes before the manifest is committed.
            for del_ref in &delete_refs {
                if del_ref.file_type == "equality" {
                    anyhow::bail!(
                        "Equality deletes are not supported. \
                         Data file: {}, delete file: {}. \
                         Equality deletes must be converted to positional deletes \
                         before explore.",
                        task.data_file_path,
                        del_ref.path
                    );
                }
            }

            // Count deleted rows by reading positional delete files
            let num_deleted_rows = if delete_refs.is_empty() {
                0
            } else {
                count_positional_deletes(&file_io, &task.data_file_path, &delete_refs).await?
            };

            // FIXME(jiaqizho): Local planning returns absolute data and delete
            // paths to C++, whose filesystem is already rooted at fs.root_path.
            // Normalize them before enabling production local data access.
            // Production does not currently support reading actual external-table
            // data from local storage, so this local data-read path is test-only.
            // Denormalize delete file paths back to scheme://bucket/path for C++.
            // The delete_refs paths are in opendal format (container@endpoint for Azure).
            let denorm_refs: Vec<DeleteFileRef> = delete_refs
                .into_iter()
                .map(|mut r| {
                    r.path = denormalize_uri(&r.path);
                    r
                })
                .collect();

            let delete_metadata_json = if denorm_refs.is_empty() {
                Vec::new() // empty metadata = no deletes
            } else {
                serde_json::to_vec(&denorm_refs)?
            };

            // record_count is required by Iceberg spec but Option in Rust.
            // Fallback: 0 (caller should handle via Parquet metadata read).
            let record_count = task.record_count.unwrap_or(0);

            // Denormalize data_file_path: strip Azure container@endpoint back to
            // scheme://container/path so C++ sees a uniform format across providers.
            result.push(IcebergFileInfo {
                data_file_path: denormalize_uri(&task.data_file_path),
                record_count,
                num_deleted_rows,
                delete_metadata_json,
            });
        }
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_hashmap() {
        let keys = vec!["k1".to_string(), "k2".to_string()];
        let values = vec!["v1".to_string(), "v2".to_string()];
        let map = vec_to_hashmap(keys, values);
        assert_eq!(map.len(), 2);
        assert_eq!(map["k1"], "v1");
        assert_eq!(map["k2"], "v2");
    }

    #[test]
    fn test_vec_to_hashmap_empty() {
        let map = vec_to_hashmap(vec![], vec![]);
        assert!(map.is_empty());
    }

    #[test]
    fn test_plan_files_rejects_null_filesystem() {
        let result = iceberg_plan_files(
            cxx::SharedPtr::null(),
            "/nonexistent/path/v1.metadata.json",
            1,
            vec![],
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_delete_metadata_types() {
        // Verify that build_delete_metadata correctly maps DataContentType
        // (equality delete rejection happens in iceberg_plan_files, not here)
        let refs = vec![DeleteFileRef {
            path: "s3://bucket/del.parquet".to_string(),
            file_type: "position".to_string(),
            equality_ids: None,
            content_offset: None,
            content_size: None,
        }];
        assert_eq!(refs[0].file_type, "position");
    }

    #[test]
    fn test_delete_file_ref_serialization() {
        let refs = vec![
            DeleteFileRef {
                path: "s3://bucket/table/data/delete-1.parquet".to_string(),
                file_type: "position".to_string(),
                equality_ids: None,
                content_offset: None,
                content_size: None,
            },
            DeleteFileRef {
                path: "s3://bucket/table/data/delete-2.parquet".to_string(),
                file_type: "equality".to_string(),
                equality_ids: Some(vec![1, 2, 3]),
                content_offset: None,
                content_size: None,
            },
        ];

        let json = serde_json::to_string(&refs).unwrap();
        assert!(json.contains("\"file_type\":\"position\""));
        assert!(json.contains("\"file_type\":\"equality\""));
        assert!(json.contains("\"equality_ids\":[1,2,3]"));
        // position delete should not have equality_ids in output
        assert!(!json.contains("\"equality_ids\":null"));
    }

    #[test]
    fn test_denormalize_uri() {
        // Strip container@endpoint → container/path
        assert_eq!(
            denormalize_uri("abfss://mycontainer@myaccount.dfs.core.windows.net/some/path"),
            "abfss://mycontainer/some/path"
        );
        // No @ → unchanged
        assert_eq!(
            denormalize_uri("abfss://mycontainer/some/path"),
            "abfss://mycontainer/some/path"
        );
        // S3 → unchanged
        assert_eq!(denormalize_uri("s3://bucket/key"), "s3://bucket/key");
        // abfs scheme
        assert_eq!(
            denormalize_uri("abfs://c@a.dfs.core.windows.net/p"),
            "abfs://c/p"
        );
    }

    #[test]
    fn test_denormalize_uri_strips_recorded_endpoint_for_all_azure_aliases() {
        for (uri, expected) in [
            ("azure://c@a.blob.core.windows.net/p", "azure://c/p"),
            ("wasb://c@a.blob.core.windows.net/p", "wasb://c/p"),
            ("wasbs://c@a.blob.core.windows.net/p", "wasbs://c/p"),
        ] {
            assert_eq!(denormalize_uri(uri), expected);
        }
    }
}
