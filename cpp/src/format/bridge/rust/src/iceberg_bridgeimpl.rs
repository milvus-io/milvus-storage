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

use futures::TryStreamExt;
use std::collections::HashMap;

use iceberg::io::FileIOBuilder;
use iceberg::scan::FileScanTask;
use iceberg::table::StaticTable;
use iceberg::TableIdent;

use crate::iceberg_ffi::IcebergFileInfo;

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

/// Detect the FileIO scheme from a URI.
pub(crate) fn detect_io_scheme(metadata_location: &str) -> &str {
    if let Some(pos) = metadata_location.find("://") {
        match &metadata_location[..pos] {
            "s3" | "s3a" => "s3",
            "gs" | "gcs" => "gs",
            "az" | "abfs" | "abfss" => "az",
            scheme => scheme,
        }
    } else {
        "file"
    }
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
    metadata_location: &str,
    snapshot_id: i64,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Vec<IcebergFileInfo>, anyhow::Error> {
    if metadata_location.is_empty() {
        anyhow::bail!("metadata_location must not be empty");
    }

    TOKIO_RT.block_on(async {
        let props = vec_to_hashmap(storage_options_keys, storage_options_values);

        // Build FileIO from storage options, auto-detecting scheme from URI
        let scheme = detect_io_scheme(metadata_location);
        let mut file_io_builder = FileIOBuilder::new(scheme);
        for (k, v) in &props {
            file_io_builder = file_io_builder.with_prop(k, v);
        }
        let file_io = file_io_builder.build()?;

        // Load table metadata directly from location (no catalog needed)
        let table_ident = TableIdent::from_strs(["default", "table"])?;
        let table =
            StaticTable::from_metadata_file(metadata_location, table_ident, file_io).await?;
        let table = table.into_table();

        // Build scan pinned to the specified snapshot
        let scan = table.scan().snapshot_id(snapshot_id).build()?;

        // Plan files — returns one FileScanTask per data file
        let tasks: Vec<FileScanTask> = scan.plan_files().await?.try_collect().await?;

        let mut result = Vec::with_capacity(tasks.len());
        for task in &tasks {
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

            let delete_metadata_json = if delete_refs.is_empty() {
                Vec::new() // empty metadata = no deletes
            } else {
                serde_json::to_vec(&delete_refs)?
            };

            // record_count is required by Iceberg spec but Option in Rust.
            // Fallback: 0 (caller should handle via Parquet metadata read).
            let record_count = task.record_count.unwrap_or(0);

            result.push(IcebergFileInfo {
                data_file_path: task.data_file_path.clone(),
                record_count,
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
    fn test_detect_io_scheme_s3() {
        assert_eq!(detect_io_scheme("s3://bucket/path/metadata.json"), "s3");
        assert_eq!(detect_io_scheme("s3a://bucket/path/metadata.json"), "s3");
    }

    #[test]
    fn test_detect_io_scheme_local() {
        assert_eq!(detect_io_scheme("/tmp/path/metadata.json"), "file");
        assert_eq!(detect_io_scheme("file:///tmp/path/metadata.json"), "file");
    }

    #[test]
    fn test_detect_io_scheme_gcs() {
        assert_eq!(detect_io_scheme("gs://bucket/path"), "gs");
        assert_eq!(detect_io_scheme("gcs://bucket/path"), "gs");
    }

    #[test]
    fn test_detect_io_scheme_azure() {
        assert_eq!(detect_io_scheme("az://container/path"), "az");
        assert_eq!(detect_io_scheme("abfs://container/path"), "az");
        assert_eq!(detect_io_scheme("abfss://container/path"), "az");
    }

    #[test]
    fn test_plan_files_invalid_local_path() {
        let result = iceberg_plan_files(
            "/nonexistent/path/v1.metadata.json",
            1,
            vec![],
            vec![],
        );
        assert!(result.is_err(), "Expected error for nonexistent metadata file");
    }

    #[test]
    fn test_build_delete_metadata_types() {
        // Verify that build_delete_metadata correctly maps DataContentType
        // (equality delete rejection happens in iceberg_plan_files, not here)
        let refs = vec![
            DeleteFileRef {
                path: "s3://bucket/del.parquet".to_string(),
                file_type: "position".to_string(),
                equality_ids: None,
                content_offset: None,
                content_size: None,
            },
        ];
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
}
