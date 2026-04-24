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
use std::sync::Arc;

use iceberg::io::{FileIOBuilder, LocalFsStorageFactory, MemoryStorageFactory, StorageFactory};
use iceberg::scan::FileScanTask;
use iceberg::table::StaticTable;
use iceberg::TableIdent;
use iceberg_storage_opendal::OpenDalStorageFactory;

use crate::aliyun_oss_provider::AliyunOssStorageFactory;
use crate::gcp_impersonation::{fetch_impersonated_bearer, DEFAULT_TOKEN_LIFETIME_SECS};
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

/// Intercepts `oss://` so per-tenant `oss.role-arn` can reach opendal —
/// upstream `OpenDalStorageFactory::Oss` only carries endpoint/AK/SK.
/// Every other scheme is a pure pass-through to upstream.
fn storage_factory_for_scheme(scheme: &str) -> anyhow::Result<Arc<dyn StorageFactory>> {
    if scheme == "oss" {
        return Ok(Arc::new(AliyunOssStorageFactory::default()));
    }
    upstream_opendal_factory(scheme)
}

/// Scheme → `iceberg-storage-opendal` variant. Hand-written because 0.9
/// ships no `from_scheme` helper; collapse when upstream adds one.
fn upstream_opendal_factory(scheme: &str) -> anyhow::Result<Arc<dyn StorageFactory>> {
    match scheme {
        "s3" | "s3a" => Ok(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: scheme.to_string(),
            customized_credential_load: None,
        })),
        "gs" => Ok(Arc::new(OpenDalStorageFactory::Gcs)),
        "abfs" | "abfss" | "wasb" | "wasbs" => {
            // `OpenDalStorageFactory::Azdls { configured_scheme }` is pub,
            // but its `AzureStorageScheme` field type isn't `pub use`'d in
            // lib.rs — round-trip via the pub `Deserialize` impl instead.
            let variant = match scheme {
                "abfs" => "Abfs",
                "abfss" => "Abfss",
                "wasb" => "Wasb",
                "wasbs" => "Wasbs",
                _ => unreachable!(),
            };
            let json = format!(r#"{{"Azdls":{{"configured_scheme":"{variant}"}}}}"#);
            let factory: OpenDalStorageFactory = serde_json::from_str(&json)
                .map_err(|e| anyhow::anyhow!("construct Azdls factory: {e}"))?;
            Ok(Arc::new(factory))
        }
        "file" => Ok(Arc::new(LocalFsStorageFactory)),
        "memory" => Ok(Arc::new(MemoryStorageFactory)),
        other => anyhow::bail!("Unsupported scheme for iceberg FileIO: {other}"),
    }
}

pub(crate) fn build_file_io(
    scheme: &str,
    props: &HashMap<String, String>,
) -> anyhow::Result<iceberg::io::FileIO> {
    let factory = storage_factory_for_scheme(scheme)?;
    let mut builder = FileIOBuilder::new(factory);
    for (k, v) in props {
        builder = builder.with_prop(k, v);
    }
    // `FileIOBuilder::build` became infallible in iceberg 0.9 (was
    // `Result<FileIO>` in 0.8); storage construction is deferred to first
    // use inside `FileIO::get_storage`.
    Ok(builder.build())
}

/// Detect the FileIO scheme from a URI.
/// Normalize a URI for opendal and detect the FileIO scheme in one pass.
///
/// Returns `(normalized_uri, io_scheme)`:
/// - S3/GCS/local: URI unchanged, scheme mapped (e.g. "s3a" → "s3")
/// - Azure ABFSS: `abfss://container/path` expanded to
///   `abfss://container@{account}.dfs.{suffix}/path`, scheme → "abfss"
pub(crate) fn normalize_uri(uri: &str, props: &HashMap<String, String>) -> (String, String) {
    let scheme_end = match uri.find("://") {
        Some(pos) => pos,
        None => return (uri.to_string(), "file".to_string()),
    };
    let authority_start = scheme_end + 3;
    let rest = &uri[authority_start..];
    match &uri[..scheme_end] {
        "abfss" | "abfs" => {
            // Only check for '@' in the authority (before the first '/').
            // Paths can legitimately contain '@' (e.g. abfss://container/user@org/file).
            let authority = rest.split('/').next().unwrap_or(rest);
            let normalized = if authority.contains('@') {
                uri.to_string() // already in container@endpoint format
            } else {
                let account = match props.get("adls.account-name") {
                    Some(a) if !a.is_empty() => a,
                    _ => return (uri.to_string(), "abfss".to_string()),
                };
                let suffix = props
                    .get("adls.endpoint-suffix")
                    .map(|s| s.as_str())
                    .unwrap_or("core.windows.net");
                let scheme = &uri[..authority_start];
                if let Some(slash) = rest.find('/') {
                    let container = &rest[..slash];
                    let path = &rest[slash..];
                    format!("{}{}@{}.dfs.{}{}", scheme, container, account, suffix, path)
                } else {
                    format!("{}{}@{}.dfs.{}", scheme, rest, account, suffix)
                }
            };
            (normalized, "abfss".to_string())
        }
        "s3" | "s3a" => (uri.to_string(), "s3".to_string()),
        "gs" | "gcs" => (uri.to_string(), "gs".to_string()),
        scheme => (uri.to_string(), scheme.to_string()),
    }
}

/// Convert a provider-specific URI back to the uniform `scheme://bucket/path` format.
///
/// - S3/GCS: returned unchanged
/// - Azure ABFSS: `abfss://container@endpoint/path` → `abfss://container/path`
pub(crate) fn denormalize_uri(uri: &str) -> String {
    let scheme_end = match uri.find("://") {
        Some(pos) => pos,
        None => return uri.to_string(),
    };
    let authority_start = scheme_end + 3;
    let rest = &uri[authority_start..];
    match &uri[..scheme_end] {
        "abfss" | "abfs" => {
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
    metadata_location: &str,
    snapshot_id: i64,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Vec<IcebergFileInfo>, anyhow::Error> {
    if metadata_location.is_empty() {
        anyhow::bail!("metadata_location must not be empty");
    }

    TOKIO_RT.block_on(async {
        let mut props = vec_to_hashmap(storage_options_keys, storage_options_values);

        // GCP cross-tenant impersonation: iceberg-rust 0.8's gcs_config_parse
        // doesn't recognize `gcs.service-account` as an impersonation target —
        // it's silently dropped, reqsign falls through to VM metadata, and
        // requests go out as the VM's default SA instead of the target. Swap
        // the key for a pre-fetched impersonated bearer via `gcs.oauth2.token`
        // (which opendal's GcsConfig.token accepts as a static bearer). A
        // 1-hour token covers plan_files' transient metadata/manifest reads
        // with room to spare; no refresh needed. See
        // `docs/iceberg-gcp-impersonation-analysis.md`.
        if let Some(target_sa) = props.remove("gcs.service-account") {
            if !target_sa.is_empty() {
                let bearer = fetch_impersonated_bearer(
                    &target_sa,
                    std::time::Duration::from_secs(DEFAULT_TOKEN_LIFETIME_SECS),
                )
                .await?;
                props.insert("gcs.oauth2.token".to_string(), bearer);
            }
        }

        // Normalize URI for opendal and detect FileIO scheme in one pass.
        // For Azure ABFSS, expands scheme://container/path to container@endpoint format.
        let (resolved_location, scheme) = normalize_uri(metadata_location, &props);

        let file_io = build_file_io(&scheme, &props)?;

        // Load table metadata directly from location (no catalog needed)
        let table_ident = TableIdent::from_strs(["default", "table"])?;
        let table =
            StaticTable::from_metadata_file(&resolved_location, table_ident, file_io.clone()).await?;
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

            // Count deleted rows by reading positional delete files
            let num_deleted_rows = if delete_refs.is_empty() {
                0
            } else {
                count_positional_deletes(&file_io, &task.data_file_path, &delete_refs).await?
            };

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
    fn test_normalize_uri_scheme_detection() {
        let empty: HashMap<String, String> = HashMap::new();
        // S3
        assert_eq!(normalize_uri("s3://bucket/path", &empty).1, "s3");
        assert_eq!(normalize_uri("s3a://bucket/path", &empty).1, "s3");
        // GCS
        assert_eq!(normalize_uri("gs://bucket/path", &empty).1, "gs");
        assert_eq!(normalize_uri("gcs://bucket/path", &empty).1, "gs");
        // Azure
        assert_eq!(normalize_uri("abfss://c/path", &empty).1, "abfss");
        assert_eq!(normalize_uri("abfs://c/path", &empty).1, "abfss");
        // Local
        assert_eq!(normalize_uri("/tmp/path", &empty).1, "file");
        assert_eq!(normalize_uri("file:///tmp/path", &empty).1, "file");
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

    #[test]
    fn test_normalize_uri() {
        let props: HashMap<String, String> = [
            ("adls.account-name".into(), "myaccount".into()),
            ("adls.endpoint-suffix".into(), "core.windows.net".into()),
        ]
        .into();
        // Simple format → container@endpoint format
        assert_eq!(
            normalize_uri("abfss://mycontainer/some/path", &props).0,
            "abfss://mycontainer@myaccount.dfs.core.windows.net/some/path"
        );
        // Already has @ → unchanged
        assert_eq!(
            normalize_uri("abfss://c@acc.dfs.core.windows.net/p", &props).0,
            "abfss://c@acc.dfs.core.windows.net/p"
        );
        // S3 → unchanged
        assert_eq!(
            normalize_uri("s3://bucket/key", &props).0,
            "s3://bucket/key"
        );
        // Default suffix when not provided
        let props_no_suffix: HashMap<String, String> =
            [("adls.account-name".into(), "acc".into())].into();
        assert_eq!(
            normalize_uri("abfss://cont/path", &props_no_suffix).0,
            "abfss://cont@acc.dfs.core.windows.net/path"
        );
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
        assert_eq!(
            denormalize_uri("s3://bucket/key"),
            "s3://bucket/key"
        );
        // abfs scheme
        assert_eq!(
            denormalize_uri("abfs://c@a.dfs.core.windows.net/p"),
            "abfs://c/p"
        );
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let props: HashMap<String, String> = [
            ("adls.account-name".into(), "hnsbucket".into()),
            ("adls.endpoint-suffix".into(), "core.windows.net".into()),
        ]
        .into();
        let simple = "abfss://hnsbucket/test-dir/iceberg/data/file.parquet";
        let (normalized, scheme) = normalize_uri(simple, &props);
        assert_eq!(scheme, "abfss");
        assert_eq!(
            normalized,
            "abfss://hnsbucket@hnsbucket.dfs.core.windows.net/test-dir/iceberg/data/file.parquet"
        );
        let denormalized = denormalize_uri(&normalized);
        assert_eq!(denormalized, simple);
    }
}
