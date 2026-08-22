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
use crate::bridge_error::{BridgeError, BridgeResult};

use arrow_array::Array;
use futures::TryStreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use crate::aliyun_oss_provider::AliyunOssStorageFactory;
use crate::aws_arn_provider::{AssumeRoleConfig, build_iceberg_factory};
use crate::azure_sas_provider::{AzureBrokerConfig, AzureSasStorageFactory};
use crate::cloud_provider_cache::{CACHE_CAPACITY, CACHE_KEY, GlobalLruCache};
use crate::gcp_impersonation::{
    GcpImpersonationConfig, GcpImpersonationStorageFactory, ICEBERG_TARGET_SERVICE_ACCOUNT,
};
use crate::iceberg_ffi::IcebergFileInfo;
use iceberg::TableIdent;
use iceberg::io::{FileIOBuilder, LocalFsStorageFactory, MemoryStorageFactory, StorageFactory};
use iceberg::scan::FileScanTask;
use iceberg::table::StaticTable;
use iceberg_storage_opendal::OpenDalStorageFactory;

const CLOUD_PROVIDER_KEY: &str = "cloud_provider";

static ICEBERG_FACTORY_CACHE: LazyLock<GlobalLruCache<Arc<dyn StorageFactory>>> =
    LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));

fn azdls_factory(scheme: &str) -> anyhow::Result<OpenDalStorageFactory> {
    // `OpenDalStorageFactory::Azdls { configured_scheme }` is public, but
    // `AzureStorageScheme` is not re-exported by iceberg-storage-opendal.
    let variant = match scheme {
        "abfs" => "Abfs",
        "abfss" => "Abfss",
        "wasb" => "Wasb",
        "wasbs" => "Wasbs",
        _ => anyhow::bail!("Unsupported Azure storage scheme: {scheme}"),
    };
    let json = format!(r#"{{"Azdls":{{"configured_scheme":"{variant}"}}}}"#);
    serde_json::from_str(&json).map_err(|error| anyhow::anyhow!("construct Azdls factory: {error}"))
}

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

/// Consumes and validates the bridge-private cloud provider selector.
pub(crate) async fn prepare_cloud_storage_options(
    props: &mut HashMap<String, String>,
) -> anyhow::Result<()> {
    match props.remove(CLOUD_PROVIDER_KEY).as_deref() {
        Some("aws" | "azure" | "gcp" | "aliyun") | None => Ok(()),
        Some(provider) => anyhow::bail!("Unsupported Iceberg cloud provider: {provider}"),
    }
}

/// Scheme → `iceberg-storage-opendal` variant. Hand-written because 0.9
/// ships no `from_scheme` helper; collapse when upstream adds one.
async fn upstream_opendal_factory(
    uri: &str,
    scheme: &str,
    props: &mut HashMap<String, String>,
) -> anyhow::Result<Arc<dyn StorageFactory>> {
    let cache_key = props.remove(CACHE_KEY).filter(|key| !key.is_empty());
    let cloud_provider = props.get(CLOUD_PROVIDER_KEY).cloned();
    let azure_broker = if cloud_provider.as_deref() == Some("azure") {
        AzureBrokerConfig::extract(props)?
    } else {
        None
    };
    let gcp_impersonation = if cloud_provider.as_deref() == Some("gcp") {
        GcpImpersonationConfig::extract(props, ICEBERG_TARGET_SERVICE_ACCOUNT)?
    } else {
        None
    };
    let assume_role = if cloud_provider.as_deref() == Some("aws") {
        let role_arn = props.remove("client.assume-role.arn").unwrap_or_default();
        let session_name = props
            .remove("client.assume-role.session-name")
            .unwrap_or_default();
        let external_id = props
            .remove("client.assume-role.external-id")
            .unwrap_or_default();
        let region = props.get("s3.region").cloned().unwrap_or_default();
        let credential_refresh_secs = props
            .remove("aws_credential_refresh_secs")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        AssumeRoleConfig::parse(
            &role_arn,
            &session_name,
            &external_id,
            &region,
            credential_refresh_secs,
        )?
    } else {
        None
    };

    match scheme {
        "abfs" | "abfss" | "wasb" | "wasbs" if azure_broker.is_some() => {
            let config = azure_broker.expect("guarded by is_some");
            let inner = azdls_factory(scheme)?;
            let scoped_cache_key = cache_key
                .as_deref()
                .map(|cache_key| format!("azure:{scheme}:{cache_key}"));
            match scoped_cache_key.as_deref() {
                Some(cache_key) => ICEBERG_FACTORY_CACHE
                    .get(cache_key, || async {
                        let factory = Arc::new(AzureSasStorageFactory::new(inner, config).await?)
                            as Arc<dyn StorageFactory>;
                        eprintln!(
                            "created cloud cache entry: consumer=iceberg, cloud=azure, mechanism=broker_sas"
                        );
                        Ok::<_, anyhow::Error>(factory)
                    })
                    .await,
                None => Ok(Arc::new(AzureSasStorageFactory::new(inner, config).await?)),
            }
        }
        "gs" if gcp_impersonation.is_some() => {
            let config = gcp_impersonation.expect("guarded by is_some");
            match cache_key.as_deref() {
                Some(cache_key) => ICEBERG_FACTORY_CACHE
                    .get(cache_key, || async {
                        let factory = Arc::new(GcpImpersonationStorageFactory::new(config).await?)
                            as Arc<dyn StorageFactory>;
                        eprintln!(
                            "created cloud cache entry: consumer=iceberg, cloud=gcp, mechanism=service_account_impersonation"
                        );
                        Ok::<_, anyhow::Error>(factory)
                    })
                    .await,
                None => Ok(Arc::new(GcpImpersonationStorageFactory::new(config).await?)),
            }
        }
        // The upstream OSS factory does not carry per-tenant `oss.role-arn`.
        "oss" if cloud_provider.as_deref() == Some("aliyun")
            && props.contains_key("oss.role-arn") =>
        {
            match cache_key.as_deref() {
                Some(cache_key) => ICEBERG_FACTORY_CACHE
                    .get(cache_key, || async {
                        let factory = Arc::new(
                            AliyunOssStorageFactory::from_uri(uri, props).await?,
                        ) as Arc<dyn StorageFactory>;
                        eprintln!(
                            "created cloud cache entry: consumer=iceberg, cloud=aliyun, mechanism=role"
                        );
                        Ok::<_, anyhow::Error>(factory)
                    })
                    .await,
                None => Ok(Arc::new(
                    AliyunOssStorageFactory::from_uri(uri, props).await?,
                )),
            }
        }
        "oss" => Ok(Arc::new(AliyunOssStorageFactory::default())),
        "s3" | "s3a" if assume_role.is_some() => {
            let config = assume_role.as_ref().unwrap();
            match cache_key.as_deref() {
                Some(cache_key) => ICEBERG_FACTORY_CACHE
                    .get(cache_key, || async {
                        let factory = build_iceberg_factory(scheme, config).await?;
                        eprintln!(
                            "created cloud cache entry: consumer=iceberg, cloud=aws, mechanism=assume_role"
                        );
                        Ok::<_, anyhow::Error>(factory)
                    })
                    .await,
                None => Ok(build_iceberg_factory(scheme, config).await?),
            }
        }
        "s3" | "s3a" => Ok(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: scheme.to_string(),
            customized_credential_load: None,
        })),
        "gs" => Ok(Arc::new(OpenDalStorageFactory::Gcs)),
        "abfs" | "abfss" | "wasb" | "wasbs" => Ok(Arc::new(azdls_factory(scheme)?)),
        "file" => Ok(Arc::new(LocalFsStorageFactory)),
        "memory" => Ok(Arc::new(MemoryStorageFactory)),
        other => anyhow::bail!("Unsupported scheme for iceberg FileIO: {other}"),
    }
}

pub(crate) async fn build_file_io(
    uri: &str,
    scheme: &str,
    props: &mut HashMap<String, String>,
) -> anyhow::Result<iceberg::io::FileIO> {
    let factory = upstream_opendal_factory(uri, scheme, props).await?;
    prepare_cloud_storage_options(props).await?;
    let mut builder = FileIOBuilder::new(factory);
    for (k, v) in props.iter() {
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
/// - Azure: Milvus `azure://container/path` is canonicalized to ABFSS, then
///   expanded to `abfss://container@{account}.dfs.{suffix}/path`
pub(crate) fn normalize_uri(uri: &str, props: &HashMap<String, String>) -> (String, String) {
    let scheme_end = match uri.find("://") {
        Some(pos) => pos,
        None => return (uri.to_string(), "file".to_string()),
    };
    let authority_start = scheme_end + 3;
    let rest = &uri[authority_start..];
    let scheme = &uri[..scheme_end];
    match scheme {
        "azure" | "abfss" | "abfs" => {
            // `azure` is the Milvus external-source scheme. Iceberg/OpenDAL
            // requires a standard Azure Data Lake Storage scheme.
            let normalized_scheme = if scheme == "azure" { "abfss" } else { scheme };
            // Only check for '@' in the authority (before the first '/').
            // Paths can legitimately contain '@' (e.g. abfss://container/user@org/file).
            let authority = rest.split('/').next().unwrap_or(rest);
            let normalized = if authority.contains('@') {
                format!("{normalized_scheme}://{rest}")
            } else {
                let account = match props.get("adls.account-name") {
                    Some(a) if !a.is_empty() => a,
                    _ => {
                        return (format!("{normalized_scheme}://{rest}"), "abfss".to_string());
                    }
                };
                let suffix = props
                    .get("adls.endpoint-suffix")
                    .map(|s| s.as_str())
                    .unwrap_or("core.windows.net");
                if let Some(slash) = rest.find('/') {
                    let container = &rest[..slash];
                    let path = &rest[slash..];
                    format!("{normalized_scheme}://{container}@{account}.dfs.{suffix}{path}")
                } else {
                    format!("{normalized_scheme}://{rest}@{account}.dfs.{suffix}")
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
    record_count: u64,
) -> Result<u64, anyhow::Error> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut positions = HashSet::new();
    for del_ref in delete_refs {
        if del_ref.file_type != "position" {
            continue;
        }

        // Read the delete file via FileIO
        let input = file_io.new_input(&del_ref.path)?;
        let bytes = input.read().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(|error| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!(
                        "cannot read positional delete Parquet metadata {}: {error}",
                        del_ref.path
                    ),
                )
            })?
            .build()
            .map_err(|error| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!(
                        "cannot decode positional delete Parquet file {}: {error}",
                        del_ref.path
                    ),
                )
            })?;

        for batch in reader {
            let batch = batch.map_err(|error| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!(
                        "cannot decode positional delete batch {}: {error}",
                        del_ref.path
                    ),
                )
            })?;
            let schema = batch.schema();
            let file_path_idx = schema.index_of("file_path").map_err(|_| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!(
                        "positional delete file is missing file_path: {}",
                        del_ref.path
                    ),
                )
            })?;
            let pos_idx = schema.index_of("pos").map_err(|_| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!("positional delete file is missing pos: {}", del_ref.path),
                )
            })?;
            let file_path_array = batch
                .column(file_path_idx)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| {
                    iceberg::Error::new(
                        iceberg::ErrorKind::DataInvalid,
                        format!(
                            "positional delete file_path must be string: {}",
                            del_ref.path
                        ),
                    )
                })?;
            let pos_array = batch
                .column(pos_idx)
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or_else(|| {
                    iceberg::Error::new(
                        iceberg::ErrorKind::DataInvalid,
                        format!("positional delete pos must be int64: {}", del_ref.path),
                    )
                })?;
            for i in 0..file_path_array.len() {
                if file_path_array.is_null(i) || pos_array.is_null(i) {
                    return Err(iceberg::Error::new(
                        iceberg::ErrorKind::DataInvalid,
                        format!(
                            "positional delete contains null file_path/pos: {}",
                            del_ref.path
                        ),
                    )
                    .into());
                }
                if file_path_array.value(i) == data_file_path {
                    let position = pos_array.value(i);
                    if position < 0 || position as u64 >= record_count {
                        return Err(iceberg::Error::new(
                            iceberg::ErrorKind::DataInvalid,
                            format!("positional delete position {position} is outside data file {data_file_path}"),
                        ).into());
                    }
                    // Deletes are a set: a snapshot may legally carry several
                    // delete files naming the same (file_path, pos), so a
                    // repeat is deduplicated rather than rejected. The
                    // out-of-range case above stays an error -- it cannot be
                    // applied to this data file at all.
                    positions.insert(position);
                }
            }
        }
    }
    Ok(positions.len() as u64)
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
) -> BridgeResult<Vec<IcebergFileInfo>> {
    if metadata_location.is_empty() {
        return Err(BridgeError::new(
            None,
            "metadata_location must not be empty".to_string(),
        ));
    }

    let result: anyhow::Result<Vec<IcebergFileInfo>> = TOKIO_RT.block_on(async {
        let mut props = vec_to_hashmap(storage_options_keys, storage_options_values);

        // Normalize URI for opendal and detect FileIO scheme in one pass.
        // For Azure ABFSS, expands scheme://container/path to container@endpoint format.
        let (resolved_location, scheme) = normalize_uri(metadata_location, &props);

        let file_io = build_file_io(&resolved_location, &scheme, &mut props).await?;

        // Load table metadata directly from location (no catalog needed)
        let table_ident = TableIdent::from_strs(["default", "table"])?;
        let table =
            StaticTable::from_metadata_file(&resolved_location, table_ident, file_io.clone())
                .await?;
        let table = table.into_table();

        // Build scan pinned to the specified snapshot
        if table.metadata().snapshot_by_id(snapshot_id).is_none() {
            return Err(BridgeError::new(
                Some(crate::bridge_error::LOON_STORAGE_NOT_FOUND),
                format!("Iceberg snapshot with id {snapshot_id} was not found"),
            )
            .into());
        }
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
                    return Err(iceberg::Error::new(
                        iceberg::ErrorKind::FeatureUnsupported,
                        format!(
                            "Equality deletes are not supported. Data file: {}, delete file: {}. \
                             Equality deletes must be converted to positional deletes before explore.",
                            task.data_file_path, del_ref.path
                        ),
                    )
                    .into());
                }
            }

            // Count deleted rows by reading positional delete files
            let record_count = task.record_count.ok_or_else(|| {
                iceberg::Error::new(
                    iceberg::ErrorKind::DataInvalid,
                    format!("Iceberg data file has no record count: {}", task.data_file_path),
                )
            })?;
            let num_deleted_rows = if delete_refs.is_empty() {
                0
            } else {
                count_positional_deletes(&file_io, &task.data_file_path, &delete_refs, record_count)
                    .await?
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
    });
    result.map_err(BridgeError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::azure_sas_provider::{
        AZURE_BROKER_ACCOUNT_NAME, AZURE_BROKER_BUCKET, AZURE_BROKER_CLIENT_ID,
        AZURE_BROKER_DURATION_SECONDS, AZURE_BROKER_ENDPOINT, AZURE_BROKER_REGION,
        AZURE_BROKER_REQUEST_TIMEOUT_MS, AZURE_BROKER_TENANT_ID,
    };
    use crate::gcp_impersonation::{
        GcpImpersonationConfig, ICEBERG_TARGET_SERVICE_ACCOUNT, TOKEN_LIFETIME_SECONDS,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const AZURE_PRIVATE_KEYS: [&str; 8] = [
        AZURE_BROKER_ENDPOINT,
        AZURE_BROKER_CLIENT_ID,
        AZURE_BROKER_TENANT_ID,
        AZURE_BROKER_ACCOUNT_NAME,
        AZURE_BROKER_REGION,
        AZURE_BROKER_BUCKET,
        AZURE_BROKER_DURATION_SECONDS,
        AZURE_BROKER_REQUEST_TIMEOUT_MS,
    ];

    fn azure_broker_props() -> HashMap<String, String> {
        HashMap::from([
            (
                AZURE_BROKER_ENDPOINT.to_string(),
                "http://127.0.0.1:1".to_string(),
            ),
            (AZURE_BROKER_CLIENT_ID.to_string(), "client".to_string()),
            (AZURE_BROKER_TENANT_ID.to_string(), "tenant".to_string()),
            (AZURE_BROKER_ACCOUNT_NAME.to_string(), "account".to_string()),
            (AZURE_BROKER_REGION.to_string(), "westus3".to_string()),
            (AZURE_BROKER_BUCKET.to_string(), "container".to_string()),
            (
                AZURE_BROKER_DURATION_SECONDS.to_string(),
                "3600".to_string(),
            ),
            (
                AZURE_BROKER_REQUEST_TIMEOUT_MS.to_string(),
                "20".to_string(),
            ),
            (
                "adls.endpoint-suffix".to_string(),
                "core.windows.net".to_string(),
            ),
        ])
    }

    fn gcp_impersonation_props() -> HashMap<String, String> {
        HashMap::from([
            (
                ICEBERG_TARGET_SERVICE_ACCOUNT.to_string(),
                "target@example.com".to_string(),
            ),
            (TOKEN_LIFETIME_SECONDS.to_string(), "3600".to_string()),
        ])
    }

    #[tokio::test]
    async fn prepare_cloud_storage_options_only_removes_supported_selector() {
        for (provider, static_options) in [
            (
                "azure",
                HashMap::from([
                    ("adls.account-name".to_string(), "account".to_string()),
                    (
                        "adls.endpoint-suffix".to_string(),
                        "core.windows.net".to_string(),
                    ),
                    ("adls.sas-token".to_string(), "static-sas".to_string()),
                ]),
            ),
            (
                "gcp",
                HashMap::from([
                    (
                        "gcs.service-account-key".to_string(),
                        "static-key".to_string(),
                    ),
                    ("gcs.oauth2.token".to_string(), "static-token".to_string()),
                ]),
            ),
        ] {
            let mut props = static_options.clone();
            props.insert(CLOUD_PROVIDER_KEY.to_string(), provider.to_string());

            prepare_cloud_storage_options(&mut props).await.unwrap();

            assert_eq!(props, static_options);
        }

        for provider in ["aws", "aliyun"] {
            let mut props = HashMap::from([
                (CLOUD_PROVIDER_KEY.to_string(), provider.to_string()),
                ("ordinary.option".to_string(), "value".to_string()),
            ]);
            prepare_cloud_storage_options(&mut props).await.unwrap();
            assert_eq!(
                props,
                HashMap::from([("ordinary.option".to_string(), "value".to_string())])
            );
        }
    }

    #[test]
    fn azure_and_gcp_private_options_are_removed_by_their_parsers() {
        let mut azure_props = azure_broker_props();
        azure_props.insert("ordinary.option".to_string(), "value".to_string());

        assert!(
            AzureBrokerConfig::extract(&mut azure_props)
                .unwrap()
                .is_some()
        );
        assert!(
            AZURE_PRIVATE_KEYS
                .iter()
                .all(|key| !azure_props.contains_key(*key))
        );
        assert_eq!(azure_props["adls.endpoint-suffix"], "core.windows.net");
        assert_eq!(azure_props["ordinary.option"], "value");

        let mut gcp_props = gcp_impersonation_props();
        gcp_props.insert("ordinary.option".to_string(), "value".to_string());
        assert!(
            GcpImpersonationConfig::extract(&mut gcp_props, ICEBERG_TARGET_SERVICE_ACCOUNT,)
                .unwrap()
                .is_some()
        );
        assert!(!gcp_props.contains_key(ICEBERG_TARGET_SERVICE_ACCOUNT));
        assert!(!gcp_props.contains_key(TOKEN_LIFETIME_SECONDS));
        assert_eq!(gcp_props["ordinary.option"], "value");
    }

    #[tokio::test]
    async fn unsupported_cloud_provider_is_rejected() {
        let mut props =
            HashMap::from([(CLOUD_PROVIDER_KEY.to_string(), "unsupported".to_string())]);

        let error = prepare_cloud_storage_options(&mut props).await.unwrap_err();

        assert_eq!(
            error.to_string(),
            "Unsupported Iceberg cloud provider: unsupported"
        );
        assert!(!props.contains_key(CLOUD_PROVIDER_KEY));
    }

    #[tokio::test]
    async fn azure_factory_cache_separates_resolved_schemes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = requests.clone();
        let response_body = serde_json::json!({
            "success": true,
            "credentials": {
                "tempAk": "account",
                "sessionToken": "sv=1&sig=scheme-separated",
                "expiredAt": (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339(),
            }
        })
        .to_string();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                server_requests.fetch_add(1, Ordering::SeqCst);
                let mut request = [0_u8; 4096];
                socket.read(&mut request).await.unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let cache_key = "test-azure-resolved-scheme-separation";
        let mut abfss_props = azure_broker_props();
        abfss_props.insert(
            AZURE_BROKER_ENDPOINT.to_string(),
            format!("http://{address}"),
        );
        abfss_props.insert(CLOUD_PROVIDER_KEY.to_string(), "azure".to_string());
        abfss_props.insert(CACHE_KEY.to_string(), cache_key.to_string());

        let mut wasbs_props = abfss_props.clone();
        let abfss_factory = upstream_opendal_factory(
            "abfss://container@account.dfs.core.windows.net/metadata/table.json",
            "abfss",
            &mut abfss_props,
        )
        .await
        .unwrap();
        let wasbs_factory = upstream_opendal_factory(
            "wasbs://container@account.blob.core.windows.net/metadata/table.json",
            "wasbs",
            &mut wasbs_props,
        )
        .await
        .unwrap();

        assert!(!Arc::ptr_eq(&abfss_factory, &wasbs_factory));
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.abort();
        let _ = server.await;
    }

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
        let result = iceberg_plan_files("/nonexistent/path/v1.metadata.json", 1, vec![], vec![]);
        assert!(
            result.is_err(),
            "Expected error for nonexistent metadata file"
        );
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
        // Milvus Azure URI → standard ABFSS URI for Iceberg/OpenDAL.
        assert_eq!(
            normalize_uri("azure://mycontainer/some/path", &props),
            (
                "abfss://mycontainer@myaccount.dfs.core.windows.net/some/path".to_string(),
                "abfss".to_string()
            )
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
        assert_eq!(denormalize_uri("s3://bucket/key"), "s3://bucket/key");
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
