// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::TOKIO_RT;

use futures::TryStreamExt;
use futures::stream::StreamExt;
use std::collections::HashMap;
use std::ops::Range;
use std::result::Result as RustResult;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use tokio::runtime::Handle;

use arrow_array58::Array;
use arrow_array58::ffi::FFI_ArrowArray;
use arrow_array58::{RecordBatch, RecordBatchReader, StructArray};
use arrow_schema58::Schema as ArrowSchema;
use arrow58::datatypes::SchemaRef;
use arrow58::error::ArrowError;
use arrow58::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};

use lance::dataset::builder::DatasetBuilder;
use lance::dataset::AutoCleanupParams;
use lance::dataset::cleanup::{CleanupPolicy, RemovalStats};
use lance::dataset::fragment::{FileFragment, FragReadConfig, FragmentReader};
use lance::dataset::optimize::{CompactionOptions as RustCompactionOptions, compact_files};
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::scanner::Scanner;
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, Dataset, ReadParams, Version, WriteMode, WriteParams};
use lance::{Error as LanceError, Result};
use lance_encoding::version::LanceFileVersion;

use crate::lance_ffi::{LanceColumnMemoryEstimate, LanceDataStorageFormat};

use lance::index::DatasetIndexExt;
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::utils::stream::ReadBatchFutStream;

use lance::io::ObjectStoreParams;
use lance::session::Session;
use lance_io::object_store::{
    ObjectStoreProvider, ObjectStoreRegistry, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use crate::aliyun_oss_provider::{AliyunOssStoreProvider, build_aliyun_oss_session};
use crate::aws_arn_provider::{
    AssumeRoleConfig, build_lance_provider as build_aws_arn_provider,
    build_lance_session as build_aws_arn_session,
};
use crate::azure_sas_provider::{AzureBrokerConfig, AzureSasStorageOptionsProvider};
use crate::cloud_provider_cache::{
    CACHE_CAPACITY, CACHE_KEY, GlobalLruCache,
};
use crate::gcp_impersonation::{ImpersonatingGcsStoreProvider, REFRESH_OFFSET_SECS};

const CLOUD_PROVIDER_KEY: &str = "cloud_provider";

static LANCE_PROVIDER_CACHE: LazyLock<GlobalLruCache<Arc<dyn ObjectStoreProvider>>> =
    LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
    object_store: Arc<lance::io::ObjectStore>,
    // Cached readers can open multiple fragment readers against the same dataset
    // concurrently. Keep Lance's scan scheduler dataset-scoped and serialize the
    // open phase, which touches Lance's async metadata/read caches.
    scan_scheduler: Arc<OnceLock<Arc<ScanScheduler>>>,
    fragment_open_mutex: Arc<Mutex<()>>,
}

impl BlockingDataset {
    fn new(inner: Dataset) -> Result<Self> {
        let object_store = TOKIO_RT.block_on(inner.object_store(None))?;
        Ok(Self {
            inner,
            object_store,
            scan_scheduler: Arc::new(OnceLock::new()),
            fragment_open_mutex: Arc::new(Mutex::new(())),
        })
    }

    fn fragment_read_config(&self, read_config: FragReadConfig) -> FragReadConfig {
        if read_config.scan_scheduler.is_some() {
            return read_config;
        }

        let scan_scheduler = self
            .scan_scheduler
            .get_or_init(|| {
                TOKIO_RT.block_on(async {
                    ScanScheduler::new(
                        self.object_store.clone(),
                        SchedulerConfig::max_bandwidth(&self.object_store),
                    )
                })
            })
            .clone();
        read_config.with_scan_scheduler(scan_scheduler)
    }

    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = TOKIO_RT.block_on(Dataset::write(reader, uri, params))?;
        Self::new(inner)
    }

    pub fn commit(
        uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let inner = TOKIO_RT.block_on(Dataset::commit(
            uri,
            operation,
            read_version,
            Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(storage_options),
                )),
                ..Default::default()
            }),
            None,
            Default::default(),
            false, // TODO: support enable_v2_manifest_paths
        ))?;
        Self::new(inner)
    }

    pub fn latest_version(&self) -> Result<u64> {
        let version = TOKIO_RT.block_on(self.inner.latest_version_id())?;
        Ok(version)
    }

    pub fn list_versions(&self) -> Result<Vec<Version>> {
        let versions = TOKIO_RT.block_on(self.inner.versions())?;
        Ok(versions)
    }

    pub fn version(&self) -> Result<Version> {
        Ok(self.inner.version())
    }

    pub fn checkout_version(&mut self, version: u64) -> Result<Self> {
        let inner = TOKIO_RT.block_on(self.inner.checkout_version(version))?;
        Self::new(inner)
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = TOKIO_RT.block_on(self.inner.checkout_version(tag))?;
        Self::new(inner)
    }

    pub fn checkout_latest(&mut self) -> Result<()> {
        TOKIO_RT.block_on(self.inner.checkout_latest())?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        TOKIO_RT.block_on(self.inner.restore())?;
        Ok(())
    }

    pub fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        let tags = TOKIO_RT.block_on(self.inner.tags().list())?;
        Ok(tags)
    }

    pub fn list_branches(&self) -> Result<HashMap<String, lance::dataset::refs::BranchContents>> {
        let branches = TOKIO_RT.block_on(self.inner.list_branches())?;
        Ok(branches)
    }

    pub fn create_branch(
        &mut self,
        branch: &str,
        version: u64,
        source_branch: Option<&str>,
    ) -> Result<Self> {
        let reference = match source_branch {
            Some(b) => Ref::from((b, version)),
            None => Ref::from(version),
        };
        let inner = TOKIO_RT.block_on(self.inner.create_branch(branch, reference, None))?;
        Self::new(inner)
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        TOKIO_RT.block_on(self.inner.delete_branch(branch))?;
        Ok(())
    }

    pub fn checkout_reference(
        &mut self,
        branch: Option<String>,
        version: Option<u64>,
        tag: Option<String>,
    ) -> Result<Self> {
        let reference = if let Some(tag_name) = tag {
            Ref::from(tag_name.as_str())
        } else {
            Ref::Version(branch, version)
        };
        let inner = TOKIO_RT.block_on(self.inner.checkout_version(reference))?;
        Self::new(inner)
    }

    pub fn create_tag(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        let reference = Ref::Version(branch.map(str::to_string), Some(version_number));
        TOKIO_RT.block_on(self.inner.tags().create(tag, reference))?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        TOKIO_RT.block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, version: u64, branch: Option<&str>) -> Result<()> {
        let reference = Ref::Version(branch.map(str::to_string), Some(version));
        TOKIO_RT.block_on(self.inner.tags().update(tag, reference))?;
        Ok(())
    }

    pub fn get_version(&self, tag: &str) -> Result<u64> {
        let version = TOKIO_RT.block_on(self.inner.tags().get_version(tag))?;
        Ok(version)
    }

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        let rows = TOKIO_RT.block_on(self.inner.count_rows(filter))?;
        Ok(rows)
    }

    pub fn delete_rows(&mut self, predicate: &str) -> Result<()> {
        TOKIO_RT.block_on(self.inner.delete(predicate))?;
        Ok(())
    }

    pub fn calculate_data_stats(&self) -> Result<DataStatistics> {
        let stats = TOKIO_RT.block_on(Arc::new(self.clone().inner).calculate_data_stats())?;
        Ok(stats)
    }

    pub fn list_indexes(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indexes = TOKIO_RT.block_on(self.inner.load_indices())?;
        Ok(indexes)
    }

    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        write_params: HashMap<String, String>,
    ) -> Result<Self> {
        let new_dataset = TOKIO_RT.block_on(
            CommitBuilder::new(Arc::new(self.clone().inner))
                .with_store_params(ObjectStoreParams {
                    storage_options_accessor: Some(Arc::new(
                        StorageOptionsAccessor::with_static_options(write_params),
                    )),
                    ..Default::default()
                })
                .execute(transaction),
        )?;
        Self::new(new_dataset)
    }

    pub fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction = TOKIO_RT.block_on(self.inner.read_transaction())?;
        Ok(transaction)
    }

    pub fn get_table_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(self.inner.metadata().clone())
    }

    pub fn compact(&mut self, options: RustCompactionOptions) -> Result<()> {
        TOKIO_RT.block_on(compact_files(&mut self.inner, options, None))?;
        Ok(())
    }

    pub fn cleanup_with_policy(&mut self, policy: CleanupPolicy) -> Result<RemovalStats> {
        Ok(TOKIO_RT.block_on(self.inner.cleanup_with_policy(policy))?)
    }

    pub fn get_all_fragments(&self) -> Vec<Fragment> {
        self.inner.manifest().fragments.clone().to_vec()
    }

    pub fn get_fragment(&self, id: u64) -> Option<Fragment> {
        self.inner
            .manifest()
            .fragments
            .iter()
            .find(|f| f.id == id)
            .cloned()
    }
}

impl BlockingDataset {
    pub fn io_stats_incremental(&self) -> crate::lance_ffi::LanceIOStats {
        let stats = self.object_store.io_stats_incremental();
        crate::lance_ffi::LanceIOStats {
            read_iops: stats.read_iops,
            read_bytes: stats.read_bytes,
        }
    }

    pub unsafe fn write_stream(&mut self, stream_ptr: *mut u8) -> Result<()> {
        let stream_ptr = stream_ptr as *mut FFI_ArrowArrayStream;
        let stream = unsafe { std::ptr::replace(stream_ptr, FFI_ArrowArrayStream::empty()) };
        let reader = ArrowArrayStreamReader::try_new(stream).map_err(|e| LanceError::IO {
            source: Box::new(e),
            location: snafu::location!(),
        })?;

        TOKIO_RT.block_on(self.inner.append(reader, None))?;
        Ok(())
    }

    pub fn get_all_fragment_ids(&self) -> Vec<u64> {
        self.inner
            .manifest()
            .fragments
            .iter()
            .map(|f| f.id)
            .collect()
    }
}

use crate::iceberg_bridgeimpl::vec_to_hashmap;

/// GCP cross-tenant impersonation parameters extracted from `storage_options`.
///
/// The C++ side (`lance::ToStorageOptions` in `lance_common.cpp`) stamps these
/// keys when `cloud_provider=gcp` and `gcp_target_service_account` is set.
/// They are bridge-private — neither lance-io nor object_store know about them
/// and we strip them here so they can't accidentally be forwarded.
struct GcpImpersonationConfig {
    target_sa: String,
    /// Mapped from `load_frequency` on the C++ side. Becomes the IAM
    /// `generateAccessToken` lifetime; the credential provider auto-refreshes
    /// `REFRESH_OFFSET_SECS` ahead of expiry.
    token_lifetime_secs: u64,
}

impl GcpImpersonationConfig {
    /// Parse from `storage_options`. Returns `Ok(None)` if
    /// `gcp_target_service_account` is not set.  Returns `Err` if
    /// `gcp_credential_refresh_secs` is missing, malformed, or out of range
    /// `[900, 3600]`.
    fn extract(storage_options: &mut HashMap<String, String>) -> Result<Option<Self>> {
        let Some(target_sa) = storage_options.remove("gcp_target_service_account") else {
            return Ok(None);
        };
        if target_sa.is_empty() {
            return Ok(None);
        }
        // Mirror `AssumeRoleConfig::parse`: missing / unparsable falls through
        // to 0 and is rejected by the range check below.  The lower bound must
        // be strictly greater than `REFRESH_OFFSET_SECS` (300s) — otherwise the
        // cached token's `needs_refresh` window opens before it even issues,
        // and every `get_credential` call hammers IAM (credential thrashing).
        // Align the lower bound with AWS at 900s.  The upper bound is GCP
        // IAM's hard cap on impersonated-token lifetime (3600s without an
        // `iam.allowServiceAccountCredentialLifetimeExtension` org policy).
        let token_lifetime_secs: u64 = storage_options
            .remove("gcp_credential_refresh_secs")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if token_lifetime_secs < 900 || token_lifetime_secs > 3600 {
            return Err(LanceError::invalid_input(
                format!(
                    "gcp_credential_refresh_secs must be in [900, 3600], got {}",
                    token_lifetime_secs
                ),
            ));
        }
        Ok(Some(Self {
            target_sa,
            token_lifetime_secs,
        }))
    }
}

/// Build a `Session` whose `ObjectStoreRegistry` overrides the `gs` scheme
/// with an `ImpersonatingGcsStoreProvider`.
///
/// A fresh `Session` is built per call so that two concurrent opens with
/// different target SAs cannot collide on a shared registry. Cache sizes
/// remain zero because index/metadata caches are managed by the caller.
fn build_gcp_impersonation_session(config: &GcpImpersonationConfig) -> Arc<Session> {
    let registry = ObjectStoreRegistry::default();
    registry.insert(
        "gs",
        Arc::new(ImpersonatingGcsStoreProvider::new(
            config.target_sa.clone(),
            std::time::Duration::from_secs(config.token_lifetime_secs),
            std::time::Duration::from_secs(REFRESH_OFFSET_SECS),
        )),
    );
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

pub fn open_dataset(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    let credential_cache_key = storage_options.remove(CACHE_KEY);
    let cloud_provider = storage_options.remove(CLOUD_PROVIDER_KEY);
    if let Some(cloud_provider) = cloud_provider.as_deref()
        && !matches!(cloud_provider, "aws" | "azure" | "gcp" | "aliyun")
    {
        return Err(LanceError::invalid_input(format!(
            "Unsupported Lance cloud provider: {cloud_provider}"
        )));
    }

    // Configure each cloud provider's cross-tenant credential path in one
    // place. AWS, GCP, and Aliyun use a per-call Session with an overridden
    // object-store provider, while Azure uses ObjectStoreParams directly.
    let mut store_params = ObjectStoreParams::default();
    let mut custom_session = None;
    match cloud_provider.as_deref() {
        Some("aws") => {
            let role_arn = storage_options.remove("aws_role_arn").unwrap_or_default();
            let session_name = storage_options
                .remove("aws_session_name")
                .unwrap_or_default();
            let external_id = storage_options.remove("aws_external_id").unwrap_or_default();
            let region = storage_options.get("aws_region").cloned().unwrap_or_default();
            let refresh_secs_str = storage_options
                .remove("aws_credential_refresh_secs")
                .unwrap_or_default();
            let credential_refresh_secs: u64 = refresh_secs_str.parse().unwrap_or(0);
            let assume_role = AssumeRoleConfig::parse(
                &role_arn,
                &session_name,
                &external_id,
                &region,
                credential_refresh_secs,
            )?;
            if let Some(config) = &assume_role {
                let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                    Some(cache_key) => TOKIO_RT.block_on(LANCE_PROVIDER_CACHE.get(
                        cache_key,
                        || async {
                            let provider = build_aws_arn_provider(config).await?;
                            eprintln!(
                                "created cloud cache entry: consumer=lance, cloud=aws, mechanism=assume_role"
                            );
                            Ok::<_, LanceError>(provider)
                        },
                    ))?,
                    None => TOKIO_RT.block_on(build_aws_arn_provider(config))?,
                };
                custom_session = Some(build_aws_arn_session(provider));
            }
        }
        Some("azure") => {
            // Lance refreshes Azure credentials through StorageOptionsAccessor;
            // the broker-backed provider supplies a fresh SAS token as needed.
            store_params.storage_options_accessor = match AzureBrokerConfig::extract(
                &mut storage_options,
            )
            .map_err(|error| LanceError::invalid_input(error.to_string()))?
            {
                Some(config) => {
                    // Emulator and unsigned modes bypass SAS authentication, so
                    // force both off when the broker is configured.
                    storage_options.insert(
                        "azure_storage_use_emulator".to_string(),
                        "false".to_string(),
                    );
                    storage_options
                        .insert("azure_skip_signature".to_string(), "false".to_string());
                    let provider = Arc::new(
                        AzureSasStorageOptionsProvider::new(config)
                            .map_err(|error| LanceError::invalid_input(error.to_string()))?,
                    );
                    // Preserve the static Azure settings and overlay refreshed SAS
                    // values returned by the provider.
                    Some(Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                        storage_options.clone(),
                        provider,
                    )))
                }
                None => None,
            };
        }
        Some("gcp") => {
            // Lance's stock GCS provider cannot refresh impersonated access
            // tokens, so replace the `gs` provider for this dataset only.
            custom_session = GcpImpersonationConfig::extract(&mut storage_options)?
                .map(|config| build_gcp_impersonation_session(&config));
        }
        Some("aliyun") if storage_options.contains_key("oss_role_arn") => {
            let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                Some(cache_key) => TOKIO_RT.block_on(LANCE_PROVIDER_CACHE.get(
                    cache_key,
                    || async {
                        let provider = Arc::new(
                            AliyunOssStoreProvider::from_uri(uri, &storage_options).await?,
                        ) as Arc<dyn ObjectStoreProvider>;
                        eprintln!(
                            "created cloud cache entry: consumer=lance, cloud=aliyun, mechanism=role"
                        );
                        Ok::<_, LanceError>(provider)
                    },
                ))?,
                None => Arc::new(TOKIO_RT.block_on(AliyunOssStoreProvider::from_uri(
                    uri,
                    &storage_options,
                ))?) as Arc<dyn ObjectStoreProvider>,
            };
            custom_session = Some(build_aliyun_oss_session(provider));
        }
        _ => {}
    }

    // Do not pass credential_refresh_secs as s3_credentials_refresh_offset here:
    // The AWS ARN credential provider handles refresh internally with REFRESH_OFFSET_SECS.
    // Passing the full session TTL (e.g. 900s) as the offset would cause Lance to
    // consider credentials expired immediately after issuance (credential thrashing).
    if store_params.storage_options_accessor.is_none() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }
    let read_params = ReadParams {
        index_cache_size_bytes: 0,
        metadata_cache_size_bytes: 0,
        store_options: Some(store_params),
        ..Default::default()
    };
    let mut builder = DatasetBuilder::from_uri(uri).with_read_params(read_params);
    if let Some(session) = custom_session {
        builder = builder.with_session(session);
    }
    let inner = TOKIO_RT.block_on(builder.load())?;
    Ok(Box::new(BlockingDataset::new(inner)?))
}

pub unsafe fn write_dataset(
    uri: &str,
    stream_ptr: *mut u8,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    data_storage_format: LanceDataStorageFormat,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    let credential_cache_key = storage_options.remove(CACHE_KEY);
    let cloud_provider = storage_options.remove(CLOUD_PROVIDER_KEY);
    if let Some(cloud_provider) = cloud_provider.as_deref()
        && !matches!(cloud_provider, "aws" | "azure" | "gcp" | "aliyun")
    {
        return Err(LanceError::invalid_input(format!(
            "Unsupported Lance cloud provider: {cloud_provider}"
        )));
    }
    // Keep write-side credential selection symmetric with open_dataset.
    let mut store_params = ObjectStoreParams::default();
    let mut custom_session = None;
    match cloud_provider.as_deref() {
        Some("aws") => {
            let role_arn = storage_options.remove("aws_role_arn").unwrap_or_default();
            let session_name = storage_options
                .remove("aws_session_name")
                .unwrap_or_default();
            let external_id = storage_options.remove("aws_external_id").unwrap_or_default();
            let region = storage_options.get("aws_region").cloned().unwrap_or_default();
            let refresh_secs_str = storage_options
                .remove("aws_credential_refresh_secs")
                .unwrap_or_default();
            let credential_refresh_secs: u64 = refresh_secs_str.parse().unwrap_or(0);
            let assume_role = AssumeRoleConfig::parse(
                &role_arn,
                &session_name,
                &external_id,
                &region,
                credential_refresh_secs,
            )?;
            if let Some(config) = &assume_role {
                let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                    Some(cache_key) => TOKIO_RT.block_on(LANCE_PROVIDER_CACHE.get(
                        cache_key,
                        || async {
                            let provider = build_aws_arn_provider(config).await?;
                            eprintln!(
                                "created cloud cache entry: consumer=lance, cloud=aws, mechanism=assume_role"
                            );
                            Ok::<_, LanceError>(provider)
                        },
                    ))?,
                    None => TOKIO_RT.block_on(build_aws_arn_provider(config))?,
                };
                custom_session = Some(build_aws_arn_session(provider));
            }
        }
        Some("azure") => {
            store_params.storage_options_accessor = match AzureBrokerConfig::extract(
                &mut storage_options,
            )
            .map_err(|error| LanceError::invalid_input(error.to_string()))?
            {
                Some(config) => {
                    storage_options.insert(
                        "azure_storage_use_emulator".to_string(),
                        "false".to_string(),
                    );
                    storage_options
                        .insert("azure_skip_signature".to_string(), "false".to_string());
                    let provider = Arc::new(
                        AzureSasStorageOptionsProvider::new(config)
                            .map_err(|error| LanceError::invalid_input(error.to_string()))?,
                    );
                    Some(Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                        storage_options.clone(),
                        provider,
                    )))
                }
                None => None,
            };
        }
        Some("gcp") => {
            custom_session = GcpImpersonationConfig::extract(&mut storage_options)?
                .map(|config| build_gcp_impersonation_session(&config));
        }
        Some("aliyun") if storage_options.contains_key("oss_role_arn") => {
            let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                Some(cache_key) => TOKIO_RT.block_on(LANCE_PROVIDER_CACHE.get(
                    cache_key,
                    || async {
                        let provider = Arc::new(
                            AliyunOssStoreProvider::from_uri(uri, &storage_options).await?,
                        ) as Arc<dyn ObjectStoreProvider>;
                        eprintln!(
                            "created cloud cache entry: consumer=lance, cloud=aliyun, mechanism=role"
                        );
                        Ok::<_, LanceError>(provider)
                    },
                ))?,
                None => Arc::new(TOKIO_RT.block_on(AliyunOssStoreProvider::from_uri(
                    uri,
                    &storage_options,
                ))?) as Arc<dyn ObjectStoreProvider>,
            };
            custom_session = Some(build_aliyun_oss_session(provider));
        }
        _ => {}
    }
    if store_params.storage_options_accessor.is_none() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }

    let stream_ptr = stream_ptr as *mut FFI_ArrowArrayStream;
    let stream = unsafe { std::ptr::replace(stream_ptr, FFI_ArrowArrayStream::empty()) };
    let reader = ArrowArrayStreamReader::try_new(stream).map_err(|e| LanceError::IO {
        source: Box::new(e),
        location: snafu::location!(),
    })?;

    let lance_file_version = match data_storage_format {
        LanceDataStorageFormat::Legacy => LanceFileVersion::Legacy,
        LanceDataStorageFormat::V2_1 => LanceFileVersion::V2_1,
        LanceDataStorageFormat::V2_2 => LanceFileVersion::V2_2,
        LanceDataStorageFormat::V2_3 => LanceFileVersion::V2_3,
        _ => LanceFileVersion::Legacy,
    };

    let mut write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(lance_file_version),
        enable_v2_manifest_paths: false,
        session: custom_session,
        auto_cleanup: Some(AutoCleanupParams::default()),
        ..Default::default()
    };
    write_params.store_params = Some(store_params);

    let inner = TOKIO_RT.block_on(Dataset::write(reader, uri, Some(write_params)))?;
    Ok(Box::new(BlockingDataset::new(inner)?))
}

struct BatchFutStreamReader {
    stream: futures::stream::Buffered<ReadBatchFutStream>,
    schema: SchemaRef,
    runtime_handle: Handle,
}

impl Iterator for BatchFutStreamReader {
    type Item = RustResult<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Use the runtime handle to block on the async stream
        self.runtime_handle
            .block_on(async { self.stream.next().await })
            .map(|res| {
                // Convert Lance Error to Arrow Error
                res.map_err(|e| ArrowError::from_external_error(Box::new(e)))
            })
    }
}

impl RecordBatchReader for BatchFutStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub trait ToFFIStream {
    fn to_ffi_stream(self, schema: SchemaRef, handle: Handle) -> FFI_ArrowArrayStream;
}

pub trait ToFFIArray {
    fn to_ffi_array(self) -> FFI_ArrowArray;
}

impl ToFFIStream for ReadBatchFutStream {
    fn to_ffi_stream(self, schema: SchemaRef, handle: Handle) -> FFI_ArrowArrayStream {
        // Buffer the stream for concurrency
        let buffered_stream = self.buffered(1); // Adjust buffer size as needed

        let reader = BatchFutStreamReader {
            stream: buffered_stream,
            schema,
            runtime_handle: handle,
        };

        // Create FFI stream from the reader
        FFI_ArrowArrayStream::new(Box::new(reader))
    }
}

impl ToFFIArray for RecordBatch {
    fn to_ffi_array(self) -> FFI_ArrowArray {
        let struct_array = StructArray::from(self);
        let data = struct_array.into_data();
        FFI_ArrowArray::new(&data)
    }
}

pub async fn collect_stream_to_batches(
    stream: ReadBatchFutStream,
    concurrency: usize,
) -> Result<Vec<RecordBatch>> {
    stream.buffered(concurrency).try_collect::<Vec<_>>().await
}

#[derive(Clone)]
pub struct BlockingFragmentReader {
    pub inner: FragmentReader,
    pub fragment: FileFragment,
    pub projection: ArrowSchema,
    sorted_deletions: Vec<u32>,
}

impl BlockingFragmentReader {
    pub fn open(
        dataset: &BlockingDataset,
        fragment: Fragment,
        arrow_projection: &ArrowSchema,
        read_config: FragReadConfig,
    ) -> Result<Self> {
        let _open_guard = dataset
            .fragment_open_mutex
            .lock()
            .map_err(|_| LanceError::Internal {
                message: "Lance fragment open mutex poisoned".into(),
                location: snafu::location!(),
            })?;

        let projection = arrow_projection.clone();
        let fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment);

        // Load deletion vector for logical→physical index mapping in take()
        let sorted_deletions = {
            let dv = TOKIO_RT.block_on(fragment.get_deletion_vector())?;
            match dv {
                Some(dv) => {
                    let mut dels: Vec<u32> = dv.as_ref().clone().into_iter().map(|i| i as u32).collect();
                    dels.sort();
                    dels
                }
                None => vec![],
            }
        };

        let meta_schema = fragment.schema();
        let meta_columns: std::collections::HashSet<_> = meta_schema
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect();

        let columns: Vec<_> = arrow_projection
            .fields()
            .iter()
            .map(|f| f.name())
            .filter(|n| meta_columns.contains(*n))
            .map(|n| n.clone())
            .collect();

        let fragment_reader = TOKIO_RT.block_on(fragment.open(&meta_schema.project(&columns)?, read_config))?;

        Ok(Self {
            inner: fragment_reader,
            fragment,
            projection,
            sorted_deletions,
        })
    }

    /// Map logical index to physical index, accounting for deletions.
    fn logical_to_physical(&self, logical: u32) -> u32 {
        if self.sorted_deletions.is_empty() {
            return logical;
        }
        let mut physical = logical;
        loop {
            let num_dels = self.sorted_deletions.partition_point(|&d| d <= physical) as u32;
            let new_physical = logical + num_dels;
            if new_physical == physical {
                break;
            }
            physical = new_physical;
        }
        physical
    }

    fn map_logical_indices(&self, logical_indices: &[u32]) -> Vec<u32> {
        if self.sorted_deletions.is_empty() {
            return logical_indices.to_vec();
        }
        logical_indices.iter().map(|&i| self.logical_to_physical(i)).collect()
    }

    pub fn number_of_rows(&self) -> Result<u64> {
        Ok(TOKIO_RT.block_on(self.fragment.count_rows(None))? as u64)
    }

    pub fn take_as_single_batch(&self, indices: &[u32], out_array: *mut u8) -> Result<()> {
        let physical_indices = self.map_logical_indices(indices);
        let ffi_array = TOKIO_RT
            .block_on(self.inner.take_as_batch(&physical_indices, None))?
            .to_ffi_array();
        let out_array = out_array as *mut FFI_ArrowArray;
        // # Safety
        // Arrow C array interface
        unsafe { std::ptr::write(out_array, ffi_array) };
        Ok(())
    }

    pub unsafe fn take_as_stream(
        &self,
        indices: &[u32],
        batch_size: u32,
        out_stream: *mut u8,
    ) -> Result<()> {
        let physical_indices = self.map_logical_indices(indices);
        let read_batch_fut_stream = TOKIO_RT.block_on(self.inner.take(&physical_indices, batch_size, None));

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
            Arc::new(self.projection.clone()),
            TOKIO_RT.handle().clone(),
        );
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        // # Safety
        // Arrow C stream interface
        unsafe { std::ptr::write(out_stream, ffi_stream) };
        Ok(())
    }

    pub unsafe fn read_all_as_stream(&self, batch_size: u32, out_stream: *mut u8) -> Result<()> {
        let read_batch_fut_stream = TOKIO_RT.block_on(self.inner.read_all(batch_size))?;

        let ffi_stream = read_batch_fut_stream.to_ffi_stream(
            Arc::new(self.projection.clone()),
            TOKIO_RT.handle().clone(),
        );
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        unsafe { std::ptr::write(out_stream, ffi_stream) };
        Ok(())
    }

    pub unsafe fn read_ranges_as_stream_internal(
        &self,
        range: Range<u32>,
        batch_size: u32,
        out_stream: *mut u8,
    ) -> Result<()> {
        let read_batch_fut_stream = TOKIO_RT.block_on(self.inner.read_range(range, batch_size))?;

        let ffi_stream = read_batch_fut_stream.to_ffi_stream(
            Arc::new(self.projection.clone()),
            TOKIO_RT.handle().clone(),
        );
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        unsafe { std::ptr::write(out_stream, ffi_stream) };
        Ok(())
    }

    pub unsafe fn read_ranges_as_stream(
        self: &BlockingFragmentReader,
        row_range_start: u32,
        row_range_end: u32,
        batch_size: u32,
        out_stream: *mut u8,
    ) -> Result<()> {
        unsafe {
            self.read_ranges_as_stream_internal(
                Range {
                    start: row_range_start,
                    end: row_range_end,
                },
                batch_size,
                out_stream,
            )
        }
    }
}

pub unsafe fn open_fragment_reader(
    dataset: &BlockingDataset,
    fragment_id: u64,
    schema_rawptr: *mut u8,
) -> Result<Box<BlockingFragmentReader>> {
    let fragment = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;

    let ffi_schema = unsafe {
        arrow58::ffi::FFI_ArrowSchema::from_raw(
            schema_rawptr as *mut arrow58::ffi::FFI_ArrowSchema,
        )
    };
    let arrow_schema =
        ArrowSchema::try_from(&ffi_schema).map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to convert schema: {}", e.to_string()).into(),
            location: snafu::location!(),
        })?;

    let reader = BlockingFragmentReader::open(
        dataset,
        fragment,
        &arrow_schema,
        dataset.fragment_read_config(FragReadConfig::default()),
    )?;
    Ok(Box::new(reader))
}

pub fn dataset_delete_rows(dataset: &mut BlockingDataset, predicate: &str) -> Result<()> {
    dataset.delete_rows(predicate)
}

/// Get sorted deletion positions for a fragment. Returns empty vec if no deletions.
pub fn get_fragment_deletion_positions(dataset: &BlockingDataset, fragment_id: u64) -> Result<Vec<u64>> {
    let fragment_meta = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;
    let fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment_meta);
    let dv = TOKIO_RT.block_on(fragment.get_deletion_vector())?;
    match dv {
        Some(dv) => {
            let mut positions: Vec<u64> = dv.as_ref().clone().into_iter().map(|i| i as u64).collect();
            positions.sort();
            Ok(positions)
        }
        None => Ok(vec![]),
    }
}

pub fn get_fragment_physical_row_count(dataset: &BlockingDataset, fragment_id: u64) -> Result<u64> {
    let fragment = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;
    fragment
        .physical_rows
        .map(|n| n as u64)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} has no physical_rows metadata", fragment_id).into(),
            location: snafu::location!(),
        })
}

pub fn get_fragment_row_count(dataset: &BlockingDataset, fragment_id: u64) -> Result<u64> {
    let fragment = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;
    fragment
        .num_rows()
        .map(|n| n as u64)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} has no row count metadata", fragment_id).into(),
            location: snafu::location!(),
        })
}

fn estimate_fragment_columns(
    dataset: &BlockingDataset,
    fragment_id: u64,
) -> Result<Vec<LanceColumnMemoryEstimate>> {
    // Match fragment reader construction: both paths reuse the dataset-scoped
    // scheduler and touch Lance's async metadata/read caches during open.
    let _open_guard = dataset
        .fragment_open_mutex
        .lock()
        .map_err(|_| LanceError::Internal {
            message: "Lance fragment open mutex poisoned".into(),
            location: snafu::location!(),
        })?;
    let fragment = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;

    // Reuse the same scheduler and object-store configuration as normal reads;
    // the estimator itself only schedules footer and column/page metadata I/O.
    let scheduler = dataset
        .fragment_read_config(FragReadConfig::default())
        .scan_scheduler
        .expect("fragment_read_config always installs a scheduler");
    TOKIO_RT.block_on(
        crate::lance_memory_estimator::estimate_fragment_column_memory(
            &dataset.inner,
            &fragment,
            scheduler,
        ),
    )
}

/// Estimate each top-level column's decoded Arrow buffer size in schema order.
///
/// The estimator reads footer and page metadata only. Variable-width columns
/// use Lance's decoded page-size target instead of interpreting page encodings.
pub fn estimate_fragment_column_memory(
    dataset: &BlockingDataset,
    fragment_id: u64,
) -> Result<Vec<LanceColumnMemoryEstimate>> {
    estimate_fragment_columns(dataset, fragment_id)
}

/// Estimate the decoded Arrow buffer size of a fragment without reading data pages.
///
/// This compatibility API is the saturating sum of the per-column estimates.
/// Errors are returned through cxx so the C++ best-effort wrapper can fall back
/// to zero.
pub fn estimate_fragment_memory(dataset: &BlockingDataset, fragment_id: u64) -> Result<u64> {
    Ok(estimate_fragment_columns(dataset, fragment_id)?
        .into_iter()
        .map(|estimate| estimate.memory_size)
        .fold(0_u64, u64::saturating_add))
}

pub unsafe fn get_fragment_schema(
    dataset: &BlockingDataset,
    fragment_id: u64,
    out_schema_ptr: *mut u8,
) -> Result<()> {
    let fragment_meta = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;

    // In Lance 7, FileFragment::schema() returns the current dataset schema. It
    // includes evolved nullable fields that may not be physically stored in this
    // fragment, matching the schema order used by the column memory estimator.
    // The clone is cheap because Dataset internally wraps state in Arcs.
    let file_fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment_meta);
    let lance_schema = file_fragment.schema();
    let arrow_schema: ArrowSchema = lance_schema.into();

    let ffi_schema = arrow58::ffi::FFI_ArrowSchema::try_from(&arrow_schema)
        .map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to export fragment schema: {}", e).into(),
            location: snafu::location!(),
        })?;

    let out_ptr = out_schema_ptr as *mut arrow58::ffi::FFI_ArrowSchema;
    unsafe { std::ptr::write(out_ptr, ffi_schema) };
    Ok(())
}

//=============================================================================
// BlockingScanner: dataset-level scan support
//=============================================================================

/// Simple RecordBatchReader backed by a Vec of batches
struct VecBatchReader {
    batches: std::vec::IntoIter<RecordBatch>,
    schema: SchemaRef,
}

impl Iterator for VecBatchReader {
    type Item = RustResult<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next().map(Ok)
    }
}

impl RecordBatchReader for VecBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub struct BlockingScanner {
    inner: Scanner,
    schema: SchemaRef,
}

impl BlockingScanner {
    pub fn count_rows(&self) -> Result<u64> {
        Ok(TOKIO_RT.block_on(self.inner.count_rows())?)
    }

    pub unsafe fn open_stream(&self, out_stream: *mut u8) -> Result<()> {
        let stream = TOKIO_RT.block_on(self.inner.try_into_stream())?;
        let batches: Vec<RecordBatch> = TOKIO_RT.block_on(stream.try_collect::<Vec<_>>())?;

        let reader = VecBatchReader {
            batches: batches.into_iter(),
            schema: self.schema.clone(),
        };
        let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let out_stream_ptr = out_stream as *mut FFI_ArrowArrayStream;
        unsafe { std::ptr::write(out_stream_ptr, ffi_stream) };
        Ok(())
    }
}

pub unsafe fn create_scanner(
    dataset: &BlockingDataset,
    schema_ptr: *mut u8,
    batch_size: u32,
) -> Result<Box<BlockingScanner>> {
    let ffi_schema = unsafe {
        arrow58::ffi::FFI_ArrowSchema::from_raw(
            schema_ptr as *mut arrow58::ffi::FFI_ArrowSchema,
        )
    };
    let arrow_schema =
        ArrowSchema::try_from(&ffi_schema).map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to convert schema: {}", e).into(),
            location: snafu::location!(),
        })?;

    let column_names: Vec<&str> = arrow_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();

    let mut scanner = dataset.inner.scan();
    scanner.project(&column_names)?;
    scanner.batch_size(batch_size as usize);

    Ok(Box::new(BlockingScanner {
        inner: scanner,
        schema: Arc::new(arrow_schema),
    }))
}

pub unsafe fn dataset_take(
    dataset: &BlockingDataset,
    indices: &[u64],
    schema_ptr: *mut u8,
    out_stream: *mut u8,
) -> Result<()> {
    let ffi_schema = unsafe {
        arrow58::ffi::FFI_ArrowSchema::from_raw(
            schema_ptr as *mut arrow58::ffi::FFI_ArrowSchema,
        )
    };
    let arrow_schema =
        ArrowSchema::try_from(&ffi_schema).map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to convert schema: {}", e).into(),
            location: snafu::location!(),
        })?;

    let column_names: Vec<&str> = arrow_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();

    let projection = dataset.inner.schema().project(&column_names)?;
    let batch = TOKIO_RT.block_on(dataset.inner.take(indices, projection))?;

    let reader = VecBatchReader {
        batches: vec![batch].into_iter(),
        schema: Arc::new(arrow_schema),
    };
    let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream_ptr = out_stream as *mut FFI_ArrowArrayStream;
    unsafe { std::ptr::write(out_stream_ptr, ffi_stream) };
    Ok(())
}
