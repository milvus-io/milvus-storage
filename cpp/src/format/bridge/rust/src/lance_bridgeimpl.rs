// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::TOKIO_RT;

use futures::TryStreamExt;
use futures::stream::StreamExt;
use std::collections::{HashMap, hash_map::RandomState};
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Range;
use std::result::Result as RustResult;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use tokio::runtime::Handle;
use url::Url;

use arrow_array58::Array;
use arrow_array58::ffi::FFI_ArrowArray;
use arrow_array58::{RecordBatch, RecordBatchReader, StructArray};
use arrow_schema58::Schema as ArrowSchema;
use arrow58::datatypes::SchemaRef;
use arrow58::error::ArrowError;
use arrow58::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};

use lance::dataset::AutoCleanupParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::{CleanupPolicy, RemovalStats};
use lance::dataset::fragment::{FileFragment, FileMetadataMap, FragReadConfig, FragmentReader};
use lance::dataset::optimize::{CompactionOptions as RustCompactionOptions, compact_files};
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::scanner::Scanner;
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, Dataset, ReadParams, Version, WriteMode, WriteParams};
use lance::{Error as LanceError, Result};
use lance_core::utils::deletion::DeletionVector;
use lance_encoding::version::LanceFileVersion;
use lance_table::rowids::RowIdSequence;

use crate::lance_ffi::{LanceColumnMemoryEstimate, LanceDataStorageFormat};

use lance::index::DatasetIndexExt;
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::utils::stream::ReadBatchFutStream;

use lance::io::ObjectStoreParams;
use lance::session::Session;
use lance_io::object_store::{
    ObjectStore, ObjectStoreProvider, StorageOptions, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use crate::aliyun_oss_provider::{
    AimdAliyunOssStoreProvider, AliyunOssStoreProvider, build_aliyun_oss_session,
};
use crate::aws_arn_provider::{
    AssumeRoleConfig, build_lance_provider as build_aws_arn_provider,
    build_lance_session as build_aws_arn_session,
};
use crate::azure_sas_provider::{
    AzureBrokerConfig, AzureSasStorageOptionsProvider,
    build_lance_provider as build_azure_sas_provider,
};
use crate::cloud_provider_cache::{CACHE_CAPACITY, CACHE_KEY, GlobalLruCache};
use crate::gcp_impersonation::{
    GcpImpersonationConfig, LANCE_TARGET_SERVICE_ACCOUNT,
    build_lance_provider as build_gcp_impersonation_provider,
    build_lance_session as build_gcp_impersonation_session,
};

const CLOUD_PROVIDER_KEY: &str = "cloud_provider";
const LANCE_IO_PARALLELISM_KEY: &str = "milvus_lance_io_parallelism";
const MAX_LANCE_IO_PARALLELISM: usize = 256;

static LANCE_PROVIDER_CACHE: LazyLock<GlobalLruCache<Arc<dyn ObjectStoreProvider>>> =
    LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));
static LANCE_AZURE_PROVIDER_CACHE: LazyLock<
    GlobalLruCache<Arc<AzureSasStorageOptionsProvider>>,
> = LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));

// Keep only an opaque, process-local representation of storage options in the
// scheduler registry. Two independent hashers make accidental collisions unlikely.
static IO_DOMAIN_HASHERS: LazyLock<[RandomState; 2]> =
    LazyLock::new(|| [RandomState::new(), RandomState::new()]);

/// Identifies datasets that may share one Lance scan scheduler.
///
/// `store_prefix` scopes the key to one object-store namespace. The configuration
/// fingerprint further separates endpoints, explicit authentication, and ObjectStore
/// behavior settings that can otherwise resolve to the same prefix. Parallelism is
/// intentionally not part of the key: the first live scheduler establishes the limit
/// for that domain.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct IoDomainKey {
    store_prefix: String,
    store_config_fingerprint: [u64; 2],
}

// Weak values let the registry reuse active schedulers without extending their
// lifetime after all datasets in the domain have been dropped.
static LANCE_IO_SCHEDULERS: LazyLock<Mutex<HashMap<IoDomainKey, Weak<ScanScheduler>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Removes and validates the bridge-only parallelism option.
///
/// An absent value or zero leaves Lance's default per-dataset scheduling in place.
/// Removing the option also prevents it from reaching an ObjectStore provider that
/// does not understand this Milvus-specific setting.
fn extract_lance_io_parallelism(
    storage_options: &mut HashMap<String, String>,
) -> Result<Option<usize>> {
    let Some(value) = storage_options.remove(LANCE_IO_PARALLELISM_KEY) else {
        return Ok(None);
    };
    let parallelism = value.parse::<usize>().map_err(|_| {
        LanceError::invalid_input(format!(
            "{LANCE_IO_PARALLELISM_KEY} must be an integer in [0, {MAX_LANCE_IO_PARALLELISM}], got '{value}'"
        ))
    })?;
    if parallelism > MAX_LANCE_IO_PARALLELISM {
        return Err(LanceError::invalid_input(format!(
            "{LANCE_IO_PARALLELISM_KEY} must be in [0, {MAX_LANCE_IO_PARALLELISM}], got {parallelism}"
        )));
    }
    Ok((parallelism != 0).then_some(parallelism))
}

/// Builds an order-independent, process-local fingerprint for scheduler isolation.
///
/// Explicit store, authentication, and ObjectStore behavior options participate in
/// the fingerprint, while scheduler capacity and provider-cache bookkeeping do not.
/// This only partitions scheduler reuse; it is not a credential validation or security
/// mechanism.
fn store_config_fingerprint(storage_options: &HashMap<String, String>) -> [u64; 2] {
    // Environment-backed ObjectStore settings and the identity resolved by a
    // default credential chain are not visible here. Callers must keep them
    // stable while a shared scheduler for the same store is active. Authentication
    // identities that may vary should be selected through the storage options passed
    // to this bridge instead of dynamically mutating process-wide credential sources.
    let mut options = storage_options
        .iter()
        .filter(|(key, _)| key.as_str() != LANCE_IO_PARALLELISM_KEY && key.as_str() != CACHE_KEY)
        .collect::<Vec<_>>();
    // HashMap iteration order is unstable, so canonicalize both keys and values.
    options.sort_unstable_by(|(left_key, left_value), (right_key, right_value)| {
        left_key
            .cmp(right_key)
            .then_with(|| left_value.cmp(right_value))
    });

    std::array::from_fn(|index| {
        let mut hasher = IO_DOMAIN_HASHERS[index].build_hasher();
        options.len().hash(&mut hasher);
        for (key, value) in &options {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        hasher.finish()
    })
}

/// Returns the ObjectStore owned by a shared scheduler.
///
/// The process-wide `LANCE_IO_THREADS` override remains authoritative. Otherwise,
/// when the requested capacity differs, create a lightweight Lance wrapper around
/// the already-authenticated inner store instead of rebuilding a cloud client.
fn scheduler_object_store(
    object_store: &Arc<ObjectStore>,
    uri: &str,
    storage_options: &HashMap<String, String>,
    parallelism: usize,
) -> Result<Arc<ObjectStore>> {
    if std::env::var_os("LANCE_IO_THREADS").is_some()
        || object_store.io_parallelism() == parallelism
    {
        return Ok(object_store.clone());
    }

    let location = Url::parse(uri).map_err(|error| {
        LanceError::invalid_input(format!(
            "Failed to parse Lance dataset URI '{uri}' for shared I/O scheduling: {error}"
        ))
    })?;
    let storage_options = StorageOptions::new(storage_options.clone());
    let download_retry_count = storage_options.download_retry_count();
    // Preserve the initialized provider, credential chain, and transport. Only
    // wrapper-level scheduler capacity and derived ObjectStore metadata change.
    Ok(Arc::new(ObjectStore::new(
        object_store.inner.clone(),
        location,
        Some(object_store.block_size()),
        None,
        object_store.use_constant_size_upload_parts,
        object_store.list_is_lexically_ordered,
        parallelism,
        download_retry_count,
        Some(&storage_options.0),
    )))
}

/// Returns the single active ScanScheduler for an I/O domain.
///
/// The registry lock covers lookup, construction, and insertion so concurrent opens
/// cannot create competing schedulers for the same domain. Only weak references are
/// retained, allowing a later open to create a fresh scheduler after the domain is idle.
fn shared_scan_scheduler(
    key: IoDomainKey,
    object_store: &Arc<ObjectStore>,
    uri: &str,
    storage_options: &HashMap<String, String>,
    parallelism: usize,
) -> Result<Arc<ScanScheduler>> {
    let mut schedulers = LANCE_IO_SCHEDULERS
        .lock()
        .map_err(|_| LanceError::Internal {
            message: "Lance I/O scheduler registry mutex poisoned".into(),
            location: snafu::location!(),
        })?;
    if let Some(scheduler) = schedulers.get(&key).and_then(Weak::upgrade) {
        return Ok(scheduler);
    }

    // Sweep expired weak entries only on a registry miss; active hits stay O(1).
    schedulers.retain(|_, scheduler| scheduler.strong_count() != 0);
    let scheduler_store = scheduler_object_store(object_store, uri, storage_options, parallelism)?;
    let scheduler = TOKIO_RT.block_on(async {
        ScanScheduler::new(
            scheduler_store.clone(),
            SchedulerConfig::max_bandwidth(&scheduler_store),
        )
    });
    schedulers.insert(key, Arc::downgrade(&scheduler));
    Ok(scheduler)
}

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
    object_store: Arc<lance::io::ObjectStore>,
    // The scheduler is dataset-local by default and domain-shared when explicitly
    // configured.
    scan_scheduler: Arc<OnceLock<Arc<ScanScheduler>>>,
}

/// Immutable file footers, deletion state, and stable row IDs for one snapshot's fragment.
#[derive(Clone)]
pub struct FragmentFilesMetadata {
    files: Arc<FileMetadataMap>,
    deletion_vector: Option<Arc<DeletionVector>>,
    sorted_deletions: Arc<[u32]>,
    row_id_sequence: Option<Arc<RowIdSequence>>,
    uri: String,
    version: u64,
    fragment_id: u64,
}

impl FragmentFilesMetadata {
    fn for_fragment(&self, dataset: &BlockingDataset, fragment_id: u64) -> Result<Self> {
        if self.fragment_id != fragment_id
            || self.uri != dataset.inner.uri()
            || self.version != dataset.inner.version().version
        {
            return Err(LanceError::invalid_input(format!(
                "File metadata does not belong to this dataset and fragment: expected fragment {} version {}, got fragment {} version {}",
                fragment_id,
                dataset.inner.version().version,
                self.fragment_id,
                self.version
            )));
        }
        Ok(self.clone())
    }
}

impl BlockingDataset {
    fn new(inner: Dataset) -> Result<Self> {
        let object_store = TOKIO_RT.block_on(inner.object_store(None))?;
        Ok(Self {
            inner,
            object_store,
            scan_scheduler: Arc::new(OnceLock::new()),
        })
    }

    // Called inside TOKIO_RT when opening fragment readers.
    fn fragment_read_config(&self, read_config: FragReadConfig) -> FragReadConfig {
        if read_config.scan_scheduler.is_some() {
            return read_config;
        }

        let scan_scheduler = self
            .scan_scheduler
            .get_or_init(|| {
                ScanScheduler::new(
                    self.object_store.clone(),
                    SchedulerConfig::max_bandwidth(&self.object_store),
                )
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

    pub fn version(&self) -> u64 {
        self.inner.version().version
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

fn build_object_store_params(
    uri: &str,
    mut storage_options: HashMap<String, String>,
) -> Result<(ObjectStoreParams, Option<Arc<Session>>)> {
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
    // object-store provider. Azure uses Lance's StorageOptionsAccessor.
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
            if let Some(config) = AzureBrokerConfig::extract(&mut storage_options)
                .map_err(|error| LanceError::invalid_input(error.to_string()))?
            {
                storage_options.insert("azure_storage_use_emulator".into(), "false".into());
                storage_options.insert("azure_skip_signature".into(), "false".into());
                let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                    Some(cache_key) => TOKIO_RT.block_on(LANCE_AZURE_PROVIDER_CACHE.get(
                        cache_key,
                        || async {
                            let provider = build_azure_sas_provider(config).await?;
                            eprintln!(
                                "created cloud cache entry: consumer=lance, cloud=azure, mechanism=broker_sas"
                            );
                            Ok::<_, LanceError>(provider)
                        },
                    ))?,
                    None => TOKIO_RT.block_on(build_azure_sas_provider(config))?,
                };
                store_params.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_initial_and_provider(
                        storage_options.clone(),
                        provider,
                    ),
                ));
            }
        }
        Some("gcp") => {
            if let Some(config) =
                GcpImpersonationConfig::extract(&mut storage_options, LANCE_TARGET_SERVICE_ACCOUNT)
                    .map_err(|error| LanceError::invalid_input(error.to_string()))?
            {
                let provider = match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
                    Some(cache_key) => TOKIO_RT.block_on(LANCE_PROVIDER_CACHE.get(
                        cache_key,
                        || async {
                            let provider = build_gcp_impersonation_provider(&config).await?;
                            eprintln!(
                                "created cloud cache entry: consumer=lance, cloud=gcp, mechanism=service_account_impersonation"
                            );
                            Ok::<_, LanceError>(provider)
                        },
                    ))?,
                    None => TOKIO_RT.block_on(build_gcp_impersonation_provider(&config))?,
                };
                custom_session = Some(build_gcp_impersonation_session(provider));
            }
        }
        Some("aliyun") => {
            let provider = if storage_options.contains_key("oss_role_arn") {
                match credential_cache_key.as_deref().filter(|key| !key.is_empty()) {
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
                }
            } else {
                Arc::new(AimdAliyunOssStoreProvider) as Arc<dyn ObjectStoreProvider>
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
    Ok((store_params, custom_session))
}

pub fn open_dataset(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    version: u64,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    let lance_io_parallelism = extract_lance_io_parallelism(&mut storage_options)?;
    // Fingerprint before provider-specific setup consumes explicit endpoint and
    // authentication options that must separate scheduler domains.
    let io_domain_fingerprint =
        lance_io_parallelism.map(|_| store_config_fingerprint(&storage_options));
    let scheduler_storage_options = lance_io_parallelism.map(|_| {
        let mut options = storage_options.clone();
        options.remove(CACHE_KEY);
        options
    });
    let (store_params, custom_session) = build_object_store_params(uri, storage_options)?;
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
    if version != 0 {
        builder = builder.with_version(version);
    }
    let inner = TOKIO_RT.block_on(builder.load())?;
    let dataset = BlockingDataset::new(inner)?;
    // Domain sharing is opt-in and cloud-only. Other stores keep the lazy,
    // dataset-local scheduler installed by fragment_read_config.
    if let (Some(parallelism), Some(store_config_fingerprint), Some(storage_options)) = (
        lance_io_parallelism,
        io_domain_fingerprint,
        scheduler_storage_options.as_ref(),
    ) && dataset.object_store.is_cloud()
    {
        let scheduler = shared_scan_scheduler(
            IoDomainKey {
                store_prefix: dataset.object_store.store_prefix.clone(),
                store_config_fingerprint,
            },
            &dataset.object_store,
            uri,
            storage_options,
            parallelism,
        )?;
        dataset
            .scan_scheduler
            .set(scheduler)
            .expect("a newly opened BlockingDataset has no scan scheduler");
    }
    Ok(Box::new(dataset))
}

pub fn resolve_latest_dataset_version(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<u64> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    extract_lance_io_parallelism(&mut storage_options)?;
    let (store_params, custom_session) = build_object_store_params(uri, storage_options)?;
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
    let (object_store, base_path, commit_handler) =
        TOKIO_RT.block_on(builder.build_object_store())?;
    let location = TOKIO_RT
        .block_on(commit_handler.resolve_latest_location(&base_path, object_store.as_ref()))?;
    Ok(location.version)
}

pub unsafe fn write_dataset(
    uri: &str,
    stream_ptr: *mut u8,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    data_storage_format: LanceDataStorageFormat,
    enable_stable_row_ids: bool,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    // The storage Lance writer is available only in test builds. Shared I/O scheduling is a
    // reader feature, so discard the option emitted by the common C++ storage-options path.
    storage_options.remove(LANCE_IO_PARALLELISM_KEY);
    let (store_params, custom_session) = build_object_store_params(uri, storage_options)?;

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
        enable_stable_row_ids,
        enable_v2_manifest_paths: false,
        session: custom_session,
        auto_cleanup: Some(AutoCleanupParams::default()),
        ..Default::default()
    };
    write_params.store_params = Some(store_params);

    let inner = TOKIO_RT.block_on(Dataset::write(reader, uri, Some(write_params)))?;
    Ok(Box::new(BlockingDataset::new(inner)?))
}

/// Convert existing stable row IDs to external files for local cache tests.
pub fn externalize_row_ids_for_testing(uri: &str) -> Result<()> {
    use lance_table::format::{ExternalFile, RowIdMeta};

    TOKIO_RT.block_on(async {
        let dataset = Dataset::open(uri).await?;
        let store = dataset.object_store(None).await?;
        let mut updated_fragments = dataset.manifest().fragments.as_ref().clone();
        for metadata in &mut updated_fragments {
            let Some(RowIdMeta::Inline(data)) = &metadata.row_id_meta else {
                return Err(LanceError::invalid_input(
                    "test requires inline stable row IDs",
                ));
            };
            let path = format!("row_ids_{}.bin", metadata.id);
            store
                .put(&dataset.branch_location().path.join(path.as_str()), data)
                .await?;
            metadata.row_id_meta = Some(RowIdMeta::External(ExternalFile {
                path,
                offset: 0,
                size: data.len() as u64,
            }));
        }
        // Publish the new row-ID references, preserving all data and deletions.
        // A false predicate records that this metadata update deletes no rows.
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids: Vec::new(),
                predicate: "false".into(),
            },
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await?;
        Ok(())
    })
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
    files_metadata: FragmentFilesMetadata,
}

impl BlockingFragmentReader {
    pub async fn open(
        dataset: &BlockingDataset,
        fragment: Fragment,
        arrow_projection: &ArrowSchema,
        mut read_config: FragReadConfig,
        files_metadata: Option<FragmentFilesMetadata>,
    ) -> Result<Self> {
        let projection = arrow_projection.clone();
        let fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment);

        // Retain all v2 footers, deletions, and row IDs before applying the initial
        // projection. Later readers share both without rereading or sorting.
        let files_metadata = match files_metadata {
            Some(metadata) => metadata,
            None => {
                let deletion_vector = fragment.get_deletion_vector().await?;
                let mut sorted_deletions = deletion_vector
                    .as_ref()
                    .map(|vector| vector.iter().collect::<Vec<_>>())
                    .unwrap_or_default();
                sorted_deletions.sort_unstable();
                let row_id_sequence = fragment.get_row_id_sequence().await?;
                let default_scheduler = read_config
                    .scan_scheduler
                    .as_ref()
                    .expect("fragment_read_config always installs a scheduler");
                let mut files = FileMetadataMap::with_capacity(fragment.metadata().files.len());
                for data_file in &fragment.metadata().files {
                    // Legacy readers use their own footer format and still open normally.
                    if data_file.is_legacy_file() {
                        continue;
                    }
                    let data_dir = match data_file.base_id {
                        Some(base_id) => {
                            let base_path = dataset
                                .inner
                                .manifest()
                                .base_paths
                                .get(&base_id)
                                .ok_or_else(|| {
                                    LanceError::invalid_input(format!(
                                        "base path {} not found",
                                        base_id
                                    ))
                                })?;
                            let path =
                                base_path.extract_path(dataset.inner.session().store_registry())?;
                            if base_path.is_dataset_root {
                                path.join("data")
                            } else {
                                path
                            }
                        }
                        None => dataset.inner.data_dir(),
                    };
                    let path = data_dir.join(data_file.path.as_str());
                    let scheduler = if data_file.base_id.is_none() {
                        default_scheduler.clone()
                    } else {
                        let object_store = dataset.inner.object_store(data_file.base_id).await?;
                        ScanScheduler::new(
                            object_store.clone(),
                            SchedulerConfig::max_bandwidth(&object_store),
                        )
                    };
                    let file_scheduler = scheduler
                        .open_file_with_priority(&path, 0, &data_file.file_size_bytes)
                        .await?;
                    let metadata = fragment.get_file_metadata(&file_scheduler).await?;
                    files.insert((data_file.base_id, data_file.path.clone()), metadata);
                }
                FragmentFilesMetadata {
                    files: Arc::new(files),
                    deletion_vector,
                    sorted_deletions: sorted_deletions.into(),
                    row_id_sequence,
                    uri: dataset.inner.uri().to_string(),
                    version: dataset.inner.version().version,
                    fragment_id: fragment.id() as u64,
                }
            }
        };
        if let Some(vector) = &files_metadata.deletion_vector {
            read_config = read_config.with_deletion_vector(vector.clone());
        }
        if let Some(sequence) = &files_metadata.row_id_sequence {
            read_config = read_config.with_row_id_sequence(sequence.clone());
        }

        let meta_schema = fragment.schema();
        let meta_columns: std::collections::HashSet<_> =
            meta_schema.fields.iter().map(|f| f.name.clone()).collect();

        let columns: Vec<_> = arrow_projection
            .fields()
            .iter()
            .map(|f| f.name())
            .filter(|n| meta_columns.contains(*n))
            .map(|n| n.clone())
            .collect();

        let projected_schema = meta_schema.project(&columns)?;
        let fragment_reader = fragment
            .open(&projected_schema, read_config, Some(&files_metadata.files))
            .await?;

        Ok(Self {
            inner: fragment_reader,
            fragment,
            projection,
            files_metadata,
        })
    }

    /// Map logical index to physical index, accounting for deletions.
    fn logical_to_physical(&self, logical: u32) -> u32 {
        if self.files_metadata.sorted_deletions.is_empty() {
            return logical;
        }
        let mut physical = logical;
        loop {
            let num_dels = self
                .files_metadata
                .sorted_deletions
                .partition_point(|&d| d <= physical) as u32;
            let new_physical = logical + num_dels;
            if new_physical == physical {
                break;
            }
            physical = new_physical;
        }
        physical
    }

    fn map_logical_indices(&self, logical_indices: &[u32]) -> Vec<u32> {
        if self.files_metadata.sorted_deletions.is_empty() {
            return logical_indices.to_vec();
        }
        logical_indices
            .iter()
            .map(|&i| self.logical_to_physical(i))
            .collect()
    }

    pub fn get_files_metadata(&self) -> Box<FragmentFilesMetadata> {
        Box::new(self.files_metadata.clone())
    }

    /// Estimate all top-level dataset columns from retained footers, without I/O.
    pub fn estimate_fragment_column_memory(&self) -> Result<Vec<LanceColumnMemoryEstimate>> {
        crate::lance_memory_estimator::estimate_fragment_column_memory(
            self.fragment.schema(),
            self.fragment.metadata(),
            &self.files_metadata.files,
            self.files_metadata.sorted_deletions.len() as u64,
        )
    }

    pub fn estimate_fragment_memory(&self) -> Result<u64> {
        Ok(self
            .estimate_fragment_column_memory()?
            .into_iter()
            .map(|estimate| estimate.memory_size)
            .fold(0_u64, u64::saturating_add))
    }

    pub fn number_of_rows(&self) -> Result<u64> {
        Ok(self.inner.num_rows() as u64)
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

async fn open_fragment_reader_impl(
    dataset: &BlockingDataset,
    fragment_id: u64,
    ffi_schema: arrow58::ffi::FFI_ArrowSchema,
    files_metadata: Option<FragmentFilesMetadata>,
) -> Result<Box<BlockingFragmentReader>> {
    let fragment = dataset
        .get_fragment(fragment_id)
        .ok_or_else(|| LanceError::InvalidInput {
            source: format!("Fragment {} not found", fragment_id).into(),
            location: snafu::location!(),
        })?;

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
        files_metadata,
    )
    .await?;
    Ok(Box::new(reader))
}

pub unsafe fn open_fragment_reader(
    dataset: &BlockingDataset,
    fragment_id: u64,
    schema_rawptr: *mut u8,
    metadata: *const FragmentFilesMetadata,
) -> Result<Box<BlockingFragmentReader>> {
    let schema = unsafe {
        arrow58::ffi::FFI_ArrowSchema::from_raw(schema_rawptr as *mut arrow58::ffi::FFI_ArrowSchema)
    };
    let files_metadata = unsafe { metadata.as_ref() }
        .map(|metadata| metadata.for_fragment(dataset, fragment_id))
        .transpose()?;
    TOKIO_RT.block_on(open_fragment_reader_impl(
        dataset,
        fragment_id,
        schema,
        files_metadata,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    // record_batch! resolves this crate name at the invocation site.
    use arrow_schema58 as arrow_schema;
    use lance::dataset::NewColumnTransform;
    use lance_table::format::{ExternalFile, RowIdMeta};
    use object_store::memory::InMemory;

    #[test]
    fn lance_io_parallelism_validation() {
        for (value, expected) in [
            ("0", None),
            ("1", Some(1)),
            ("64", Some(64)),
            ("256", Some(256)),
        ] {
            let mut options =
                HashMap::from([(LANCE_IO_PARALLELISM_KEY.to_string(), value.to_string())]);
            assert_eq!(
                extract_lance_io_parallelism(&mut options).unwrap(),
                expected
            );
            assert!(!options.contains_key(LANCE_IO_PARALLELISM_KEY));
        }

        for value in ["257", "-1", "invalid"] {
            let mut options =
                HashMap::from([(LANCE_IO_PARALLELISM_KEY.to_string(), value.to_string())]);
            assert!(extract_lance_io_parallelism(&mut options).is_err());
        }

        assert_eq!(
            extract_lance_io_parallelism(&mut HashMap::new()).unwrap(),
            None
        );
    }

    #[test]
    fn store_fingerprint_is_order_independent_and_ignores_private_keys() {
        let first = HashMap::from([
            ("cloud_provider".to_string(), "aws".to_string()),
            (
                "aws_endpoint".to_string(),
                "https://s3.example.com".to_string(),
            ),
            ("aws_region".to_string(), "us-west-2".to_string()),
        ]);
        let second = HashMap::from([
            ("aws_region".to_string(), "us-west-2".to_string()),
            ("cloud_provider".to_string(), "aws".to_string()),
            (
                "aws_endpoint".to_string(),
                "https://s3.example.com".to_string(),
            ),
        ]);
        assert_eq!(
            store_config_fingerprint(&first),
            store_config_fingerprint(&second)
        );

        let mut with_private_keys = first.clone();
        with_private_keys.insert(CACHE_KEY.to_string(), "provider-cache-key".to_string());
        with_private_keys.insert(LANCE_IO_PARALLELISM_KEY.to_string(), "256".to_string());
        assert_eq!(
            store_config_fingerprint(&first),
            store_config_fingerprint(&with_private_keys)
        );
    }

    #[test]
    fn store_fingerprint_tracks_object_store_and_credential_inputs() {
        let base = HashMap::from([
            ("cloud_provider".to_string(), "aws".to_string()),
            (
                "aws_endpoint".to_string(),
                "https://s3.example.com".to_string(),
            ),
            ("aws_region".to_string(), "us-west-2".to_string()),
            ("aws_access_key_id".to_string(), "access-key".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "secret-key".to_string(),
            ),
            ("aws_role_arn".to_string(), "role-arn".to_string()),
            ("aws_session_name".to_string(), "session".to_string()),
            ("aws_external_id".to_string(), "external-id".to_string()),
            ("aws_credential_refresh_secs".to_string(), "900".to_string()),
            (
                "gcp_target_service_account".to_string(),
                "target@example.iam.gserviceaccount.com".to_string(),
            ),
            (
                "azure_broker_endpoint".to_string(),
                "https://broker.example.com".to_string(),
            ),
            (
                "azure_broker_client_id".to_string(),
                "client-id".to_string(),
            ),
            (
                "azure_broker_tenant_id".to_string(),
                "tenant-id".to_string(),
            ),
            ("lance_aimd_initial_rate".to_string(), "10".to_string()),
            ("lance_aimd_max_rate".to_string(), "100".to_string()),
        ]);
        let base_fingerprint = store_config_fingerprint(&base);
        for key in base.keys() {
            let mut changed = base.clone();
            changed.insert(key.clone(), format!("{}-changed", base[key]));
            assert_ne!(
                base_fingerprint,
                store_config_fingerprint(&changed),
                "{key}"
            );
        }
    }

    #[test]
    fn scheduler_object_store_changes_only_the_requested_capacity() {
        if std::env::var_os("LANCE_IO_THREADS").is_some() {
            return;
        }

        let object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://shared-bucket/dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            64,
            3,
            None,
        ));
        let unchanged = scheduler_object_store(
            &object_store,
            "s3://shared-bucket/dataset",
            &HashMap::new(),
            64,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&object_store, &unchanged));

        let custom = scheduler_object_store(
            &object_store,
            "s3://shared-bucket/dataset",
            &HashMap::new(),
            17,
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&object_store, &custom));
        assert_eq!(custom.block_size(), object_store.block_size());
        assert_eq!(custom.io_parallelism(), 17);
        assert_eq!(custom.store_prefix, object_store.store_prefix);
    }

    #[test]
    fn scheduler_registry_reuses_active_scheduler_without_retaining_it() {
        let object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://shared-bucket/dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            64,
            3,
            None,
        ));
        let key = IoDomainKey {
            store_prefix: object_store.store_prefix.clone(),
            store_config_fingerprint: [0x1234, 0x5678],
        };
        let options = HashMap::new();

        let first = shared_scan_scheduler(
            key.clone(),
            &object_store,
            "s3://shared-bucket/dataset-a",
            &options,
            64,
        )
        .unwrap();
        let second = shared_scan_scheduler(
            key.clone(),
            &object_store,
            "s3://shared-bucket/dataset-b",
            &options,
            64,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let later_value = shared_scan_scheduler(
            key.clone(),
            &object_store,
            "s3://shared-bucket/dataset-c",
            &options,
            1,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&first, &later_value));

        let different_fingerprint = shared_scan_scheduler(
            IoDomainKey {
                store_prefix: object_store.store_prefix.clone(),
                store_config_fingerprint: [0x1234, 0x5679],
            },
            &object_store,
            "s3://shared-bucket/dataset-d",
            &options,
            64,
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&first, &different_fingerprint));

        let scheduler = Arc::downgrade(&first);
        drop(first);
        drop(second);
        drop(later_value);
        assert!(scheduler.upgrade().is_none());

        let replacement = shared_scan_scheduler(
            key,
            &object_store,
            "s3://shared-bucket/dataset-e",
            &options,
            64,
        )
        .unwrap();
        assert!(!Weak::ptr_eq(&scheduler, &Arc::downgrade(&replacement)));
    }

    #[test]
    fn scheduler_registry_get_or_create_is_atomic() {
        let object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://concurrent-bucket/dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            64,
            3,
            None,
        ));
        let key = IoDomainKey {
            store_prefix: object_store.store_prefix.clone(),
            store_config_fingerprint: [0x9abc, 0xdef0],
        };
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let object_store = object_store.clone();
                let key = key.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    shared_scan_scheduler(
                        key,
                        &object_store,
                        "s3://concurrent-bucket/dataset",
                        &HashMap::new(),
                        64,
                    )
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        let schedulers = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        for scheduler in schedulers.iter().skip(1) {
            assert!(Arc::ptr_eq(&schedulers[0], scheduler));
        }
    }
    #[test]
    fn projected_readers_reuse_stable_row_ids_across_sessions() {
        for external in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let uri = directory.path().to_str().unwrap();
            let batch =
                arrow_array58::record_batch!(("i", Int32, [1, 2, 3]), ("j", Int32, [10, 20, 30]))
                    .unwrap();
            let mut manifest = TOKIO_RT.block_on(async {
                let mut dataset = Dataset::write(
                    arrow_array58::RecordBatchIterator::new(
                        vec![Ok(batch.clone())],
                        batch.schema(),
                    ),
                    uri,
                    Some(WriteParams {
                        enable_stable_row_ids: true,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();
                dataset.delete("i = 1").await.unwrap();
                dataset.manifest().clone()
            });
            if external {
                let Some(RowIdMeta::Inline(data)) = &manifest.fragments[0].row_id_meta else {
                    panic!("small fixture should have inline row IDs");
                };
                let size = data.len() as u64;
                std::fs::write(directory.path().join("row_ids.bin"), data).unwrap();
                Arc::make_mut(&mut manifest.fragments)[0].row_id_meta =
                    Some(RowIdMeta::External(ExternalFile {
                        path: "row_ids.bin".into(),
                        offset: 0,
                        size,
                    }));
            }
            let serialized = manifest.serialized();
            let dataset = TOKIO_RT
                .block_on(
                    DatasetBuilder::from_uri(uri)
                        .with_serialized_manifest(&serialized)
                        .unwrap()
                        .with_session(Arc::new(lance::session::Session::new(
                            0,
                            0,
                            Default::default(),
                        )))
                        .load(),
                )
                .unwrap();
            let dataset = BlockingDataset::new(dataset).unwrap();
            let fragment = dataset.inner.manifest().fragments[0].clone();
            let projection = ArrowSchema::from(&dataset.inner.schema().project(&["i"]).unwrap());
            dataset.object_store.io_stats_incremental();
            let reader = TOKIO_RT
                .block_on(async {
                    BlockingFragmentReader::open(
                        &dataset,
                        fragment.clone(),
                        &projection,
                        dataset.fragment_read_config(FragReadConfig::default().with_row_id(true)),
                        None,
                    )
                    .await
                })
                .unwrap();
            let cold_io = dataset.object_store.io_stats_incremental();
            assert_eq!(cold_io.read_iops, 2 + u64::from(external));
            let metadata = reader.get_files_metadata();
            let session = Arc::downgrade(&dataset.inner.session());
            drop(reader);
            drop(dataset);
            assert!(session.upgrade().is_none());

            let dataset = TOKIO_RT
                .block_on(
                    DatasetBuilder::from_uri(uri)
                        .with_serialized_manifest(&serialized)
                        .unwrap()
                        .with_session(Arc::new(lance::session::Session::new(
                            0,
                            0,
                            Default::default(),
                        )))
                        .load(),
                )
                .unwrap();
            let dataset = BlockingDataset::new(dataset).unwrap();
            let metadata = metadata.for_fragment(&dataset, fragment.id).unwrap();
            let projection = ArrowSchema::from(&dataset.inner.schema().project(&["j"]).unwrap());
            dataset.object_store.io_stats_incremental();
            let reader = TOKIO_RT
                .block_on(async {
                    BlockingFragmentReader::open(
                        &dataset,
                        fragment,
                        &projection,
                        dataset.fragment_read_config(FragReadConfig::default().with_row_id(true)),
                        Some(metadata),
                    )
                    .await
                })
                .unwrap();
            let warm_io = dataset.object_store.io_stats_incremental();
            assert_eq!(warm_io.read_iops, 0, "external row IDs: {external}");
            assert_eq!(warm_io.read_bytes, 0, "external row IDs: {external}");
            let batches = TOKIO_RT
                .block_on(async {
                    reader
                        .inner
                        .read_all(3)
                        .await?
                        .buffered(1)
                        .try_collect::<Vec<_>>()
                        .await
                })
                .unwrap();
            assert_eq!(batches.len(), 1);
            let batch = &batches[0];
            assert_eq!(batch.num_rows(), 2);
            let values = batch["j"]
                .as_any()
                .downcast_ref::<arrow_array58::Int32Array>()
                .unwrap();
            assert_eq!(values.values().as_ref(), &[20, 30]);
            let ids = batch["_rowid"]
                .as_any()
                .downcast_ref::<arrow_array58::UInt64Array>()
                .unwrap();
            assert_eq!(ids.values().as_ref(), &[1, 2]);
        }
    }

    #[test]
    fn legacy_reader_opens_without_v2_footers() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().to_str().unwrap();
        let batch = arrow_array58::record_batch!(("i", Int32, [1, 2, 3])).unwrap();
        let projection = batch.schema();
        let dataset = TOKIO_RT
            .block_on(Dataset::write(
                arrow_array58::RecordBatchIterator::new(vec![Ok(batch)], projection.clone()),
                uri,
                Some(lance::dataset::WriteParams {
                    data_storage_version: Some(lance_encoding::version::LanceFileVersion::Legacy),
                    ..Default::default()
                }),
            ))
            .unwrap();
        let dataset = BlockingDataset::new(dataset).unwrap();
        let fragment = dataset.inner.manifest().fragments[0].clone();
        assert!(fragment.files[0].is_legacy_file());
        let reader = TOKIO_RT
            .block_on(async {
                BlockingFragmentReader::open(
                    &dataset,
                    fragment,
                    &projection,
                    dataset.fragment_read_config(FragReadConfig::default()),
                    None,
                )
                .await
            })
            .unwrap();
        assert!(reader.get_files_metadata().files.is_empty());
        assert!(reader.estimate_fragment_column_memory().is_err());
        let batches = TOKIO_RT
            .block_on(async {
                let stream = reader.inner.read_all(3).await?;
                stream.buffered(1).try_collect::<Vec<_>>().await
            })
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }

    #[test]
    fn narrow_projection_retains_footers_for_all_physical_files() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().to_str().unwrap();
        let batch = arrow_array58::record_batch!(("i", Int32, [1, 2, 3])).unwrap();
        let dataset = TOKIO_RT.block_on(async {
            let mut dataset = Dataset::write(
                arrow_array58::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                uri,
                None,
            )
            .await
            .unwrap();
            dataset
                .add_columns(
                    NewColumnTransform::SqlExpressions(vec![("j".into(), "i + 1".into())]),
                    None,
                    None,
                )
                .await
                .unwrap();
            dataset.delete("i = 1").await.unwrap();
            dataset
        });
        let dataset = BlockingDataset::new(dataset).unwrap();
        let fragment = dataset.inner.manifest().fragments[0].clone();
        assert_eq!(fragment.files.len(), 2);
        let projection = ArrowSchema::from(&dataset.inner.schema().project(&["i"]).unwrap());
        let reader = TOKIO_RT
            .block_on(async {
                BlockingFragmentReader::open(
                    &dataset,
                    fragment.clone(),
                    &projection,
                    dataset.fragment_read_config(FragReadConfig::default()),
                    None,
                )
                .await
            })
            .unwrap();
        let metadata = reader.get_files_metadata();
        assert_eq!(reader.number_of_rows().unwrap(), 2);
        assert_eq!(metadata.sorted_deletions.as_ref(), &[0]);
        let previous = TOKIO_RT
            .block_on(
                dataset
                    .inner
                    .checkout_version(dataset.inner.version().version - 1),
            )
            .unwrap();
        let previous = BlockingDataset::new(previous).unwrap();
        assert!(metadata.for_fragment(&previous, fragment.id).is_err());
        drop(previous);
        assert_eq!(metadata.files.len(), 2);
        assert!(metadata.for_fragment(&dataset, fragment.id + 1).is_err());
        assert!(Arc::ptr_eq(
            &metadata.for_fragment(&dataset, fragment.id).unwrap().files,
            &metadata.files,
        ));
        assert!(Arc::ptr_eq(&metadata.files, &reader.files_metadata.files));
        assert!(
            reader
                .estimate_fragment_column_memory()
                .unwrap()
                .iter()
                .all(|estimate| estimate.memory_size > 0)
        );

        // Footers and deletion state can be reused by another Session for the
        // same snapshot after the original Dataset and reader are gone.
        let version = dataset.inner.version().version;
        let previous_session = Arc::downgrade(&dataset.inner.session());
        drop(reader);
        drop(dataset);
        assert!(previous_session.upgrade().is_none());
        let dataset = TOKIO_RT
            .block_on(
                lance::dataset::builder::DatasetBuilder::from_uri(uri)
                    .with_version(version)
                    .with_session(Arc::new(lance::session::Session::new(
                        0,
                        0,
                        Default::default(),
                    )))
                    .load(),
            )
            .unwrap();
        let dataset = BlockingDataset::new(dataset).unwrap();
        assert!(!previous_session.ptr_eq(&Arc::downgrade(&dataset.inner.session())));
        let shared_metadata = metadata.for_fragment(&dataset, fragment.id).unwrap();
        assert!(Arc::ptr_eq(&shared_metadata.files, &metadata.files));
        dataset.object_store.io_stats_incremental();
        let projection = ArrowSchema::from(&dataset.inner.schema().project(&["j"]).unwrap());
        let reader = TOKIO_RT
            .block_on(async {
                BlockingFragmentReader::open(
                    &dataset,
                    fragment,
                    &projection,
                    dataset.fragment_read_config(FragReadConfig::default()),
                    Some(shared_metadata),
                )
                .await
            })
            .unwrap();
        let open_io = dataset.object_store.io_stats_incremental();
        assert_eq!(open_io.read_iops, 0);
        assert_eq!(open_io.read_bytes, 0);
        let batches = TOKIO_RT
            .block_on(async {
                let stream = reader.inner.read_all(3).await?;
                stream.buffered(1).try_collect::<Vec<_>>().await
            })
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "j");
        assert_eq!(reader.map_logical_indices(&[0, 1]), vec![1, 2]);
        assert!(Arc::ptr_eq(
            &metadata.sorted_deletions,
            &reader.files_metadata.sorted_deletions
        ));
    }
}
