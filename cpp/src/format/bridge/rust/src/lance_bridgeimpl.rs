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
    ObjectStore, ObjectStoreProvider, StorageOptions, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use crate::aliyun_oss_provider::{AliyunOssStoreProvider, build_aliyun_oss_session};
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
const IO_DOMAIN_FINGERPRINT_VERSION: &str = "milvus-lance-io-domain-v1";

static LANCE_PROVIDER_CACHE: LazyLock<GlobalLruCache<Arc<dyn ObjectStoreProvider>>> =
    LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));
static LANCE_AZURE_PROVIDER_CACHE: LazyLock<
    GlobalLruCache<Arc<AzureSasStorageOptionsProvider>>,
> = LazyLock::new(|| GlobalLruCache::new(CACHE_CAPACITY));

static IO_DOMAIN_HASHERS: LazyLock<[RandomState; 2]> =
    LazyLock::new(|| [RandomState::new(), RandomState::new()]);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct IoDomainKey {
    store_prefix: String,
    store_config_fingerprint: [u64; 2],
}

static LANCE_IO_SCHEDULERS: LazyLock<Mutex<HashMap<IoDomainKey, Weak<ScanScheduler>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

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
    options.sort_unstable_by(|(left_key, left_value), (right_key, right_value)| {
        left_key
            .cmp(right_key)
            .then_with(|| left_value.cmp(right_value))
    });

    std::array::from_fn(|index| {
        let mut hasher = IO_DOMAIN_HASHERS[index].build_hasher();
        IO_DOMAIN_FINGERPRINT_VERSION.hash(&mut hasher);
        options.len().hash(&mut hasher);
        for (key, value) in &options {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        hasher.finish()
    })
}

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
    Ok((store_params, custom_session))
}

pub fn open_dataset(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    let lance_io_parallelism = extract_lance_io_parallelism(&mut storage_options)?;
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
    let inner = TOKIO_RT.block_on(builder.load())?;
    let dataset = BlockingDataset::new(inner)?;
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

pub unsafe fn write_dataset(
    uri: &str,
    stream_ptr: *mut u8,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    data_storage_format: LanceDataStorageFormat,
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
