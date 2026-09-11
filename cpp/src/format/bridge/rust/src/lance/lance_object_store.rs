// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Lance storage integration policy.
//!
//! Dataset reads and mutations intentionally use different storage stacks. Reads use
//! [`FilesystemObjectStore`] over the C++-created Arrow filesystem and accept only
//! credential-free adapter options. Writes and deletes use Lance's native mutation stores and
//! reject dynamic credential modes that cannot preserve the caller's identity safely.
//!
//! Every local or remote reader joins the `ScanScheduler` keyed by `milvus_fs_cache_key`. A
//! configured parallelism of zero becomes Lance's default of 64; it does not disable sharing.
//! While a scheduler is active, the first reader's ObjectStore, AIMD settings, and parallelism
//! remain authoritative. Later readers for the same filesystem reuse them until the last strong
//! scheduler reference is dropped; the registry itself retains only a weak reference.

use std::collections::HashMap;
use std::path::{Component, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use cxx::SharedPtr;
use lance::io::ObjectStoreParams;
use lance::session::Session;
use lance::{Error as LanceError, Result};
use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_DOWNLOAD_RETRY_COUNT, ObjectStore, ObjectStoreProvider,
    ObjectStoreRegistry, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use url::Url;

use crate::TOKIO_RT;
use crate::filesystem_object_store::{FFIPathMapper, FilesystemObjectStore};
use crate::lance_ffi::FileSystemWrapper;

const FS_CACHE_KEY: &str = "milvus_fs_cache_key";
const FS_ROOT_PATH_KEY: &str = "milvus_fs_root_path";
const FS_IS_LOCAL_KEY: &str = "milvus_fs_is_local";
const LANCE_IO_PARALLELISM_KEY: &str = "lance_io_parallelism";
const AIMD_INITIAL_RATE_KEY: &str = "lance_aimd_initial_rate";
const AIMD_MAX_RATE_KEY: &str = "lance_aimd_max_rate";
const MAX_LANCE_IO_PARALLELISM: usize = 256;

pub(crate) fn native_mutation_store_params(
    mut storage_options: HashMap<String, String>,
) -> Result<ObjectStoreParams> {
    // Mutations never use the C++ read adapter. Remove read-only tuning and
    // reject bridge-private credential modes so Lance cannot silently mutate
    // with an ambient identity different from the caller's resolved filesystem.
    storage_options.remove(LANCE_IO_PARALLELISM_KEY);
    const DYNAMIC_CREDENTIAL_KEYS: &[&str] = &[
        "cloud_provider",
        "milvus_fs_cache_key",
        "aws_role_arn",
        "aws_session_name",
        "aws_external_id",
        "aws_credential_refresh_secs",
        "azure_broker_endpoint",
        "azure_broker_client_id",
        "azure_broker_tenant_id",
        "azure_broker_account_name",
        "azure_broker_region",
        "azure_broker_bucket",
        "azure_broker_duration_seconds",
        "azure_broker_request_timeout_ms",
        "gcp_target_service_account",
        "gcp_credential_refresh_secs",
        "oss_role_arn",
        "oss_role_session_name",
        "oss_external_id",
        "oss_credential_refresh_secs",
    ];
    if let Some(key) = DYNAMIC_CREDENTIAL_KEYS
        .iter()
        .find(|key| storage_options.contains_key(**key))
    {
        return Err(LanceError::invalid_input(format!(
            "unsupported Lance native mutation option: {key}"
        )));
    }

    Ok(ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            storage_options,
        ))),
        ..Default::default()
    })
}

#[derive(Clone, Debug)]
pub(crate) struct FFIReadOptions {
    fs_cache_key: String,
    pub(crate) is_local: bool,
    root_path: Option<PathBuf>,
    pub(crate) io_parallelism: usize,
    pub(crate) aimd_options: HashMap<String, String>,
}

impl FFIReadOptions {
    pub(crate) fn parse(mut options: HashMap<String, String>) -> Result<Self> {
        // This is the identity of the cached C++ filesystem, not the dataset URI.
        // The scheduler registry uses it to share one store and limiter domain.
        let fs_cache_key = options
            .remove(FS_CACHE_KEY)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                LanceError::invalid_input(format!("missing required option: {FS_CACHE_KEY}"))
            })?;
        fs_cache_key
            .strip_prefix("fs:")
            .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
            .ok_or_else(|| {
                LanceError::invalid_input(format!(
                    "{FS_CACHE_KEY} must have the form fs:<unsigned integer>"
                ))
            })?
            .parse::<u64>()
            .map_err(|_| {
                LanceError::invalid_input(format!(
                    "{FS_CACHE_KEY} must have the form fs:<unsigned integer>"
                ))
            })?;

        let is_local_value = options.remove(FS_IS_LOCAL_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {FS_IS_LOCAL_KEY}"))
        })?;
        let is_local = is_local_value.parse::<bool>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{FS_IS_LOCAL_KEY} must be 'true' or 'false', got '{is_local_value}'"
            ))
        })?;

        let root_path = options.remove(FS_ROOT_PATH_KEY);
        let root_path = if is_local {
            let root = root_path.filter(|value| !value.is_empty()).ok_or_else(|| {
                LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} is required for a local filesystem"
                ))
            })?;
            let path = PathBuf::from(root);
            if !path.is_absolute()
                || path
                    .components()
                    .any(|component| matches!(component, Component::ParentDir))
            {
                return Err(LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} must be an absolute normalized path"
                )));
            }
            Some(path)
        } else {
            if root_path.is_some() {
                return Err(LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} is valid only for a local filesystem"
                )));
            }
            None
        };

        let parallelism_value = options.remove(LANCE_IO_PARALLELISM_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!(
                "missing required option: {LANCE_IO_PARALLELISM_KEY}"
            ))
        })?;
        let io_parallelism = parallelism_value.parse::<usize>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{LANCE_IO_PARALLELISM_KEY} must be an integer in [0, {MAX_LANCE_IO_PARALLELISM}], got '{parallelism_value}'"
            ))
        })?;
        if io_parallelism > MAX_LANCE_IO_PARALLELISM {
            return Err(LanceError::invalid_input(format!(
                "{LANCE_IO_PARALLELISM_KEY} must be in [0, {MAX_LANCE_IO_PARALLELISM}], got {io_parallelism}"
            )));
        }
        // Sharing is unconditional. Zero is only the compatibility spelling
        // for Lance's default capacity, not a switch back to per-dataset state.
        let io_parallelism = if io_parallelism == 0 {
            DEFAULT_CLOUD_IO_PARALLELISM
        } else {
            io_parallelism
        };

        let initial_rate_value = options.remove(AIMD_INITIAL_RATE_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {AIMD_INITIAL_RATE_KEY}"))
        })?;
        let max_rate_value = options.remove(AIMD_MAX_RATE_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {AIMD_MAX_RATE_KEY}"))
        })?;
        let initial_rate = initial_rate_value.parse::<f64>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must be a positive number, got '{initial_rate_value}'"
            ))
        })?;
        let max_rate = max_rate_value.parse::<f64>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{AIMD_MAX_RATE_KEY} must be a non-negative number, got '{max_rate_value}'"
            ))
        })?;
        if !initial_rate.is_finite() || initial_rate <= 0.0 {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must be a positive finite number"
            )));
        }
        if !max_rate.is_finite() || max_rate < 0.0 {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_MAX_RATE_KEY} must be a non-negative finite number"
            )));
        }
        if max_rate > 0.0 && initial_rate > max_rate {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must not exceed {AIMD_MAX_RATE_KEY}"
            )));
        }

        if let Some((key, _)) = options.into_iter().next() {
            return Err(LanceError::invalid_input(format!(
                "unsupported Lance filesystem read option: {key}"
            )));
        }

        Ok(Self {
            fs_cache_key,
            is_local,
            root_path,
            io_parallelism,
            aimd_options: HashMap::from([
                (AIMD_INITIAL_RATE_KEY.into(), initial_rate_value),
                (AIMD_MAX_RATE_KEY.into(), max_rate_value),
            ]),
        })
    }

    fn path_mapper(&self, dataset_url: &Url) -> Result<FFIPathMapper> {
        // Validate the URI against the already-bound bucket or local root before
        // any path becomes relative to that filesystem.
        if self.is_local {
            let mapper = FFIPathMapper::local(
                self.root_path
                    .as_ref()
                    .expect("validated local read options contain a root path"),
            );
            mapper
                .from_url(dataset_url)
                .map_err(|error| LanceError::invalid_input(error.to_string()))?;
            Ok(mapper)
        } else {
            let authority = dataset_url.host_str().ok_or_else(|| {
                LanceError::invalid_input("remote Lance dataset URI must contain an authority")
            })?;
            let mapper = FFIPathMapper::remote(dataset_url.scheme(), authority);
            mapper
                .from_url(dataset_url)
                .map_err(|error| LanceError::invalid_input(error.to_string()))?;
            Ok(mapper)
        }
    }

    pub(crate) fn object_store_prefix(&self) -> &str {
        &self.fs_cache_key
    }
}

pub(crate) struct FFIObjectStoreProvider {
    filesystem: SharedPtr<FileSystemWrapper>,
    mapper: FFIPathMapper,
    fs_cache_key: String,
    block_size: usize,
    io_parallelism: usize,
    aimd_options: HashMap<String, String>,
}

impl FFIObjectStoreProvider {
    pub(crate) fn new(
        filesystem: SharedPtr<FileSystemWrapper>,
        dataset_url: &Url,
        options: &FFIReadOptions,
    ) -> Result<Self> {
        Ok(Self {
            filesystem,
            mapper: options.path_mapper(dataset_url)?,
            fs_cache_key: options.fs_cache_key.clone(),
            block_size: if options.is_local {
                4 * 1024
            } else {
                64 * 1024
            },
            io_parallelism: options.io_parallelism,
            aimd_options: options.aimd_options.clone(),
        })
    }
}

fn build_unwrapped_ffi_store(
    inner: Arc<dyn object_store::ObjectStore>,
    store_prefix: &str,
    block_size: usize,
    io_parallelism: usize,
    download_retry_count: usize,
) -> ObjectStore {
    // `ObjectStoreRegistry::get_store` owns tracing, custom wrappers, and I/O
    // tracking. Use Lance's public constructor only to initialize its private
    // fields, then restore the raw inner store before returning to the registry.
    // This registered scheme has a pure prefix calculation and, unlike `file`,
    // forces Lance to use the supplied ObjectStore. It therefore initializes
    // private wrapper fields without consulting any native cloud provider or
    // reading provider environment variables.
    let constructor_location =
        Url::parse("file-object-store:///").expect("constant file-object-store URL is valid");
    let mut store = ObjectStore::new(
        inner.clone(),
        constructor_location,
        Some(block_size),
        None,
        false,
        true,
        io_parallelism,
        download_retry_count,
        None,
    );
    store.inner = inner;
    store.store_prefix = store_prefix.to_string();
    store
}

impl std::fmt::Debug for FFIObjectStoreProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FFIObjectStoreProvider(filesystem_c)")
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for FFIObjectStoreProvider {
    async fn new_store(&self, base_path: Url, _params: &ObjectStoreParams) -> Result<ObjectStore> {
        self.mapper
            .from_url(&base_path)
            .map_err(|error| LanceError::invalid_input(error.to_string()))?;

        // Build the read-only adapter first, then apply AIMD exactly once below.
        // The scheduler may retain that wrapped store, making its limiter shared.
        let inner = Arc::new(FilesystemObjectStore::new(
            self.filesystem.clone(),
            self.mapper.clone(),
        )) as Arc<dyn object_store::ObjectStore>;
        let throttle_config = AimdThrottleConfig::from_storage_options(Some(&self.aimd_options))?;
        let inner = if throttle_config.is_disabled() {
            inner
        } else {
            Arc::new(AimdThrottledStore::new(inner, throttle_config)?)
                as Arc<dyn object_store::ObjectStore>
        };

        Ok(build_unwrapped_ffi_store(
            inner,
            &self.fs_cache_key,
            self.block_size,
            self.io_parallelism,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        ))
    }

    fn extract_path(&self, url: &Url) -> Result<object_store::path::Path> {
        self.mapper
            .from_url(url)
            .map_err(|error| LanceError::invalid_input(error.to_string()))
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        self.extract_path(url)?;
        Ok(self.fs_cache_key.clone())
    }
}

pub(crate) fn build_filesystem_session(
    scheme: &str,
    provider: Arc<dyn ObjectStoreProvider>,
) -> Arc<Session> {
    // Start empty deliberately: only the bound scheme may resolve. Falling back
    // to Lance's native registry could rebuild a provider from ambient credentials.
    let registry = ObjectStoreRegistry::empty();
    registry.insert(scheme, provider);
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

// Weak values let the registry reuse active schedulers without extending their
// lifetime after all datasets in the domain have been dropped.
static LANCE_IO_SCHEDULERS: LazyLock<Mutex<HashMap<String, Weak<ScanScheduler>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns the single active ScanScheduler for a filesystem.
///
/// The registry lock covers lookup, construction, and insertion so concurrent opens
/// cannot create competing schedulers for the same filesystem. Only weak references are
/// retained, allowing a later open to create a fresh scheduler after the domain is idle.
///
/// Reusing this scheduler also reuses the ObjectStore captured when it was first
/// created. Therefore, AIMD throttling options are set by the first active reader;
/// later readers for the same filesystem cannot change them until the shared
/// scheduler is dropped and recreated.
pub(crate) fn shared_scan_scheduler(
    fs_cache_key: String,
    object_store: &Arc<ObjectStore>,
) -> Result<Arc<ScanScheduler>> {
    let mut schedulers = LANCE_IO_SCHEDULERS
        .lock()
        .map_err(|_| LanceError::Internal {
            message: "Lance I/O scheduler registry mutex poisoned".into(),
            location: snafu::location!(),
        })?;
    // A hit intentionally ignores later reader settings: the live scheduler
    // carries the first reader's ObjectStore, AIMD controller, and capacity.
    if let Some(scheduler) = schedulers.get(&fs_cache_key).and_then(Weak::upgrade) {
        return Ok(scheduler);
    }

    // Sweep expired weak entries only on a registry miss; active hits stay O(1).
    schedulers.retain(|_, scheduler| scheduler.strong_count() != 0);
    // Construct while holding the registry lock so concurrent first opens
    // cannot select different stores for the same filesystem identity.
    let scheduler = TOKIO_RT.block_on(async {
        ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(object_store),
        )
    });
    schedulers.insert(fs_cache_key, Arc::downgrade(&scheduler));
    Ok(scheduler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn native_mutation_options_reject_dynamic_credentials() {
        for key in [
            "cloud_provider",
            "milvus_fs_cache_key",
            "aws_role_arn",
            "azure_broker_endpoint",
            "gcp_target_service_account",
            "oss_role_arn",
        ] {
            let options = HashMap::from([(key.to_string(), "value".to_string())]);
            let error = native_mutation_store_params(options).err().unwrap();
            assert!(error.to_string().contains(key));
        }

        let static_options = HashMap::from([
            ("aws_access_key_id".to_string(), "access-key".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "secret-key".to_string(),
            ),
        ]);
        assert!(native_mutation_store_params(static_options).is_ok());
    }

    fn valid_filesystem_read_options(is_local: bool) -> HashMap<String, String> {
        let mut options = HashMap::from([
            (FS_CACHE_KEY.into(), "fs:123".into()),
            (FS_IS_LOCAL_KEY.into(), is_local.to_string()),
            (LANCE_IO_PARALLELISM_KEY.into(), "64".into()),
            (AIMD_INITIAL_RATE_KEY.into(), "2000".into()),
            (AIMD_MAX_RATE_KEY.into(), "5000".into()),
        ]);
        if is_local {
            options.insert(FS_ROOT_PATH_KEY.into(), "/data/root".into());
        }
        options
    }

    #[test]
    fn read_options_reject_credentials() {
        let mut options = valid_filesystem_read_options(false);
        options.insert("aws_access_key_id".into(), "secret".into());
        assert!(FFIReadOptions::parse(options).is_err());
    }

    #[test]
    fn read_options_validate_required_identity_and_limits() {
        let mut missing_cache_key = valid_filesystem_read_options(false);
        missing_cache_key.remove(FS_CACHE_KEY);
        assert!(FFIReadOptions::parse(missing_cache_key).is_err());

        let mut invalid_bool = valid_filesystem_read_options(false);
        invalid_bool.insert(FS_IS_LOCAL_KEY.into(), "remote".into());
        assert!(FFIReadOptions::parse(invalid_bool).is_err());

        let mut missing_local_root = valid_filesystem_read_options(true);
        missing_local_root.remove(FS_ROOT_PATH_KEY);
        assert!(FFIReadOptions::parse(missing_local_root).is_err());

        let mut excessive_parallelism = valid_filesystem_read_options(false);
        excessive_parallelism.insert(LANCE_IO_PARALLELISM_KEY.into(), "257".into());
        assert!(FFIReadOptions::parse(excessive_parallelism).is_err());
    }

    #[test]
    fn read_options_zero_parallelism_uses_default() {
        let mut options = valid_filesystem_read_options(false);
        options.insert(LANCE_IO_PARALLELISM_KEY.into(), "0".into());

        let options = FFIReadOptions::parse(options).unwrap();
        assert_eq!(options.io_parallelism, DEFAULT_CLOUD_IO_PARALLELISM);
    }

    #[test]
    fn read_options_accept_valid_remote_and_local_maps() {
        let remote = FFIReadOptions::parse(valid_filesystem_read_options(false)).unwrap();
        assert!(!remote.is_local);
        assert_eq!(remote.root_path, None);
        assert_eq!(remote.io_parallelism, 64);

        let local = FFIReadOptions::parse(valid_filesystem_read_options(true)).unwrap();
        assert!(local.is_local);
        assert_eq!(
            local.root_path.as_deref(),
            Some(std::path::Path::new("/data/root"))
        );
        assert_eq!(local.io_parallelism, 64);
    }

    #[test]
    fn filesystem_provider_uses_configured_parallelism() {
        for (is_local, dataset_uri) in [
            (true, "file:///data/root/table"),
            (false, "s3://bucket/table"),
        ] {
            let mut options = valid_filesystem_read_options(is_local);
            options.insert(LANCE_IO_PARALLELISM_KEY.into(), "17".into());
            let options = FFIReadOptions::parse(options).unwrap();
            let dataset_url = Url::parse(dataset_uri).unwrap();
            let provider =
                FFIObjectStoreProvider::new(SharedPtr::null(), &dataset_url, &options).unwrap();

            assert_eq!(provider.io_parallelism, 17);
        }
    }

    #[test]
    fn filesystem_provider_identity_and_paths_are_strict() {
        let options = FFIReadOptions::parse(valid_filesystem_read_options(false)).unwrap();
        let dataset_url = Url::parse("s3://bucket/table").unwrap();
        let mapper = options.path_mapper(&dataset_url).unwrap();

        assert!(mapper.from_url(&dataset_url).is_ok());
        assert!(
            mapper
                .from_url(&Url::parse("gs://bucket/table").unwrap())
                .is_err()
        );
        assert!(
            mapper
                .from_url(&Url::parse("s3://other/table").unwrap())
                .is_err()
        );
        assert_eq!(options.object_store_prefix(), "fs:123");

        assert!(!options.object_store_prefix().contains("0x"));
    }

    #[test]
    fn filesystem_store_leaves_wrapping_and_tracking_to_registry() {
        let inner = Arc::new(InMemory::new()) as Arc<dyn object_store::ObjectStore>;
        let store = build_unwrapped_ffi_store(
            inner.clone(),
            "fs:123",
            64 * 1024,
            64,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        );

        assert!(Arc::ptr_eq(&store.inner, &inner));
        assert_eq!(store.store_prefix, "fs:123");
        assert_eq!(store.scheme(), "file-object-store");
    }

    #[derive(Debug)]
    struct UnusedObjectStoreProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for UnusedObjectStoreProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            unreachable!("registry membership test never creates a store")
        }
    }

    #[test]
    fn filesystem_session_registry_has_no_native_fallback() {
        let session = build_filesystem_session("s3", Arc::new(UnusedObjectStoreProvider));
        let registry = session.store_registry();

        assert!(registry.get_provider("s3").is_some());
        assert!(registry.get_provider("gs").is_none());
        assert!(registry.get_provider("file").is_none());
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
        let key = "fs:scheduler-reuse".to_string();

        let first = shared_scan_scheduler(key.clone(), &object_store).unwrap();
        let second = shared_scan_scheduler(key.clone(), &object_store).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let later_object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://shared-bucket/other-dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            1,
            3,
            None,
        ));
        let later_value = shared_scan_scheduler(key.clone(), &later_object_store).unwrap();
        assert!(Arc::ptr_eq(&first, &later_value));

        let scheduler = Arc::downgrade(&first);
        drop(first);
        drop(second);
        drop(later_value);
        assert!(scheduler.upgrade().is_none());

        let replacement = shared_scan_scheduler(key, &object_store).unwrap();
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
        let key = "fs:scheduler-atomic".to_string();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let object_store = object_store.clone();
                let key = key.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    shared_scan_scheduler(key, &object_store).unwrap()
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
