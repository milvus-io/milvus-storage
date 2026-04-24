// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::TOKIO_RT;

use futures::TryStreamExt;
use futures::stream::StreamExt;
use std::collections::HashMap;
use std::ops::Range;
use std::result::Result as RustResult;
use std::sync::Arc;
use tokio::runtime::Handle;

use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::Array;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::{RecordBatch, RecordBatchReader, StructArray};
use arrow_schema::Schema as ArrowSchema;

use lance::dataset::builder::DatasetBuilder;
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

use crate::lance_ffi::LanceDataStorageFormat;

use lance_index::traits::DatasetIndexExt;
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::utils::stream::ReadBatchFutStream;

use lance::io::ObjectStoreParams;
use lance::session::Session;
use lance_io::object_store::{ObjectStoreRegistry, StorageOptionsProvider};

use crate::gcp_impersonation::{ImpersonatingGcsStoreProvider, REFRESH_OFFSET_SECS};

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = TOKIO_RT.block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        uri: &str,
        version: Option<i32>,
        block_size: Option<i32>,
        index_cache_size_bytes: i64,
        metadata_cache_size_bytes: i64,
        storage_options: HashMap<String, String>,
        serialized_manifest: Option<&[u8]>,
        aws_credentials: Option<object_store::aws::AwsCredentialProvider>,
        s3_credentials_refresh_offset_seconds: Option<u64>,
        // Caller-supplied Session, e.g. one whose ObjectStoreRegistry has the
        // GCS scheme overridden with an ImpersonatingGcsStoreProvider. When
        // None, lance falls back to its own default session (default registry).
        session: Option<Arc<Session>>,
    ) -> Result<Self> {
        let mut store_params = ObjectStoreParams {
            block_size: block_size.map(|size| size as usize),
            storage_options: Some(storage_options.clone()),
            ..Default::default()
        };
        if let Some(offset_seconds) = s3_credentials_refresh_offset_seconds {
            store_params.s3_credentials_refresh_offset =
                std::time::Duration::from_secs(offset_seconds);
        }
        if let Some(creds) = aws_credentials {
            store_params.aws_credentials = Some(creds);
        }
        let params = ReadParams {
            index_cache_size_bytes: index_cache_size_bytes as usize,
            metadata_cache_size_bytes: metadata_cache_size_bytes as usize,
            store_options: Some(store_params),
            ..Default::default()
        };

        let mut builder = DatasetBuilder::from_uri(uri).with_read_params(params);

        if let Some(ver) = version {
            builder = builder.with_version(ver as u64);
        }
        builder = builder.with_storage_options(storage_options);
        if let Some(offset_seconds) = s3_credentials_refresh_offset_seconds {
            builder = builder
                .with_s3_credentials_refresh_offset(std::time::Duration::from_secs(offset_seconds));
        }
        if let Some(session) = session {
            builder = builder.with_session(session);
        }

        if let Some(serialized_manifest) = serialized_manifest {
            builder = builder.with_serialized_manifest(serialized_manifest)?;
        }

        let inner = TOKIO_RT.block_on(builder.load())?;
        Ok(Self { inner })
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
                storage_options: Some(storage_options),
                ..Default::default()
            }),
            None,
            Default::default(),
            false, // TODO: support enable_v2_manifest_paths
        ))?;
        Ok(Self { inner })
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
        Ok(Self { inner })
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = TOKIO_RT.block_on(self.inner.checkout_version(tag))?;
        Ok(Self { inner })
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
        Ok(Self { inner })
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
        Ok(Self { inner })
    }

    pub fn create_tag(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        TOKIO_RT.block_on(
            self.inner
                .tags()
                .create_on_branch(tag, version_number, branch),
        )?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        TOKIO_RT.block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, version: u64, branch: Option<&str>) -> Result<()> {
        TOKIO_RT.block_on(self.inner.tags().update_on_branch(tag, version, branch))?;
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
                    storage_options: Some(write_params),
                    ..Default::default()
                })
                .execute(transaction),
        )?;
        Ok(BlockingDataset { inner: new_dataset })
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

/// Configuration for AWS STS AssumeRole credentials.
struct AssumeRoleConfig {
    role_arn: String,
    session_name: String,
    external_id: String,
    credential_refresh_secs: u64,
}

impl AssumeRoleConfig {
    /// Parse from raw parameters. Returns None if role_arn is empty.
    /// Returns Err if credential_refresh_secs is out of range [900, 43200].
    /// 43200s (12h) is AWS STS `AssumeRole`'s hard upper bound on
    /// `DurationSeconds`, reachable only when the target IAM role's
    /// `MaxSessionDuration` is raised from the 3600s default.
    fn parse(
        role_arn: &str,
        session_name: &str,
        external_id: &str,
        credential_refresh_secs: u64,
    ) -> Result<Option<Self>> {
        if role_arn.is_empty() {
            return Ok(None);
        }
        if credential_refresh_secs < 900 || credential_refresh_secs > 43200 {
            return Err(LanceError::invalid_input(
                format!(
                    "credential_refresh_secs must be in [900, 43200], got {}",
                    credential_refresh_secs
                ),
                snafu::location!(),
            ));
        }
        Ok(Some(Self {
            role_arn: role_arn.to_string(),
            session_name: session_name.to_string(),
            external_id: external_id.to_string(),
            credential_refresh_secs,
        }))
    }

    /// Build AWS credentials by calling STS AssumeRole.
    async fn build_credentials(&self) -> Result<object_store::aws::AwsCredentialProvider> {
        use aws_config::sts::AssumeRoleProvider;
        use lance_io::object_store::providers::aws::AwsCredentialAdapter;

        // session_length = STS token TTL (credential_refresh_secs, e.g. 900s).
        // refresh_offset = how early before expiry AwsCredentialAdapter triggers a
        //   refresh.  Must be strictly less than session_length, otherwise the cache
        //   is considered expired immediately after issuance and every credential
        //   check triggers a new STS call (credential thrashing).
        //   Use a fixed 300s offset to leave enough safety margin.
        const REFRESH_OFFSET_SECS: u64 = 300;

        let mut builder = AssumeRoleProvider::builder(&self.role_arn)
            .session_length(std::time::Duration::from_secs(self.credential_refresh_secs));

        if !self.session_name.is_empty() {
            builder = builder.session_name(&self.session_name);
        }
        if !self.external_id.is_empty() {
            builder = builder.external_id(&self.external_id);
        }

        // build() auto-loads base credentials from the default chain (IAM / IRSA / env vars)
        let assume_role_provider = builder.build().await;

        Ok(Arc::new(AwsCredentialAdapter::new(
            Arc::new(assume_role_provider),
            std::time::Duration::from_secs(REFRESH_OFFSET_SECS),
        )))
    }
}

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
                snafu::location!(),
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
/// match the values the FFI entry points already pass to `BlockingDataset::open`
/// (zero — index/metadata caches are managed by the caller, not us).
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

/// Pick a per-call Session if any cross-tenant credential feature is active.
/// The two supported features are mutually exclusive at the URI level (a URI
/// is either `gs://` or `oss://`), so at most one override is installed per
/// call. Returns `None` when no override is needed, so lance falls back to
/// its default session.
fn pick_custom_session(
    storage_options: &mut HashMap<String, String>,
) -> Result<Option<Arc<Session>>> {
    if let Some(cfg) = GcpImpersonationConfig::extract(storage_options)? {
        return Ok(Some(build_gcp_impersonation_session(&cfg)));
    }
    if storage_options.contains_key("oss_role_arn") {
        return Ok(Some(crate::aliyun_oss_provider::build_aliyun_oss_session()));
    }
    Ok(None)
}

pub fn open_dataset(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);

    // Extract ARN fields from storage_options (set by lance::ToStorageOptions on the C++ side)
    let role_arn = storage_options.remove("aws_role_arn").unwrap_or_default();
    let session_name = storage_options.remove("aws_session_name").unwrap_or_default();
    let external_id = storage_options.remove("aws_external_id").unwrap_or_default();
    let refresh_secs_str = storage_options.remove("aws_credential_refresh_secs").unwrap_or_default();
    let credential_refresh_secs: u64 = refresh_secs_str.parse().unwrap_or(0);

    let assume_role = AssumeRoleConfig::parse(&role_arn, &session_name, &external_id, credential_refresh_secs)?;

    let aws_creds = match &assume_role {
        Some(config) => Some(TOKIO_RT.block_on(config.build_credentials())?),
        None => None,
    };

    // Install a custom lance Session for cross-tenant credential features:
    // - GCP Service Account Impersonation under the `gs` scheme
    //   (lance-io's stock GCS provider only accepts a static bearer token
    //   via `google_storage_token`, no refresh hook — we replace wholesale).
    // - Aliyun per-tenant `role_arn` under the `oss` scheme
    //   (lance-io's stock OSS provider only forwards 4 storage_options keys
    //   — we add `oss_role_arn` / `oss_role_session_name` forwarding).
    // At most one override is installed per call (URIs are mutually exclusive).
    let custom_session = pick_custom_session(&mut storage_options)?;

    // Do not pass credential_refresh_secs as s3_credentials_refresh_offset here:
    // AwsCredentialAdapter already handles refresh internally with REFRESH_OFFSET_SECS.
    // Passing the full session TTL (e.g. 900s) as the offset would cause Lance to
    // consider credentials expired immediately after issuance (credential thrashing).
    let ds = BlockingDataset::open(
        uri, None, None, 0, 0, storage_options, None, aws_creds, None, custom_session,
    )?;
    Ok(Box::new(ds))
}

pub unsafe fn write_dataset(
    uri: &str,
    stream_ptr: *mut u8,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    data_storage_format: LanceDataStorageFormat,
) -> Result<Box<BlockingDataset>> {
    let mut storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    // Symmetric with `open_dataset`: install a custom Session on WriteParams
    // for GCP impersonation or Aliyun role_arn. `WriteParams::store_registry()`
    // reads through Session for object-store creation during write.
    let custom_session = pick_custom_session(&mut storage_options)?;

    let stream_ptr = stream_ptr as *mut FFI_ArrowArrayStream;
    let stream = unsafe { std::ptr::replace(stream_ptr, FFI_ArrowArrayStream::empty()) };
    let reader = ArrowArrayStreamReader::try_new(stream).map_err(|e| LanceError::IO {
        source: Box::new(e),
        location: snafu::location!(),
    })?;

    let lance_file_version = match data_storage_format {
        LanceDataStorageFormat::Legacy => LanceFileVersion::Legacy,
        LanceDataStorageFormat::Stable => LanceFileVersion::V2_0,  // Stable resolves to V2_0
        _ => LanceFileVersion::Legacy,
    };

    let mut write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(lance_file_version),
        session: custom_session,
        ..Default::default()
    };
    write_params.store_params = Some(ObjectStoreParams {
        storage_options: Some(storage_options),
        ..Default::default()
    });

    let inner = TOKIO_RT.block_on(Dataset::write(reader, uri, Some(write_params)))?;
    Ok(Box::new(BlockingDataset { inner }))
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
        let read_batch_fut_stream = TOKIO_RT.block_on(async { self.inner.read_all(batch_size) });

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
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
        let read_batch_fut_stream = TOKIO_RT.block_on(async { self.inner.read_range(range, batch_size) });

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
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
        arrow::ffi::FFI_ArrowSchema::from_raw(schema_rawptr as *mut arrow::ffi::FFI_ArrowSchema)
    };
    let arrow_schema =
        ArrowSchema::try_from(&ffi_schema).map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to convert schema: {}", e.to_string()).into(),
            location: snafu::location!(),
        })?;

    let reader =
        BlockingFragmentReader::open(dataset, fragment, &arrow_schema, FragReadConfig::default())?;
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

    // Lance only exposes fragment schema via FileFragment::schema(), which requires an
    // Arc<Dataset>. The clone here is cheap — Dataset internally wraps state in Arcs, so
    // this is essentially a ref-count bump rather than a deep copy.
    let file_fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment_meta);
    let lance_schema = file_fragment.schema();
    let arrow_schema: ArrowSchema = lance_schema.into();

    let ffi_schema = arrow::ffi::FFI_ArrowSchema::try_from(&arrow_schema)
        .map_err(|e| LanceError::InvalidInput {
            source: format!("Failed to export fragment schema: {}", e).into(),
            location: snafu::location!(),
        })?;

    let out_ptr = out_schema_ptr as *mut arrow::ffi::FFI_ArrowSchema;
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
        arrow::ffi::FFI_ArrowSchema::from_raw(schema_ptr as *mut arrow::ffi::FFI_ArrowSchema)
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
        arrow::ffi::FFI_ArrowSchema::from_raw(schema_ptr as *mut arrow::ffi::FFI_ArrowSchema)
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
