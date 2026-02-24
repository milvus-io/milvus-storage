// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::LANCE_RT;

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

use lance::io::{ObjectStore, ObjectStoreParams};
use lance_io::object_store::{ObjectStoreRegistry, StorageOptionsProvider};

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn drop(uri: &str, storage_options: HashMap<String, String>) -> Result<()> {
        LANCE_RT.block_on(async move {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams {
                storage_options: Some(storage_options.clone()),
                ..Default::default()
            };
            let (object_store, path) =
                ObjectStore::from_uri_and_params(registry, uri, &object_store_params).await?;
            object_store.remove_dir_all(path).await?;
            Ok(())
        })
    }

    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = LANCE_RT.block_on(Dataset::write(reader, uri, params))?;
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
        storage_options_provider: Option<Arc<dyn StorageOptionsProvider>>,
        s3_credentials_refresh_offset_seconds: Option<u64>,
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
        if let Some(provider) = storage_options_provider.clone() {
            store_params.storage_options_provider = Some(provider);
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
        if let Some(provider) = storage_options_provider {
            builder = builder.with_storage_options_provider(provider)
        }
        if let Some(offset_seconds) = s3_credentials_refresh_offset_seconds {
            builder = builder
                .with_s3_credentials_refresh_offset(std::time::Duration::from_secs(offset_seconds));
        }

        if let Some(serialized_manifest) = serialized_manifest {
            builder = builder.with_serialized_manifest(serialized_manifest)?;
        }

        let inner = LANCE_RT.block_on(builder.load())?;
        Ok(Self { inner })
    }

    pub fn commit(
        uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let inner = LANCE_RT.block_on(Dataset::commit(
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
        let version = LANCE_RT.block_on(self.inner.latest_version_id())?;
        Ok(version)
    }

    pub fn list_versions(&self) -> Result<Vec<Version>> {
        let versions = LANCE_RT.block_on(self.inner.versions())?;
        Ok(versions)
    }

    pub fn version(&self) -> Result<Version> {
        Ok(self.inner.version())
    }

    pub fn checkout_version(&mut self, version: u64) -> Result<Self> {
        let inner = LANCE_RT.block_on(self.inner.checkout_version(version))?;
        Ok(Self { inner })
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = LANCE_RT.block_on(self.inner.checkout_version(tag))?;
        Ok(Self { inner })
    }

    pub fn checkout_latest(&mut self) -> Result<()> {
        LANCE_RT.block_on(self.inner.checkout_latest())?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        LANCE_RT.block_on(self.inner.restore())?;
        Ok(())
    }

    pub fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        let tags = LANCE_RT.block_on(self.inner.tags().list())?;
        Ok(tags)
    }

    pub fn list_branches(&self) -> Result<HashMap<String, lance::dataset::refs::BranchContents>> {
        let branches = LANCE_RT.block_on(self.inner.list_branches())?;
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
        let inner = LANCE_RT.block_on(self.inner.create_branch(branch, reference, None))?;
        Ok(Self { inner })
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        LANCE_RT.block_on(self.inner.delete_branch(branch))?;
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
        let inner = LANCE_RT.block_on(self.inner.checkout_version(reference))?;
        Ok(Self { inner })
    }

    pub fn create_tag(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        LANCE_RT.block_on(
            self.inner
                .tags()
                .create_on_branch(tag, version_number, branch),
        )?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        LANCE_RT.block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, version: u64, branch: Option<&str>) -> Result<()> {
        LANCE_RT.block_on(self.inner.tags().update_on_branch(tag, version, branch))?;
        Ok(())
    }

    pub fn get_version(&self, tag: &str) -> Result<u64> {
        let version = LANCE_RT.block_on(self.inner.tags().get_version(tag))?;
        Ok(version)
    }

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        let rows = LANCE_RT.block_on(self.inner.count_rows(filter))?;
        Ok(rows)
    }

    pub fn calculate_data_stats(&self) -> Result<DataStatistics> {
        let stats = LANCE_RT.block_on(Arc::new(self.clone().inner).calculate_data_stats())?;
        Ok(stats)
    }

    pub fn list_indexes(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indexes = LANCE_RT.block_on(self.inner.load_indices())?;
        Ok(indexes)
    }

    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        write_params: HashMap<String, String>,
    ) -> Result<Self> {
        let new_dataset = LANCE_RT.block_on(
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
        let transaction = LANCE_RT.block_on(self.inner.read_transaction())?;
        Ok(transaction)
    }

    pub fn get_table_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(self.inner.metadata().clone())
    }

    pub fn compact(&mut self, options: RustCompactionOptions) -> Result<()> {
        LANCE_RT.block_on(compact_files(&mut self.inner, options, None))?;
        Ok(())
    }

    pub fn cleanup_with_policy(&mut self, policy: CleanupPolicy) -> Result<RemovalStats> {
        Ok(LANCE_RT.block_on(self.inner.cleanup_with_policy(policy))?)
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

        LANCE_RT.block_on(self.inner.append(reader, None))?;
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

fn vec_to_hashmap(keys: Vec<String>, values: Vec<String>) -> HashMap<String, String> {
    keys.into_iter().zip(values.into_iter()).collect()
}

pub fn open_dataset(
    uri: &str,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
) -> Result<Box<BlockingDataset>> {
    let storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
    let ds = BlockingDataset::open(uri, None, None, 0, 0, storage_options, None, None, None)?;
    Ok(Box::new(ds))
}

pub unsafe fn write_dataset(
    uri: &str,
    stream_ptr: *mut u8,
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    data_storage_format: LanceDataStorageFormat,
) -> Result<Box<BlockingDataset>> {
    let storage_options = vec_to_hashmap(storage_options_keys, storage_options_values);
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
        ..Default::default()
    };
    write_params.store_params = Some(ObjectStoreParams {
        storage_options: Some(storage_options),
        ..Default::default()
    });

    let inner = LANCE_RT.block_on(Dataset::write(reader, uri, Some(write_params)))?;
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

        let fragment_reader = LANCE_RT.block_on(fragment.open(&meta_schema.project(&columns)?, read_config))?;

        Ok(Self {
            inner: fragment_reader,
            fragment: fragment,
            projection: projection,
        })
    }

    pub fn number_of_rows(&self) -> Result<u64> {
        Ok(LANCE_RT.block_on(self.fragment.count_rows(None))? as u64)
    }

    pub fn take_as_single_batch(&self, indices: &[u32], out_array: *mut u8) -> Result<()> {
        let ffi_array = LANCE_RT
            .block_on(self.inner.take_as_batch(indices, None))?
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
        let read_batch_fut_stream = LANCE_RT.block_on(self.inner.take(indices, batch_size, None));

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
            Arc::new(self.projection.clone()),
            LANCE_RT.handle().clone(),
        );
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        // # Safety
        // Arrow C stream interface
        unsafe { std::ptr::write(out_stream, ffi_stream) };
        Ok(())
    }

    pub unsafe fn read_all_as_stream(&self, batch_size: u32, out_stream: *mut u8) -> Result<()> {
        let read_batch_fut_stream = LANCE_RT.block_on(async { self.inner.read_all(batch_size) });

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
            Arc::new(self.projection.clone()),
            LANCE_RT.handle().clone(),
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
        let read_batch_fut_stream = LANCE_RT.block_on(async { self.inner.read_range(range, batch_size) });

        let ffi_stream = read_batch_fut_stream?.to_ffi_stream(
            Arc::new(self.projection.clone()),
            LANCE_RT.handle().clone(),
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
        Ok(LANCE_RT.block_on(self.inner.count_rows())?)
    }

    pub unsafe fn open_stream(&self, out_stream: *mut u8) -> Result<()> {
        let stream = LANCE_RT.block_on(self.inner.try_into_stream())?;
        let batches: Vec<RecordBatch> = LANCE_RT.block_on(stream.try_collect::<Vec<_>>())?;

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
    let batch = LANCE_RT.block_on(dataset.inner.take(indices, projection))?;

    let reader = VecBatchReader {
        batches: vec![batch].into_iter(),
        schema: Arc::new(arrow_schema),
    };
    let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream_ptr = out_stream as *mut FFI_ArrowArrayStream;
    unsafe { std::ptr::write(out_stream_ptr, ffi_stream) };
    Ok(())
}
