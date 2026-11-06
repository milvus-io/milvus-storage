// Copyright 2023 Zilliz
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

// Lance, Paimon and Iceberg cxx bridges. They live apart from lib.rs because
// cxxbridge drops #[cfg(feature)] blocks unless --cfg is passed, which Corrosion
// does not do. CMake lists this file only when the formats are enabled.

use crate::iceberg_bridgeimpl::*;
use crate::iceberg_testutil::*;
use crate::lance_bridgeimpl::*;
use crate::paimon_bridgeimpl::*;
use crate::paimon_testutil::*;

#[cxx::bridge(namespace = "milvus_storage::lance::ffi")]
pub mod lance_ffi {
    unsafe extern "C++" {
        include!("milvus-storage/filesystem/ffi/filesystem_internal.h");

        // Lance may share one ScanScheduler across datasets. The scheduler
        // retains the ObjectStore created by the first dataset, so a
        // reader-owned C++ filesystem holder cannot keep that ObjectStore's
        // wrapper address valid after the originating dataset is destroyed.
        // Pass an opaque shared lease instead, allowing the Rust ObjectStore
        // and scheduler to retain the exact FileSystemWrapper they use.
        // Vortex's reader-local holder is sufficient because it does not have
        // this cross-dataset scheduler ownership path.
        #[namespace = ""]
        type FileSystemWrapper;
    }

    /// Lance data storage format
    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum LanceDataStorageFormat {
        Legacy = 0,
        V2_1 = 1,
        V2_2 = 2,
        V2_3 = 3,
    }

    /// Test-only Lance IO statistics with read-and-reset semantics.
    struct LanceIOStats {
        read_iops: u64,
        read_bytes: u64,
    }

    struct LanceColumnMemoryEstimate {
        field_id: i32,
        memory_size: u64,
    }

    extern "Rust" {

        type BlockingDataset;
        /// Test instrumentation for the ObjectStore retained by this dataset.
        ///
        /// When datasets share a ScanScheduler, scheduled reads are tracked by the
        /// ObjectStore captured from the dataset that created that scheduler. This
        /// method therefore must not be interpreted as per-dataset accounting.
        pub fn io_stats_incremental(self: &BlockingDataset) -> LanceIOStats;
        /// Open the latest Dataset when version is zero, or the exact snapshot
        /// identified by a non-zero version.
        pub fn open_dataset(
            filesystem: SharedPtr<FileSystemWrapper>,
            uri: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            version: u64,
        ) -> Result<Box<BlockingDataset>>;
        pub unsafe fn open_dataset_async(
            filesystem: SharedPtr<FileSystemWrapper>,
            uri: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            version: u64,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        pub unsafe fn take_dataset_handle(handle: u64) -> Box<BlockingDataset>;
        /// Resolve only the latest manifest location and return its version.
        /// This does not load or decode the manifest, construct a Dataset, or
        /// initialize a ScanScheduler.
        pub fn resolve_latest_dataset_version(
            filesystem: SharedPtr<FileSystemWrapper>,
            uri: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<u64>;
        pub unsafe fn resolve_latest_dataset_version_async(
            filesystem: SharedPtr<FileSystemWrapper>,
            uri: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        pub unsafe fn write_dataset(
            uri: &str,
            stream_ptr: *mut u8,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            data_storage_format: LanceDataStorageFormat,
            enable_stable_row_ids: bool,
        ) -> Result<Vec<u64>>;
        pub fn delete_rows(
            uri: &str,
            predicate: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<()>;

        pub fn externalize_row_ids_for_testing(uri: &str) -> Result<()>;

        pub fn version(self: &BlockingDataset) -> u64;
        pub fn get_all_fragment_ids(self: &BlockingDataset) -> Vec<u64>;
        pub fn get_fragment_deletion_positions(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<Vec<u64>>;
        pub fn get_fragment_physical_row_count(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<u64>;
        pub fn get_fragment_row_count(dataset: &BlockingDataset, fragment_id: u64) -> Result<u64>;
        pub unsafe fn get_fragment_row_count_async(
            dataset: &BlockingDataset,
            fragment_id: u64,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        type FragmentFilesMetadata;
        pub unsafe fn take_column_memory_sizes(handle: u64) -> Vec<u64>;
        pub unsafe fn get_fragment_schema(
            dataset: &BlockingDataset,
            fragment_id: u64,
            out_schema_ptr: *mut u8,
        ) -> Result<()>;

        type BlockingFragmentReader;
        pub unsafe fn open_fragment_reader(
            dataset: &BlockingDataset,
            fragment_id: u64,
            schema_rawptr: *mut u8,
            metadata: *const FragmentFilesMetadata,
        ) -> Result<Box<BlockingFragmentReader>>;
        pub unsafe fn open_fragment_reader_async(
            dataset: &BlockingDataset,
            fragment_id: u64,
            schema_rawptr: *mut u8,
            metadata: *const FragmentFilesMetadata,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        pub unsafe fn take_fragment_reader_handle(handle: u64) -> Box<BlockingFragmentReader>;

        // BlockingFragmentReader functions
        pub fn get_files_metadata(self: &BlockingFragmentReader) -> Box<FragmentFilesMetadata>;
        // Top-level dataset columns in schema order, using retained footers.
        pub fn estimate_fragment_column_memory(
            self: &BlockingFragmentReader,
        ) -> Result<Vec<LanceColumnMemoryEstimate>>;
        pub unsafe fn estimate_fragment_column_memory_async(
            self: &BlockingFragmentReader,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        pub fn estimate_fragment_memory(self: &BlockingFragmentReader) -> Result<u64>;
        pub fn number_of_rows(self: &BlockingFragmentReader) -> Result<u64>;
        pub unsafe fn take_as_single_batch(
            self: &BlockingFragmentReader,
            indices: &[u32],
            out_array: *mut u8,
        ) -> Result<()>;

        pub unsafe fn take_as_stream(
            self: &BlockingFragmentReader,
            indices: &[u32],
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;

        pub unsafe fn take_async(
            self: &BlockingFragmentReader,
            indices: &[u32],
            out_stream: *mut u8,
            callback: usize,
            context: usize,
        ) -> Result<()>;

        pub unsafe fn read_all_as_stream(
            self: &BlockingFragmentReader,
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;

        pub unsafe fn read_ranges_as_stream(
            self: &BlockingFragmentReader,
            row_range_start: u32,
            row_range_end: u32,
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;
        pub unsafe fn read_ranges_async(
            self: &BlockingFragmentReader,
            row_range_start: u32,
            row_range_end: u32,
            batch_size: u32,
            callback: usize,
            context: usize,
        ) -> Result<()>;
        pub unsafe fn take_record_batch_stream(handle: u64, out_stream: *mut u8);

        // BlockingScanner: dataset-level scan
        type BlockingScanner;
        pub unsafe fn create_scanner(
            dataset: &BlockingDataset,
            schema_ptr: *mut u8,
            batch_size: u32,
        ) -> Result<Box<BlockingScanner>>;

        pub fn count_rows(self: &BlockingScanner) -> Result<u64>;

        pub unsafe fn open_stream(self: &BlockingScanner, out_stream: *mut u8) -> Result<()>;

        // Dataset-level take
        pub unsafe fn dataset_take(
            dataset: &BlockingDataset,
            indices: &[u64],
            schema_ptr: *mut u8,
            out_stream: *mut u8,
        ) -> Result<()>;

    }
}

// FileSystemWrapper is immutable after construction. Its shared Arrow
// filesystem is already used concurrently by C++ format readers.
unsafe impl Send for lance_ffi::FileSystemWrapper {}
unsafe impl Sync for lance_ffi::FileSystemWrapper {}

#[cxx::bridge(namespace = "milvus_storage::paimon::ffi")]
pub mod paimon_ffi {
    unsafe extern "C++" {
        #[namespace = ""]
        #[cxx_name = "FileSystemWrapper"]
        type PaimonFileSystemWrapper = crate::lance_ffi::FileSystemWrapper;
    }

    /// A single physical file or atomic DataSplit returned by Paimon planning.
    struct PaimonFileInfo {
        path: String,
        file_size: u64,
        metadata_json: String,
    }

    extern "Rust" {
        type BlockingPaimonDataSplitReader;

        fn paimon_plan_files(
            filesystem: SharedPtr<PaimonFileSystemWrapper>,
            table_location: &str,
            snapshot_id: i64,
            scan_mode: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<Vec<PaimonFileInfo>>;

        fn paimon_read_deletion_vector(
            filesystem: SharedPtr<PaimonFileSystemWrapper>,
            path: &str,
            data_file_path: &str,
            offset: u64,
            length: u64,
            expected_cardinality: i64,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<Vec<u64>>;

        fn paimon_open_data_split_reader(
            filesystem: SharedPtr<PaimonFileSystemWrapper>,
            metadata_json: &str,
            expected_table_location: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<Box<BlockingPaimonDataSplitReader>>;

        unsafe fn export_schema(
            self: &BlockingPaimonDataSplitReader,
            out_schema_ptr: *mut u8,
        ) -> Result<()>;

        unsafe fn open_stream(
            self: &BlockingPaimonDataSplitReader,
            projected_columns: Vec<String>,
            out_stream_ptr: *mut u8,
        ) -> Result<()>;
    }
}

#[cxx::bridge(namespace = "milvus_storage::paimon::ffi")]
pub mod paimon_test_ffi {
    struct PaimonTestTableInfo {
        snapshot_ids: Vec<i64>,
    }

    extern "Rust" {
        /// Create a repository-owned local Paimon fixture through the public
        /// paimon-rust writer/commit APIs.
        fn paimon_create_test_table(
            table_location: &str,
            num_rows: u64,
            mode: &str,
            deleted_positions: Vec<i64>,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            file_format: &str,
            dimension: u32,
        ) -> Result<PaimonTestTableInfo>;
    }
}

#[cxx::bridge(namespace = "milvus_storage::iceberg::ffi")]
pub mod iceberg_ffi {
    unsafe extern "C++" {
        #[namespace = ""]
        #[cxx_name = "FileSystemWrapper"]
        type IcebergFileSystemWrapper = crate::lance_ffi::FileSystemWrapper;
    }

    /// Per-file info returned from plan_files.
    struct IcebergFileInfo {
        /// Absolute data file URI from Iceberg metadata
        data_file_path: String,
        /// Physical row count (before deletes)
        record_count: u64,
        /// Number of rows deleted from this data file
        num_deleted_rows: u64,
        /// JSON-serialized delete file references.
        /// Empty Vec when no deletes apply to this file.
        delete_metadata_json: Vec<u8>,
    }

    extern "Rust" {
        /// Plan files for a given Iceberg table + snapshot.
        /// Returns one IcebergFileInfo per data file.
        ///
        /// Uses iceberg-rust's TableScan::plan_files() internally
        /// which handles snapshot resolution, delete file association,
        /// sequence number filtering, and partition matching.
        fn iceberg_plan_files(
            filesystem: SharedPtr<IcebergFileSystemWrapper>,
            metadata_location: &str,
            snapshot_id: i64,
            read_option_keys: Vec<String>,
            read_option_values: Vec<String>,
        ) -> Result<Vec<IcebergFileInfo>>;
    }
}

#[cxx::bridge(namespace = "milvus_storage::iceberg::ffi")]
pub mod iceberg_test_ffi {
    /// Info returned after creating a test Iceberg table.
    struct IcebergTestTableInfo {
        /// Path to the metadata.json file
        metadata_location: String,
        /// Snapshot ID of the created snapshot
        snapshot_id: i64,
        /// URI of the data file written
        data_file_uri: String,
    }

    extern "Rust" {
        /// Create a test Iceberg table on local filesystem or cloud storage.
        ///
        /// Schema: id (int64), name (string), value (float64)
        /// Data: id=0..N-1, name="row_0".."row_{N-1}", value=i*1.5
        ///
        /// For cloud storage, pass storage options (e.g., s3.access-key-id, s3.region).
        /// For local filesystem, pass empty storage options.
        fn iceberg_create_test_table(
            table_dir: &str,
            num_rows: u64,
            with_positional_deletes: bool,
            deleted_positions: Vec<i64>,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            // Empty string = no override. Set e.g. to "gs" when physically
            // writing via "s3://" S3-compat to GCS but intending to read via
            // native "gs://" — iceberg-rust otherwise bakes the write-time
            // scheme into every level of metadata and never swaps it on read.
            record_scheme_override: &str,
        ) -> Result<IcebergTestTableInfo>;
    }
}
