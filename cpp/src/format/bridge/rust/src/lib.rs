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

mod aliyun_oss_provider;
mod gcp_impersonation;
mod iceberg_bridgeimpl;
mod iceberg_testutil;
mod lance_bridgeimpl;
mod lance_memory_estimator;
mod predicate_parser;
mod rust_runtime;
mod vortex_bridgeimpl;
mod vortex_layout_strategy_v2;

mod filesystem_c;
use iceberg_bridgeimpl::*;
use iceberg_testutil::*;
use lance_bridgeimpl::*;
use rust_runtime::configure_rust_runtime;
use vortex_bridgeimpl::*;

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutEncodingRef;
use vortex::layout::session::LayoutSessionExt;
use vortex::session::VortexSession;

use crate::vortex_layout_strategy_v2::RowGroupZoneMapLayoutEncoding;

pub(crate) use rust_runtime::TOKIO_RT;

/// Vortex runtime adapter backed by the shared Tokio runtime.
///
/// This is not a second Tokio runtime; it only gives Vortex APIs a
/// `BlockingRuntime` view over `TOKIO_RT`.
static VORTEX_RT: LazyLock<TokioRuntime> =
    LazyLock::new(|| TokioRuntime::new(TOKIO_RT.handle().clone()));

static VORTEX_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_handle(VORTEX_RT.handle());
    session.layouts().register(LayoutEncodingRef::new_ref(
        RowGroupZoneMapLayoutEncoding.as_ref(),
    ));
    session
});

#[cxx::bridge(namespace = "milvus_storage::rust_bridge::ffi")]
pub mod rust_runtime_ffi {
    extern "Rust" {
        fn configure_rust_runtime(worker_threads: u32, max_blocking_threads: u32) -> Result<()>;
    }
}

#[cxx::bridge(namespace = "milvus_storage::lance::ffi")]
pub mod lance_ffi {
    /// Lance data storage format
    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum LanceDataStorageFormat {
        Legacy = 0,
        V2_1 = 1,
        V2_2 = 2,
        V2_3 = 3,
    }

    /// IO statistics returned by io_stats_incremental (read-and-reset)
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
        pub fn io_stats_incremental(self: &BlockingDataset) -> LanceIOStats;
        pub fn open_dataset(
            uri: &str,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
        ) -> Result<Box<BlockingDataset>>;
        pub unsafe fn write_dataset(
            uri: &str,
            stream_ptr: *mut u8,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
            data_storage_format: LanceDataStorageFormat,
        ) -> Result<Box<BlockingDataset>>;

        pub unsafe fn write_stream(self: &mut BlockingDataset, stream_ptr: *mut u8) -> Result<()>;
        pub fn get_all_fragment_ids(self: &BlockingDataset) -> Vec<u64>;
        pub fn dataset_delete_rows(dataset: &mut BlockingDataset, predicate: &str) -> Result<()>;
        pub fn get_fragment_deletion_positions(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<Vec<u64>>;
        pub fn get_fragment_physical_row_count(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<u64>;
        pub fn get_fragment_row_count(dataset: &BlockingDataset, fragment_id: u64) -> Result<u64>;
        // Top-level dataset columns in schema order.
        pub fn estimate_fragment_column_memory(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<Vec<LanceColumnMemoryEstimate>>;
        pub fn estimate_fragment_memory(
            dataset: &BlockingDataset,
            fragment_id: u64,
        ) -> Result<u64>;
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
        ) -> Result<Box<BlockingFragmentReader>>;

        // BlockingFragmentReader functions
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
} // mod lance_ffi

#[cxx::bridge(namespace = "milvus_storage::vortex::ffi")]
pub mod vortex_ffi {
    struct VortexWriteSummary {
        file_size: u64,
        footer_size: u64,
    }

    struct CoalescingWindow {
        distance: u64,
        max_size: u64,
    }

    struct RowGroupZoneMapPruningStats {
        prune_eval_count: u64,
        pruned_row_group_count: u64,
    }

    extern "Rust" {
        type DType;
        // Factory functions for creating DType
        fn dtype_null() -> Box<DType>;
        fn dtype_bool(nullable: bool) -> Box<DType>;
        fn dtype_primitive(ptype: PType, nullable: bool) -> Box<DType>;
        fn dtype_decimal(precision: u8, scale: i8, nullable: bool) -> Box<DType>;
        fn dtype_utf8(nullable: bool) -> Box<DType>;
        fn dtype_binary(nullable: bool) -> Box<DType>;
        unsafe fn from_arrow(ffi_schema: *mut u8, non_nullable: bool) -> Result<Box<DType>>;
        // Methods for DType
        fn to_string(self: &DType) -> String;

        type Scalar;
        fn bool_scalar_new(value: bool) -> Box<Scalar>;
        fn i8_scalar_new(value: i8) -> Box<Scalar>;
        fn i16_scalar_new(value: i16) -> Box<Scalar>;
        fn i32_scalar_new(value: i32) -> Box<Scalar>;
        fn i64_scalar_new(value: i64) -> Box<Scalar>;
        fn u8_scalar_new(value: u8) -> Box<Scalar>;
        fn u16_scalar_new(value: u16) -> Box<Scalar>;
        fn u32_scalar_new(value: u32) -> Box<Scalar>;
        fn u64_scalar_new(value: u64) -> Box<Scalar>;
        fn f32_scalar_new(value: f32) -> Box<Scalar>;
        fn f64_scalar_new(value: f64) -> Box<Scalar>;
        fn string_scalar_new(value: &str) -> Box<Scalar>;
        fn binary_scalar_new(value: &[u8]) -> Box<Scalar>;
        fn cast_scalar(self: &Scalar, dtype: &DType) -> Result<Box<Scalar>>;

        type Expr;
        fn literal(scalar: Box<Scalar>) -> Box<Expr>;
        fn root() -> Box<Expr>;
        fn column(name: String) -> Box<Expr>;
        fn get_item(field: String, child: Box<Expr>) -> Box<Expr>;
        fn not_(child: Box<Expr>) -> Box<Expr>;
        fn is_null(child: Box<Expr>) -> Box<Expr>;
        // binary op
        fn eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn not_eq_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn and_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn or_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn checked_add(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr>;

        type ParsedPredicate;
        fn parse_predicate_string(
            predicate: &str,
            column_names: Vec<String>,
            column_type_tags: Vec<u8>,
        ) -> Result<Box<ParsedPredicate>>;
        fn has_filter(self: &ParsedPredicate) -> bool;
        fn take_filter(self: &mut ParsedPredicate) -> Box<Expr>;

        // writer
        type VortexWriter;
        unsafe fn open_writer(
            fswrapper_ptr: *mut u8,
            path: &str,
            enable_stats: bool,
            format_version: u32,
            row_group_size: u64,
        ) -> Result<Box<VortexWriter>>;
        unsafe fn write(
            self: &mut VortexWriter,
            in_schema: *mut u8,
            in_array: *mut u8,
        ) -> Result<()>;
        unsafe fn flush(self: &mut VortexWriter) -> Result<()>;
        unsafe fn close(self: &mut VortexWriter) -> Result<VortexWriteSummary>;

        // reader
        type VortexFile;
        fn row_count(self: &VortexFile) -> u64;
        unsafe fn get_schema(self: &VortexFile, out_schema: *mut u8) -> Result<()>;
        fn scan_builder(
            self: &VortexFile,
            window: CoalescingWindow,
        ) -> Result<Box<VortexScanBuilder>>;
        unsafe fn scan_builder_with_schema(
            self: &VortexFile,
            in_schema: *mut u8,
        ) -> Result<Box<VortexScanBuilder>>;
        fn splits(self: &VortexFile) -> Result<Vec<u64>>;
        fn uncompressed_sizes(self: &VortexFile) -> Vec<u64>;
        fn root_layout_encoding(self: &VortexFile) -> String;
        fn row_group_zone_map_count(self: &VortexFile) -> Result<u64>;
        fn row_group_zone_map_data_before_zones(self: &VortexFile) -> Result<bool>;
        fn zone_map_segment_ids(self: &VortexFile) -> Result<Vec<u64>>;
        fn footer_byte_range(self: &VortexFile, file_size: u64) -> Result<Vec<u64>>;
        fn segment_bytes(self: &VortexFile, flat_segment_id: u64) -> Result<Vec<u64>>;
        fn field_layout_units(self: &VortexFile, field_name: &str) -> Result<Vec<u64>>;
        fn prune_row_groups(
            self: &VortexFile,
            predicate: &str,
            candidate_row_group_ids: &[u64],
        ) -> Result<Vec<u64>>;
        fn vortex_eof_size() -> u64;

        unsafe fn open_file(
            fswrapper_ptr: *mut u8,
            path: &str,
            file_size: u64,
            footer_size: u64,
        ) -> Result<Box<VortexFile>>;

        type VortexScanBuilder;
        fn with_filter(self: &mut VortexScanBuilder, filter: Box<Expr>);
        fn with_filter_ref(self: &mut VortexScanBuilder, filter: &Expr);
        fn with_projection(self: &mut VortexScanBuilder, projection: Box<Expr>);
        fn with_projection_ref(self: &mut VortexScanBuilder, projection: &Expr);
        fn with_row_indices_projection(self: &mut VortexScanBuilder, field_name: &str);
        fn with_split_row_indices(self: &mut VortexScanBuilder, split_row_indices: bool);
        fn with_row_range(self: &mut VortexScanBuilder, row_range_start: u64, row_range_end: u64);
        fn with_row_ranges(self: &mut VortexScanBuilder, starts: &[u64], ends: &[u64]);
        fn with_include_by_index(self: &mut VortexScanBuilder, include_by_index: &[u64]);
        fn with_limit(self: &mut VortexScanBuilder, limit: usize);
        unsafe fn with_output_schema(
            self: &mut VortexScanBuilder,
            output_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
        fn scan_builder_into_raw_handle(builder: Box<VortexScanBuilder>) -> usize;

        // IO trace
        fn reset_io_trace_ffi();
        fn print_io_trace_ffi();
        fn disable_io_trace_ffi();

        // Row-group zonemap pruning diagnostics
        fn reset_row_group_zone_map_pruning_stats_ffi();
        fn row_group_zone_map_pruning_stats_ffi() -> RowGroupZoneMapPruningStats;
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum PType {
        U8,
        U16,
        U32,
        U64,
        I8,
        I16,
        I32,
        I64,
        F16,
        F32,
        F64,
    }
}

#[cxx::bridge(namespace = "milvus_storage::iceberg::ffi")]
pub mod iceberg_ffi {
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
            metadata_location: &str,
            snapshot_id: i64,
            storage_options_keys: Vec<String>,
            storage_options_values: Vec<String>,
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
