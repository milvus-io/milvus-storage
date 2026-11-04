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

// Some shared filesystem helpers are only used by the external table formats.
#![cfg_attr(not(feature = "external-table-formats"), allow(dead_code, unused_imports))]

#[cfg(feature = "external-table-formats")]
#[path = "filesystem/filesystem_opendal.rs"]
mod filesystem_opendal;
#[cfg(feature = "external-table-formats")]
#[path = "iceberg/iceberg_bridgeimpl.rs"]
mod iceberg_bridgeimpl;
#[cfg(feature = "external-table-formats")]
#[path = "iceberg/iceberg_opendal.rs"]
mod iceberg_opendal;
#[cfg(feature = "external-table-formats")]
#[path = "iceberg/iceberg_testutil.rs"]
mod iceberg_testutil;
#[cfg(feature = "external-table-formats")]
#[path = "lance/lance_bridgeimpl.rs"]
mod lance_bridgeimpl;
#[cfg(feature = "external-table-formats")]
#[path = "lance/lance_memory_estimator.rs"]
mod lance_memory_estimator;
#[cfg(feature = "external-table-formats")]
#[path = "lance/lance_object_store.rs"]
mod lance_object_store;
#[cfg(feature = "external-table-formats")]
#[path = "paimon/paimon_bridgeimpl.rs"]
mod paimon_bridgeimpl;
#[cfg(feature = "external-table-formats")]
#[path = "paimon/paimon_opendal.rs"]
mod paimon_opendal;
#[cfg(feature = "external-table-formats")]
#[path = "paimon/paimon_split_serde.rs"]
mod paimon_split_serde;
#[cfg(feature = "external-table-formats")]
#[path = "paimon/paimon_testutil.rs"]
mod paimon_testutil;
#[path = "vortex/predicate_parser.rs"]
mod predicate_parser;
#[cfg(feature = "external-table-formats")]
#[path = "external_formats_ffi.rs"]
mod external_formats_ffi;
#[cfg(feature = "external-table-formats")]
pub use external_formats_ffi::{iceberg_ffi, iceberg_test_ffi, lance_ffi, paimon_ffi, paimon_test_ffi};
#[path = "runtime/rust_runtime.rs"]
mod rust_runtime;
#[cfg(feature = "talon")]
#[path = "talon/talon_bridge.rs"]
mod talon_bridge;
#[path = "vortex/vortex_bridgeimpl.rs"]
mod vortex_bridgeimpl;
#[path = "vortex/vortex_layout_strategy_v2.rs"]
mod vortex_layout_strategy_v2;

#[path = "filesystem/filesystem_c.rs"]
mod filesystem_c;
#[cfg(feature = "external-table-formats")]
#[path = "filesystem/filesystem_object_store.rs"]
mod filesystem_object_store;
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
        fn with_limit(self: &mut VortexScanBuilder, limit: u64);
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

