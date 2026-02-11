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

mod lance_bridgeimpl;
mod vortex_bridgeimpl;

mod filesystem_c;
use lance_bridgeimpl::*;
use vortex_bridgeimpl::*;

use std::sync::LazyLock;
use vortex::VortexSessionDefault;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// By default, the C++ API uses a current-thread runtime, providing control of the threading
/// model to the C++ side.
///
// TODO(ngates): in the future, we could expose an API for C++ to spawn threads that can drive
//  this runtime.
static VORTEX_RT: LazyLock<CurrentThreadRuntime> =
    LazyLock::new(CurrentThreadRuntime::new);

static VORTEX_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(VORTEX_RT.handle()));

static LANCE_RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

#[cxx::bridge(namespace = "milvus_storage::lance::ffi")]
pub mod lance_ffi {
    /// Lance data storage format
    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum LanceDataStorageFormat {
        Legacy = 0,
        Stable = 1,
    }

    extern "Rust" {

        type BlockingDataset;
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

        pub unsafe fn open_stream(
            self: &BlockingScanner,
            out_stream: *mut u8,
        ) -> Result<()>;

        // Dataset-level take
        pub unsafe fn dataset_take(
            dataset: &BlockingDataset,
            indices: &[u64],
            schema_ptr: *mut u8,
            out_stream: *mut u8,
        ) -> Result<()>;
    }
}  // mod lance_ffi

#[cxx::bridge(namespace = "milvus_storage::vortex::ffi")]
pub mod vortex_ffi {
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

        // writer
        type VortexWriter;
        unsafe fn open_writer(fswrapper_ptr: *mut u8, path: &str, enable_stats: bool) -> Result<Box<VortexWriter>>;
        // unsafe fn write(self: &mut VortexWriter, in_stream: *mut u8) -> Result<()>;
        unsafe fn write(self: &mut VortexWriter, in_schema: *mut u8, in_array: *mut u8) -> Result<()>;
        unsafe fn close(self: &mut VortexWriter) -> Result<()>;

        // reader
        type VortexFile;
        fn row_count(self: &VortexFile) -> u64;
        fn scan_builder(self: &VortexFile) -> Result<Box<VortexScanBuilder>>;
        unsafe fn scan_builder_with_schema(self: &VortexFile, in_schema: *mut u8) -> Result<Box<VortexScanBuilder>>;
        fn splits(self: &VortexFile) -> Result<Vec<u64>>;
        fn uncompressed_sizes(self: &VortexFile) -> Vec<u64>;

        unsafe fn open_file(fswrapper_ptr: *mut u8, path: &str) -> Result<Box<VortexFile>>;

        type VortexScanBuilder;
        fn with_filter(self: &mut VortexScanBuilder, filter: Box<Expr>);
        fn with_filter_ref(self: &mut VortexScanBuilder, filter: &Expr);
        fn with_projection(self: &mut VortexScanBuilder, projection: Box<Expr>);
        fn with_projection_ref(self: &mut VortexScanBuilder, projection: &Expr);
        fn with_row_range(self: &mut VortexScanBuilder, row_range_start: u64, row_range_end: u64);
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
