// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::fmt::{Display, Formatter};
use std::ffi::c_void;
use anyhow::Result;

use arrow_array::{Array, ArrayRef as ArrowArrayRef, RecordBatch, RecordBatchReader, FixedSizeBinaryArray, FixedSizeListArray, UInt8Array};
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{FFI_ArrowArrayStream};
use arrow_data::ffi::{FFI_ArrowArray};
use arrow_data::ArrayData;
use arrow_schema::{Field, ArrowError, DataType, Schema, SchemaRef};


use vortex::stats::Precision;
use vortex::stats::Stat;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::buffer::Buffer;
use vortex::file::{BlockingWriter, OpenOptionsSessionExt};
use vortex::scan::ScanBuilder;
use vortex::dtype::{DType as RustDType, DecimalDType, Nullability, PType as RustPType, FieldName};
use vortex::dtype::arrow::FromArrowType;
use vortex::expr::Expression;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::error::VortexError;

use std::collections::VecDeque;
use async_trait::async_trait;
use async_stream::try_stream;
use futures::{StreamExt as FuturesStreamExt, pin_mut};

use vortex::file::VortexWriteOptions;
use vortex::file::WriteStrategyBuilder;
use vortex::IntoArray;
use vortex::arrays::ChunkedArray;
use vortex::layout::LayoutStrategy;
use vortex::layout::LayoutRef as VortexLayoutRef;
use vortex::layout::segments::SegmentSinkRef;
use vortex::layout::sequence::{SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt};
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::collect::CollectStrategy;
use vortex::layout::layouts::struct_::writer::StructStrategy;
use vortex::io::runtime::Handle;
use vortex::ArrayContext;

use crate::filesystem_c::*;
use crate::VORTEX_RT;
use crate::VORTEX_SESSION;
use crate::vortex_ffi as ffi;

/*
 * Type
 */
pub(crate) struct DType {
    pub(crate) inner: RustDType,
}

pub(crate) fn dtype_null() -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Null,
    })
}

pub(crate) fn dtype_bool(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Bool(nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_primitive(ptype: ffi::PType, nullable: bool) -> Box<DType> {
    let vortex_ptype = match ptype {
        ffi::PType::U8 => RustPType::U8,
        ffi::PType::U16 => RustPType::U16,
        ffi::PType::U32 => RustPType::U32,
        ffi::PType::U64 => RustPType::U64,
        ffi::PType::I8 => RustPType::I8,
        ffi::PType::I16 => RustPType::I16,
        ffi::PType::I32 => RustPType::I32,
        ffi::PType::I64 => RustPType::I64,
        ffi::PType::F16 => RustPType::F16,
        ffi::PType::F32 => RustPType::F32,
        ffi::PType::F64 => RustPType::F64,
        _ => unreachable!(),
    };
    Box::new(DType {
        inner: RustDType::Primitive(vortex_ptype, nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_decimal(precision: u8, scale: i8, nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Decimal(
            DecimalDType::new(precision, scale),
            nullability_from_bool(nullable),
        ),
    })
}

pub(crate) fn dtype_utf8(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Utf8(nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_binary(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Binary(nullability_from_bool(nullable)),
    })
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}", self.inner)
    }
}

pub(crate) fn nullability_from_bool(nullable: bool) -> Nullability {
    if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

pub(crate) unsafe fn from_arrow(ffi_schema: *mut u8, non_nullable: bool) -> Result<Box<DType>> {
    let arrow_schema = unsafe { FFI_ArrowSchema::from_raw(ffi_schema as *mut FFI_ArrowSchema) };
    let arrow_dtype = arrow_schema::DataType::try_from(&arrow_schema)?;
    Ok(Box::new(DType {
        inner: RustDType::from_arrow(&Field::new("_", arrow_dtype, !non_nullable)),
    }))
}

/*
 * Scalar
 */
pub(crate) struct Scalar {
    pub(crate) inner: vortex::scalar::Scalar,
}

macro_rules! primitive_scalar_new {
    ($name:ident, $type:ty) => {
        pub(crate) fn $name(value: $type) -> Box<Scalar> {
            Box::new(Scalar {
                inner: vortex::scalar::Scalar::from(value),
            })
        }
    };
}

primitive_scalar_new!(bool_scalar_new, bool); // bool is not primitive but reuse the macro here
primitive_scalar_new!(i8_scalar_new, i8);
primitive_scalar_new!(i16_scalar_new, i16);
primitive_scalar_new!(i32_scalar_new, i32);
primitive_scalar_new!(i64_scalar_new, i64);
primitive_scalar_new!(u8_scalar_new, u8);
primitive_scalar_new!(u16_scalar_new, u16);
primitive_scalar_new!(u32_scalar_new, u32);
primitive_scalar_new!(u64_scalar_new, u64);
primitive_scalar_new!(f32_scalar_new, f32);
primitive_scalar_new!(f64_scalar_new, f64);

pub(crate) fn string_scalar_new(value: &str) -> Box<Scalar> {
    Box::new(Scalar {
        inner: vortex::scalar::Scalar::from(value),
    })
}

pub(crate) fn binary_scalar_new(value: &[u8]) -> Box<Scalar> {
    Box::new(Scalar {
        inner: vortex::scalar::Scalar::from(value),
    })
}

impl Scalar {
    pub(crate) fn cast_scalar(&self, dtype: &DType) -> Result<Box<Scalar>> {
        Ok(Box::new(Scalar {
            inner: self.inner.cast(&dtype.inner)?,
        }))
    }
}

/*
 * expr
 */
pub(crate) struct Expr {
    pub(crate) inner: Expression,
}

pub(crate) fn literal(scalar: Box<Scalar>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::lit(scalar.inner),
    })
}

pub(crate) fn root() -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::root(),
    })
}

pub(crate) fn column(name: String) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::get_item(name, vortex::expr::root()),
    })
}

pub(crate) fn get_item(field: String, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::get_item(field, child.inner),
    })
}

pub(crate) fn not_(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::not(child.inner),
    })
}

pub(crate) fn is_null(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::is_null(child.inner),
    })
}

macro_rules! binary_op {
    ($fn_name:ident $(, $suffix:tt)?) => {
        paste::paste! {
            pub(crate) fn [<$fn_name $($suffix)?>](
                lhs: Box<Expr>,
                rhs: Box<Expr>,
            ) -> Box<Expr> {
                Box::new(Expr {
                    inner: vortex::expr::$fn_name(lhs.inner, rhs.inner),
                })
            }
        }
    };
}

binary_op!(eq);
binary_op!(not_eq, _);
binary_op!(gt);
binary_op!(gt_eq);
binary_op!(lt);
binary_op!(lt_eq);
binary_op!(and, _);
binary_op!(or, _);
binary_op!(checked_add);

pub(crate) fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::select(
            fields.into_iter().map(FieldName::from).collect::<Vec<_>>(),
            child.inner,
        ),
    })
}

/*
 * FixedSizeBinary <-> FixedSizeList<u8> conversion utilities
 *
 * Vortex doesn't support Arrow FixedSizeBinary type directly.
 * We convert FixedSizeBinary(N) <-> FixedSizeList<u8, N> transparently.
 * Both types have identical memory layout, enabling zero-copy conversion.
 */

/// Check if a DataType is FixedSizeBinary
fn is_fixed_size_binary(dt: &DataType) -> bool {
    matches!(dt, DataType::FixedSizeBinary(_))
}

/// Check if schema contains any FixedSizeBinary fields
fn schema_has_fixed_size_binary(schema: &Schema) -> bool {
    schema.fields().iter().any(|f| is_fixed_size_binary(f.data_type()))
}

/// Convert FixedSizeBinary type to FixedSizeList<u8>
fn fixed_size_binary_to_list_type(byte_width: i32) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        byte_width,
    )
}

/// Convert schema: replace FixedSizeBinary with FixedSizeList<u8>
fn convert_schema_for_vortex(schema: &Schema) -> Schema {
    if !schema_has_fixed_size_binary(schema) {
        return schema.clone();
    }

    let new_fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|field| {
            if let DataType::FixedSizeBinary(byte_width) = field.data_type() {
                Arc::new(Field::new(
                    field.name(),
                    fixed_size_binary_to_list_type(*byte_width),
                    field.is_nullable(),
                ))
            } else {
                field.clone()
            }
        })
        .collect();

    Schema::new_with_metadata(new_fields, schema.metadata().clone())
}

/// Convert FixedSizeBinary array to FixedSizeList<u8> array (zero-copy)
fn convert_fixed_size_binary_to_list(array: &FixedSizeBinaryArray) -> ArrowArrayRef {
    let byte_width = array.value_length();

    // Get the underlying data buffer directly (zero-copy via Arc)
    let data = array.to_data();
    let values_buffer = data.buffers()[0].clone();

    // Create UInt8Array from the buffer directly (zero-copy)
    let child_data = ArrayData::builder(DataType::UInt8)
        .len(array.len() * byte_width as usize)
        .add_buffer(values_buffer)
        .build()
        .expect("Failed to build UInt8 ArrayData");
    let child_array = UInt8Array::from(child_data);

    // Create FixedSizeList array
    let list_field = Arc::new(Field::new("item", DataType::UInt8, false));
    let nulls = array.nulls().cloned();

    Arc::new(FixedSizeListArray::new(
        list_field,
        byte_width,
        Arc::new(child_array),
        nulls,
    ))
}

/// Convert FixedSizeList<u8> array to FixedSizeBinary array (zero-copy)
fn convert_list_to_fixed_size_binary(array: &FixedSizeListArray, byte_width: i32) -> ArrowArrayRef {
    let values = array.values();

    // Get the u8 child array
    let u8_array = values.as_any().downcast_ref::<UInt8Array>()
        .expect("Expected UInt8 child array");

    // Get the underlying buffer directly (zero-copy via Arc)
    let u8_data = u8_array.to_data();
    let values_buffer = u8_data.buffers()[0].clone();

    // Build FixedSizeBinaryArray from the buffer directly (zero-copy)
    let mut builder = ArrayData::builder(DataType::FixedSizeBinary(byte_width))
        .len(array.len())
        .add_buffer(values_buffer);

    if let Some(nulls) = array.nulls() {
        builder = builder.null_bit_buffer(Some(nulls.buffer().clone()));
    }

    let fsb_data = builder.build().expect("Failed to build FixedSizeBinary ArrayData");
    Arc::new(FixedSizeBinaryArray::from(fsb_data))
}

/// Convert RecordBatch: replace FixedSizeList<u8> columns with FixedSizeBinary
/// based on the original schema that specifies FixedSizeBinary
fn convert_record_batch_from_vortex(batch: &RecordBatch, original_schema: &Schema) -> RecordBatch {
    if !schema_has_fixed_size_binary(original_schema) {
        return batch.clone();
    }

    let new_columns: Vec<ArrowArrayRef> = batch
        .columns()
        .iter()
        .zip(original_schema.fields().iter())
        .map(|(col, orig_field)| {
            if let DataType::FixedSizeBinary(byte_width) = orig_field.data_type() {
                if let DataType::FixedSizeList(_, _) = col.data_type() {
                    let fsl_array = col.as_any().downcast_ref::<FixedSizeListArray>()
                        .expect("Expected FixedSizeListArray");
                    convert_list_to_fixed_size_binary(fsl_array, *byte_width)
                } else {
                    col.clone()
                }
            } else {
                col.clone()
            }
        })
        .collect();

    RecordBatch::try_new(Arc::new(original_schema.clone()), new_columns)
        .expect("Failed to create converted RecordBatch")
}

/// Convert StructArray: replace FixedSizeBinary columns with FixedSizeList<u8>
fn convert_struct_array_for_vortex(struct_array: &arrow_array::StructArray) -> arrow_array::StructArray {
    let schema = Schema::new(struct_array.fields().clone());
    if !schema_has_fixed_size_binary(&schema) {
        return struct_array.clone();
    }

    let new_fields: Vec<_> = struct_array
        .fields()
        .iter()
        .map(|field| {
            if let DataType::FixedSizeBinary(byte_width) = field.data_type() {
                Arc::new(Field::new(
                    field.name(),
                    fixed_size_binary_to_list_type(*byte_width),
                    field.is_nullable(),
                ))
            } else {
                field.clone()
            }
        })
        .collect();

    let new_columns: Vec<ArrowArrayRef> = struct_array
        .columns()
        .iter()
        .map(|col| {
            if let DataType::FixedSizeBinary(_) = col.data_type() {
                let fsb_array = col.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .expect("Expected FixedSizeBinaryArray");
                convert_fixed_size_binary_to_list(fsb_array)
            } else {
                col.clone()
            }
        })
        .collect();

    arrow_array::StructArray::try_new(new_fields.into(), new_columns, struct_array.nulls().cloned())
        .expect("Failed to create converted StructArray")
}

/*
 * writer
 */

pub const VORTEX_BASIC_STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::NaNCount,
    Stat::UncompressedSizeInBytes
];

pub const VORTEX_NON_STATS: &[Stat] = &[
    Stat::UncompressedSizeInBytes
];

const VORTEX_FORMAT_V1: u32 = 1;
const VORTEX_FORMAT_V2: u32 = 2;

/// Options for byte-size-based row group splitting.
#[derive(Clone)]
struct RowGroupSplitOptions {
    block_size_minimum: u64,
    canonicalize: bool,
}

/// Splits a stream of arrays into row groups based on uncompressed byte size.
///
/// Unlike `RepartitionStrategy` which slices incoming chunks into fixed-row-count
/// pieces via `block_len_multiple`, this strategy keeps incoming chunks intact and
/// flushes all accumulated data as a single row group when the byte threshold is met.
struct RowGroupSplitStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RowGroupSplitOptions,
}

impl RowGroupSplitStrategy {
    fn new<S: LayoutStrategy>(child: S, options: RowGroupSplitOptions) -> Self {
        Self {
            child: Arc::new(child),
            options,
        }
    }
}

/// Simple accumulator that buffers chunks until a byte-size threshold is reached,
/// then drains them as size-limited row groups.
///
/// Each entry stores (chunk, estimated_bytes) because `nbytes()` on a sliced array
/// may return the underlying buffer's total size, not the slice's portion.
struct RowGroupBuffer {
    data: VecDeque<(vortex::ArrayRef, u64)>,
    nbytes: u64,
    block_size_minimum: u64,
}

impl RowGroupBuffer {
    fn new(block_size_minimum: u64) -> Self {
        Self {
            data: VecDeque::new(),
            nbytes: 0,
            block_size_minimum,
        }
    }

    fn push(&mut self, chunk: vortex::ArrayRef) {
        let nbytes = chunk.nbytes() as u64;
        self.nbytes += nbytes;
        self.data.push_back((chunk, nbytes));
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.block_size_minimum
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Drain one row group (~block_size_minimum bytes) from the front of the buffer.
    /// If a chunk would overshoot the limit, slice it and put the remainder back.
    fn drain_one_group(&mut self, dtype: &vortex::dtype::DType) -> Option<vortex::ArrayRef> {
        if self.data.is_empty() {
            return None;
        }

        let mut group = Vec::new();
        let mut group_bytes: u64 = 0;

        while let Some((chunk, est_bytes)) = self.data.pop_front() {
            let chunk_len = chunk.len();
            self.nbytes -= est_bytes;

            if group_bytes + est_bytes <= self.block_size_minimum {
                group_bytes += est_bytes;
                group.push(chunk);
                if group_bytes >= self.block_size_minimum {
                    break;
                }
            } else {
                // This chunk would overshoot — slice to fit
                let space_left = self.block_size_minimum - group_bytes;
                let rows_to_take = if est_bytes > 0 {
                    ((space_left * chunk_len as u64 + est_bytes - 1) / est_bytes) as usize
                } else {
                    chunk_len
                }.max(1).min(chunk_len);

                let left = chunk.slice(0..rows_to_take);
                group.push(left);

                if rows_to_take < chunk_len {
                    let right = chunk.slice(rows_to_take..chunk_len);
                    let right_est = est_bytes * (chunk_len - rows_to_take) as u64 / chunk_len as u64;
                    self.nbytes += right_est;
                    self.data.push_front((right, right_est));
                }
                break;
            }
        }

        let chunked = ChunkedArray::try_new(group, dtype.clone()).ok()?;
        Some(chunked.to_canonical().into_array())
    }
}

#[async_trait]
impl LayoutStrategy for RowGroupSplitStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> vortex::error::VortexResult<VortexLayoutRef> {
        let dtype = stream.dtype().clone();
        let stream = if self.options.canonicalize {
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream.map(|chunk| {
                    let (sequence_id, chunk) = chunk?;
                    vortex::error::VortexResult::Ok((sequence_id, chunk.to_canonical().into_array()))
                }),
            )
            .sendable()
        } else {
            stream
        };

        let dtype_clone = dtype.clone();
        let block_size_minimum = self.options.block_size_minimum;
        let repartitioned_stream = try_stream! {
            let stream = stream.peekable();
            pin_mut!(stream);
            let mut buffer = RowGroupBuffer::new(block_size_minimum);

            // Each input chunk comes from a C++ Write() call via BlockingWriter::push().
            // The stream closes when C++ calls Close() (BlockingWriter::finish()).
            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                // Create a child sequence pointer for this input chunk.
                // Each advance() produces a unique, ordered ID (A.0, A.1, ...)
                // so downstream ChunkedLayoutStrategy can write row groups concurrently
                // while preserving deterministic ordering in the final file.
                let mut sp = sequence_id.descend();

                if chunk.len() > 0 {
                    buffer.push(chunk);
                }

                // Check if this is the last chunk (writer is closing).
                let is_eof = stream.as_mut().peek().await.is_none();

                // Drain row groups from the buffer:
                // - Normally: only drain when accumulated bytes >= block_size_minimum
                // - At EOF: drain all remaining data as size-limited row groups
                while buffer.have_enough() || (is_eof && !buffer.is_empty()) {
                    if let Some(array) = buffer.drain_one_group(&dtype_clone) {
                        yield (sp.advance(), array)
                    } else {
                        break;
                    }
                }
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
                eof,
                handle,
            )
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}

fn build_row_group_strategy(row_group_max_size: u64) -> Arc<dyn LayoutStrategy> {
    let flat = FlatLayoutStrategy { inline_array_node: true, ..Default::default() };
    let compress_flat = CompressingStrategy::new_btrblocks(flat, false);
    let chunked_inner = ChunkedLayoutStrategy::new(compress_flat.clone());
    let validity = CollectStrategy::new(compress_flat);
    let struct_inner = StructStrategy::new(chunked_inner, validity);
    let chunked_outer = ChunkedLayoutStrategy::new(struct_inner);
    // RowGroupSplit is the top-level strategy: it receives the full push stream,
    // accumulates across batch boundaries, and yields row-group-sized arrays.
    // Each yield becomes one chunk in the outer ChunkedLayout.
    Arc::new(RowGroupSplitStrategy::new(
        chunked_outer,
        RowGroupSplitOptions {
            block_size_minimum: row_group_max_size,
            canonicalize: false,
        },
    ))
}

pub(crate) struct VortexWriter {
    pub fswrapper_ptr: *mut u8,
    pub path: String,
    pub inner_writer: Option<BlockingWriter<'static, 'static, TokioRuntime>>,
    pub enable_stats: bool,
    pub format_version: u32,
    pub row_group_max_size: u64,
}

pub(crate) unsafe fn open_writer(fswrapper_ptr: *mut u8, path: &str, enable_stats: bool, format_version: u32, row_group_max_size: u64)
    -> Result<Box<VortexWriter>, Box<dyn std::error::Error>> {
    if format_version == VORTEX_FORMAT_V2 && row_group_max_size == 0 {
        return Err("format_version=V2 requires row_group_max_size > 0".into());
    }
    Ok(Box::new(VortexWriter {
        fswrapper_ptr,
        path: path.to_string(),
        inner_writer: None,
        enable_stats,
        format_version,
        row_group_max_size,
    }))
}

impl VortexWriter {

pub(crate) unsafe fn write(&mut self, in_schema: *mut u8, in_array: *mut u8) -> Result<()> {
    let ffi_array = unsafe {
        FFI_ArrowArray::from_raw(in_array as *mut FFI_ArrowArray)
    };

    let ffi_schema = unsafe {
         FFI_ArrowSchema::from_raw(in_schema as *mut FFI_ArrowSchema)
    };
    let arrow_schema = Schema::try_from(&ffi_schema)?;

    let arrow_array_data = arrow_array::array::StructArray::from(unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema) }
        .map_err(|e| VortexError::from(e))?);

    // Convert FixedSizeBinary columns to FixedSizeList<u8> for Vortex compatibility
    let converted_array = convert_struct_array_for_vortex(&arrow_array_data);
    let converted_schema = convert_schema_for_vortex(&arrow_schema);
    let vortex_schema = RustDType::from_arrow(&converted_schema);

    // lazy init the inner_writer
    if self.inner_writer.is_none() {
        let objw = ObjectStoreWriterCpp::new(self.fswrapper_ptr as *mut c_void, &self.path)
            .map_err(|e| VortexError::from(e))?;

        // stats options
        let stats_options = if self.enable_stats {
            VORTEX_BASIC_STATS.to_vec()
        } else {
            VORTEX_NON_STATS.to_vec()
        };
        let strategy = if self.format_version == VORTEX_FORMAT_V2 {
            build_row_group_strategy(self.row_group_max_size)
        } else {
            WriteStrategyBuilder::new()
                .with_inline_array_node(true)
                .build()
        };

        let blocking_writer = VortexWriteOptions::new(VORTEX_SESSION.clone())
            .with_file_statistics(stats_options)
            .with_strategy(strategy)
            .blocking(&*VORTEX_RT)
            .writer(objw, vortex_schema);

        self.inner_writer = Some(blocking_writer);
    }
    let mut inner_writer = self.inner_writer.take().unwrap();

    inner_writer.push(ArrayRef::from_arrow(&converted_array, false))
        .map_err(|e| Box::new(VortexError::from(e)))?;

    self.inner_writer = Some(inner_writer);
    Ok(())
}

pub(crate) unsafe fn close(&mut self) -> Result<crate::vortex_ffi::VortexWriteSummary, Box<dyn std::error::Error>> {
    if let Some(w) = self.inner_writer.take() {
        let summary = w.finish().map_err(|e| Box::new(VortexError::from(e)) as Box<dyn std::error::Error>)?;
        let file_size = summary.size();

        // Re-serialize the footer to compute the exact footer region size on disk.
        let footer_size: u64 = summary.footer().clone()
            .into_serializer()
            .serialize()
            .map_err(|e| Box::new(VortexError::from(e)) as Box<dyn std::error::Error>)?
            .iter()
            .map(|b| b.len() as u64)
            .sum();

        return Ok(crate::vortex_ffi::VortexWriteSummary { file_size, footer_size });
    }
    Ok(crate::vortex_ffi::VortexWriteSummary { file_size: 0, footer_size: 0 })
}

}

pub(crate) struct VortexFile {
    inner: vortex::file::VortexFile,
}

impl VortexFile {
    pub(crate) fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    pub(crate) fn scan_builder(&self) -> Result<Box<VortexScanBuilder>> {
        Ok(Box::new(VortexScanBuilder {
            inner: self.inner.scan()?.with_split_row_indices(false),
            output_schema: None,
            original_schema: None,
        }))
    }

    pub(crate) fn scan_builder_with_schema(&self, in_schema: *mut u8) -> Result<Box<VortexScanBuilder>> {
        let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(in_schema as *mut FFI_ArrowSchema) };
        let original_schema = Arc::new(Schema::try_from(&ffi_schema)?);

        // Convert schema for Vortex (FixedSizeBinary -> FixedSizeList<u8>)
        let converted_schema = Arc::new(convert_schema_for_vortex(&original_schema));

        Ok(Box::new(VortexScanBuilder {
            inner: self.inner.scan()?.with_split_row_indices(false),
            output_schema: Some(converted_schema),
            original_schema: Some(original_schema),
        }))
    }

    pub(crate) unsafe fn get_schema(&self, out_schema: *mut u8) -> Result<()> {
        let dtype = self.inner.dtype();
        let arrow_schema = dtype.to_arrow_schema()?;
        let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe { std::ptr::write(out_schema as *mut FFI_ArrowSchema, ffi_schema) };
        Ok(())
    }

    pub(crate) fn splits(&self) -> Result<Vec<u64>> {
        // get the Vec<Range<u64>> from the inner file
        let ranges = self.inner.splits()
            .map_err(|e| VortexError::from(e))?;

        // map each Range<u64> to its end (right-hand side)
        let ends = ranges.into_iter()
            .map(|r: Range<u64>| r.end - r.start)
            .collect::<Vec<u64>>();

        Ok(ends)
    }

    pub(crate) fn uncompressed_sizes(&self) -> Vec<u64> {
        let stats_opt = self.inner.footer().statistics();
        
        match stats_opt {
            None => vec![],
            Some(arc_slice) => {
                let mut sizes = Vec::with_capacity(arc_slice.len());
                arc_slice.iter().for_each(|stats| {
                    let byte_size = stats
                        .get_as::<u64>(Stat::UncompressedSizeInBytes, &RustPType::U64.into())
                        .unwrap_or_else(|| Precision::inexact(u64::MAX))
                        .into_inexact();

                    let byte_size = match byte_size.as_inexact() {
                        Some(v) => v,
                        None => u64::MAX,
                    };
                    
                    sizes.push(byte_size);
                });
                
                sizes
            }
        }
    }

}

pub(crate) unsafe fn open_file(
    fswrapper_ptr: *mut u8,
    path: &str,
    file_size: u64,
    footer_size: u64) -> Result<Box<VortexFile>> {

    let read_source = ObjectStoreReadSourceCpp::new(fswrapper_ptr as *mut c_void, path, file_size)
        .map_err(VortexError::from)?;
    let mut open_options = VORTEX_SESSION.open_options();
    if file_size > 0 {
        // Use pre-known file size to skip the S3 HEAD request that size() would trigger.
        open_options = open_options.with_file_size(file_size);
    }
    if footer_size > 0 {
        // Use cached footer size as initial read size to read entire footer in one IO.
        // Add EOF_SIZE for the EOF marker (version + postscript length + magic) that follows the footer.
        open_options = open_options
            .with_initial_read_size(footer_size as usize + vortex::file::EOF_SIZE);
    }
    let file = VORTEX_RT.block_on(async move {
        open_options
            .open(read_source)
            .await
            .map_err(VortexError::from)
    })?;

    Ok(Box::new(VortexFile { inner: file }))
}

pub(crate) struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
    output_schema: Option<SchemaRef>,          // Converted schema for Vortex (FixedSizeList<u8>)
    original_schema: Option<SchemaRef>,        // Original schema from user (may contain FixedSizeBinary)
}

impl VortexScanBuilder {
    pub(crate) fn with_filter(&mut self, filter: Box<Expr>) {
        take_mut::take(&mut self.inner, |inner| inner.with_filter(filter.inner));
    }

    pub(crate) fn with_filter_ref(&mut self, filter: &Expr) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_filter(filter.inner.clone())
        });
    }

    pub(crate) fn with_projection(&mut self, filter: Box<Expr>) {
        take_mut::take(&mut self.inner, |inner| inner.with_projection(filter.inner));
    }

    pub(crate) fn with_projection_ref(&mut self, filter: &Expr) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_projection(filter.inner.clone())
        });
    }

    pub(crate) fn with_row_range(&mut self, row_range_start: u64, row_range_end: u64) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_row_range(row_range_start..row_range_end)
        });
    }

    pub(crate) fn with_include_by_index(&mut self, include_by_index: &[u64]) {
        let selection =
            vortex::scan::Selection::IncludeByIndex(Buffer::copy_from(include_by_index));
        take_mut::take(&mut self.inner, |inner| {
            inner
                .with_selection(selection)
                // For point queries, increase concurrency so that all natural splits
                // fit within the buffered() window. This ensures all IO requests are
                // visible to the IO driver at once, reducing from ~5 IO rounds to 2.
                // Default is 4 per-thread; with 16 cores this gives buffered(64) which
                // is too small for files with many chunks (e.g. 370 embedding chunks).
                // Setting 128 gives buffered(128*16=2048), covering any realistic file.
                .with_concurrency(128)
        });
    }

    pub(crate) fn with_limit(&mut self, limit: usize) {
        take_mut::take(&mut self.inner, |inner| inner.with_limit(limit));
    }

    pub(crate) unsafe fn with_output_schema(&mut self, output_schema: *mut u8) -> Result<()> {
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(output_schema as *mut FFI_ArrowSchema) };
        let original_schema = Arc::new(Schema::try_from(&ffi_schema)?);

        // Convert schema for Vortex (FixedSizeBinary -> FixedSizeList<u8>)
        let converted_schema = Arc::new(convert_schema_for_vortex(&original_schema));

        self.output_schema = Some(converted_schema);
        self.original_schema = Some(original_schema);
        Ok(())
    }
}

/// A wrapper RecordBatchReader that converts FixedSizeList<u8> back to FixedSizeBinary
struct ConvertingRecordBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
    original_schema: SchemaRef,
}

impl std::iter::Iterator for ConvertingRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok(batch)) => {
                let converted = convert_record_batch_from_vortex(&batch, &self.original_schema);
                Some(Ok(converted))
            }
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }
}

impl RecordBatchReader for ConvertingRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.original_schema.clone()
    }
}

/// # Safety
///
/// out_stream should be properly aligned according to the Arrow C stream interface and valid for write.
pub(crate) unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<()> {
    let (vortex_schema, original_schema) = match (builder.output_schema, builder.original_schema) {
        (Some(vs), Some(os)) => (vs, os),
        (Some(vs), None) => (vs.clone(), vs),
        (None, _) => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = Arc::new(dtype.to_arrow_schema()?);
            (arrow_schema.clone(), arrow_schema)
        }
    };

    let reader = builder.inner.into_record_batch_reader(vortex_schema, &*VORTEX_RT)?;

    // Wrap with converting reader if schema has FixedSizeBinary
    let final_reader: Box<dyn RecordBatchReader + Send> = if schema_has_fixed_size_binary(&original_schema) {
        Box::new(ConvertingRecordBatchReader {
            inner: Box::new(reader),
            original_schema,
        })
    } else {
        Box::new(reader)
    };

    let stream = FFI_ArrowArrayStream::new(final_reader);
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C stream interface
    unsafe { std::ptr::write(out_stream, stream) };
    Ok(())
}

pub fn reset_io_trace_ffi() {
    crate::filesystem_c::reset_io_trace();
}

pub fn print_io_trace_ffi() {
    crate::filesystem_c::print_io_trace();
}

pub fn disable_io_trace_ffi() {
    crate::filesystem_c::disable_io_trace();
}

