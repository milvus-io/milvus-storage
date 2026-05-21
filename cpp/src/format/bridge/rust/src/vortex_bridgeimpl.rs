// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{
    Array, ArrayRef as ArrowArrayRef, FixedSizeBinaryArray, FixedSizeListArray, RecordBatch,
    RecordBatchReader, StructArray, UInt8Array, make_array,
};
use arrow_data::ArrayData;
use arrow_data::ffi::FFI_ArrowArray;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

use vortex::ArrayRef;
use vortex::arrow::{FromArrowArray, IntoArrowArray};
use vortex::buffer::Buffer;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::{DType as RustDType, DecimalDType, FieldName, Nullability, PType as RustPType};
use vortex::error::VortexError;
use vortex::expr::Expression;
use vortex::file::{BlockingWriter, OpenOptionsSessionExt};
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::scan::ScanBuilder;
use vortex::stats::Precision;
use vortex::stats::Stat;

use vortex::file::VortexWriteOptions;
use vortex::file::WriteStrategyBuilder;

use crate::VORTEX_RT;
use crate::VORTEX_SESSION;
use crate::filesystem_c::*;
use crate::vortex_ffi as ffi;
use crate::vortex_layout_strategy_v2::build_row_group_strategy;

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

pub(crate) struct ParsedPredicate {
    filter: Option<Expression>,
}

pub(crate) fn parse_predicate_string(
    predicate: &str,
    column_names: Vec<String>,
    column_type_tags: Vec<u8>,
) -> anyhow::Result<Box<ParsedPredicate>> {
    use crate::predicate_parser::ColumnType;
    if column_names.len() != column_type_tags.len() {
        anyhow::bail!(
            "schema length mismatch: {} names vs {} tags",
            column_names.len(),
            column_type_tags.len()
        );
    }
    let schema: Vec<(String, ColumnType)> = column_names
        .into_iter()
        .zip(column_type_tags.into_iter())
        .map(|(n, t)| {
            let ct = match t {
                0 => ColumnType::Int,
                1 => ColumnType::UInt,
                2 => ColumnType::Float,
                3 => ColumnType::Utf8,
                4 => ColumnType::Bool,
                _ => ColumnType::Other,
            };
            (n, ct)
        })
        .collect();
    let filter = crate::predicate_parser::parse_predicate_with_schema(predicate, &schema)?;
    Ok(Box::new(ParsedPredicate { filter }))
}

impl ParsedPredicate {
    pub(crate) fn has_filter(&self) -> bool {
        self.filter.is_some()
    }

    pub(crate) fn take_filter(&mut self) -> Box<Expr> {
        let f = self
            .filter
            .take()
            .expect("take_filter called when no filter is present");
        Box::new(Expr { inner: f })
    }
}

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

#[derive(Clone)]
enum VortexArrayConversion {
    FixedSizeBinary {
        converted_type: DataType,
    },
    List {
        converted_type: DataType,
        child: Box<VortexArrayConversion>,
    },
    Struct {
        converted_type: DataType,
        children: Vec<Option<VortexArrayConversion>>,
    },
}

impl VortexArrayConversion {
    fn converted_type(&self) -> &DataType {
        match self {
            Self::FixedSizeBinary { converted_type }
            | Self::List { converted_type, .. }
            | Self::Struct { converted_type, .. } => converted_type,
        }
    }
}

struct VortexSchemaConversion {
    schema: Schema,
    fields: Vec<Option<VortexArrayConversion>>,
}

fn convert_field_for_vortex(
    field: &Field,
) -> Result<Option<(Arc<Field>, VortexArrayConversion)>, ArrowError> {
    let Some(conversion) = build_vortex_array_conversion(field.data_type())? else {
        return Ok(None);
    };
    Ok(Some((
        Arc::new(
            field
                .clone()
                .with_data_type(conversion.converted_type().clone()),
        ),
        conversion,
    )))
}

fn build_vortex_array_conversion(
    dt: &DataType,
) -> Result<Option<VortexArrayConversion>, ArrowError> {
    match dt {
        DataType::FixedSizeBinary(byte_width) => Ok(Some(VortexArrayConversion::FixedSizeBinary {
            converted_type: DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                *byte_width,
            ),
        })),
        DataType::List(field) => {
            Ok(
                convert_field_for_vortex(field)?.map(|(field, child)| {
                    VortexArrayConversion::List {
                        converted_type: DataType::List(field),
                        child: Box::new(child),
                    }
                }),
            )
        }
        DataType::LargeList(field) => {
            Ok(
                convert_field_for_vortex(field)?.map(|(field, child)| {
                    VortexArrayConversion::List {
                        converted_type: DataType::LargeList(field),
                        child: Box::new(child),
                    }
                }),
            )
        }
        DataType::ListView(field) => {
            Ok(
                convert_field_for_vortex(field)?.map(|(field, child)| {
                    VortexArrayConversion::List {
                        converted_type: DataType::ListView(field),
                        child: Box::new(child),
                    }
                }),
            )
        }
        DataType::LargeListView(field) => Ok(convert_field_for_vortex(field)?.map(
            |(field, child)| VortexArrayConversion::List {
                converted_type: DataType::LargeListView(field),
                child: Box::new(child),
            },
        )),
        DataType::FixedSizeList(field, list_size) => Ok(convert_field_for_vortex(field)?.map(
            |(field, child)| VortexArrayConversion::List {
                converted_type: DataType::FixedSizeList(field, *list_size),
                child: Box::new(child),
            },
        )),
        DataType::Struct(fields) => {
            let mut changed = false;
            let mut child_conversions = Vec::with_capacity(fields.len());
            let mut converted_fields = Vec::with_capacity(fields.len());
            for field in fields {
                match convert_field_for_vortex(field)? {
                    Some((converted_field, conversion)) => {
                        changed = true;
                        child_conversions.push(Some(conversion));
                        converted_fields.push(converted_field);
                    }
                    None => {
                        child_conversions.push(None);
                        converted_fields.push(field.clone());
                    }
                }
            }
            if changed {
                Ok(Some(VortexArrayConversion::Struct {
                    converted_type: DataType::Struct(converted_fields.into()),
                    children: child_conversions,
                }))
            } else {
                Ok(None)
            }
        }
        DataType::Dictionary(_, _) if contains_fixed_size_binary(dt) => {
            Err(unsupported_container_error(
                "Dictionary",
                "FixedSizeBinary nested inside Dictionary is not supported by the Vortex bridge conversion",
            ))
        }
        DataType::Map(_, _) if contains_fixed_size_binary(dt) => Err(unsupported_container_error(
            "Map",
            "FixedSizeBinary nested inside Map is not supported by the Vortex bridge conversion",
        )),
        DataType::Union(_, _) if contains_fixed_size_binary(dt) => {
            Err(unsupported_container_error(
                "Union",
                "FixedSizeBinary nested inside Union is not supported by the Vortex bridge conversion",
            ))
        }
        DataType::RunEndEncoded(_, _) if contains_fixed_size_binary(dt) => {
            Err(unsupported_container_error(
                "RunEndEncoded",
                "FixedSizeBinary nested inside RunEndEncoded is not supported by the Vortex bridge conversion",
            ))
        }
        _ => Ok(None),
    }
}

fn contains_fixed_size_binary(dt: &DataType) -> bool {
    match dt {
        DataType::FixedSizeBinary(_) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => contains_fixed_size_binary(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_fixed_size_binary(field.data_type())),
        DataType::Dictionary(_, value_type) => contains_fixed_size_binary(value_type),
        DataType::Map(field, _) => contains_fixed_size_binary(field.data_type()),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_fixed_size_binary(field.data_type())),
        DataType::RunEndEncoded(run_ends, values) => {
            contains_fixed_size_binary(run_ends.data_type())
                || contains_fixed_size_binary(values.data_type())
        }
        _ => false,
    }
}

fn unsupported_container_error(container: &str, message: &str) -> ArrowError {
    arrow_conversion_error(format!("{container} conversion is unsupported: {message}"))
}

/// Convert schema: replace FixedSizeBinary with FixedSizeList<u8>
fn convert_schema_for_vortex(
    schema: &Schema,
) -> Result<Option<VortexSchemaConversion>, ArrowError> {
    let mut changed = false;
    let mut field_conversions = Vec::with_capacity(schema.fields().len());
    let mut new_fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        match convert_field_for_vortex(field)? {
            Some((converted_field, conversion)) => {
                changed = true;
                field_conversions.push(Some(conversion));
                new_fields.push(converted_field);
            }
            None => {
                field_conversions.push(None);
                new_fields.push(field.clone());
            }
        }
    }

    if changed {
        Ok(Some(VortexSchemaConversion {
            schema: Schema::new_with_metadata(new_fields, schema.metadata().clone()),
            fields: field_conversions,
        }))
    } else {
        Ok(None)
    }
}

fn arrow_conversion_error(message: impl Into<String>) -> ArrowError {
    ArrowError::InvalidArgumentError(message.into())
}

/// Convert FixedSizeBinary array to FixedSizeList<u8> array (zero-copy)
fn convert_fixed_size_binary_to_list(
    array: &FixedSizeBinaryArray,
) -> Result<ArrowArrayRef, ArrowError> {
    let byte_width = array.value_length();
    if byte_width <= 0 {
        return Err(arrow_conversion_error(format!(
            "FixedSizeBinary byte width must be positive, got {byte_width}"
        )));
    }

    // Get the underlying data buffer directly (zero-copy via Arc)
    let data = array.to_data();
    let values_buffer = data.buffers().first().cloned().ok_or_else(|| {
        arrow_conversion_error("FixedSizeBinary array is missing its values buffer")
    })?;
    let values_offset = data.offset() * byte_width as usize;

    // Create UInt8Array from the buffer directly (zero-copy)
    let child_data = ArrayData::builder(DataType::UInt8)
        .len(array.len() * byte_width as usize)
        .offset(values_offset)
        .add_buffer(values_buffer)
        .build()?;
    let child_array = UInt8Array::from(child_data);

    // Create FixedSizeList array
    let list_field = Arc::new(Field::new("item", DataType::UInt8, false));
    let nulls = array.nulls().cloned();

    Ok(Arc::new(FixedSizeListArray::try_new(
        list_field,
        byte_width,
        Arc::new(child_array),
        nulls,
    )?))
}

/// Convert FixedSizeList<u8> array to FixedSizeBinary array (zero-copy)
fn convert_list_to_fixed_size_binary(
    array: &FixedSizeListArray,
    byte_width: i32,
) -> Result<ArrowArrayRef, ArrowError> {
    if byte_width <= 0 {
        return Err(arrow_conversion_error(format!(
            "FixedSizeBinary byte width must be positive, got {byte_width}"
        )));
    }
    if array.value_length() != byte_width {
        return Err(arrow_conversion_error(format!(
            "FixedSizeList<u8> value width {} does not match target FixedSizeBinary width {}",
            array.value_length(),
            byte_width
        )));
    }

    let values = array.values();

    // Get the u8 child array
    let u8_array = values
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| arrow_conversion_error("Expected UInt8 child array"))?;

    // Milvus only creates this FixedSizeList<u8> as a bridge representation for
    // FixedSizeBinary, whose nullability is row-level. Byte-level child nulls
    // cannot be represented when converting back to FixedSizeBinary.
    if u8_array.nulls().is_some() {
        return Err(arrow_conversion_error(format!(
            "FixedSizeList<u8> child UInt8 array must not contain byte-level nulls when converting to FixedSizeBinary; null_count={}",
            u8_array.null_count()
        )));
    }

    // Get the underlying buffer directly (zero-copy via Arc)
    let u8_data = u8_array.to_data();
    let values_buffer =
        u8_data.buffers().first().cloned().ok_or_else(|| {
            arrow_conversion_error("UInt8 child array is missing its values buffer")
        })?;
    let byte_width_usize = byte_width as usize;
    if u8_data.offset() % byte_width_usize != 0 {
        return Err(arrow_conversion_error(format!(
            "FixedSizeList<u8> child offset {} must align to FixedSizeBinary width {}",
            u8_data.offset(),
            byte_width
        )));
    }
    let values_offset = array.offset() + u8_data.offset() / byte_width_usize;

    // Build FixedSizeBinaryArray from the buffer directly (zero-copy)
    let fsb_data = ArrayData::builder(DataType::FixedSizeBinary(byte_width))
        .len(array.len())
        .offset(values_offset)
        .add_buffer(values_buffer)
        .nulls(array.nulls().cloned())
        .build()?;
    Ok(Arc::new(FixedSizeBinaryArray::from(fsb_data)))
}

fn rebuild_array_with_children(
    array: &ArrowArrayRef,
    data_type: DataType,
    child_data: Vec<ArrayData>,
    context: &str,
) -> Result<ArrowArrayRef, ArrowError> {
    let data = array.to_data();
    let converted_data = data
        .into_builder()
        .data_type(data_type)
        .child_data(child_data)
        .build()
        .map_err(|e| arrow_conversion_error(format!("{context}: {e}")))?;
    Ok(make_array(converted_data))
}

fn apply_vortex_array_conversion(
    array: &ArrowArrayRef,
    conversion: &VortexArrayConversion,
) -> Result<ArrowArrayRef, ArrowError> {
    match conversion {
        VortexArrayConversion::FixedSizeBinary { .. } => {
            let fsb_array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| arrow_conversion_error("Expected FixedSizeBinaryArray"))?;
            convert_fixed_size_binary_to_list(fsb_array)
        }
        VortexArrayConversion::List {
            converted_type,
            child,
        } => {
            let data = array.to_data();
            let child_data = data.child_data();
            if child_data.len() != 1 {
                return Err(arrow_conversion_error(format!(
                    "Expected one child array for {}, got {}",
                    array.data_type(),
                    child_data.len()
                )));
            }
            let child_array = make_array(child_data[0].clone());
            let converted_child = apply_vortex_array_conversion(&child_array, child)?;
            rebuild_array_with_children(
                array,
                converted_type.clone(),
                vec![converted_child.to_data()],
                "Failed to build Arrow nested array converted for Vortex",
            )
        }
        VortexArrayConversion::Struct {
            converted_type,
            children,
        } => {
            let data = array.to_data();
            let child_data = data.child_data();
            if child_data.len() != children.len() {
                return Err(arrow_conversion_error(format!(
                    "Struct child count mismatch for {}: array has {}, conversion expects {}",
                    array.data_type(),
                    child_data.len(),
                    children.len()
                )));
            }

            let mut converted_children = Vec::with_capacity(child_data.len());
            for (child, child_conversion) in child_data.iter().zip(children.iter()) {
                match child_conversion {
                    Some(child_conversion) => {
                        let child_array = make_array(child.clone());
                        converted_children.push(
                            apply_vortex_array_conversion(&child_array, child_conversion)?
                                .to_data(),
                        );
                    }
                    None => {
                        converted_children.push(child.clone());
                    }
                }
            }
            rebuild_array_with_children(
                array,
                converted_type.clone(),
                converted_children,
                "Failed to build Arrow struct array converted for Vortex",
            )
        }
    }
}

fn apply_arrow_array_conversion(
    array: &ArrowArrayRef,
    target_type: &DataType,
    conversion: &VortexArrayConversion,
) -> Result<ArrowArrayRef, ArrowError> {
    match conversion {
        VortexArrayConversion::FixedSizeBinary { .. } => {
            let DataType::FixedSizeBinary(byte_width) = target_type else {
                return Err(arrow_conversion_error(format!(
                    "Expected FixedSizeBinary target type for read-back conversion, got {target_type}"
                )));
            };
            let fsl_array = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| arrow_conversion_error("Expected FixedSizeListArray"))?;
            convert_list_to_fixed_size_binary(fsl_array, *byte_width)
        }
        VortexArrayConversion::List { child, .. } => {
            let data = array.to_data();
            let child_data = data.child_data();
            if child_data.len() != 1 {
                return Err(arrow_conversion_error(format!(
                    "Expected one child array for {}, got {}",
                    target_type,
                    child_data.len()
                )));
            }
            let child_array = make_array(child_data[0].clone());
            let target_child_type = match target_type {
                DataType::List(field)
                | DataType::LargeList(field)
                | DataType::ListView(field)
                | DataType::LargeListView(field)
                | DataType::FixedSizeList(field, _) => field.data_type(),
                _ => {
                    return Err(arrow_conversion_error(format!(
                        "Expected list target type for read-back conversion, got {target_type}"
                    )));
                }
            };
            let converted_child =
                apply_arrow_array_conversion(&child_array, target_child_type, child)?;
            rebuild_array_with_children(
                array,
                target_type.clone(),
                vec![converted_child.to_data()],
                "Failed to build Arrow nested array converted from Vortex",
            )
        }
        VortexArrayConversion::Struct { children, .. } => {
            let DataType::Struct(fields) = target_type else {
                return Err(arrow_conversion_error(format!(
                    "Expected struct target type for read-back conversion, got {target_type}"
                )));
            };
            let data = array.to_data();
            let child_data = data.child_data();
            if child_data.len() != fields.len() {
                return Err(arrow_conversion_error(format!(
                    "Struct child count mismatch for read-back conversion: array has {}, target schema expects {} ({})",
                    child_data.len(),
                    fields.len(),
                    target_type
                )));
            }
            if child_data.len() != children.len() {
                return Err(arrow_conversion_error(format!(
                    "Struct child count mismatch for read-back conversion plan: array has {}, conversion expects {}",
                    child_data.len(),
                    children.len()
                )));
            }

            let mut converted_children = Vec::with_capacity(child_data.len());
            for ((child, field), child_conversion) in
                child_data.iter().zip(fields.iter()).zip(children.iter())
            {
                match child_conversion {
                    Some(child_conversion) => {
                        let child_array = make_array(child.clone());
                        converted_children.push(
                            apply_arrow_array_conversion(
                                &child_array,
                                field.data_type(),
                                child_conversion,
                            )?
                            .to_data(),
                        );
                    }
                    None => converted_children.push(child.clone()),
                }
            }
            rebuild_array_with_children(
                array,
                target_type.clone(),
                converted_children,
                "Failed to build Arrow struct array converted from Vortex",
            )
        }
    }
}

/// Convert RecordBatch: replace FixedSizeList<u8> columns with FixedSizeBinary
/// based on the original schema that specifies FixedSizeBinary
fn convert_record_batch_from_vortex(
    batch: &RecordBatch,
    original_schema: &Schema,
    conversion: &VortexSchemaConversion,
) -> Result<RecordBatch, ArrowError> {
    if batch.num_columns() != original_schema.fields().len() {
        return Err(arrow_conversion_error(format!(
            "RecordBatch column count mismatch for read-back conversion: batch has {}, original schema expects {}",
            batch.num_columns(),
            original_schema.fields().len()
        )));
    }
    if batch.num_columns() != conversion.fields.len() {
        return Err(arrow_conversion_error(format!(
            "RecordBatch column count mismatch for read-back conversion plan: batch has {}, conversion expects {}",
            batch.num_columns(),
            conversion.fields.len()
        )));
    }

    let mut new_columns = Vec::with_capacity(batch.num_columns());
    for ((col, orig_field), field_conversion) in batch
        .columns()
        .iter()
        .zip(original_schema.fields().iter())
        .zip(conversion.fields.iter())
    {
        match field_conversion {
            Some(field_conversion) => {
                new_columns.push(apply_arrow_array_conversion(
                    col,
                    orig_field.data_type(),
                    field_conversion,
                )?);
            }
            None => new_columns.push(col.clone()),
        }
    }

    Ok(RecordBatch::try_new(
        Arc::new(original_schema.clone()),
        new_columns,
    )?)
}

/// Convert StructArray: replace FixedSizeBinary columns with FixedSizeList<u8>
fn convert_struct_array_for_vortex(
    struct_array: &StructArray,
    conversion: &VortexSchemaConversion,
) -> Result<StructArray, ArrowError> {
    let array = Arc::new(struct_array.clone()) as ArrowArrayRef;
    let root_conversion = VortexArrayConversion::Struct {
        converted_type: DataType::Struct(conversion.schema.fields().clone()),
        children: conversion.fields.clone(),
    };
    Ok(apply_vortex_array_conversion(&array, &root_conversion)?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| arrow_conversion_error("Expected converted StructArray"))?
        .clone())
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
    Stat::UncompressedSizeInBytes,
];

pub const VORTEX_NON_STATS: &[Stat] = &[Stat::UncompressedSizeInBytes];

const VORTEX_FORMAT_V1: u32 = 1;
const VORTEX_FORMAT_V2: u32 = 2;

pub(crate) struct VortexWriter {
    pub fswrapper_ptr: *mut u8,
    pub path: String,
    pub inner_writer: Option<BlockingWriter<'static, 'static, TokioRuntime>>,
    pub enable_stats: bool,
    pub format_version: u32,
    pub row_group_max_size: u64,
}

pub(crate) unsafe fn open_writer(
    fswrapper_ptr: *mut u8,
    path: &str,
    enable_stats: bool,
    format_version: u32,
    row_group_max_size: u64,
) -> Result<Box<VortexWriter>, Box<dyn std::error::Error>> {
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
        let ffi_array = unsafe { FFI_ArrowArray::from_raw(in_array as *mut FFI_ArrowArray) };

        let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(in_schema as *mut FFI_ArrowSchema) };
        let arrow_schema = Schema::try_from(&ffi_schema)?;

        let arrow_array_data = arrow_array::array::StructArray::from(
            unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema) }
                .map_err(|e| VortexError::from(e))?,
        );

        let (converted_schema, converted_array) = if let Some(conversion) =
            convert_schema_for_vortex(&arrow_schema)?
        {
            let converted_array = convert_struct_array_for_vortex(&arrow_array_data, &conversion)?;
            (conversion.schema, converted_array)
        } else {
            (arrow_schema, arrow_array_data)
        };
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
                build_row_group_strategy(
                    self.row_group_max_size,
                    self.enable_stats,
                    Arc::<[Stat]>::from(stats_options.clone()),
                )
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

        inner_writer
            .push(ArrayRef::from_arrow(&converted_array, false))
            .map_err(|e| Box::new(VortexError::from(e)))?;

        self.inner_writer = Some(inner_writer);
        Ok(())
    }

    pub(crate) unsafe fn close(
        &mut self,
    ) -> Result<crate::vortex_ffi::VortexWriteSummary, Box<dyn std::error::Error>> {
        if let Some(w) = self.inner_writer.take() {
            let summary = w
                .finish()
                .map_err(|e| Box::new(VortexError::from(e)) as Box<dyn std::error::Error>)?;
            let file_size = summary.size();

            // Re-serialize the footer to compute the exact footer region size on disk.
            let footer_size: u64 = summary
                .footer()
                .clone()
                .into_serializer()
                .serialize()
                .map_err(|e| Box::new(VortexError::from(e)) as Box<dyn std::error::Error>)?
                .iter()
                .map(|b| b.len() as u64)
                .sum();

            return Ok(crate::vortex_ffi::VortexWriteSummary {
                file_size,
                footer_size,
            });
        }
        Ok(crate::vortex_ffi::VortexWriteSummary {
            file_size: 0,
            footer_size: 0,
        })
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
            conversion_plan: None,
            row_range: None,
        }))
    }

    pub(crate) fn scan_builder_with_schema(
        &self,
        in_schema: *mut u8,
    ) -> Result<Box<VortexScanBuilder>> {
        let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(in_schema as *mut FFI_ArrowSchema) };
        let original_schema = Arc::new(Schema::try_from(&ffi_schema)?);

        let schema_conversion = convert_schema_for_vortex(original_schema.as_ref())?;
        let converted_schema = schema_conversion
            .as_ref()
            .map(|conversion| Arc::new(conversion.schema.clone()))
            .unwrap_or_else(|| original_schema.clone());

        Ok(Box::new(VortexScanBuilder {
            inner: self.inner.scan()?.with_split_row_indices(false),
            output_schema: Some(converted_schema),
            original_schema: Some(original_schema),
            conversion_plan: schema_conversion,
            row_range: None,
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
        let ranges = self.inner.splits().map_err(|e| VortexError::from(e))?;

        // map each Range<u64> to its end (right-hand side)
        let ends = ranges
            .into_iter()
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

    pub(crate) fn root_layout_encoding(&self) -> String {
        self.inner
            .footer()
            .layout()
            .encoding_id()
            .as_ref()
            .to_string()
    }

    pub(crate) fn row_group_zone_map_count(&self) -> Result<u64> {
        let root = self.inner.footer().layout();
        if root.encoding_id().as_ref() != "milvus.v2_zoned_row_group" {
            return Ok(0);
        }
        Ok(root.child(1)?.row_count())
    }

    pub(crate) fn row_group_zone_map_data_before_zones(&self) -> Result<bool> {
        let root = self.inner.footer().layout();
        if root.encoding_id().as_ref() != "milvus.v2_zoned_row_group" {
            return Ok(false);
        }

        let data_child = root.child(0)?;
        let zones_child = root.child(1)?;
        let mut data_segment_ids = Vec::new();
        collect_layout_segment_ids(&data_child, &mut data_segment_ids)?;
        let mut zones_segment_ids = Vec::new();
        collect_layout_segment_ids(&zones_child, &mut zones_segment_ids)?;

        if data_segment_ids.is_empty() || zones_segment_ids.is_empty() {
            return Ok(false);
        }

        let segments = self.inner.footer().segment_map();
        let max_data_offset = data_segment_ids
            .iter()
            .map(|idx| segments[*idx].offset)
            .max()
            .expect("checked non-empty data segments");
        let min_zones_offset = zones_segment_ids
            .iter()
            .map(|idx| segments[*idx].offset)
            .min()
            .expect("checked non-empty zones segments");

        Ok(max_data_offset < min_zones_offset)
    }
}

fn collect_layout_segment_ids(
    layout: &vortex::layout::LayoutRef,
    out: &mut Vec<usize>,
) -> Result<()> {
    for segment_id in layout.segment_ids() {
        out.push(usize::try_from(*segment_id)?);
    }
    for child in layout.children()? {
        collect_layout_segment_ids(&child, out)?;
    }
    Ok(())
}

pub(crate) unsafe fn open_file(
    fswrapper_ptr: *mut u8,
    path: &str,
    file_size: u64,
    footer_size: u64,
) -> Result<Box<VortexFile>> {
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
        open_options =
            open_options.with_initial_read_size(footer_size as usize + vortex::file::EOF_SIZE);
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
    output_schema: Option<SchemaRef>, // Converted schema for Vortex (FixedSizeList<u8>)
    original_schema: Option<SchemaRef>, // Original schema from user (may contain FixedSizeBinary)
    conversion_plan: Option<VortexSchemaConversion>,
    row_range: Option<Range<u64>>,
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
        self.row_range = Some(row_range_start..row_range_end);
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

        let schema_conversion = convert_schema_for_vortex(original_schema.as_ref())?;
        let converted_schema = schema_conversion
            .as_ref()
            .map(|conversion| Arc::new(conversion.schema.clone()))
            .unwrap_or_else(|| original_schema.clone());

        self.output_schema = Some(converted_schema);
        self.original_schema = Some(original_schema);
        self.conversion_plan = schema_conversion;
        Ok(())
    }
}

struct VortexRecordBatchReader<I> {
    iter: I,
    schema: SchemaRef,
    data_type: DataType,
}

impl<I> std::iter::Iterator for VortexRecordBatchReader<I>
where
    I: Iterator<Item = vortex::error::VortexResult<ArrayRef>>,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|result| {
            result
                .and_then(|chunk| {
                    let arrow = chunk.into_arrow(&self.data_type)?;
                    Ok(RecordBatch::from(arrow.as_struct().clone()))
                })
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
        })
    }
}

impl<I> RecordBatchReader for VortexRecordBatchReader<I>
where
    I: Iterator<Item = vortex::error::VortexResult<ArrayRef>>,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// A wrapper RecordBatchReader that converts FixedSizeList<u8> back to FixedSizeBinary
struct ConvertingRecordBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
    original_schema: SchemaRef,
    plan: Option<VortexSchemaConversion>,
}

impl std::iter::Iterator for ConvertingRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok(batch)) => match self.plan.as_ref() {
                Some(plan) => Some(convert_record_batch_from_vortex(
                    &batch,
                    &self.original_schema,
                    plan,
                )),
                None => Some(Ok(batch)),
            },
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
    let VortexScanBuilder {
        inner,
        output_schema,
        original_schema,
        conversion_plan,
        row_range,
    } = *builder;

    let (vortex_schema, original_schema, plan) =
        match (output_schema, original_schema, conversion_plan) {
            (Some(vs), Some(os), plan) => (vs, os, plan),
            (Some(vs), None, _) => (vs.clone(), vs, None),
            (None, _, _) => {
                let dtype = inner.dtype()?;
                let arrow_schema = Arc::new(dtype.to_arrow_schema()?);
                (arrow_schema.clone(), arrow_schema, None)
            }
        };

    let scan = inner.prepare()?;
    let iter = scan.execute_array_iter(row_range, &*VORTEX_RT)?;
    let data_type = DataType::Struct(vortex_schema.fields().clone());
    let reader = VortexRecordBatchReader {
        iter,
        schema: vortex_schema,
        data_type,
    };

    let final_reader: Box<dyn RecordBatchReader + Send> = if plan.is_some() {
        Box::new(ConvertingRecordBatchReader {
            inner: Box::new(reader),
            original_schema,
            plan,
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

pub fn reset_row_group_zone_map_pruning_stats_ffi() {
    crate::vortex_layout_strategy_v2::reset_row_group_zone_map_pruning_stats();
}

pub fn row_group_zone_map_pruning_stats_ffi() -> crate::vortex_ffi::RowGroupZoneMapPruningStats {
    let (prune_eval_count, pruned_row_group_count) =
        crate::vortex_layout_strategy_v2::row_group_zone_map_pruning_stats();
    crate::vortex_ffi::RowGroupZoneMapPruningStats {
        prune_eval_count,
        pruned_row_group_count,
    }
}
