// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::fmt::{Display, Formatter};
use std::io::Write;
use anyhow::Result;
use bytes::BytesMut;
use futures::stream::{TryStreamExt, FuturesUnordered};

use object_store::{MultipartUpload, ObjectStore, PutPayload, PutResult};
use object_store::local::LocalFileSystem;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{FFI_ArrowArrayStream};
use arrow_array::{Array as ArrowArray};
use arrow_data::ffi::{FFI_ArrowArray};
use arrow_schema::{Field, ArrowError, DataType, Schema, SchemaRef};


use vortex::pipeline::vec;
use vortex::stats::Precision;
use vortex::stats::{Stat, StatsProvider};
use vortex::{ArrayRef};
use vortex::arrow::{FromArrowArray, IntoArrowArray};
use vortex::buffer::Buffer;
use vortex::file::{VortexOpenOptions, Writer as VortexBlockWriter, VortexWriteOptions, BlockingWriter};
use vortex::scan::ScanBuilder;
use vortex::scan::arrow::RecordBatchIteratorAdapter;
use vortex::dtype::{DType as RustDType, DecimalDType, Nullability, PType as RustPType, FieldName};
use vortex::dtype::arrow::FromArrowType;
use vortex::expr::ExprRef;
use vortex::io::runtime::current::{CurrentThreadRuntime};
use vortex::io::runtime::BlockingRuntime;
use vortex::error::VortexError;

use crate::ffi;
use crate::RUNTIME;

/*
 * Blocking ObjectStoreWriter
 */
pub struct ObjectStoreWriter {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
    put_result: Option<PutResult>,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWriter {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> Result<Self, VortexError> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            upload,
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            put_result: None,
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }
}

impl ObjectStoreWriter {
    pub async fn shutdown(&mut self) -> Result<()> {
        // flush remaining buffer
        self.flush()?;

        if !self.buffer.is_empty() {
            let payload = std::mem::take(&mut self.buffer).freeze();
            self.upload
                .put_part(PutPayload::from_bytes(payload))
                .await?;
        }

        // complete the multipart upload
        let put_result = self.upload.complete().await?;
        self.put_result = Some(put_result);
        Ok(())
    }
}

impl Drop for ObjectStoreWriter {
    fn drop(&mut self) {
        // Use the crate-global CurrentThreadRuntime instead of creating a new Tokio runtime.
        // Dereference the LazyLock to call methods on the inner runtime instance.
        RUNTIME.block_on(|_h| async {
            let _ = self.shutdown().await;
        });

    }
}

impl Write for ObjectStoreWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    // Use the crate-global runtime instead of creating a fresh Tokio runtime.
        self.buffer.extend_from_slice(buf);
        let parts = FuturesUnordered::new();

        // If the buffer is full
        if self.buffer.len() > BUFFER_SIZE {
            // Split off chunks while buffer is larger than CHUNKS_SIZE
            while self.buffer.len() > CHUNK_SIZE {
                let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
                let part_fut = self.upload.put_part(PutPayload::from_bytes(payload));

                parts.push(part_fut);
            }
        }

        RUNTIME.block_on(|_| parts.try_collect::<Vec<_>>())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let parts = FuturesUnordered::new();

        while self.buffer.len() > CHUNK_SIZE {
            let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
            let part_fut = self.upload.put_part(PutPayload::from_bytes(payload));

            parts.push(part_fut);
        }

        RUNTIME.block_on(|_| parts.try_collect::<Vec<_>>())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

        Ok(())
    }
}

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
    pub(crate) inner: ExprRef,
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
 * reader
 */
pub(crate) struct ObjectStoreWrapper {
    pub inner: Arc<dyn object_store::ObjectStore>,
}

pub(crate) fn open_object_store(
    ostype: &str, 
    endpoint: &str,
    access_key_id: &str,
    secret_access_key: &str,
    region: &str,
    bucket_name: &str) -> Result<Box<ObjectStoreWrapper>, Box<dyn std::error::Error>> {

    let boxed_osw = match ostype {
        "local" => {
            // LocalFileSystem requires the endpoint to be a valid absolute path
            // more info check the properties of `fs.root_path`
            let store = match LocalFileSystem::new_with_prefix(endpoint) {
                Ok(store) => store,
                Err(err) => return Err(Box::new(err)),
            };

            let osw: Box<ObjectStoreWrapper> = Box::new(ObjectStoreWrapper {
                 inner : Arc::new(store)
            });
            osw
        }
        "remote" => {
            let store = match AmazonS3Builder::new()
                .with_endpoint(endpoint)
                .with_bucket_name(bucket_name)
                .with_region(region)
                .with_access_key_id(access_key_id)
                .with_secret_access_key(secret_access_key)
                .with_allow_http(true)
                .build()
            {
                Ok(store) => store,
                Err(err) => return Err(Box::new(err)),
            };
            let osw: Box<ObjectStoreWrapper> = Box::new(ObjectStoreWrapper {
                 inner : Arc::new(store)
            });
            osw
        }
        _ => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("unsupported object store type: {}", ostype)))),
    };
    
    Ok(boxed_osw)
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

pub(crate) struct VortexWriter {
    pub objstore: Arc<dyn object_store::ObjectStore>,
    pub path: Path,
    pub inner_writer: Option<BlockingWriter<'static, CurrentThreadRuntime>>,
    pub enable_stats: bool,
}

pub(crate) fn open_writer(object_store_wrapper: &Box<ObjectStoreWrapper>, path: &str, enable_stats: bool) 
    -> Result<Box<VortexWriter>, Box<dyn std::error::Error>> {
    Ok(Box::new(VortexWriter { 
        objstore: object_store_wrapper.inner.clone(),
        path: Path::parse(path).map_err(|e| Box::new(e))?,
        inner_writer : None,
        enable_stats : enable_stats,
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
    let arrow_schema =  Schema::try_from(&ffi_schema)?;
    let vortex_schema = RustDType::from_arrow(&arrow_schema);

    let arrow_array_data = arrow_array::array::StructArray::from(unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema) }
        .map_err(|e| VortexError::from(e))?);
    
    // lazy init the inner_writer
    if self.inner_writer.is_none() {
        let objw: ObjectStoreWriter = RUNTIME.block_on(|_h| async {
            ObjectStoreWriter::new(self.objstore.clone(), &self.path).await
        }).map_err(|e| VortexError::from(e))?;
        
        // stats options
        let stats_options = if self.enable_stats {
            VORTEX_BASIC_STATS.to_vec()
        } else {
            VORTEX_NON_STATS.to_vec()
        };

        let blocking_writer = vortex::file::VortexWriteOptions::default()
            .with_file_statistics(stats_options)
            .blocking::<CurrentThreadRuntime>()
            .writer(objw, vortex_schema);

        self.inner_writer = Some(blocking_writer);
    }
    let mut inner_writer = self.inner_writer.take().unwrap();

    inner_writer.push(ArrayRef::from_arrow(&arrow_array_data, false))
        .map_err(|e| Box::new(VortexError::from(e)))?;

    self.inner_writer = Some(inner_writer);
    Ok(())
}

pub(crate) unsafe fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(mut w) = self.inner_writer.take() {
        w.finish().map_err(|e| Box::new(VortexError::from(e)))?;
    }
    Ok(())
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
            inner: self.inner.scan()?,
            output_schema: None,
        }))
    }

    pub(crate) fn splits(&self) -> Result<Vec<u64>> {
        // get the Vec<Range<u64>> from the inner file
        let ranges = self.inner.splits()
            .map_err(|e| VortexError::from(e))?;

        // map each Range<u64> to its end (right-hand side)
        let ends = ranges.into_iter()
            .map(|r: Range<u64>| r.end)
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

pub(crate) fn open_file(
    object_store_wrapper: &Box<ObjectStoreWrapper>,
    path: &str) -> Result<Box<VortexFile>> {

    let file = RUNTIME.block_on(|h| {
        VortexOpenOptions::new()
            .with_handle(h)
            .open_object_store(&object_store_wrapper.inner, path)
    })?;

    Ok(Box::new(VortexFile { inner: file }))
}

pub(crate) struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
    output_schema: Option<SchemaRef>,
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
        take_mut::take(&mut self.inner, |inner| inner.with_selection(selection));
    }

    pub(crate) fn with_limit(&mut self, limit: usize) {
        take_mut::take(&mut self.inner, |inner| inner.with_limit(limit));
    }

    pub(crate) unsafe fn with_output_schema(&mut self, output_schema: *mut u8) -> Result<()> {
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(output_schema as *mut FFI_ArrowSchema) };
        self.output_schema = Some(Arc::new(Schema::try_from(&ffi_schema)?));
        Ok(())
    }
}

/// # Safety
///
/// out_stream should be properly aligned according to the Arrow C stream interface and valid for write.
pub(crate) unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<()> {
    let schema = match builder.output_schema {
        Some(schema) => schema,
        None => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = dtype.to_arrow_schema()?;
            Arc::new(arrow_schema)
        }
    };
    let reader = builder.inner.into_record_batch_reader(schema, &*RUNTIME)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C stream interface
    unsafe { std::ptr::write(out_stream, stream) };
    Ok(())
}

trait ThreadsafeCloneableReaderTrait: RecordBatchReader + Send + 'static {
    fn clone_boxed(&self) -> Box<dyn ThreadsafeCloneableReaderTrait>;
}

impl<T> ThreadsafeCloneableReaderTrait for T
where
    T: RecordBatchReader + Send + Clone + 'static,
{
    fn clone_boxed(&self) -> Box<dyn ThreadsafeCloneableReaderTrait> {
        Box::new(self.clone())
    }
}

pub(crate) struct ThreadsafeCloneableReader {
    inner: Box<dyn ThreadsafeCloneableReaderTrait>,
}

pub(crate) fn scan_builder_into_threadsafe_cloneable_reader(
    builder: Box<VortexScanBuilder>,
) -> Result<Box<ThreadsafeCloneableReader>, Box<dyn std::error::Error + Send + Sync>> {
    let schema = match builder.output_schema {
        Some(schema) => schema,
        None => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = dtype.to_arrow_schema()?;
            Arc::new(arrow_schema)
        }
    };
    let data_type = DataType::Struct(schema.fields().clone());

    let stream = builder
        .inner
        .with_handle(RUNTIME.handle())
        .map(move |b| {
            b.into_arrow(&data_type)
                .map(|struct_array| RecordBatch::from(struct_array.as_struct()))
        })
        .into_stream()?
        .map_err(|e| ArrowError::ExternalError(Box::new(e)));

    let iter = RUNTIME.block_on_stream_thread_safe(|_h| stream);
    let rbr = RecordBatchIteratorAdapter::new(iter, schema);

    Ok(Box::new(ThreadsafeCloneableReader {
        inner: Box::new(rbr),
    }))
}

impl ThreadsafeCloneableReader {
    pub(crate) fn clone_a_stream(&self, out_stream: *mut u8) {
        let cloned_reader = self.inner.clone_boxed();
        let stream = FFI_ArrowArrayStream::new(cloned_reader);
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        // # Safety
        // Arrow C stream interface
        unsafe { std::ptr::write(out_stream, stream) };
    }
}