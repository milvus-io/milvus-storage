use libc::{c_char, c_int};
use std::ffi::{CStr};
use futures::FutureExt;
use anyhow::Result;
use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use tokio::runtime::Runtime;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexError;
use vortex_file::VortexWriteOptions;
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::stream::ArrayStream;
use vortex_io::ObjectStoreWriter;
use vortex_io::VortexWrite;

use object_store::path::Path;
use crate::objstore::ObjectStoreWrapper;
use crate::errcode::{SUCCESS, ERR_INVALID_ARGS, ERR_UTF8_CONVERSION, ERR_BUILD_FAILED};

pub struct ObjectStoreWriterWrapper {
    pub inner: Option<ObjectStoreWriter>,
    rt: Runtime, // is that used?
}

#[unsafe(no_mangle)]
pub extern "C" fn create_object_store_writer(
    object_store: *const ObjectStoreWrapper,
    location: *const c_char,
    out_writer: *mut *mut ObjectStoreWriterWrapper,
) -> c_int {
    if object_store.is_null() || location.is_null() || out_writer.is_null() {
        return ERR_INVALID_ARGS;
    }
    
    let location_str = match unsafe { CStr::from_ptr(location) }.to_str() {
        Ok(s) => s,
        Err(_) => return ERR_UTF8_CONVERSION,
    };
    
    let object_store_wrapper = unsafe { &*object_store };
    let path = Path::from(location_str);
    
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ERR_BUILD_FAILED,
    };
    
    let writer = match rt.block_on(async {
        ObjectStoreWriter::new(object_store_wrapper.inner.clone(), &path).await
    }) {
        Ok(writer) => writer,
        Err(_) => return ERR_BUILD_FAILED,
    };
    
    let wrapper = Box::new(ObjectStoreWriterWrapper {
        inner: Some(writer),
        rt,
    });
    
    unsafe {
        *out_writer = Box::into_raw(wrapper);
    }
    
    SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn free_object_store_writer(
    writer: *mut ObjectStoreWriterWrapper,
) {
    if writer.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(writer);
    }
}

fn arrow_stream_to_vortex_stream(reader: ArrowArrayStreamReader) -> Result<impl ArrayStream> {
    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|result| {
            result
                .map(|record_batch| ArrayRef::from_arrow(record_batch, false))
                .map_err(|e|VortexError::from(e))
        }),
    );

    Ok(array_iter.into_array_stream())
}

#[unsafe(no_mangle)]
pub extern "C" fn write_array_stream(
    object_store: *const ObjectStoreWrapper,
    input_stream: *mut u8,
    path_raw: *const c_char,
) -> c_int {
    let tokio_rt  = Runtime::new().unwrap();
    let writer_opt = VortexWriteOptions::default();
    let object_store_wrapper = unsafe { &*object_store };

    let stream_reader =
        unsafe { ArrowArrayStreamReader::from_raw(input_stream as *mut FFI_ArrowArrayStream).unwrap() };

    let vortex_stream = arrow_stream_to_vortex_stream(stream_reader).unwrap();
    let path = match unsafe { CStr::from_ptr(path_raw) }
            .to_str()
            .map_err(|_| ERR_UTF8_CONVERSION) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let path = Path::parse(path).unwrap();

    tokio_rt.block_on(async {
        let _ = writer_opt.write(
            ObjectStoreWriter::new(object_store_wrapper.inner.clone(), &path).await.unwrap(),
            vortex_stream
        )
        .boxed()
        .await.unwrap()
        .shutdown()
        .await.unwrap();
    });

    SUCCESS
}
