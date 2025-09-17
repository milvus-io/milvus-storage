use std::{ffi::CStr};
use libc::{c_char, c_int};
use object_store::{path::Path};
use tokio::runtime::Runtime;
use vortex::{arrays::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray}, buffer::{Buffer}, dtype::{DType, Nullability}, validity::Validity, IntoArray as _};

use crate::errcode::{SUCCESS, ERR_UTF8_CONVERSION};
use vortex::file::{VortexWriteOptions};
use vortex_io::VortexWrite;
use crate::objstore::ObjectStoreWrapper;
use futures::FutureExt;
use vortex_io::ObjectStoreWriter;

#[unsafe(no_mangle)]
pub extern "C" fn test_bridge_object_store_async_to_sync(
    objstore_raw: *mut ObjectStoreWrapper,
    path_raw: *const c_char) -> c_int {
    let vxwriter_opt = VortexWriteOptions::default();
    let tokio_rt  = Runtime::new().unwrap();

    let path = match unsafe { CStr::from_ptr(path_raw) }.to_str() {
        Ok(s) => s,
        Err(_) => return ERR_UTF8_CONVERSION,
    };
    let path = Path::parse(path).unwrap();

    // perpare rb 
    let values = Buffer::copy_from("hello worldhello world this is a long string".as_bytes());
    let offsets: PrimitiveArray = PrimitiveArray::from_iter([0, 11, 44]);
    let chunk1 = VarBinArray::try_new(
            offsets.into_array(),
            values,
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
    let chunk2 = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array();
    let chunkarray1 = ChunkedArray::from_iter([chunk1, chunk2]).into_array();
    let chunk3 = PrimitiveArray::from_iter([100i32, 101, 102]).into_array();
    let chunk4 = PrimitiveArray::from_iter([200i32, 201, 202]).into_array();
    let chunkarray2 = ChunkedArray::from_iter([chunk3, chunk4]).into_array();
    let st_stream = StructArray::from_fields(&[("strings", chunkarray1), ("numbers", chunkarray2)]).unwrap().to_array_stream();

    unsafe {
        let objstore = Box::from_raw(objstore_raw);

        tokio_rt.block_on(async {
            let _ = vxwriter_opt.write(
                    ObjectStoreWriter::new(objstore.inner.clone(), &path).await.unwrap(),
                    st_stream,
                )
                .boxed()
                .await.unwrap()
                .shutdown()
                .await.unwrap();
        });

        let _ = Box::into_raw(objstore);
    }
    

    SUCCESS
}
