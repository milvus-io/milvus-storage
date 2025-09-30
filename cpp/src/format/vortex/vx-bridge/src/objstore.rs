
use libc::{c_char, c_int};
use std::ffi::{CStr};

use std::sync::Arc;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use crate::errcode::{SUCCESS, ERR_INVALID_ARGS, ERR_UTF8_CONVERSION, ERR_BUILD_FAILED};

pub struct ObjectStoreWrapper {
    pub inner: Arc<dyn ObjectStore>,
}

#[unsafe(no_mangle)]
pub extern "C" fn create_object_store(
    ostype: *const c_char, // TODO: unused 
    endpoint: *const c_char,
    access_key_id: *const c_char,
    secret_access_key: *const c_char,
    region: *const c_char,
    bucket_name: *const c_char,
    out_store: *mut *mut ObjectStoreWrapper,
) -> c_int {
    if ostype.is_null()
        || endpoint.is_null()
        || access_key_id.is_null()
        || secret_access_key.is_null()
        || region.is_null()
        || bucket_name.is_null()
        || out_store.is_null()
    {
        return ERR_INVALID_ARGS;
    }

    let rust_str = |ptr| {
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .map_err(|_| ERR_UTF8_CONVERSION)
    };

    let _ostype = match rust_str(ostype) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let endpoint = match rust_str(endpoint) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let access_key_id = match rust_str(access_key_id) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let secret_access_key = match rust_str(secret_access_key) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let region = match rust_str(region) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let bucket_name = match rust_str(bucket_name) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let store = match AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket_name)
        .with_region(region)
        .with_access_key_id(access_key_id)
        .with_secret_access_key(secret_access_key)
        .with_allow_http(true)
        .build()
    {
        Ok(store) => Arc::new(store),
        Err(_) => return ERR_BUILD_FAILED,
    };

    let wrapper = Box::new(ObjectStoreWrapper { inner: store });

    // Arc to raw C ptr
    unsafe {
        *out_store = Box::into_raw(wrapper);
    }

    SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn free_object_store_wrapper(ptr: *mut ObjectStoreWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}
