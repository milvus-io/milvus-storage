
use std::ffi::{c_void};
use std::sync::Arc;
use std::io::Write;
use std::marker::PhantomData;

use async_compat::Compat;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};

use vortex::buffer::ByteBufferMut;
use vortex::error::{VortexError, VortexResult, vortex_err};
use vortex::io::runtime::Handle;
use vortex::io::file::{CoalesceWindow, IntoReadSource, ReadSource, ReadSourceRef, IoRequest};

#[repr(C)]
pub struct ffi_result {
    pub err_code: i32,
    pub message: *mut std::ffi::c_char,
}

unsafe extern "C" {
    unsafe fn FreeFFIResult(result: *mut ffi_result);

    // C-ABI: write data from pointer + size, return number of bytes written or negative error code
    unsafe fn fscpp_open_writer(fs :*mut std::ffi::c_void, path: *const u8, path_len: u64,
        out_writer_handle :*mut *mut std::ffi::c_void) 
        -> ffi_result;

    unsafe fn fscpp_write(
        writer: *mut std::ffi::c_void,
        data: *const u8,
        size: u64,
    ) -> ffi_result;
    unsafe fn fscpp_flush(writer: *mut std::ffi::c_void) -> ffi_result;
    unsafe fn fscpp_close(writer: *mut std::ffi::c_void) -> ffi_result;
    unsafe fn fscpp_destroy_writer(writer: *mut std::ffi::c_void);

    // C-ABI: pass path as pointer + len, avoid Rust String across FFI
    unsafe fn fscpp_head_object(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: u64,
        out_size: *mut u64
    ) -> ffi_result;

    // C-ABI: pass path pointer/len and explicit range bounds to avoid non-FFI-safe Rust types
    unsafe fn fscpp_get_object(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: u64,
        start: u64,
        len: u64,
        out_buf: *mut u8, // need pre-allocated buffer
    ) -> ffi_result;
}


// Helper to check ffi_result and convert to VortexError if needed.
fn check_ffi_result(result: &mut ffi_result, context: &str) -> Result<(), VortexError> {
    if result.err_code != 0 {
        // Safely copy C string into owned Rust String, handle null and invalid UTF-8.
        let message = unsafe {
            if result.message.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(result.message).to_string_lossy().into_owned()
            }
        };
        unsafe { FreeFFIResult(result as *mut ffi_result) };
        return Err(vortex_err!(Generic: "{}: {}", context, message));
    }
    Ok(())
}

struct ThreadSafePtr<T> {
    ptr: *mut c_void,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for ThreadSafePtr<T> {}
unsafe impl<T> Sync for ThreadSafePtr<T> {}

impl<T> ThreadSafePtr<T> {
    fn new(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }
    
    // Add methods to safely access the pointer
    // every time we call as_ptr, we clone it ensure
    // the pointer is not moved
    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    // move out the raw pointer
    fn as_raw_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl<T> Clone for ThreadSafePtr<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

pub struct ObjectStoreWriterCpp {
    inner: ThreadSafePtr<std::ffi::c_void>,
    path: String,
    writer: ThreadSafePtr<std::ffi::c_void>,
}

impl ObjectStoreWriterCpp {
    pub fn new(fs_rawptr: *mut std::ffi::c_void, path: &String) -> Result<Self, VortexError> {
        Ok(Self {
            inner: ThreadSafePtr::new(fs_rawptr),
            path: path.clone(),
            writer: ThreadSafePtr::new(std::ptr::null_mut()),
        })
    }
}

impl Drop for ObjectStoreWriterCpp {
    fn drop(&mut self) {
        unsafe { 
            if !self.writer.as_ptr().is_null() {
                let mut result = fscpp_close(self.writer.as_ptr());
                check_ffi_result(&mut result, "Failed to close ObjectStoreWriterCpp")
                    .unwrap_or(());

                fscpp_destroy_writer(self.writer.as_raw_ptr());
            }
        };
    }
}

impl Write for ObjectStoreWriterCpp {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = unsafe { 
            if self.writer.as_ptr().is_null() {
                let mut writer_raw: *mut c_void = std::ptr::null_mut();
                let path = std::ffi::CString::new(self.path.clone()).unwrap();
                let mut result = fscpp_open_writer(self.inner.as_ptr(), path.as_ptr() as *const u8, path.as_bytes().len() as u64,
                    &mut writer_raw);
                check_ffi_result(&mut result, "Failed to open ObjectStoreWriterCpp")
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                self.writer = ThreadSafePtr::new(writer_raw);
            }

            let mut result = fscpp_write(self.writer.as_ptr(), buf.as_ptr(), buf.len() as u64);
            check_ffi_result(&mut result, "Failed to write data to ObjectStoreWriterCpp")
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            buf.len()
        };
        
        Ok(written as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut result = unsafe { fscpp_flush(self.writer.as_ptr()) };
        check_ffi_result(&mut result, "Failed to flush data to ObjectStoreWriterCpp")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }
}

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
const CONCURRENCY: usize = 192;

pub struct ObjectStoreReadSourceCpp {
    inner: ThreadSafePtr<std::ffi::c_void>,
    path: String,
    uri: Arc<str>,
    coalesce_window: Option<CoalesceWindow>,
}

impl ObjectStoreReadSourceCpp {
    pub fn new(fs_rawptr: *mut std::ffi::c_void, path: &str) -> VortexResult<Self> {
        Ok(Self {
            inner: ThreadSafePtr::new(fs_rawptr),
            path: path.to_string(),
            uri: Arc::from(path.to_string()),
            coalesce_window: Some(COALESCING_WINDOW),
        })
    }
}


impl IntoReadSource for ObjectStoreReadSourceCpp {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        Ok(Arc::new(ObjectStoreIoSourceCpp { io: self, handle }))
    }
}

struct ObjectStoreIoSourceCpp {
    io: ObjectStoreReadSourceCpp,
    handle: Handle,
}

impl ReadSource for ObjectStoreIoSourceCpp {
    fn uri(&self) -> &Arc<str> {
        &self.io.uri
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        self.io.coalesce_window
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        // move owned values into the async block so the future is 'static
        let inner = self.io.inner.clone();
        let path = self.io.path.clone();
        let handle = self.handle.clone();
        Compat::new(async move {
            // Pass path as bytes to FFI (no allocation across FFI boundaries)
            let path_bytes = path.into_bytes();
            // Return Result from blocking task to propagate errors cleanly
            let task = handle.spawn_blocking(move || {
                unsafe {
                    let mut out_size: u64 = 0;
                    let mut result = fscpp_head_object(
                        inner.as_ptr(),
                        path_bytes.as_ptr(),
                        path_bytes.len() as u64,
                        &mut out_size,
                    );
                    check_ffi_result(
                        &mut result,
                        "Failed to get object size from ObjectStoreIoSourceCpp",
                    )
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                    Ok::<u64, std::io::Error>(out_size)
                }
            });
            let size: u64 = Compat::new(task)
                .await
                .map_err(|e| vortex_err!(Generic: "{}", e))?;
            Ok(size)
        })
        .boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        let self2 = self.clone();
        requests
        .map(move |req| {
            let store = self.io.inner.clone();
            let path = self.io.path.clone();

            let range = req.range();
            let start = range.start;
            let end = range.end;
            let len = end - start;

            let alignment = req.alignment();

            // Offload sync FFI to blocking pool, copy into Rust-owned Vec<u8>
            let blocking = self.handle.spawn_blocking(move || -> VortexResult<Vec<u8>> {
                let path_bytes = path.into_bytes();
                // Preallocate buffer with exact capacity; FFI will fill it.
                let mut owned: Vec<u8> = Vec::with_capacity(len as usize);

                unsafe {
                    let mut result = fscpp_get_object(
                        store.as_ptr(),
                        path_bytes.as_ptr(),
                        path_bytes.len() as u64,
                        start,
                        len,
                        owned.as_mut_ptr(),
                    );

                    // Convert FFI error to VortexError (also frees message via FreeFFIResult)
                    check_ffi_result(
                        &mut result,
                        "Failed to get object range from ObjectStoreIoSourceCpp",
                    )?;

                    // Mark the bytes as initialized after successful fill
                    owned.set_len(len as usize);
                }

                Ok(owned)
            });

            let fut = async move {
                let bytes: Vec<u8> = Compat::new(blocking).await?;
                let mut buffer = ByteBufferMut::with_capacity_aligned(len as usize, alignment);
                buffer.extend_from_slice(&bytes);
                Ok(buffer.freeze())
            };

            async move { req.resolve(Compat::new(fut).await) }
        })
        .map(move |f| self2.handle.spawn(f))
        .buffer_unordered(CONCURRENCY)
        .collect::<()>()
        .boxed()
    }
}
