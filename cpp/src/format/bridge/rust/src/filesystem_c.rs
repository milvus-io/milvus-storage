
use std::ffi::{c_void};
use std::sync::{Arc, Mutex};
use std::io::Write;
use std::marker::PhantomData;
use std::time::Instant;

use async_compat::Compat;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};

use vortex::buffer::{ByteBuffer, ByteBufferMut};
use vortex::error::{VortexError, VortexResult, vortex_err};
use vortex::io::runtime::Handle;
use vortex::io::file::{CoalesceWindow, IntoReadSource, ReadSource, ReadSourceRef, IoRequest};

//=============================================================================
// IO Trace Collector
//=============================================================================

#[derive(Clone)]
struct IoTraceEntry {
    seq: u32,
    start_us: u64,   // microseconds since trace reset
    end_us: u64,
    offset: u64,
    size: u64,
}

struct IoTraceState {
    enabled: bool,
    epoch: Instant,
    entries: Vec<IoTraceEntry>,
    seq_counter: u32,
}

static IO_TRACE: std::sync::LazyLock<Mutex<IoTraceState>> =
    std::sync::LazyLock::new(|| Mutex::new(IoTraceState {
        enabled: false,
        epoch: Instant::now(),
        entries: Vec::new(),
        seq_counter: 0,
    }));

pub(crate) fn reset_io_trace() {
    let mut state = IO_TRACE.lock().unwrap();
    state.enabled = true;
    state.epoch = Instant::now();
    state.entries.clear();
    state.seq_counter = 0;
}

pub(crate) fn disable_io_trace() {
    let mut state = IO_TRACE.lock().unwrap();
    state.enabled = false;
    state.entries.clear();
}

fn record_io_start() -> (bool, Instant) {
    let state = IO_TRACE.lock().unwrap();
    (state.enabled, Instant::now())
}

fn record_io_end(enabled: bool, start_instant: Instant, offset: u64, size: u64) {
    if !enabled { return; }
    let mut state = IO_TRACE.lock().unwrap();
    let epoch = state.epoch;
    let seq = state.seq_counter;
    state.seq_counter += 1;
    state.entries.push(IoTraceEntry {
        seq,
        start_us: start_instant.duration_since(epoch).as_micros() as u64,
        end_us: Instant::now().duration_since(epoch).as_micros() as u64,
        offset,
        size,
    });
}

pub(crate) fn print_io_trace() {
    let state = IO_TRACE.lock().unwrap();
    if state.entries.is_empty() {
        eprintln!("[IO Trace] No entries recorded");
        return;
    }

    let mut entries = state.entries.clone();
    entries.sort_by_key(|e| e.start_us);

    // Group into rounds: a new round starts when a request's start_us is after
    // the previous request's end_us (i.e., sequential dependency)
    let mut rounds: Vec<Vec<&IoTraceEntry>> = Vec::new();
    let mut current_round: Vec<&IoTraceEntry> = Vec::new();
    let mut round_end_us: u64 = 0;

    for entry in &entries {
        if current_round.is_empty() || entry.start_us < round_end_us + 2000 {
            // Same round (started within 2ms of round begin)
            current_round.push(entry);
            if entry.end_us > round_end_us {
                round_end_us = entry.end_us;
            }
        } else {
            rounds.push(current_round);
            current_round = vec![entry];
            round_end_us = entry.end_us;
        }
    }
    if !current_round.is_empty() {
        rounds.push(current_round);
    }

    eprintln!("[IO Trace] {} total requests, {} rounds", entries.len(), rounds.len());
    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let wall_us = entries.iter().map(|e| e.end_us).max().unwrap_or(0)
                - entries.iter().map(|e| e.start_us).min().unwrap_or(0);
    eprintln!("[IO Trace] total_bytes={:.2}MB  wall={:.1}ms",
             total_bytes as f64 / (1024.0 * 1024.0), wall_us as f64 / 1000.0);

    for (ri, round) in rounds.iter().enumerate() {
        let r_start = round.iter().map(|e| e.start_us).min().unwrap_or(0);
        let r_end = round.iter().map(|e| e.end_us).max().unwrap_or(0);
        let r_wall = r_end - r_start;
        let longest = round.iter().map(|e| e.end_us - e.start_us).max().unwrap_or(0);
        let r_bytes: u64 = round.iter().map(|e| e.size).sum();
        eprintln!("    R{} — {} req, wall={:.1}ms, longest={:.1}ms, bytes={:.2}MB",
                 ri + 1, round.len(), r_wall as f64 / 1000.0,
                 longest as f64 / 1000.0, r_bytes as f64 / (1024.0 * 1024.0));
        for entry in round.iter() {
            eprintln!("      seq={:<3} start={:>8.1}ms end={:>8.1}ms dur={:>6.1}ms size={:>8} range={}..{}",
                     entry.seq,
                     entry.start_us as f64 / 1000.0,
                     entry.end_us as f64 / 1000.0,
                     (entry.end_us - entry.start_us) as f64 / 1000.0,
                     entry.size,
                     entry.offset,
                     entry.offset + entry.size);
        }
    }
}

#[repr(C)]
pub struct LoonFFIResult {
    pub err_code: i32,
    pub message: *mut std::ffi::c_char,
}

#[repr(C)]
pub struct LoonFileSystemMeta {
    pub key: *mut std::ffi::c_char,
    pub value: *mut std::ffi::c_char,
}

unsafe extern "C" {
    unsafe fn loon_ffi_free_result(result: *mut LoonFFIResult);
    unsafe fn loon_filesystem_free_meta_array(meta_array: *mut LoonFileSystemMeta, meta_count: u32);

    // C-ABI: write data from pointer + size, return number of bytes written or negative error code
    unsafe fn loon_filesystem_open_writer(
        fs: *mut std::ffi::c_void,
        path: *const u8,
        path_len: u32,
        meta_array: *const LoonFileSystemMeta,
        num_of_meta: u32,
        out_handle: *mut *mut std::ffi::c_void,
    ) -> LoonFFIResult;

    unsafe fn loon_filesystem_writer_write(
        writer: *mut std::ffi::c_void,
        data: *const u8,
        size: u64,
    ) -> LoonFFIResult;
    unsafe fn loon_filesystem_writer_flush(writer: *mut std::ffi::c_void) -> LoonFFIResult;
    unsafe fn loon_filesystem_writer_close(writer: *mut std::ffi::c_void) -> LoonFFIResult;
    unsafe fn loon_filesystem_writer_destroy(writer: *mut std::ffi::c_void);

    // C-ABI: pass path as pointer + len, avoid Rust String across FFI
    unsafe fn loon_filesystem_get_file_info(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: u64,
        out_size: *mut u64
    ) -> LoonFFIResult;

    // C-ABI: pass path pointer/len and explicit range bounds to avoid non-FFI-safe Rust types
    unsafe fn loon_filesystem_read_file(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: u64,
        start: u64,
        len: u64,
        out_buf: *mut u8, // need pre-allocated buffer
    ) -> LoonFFIResult;

    // C-ABI: open a RandomAccessFile reader handle (one OpenInputFile, reuse for multiple ReadAt)
    unsafe fn loon_filesystem_open_reader(
        fs: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: u32,
        file_size: u64,
        out_reader_ptr: *mut *mut std::ffi::c_void,
    ) -> LoonFFIResult;

    unsafe fn loon_filesystem_reader_readat(
        reader: *mut std::ffi::c_void,
        offset: u64,
        nbytes: u64,
        out_data: *mut u8,
    ) -> LoonFFIResult;

    unsafe fn loon_filesystem_reader_close(reader: *mut std::ffi::c_void) -> LoonFFIResult;
    unsafe fn loon_filesystem_reader_destroy(reader: *mut std::ffi::c_void);
}


// Helper to check LoonFFIResult and convert to VortexError if needed.
fn check_loon_ffi_result(result: &mut LoonFFIResult, context: &str) -> Result<(), VortexError> {
    if result.err_code != 0 {
        // Safely copy C string into owned Rust String, handle null and invalid UTF-8.
        let message = unsafe {
            if result.message.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(result.message).to_string_lossy().into_owned()
            }
        };
        unsafe { loon_ffi_free_result(result as *mut LoonFFIResult) };
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
                let mut result = loon_filesystem_writer_close(self.writer.as_ptr());
                check_loon_ffi_result(&mut result, "Failed to close ObjectStoreWriterCpp")
                    .unwrap_or(());

                loon_filesystem_writer_destroy(self.writer.as_raw_ptr());
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
                let mut result = loon_filesystem_open_writer(self.inner.as_ptr(),
                    path.as_ptr() as *const u8,
                    path.as_bytes().len() as u32,
                    std::ptr::null(), // meta_array
                    0 as u32,         // num_of_meta
                    &mut writer_raw);
                check_loon_ffi_result(&mut result, "Failed to open ObjectStoreWriterCpp")
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                self.writer = ThreadSafePtr::new(writer_raw);
            }

            let mut result = loon_filesystem_writer_write(self.writer.as_ptr(), buf.as_ptr(), buf.len() as u64);
            check_loon_ffi_result(&mut result, "Failed to write data to ObjectStoreWriterCpp")
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            buf.len()
        };
        
        Ok(written as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut result = unsafe { loon_filesystem_writer_flush(self.writer.as_ptr()) };
        check_loon_ffi_result(&mut result, "Failed to flush data to ObjectStoreWriterCpp")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }
}

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 1024 * 1024,       // 1 MB
    max_size: 1 * 1024 * 1024,  // 1 MB
};
const CONCURRENCY: usize = 256;

/// Arc-wrapped reader handle that ensures the underlying C++ RandomAccessFile
/// is only closed/destroyed after all concurrent spawn_blocking tasks are done.
struct ReaderHandle {
    ptr: *mut c_void,
}

unsafe impl Send for ReaderHandle {}
unsafe impl Sync for ReaderHandle {}

impl ReaderHandle {
    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for ReaderHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                let mut result = loon_filesystem_reader_close(self.ptr);
                if let Err(e) = check_loon_ffi_result(&mut result, "Failed to close ReaderHandle") {
                    eprintln!("Warning: ReaderHandle close failed: {e}");
                }
                loon_filesystem_reader_destroy(self.ptr);
            }
        }
    }
}

pub struct ObjectStoreReadSourceCpp {
    inner: ThreadSafePtr<std::ffi::c_void>,
    reader: Arc<ReaderHandle>,
    path: String,
    uri: Arc<str>,
    coalesce_window: Option<CoalesceWindow>,
}

impl ObjectStoreReadSourceCpp {
    pub fn new(fs_rawptr: *mut std::ffi::c_void, path: &str, file_size: u64) -> VortexResult<Self> {
        let mut reader_raw: *mut c_void = std::ptr::null_mut();
        let path_bytes = path.as_bytes();
        unsafe {
            let mut result = loon_filesystem_open_reader(
                fs_rawptr,
                path_bytes.as_ptr(),
                path_bytes.len() as u32,
                file_size,
                &mut reader_raw,
            );
            check_loon_ffi_result(&mut result, "Failed to open reader in ObjectStoreReadSourceCpp")?;
        }
        Ok(Self {
            inner: ThreadSafePtr::new(fs_rawptr),
            reader: Arc::new(ReaderHandle { ptr: reader_raw }),
            path: path.to_string(),
            uri: Arc::from(path.to_string()),
            coalesce_window: Some(COALESCING_WINDOW),
        })
    }
}

// Drop is handled by Arc<ReaderHandle> which closes and destroys the reader.

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
                    let mut result = loon_filesystem_get_file_info(
                        inner.as_ptr(),
                        path_bytes.as_ptr(),
                        path_bytes.len() as u64,
                        &mut out_size,
                    );
                    check_loon_ffi_result(
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
                let reader = self.io.reader.clone();

                let range = req.range();
                let start = range.start;
                let end = range.end;
                let len = end - start;

                let alignment = req.alignment();

                // Offload sync FFI to blocking pool, read directly into aligned buffer
                let blocking = self.handle.spawn_blocking(move || -> VortexResult<ByteBuffer> {
                    let (trace_enabled, trace_start) = record_io_start();
                    let mut buffer = ByteBufferMut::with_capacity_aligned(len as usize, alignment);

                    unsafe {
                        let dst = buffer.spare_capacity_mut().as_mut_ptr() as *mut u8;
                        let mut result = loon_filesystem_reader_readat(
                            reader.as_ptr(),
                            start,
                            len,
                            dst,
                        );

                        check_loon_ffi_result(
                            &mut result,
                            "Failed to readat from ObjectStoreIoSourceCpp",
                        )?;

                        buffer.set_len(len as usize);
                    }

                    record_io_end(trace_enabled, trace_start, start, len);
                    Ok(buffer.freeze())
                });

                let fut = async move {
                    let buffer: ByteBuffer = Compat::new(blocking).await?;
                    Ok(buffer)
                };

                async move { req.resolve(Compat::new(fut).await) }
            })
            .map(move |f| self2.handle.spawn(f))
            .buffer_unordered(CONCURRENCY)
        .collect::<()>()
        .boxed()
    }
}