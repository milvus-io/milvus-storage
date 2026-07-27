use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::ffi::{CString, c_void};
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
#[cfg(feature = "io-trace")]
use std::time::Instant;

use async_compat::Compat;
#[cfg(feature = "s3-crt-async")]
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::FutureExt;

use vortex::array::buffer::BufferHandle;
#[cfg(feature = "s3-crt-async")]
use vortex::buffer::Alignment;
use vortex::buffer::{ByteBuffer, ByteBufferMut};
use vortex::error::{VortexError, VortexResult, vortex_err};
use vortex::io::runtime::Handle;
use vortex::io::{CoalesceConfig, IoBuf, VortexReadAt, VortexWrite};

//=============================================================================
// IO Trace Collector
//=============================================================================

#[cfg(feature = "io-trace")]
#[derive(Clone)]
struct IoTraceEntry {
    seq: u32,
    start_us: u64, // microseconds since trace reset
    end_us: u64,
    offset: u64,
    size: u64,
}

#[cfg(feature = "io-trace")]
struct IoTraceState {
    enabled: bool,
    epoch: Instant,
    entries: Vec<IoTraceEntry>,
    seq_counter: u32,
}

#[cfg(feature = "io-trace")]
static IO_TRACE: std::sync::LazyLock<Mutex<IoTraceState>> = std::sync::LazyLock::new(|| {
    Mutex::new(IoTraceState {
        enabled: false,
        epoch: Instant::now(),
        entries: Vec::new(),
        seq_counter: 0,
    })
});

#[cfg(feature = "io-trace")]
struct IoTraceStart {
    enabled: bool,
    start: Instant,
}

#[cfg(not(feature = "io-trace"))]
type IoTraceStart = ();

#[cfg(feature = "io-trace")]
pub(crate) fn reset_io_trace() {
    let mut state = IO_TRACE.lock().unwrap();
    state.enabled = true;
    state.epoch = Instant::now();
    state.entries.clear();
    state.seq_counter = 0;
}

#[cfg(not(feature = "io-trace"))]
pub(crate) fn reset_io_trace() {}

#[cfg(feature = "io-trace")]
pub(crate) fn disable_io_trace() {
    let mut state = IO_TRACE.lock().unwrap();
    state.enabled = false;
    state.entries.clear();
}

#[cfg(not(feature = "io-trace"))]
pub(crate) fn disable_io_trace() {}

#[cfg(feature = "io-trace")]
fn record_io_start() -> IoTraceStart {
    let state = IO_TRACE.lock().unwrap();
    IoTraceStart {
        enabled: state.enabled,
        start: Instant::now(),
    }
}

#[cfg(not(feature = "io-trace"))]
#[inline(always)]
fn record_io_start() -> IoTraceStart {}

#[cfg(feature = "io-trace")]
fn record_io_end(trace_start: IoTraceStart, offset: u64, size: u64) {
    if !trace_start.enabled {
        return;
    }
    let mut state = IO_TRACE.lock().unwrap();
    let epoch = state.epoch;
    let seq = state.seq_counter;
    state.seq_counter += 1;
    state.entries.push(IoTraceEntry {
        seq,
        start_us: trace_start.start.duration_since(epoch).as_micros() as u64,
        end_us: Instant::now().duration_since(epoch).as_micros() as u64,
        offset,
        size,
    });
}

#[cfg(not(feature = "io-trace"))]
#[inline(always)]
fn record_io_end(_trace_start: IoTraceStart, _offset: u64, _size: u64) {}

#[cfg(feature = "io-trace")]
pub(crate) fn print_io_trace() {
    let state = IO_TRACE.lock().unwrap();
    if state.entries.is_empty() {
        eprintln!("[IO Trace] No entries recorded");
        return;
    }

    let mut entries = state.entries.clone();
    entries.sort_by_key(|e| e.start_us);

    // Group into rounds: a new round starts when a request's start_us is after
    // the previous request's end_us (i.e., sequential dependency).
    let mut rounds: Vec<Vec<&IoTraceEntry>> = Vec::new();
    let mut current_round: Vec<&IoTraceEntry> = Vec::new();
    let mut round_end_us: u64 = 0;

    for entry in &entries {
        if current_round.is_empty() || entry.start_us < round_end_us + 2000 {
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

    eprintln!(
        "[IO Trace] {} total requests, {} rounds",
        entries.len(),
        rounds.len()
    );
    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let wall_us = entries.iter().map(|e| e.end_us).max().unwrap_or(0)
        - entries.iter().map(|e| e.start_us).min().unwrap_or(0);
    eprintln!(
        "[IO Trace] total_bytes={:.2}MB  wall={:.1}ms",
        total_bytes as f64 / (1024.0 * 1024.0),
        wall_us as f64 / 1000.0
    );

    for (ri, round) in rounds.iter().enumerate() {
        let r_start = round.iter().map(|e| e.start_us).min().unwrap_or(0);
        let r_end = round.iter().map(|e| e.end_us).max().unwrap_or(0);
        let r_wall = r_end - r_start;
        let longest = round
            .iter()
            .map(|e| e.end_us - e.start_us)
            .max()
            .unwrap_or(0);
        let r_bytes: u64 = round.iter().map(|e| e.size).sum();
        eprintln!(
            "    R{} - {} req, wall={:.1}ms, longest={:.1}ms, bytes={:.2}MB",
            ri + 1,
            round.len(),
            r_wall as f64 / 1000.0,
            longest as f64 / 1000.0,
            r_bytes as f64 / (1024.0 * 1024.0)
        );
        for entry in round.iter() {
            eprintln!(
                "      seq={:<3} start={:>8.1}ms end={:>8.1}ms dur={:>6.1}ms size={:>8} range={}..{}",
                entry.seq,
                entry.start_us as f64 / 1000.0,
                entry.end_us as f64 / 1000.0,
                (entry.end_us - entry.start_us) as f64 / 1000.0,
                entry.size,
                entry.offset,
                entry.offset + entry.size
            );
        }
    }
}

#[cfg(not(feature = "io-trace"))]
pub(crate) fn print_io_trace() {
    eprintln!("[IO Trace] not compiled in");
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

#[cfg(feature = "s3-crt-async")]
type LoonFileSystemReadAsyncCallback =
    unsafe extern "C" fn(*mut std::ffi::c_void, LoonFFIResult, u64);

unsafe extern "C" {
    unsafe fn loon_ffi_free_result(result: *mut LoonFFIResult);

    // C-ABI: write data from pointer + size, return number of bytes written or negative error code
    unsafe fn loon_filesystem_open_writer(
        fs: *mut std::ffi::c_void,
        path: *const u8,
        path_len: u32,
        meta_array: *const LoonFileSystemMeta,
        num_of_meta: u32,
        conditional: bool,
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
        path_len: u32,
        out_size: *mut u64,
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

#[cfg(feature = "s3-crt-async")]
unsafe extern "C" {
    unsafe fn loon_filesystem_reader_supports_async(
        reader: *mut std::ffi::c_void,
        out_supported: *mut bool,
    ) -> LoonFFIResult;

    unsafe fn loon_filesystem_reader_readat_async(
        reader: *mut std::ffi::c_void,
        offset: u64,
        nbytes: u64,
        out_data: *mut u8,
        callback: LoonFileSystemReadAsyncCallback,
        user_data: *mut std::ffi::c_void,
    ) -> LoonFFIResult;
}

const LOON_VORTEX_FFI_ERRCODE_MARKER: &str = "__LOON_VORTEX_FFI_ERRCODE__=";

#[derive(Debug)]
struct LoonFfiError {
    err_code: i32,
    context: String,
    message: String,
}

impl std::fmt::Display for LoonFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{LOON_VORTEX_FFI_ERRCODE_MARKER}{}; {}: {}",
            self.err_code, self.context, self.message
        )
    }
}

impl std::error::Error for LoonFfiError {}

fn ffi_err(err_code: i32, context: &str, message: String) -> VortexError {
    vortex_err!(External: LoonFfiError {
        err_code,
        context: context.to_string(),
        message,
    })
}

// Helper to check LoonFFIResult and convert to VortexError if needed.
fn check_loon_ffi_result(result: &mut LoonFFIResult, context: &str) -> Result<(), VortexError> {
    if result.err_code != 0 {
        let err_code = result.err_code;
        // Safely copy C string into owned Rust String, handle null and invalid UTF-8.
        let message = unsafe {
            if result.message.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(result.message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        unsafe { loon_ffi_free_result(result as *mut LoonFFIResult) };
        return Err(ffi_err(err_code, context, message));
    }
    Ok(())
}

#[cfg(feature = "s3-crt-async")]
fn reader_supports_async(reader: *mut c_void) -> Result<bool, VortexError> {
    let mut supported = false;
    let mut result =
        unsafe { loon_filesystem_reader_supports_async(reader, &mut supported as *mut bool) };
    check_loon_ffi_result(&mut result, "Failed to check reader async support")?;
    Ok(supported)
}

#[cfg(not(feature = "s3-crt-async"))]
fn reader_supports_async(_reader: *mut c_void) -> Result<bool, VortexError> {
    Ok(false)
}

#[cfg(feature = "s3-crt-async")]
type AsyncReadResult = Result<ByteBuffer, VortexError>;

#[cfg(feature = "s3-crt-async")]
struct AsyncReadCallbackState {
    sender: Option<oneshot::Sender<AsyncReadResult>>,
    reader: Arc<ReaderHandle>,
    buffer: ByteBufferMut,
    start: u64,
    expected_len: u64,
    trace_start: IoTraceStart,
}

#[cfg(feature = "s3-crt-async")]
unsafe extern "C" fn async_read_callback(
    user_data: *mut c_void,
    mut result: LoonFFIResult,
    bytes_read: u64,
) {
    if user_data.is_null() {
        if result.err_code != 0 {
            unsafe { loon_ffi_free_result(&mut result as *mut LoonFFIResult) };
        }
        return;
    }

    let AsyncReadCallbackState {
        mut sender,
        reader: _reader,
        mut buffer,
        start,
        expected_len,
        trace_start,
    } = *unsafe { Box::from_raw(user_data.cast::<AsyncReadCallbackState>()) };

    let read_result = if result.err_code != 0 {
        let message = unsafe {
            if result.message.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(result.message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        unsafe { loon_ffi_free_result(&mut result as *mut LoonFFIResult) };
        Err(ffi_err(result.err_code, "Async readat failed", message))
    } else {
        record_io_end(trace_start, start, bytes_read);

        if bytes_read != expected_len {
            Err(vortex_err!(
                "Async readat returned {} bytes for range {}..{}, expected {}",
                bytes_read,
                start,
                start + expected_len,
                expected_len
            ))
        } else {
            unsafe { buffer.set_len(bytes_read as usize) };
            Ok(buffer.freeze())
        }
    };

    if let Some(sender) = sender.take() {
        let _ = sender.send(read_result);
    }
}

#[cfg(feature = "s3-crt-async")]
async fn read_async_via_ffi(
    reader: Arc<ReaderHandle>,
    start: u64,
    len: u64,
    alignment: Alignment,
) -> VortexResult<ByteBuffer> {
    if len == 0 {
        return Ok(ByteBufferMut::with_capacity_aligned(0, alignment).freeze());
    }

    let trace_start = record_io_start();
    let mut buffer = ByteBufferMut::with_capacity_aligned(len as usize, alignment);
    let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let (sender, receiver) = oneshot::channel();
    let state = Box::new(AsyncReadCallbackState {
        sender: Some(sender),
        reader: reader.clone(),
        buffer,
        start,
        expected_len: len,
        trace_start,
    });
    let state_ptr = Box::into_raw(state);

    {
        let mut result = unsafe {
            loon_filesystem_reader_readat_async(
                reader.as_ptr(),
                start,
                len,
                out_data,
                async_read_callback,
                state_ptr.cast::<c_void>(),
            )
        };
        if result.err_code != 0 {
            unsafe {
                drop(Box::from_raw(state_ptr));
            }
            check_loon_ffi_result(&mut result, "Failed to submit async readat")?;
        }
    }
    drop(reader);

    let read_result = receiver
        .await
        .map_err(|_| vortex_err!("Async readat completion channel closed"))?;
    read_result
}

pub(crate) struct ThreadSafePtr<T> {
    ptr: *mut c_void,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for ThreadSafePtr<T> {}
unsafe impl<T> Sync for ThreadSafePtr<T> {}

impl<T> ThreadSafePtr<T> {
    pub(crate) fn new(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    // Add methods to safely access the pointer
    // every time we call as_ptr, we clone it ensure
    // the pointer is not moved
    pub(crate) fn as_ptr(&self) -> *mut c_void {
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

enum WriterSlot {
    Unopened,
    Open(ThreadSafePtr<c_void>),
    Closed,
}

struct ObjectStoreWriterInner {
    inner: ThreadSafePtr<c_void>,
    path: CString,
    writer: Mutex<WriterSlot>,
}

impl ObjectStoreWriterInner {
    fn new(fs_rawptr: *mut c_void, path: &str) -> Result<Self, VortexError> {
        Ok(Self {
            inner: ThreadSafePtr::new(fs_rawptr),
            path: CString::new(path)
                .map_err(|e| vortex_err!("ObjectStoreWriterCpp path contains nul byte: {}", e))?,
            writer: Mutex::new(WriterSlot::Unopened),
        })
    }

    fn open_writer(&self) -> Result<ThreadSafePtr<c_void>, VortexError> {
        let mut writer_raw: *mut c_void = std::ptr::null_mut();
        let mut result = unsafe {
            loon_filesystem_open_writer(
                self.inner.as_ptr(),
                self.path.as_ptr() as *const u8,
                self.path.as_bytes().len() as u32,
                std::ptr::null(), // meta_array
                0 as u32,         // num_of_meta
                false,            // conditional
                &mut writer_raw,
            )
        };
        check_loon_ffi_result(&mut result, "Failed to open ObjectStoreWriterCpp")?;
        Ok(ThreadSafePtr::new(writer_raw))
    }

    fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut writer = self.writer.lock().unwrap();
        let writer_ptr = match &mut *writer {
            WriterSlot::Unopened => {
                let opened = self
                    .open_writer()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let writer_ptr = opened.as_ptr();
                *writer = WriterSlot::Open(opened);
                writer_ptr
            }
            WriterSlot::Open(writer) => writer.as_ptr(),
            WriterSlot::Closed => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "cannot write: ObjectStoreWriterCpp is closed",
                ));
            }
        };

        let mut result =
            unsafe { loon_filesystem_writer_write(writer_ptr, buf.as_ptr(), buf.len() as u64) };
        check_loon_ffi_result(&mut result, "Failed to write data to ObjectStoreWriterCpp")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        Ok(buf.len())
    }

    fn flush(&self) -> Result<(), VortexError> {
        let writer = self.writer.lock().unwrap();
        match &*writer {
            WriterSlot::Unopened => Ok(()),
            WriterSlot::Open(writer) => {
                let mut result = unsafe { loon_filesystem_writer_flush(writer.as_ptr()) };
                check_loon_ffi_result(&mut result, "Failed to flush data to ObjectStoreWriterCpp")
            }
            WriterSlot::Closed => Err(vortex_err!("cannot flush: ObjectStoreWriterCpp is closed")),
        }
    }

    fn close(&self) -> Result<(), VortexError> {
        let mut writer = self.writer.lock().unwrap();
        match std::mem::replace(&mut *writer, WriterSlot::Closed) {
            WriterSlot::Unopened | WriterSlot::Closed => Ok(()),
            WriterSlot::Open(writer) => {
                let writer_raw = writer.as_raw_ptr();
                let mut result = unsafe { loon_filesystem_writer_close(writer_raw) };
                let close_result =
                    check_loon_ffi_result(&mut result, "Failed to close ObjectStoreWriterCpp");
                unsafe { loon_filesystem_writer_destroy(writer_raw) };
                close_result
            }
        }
    }
}

impl Drop for ObjectStoreWriterInner {
    fn drop(&mut self) {
        let mut writer = self.writer.lock().unwrap();
        let writer = std::mem::replace(&mut *writer, WriterSlot::Closed);
        if let WriterSlot::Open(writer) = writer {
            // Only explicit close may complete the object. Drop can run after writer
            // errors, so calling close here would finalize partial data; release only
            // the C++ wrapper.
            unsafe { loon_filesystem_writer_destroy(writer.as_raw_ptr()) };
        }
    }
}

struct ObjectStoreWriter {
    inner: Arc<ObjectStoreWriterInner>,
    handle: Handle,
}

impl ObjectStoreWriter {
    fn new(
        fs_rawptr: *mut c_void,
        path: &str,
        handle: Handle,
    ) -> Result<(ObjectStoreWriterCpp, ObjectStoreWriterHandle), VortexError> {
        let inner = Arc::new(ObjectStoreWriterInner::new(fs_rawptr, path)?);
        let writer = Self {
            inner: inner.clone(),
            handle: handle.clone(),
        };
        let control = Self { inner, handle };

        Ok((
            ObjectStoreWriterCpp { writer },
            ObjectStoreWriterHandle { writer: control },
        ))
    }

    async fn write_byte_buffer(&self, buffer: ByteBuffer) -> io::Result<ByteBuffer> {
        let inner = self.inner.clone();
        self.handle
            .spawn_blocking(move || {
                inner.write(buffer.as_slice())?;
                Ok::<ByteBuffer, io::Error>(buffer)
            })
            .await
    }

    async fn flush(&self) -> Result<(), VortexError> {
        let inner = self.inner.clone();
        self.handle.spawn_blocking(move || inner.flush()).await
    }

    async fn close(&self) -> Result<(), VortexError> {
        let inner = self.inner.clone();
        self.handle.spawn_blocking(move || inner.close()).await
    }
}

pub struct ObjectStoreWriterHandle {
    writer: ObjectStoreWriter,
}

impl ObjectStoreWriterHandle {
    pub async fn flush(&self) -> Result<(), VortexError> {
        self.writer.flush().await
    }

    pub async fn close(&self) -> Result<(), VortexError> {
        self.writer.close().await
    }
}

pub struct ObjectStoreWriterCpp {
    writer: ObjectStoreWriter,
}

impl ObjectStoreWriterCpp {
    pub fn new(
        fs_rawptr: *mut c_void,
        path: &str,
        handle: Handle,
    ) -> Result<(Self, ObjectStoreWriterHandle), VortexError> {
        ObjectStoreWriter::new(fs_rawptr, path, handle)
    }
}

impl Write for ObjectStoreWriterCpp {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer
            .inner
            .flush()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }
}

fn into_vortex_write_byte_buffer<B: IoBuf>(buffer: B) -> Result<ByteBuffer, B> {
    let buffer: Box<dyn Any> = Box::new(buffer);
    match buffer.downcast::<ByteBuffer>() {
        Ok(buffer) => Ok(*buffer),
        Err(buffer) => Err(*buffer
            .downcast::<B>()
            .unwrap_or_else(|_| unreachable!("write buffer type changed during downcast"))),
    }
}

fn from_vortex_write_byte_buffer<B: IoBuf>(buffer: ByteBuffer) -> B {
    let buffer: Box<dyn Any> = Box::new(buffer);
    *buffer
        .downcast::<B>()
        .unwrap_or_else(|_| unreachable!("ByteBuffer write result requested as a different type"))
}

impl VortexWrite for ObjectStoreWriterCpp {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        let buffer = into_vortex_write_byte_buffer(buffer).map_err(|_| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "ObjectStoreWriterCpp only supports ByteBuffer writes without copying",
            )
        })?;
        self.writer
            .write_byte_buffer(buffer)
            .await
            .map(from_vortex_write_byte_buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.writer
            .flush()
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        // The current Vortex file writer does not call VortexWrite::shutdown()
        // during finish(); it only flushes. Keep this a no-op so object commit
        // remains controlled by ObjectStoreWriterHandle::close() after finish().
        Ok(())
    }
}

pub(crate) const DEFAULT_COALESCING_WINDOW: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,     // 1 MB
    max_size: 1 * 1024 * 1024, // 1 MB
};
const CONCURRENCY: usize = 256;

fn vortex_read_source_uri(path: &str) -> Arc<str> {
    if !path.contains("://") {
        return Arc::from(path.to_string());
    }

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    Arc::from(format!("vortex-object-{:016x}.vortex", hasher.finish()))
}

/// Arc-wrapped reader handle that ensures the underlying C++ RandomAccessFile
/// is only closed/destroyed after all concurrent read tasks are done.
struct ReaderHandle {
    ptr: *mut c_void,
    #[cfg_attr(not(feature = "s3-crt-async"), allow(dead_code))]
    supports_async: bool,
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
    coalesce_config: Option<CoalesceConfig>,
    handle: Handle,
}

impl ObjectStoreReadSourceCpp {
    pub fn new(
        fs_rawptr: *mut std::ffi::c_void,
        path: &str,
        file_size: u64,
        coalesce_config: CoalesceConfig,
        handle: Handle,
    ) -> VortexResult<Self> {
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
            check_loon_ffi_result(
                &mut result,
                "Failed to open reader in ObjectStoreReadSourceCpp",
            )?;
        }
        let supports_async = match reader_supports_async(reader_raw) {
            Ok(supported) => supported,
            Err(e) => unsafe {
                let mut result = loon_filesystem_reader_close(reader_raw);
                if let Err(close_err) =
                    check_loon_ffi_result(&mut result, "Failed to close ReaderHandle")
                {
                    eprintln!("Warning: ReaderHandle close failed: {close_err}");
                }
                loon_filesystem_reader_destroy(reader_raw);
                return Err(e);
            },
        };

        Ok(Self {
            inner: ThreadSafePtr::new(fs_rawptr),
            reader: Arc::new(ReaderHandle {
                ptr: reader_raw,
                supports_async,
            }),
            path: path.to_string(),
            uri: vortex_read_source_uri(path),
            coalesce_config: Some(coalesce_config),
            handle,
        })
    }
}

// Drop is handled by Arc<ReaderHandle> which closes and destroys the reader.

impl VortexReadAt for ObjectStoreReadSourceCpp {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        // move owned values into the async block so the future is 'static
        let inner = self.inner.clone();
        let path = self.path.clone();
        let handle = self.handle.clone();
        Compat::new(async move {
            // Pass path as bytes to FFI (no allocation across FFI boundaries)
            let path_bytes = path.into_bytes();
            // Return Result from blocking task to propagate errors cleanly
            let task = handle.spawn_blocking(move || unsafe {
                let mut out_size: u64 = 0;
                let mut result = loon_filesystem_get_file_info(
                    inner.as_ptr(),
                    path_bytes.as_ptr(),
                    path_bytes.len() as u32,
                    &mut out_size,
                );
                check_loon_ffi_result(
                    &mut result,
                    "Failed to get object size from ObjectStoreIoSourceCpp",
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                Ok::<u64, std::io::Error>(out_size)
            });
            let size: u64 = Compat::new(task).await.map_err(|e| vortex_err!("{}", e))?;
            Ok(size)
        })
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: vortex::buffer::Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let reader = self.reader.clone();
        let handle = self.handle.clone();

        async move {
            let len = u64::try_from(length)
                .map_err(|_| vortex_err!("read length does not fit in u64"))?;

            #[cfg(feature = "s3-crt-async")]
            if reader.supports_async {
                let buffer = read_async_via_ffi(reader, offset, len, alignment).await?;
                return Ok(BufferHandle::new_host(buffer));
            }

            // Offload sync FFI to the blocking pool and reuse the pre-opened reader handle.
            let blocking = handle.spawn_blocking(move || -> VortexResult<ByteBuffer> {
                let trace_start = record_io_start();
                let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
                let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();

                let mut result = unsafe {
                    loon_filesystem_reader_readat(reader.as_ptr(), offset, len, out_data)
                };

                check_loon_ffi_result(
                    &mut result,
                    "Failed to readat from ObjectStoreReadSourceCpp",
                )?;
                record_io_end(trace_start, offset, len);

                unsafe { buffer.set_len(length) };
                Ok(buffer.freeze())
            });

            let buffer: ByteBuffer = Compat::new(blocking).await?;
            Ok(BufferHandle::new_host(buffer))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vortex_write<T: vortex::io::VortexWrite + Unpin>() {}

    #[test]
    fn byte_buffer_write_input_round_trips_without_copy() {
        let buffer = ByteBuffer::copy_from([1_u8, 2, 3, 4]);
        let ptr = buffer.as_ptr();

        let buffer = match into_vortex_write_byte_buffer(buffer) {
            Ok(buffer) => buffer,
            Err(_) => panic!("ByteBuffer should be accepted"),
        };
        assert_eq!(buffer.as_ptr(), ptr);

        let buffer: ByteBuffer = from_vortex_write_byte_buffer(buffer);
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn non_byte_buffer_write_input_is_rejected() {
        let buffer = vec![1_u8, 2, 3, 4];
        let ptr = buffer.as_ptr();

        let buffer = match into_vortex_write_byte_buffer(buffer) {
            Ok(_) => panic!("Vec<u8> should not be accepted"),
            Err(buffer) => buffer,
        };
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn object_store_writer_cpp_is_a_vortex_write_sink() {
        assert_vortex_write::<ObjectStoreWriterCpp>();
    }
}
