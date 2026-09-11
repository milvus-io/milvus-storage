// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Read-only OpenDAL access backed by the cached C++ Arrow filesystem.
//!
//! This module is the storage-agnostic bridge between OpenDAL and the filesystem
//! FFI. Callers provide paths relative to the bound filesystem; this layer does
//! not parse cloud URIs, select buckets, or build native cloud providers.
//!
//! Data flows from OpenDAL `stat` / `read` / `list` operations through the C
//! ABI into the shared C++ filesystem. The C++ side remains the owner of cloud
//! clients, credentials, retries, metrics, and provider lifetimes.

use std::ffi::c_void;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
#[cfg(feature = "s3-crt-async")]
use futures::channel::oneshot;
use opendal::raw::AccessorInfo;
use opendal::{Capability, Operator, OperatorBuilder};

use crate::TOKIO_RT;
use crate::filesystem_c::{
    LoonFFIError, LoonFileInfoList, ReaderHandle, check_loon_ffi_result,
    loon_errcode_arrow, loon_errcode_aws_access_denied, loon_errcode_aws_not_found,
    loon_errcode_aws_precondition_failed, loon_errcode_file_not_found,
    loon_errcode_not_support, loon_errcode_permission_denied, loon_errcode_transient_throttling,
    loon_filesystem_free_file_info_list, loon_filesystem_get_object_info,
    loon_filesystem_list_dir, loon_filesystem_open_reader, loon_filesystem_reader_close,
    loon_filesystem_reader_destroy, loon_filesystem_reader_readat, reader_supports_async,
};
#[cfg(feature = "s3-crt-async")]
use crate::filesystem_c::{LoonFFIResult, loon_filesystem_reader_readat_async};

// TODO(jiaqizho): Move shared stat/read/list and CRT reader lifecycle plumbing
// from the ObjectStore and OpenDAL adapters into filesystem_c.rs in a separate
// PR. Keep adapter-specific errors and result types at their current boundaries.

// -----------------------------------------------------------------------------
// FFI error and metadata translation
// -----------------------------------------------------------------------------

/// C++ filesystems use both generic and provider-specific not-found codes.
/// OpenDAL must see one stable category so its higher layers can handle them
/// without knowing which cloud implementation owns the request.
fn is_not_found_error_code(code: i32) -> bool {
    // SAFETY: these are immutable error-code constants exported by the C++ ABI.
    code == unsafe { loon_errcode_file_not_found } || code == unsafe { loon_errcode_aws_not_found }
}

/// Preserve actionable C++ storage categories at the OpenDAL boundary.
///
/// In particular, throttling stays both `RateLimited` and temporary. Consumers
/// above OpenDAL can therefore retain retry/AIMD behavior instead of seeing an
/// undifferentiated bridge failure.
fn into_opendal_error(
    error: LoonFFIError,
    operation: &'static str,
    path: &str,
) -> opendal::Error {
    let code = error.err_code;
    let kind = if is_not_found_error_code(code) {
        opendal::ErrorKind::NotFound
    } else if code == unsafe { loon_errcode_permission_denied }
        || code == unsafe { loon_errcode_aws_access_denied }
    {
        opendal::ErrorKind::PermissionDenied
    } else if code == unsafe { loon_errcode_aws_precondition_failed } {
        opendal::ErrorKind::ConditionNotMatch
    } else if code == unsafe { loon_errcode_not_support } {
        opendal::ErrorKind::Unsupported
    } else if code == unsafe { loon_errcode_transient_throttling } {
        opendal::ErrorKind::RateLimited
    } else {
        opendal::ErrorKind::Unexpected
    };
    let result = opendal::Error::new(kind, "C++ filesystem operation failed")
        .with_operation(operation)
        .with_context("path", path)
        .set_source(error);
    if code == unsafe { loon_errcode_transient_throttling } {
        result.set_temporary()
    } else {
        result
    }
}

/// Minimal metadata returned by the filesystem ABI before conversion to
/// OpenDAL's richer metadata type.
#[derive(Clone, Copy)]
struct ObjectInfo {
    size: u64,
    mtime_ns: i64,
    is_dir: bool,
}

impl ObjectInfo {
    fn into_metadata(self) -> opendal::Result<opendal::Metadata> {
        let mode = if self.is_dir {
            opendal::EntryMode::DIR
        } else {
            opendal::EntryMode::FILE
        };
        let mut metadata = opendal::Metadata::new(mode);
        metadata.set_content_length(self.size);
        if self.mtime_ns > 0 {
            let seconds = self.mtime_ns.div_euclid(1_000_000_000);
            let nanoseconds = self.mtime_ns.rem_euclid(1_000_000_000) as i32;
            metadata.set_last_modified(opendal::raw::Timestamp::new(seconds, nanoseconds)?);
        }
        Ok(metadata)
    }
}

/// Owns a C-allocated listing until every borrowed entry has been copied into
/// Rust-owned OpenDAL entries.
struct FileInfoListGuard(LoonFileInfoList);

impl Drop for FileInfoListGuard {
    fn drop(&mut self) {
        // SAFETY: this guard is the unique owner of the list returned by the
        // filesystem ABI, and the free function accepts an empty list as well.
        unsafe { loon_filesystem_free_file_info_list(&mut self.0) }
    }
}

/// OpenDAL's streaming list interface over a listing materialized by C++.
///
/// The C ABI returns a complete vector, so this lister only advances through
/// already-owned entries and performs no additional filesystem calls.
pub(crate) struct FilesystemLister {
    entries: std::vec::IntoIter<opendal::raw::oio::Entry>,
}

impl FilesystemLister {
    fn new(entries: Vec<opendal::raw::oio::Entry>) -> Self {
        Self {
            entries: entries.into_iter(),
        }
    }
}

impl opendal::raw::oio::List for FilesystemLister {
    async fn next(&mut self) -> opendal::Result<Option<opendal::raw::oio::Entry>> {
        Ok(self.entries.next())
    }
}

// -----------------------------------------------------------------------------
// Optional CRT asynchronous range-read bridge
// -----------------------------------------------------------------------------

#[cfg(feature = "s3-crt-async")]
/// State whose ownership is transferred to the C callback.
///
/// Keeping the reader and buffer in the same box guarantees that the native
/// reader and the buffer allocation outlive the asynchronous write. A successful
/// submission transfers this box to exactly one callback; an immediate submit
/// failure reclaims it synchronously in `read_async_via_ffi`.
struct FilesystemAsyncReadCallbackState {
    sender: Option<oneshot::Sender<(opendal::Result<Bytes>, Arc<ReaderHandle>)>>,
    reader: Arc<ReaderHandle>,
    buffer: BytesMut,
    path: String,
    expected_len: usize,
}

#[cfg(feature = "s3-crt-async")]
fn send_async_read_completion<R: Send + 'static>(
    sender: Option<oneshot::Sender<(opendal::Result<Bytes>, R)>>,
    read_result: opendal::Result<Bytes>,
    reader: R,
) {
    match sender {
        Some(sender) => {
            if let Err((_read_result, reader)) = sender.send((read_result, reader)) {
                let _ = TOKIO_RT.spawn_blocking(move || drop(reader));
            }
        }
        None => {
            let _ = TOKIO_RT.spawn_blocking(move || drop(reader));
        }
    }
}

#[cfg(feature = "s3-crt-async")]
/// # Safety
///
/// `user_data` must be either null or the unique pointer returned by
/// `Box::into_raw` in `read_async_via_ffi`. After successful submission the C
/// ABI must invoke this callback exactly once.
unsafe extern "C" fn filesystem_async_read_callback(
    user_data: *mut c_void,
    mut result: LoonFFIResult,
    bytes_read: u64,
) {
    if user_data.is_null() {
        let _ = check_loon_ffi_result(&mut result, "async OpenDAL read failed");
        return;
    }

    // SAFETY: `user_data` came from `Box::into_raw` below. The filesystem ABI
    // promises one callback after a successful submission and no callback after
    // an immediate submission error, so this is the unique ownership recovery.
    let FilesystemAsyncReadCallbackState {
        sender,
        reader,
        mut buffer,
        path,
        expected_len,
    } = *unsafe { Box::from_raw(user_data.cast::<FilesystemAsyncReadCallbackState>()) };
    let read_result = match check_loon_ffi_result(&mut result, "async OpenDAL read failed") {
        Err(error) => Err(into_opendal_error(error, "read", &path)),
        Ok(()) if bytes_read != expected_len as u64 => Err(opendal::Error::new(
            opendal::ErrorKind::Unexpected,
            format!("filesystem read returned {bytes_read} bytes, expected {expected_len}"),
        )
        .with_operation("read")
        .with_context("path", &path)),
        Ok(()) => {
            // SAFETY: C++ reported exactly `expected_len` initialized bytes in
            // the allocation's spare capacity, and the allocation stayed pinned
            // inside the callback state for the entire native operation.
            unsafe { buffer.set_len(expected_len) };
            Ok(buffer.freeze())
        }
    };

    // Transfer the reader to the awaiting task so the CRT callback never drops
    // the last handle. If the task was cancelled, defer the drop off the CRT
    // callback thread instead.
    send_async_read_completion(sender, read_result, reader);
}

#[cfg(feature = "s3-crt-async")]
/// Submit one range read without blocking a Tokio worker.
///
/// The callback state is deliberately self-contained: cancellation can drop the
/// receiver, but the native operation still owns the state until its callback
/// transfers or defers the reader release and frees the buffer safely.
async fn read_async_via_ffi(
    reader: Arc<ReaderHandle>,
    path: String,
    range: Range<u64>,
) -> opendal::Result<Bytes> {
    let length = range.end - range.start;
    let expected_len = usize::try_from(length).map_err(|source| {
        opendal::Error::new(
            opendal::ErrorKind::Unexpected,
            "filesystem read length does not fit in usize",
        )
        .with_operation("read")
        .with_context("path", &path)
        .set_source(source)
    })?;
    let mut buffer = BytesMut::with_capacity(expected_len);
    // The pointer remains stable because ownership of `buffer` moves into the
    // callback state and no Rust code grows or reallocates it while C++ writes.
    let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let (sender, receiver) = oneshot::channel();
    let state = Box::new(FilesystemAsyncReadCallbackState {
        sender: Some(sender),
        reader: reader.clone(),
        buffer,
        path: path.clone(),
        expected_len,
    });
    // Ownership crosses the FFI boundary only after this conversion. See the
    // callback's matching `Box::from_raw` and the immediate-error branch below.
    let state_ptr = Box::into_raw(state);

    {
        let mut result = unsafe {
            loon_filesystem_reader_readat_async(
                reader.as_ptr(),
                range.start,
                length,
                out_data,
                filesystem_async_read_callback,
                state_ptr.cast::<c_void>(),
            )
        };
        if result.err_code != 0 {
            // Submission failed synchronously, so the ABI will not invoke the
            // callback and Rust must recover the state itself.
            unsafe { drop(Box::from_raw(state_ptr)) };
            check_loon_ffi_result(&mut result, "submit async OpenDAL read")
                .map_err(|error| into_opendal_error(error, "read", &path))?;
        }
    }
    // The callback state holds its own Arc until completion transfers that
    // reader back to the awaiting task.
    drop(reader);

    let (read_result, completed_reader) = receiver.await.map_err(|source| {
        opendal::Error::new(
            opendal::ErrorKind::Unexpected,
            "async filesystem read completion channel closed",
        )
        .with_operation("read")
        .with_context("path", &path)
        .set_source(source)
    })?;
    drop(completed_reader);
    read_result
}

/// Convert OpenDAL's open-ended range representation into a bounded file range.
/// Reads are clipped at EOF, while an offset beyond EOF is rejected explicitly.
fn resolve_read_range(range: opendal::raw::BytesRange, size: u64) -> opendal::Result<Range<u64>> {
    let start = range.offset();
    if start > size {
        return Err(opendal::Error::new(
            opendal::ErrorKind::RangeNotSatisfied,
            "filesystem read range starts beyond the end of the object",
        )
        .with_operation("read"));
    }
    let available = size - start;
    let length = range.size().unwrap_or(available).min(available);
    Ok(start..start + length)
}

// -----------------------------------------------------------------------------
// Generic read-only OpenDAL Access implementation
// -----------------------------------------------------------------------------

/// OpenDAL accessor that delegates relative paths to one shared C++ filesystem.
///
/// The shared pointer is the lifetime lease: cloning this accessor keeps the
/// cached filesystem alive even if `FilesystemCache` is cleared concurrently.
/// The wrapper is immutable after construction and is declared Send/Sync at the
/// CXX boundary because the underlying Arrow filesystem is concurrency-safe.
pub(crate) struct FilesystemAccess {
    filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
    info: Arc<AccessorInfo>,
}

impl fmt::Debug for FilesystemAccess {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Do not expose the filesystem pointer or provider configuration.
        formatter
            .debug_struct("FilesystemAccess")
            .finish_non_exhaustive()
    }
}

impl FilesystemAccess {
    /// Bind OpenDAL to an existing filesystem. This layer never creates or
    /// selects a cloud provider on its own.
    pub(crate) fn new(
        filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
    ) -> opendal::Result<Self> {
        if filesystem.is_null() {
            return Err(opendal::Error::new(
                opendal::ErrorKind::ConfigInvalid,
                "filesystem-backed OpenDAL access requires a filesystem",
            ));
        }
        Ok(Self::from_filesystem(filesystem))
    }

    fn from_filesystem(
        filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
    ) -> Self {
        let info = AccessorInfo::default();
        // Advertise only operations implemented below. OpenDAL rejects all
        // mutation methods before they can reach the filesystem ABI.
        info.set_scheme("milvus-filesystem")
            .set_root("/")
            .set_name("milvus-filesystem")
            .set_native_capability(Capability {
                stat: true,
                read: true,
                list: true,
                ..Default::default()
            });
        Self {
            filesystem,
            info: Arc::new(info),
        }
    }

    /// Finish the raw accessor without adding retry or credential layers. Those
    /// policies remain owned by the C++ filesystem.
    pub(crate) fn into_operator(self) -> Operator {
        OperatorBuilder::new(self).finish()
    }

    /// Fetch metadata for one relative filesystem path.
    ///
    /// No URI normalization belongs here; callers must bind and strip absolute
    /// URIs before entering the generic access layer.
    async fn object_info(&self, path: &str) -> opendal::Result<ObjectInfo> {
        // OpenDAL treats its empty root as a directory. Avoid a provider call
        // for this synthetic entry because some object stores have no root key.
        if path.is_empty() {
            return Ok(ObjectInfo {
                size: 0,
                mtime_ns: 0,
                is_dir: true,
            });
        }

        let filesystem = self.filesystem.clone();
        let ffi_path = path.to_string();
        let error_path = ffi_path.clone();
        // Arrow filesystem calls are synchronous. Move them off Tokio workers;
        // the cloned shared pointer keeps the C++ object alive in the task.
        TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|source| {
                    opendal::Error::new(
                        opendal::ErrorKind::Unexpected,
                        "filesystem path length does not fit in u32",
                    )
                    .with_operation("stat")
                    .with_context("path", &ffi_path)
                    .set_source(source)
                })?;
                let mut size = 0;
                let mut mtime_ns = 0;
                let mut is_dir = false;
                // SAFETY: the shared filesystem and path bytes outlive this
                // synchronous call; all output pointers reference stack values.
                let mut result = unsafe {
                    loon_filesystem_get_object_info(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        &mut size,
                        &mut mtime_ns,
                        &mut is_dir,
                    )
                };
                check_loon_ffi_result(&mut result, "get OpenDAL object metadata")
                    .map_err(|error| into_opendal_error(error, "stat", &ffi_path))?;
                Ok(ObjectInfo {
                    size,
                    mtime_ns,
                    is_dir,
                })
            })
            .await
            .map_err(|source| {
                opendal::Error::new(
                    opendal::ErrorKind::Unexpected,
                    "filesystem metadata blocking task failed",
                )
                .with_operation("stat")
                .with_context("path", &error_path)
                .set_source(source)
            })?
    }

    /// Read one already-bounded range from a relative path.
    ///
    /// A reader handle is opened per OpenDAL read request. Its RAII wrapper
    /// closes the C++ RandomAccessFile after either the callback or blocking
    /// ReadAt path completes.
    async fn read_range(
        &self,
        path: &str,
        range: Range<u64>,
        size: u64,
    ) -> opendal::Result<Bytes> {
        if range.start > range.end || range.end > size {
            return Err(opendal::Error::new(
                opendal::ErrorKind::RangeNotSatisfied,
                "filesystem read range is outside the object",
            )
            .with_operation("read")
            .with_context("path", path));
        }
        let length = range.end - range.start;
        if length == 0 {
            return Ok(Bytes::new());
        }
        let length_usize = usize::try_from(length).map_err(|source| {
            opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                "filesystem read length does not fit in usize",
            )
            .with_operation("read")
            .with_context("path", path)
            .set_source(source)
        })?;

        enum ReadOutcome {
            #[cfg(feature = "s3-crt-async")]
            Async(Arc<ReaderHandle>),
            Complete(Bytes),
        }

        let filesystem = self.filesystem.clone();
        let ffi_path = path.to_string();
        let error_path = ffi_path.clone();
        let offset = range.start;
        // Open and complete blocking reads in one task. CRT readers return their
        // handle so the existing callback path can submit without occupying a
        // blocking-pool thread while network I/O is in flight.
        let outcome = TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|source| {
                    opendal::Error::new(
                        opendal::ErrorKind::Unexpected,
                        "filesystem path length does not fit in u32",
                    )
                    .with_operation("read")
                    .with_context("path", &ffi_path)
                    .set_source(source)
                })?;
                let mut reader_raw = std::ptr::null_mut();
                // SAFETY: inputs stay alive for the synchronous open call and
                // `reader_raw` is initialized by C++ on success.
                let mut result = unsafe {
                    loon_filesystem_open_reader(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        size,
                        &mut reader_raw,
                    )
                };
                check_loon_ffi_result(&mut result, "open OpenDAL object reader")
                    .map_err(|error| into_opendal_error(error, "read", &ffi_path))?;

                let supports_async = match reader_supports_async(reader_raw) {
                    Ok(supported) => supported,
                    Err(error) => unsafe {
                        // Ownership has not reached ReaderHandle yet. Close and
                        // destroy the raw reader on this error path explicitly.
                        let mut close_result = loon_filesystem_reader_close(reader_raw);
                        if let Err(close_error) =
                            check_loon_ffi_result(&mut close_result, "close OpenDAL object reader")
                        {
                            eprintln!("Warning: ReaderHandle close failed: {close_error}");
                        }
                        loon_filesystem_reader_destroy(reader_raw);
                        return Err(into_opendal_error(error, "read", &ffi_path));
                    },
                };
                let reader = ReaderHandle {
                    ptr: reader_raw,
                    supports_async,
                };

                #[cfg(feature = "s3-crt-async")]
                if reader.supports_async {
                    return Ok(ReadOutcome::Async(Arc::new(reader)));
                }

                let mut buffer = BytesMut::with_capacity(length_usize);
                let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
                // SAFETY: ReaderHandle owns a valid native reader and the spare
                // capacity is at least `length` bytes for this synchronous call.
                let mut result = unsafe {
                    loon_filesystem_reader_readat(reader.as_ptr(), offset, length, out_data)
                };
                check_loon_ffi_result(&mut result, "read OpenDAL object range")
                    .map_err(|error| into_opendal_error(error, "read", &ffi_path))?;
                // SAFETY: a successful ReadAt fills the complete requested
                // range; the FFI contract rejects short reads.
                unsafe { buffer.set_len(length_usize) };
                Ok(ReadOutcome::Complete(buffer.freeze()))
            })
            .await
            .map_err(|source| {
                opendal::Error::new(
                    opendal::ErrorKind::Unexpected,
                    "filesystem read blocking task failed",
                )
                .with_operation("read")
                .with_context("path", &error_path)
                .set_source(source)
            })??;

        match outcome {
            #[cfg(feature = "s3-crt-async")]
            ReadOutcome::Async(reader) => {
                read_async_via_ffi(reader, path.to_string(), range).await
            }
            ReadOutcome::Complete(bytes) => Ok(bytes),
        }
    }

    /// Materialize one directory listing from the C filesystem ABI.
    ///
    /// Results are copied before the C allocation is released, then exposed
    /// through `FilesystemLister` as OpenDAL's asynchronous iterator shape.
    async fn list_entries(
        &self,
        path: &str,
        recursive: bool,
    ) -> opendal::Result<Vec<opendal::raw::oio::Entry>> {
        let filesystem = self.filesystem.clone();
        let ffi_path = path.to_string();
        let error_path = ffi_path.clone();
        // Listing is synchronous and returns C-owned memory. The guard below
        // releases that memory after all fields have been copied.
        TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|source| {
                    opendal::Error::new(
                        opendal::ErrorKind::Unexpected,
                        "filesystem path length does not fit in u32",
                    )
                    .with_operation("list")
                    .with_context("path", &ffi_path)
                    .set_source(source)
                })?;
                let mut list = FileInfoListGuard(LoonFileInfoList {
                    entries: std::ptr::null_mut(),
                    count: 0,
                });
                // SAFETY: inputs live through this synchronous call and `list`
                // is initialized by C++ then owned by FileInfoListGuard.
                let mut result = unsafe {
                    loon_filesystem_list_dir(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        recursive,
                        &mut list.0,
                    )
                };
                if let Err(list_error) =
                    check_loon_ffi_result(&mut result, "list OpenDAL filesystem directory")
                {
                    if is_not_found_error_code(list_error.err_code) {
                        // Object-store listing of a missing prefix is empty,
                        // unlike stat/read where not-found remains an error.
                        return Ok(Vec::new());
                    }

                    // Arrow rejects a directory-shaped path that resolves to a
                    // file with ENOTDIR. OpenDAL filesystem listing treats that
                    // path as an empty directory. Probe metadata only after this
                    // generic Arrow failure so normal lists pay no extra request.
                    if list_error.err_code == unsafe { loon_errcode_arrow } {
                        let mut size = 0;
                        let mut mtime_ns = 0;
                        let mut is_dir = false;
                        let mut info_result = unsafe {
                            loon_filesystem_get_object_info(
                                filesystem.as_mut_ptr().cast::<c_void>(),
                                ffi_path.as_ptr(),
                                path_len,
                                &mut size,
                                &mut mtime_ns,
                                &mut is_dir,
                            )
                        };
                        match check_loon_ffi_result(
                            &mut info_result,
                            "inspect failed OpenDAL filesystem list path",
                        ) {
                            Ok(()) if !is_dir => return Ok(Vec::new()),
                            Err(info_error) if is_not_found_error_code(info_error.err_code) => {
                                return Ok(Vec::new());
                            }
                            Ok(()) | Err(_) => {}
                        }
                    }

                    return Err(into_opendal_error(list_error, "list", &ffi_path));
                }
                if list.0.count != 0 && list.0.entries.is_null() {
                    return Err(opendal::Error::new(
                        opendal::ErrorKind::Unexpected,
                        "filesystem list returned a null entries pointer",
                    )
                    .with_operation("list")
                    .with_context("path", &ffi_path));
                }
                let infos = if list.0.count == 0 {
                    &[]
                } else {
                    // SAFETY: a nonzero count was validated to have a non-null
                    // pointer, and FileInfoListGuard keeps the allocation alive.
                    unsafe {
                        std::slice::from_raw_parts(list.0.entries, list.0.count as usize)
                    }
                };
                let mut entries = Vec::with_capacity(infos.len());
                for info in infos {
                    if info.path.is_null() {
                        return Err(opendal::Error::new(
                            opendal::ErrorKind::Unexpected,
                            "filesystem list returned a null path",
                        )
                        .with_operation("list")
                        .with_context("path", &ffi_path));
                    }
                    // SAFETY: each path allocation remains owned by `list` and
                    // is valid for `path_len` bytes until the guard is dropped.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            info.path.cast::<u8>(),
                            info.path_len as usize,
                        )
                    };
                    let mut entry_path = std::str::from_utf8(bytes)
                        .map_err(|source| {
                            opendal::Error::new(
                                opendal::ErrorKind::Unexpected,
                                "filesystem list returned a non-UTF-8 path",
                            )
                            .with_operation("list")
                            .with_context("path", &ffi_path)
                            .set_source(source)
                        })?
                        .to_string();
                    if info.is_dir && !entry_path.ends_with('/') {
                        entry_path.push('/');
                    }
                    let metadata = ObjectInfo {
                        size: info.size,
                        mtime_ns: info.mtime_ns,
                        is_dir: info.is_dir,
                    }
                    .into_metadata()?;
                    entries.push(opendal::raw::oio::Entry::new(&entry_path, metadata));
                }
                // Stable ordering keeps OpenDAL consumers deterministic even if
                // different Arrow filesystem providers enumerate differently.
                entries.sort_by(|left, right| left.path().cmp(right.path()));
                Ok(entries)
            })
            .await
            .map_err(|source| {
                opendal::Error::new(
                    opendal::ErrorKind::Unexpected,
                    "filesystem list blocking task failed",
                )
                .with_operation("list")
                .with_context("path", &error_path)
                .set_source(source)
            })?
    }
}

/// Raw OpenDAL surface exposed by the bridge. Mutation methods are not
/// overridden, so OpenDAL's `Access` defaults return Unsupported. The `()`
/// associated types and false capability flags reflect that same contract.
impl opendal::raw::Access for FilesystemAccess {
    type Reader = opendal::Buffer;
    type Writer = ();
    type Lister = FilesystemLister;
    type Deleter = ();

    fn info(&self) -> Arc<AccessorInfo> {
        self.info.clone()
    }

    async fn stat(
        &self,
        path: &str,
        _args: opendal::raw::OpStat,
    ) -> opendal::Result<opendal::raw::RpStat> {
        let info = self.object_info(path).await?;
        Ok(opendal::raw::RpStat::new(info.into_metadata()?))
    }

    async fn read(
        &self,
        path: &str,
        args: opendal::raw::OpRead,
    ) -> opendal::Result<(opendal::raw::RpRead, Self::Reader)> {
        // TODO(jiaqizho): Avoid this pre-read metadata lookup if multi-manifest snapshots
        // become common in this workload. Today the no-delete path reads one manifest,
        // so the potential saving is only three metadata RPCs per refresh (table
        // metadata, manifest list, and manifest). A future fix should reuse the C++
        // input stream's cached size rather than adding another metadata cache.
        // TODO(jiaqizho): Replace the Buffer reader with a stateful oio::Read before
        // enabling ranged FileRead consumers. Otherwise every range for one file
        // repeats this stat plus reader open and close.
        let info = self.object_info(path).await?;
        if info.is_dir {
            return Err(opendal::Error::new(
                opendal::ErrorKind::IsADirectory,
                "cannot read a directory",
            )
            .with_operation("read")
            .with_context("path", path));
        }
        let range = resolve_read_range(args.range(), info.size)?;
        let bytes = self.read_range(path, range, info.size).await?;
        let read_size = u64::try_from(bytes.len()).map_err(|source| {
            opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                "filesystem read size does not fit in u64",
            )
            .with_operation("read")
            .with_context("path", path)
            .set_source(source)
        })?;
        Ok((
            opendal::raw::RpRead::new().with_size(Some(read_size)),
            opendal::Buffer::from(bytes),
        ))
    }

    async fn list(
        &self,
        path: &str,
        args: opendal::raw::OpList,
    ) -> opendal::Result<(opendal::raw::RpList, Self::Lister)> {
        let entries = self.list_entries(path, args.recursive()).await?;
        Ok((
            opendal::raw::RpList::default(),
            FilesystemLister::new(entries),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filesystem_c::LoonFFIResult;
    use std::ffi::{CString, c_char};

    #[repr(C)]
    struct TestLoonProperty {
        key: *mut c_char,
        value: *mut c_char,
    }

    #[repr(C)]
    struct TestLoonProperties {
        properties: *mut TestLoonProperty,
        count: usize,
    }

    unsafe extern "C" {
        fn loon_filesystem_get(
            properties: *const TestLoonProperties,
            path: *const c_char,
            path_len: u32,
            out_handle: *mut usize,
        ) -> LoonFFIResult;
    }

    struct LocalOperatorFixture {
        operator: Operator,
        root: tempfile::TempDir,
    }

    #[cfg(feature = "s3-crt-async")]
    struct DropThreadProbe(std::sync::mpsc::Sender<std::thread::ThreadId>);

    #[cfg(feature = "s3-crt-async")]
    impl Drop for DropThreadProbe {
        fn drop(&mut self) {
            let _ = self.0.send(std::thread::current().id());
        }
    }

    impl LocalOperatorFixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();

            let storage_type_key = CString::new("fs.storage_type").unwrap();
            let root_path_key = CString::new("fs.root_path").unwrap();
            let storage_type = CString::new("local").unwrap();
            let root_path = CString::new(root.path().to_str().unwrap()).unwrap();
            let mut entries = [
                TestLoonProperty {
                    key: storage_type_key.as_ptr().cast_mut(),
                    value: storage_type.as_ptr().cast_mut(),
                },
                TestLoonProperty {
                    key: root_path_key.as_ptr().cast_mut(),
                    value: root_path.as_ptr().cast_mut(),
                },
            ];
            let properties = TestLoonProperties {
                properties: entries.as_mut_ptr(),
                count: entries.len(),
            };

            let mut handle = 0;
            let mut result =
                unsafe { loon_filesystem_get(&properties, std::ptr::null(), 0, &mut handle) };
            check_loon_ffi_result(&mut result, "create local filesystem for OpenDAL test").unwrap();
            assert_ne!(handle, 0);

            // SAFETY: `loon_filesystem_get` transfers a `FileSystemWrapper`
            // allocated by C++ `new`. SharedPtr adopts that allocation and is
            // its only owner on the Rust side.
            let filesystem = unsafe {
                cxx::SharedPtr::from_raw(handle as *mut crate::lance_ffi::FileSystemWrapper)
            };
            let operator = FilesystemAccess::new(filesystem).unwrap().into_operator();
            Self { operator, root }
        }
    }

    #[test]
    fn access_rejects_null_filesystem() {
        let result = FilesystemAccess::new(cxx::SharedPtr::null());

        assert!(result.is_err());
    }

    #[test]
    fn generic_permission_denied_maps_to_permission_denied() {
        let mut result = LoonFFIResult {
            err_code: unsafe { crate::filesystem_c::loon_errcode_permission_denied },
            message: std::ptr::null_mut(),
        };
        let error = check_loon_ffi_result(&mut result, "permission denied").unwrap_err();

        let error = into_opendal_error(error, "read", "restricted");

        assert_eq!(error.kind(), opendal::ErrorKind::PermissionDenied);
    }

    #[test]
    fn access_advertises_only_read_capabilities() {
        use opendal::raw::Access;

        let access = FilesystemAccess::from_filesystem(cxx::SharedPtr::null());
        let capability = access.info().native_capability();

        assert!(capability.stat);
        assert!(capability.read);
        assert!(capability.list);
        assert!(!capability.write);
        assert!(!capability.delete);
        assert!(!capability.copy);
        assert!(!capability.rename);
    }

    #[cfg(feature = "s3-crt-async")]
    #[test]
    fn canceled_async_completion_defers_reader_drop() {
        let callback_thread = std::thread::current().id();
        let (sender, receiver) = oneshot::channel();
        drop(receiver);
        let (drop_sender, drop_receiver) = std::sync::mpsc::channel();

        send_async_read_completion(
            Some(sender),
            Ok(Bytes::new()),
            DropThreadProbe(drop_sender),
        );

        let drop_thread = drop_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("deferred reader drop did not complete");
        assert_ne!(drop_thread, callback_thread);
    }

    #[tokio::test]
    async fn access_rejects_mutations_without_calling_filesystem() {
        let operator = FilesystemAccess::from_filesystem(cxx::SharedPtr::null()).into_operator();

        let write_error = operator
            .write("forbidden", Bytes::from_static(b"data"))
            .await
            .unwrap_err();
        let delete_error = operator.delete("forbidden").await.unwrap_err();

        assert_eq!(write_error.kind(), opendal::ErrorKind::Unsupported);
        assert_eq!(delete_error.kind(), opendal::ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn operator_stat_reports_file_metadata() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("payload.bin"), b"0123456789").unwrap();

        let metadata = fixture.operator.stat("payload.bin").await.unwrap();

        assert!(metadata.mode().is_file());
        assert_eq!(metadata.content_length(), 10);
    }

    #[tokio::test]
    async fn operator_stat_reports_directory_metadata() {
        let fixture = LocalOperatorFixture::new();
        std::fs::create_dir(fixture.root.path().join("nested")).unwrap();

        let metadata = fixture.operator.stat("nested/").await.unwrap();

        assert!(metadata.mode().is_dir());
        assert_eq!(metadata.content_length(), 0);
    }

    #[tokio::test]
    async fn operator_reads_complete_file_content() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("payload.bin"), b"0123456789").unwrap();

        let content = fixture.operator.read("payload.bin").await.unwrap();

        assert_eq!(content.to_bytes().as_ref(), b"0123456789");
    }

    #[tokio::test]
    async fn operator_reads_requested_byte_range() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("payload.bin"), b"0123456789").unwrap();

        let content = fixture
            .operator
            .read_with("payload.bin")
            .range(2..6)
            .await
            .unwrap();

        assert_eq!(content.to_bytes().as_ref(), b"2345");
    }

    #[tokio::test]
    async fn operator_reads_empty_file() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("empty.bin"), b"").unwrap();

        let content = fixture.operator.read("empty.bin").await.unwrap();

        assert!(content.is_empty());
    }

    #[tokio::test]
    async fn operator_lists_immediate_children() {
        let fixture = LocalOperatorFixture::new();
        std::fs::create_dir(fixture.root.path().join("nested")).unwrap();
        std::fs::write(fixture.root.path().join("empty.bin"), b"").unwrap();
        std::fs::write(fixture.root.path().join("payload.bin"), b"data").unwrap();
        std::fs::write(fixture.root.path().join("nested/child.bin"), b"child").unwrap();

        let entries = fixture.operator.list("").await.unwrap();

        assert_eq!(
            entries.iter().map(|entry| entry.path()).collect::<Vec<_>>(),
            vec!["empty.bin", "nested/", "payload.bin"]
        );
        assert!(entries[0].metadata().mode().is_file());
        assert_eq!(entries[0].metadata().content_length(), 0);
        assert!(entries[1].metadata().mode().is_dir());
        assert_eq!(entries[1].metadata().content_length(), 0);
        assert!(entries[2].metadata().mode().is_file());
        assert_eq!(entries[2].metadata().content_length(), 4);
    }

    #[tokio::test]
    async fn operator_lists_directory_shaped_file_path_as_empty() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("payload.bin"), b"data").unwrap();

        let entries = fixture.operator.list("payload.bin/").await.unwrap();

        assert!(entries.is_empty());
    }
}
