// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Read-only `object_store` adapter backed by a C++ Arrow filesystem.
//!
//! A CXX `SharedPtr` keeps the C++ filesystem alive for every Rust clone. Object paths stay
//! relative to the filesystem's bound bucket or normalized local root, so reads reuse the
//! already-resolved credentials, retries, and metrics instead of constructing another provider.
//! Metadata is single-flighted and retained in a bounded LRU cache; mutation methods deliberately
//! return `NotImplemented`.
//!
//! Each FFI range read is materialized before `get_opts` returns so storage errors remain inside
//! the outer Lance AIMD retry boundary. For CRT asynchronous reads, callback state owns both the
//! destination buffer and reader handle until native completion. Cancelling the awaiting Rust task
//! drops only the receiver and cannot invalidate memory still used by the callback.

use std::ffi::c_void;
use std::fmt;
use std::future::Future;
use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Component, Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use cxx::SharedPtr;
use futures::StreamExt;
#[cfg(feature = "s3-crt-async")]
use futures::channel::oneshot;
use futures::stream::{self, BoxStream};
use lru::LruCache;
use object_store::path::Path as ObjectPath;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
};

use crate::TOKIO_RT;
use crate::filesystem_c::{
    LoonFFIError, LoonFileInfoList, ReaderHandle, check_loon_ffi_result, loon_errcode_arrow,
    loon_errcode_aws_access_denied, loon_errcode_aws_not_found,
    loon_errcode_aws_precondition_failed, loon_errcode_file_not_found, loon_errcode_not_support,
    loon_errcode_permission_denied, loon_errcode_transient_throttling,
    loon_filesystem_free_file_info_list, loon_filesystem_get_object_info, loon_filesystem_list_dir,
    loon_filesystem_open_reader, loon_filesystem_reader_close, loon_filesystem_reader_destroy,
    loon_filesystem_reader_readat, reader_supports_async,
};
#[cfg(feature = "s3-crt-async")]
use crate::filesystem_c::{LoonFFIResult, loon_filesystem_reader_readat_async};
use crate::lance_ffi::FileSystemWrapper;

#[cfg(feature = "s3-crt-async")]
struct ObjectStoreAsyncReadCallbackState {
    sender: Option<oneshot::Sender<(object_store::Result<Bytes>, Arc<ReaderHandle>)>>,
    reader: Arc<ReaderHandle>,
    buffer: BytesMut,
    location: ObjectPath,
    expected_len: usize,
}

#[cfg(feature = "s3-crt-async")]
unsafe extern "C" fn object_store_async_read_callback(
    user_data: *mut c_void,
    mut result: LoonFFIResult,
    bytes_read: u64,
) {
    if user_data.is_null() {
        let _ = check_loon_ffi_result(&mut result, "async object read failed");
        return;
    }

    // SAFETY: a successful native submission transfers exactly one Box to an
    // exactly-once callback. Immediate submission failure is reclaimed by the
    // caller instead, so ownership cannot be shared between both paths.
    let ObjectStoreAsyncReadCallbackState {
        mut sender,
        reader,
        mut buffer,
        location,
        expected_len,
    } = *unsafe { Box::from_raw(user_data.cast::<ObjectStoreAsyncReadCallbackState>()) };

    let read_result = match check_loon_ffi_result(&mut result, "async object read failed") {
        Err(error) => Err(into_object_store_error(error, &location)),
        Ok(()) if bytes_read != expected_len as u64 => Err(object_store::Error::Generic {
            store: "filesystem_c",
            source: Box::new(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("async object read returned {bytes_read} bytes, expected {expected_len}"),
            )),
        }),
        Ok(()) => {
            // SAFETY: native completion reported the exact requested length and
            // has finished writing those bytes into the allocation.
            unsafe { buffer.set_len(expected_len) };
            Ok(buffer.freeze())
        }
    };

    // Transfer the reader to the awaiting task so the CRT callback never drops
    // the last handle. If the task was cancelled, defer the drop off the CRT
    // callback thread instead.
    match sender.take() {
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
async fn read_object_store_async_via_ffi(
    reader: Arc<ReaderHandle>,
    location: ObjectPath,
    range: Range<u64>,
) -> object_store::Result<Bytes> {
    let length = range.end - range.start;
    let expected_len = usize::try_from(length).map_err(|source| object_store::Error::Generic {
        store: "filesystem_c",
        source: Box::new(source),
    })?;
    let mut buffer = BytesMut::with_capacity(expected_len);
    let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let (sender, receiver) = oneshot::channel();
    let state = Box::new(ObjectStoreAsyncReadCallbackState {
        sender: Some(sender),
        reader: reader.clone(),
        buffer,
        location: location.clone(),
        expected_len,
    });
    // Moving BytesMut into this Box keeps its allocation stable. After a
    // successful submission the callback owns the state, buffer, and reader;
    // an immediate submission error reclaims the same pointer below.
    let state_ptr = Box::into_raw(state);

    {
        let mut result = unsafe {
            loon_filesystem_reader_readat_async(
                reader.as_ptr(),
                range.start,
                length,
                out_data,
                object_store_async_read_callback,
                state_ptr.cast::<c_void>(),
            )
        };
        if result.err_code != 0 {
            unsafe { drop(Box::from_raw(state_ptr)) };
            check_loon_ffi_result(&mut result, "submit async object read")
                .map_err(|error| into_object_store_error(error, &location))?;
        }
    }
    drop(reader);

    let (read_result, completed_reader) =
        receiver.await.map_err(|_| object_store::Error::Generic {
            store: "filesystem_c",
            source: Box::new(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "async object read completion channel closed",
            )),
        })?;
    drop(completed_reader);
    read_result
}

#[derive(Clone, Debug)]
pub(crate) enum FFIPathMapper {
    Remote { scheme: String, authority: String },
    Local { normalized_root: PathBuf },
}

impl FFIPathMapper {
    pub(crate) fn remote(scheme: impl Into<String>, authority: impl Into<String>) -> Self {
        Self::Remote {
            scheme: scheme.into(),
            authority: authority.into(),
        }
    }

    pub(crate) fn local(root: impl AsRef<FsPath>) -> Self {
        Self::Local {
            normalized_root: root.as_ref().to_path_buf(),
        }
    }

    pub(crate) fn from_url(&self, url: &url::Url) -> object_store::Result<ObjectPath> {
        // Never turn an arbitrary URI directly into a relative object key. The
        // storage identity or local-root check must succeed first.
        match self {
            Self::Remote { scheme, authority } => {
                if url.scheme() != scheme || url.host_str() != Some(authority.as_str()) {
                    return Err(object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "URI storage identity does not match the bound filesystem",
                        )),
                    });
                }
                ObjectPath::from_url_path(url.path()).map_err(Into::into)
            }
            Self::Local { .. } => {
                let path = url
                    .to_file_path()
                    .map_err(|_| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "local dataset URI is not a valid file path",
                        )),
                    })?;
                self.from_local_path(path)
            }
        }
    }

    fn from_local_path(&self, path: impl AsRef<FsPath>) -> object_store::Result<ObjectPath> {
        let Self::Local { normalized_root } = self else {
            return Err(object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local path used with a remote filesystem",
                )),
            });
        };
        let path = path.as_ref();
        if !path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local dataset path must be absolute and must not contain '..'",
                )),
            });
        }

        let relative =
            path.strip_prefix(normalized_root)
                .map_err(|_| object_store::Error::Generic {
                    store: "filesystem_c",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "local dataset path is outside the bound filesystem root",
                    )),
                })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local dataset path is not valid UTF-8",
                )),
            })?
            .replace(std::path::MAIN_SEPARATOR, "/");
        ObjectPath::parse(relative).map_err(Into::into)
    }

    fn to_ffi_path(&self, location: &ObjectPath) -> String {
        location.to_string()
    }

    fn from_ffi_path(&self, path: &str) -> object_store::Result<ObjectPath> {
        ObjectPath::parse(path).map_err(Into::into)
    }
}

const FFI_OBJECT_META_CACHE_CAPACITY: usize = 256;
type MetaCell = Arc<tokio::sync::OnceCell<ObjectMeta>>;

#[derive(Clone)]
struct MetadataCache {
    entries: Arc<Mutex<LruCache<ObjectPath, MetaCell>>>,
}

impl MetadataCache {
    fn new() -> Self {
        // LruCache::new(capacity) preallocates the full hash table. Keep the
        // 256-entry limit while avoiding that fixed cost for every opened dataset.
        let mut entries = LruCache::new(NonZeroUsize::new(1).unwrap());
        entries.resize(NonZeroUsize::new(FFI_OBJECT_META_CACHE_CAPACITY).unwrap());
        Self {
            entries: Arc::new(Mutex::new(entries)),
        }
    }

    async fn get_or_load<F, Fut>(
        &self,
        location: ObjectPath,
        loader: F,
    ) -> object_store::Result<ObjectMeta>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = object_store::Result<ObjectMeta>>,
    {
        // The mutex protects only LRU lookup/insertion. OnceCell performs the
        // asynchronous single-flight after this scope releases the mutex.
        let cell = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(cell) = entries.get(&location) {
                cell.clone()
            } else {
                let cell = Arc::new(tokio::sync::OnceCell::new());
                entries.put(location, cell.clone());
                cell
            }
        };

        cell.get_or_try_init(loader).await.cloned()
    }

    fn prime(&self, meta: ObjectMeta) {
        let cell = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(cell) = entries.get(&meta.location) {
                cell.clone()
            } else {
                let cell = Arc::new(tokio::sync::OnceCell::new());
                entries.put(meta.location.clone(), cell.clone());
                cell
            }
        };
        let _ = cell.set(meta);
    }
}

struct FileInfoListGuard(LoonFileInfoList);

impl Drop for FileInfoListGuard {
    fn drop(&mut self) {
        unsafe { loon_filesystem_free_file_info_list(&mut self.0) }
    }
}

fn is_not_found_error_code(code: i32) -> bool {
    code == unsafe { loon_errcode_file_not_found } || code == unsafe { loon_errcode_aws_not_found }
}

#[derive(Debug)]
struct AimdThrottleError(LoonFFIError);

impl fmt::Display for AimdThrottleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Lance 7 identifies throttling from this RetryError fragment because
        // object_store does not expose a typed throttling error for custom stores.
        write!(f, "{}; AIMD compatibility: retries, max_retries", self.0)
    }
}

impl std::error::Error for AimdThrottleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

fn into_object_store_error(error: LoonFFIError, path: &ObjectPath) -> object_store::Error {
    let code = error.err_code;
    if is_not_found_error_code(code) {
        return object_store::Error::NotFound {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_permission_denied }
        || code == unsafe { loon_errcode_aws_access_denied }
    {
        return object_store::Error::PermissionDenied {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_aws_precondition_failed } {
        return object_store::Error::Precondition {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_not_support } {
        return object_store::Error::NotSupported {
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_transient_throttling } {
        return object_store::Error::Generic {
            store: "filesystem_c",
            source: Box::new(AimdThrottleError(error)),
        };
    }
    object_store::Error::Generic {
        store: "filesystem_c",
        source: Box::new(error),
    }
}

fn validate_get_options(options: &GetOptions) -> object_store::Result<()> {
    let unsupported = if options.version.is_some() {
        Some("version")
    } else if options.if_match.is_some() {
        Some("if_match")
    } else if options.if_none_match.is_some() {
        Some("if_none_match")
    } else if options.if_modified_since.is_some() {
        Some("if_modified_since")
    } else if options.if_unmodified_since.is_some() {
        Some("if_unmodified_since")
    } else {
        None
    };

    match unsupported {
        Some(option) => Err(object_store::Error::NotSupported {
            source: Box::new(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("filesystem_c does not support get option '{option}'"),
            )),
        }),
        None => Ok(()),
    }
}

#[derive(Clone)]
pub(crate) struct FilesystemObjectStore {
    filesystem: SharedPtr<FileSystemWrapper>,
    mapper: FFIPathMapper,
    metadata: MetadataCache,
}

impl FilesystemObjectStore {
    pub(crate) fn new(filesystem: SharedPtr<FileSystemWrapper>, mapper: FFIPathMapper) -> Self {
        Self {
            filesystem,
            mapper,
            metadata: MetadataCache::new(),
        }
    }

    fn object_meta(location: ObjectPath, size: u64, mtime_ns: i64) -> ObjectMeta {
        let last_modified = if mtime_ns <= 0 {
            chrono::DateTime::<chrono::Utc>::UNIX_EPOCH
        } else {
            let seconds = mtime_ns.div_euclid(1_000_000_000);
            let nanoseconds = mtime_ns.rem_euclid(1_000_000_000) as u32;
            chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanoseconds)
                .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
        };
        ObjectMeta {
            location,
            last_modified,
            size,
            e_tag: None,
            version: None,
        }
    }

    async fn load_metadata(&self, location: &ObjectPath) -> object_store::Result<ObjectMeta> {
        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(location);
        let object_path = location.clone();
        self.metadata
            .get_or_load(location.clone(), move || async move {
                // Arrow filesystem metadata calls are synchronous and may issue
                // remote I/O, so they must not block a Tokio worker thread.
                TOKIO_RT
                    .spawn_blocking(move || {
                        let path_len = u32::try_from(ffi_path.len()).map_err(|error| {
                            object_store::Error::Generic {
                                store: "filesystem_c",
                                source: Box::new(error),
                            }
                        })?;
                        let mut size = 0;
                        let mut mtime_ns = 0;
                        let mut is_dir = false;
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
                        check_loon_ffi_result(&mut result, "get object metadata")
                            .map_err(|error| into_object_store_error(error, &object_path))?;
                        if is_dir {
                            return Err(object_store::Error::Generic {
                                store: "filesystem_c",
                                source: Box::new(io::Error::new(
                                    io::ErrorKind::IsADirectory,
                                    format!("{} is a directory", object_path),
                                )),
                            });
                        }
                        Ok(Self::object_meta(object_path, size, mtime_ns))
                    })
                    .await
                    .map_err(|source| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    })?
            })
            .await
    }

    async fn list_impl(
        &self,
        prefix: ObjectPath,
        recursive: bool,
    ) -> object_store::Result<ListResult> {
        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(&prefix);
        let error_path = prefix.clone();
        // Keep C allocation access and copying inside the blocking task. The
        // guard invokes the matching C free function before the task returns.
        let entries = TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|error| {
                    object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(error),
                    }
                })?;
                let mut list = FileInfoListGuard(LoonFileInfoList {
                    entries: std::ptr::null_mut(),
                    count: 0,
                });
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
                    check_loon_ffi_result(&mut result, "list filesystem directory")
                {
                    if is_not_found_error_code(list_error.err_code) {
                        return Ok(Vec::new());
                    }

                    // Arrow directory listing rejects an exact file path with
                    // ENOTDIR. ObjectStore listing defines exact-file and
                    // nonexistent prefixes as successful empty results. Only
                    // probe metadata after this generic Arrow failure so normal
                    // directory lists do not pay for an extra remote request.
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
                            "inspect failed filesystem list prefix",
                        ) {
                            Ok(()) if !is_dir => return Ok(Vec::new()),
                            Err(info_error) if is_not_found_error_code(info_error.err_code) => {
                                return Ok(Vec::new());
                            }
                            Ok(()) | Err(_) => {}
                        }
                    }

                    return Err(into_object_store_error(list_error, &error_path));
                }

                if list.0.count != 0 && list.0.entries.is_null() {
                    return Err(object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "filesystem list returned a null entries pointer",
                        )),
                    });
                }

                if list.0.count == 0 {
                    return Ok(Vec::new());
                }

                let infos =
                    unsafe { std::slice::from_raw_parts(list.0.entries, list.0.count as usize) };
                let mut owned = Vec::with_capacity(infos.len());
                for info in infos {
                    if info.path.is_null() {
                        return Err(object_store::Error::Generic {
                            store: "filesystem_c",
                            source: Box::new(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "filesystem list returned a null path",
                            )),
                        });
                    }
                    let bytes = unsafe {
                        std::slice::from_raw_parts(info.path.cast::<u8>(), info.path_len as usize)
                    };
                    let path = std::str::from_utf8(bytes)
                        .map_err(|source| object_store::Error::Generic {
                            store: "filesystem_c",
                            source: Box::new(source),
                        })?
                        .to_string();
                    owned.push((path, info.is_dir, info.size, info.mtime_ns));
                }
                Ok::<_, object_store::Error>(owned)
            })
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })??;

        let mut objects = Vec::new();
        let mut common_prefixes = Vec::new();
        for (path, is_dir, size, mtime_ns) in entries {
            let location = self.mapper.from_ffi_path(&path)?;
            if is_dir {
                if !recursive {
                    common_prefixes.push(location);
                }
            } else {
                let meta = Self::object_meta(location, size, mtime_ns);
                self.metadata.prime(meta.clone());
                objects.push(meta);
            }
        }
        objects.sort_by(|left, right| left.location.cmp(&right.location));
        common_prefixes.sort();
        Ok(ListResult {
            common_prefixes,
            objects,
        })
    }

    async fn read_range(
        &self,
        location: &ObjectPath,
        size: u64,
        range: Range<u64>,
    ) -> object_store::Result<Bytes> {
        let length =
            range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| object_store::Error::Generic {
                    store: "filesystem_c",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "read range end precedes its start",
                    )),
                })?;
        if length == 0 {
            return Ok(Bytes::new());
        }
        let length_usize =
            usize::try_from(length).map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })?;

        enum ReadOutcome {
            #[cfg(feature = "s3-crt-async")]
            Async(Arc<ReaderHandle>),
            Complete(Bytes),
        }

        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(location);
        let object_path = location.clone();
        let offset = range.start;
        // Passing the metadata size avoids a second HEAD request. Open and
        // complete blocking reads in one task; CRT readers return their handle
        // to the existing callback path instead.
        let outcome = TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|source| {
                    object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    }
                })?;
                let mut reader_raw = std::ptr::null_mut();
                let mut result = unsafe {
                    loon_filesystem_open_reader(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        size,
                        &mut reader_raw,
                    )
                };
                check_loon_ffi_result(&mut result, "open object reader")
                    .map_err(|error| into_object_store_error(error, &object_path))?;

                let supports_async = match reader_supports_async(reader_raw) {
                    Ok(supported) => supported,
                    Err(error) => unsafe {
                        let mut close_result = loon_filesystem_reader_close(reader_raw);
                        if let Err(close_error) =
                            check_loon_ffi_result(&mut close_result, "close object reader")
                        {
                            eprintln!("Warning: ReaderHandle close failed: {close_error}");
                        }
                        loon_filesystem_reader_destroy(reader_raw);
                        return Err(into_object_store_error(error, &object_path));
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
                let mut result = unsafe {
                    loon_filesystem_reader_readat(reader.as_ptr(), offset, length, out_data)
                };
                check_loon_ffi_result(&mut result, "read object range")
                    .map_err(|error| into_object_store_error(error, &object_path))?;
                // SAFETY: the FFI contract returns success only after filling the
                // entire requested range; the allocation stays inside this task.
                unsafe { buffer.set_len(length_usize) };
                Ok(ReadOutcome::Complete(buffer.freeze()))
            })
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })??;

        match outcome {
            #[cfg(feature = "s3-crt-async")]
            ReadOutcome::Async(reader) => {
                read_object_store_async_via_ffi(reader, location.clone(), range).await
            }
            ReadOutcome::Complete(bytes) => Ok(bytes),
        }
    }
}

impl fmt::Debug for FilesystemObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FilesystemObjectStore(filesystem_c)")
    }
}

impl fmt::Display for FilesystemObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("filesystem_c")
    }
}

#[async_trait]
impl object_store::ObjectStore for FilesystemObjectStore {
    async fn put_opts(
        &self,
        _location: &ObjectPath,
        _payload: PutPayload,
        _options: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(object_store::Error::NotImplemented {
            operation: "put_opts".into(),
            implementer: "FilesystemObjectStore".into(),
        })
    }

    async fn put_multipart_opts(
        &self,
        _location: &ObjectPath,
        _options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(object_store::Error::NotImplemented {
            operation: "put_multipart_opts".into(),
            implementer: "FilesystemObjectStore".into(),
        })
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        validate_get_options(&options)?;
        let meta = self.load_metadata(location).await?;
        if options.head {
            return Ok(GetResult {
                payload: GetResultPayload::Stream(stream::once(async { Ok(Bytes::new()) }).boxed()),
                meta,
                range: 0..0,
                attributes: Attributes::default(),
            });
        }

        let range = match options.range {
            Some(range) => {
                range
                    .as_range(meta.size)
                    .map_err(|source| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    })?
            }
            None => 0..meta.size,
        };
        // FFI range reads return one complete Bytes value. Resolve it before
        // returning GetResult so the outer AIMD wrapper can classify and retry
        // throttling errors instead of losing them in the deferred payload.
        let bytes = self.read_range(location, meta.size, range.clone()).await?;
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes: Attributes::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        stream::once(async {
            Err(object_store::Error::NotImplemented {
                operation: "delete_stream".into(),
                implementer: "FilesystemObjectStore".into(),
            })
        })
        .boxed()
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let store = self.clone();
        let prefix = prefix.cloned().unwrap_or(ObjectPath::ROOT);
        async_stream::try_stream! {
            let result = store.list_impl(prefix, true).await?;
            for meta in result.objects {
                yield meta;
            }
        }
        .boxed()
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        self.list_impl(prefix.cloned().unwrap_or(ObjectPath::ROOT), false)
            .await
    }

    async fn copy_opts(
        &self,
        _from: &ObjectPath,
        _to: &ObjectPath,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented {
            operation: "copy_opts".into(),
            implementer: "FilesystemObjectStore".into(),
        })
    }

    async fn rename_opts(
        &self,
        _from: &ObjectPath,
        _to: &ObjectPath,
        _options: RenameOptions,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented {
            operation: "rename_opts".into(),
            implementer: "FilesystemObjectStore".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn remote_uri_path_is_relative_to_bucket() {
        let mapper = FFIPathMapper::remote("s3", "bucket");
        assert_eq!(
            mapper
                .from_url(&url::Url::parse("s3://bucket/a/b.lance").unwrap())
                .unwrap(),
            object_store::path::Path::from("a/b.lance")
        );
    }

    #[test]
    fn remote_uri_requires_matching_storage_identity() {
        let mapper = FFIPathMapper::remote("s3", "bucket");
        assert!(
            mapper
                .from_url(&url::Url::parse("gs://bucket/a.lance").unwrap())
                .is_err()
        );
        assert!(
            mapper
                .from_url(&url::Url::parse("s3://other/a.lance").unwrap())
                .is_err()
        );
    }

    #[test]
    fn local_path_is_relative_to_normalized_root() {
        let mapper = FFIPathMapper::local("/data/root/");
        assert_eq!(
            mapper.from_local_path("/data/root/a/b.lance").unwrap(),
            object_store::path::Path::from("a/b.lance")
        );
    }

    #[test]
    fn local_path_outside_root_is_rejected() {
        let mapper = FFIPathMapper::local("/data/root");
        assert!(mapper.from_local_path("/other/file.lance").is_err());
        assert!(
            mapper
                .from_local_path("/data/root/../other/file.lance")
                .is_err()
        );
    }

    #[test]
    fn versioned_and_conditional_gets_are_rejected() {
        let now = chrono::Utc::now();
        let unsupported = [
            (
                "version",
                object_store::GetOptions::new().with_version(Some("7")),
            ),
            (
                "if_match",
                object_store::GetOptions::new().with_if_match(Some("etag")),
            ),
            (
                "if_none_match",
                object_store::GetOptions::new().with_if_none_match(Some("etag")),
            ),
            (
                "if_modified_since",
                object_store::GetOptions::new().with_if_modified_since(Some(now)),
            ),
            (
                "if_unmodified_since",
                object_store::GetOptions::new().with_if_unmodified_since(Some(now)),
            ),
        ];

        for (name, options) in unsupported {
            assert!(
                validate_get_options(&options).is_err(),
                "{name} must be rejected"
            );
        }

        assert!(
            validate_get_options(
                &object_store::GetOptions::new()
                    .with_range(Some(1_u64..2))
                    .with_head(true)
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn metadata_cache_single_flights_concurrent_loads() {
        let cache = Arc::new(MetadataCache::new());
        let loads = Arc::new(AtomicUsize::new(0));
        let location = object_store::path::Path::from("same.lance");
        let mut tasks = Vec::new();

        for _ in 0..16 {
            let cache = cache.clone();
            let loads = loads.clone();
            let location = location.clone();
            tasks.push(tokio::spawn(async move {
                cache
                    .get_or_load(location.clone(), || async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok(object_store::ObjectMeta {
                            location,
                            last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                            size: 42,
                            e_tag: None,
                            version: None,
                        })
                    })
                    .await
            }));
        }

        for task in tasks {
            assert_eq!(task.await.unwrap().unwrap().size, 42);
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn metadata_cache_retries_failed_initialization() {
        let cache = MetadataCache::new();
        let location = object_store::path::Path::from("retry.lance");
        let first = cache
            .get_or_load(location.clone(), || async {
                Err(object_store::Error::Generic {
                    store: "test",
                    source: "first load failed".into(),
                })
            })
            .await;
        assert!(first.is_err());

        let second = cache
            .get_or_load(location.clone(), || async {
                Ok(object_store::ObjectMeta {
                    location,
                    last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                    size: 7,
                    e_tag: None,
                    version: None,
                })
            })
            .await
            .unwrap();
        assert_eq!(second.size, 7);
    }

    #[test]
    fn metadata_cache_capacity_is_bounded() {
        let cache = MetadataCache::new();
        for index in 0..(FFI_OBJECT_META_CACHE_CAPACITY + 10) {
            cache.prime(object_store::ObjectMeta {
                location: object_store::path::Path::from(format!("{index}.lance")),
                last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                size: index as u64,
                e_tag: None,
                version: None,
            });
        }

        assert_eq!(
            cache.entries.lock().unwrap().len(),
            FFI_OBJECT_META_CACHE_CAPACITY
        );
    }
}
