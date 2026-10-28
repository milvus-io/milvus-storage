use std::{
    ffi::{CString, c_char, c_void},
    panic::AssertUnwindSafe,
    sync::Arc,
};

use anyhow::{Result, bail};
use futures::FutureExt;
use talon::{
    BlockReadError, CacheReadError, Client, ClientBuilder, ObjectId, ObjectStat, RequestOptions,
    UNCHECKED_VERSION, parse_uri,
};
use tokio::sync::OnceCell;

#[cxx::bridge(namespace = "milvus_storage::talon::ffi")]
pub mod ffi {
    #[repr(i32)]
    enum TalonErrorCode {
        InvalidArgument = 1,
        Io = 2,
        NotFound = 3,
        Timeout = 4,
        Unavailable = 5,
        RateLimited = 6,
        VersionMismatch = 7,
        NonRetryable = 8,
    }

    extern "Rust" {
        type TalonClient;
        type TalonObjectReader;

        fn new_talon_client(
            coordinator: &str,
            block_size: u32,
            max_idle_per_addr: u32,
            namespace_name: &str,
            credential_broker: &str,
        ) -> Result<Box<TalonClient>>;
        fn with_namespace(self: &TalonClient, namespace_name: &str) -> Box<TalonClient>;
        fn with_credential_broker(self: &TalonClient, credential_broker: &str) -> Box<TalonClient>;
        fn open_talon_object(
            client: &TalonClient,
            cloud_provider: &str,
            bucket: &str,
            key: &str,
            has_initial_stat: bool,
            initial_size: u64,
            initial_version: &str,
        ) -> Result<Box<TalonObjectReader>>;
        fn talon_object_known_size(reader: &TalonObjectReader) -> i64;
    }
}

type TalonIoCallback = unsafe extern "C" fn(
    context: *mut c_void,
    error_code: i32,
    value: u64,
    error_msg: *const c_char,
);

struct TalonIoResult {
    code: i32,
    value: u64,
    message: String,
}

pub struct TalonClient {
    inner: Arc<TalonClientState>,
}

// Keep one descriptor per shared client, including requests submitted before
// the C++ owner is released. Readers only clone this Arc.
struct TalonClientState {
    client: Arc<Client>,
    namespace: Option<Arc<str>>,
    credential_broker: Option<Arc<str>>,
}

impl TalonClientState {
    fn request_options(&self) -> RequestOptions<'_> {
        let options =
            RequestOptions::default().with_credential_broker(self.credential_broker.as_deref());
        match self.namespace.as_deref() {
            Some(namespace) => options.with_namespace(namespace),
            None => options,
        }
    }
}

#[derive(Clone)]
pub struct TalonObjectReader {
    client: Arc<TalonClientState>,
    object: ObjectId,
    stat: Arc<OnceCell<ObjectStat>>,
}

// Static requests retain deployment-configured origins. Broker requests carry
// their own origin and credential scope; GCP then uses Talon's bearer-token backend.
fn talon_uri(provider: &str, bucket: &str, key: &str, broker_credentials: bool) -> Result<String> {
    let scheme = match provider {
        "gcp" if broker_credentials => "gcs",
        // Milvus accesses these providers through their S3-compatible APIs, so
        // the corresponding Talon deployment must use its S3 backend as well.
        "aws" | "aliyun" | "tencent" | "huawei" | "gcp" | "volcengine" => "s3",
        "azure" => "az",
        value => bail!("Talon does not support cloud provider {value:?}"),
    };
    Ok(format!("{scheme}://{bucket}/{key}"))
}

pub fn new_talon_client(
    coordinator: &str,
    block_size: u32,
    max_idle_per_addr: u32,
    namespace_name: &str,
    credential_broker: &str,
) -> Result<Box<TalonClient>> {
    Ok(Box::new(TalonClient {
        inner: Arc::new(TalonClientState {
            client: Arc::new(
                ClientBuilder::default()
                    .with_coordinator(coordinator)
                    .with_block_size(block_size)
                    .with_max_idle_per_addr(max_idle_per_addr as usize)
                    .build()?,
            ),
            namespace: (!namespace_name.is_empty()).then(|| Arc::from(namespace_name)),
            credential_broker: (!credential_broker.is_empty())
                .then(|| Arc::from(credential_broker)),
        }),
    }))
}

impl TalonClient {
    // Each method creates an immutable request view. The unchanged option and
    // the native client's connection pools and discovery state remain shared.
    pub fn with_namespace(&self, namespace_name: &str) -> Box<Self> {
        Box::new(Self {
            inner: Arc::new(TalonClientState {
                client: Arc::clone(&self.inner.client),
                namespace: (!namespace_name.is_empty()).then(|| Arc::from(namespace_name)),
                credential_broker: self.inner.credential_broker.clone(),
            }),
        })
    }

    pub fn with_credential_broker(&self, credential_broker: &str) -> Box<Self> {
        Box::new(Self {
            inner: Arc::new(TalonClientState {
                client: Arc::clone(&self.inner.client),
                namespace: self.inner.namespace.clone(),
                credential_broker: (!credential_broker.is_empty())
                    .then(|| Arc::from(credential_broker)),
            }),
        })
    }
}

pub fn open_talon_object(
    client: &TalonClient,
    cloud_provider: &str,
    bucket: &str,
    key: &str,
    has_initial_stat: bool,
    initial_size: u64,
    initial_version: &str,
) -> Result<Box<TalonObjectReader>> {
    if bucket.is_empty() || key.is_empty() {
        bail!("Talon bucket and object key must be non-empty");
    }
    let initial_stat = has_initial_stat.then(|| ObjectStat {
        size: initial_size,
        version: if initial_version.is_empty() {
            UNCHECKED_VERSION.into()
        } else {
            initial_version.into()
        },
    });
    Ok(Box::new(TalonObjectReader {
        client: Arc::clone(&client.inner),
        object: parse_uri(&talon_uri(
            cloud_provider,
            bucket,
            key,
            client.inner.credential_broker.is_some(),
        )?)?,
        stat: Arc::new(OnceCell::new_with(initial_stat)),
    }))
}

pub fn talon_object_known_size(reader: &TalonObjectReader) -> i64 {
    reader
        .stat
        .get()
        .map(|stat| stat.size)
        .and_then(|size| i64::try_from(size).ok())
        .unwrap_or(-1)
}

impl TalonObjectReader {
    async fn resolved_stat(&self) -> Result<&ObjectStat, talon::Error> {
        self.stat
            .get_or_try_init(|| async {
                self.client
                    .client
                    .stat(&self.object, &self.client.request_options())
                    .await
            })
            .await
    }
}

struct ReadBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ReadBuffer {}

impl ReadBuffer {
    unsafe fn into_mut_slice(self, len: usize) -> &'static mut [u8] {
        debug_assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

fn talon_error_result(error: talon::Error) -> TalonIoResult {
    // Capture the complete diagnostic before unwrapping the typed source so
    // exhausted replicas retain both the last worker address and its failure.
    let message = error.to_string();
    let code = match error {
        talon::Error::InvalidUri(_) | talon::Error::InvalidArgument(_) => {
            ffi::TalonErrorCode::InvalidArgument
        }
        talon::Error::Coordinator(error) => cache_error_code(error.into()),
        talon::Error::Block(error) => match error {
            BlockReadError::Coordinator(error) => cache_error_code(error.into()),
            BlockReadError::Discovery(error) => cache_error_code(error.as_ref().into()),
            BlockReadError::Worker(error)
            | BlockReadError::Target { source: error, .. }
            | BlockReadError::AllReplicasFailed { source: error, .. } => {
                cache_error_code(error.into())
            }
            BlockReadError::NoOwners => ffi::TalonErrorCode::Unavailable,
        },
    };
    TalonIoResult {
        code: code.repr,
        value: 0,
        message,
    }
}

fn cache_error_code(error: CacheReadError) -> ffi::TalonErrorCode {
    match error {
        CacheReadError::InvalidRequest(_) => ffi::TalonErrorCode::InvalidArgument,
        CacheReadError::NotFound(_) => ffi::TalonErrorCode::NotFound,
        CacheReadError::Timeout(_) => ffi::TalonErrorCode::Timeout,
        CacheReadError::Unavailable(_) => ffi::TalonErrorCode::Unavailable,
        CacheReadError::RateLimited(_) => ffi::TalonErrorCode::RateLimited,
        CacheReadError::VersionMismatch(_) => ffi::TalonErrorCode::VersionMismatch,
        CacheReadError::Origin(_)
        | CacheReadError::Protocol(_)
        | CacheReadError::Internal(_)
        | CacheReadError::Unknown(_)
        | CacheReadError::CacheMiss(_) => ffi::TalonErrorCode::NonRetryable,
    }
}

fn complete_talon_io(callback: TalonIoCallback, context: *mut c_void, result: TalonIoResult) {
    let error_msg = if result.code == 0 {
        std::ptr::null()
    } else {
        CString::new(result.message.replace('\0', "\\0"))
            .unwrap()
            .into_raw()
    };
    unsafe { callback(context, result.code, result.value, error_msg) };
}

fn complete_talon_error(
    callback: TalonIoCallback,
    context: *mut c_void,
    code: i32,
    message: impl ToString,
) {
    complete_talon_io(
        callback,
        context,
        TalonIoResult {
            code,
            value: 0,
            message: message.to_string(),
        },
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_dispatch(
    callback: unsafe extern "C" fn(*mut c_void),
    context: *mut c_void,
) {
    let context_addr = context as usize;
    // Metadata submission can fall back to a synchronous provider, and final
    // origin SDK destruction can wait for callbacks. Reuse the shared blocking
    // pool for these short bridge tasks. Native CRT HEAD returns its future
    // immediately, so no thread is held while that request is pending.
    crate::TOKIO_RT.spawn_blocking(move || unsafe { callback(context_addr as *mut c_void) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_free_error_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_object_stat_async(
    reader: *const c_void,
    callback: TalonIoCallback,
    context: *mut c_void,
) {
    let Some(reader) = (unsafe { (reader as *const TalonObjectReader).as_ref() }) else {
        complete_talon_error(callback, context, 1, "Talon stat received a null reader");
        return;
    };
    let reader = reader.clone();
    let context_addr = context as usize;
    crate::TOKIO_RT.spawn(async move {
        let result = match AssertUnwindSafe(reader.resolved_stat())
            .catch_unwind()
            .await
        {
            Ok(Ok(stat)) => TalonIoResult {
                code: 0,
                value: stat.size,
                message: String::new(),
            },
            Ok(Err(error)) => talon_error_result(error),
            Err(_) => TalonIoResult {
                code: 2,
                value: 0,
                message: "Talon stat task panicked".to_string(),
            },
        };
        complete_talon_io(callback, context_addr as *mut c_void, result);
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_object_read_async(
    reader: *const c_void,
    offset: u64,
    length: u64,
    dst: *mut u8,
    callback: TalonIoCallback,
    context: *mut c_void,
) {
    if offset.checked_add(length).is_none() {
        complete_talon_error(callback, context, 1, "Talon read range overflows u64");
        return;
    }
    if length > isize::MAX as u64 {
        complete_talon_error(callback, context, 1, "Talon read length exceeds isize::MAX");
        return;
    }
    if length > 0 && dst.is_null() {
        complete_talon_error(callback, context, 1, "Talon read destination is null");
        return;
    }
    let Some(reader) = (unsafe { (reader as *const TalonObjectReader).as_ref() }) else {
        complete_talon_error(callback, context, 1, "Talon read received a null reader");
        return;
    };

    let reader = reader.clone();
    let buffer = ReadBuffer {
        ptr: dst,
        len: length as usize,
    };
    let context_addr = context as usize;
    crate::TOKIO_RT.spawn(async move {
        let read_result = AssertUnwindSafe(async {
            let stat = if buffer.len == 0 {
                None
            } else {
                Some(reader.resolved_stat().await?)
            };
            let len = stat.map_or(0, |stat| {
                (buffer.len as u64).min(stat.size.saturating_sub(offset)) as usize
            });
            // The SDK validates request options even for empty reads, without I/O.
            let dst = if len == 0 {
                &mut []
            } else {
                unsafe { buffer.into_mut_slice(len) }
            };
            reader
                .client
                .client
                .read_into(
                    &reader.object,
                    offset,
                    dst,
                    stat,
                    &reader.client.request_options(),
                )
                .await
        })
        .catch_unwind()
        .await;

        let result = match read_result {
            Ok(Ok(bytes_written)) => TalonIoResult {
                code: 0,
                value: bytes_written as u64,
                message: String::new(),
            },
            Ok(Err(error)) => talon_error_result(error),
            Err(_) => TalonIoResult {
                code: 2,
                value: 0,
                message: "Talon read task panicked".to_string(),
            },
        };
        complete_talon_io(callback, context_addr as *mut c_void, result);
    });
}

// These records borrow their strings and arrays until the completion callback returns.
#[repr(C)]
pub struct TalonRawRequest {
    uri: *const u8,
    uri_len: usize,
    size: u64,
    version: *const u8,
    version_len: usize,
}
#[repr(C)]
struct TalonRawLoadResult {
    size: u64,
    blocks: u64,
}
#[repr(C)]
struct TalonRawFailure {
    index: u64,
    uncertain: bool,
    error: *const c_char,
}
#[repr(C)]
pub struct TalonRawResult {
    code: i32,
    error: *const c_char,
    size: u64,
    version: *const c_char,
    blocks: u64,
    results: *const TalonRawLoadResult,
    results_len: usize,
    failures: *const TalonRawFailure,
    failures_len: usize,
}
type TalonRawCallback = unsafe extern "C" fn(*mut c_void, *const TalonRawResult);

unsafe fn raw_string(ptr: *const u8, len: usize) -> Result<String, talon::Error> {
    if len == 0 {
        return Ok(String::new());
    }
    if ptr.is_null() {
        return Err(talon::Error::InvalidArgument(
            "null Talon input string".into(),
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| talon::Error::InvalidArgument(error.to_string()))
}

fn complete_raw(
    callback: TalonRawCallback,
    context: usize,
    io: TalonIoResult,
    version: &str,
    blocks: u64,
    results: &[talon::LoadResult],
    failures: &[talon::LoadFailure],
) {
    // Escaping an interior NUL preserves the rest of the original diagnostic.
    let error = CString::new(io.message.replace('\0', "\\0")).unwrap();
    let version = CString::new(version.replace('\0', "\\0")).unwrap();
    let failure_errors = failures
        .iter()
        .map(|failure| CString::new(failure.error.replace('\0', "\\0")).unwrap())
        .collect::<Vec<_>>();
    let failures = failures
        .iter()
        .zip(&failure_errors)
        .map(|(failure, error)| TalonRawFailure {
            index: failure.index as u64,
            uncertain: failure.uncertain,
            error: error.as_ptr(),
        })
        .collect::<Vec<_>>();
    let results = results
        .iter()
        .map(|result| TalonRawLoadResult {
            size: result.size,
            blocks: result.blocks,
        })
        .collect::<Vec<_>>();
    let result = TalonRawResult {
        code: io.code,
        error: error.as_ptr(),
        size: io.value,
        version: version.as_ptr(),
        blocks,
        results: results.as_ptr(),
        results_len: results.len(),
        failures: failures.as_ptr(),
        failures_len: failures.len(),
    };
    unsafe { callback(context as *mut c_void, &result) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_raw_stat_async(
    client: *const c_void,
    uri: *const u8,
    uri_len: usize,
    callback: TalonRawCallback,
    context: *mut c_void,
) {
    let context = context as usize;
    let parsed = unsafe { raw_string(uri, uri_len) }.and_then(|uri| parse_uri(&uri));
    let object = match parsed {
        Ok(object) => object,
        Err(error) => {
            complete_raw(
                callback,
                context,
                talon_error_result(error),
                "",
                0,
                &[],
                &[],
            );
            return;
        }
    };
    // The C++ owner may be released immediately after submission.
    let client = Arc::clone(&unsafe { &*(client as *const TalonClient) }.inner);
    crate::TOKIO_RT.spawn(async move {
        match AssertUnwindSafe(client.client.stat(&object, &client.request_options()))
            .catch_unwind()
            .await
        {
            Ok(Ok(stat)) => complete_raw(
                callback,
                context,
                TalonIoResult {
                    code: 0,
                    value: stat.size,
                    message: String::new(),
                },
                &stat.version,
                0,
                &[],
                &[],
            ),
            Ok(Err(error)) => complete_raw(
                callback,
                context,
                talon_error_result(error),
                "",
                0,
                &[],
                &[],
            ),
            Err(_) => complete_raw(
                callback,
                context,
                TalonIoResult {
                    code: -1,
                    value: 0,
                    message: "Talon stat task panicked".into(),
                },
                "",
                0,
                &[],
                &[],
            ),
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_raw_read_async(
    client: *const c_void,
    uri: *const u8,
    uri_len: usize,
    offset: u64,
    length: u64,
    dst: *mut u8,
    has_stat: bool,
    size: u64,
    version: *const u8,
    version_len: usize,
    callback: TalonIoCallback,
    context: *mut c_void,
) {
    let parsed = unsafe { raw_string(uri, uri_len) }
        .and_then(|uri| parse_uri(&uri))
        .and_then(|object| {
            unsafe { raw_string(version, version_len) }.map(|version| (object, version))
        });
    let (object, version) = match parsed {
        Ok(value) => value,
        Err(error) => {
            complete_talon_io(callback, context, talon_error_result(error));
            return;
        }
    };
    let reader = TalonObjectReader {
        client: Arc::clone(&unsafe { &*(client as *const TalonClient) }.inner),
        object,
        stat: Arc::new(OnceCell::new_with(has_stat.then(|| ObjectStat {
            size,
            version: if version.is_empty() {
                UNCHECKED_VERSION.into()
            } else {
                version
            },
        }))),
    };
    // Existing submission clones reader state before returning and retains the buffer until completion.
    unsafe {
        talon_object_read_async(
            &reader as *const _ as *const c_void,
            offset,
            length,
            dst,
            callback,
            context,
        )
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn talon_raw_load_async(
    client: *const c_void,
    requests: *const TalonRawRequest,
    count: usize,
    batch: bool,
    callback: TalonRawCallback,
    context: *mut c_void,
) {
    let context = context as usize;
    let requests = if count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(requests, count) }
    };
    // Parse and own every URI/version before dispatch, including empty-file requests.
    let requests = requests
        .iter()
        .map(|request| {
            let object = parse_uri(&unsafe { raw_string(request.uri, request.uri_len) }?)?;
            let version = unsafe { raw_string(request.version, request.version_len) }?;
            Ok(talon::LoadRequest {
                object,
                size: request.size,
                version: if version.is_empty() {
                    talon::Version::new(UNCHECKED_VERSION)
                } else {
                    talon::Version::new(version)
                },
            })
        })
        .collect::<Result<Vec<_>, talon::Error>>();
    let requests = match requests {
        Ok(requests) => requests,
        Err(error) => {
            complete_raw(
                callback,
                context,
                talon_error_result(error),
                "",
                0,
                &[],
                &[],
            );
            return;
        }
    };
    let client = Arc::clone(&unsafe { &*(client as *const TalonClient) }.inner);
    crate::TOKIO_RT.spawn(async move {
        let operation = AssertUnwindSafe(async {
            if batch {
                client
                    .client
                    .batch_load(&requests, &client.request_options())
                    .await
            } else {
                let request = &requests[0];
                client
                    .client
                    .load(
                        &request.object,
                        &request.version,
                        request.size,
                        &client.request_options(),
                    )
                    .await
                    .map(|result| vec![result])
            }
        })
        .catch_unwind()
        .await;
        match operation {
            Ok(Ok(results)) => {
                let first = results.first();
                complete_raw(
                    callback,
                    context,
                    TalonIoResult {
                        code: 0,
                        value: first.map_or(0, |result| result.size),
                        message: String::new(),
                    },
                    "",
                    first.map_or(0, |result| result.blocks),
                    if batch { &results } else { &[] },
                    &[],
                );
            }
            Ok(Err(error)) => {
                // Preserve the outer diagnostic before moving the failure list and classifying the cause.
                let message = error.to_string();
                let (cause, failures) = match error {
                    talon::LoadError::Batch {
                        source,
                        failed_files,
                    } => (*source, failed_files),
                    other => (other, Vec::new()),
                };
                let mut result = talon_load_error_result(cause);
                result.message = message;
                complete_raw(callback, context, result, "", 0, &[], &failures);
            }
            Err(_) => complete_raw(
                callback,
                context,
                TalonIoResult {
                    code: -1,
                    value: 0,
                    message: "Talon load task panicked".into(),
                },
                "",
                0,
                &[],
                &[],
            ),
        }
    });
}

fn talon_load_error_result(error: talon::LoadError) -> TalonIoResult {
    // Retain the outer diagnostic, including the worker address and block offset.
    let message = error.to_string();
    let worker_code = |source: talon::WorkerLoadError| match source {
        talon::WorkerLoadError::Instance(error) => cache_error_code(error.into()),
        talon::WorkerLoadError::Io(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            ffi::TalonErrorCode::Timeout
        }
        talon::WorkerLoadError::Io(_) => ffi::TalonErrorCode::Unavailable,
        talon::WorkerLoadError::Codec(_)
        | talon::WorkerLoadError::Protocol(_)
        | talon::WorkerLoadError::Rejected(_) => ffi::TalonErrorCode::NonRetryable,
    };
    let code = match error {
        talon::LoadError::Batch { source, .. } => talon_load_error_result(*source).code,
        talon::LoadError::InvalidArgument(_) => ffi::TalonErrorCode::InvalidArgument.repr,
        talon::LoadError::Placement(error) => talon_error_result(error.into()).code,
        talon::LoadError::Worker { source, .. } | talon::LoadError::WorkerBatch { source, .. } => {
            worker_code(source).repr
        }
        talon::LoadError::Timeout => ffi::TalonErrorCode::Timeout.repr,
        talon::LoadError::MembershipChanged => ffi::TalonErrorCode::Unavailable.repr,
    };
    TalonIoResult {
        code,
        value: 0,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talon::{CoordinatorError, DataErrorCode, DataPlaneError, WorkerError};

    #[test]
    fn load_errors_preserve_classification_and_complete_diagnostics() {
        use talon::{LoadError, LoadFailure, WorkerLoadError};
        let errors = [
            (
                LoadError::InvalidArgument("bad size".into()),
                ffi::TalonErrorCode::InvalidArgument,
            ),
            (LoadError::Timeout, ffi::TalonErrorCode::Timeout),
            (
                LoadError::MembershipChanged,
                ffi::TalonErrorCode::Unavailable,
            ),
            (
                LoadError::Placement(BlockReadError::NoOwners),
                ffi::TalonErrorCode::Unavailable,
            ),
            (
                LoadError::Worker {
                    worker: "worker:9000".into(),
                    offset: 8388608,
                    source: WorkerLoadError::Io(std::io::ErrorKind::TimedOut.into()),
                },
                ffi::TalonErrorCode::Timeout,
            ),
            (
                LoadError::WorkerBatch {
                    worker: "worker:9000".into(),
                    source: WorkerLoadError::Rejected("original rejection".into()),
                },
                ffi::TalonErrorCode::NonRetryable,
            ),
            (
                LoadError::Batch {
                    failed_files: vec![LoadFailure {
                        index: 3,
                        uncertain: true,
                        error: "original rejection".into(),
                    }],
                    source: Box::new(LoadError::WorkerBatch {
                        worker: "worker:9000".into(),
                        source: WorkerLoadError::Io(std::io::ErrorKind::ConnectionReset.into()),
                    }),
                },
                ffi::TalonErrorCode::Unavailable,
            ),
        ];
        for (error, expected) in errors {
            let message = error.to_string();
            let result = talon_load_error_result(error);
            assert_eq!(result.code, expected.repr, "{message}");
            assert_eq!(result.message, message);
        }
    }

    #[test]
    fn worker_errors_preserve_codes_and_replica_diagnostics() {
        for (remote_code, expected) in [
            (
                DataErrorCode::InvalidRequest,
                ffi::TalonErrorCode::InvalidArgument,
            ),
            (DataErrorCode::NotFound, ffi::TalonErrorCode::NotFound),
            (DataErrorCode::Timeout, ffi::TalonErrorCode::Timeout),
            (DataErrorCode::Unavailable, ffi::TalonErrorCode::Unavailable),
            (DataErrorCode::RateLimited, ffi::TalonErrorCode::RateLimited),
            (
                DataErrorCode::VersionMismatch,
                ffi::TalonErrorCode::VersionMismatch,
            ),
            (DataErrorCode::Origin, ffi::TalonErrorCode::NonRetryable),
            (DataErrorCode::Internal, ffi::TalonErrorCode::NonRetryable),
            (DataErrorCode::Unknown, ffi::TalonErrorCode::NonRetryable),
            (DataErrorCode::CacheMiss, ffi::TalonErrorCode::NonRetryable),
        ] {
            for (targeted, exhausted) in [(false, false), (true, false), (true, true)] {
                let source = WorkerError::Remote(DataPlaneError {
                    code: remote_code,
                    // Classification must use the code, never diagnostic text.
                    message: "NotFound Timeout: original worker diagnostic".into(),
                });
                let error = if exhausted {
                    BlockReadError::AllReplicasFailed {
                        worker: "10.0.0.8:9000".into(),
                        worker_id: "worker-1".into(),
                        instance_id: Some("instance-1".into()),
                        source,
                    }
                } else if targeted {
                    BlockReadError::Target {
                        worker_id: "worker-1".into(),
                        instance_id: "instance-1".into(),
                        address: "10.0.0.8:9000".into(),
                        source,
                    }
                } else {
                    BlockReadError::Worker(source)
                };
                let message = error.to_string();
                let result = talon_error_result(error.into());
                assert_eq!(result.code, expected.repr, "{message}");
                assert_eq!(result.message, message);
                assert_eq!(result.value, 0);
                assert_eq!(result.message.contains("10.0.0.8:9000"), targeted);
                assert_eq!(result.message.contains("worker-1"), targeted);
                assert_eq!(result.message.contains("instance-1"), targeted);
                assert_eq!(result.message.contains("all replicas failed"), exhausted);
            }
        }
    }

    #[test]
    fn transport_and_placement_errors_keep_their_classification() {
        for (kind, expected) in [
            (std::io::ErrorKind::TimedOut, ffi::TalonErrorCode::Timeout),
            (
                std::io::ErrorKind::ConnectionReset,
                ffi::TalonErrorCode::Unavailable,
            ),
        ] {
            let errors = [
                talon::Error::Coordinator(CoordinatorError::Io(kind.into())),
                BlockReadError::Coordinator(CoordinatorError::Io(kind.into())).into(),
                BlockReadError::Worker(WorkerError::Io(kind.into())).into(),
                BlockReadError::Target {
                    worker_id: "worker-1".into(),
                    instance_id: "instance-1".into(),
                    address: "10.0.0.8:9000".into(),
                    source: WorkerError::Io(kind.into()),
                }
                .into(),
                BlockReadError::AllReplicasFailed {
                    worker: "10.0.0.8:9000".into(),
                    worker_id: "worker-1".into(),
                    instance_id: Some("instance-1".into()),
                    source: WorkerError::Io(kind.into()),
                }
                .into(),
            ];
            for error in errors {
                assert_eq!(talon_error_result(error).code, expected.repr);
            }
        }
        assert_eq!(
            talon_error_result(BlockReadError::NoOwners.into()).code,
            ffi::TalonErrorCode::Unavailable.repr
        );
        assert_eq!(
            talon_error_result(talon::Error::InvalidArgument("bad range".into())).code,
            ffi::TalonErrorCode::InvalidArgument.repr
        );
        assert_eq!(
            talon_error_result(parse_uri("invalid").unwrap_err()).code,
            ffi::TalonErrorCode::InvalidArgument.repr
        );
        let protocol = BlockReadError::Worker(WorkerError::RangeLengthMismatch {
            expected: 10,
            actual: 1,
        });
        assert_eq!(
            talon_error_result(protocol.into()).code,
            ffi::TalonErrorCode::NonRetryable.repr
        );
    }

    #[test]
    fn maps_storage_providers_to_canonical_talon_uris() {
        for provider in ["aws", "aliyun", "tencent", "huawei", "gcp", "volcengine"] {
            assert_eq!(
                talon_uri(provider, "bucket", "path/a", false).unwrap(),
                "s3://bucket/path/a"
            );
        }
        assert_eq!(
            talon_uri("azure", "container", "path/a", false).unwrap(),
            "az://container/path/a"
        );
        assert!(talon_uri("local", "bucket", "path/a", false).is_err());
    }

    #[test]
    fn open_object_with_stat_initializes_metadata_without_resolving() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024, 8, "", "").unwrap();
        let reader = open_talon_object(
            &client,
            "aws",
            "test-bucket",
            "path/a",
            true,
            123,
            "etag-v1",
        )
        .unwrap();
        assert_eq!(talon_object_known_size(&reader), 123);
        assert_eq!(reader.stat.get().unwrap().size, 123);
        assert_eq!(reader.stat.get().unwrap().version, "etag-v1");
    }

    #[test]
    fn open_object_with_size_only_skips_etag_matching() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024, 8, "", "").unwrap();
        for size in [0, 123] {
            let reader =
                open_talon_object(&client, "aws", "test-bucket", "path/a", true, size, "").unwrap();
            assert_eq!(talon_object_known_size(&reader), size as i64);
            assert_eq!(reader.stat.get().unwrap().version, "__talon_unchecked__");
        }
    }

    #[test]
    fn open_object_without_stat_leaves_metadata_unresolved() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024, 8, "", "").unwrap();
        let reader =
            open_talon_object(&client, "aws", "test-bucket", "path/a", false, 0, "").unwrap();
        assert_eq!(talon_object_known_size(&reader), -1);
        assert!(reader.stat.get().is_none());
    }

    #[test]
    fn resolved_stat_is_shared_by_reader_clones() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024, 8, "", "").unwrap();
        let reader =
            open_talon_object(&client, "aws", "test-bucket", "path/a", false, 0, "").unwrap();
        reader
            .stat
            .set(ObjectStat {
                size: 7,
                version: "v1".into(),
            })
            .unwrap();
        let clone = reader.clone();
        assert_eq!(clone.stat.get().unwrap().size, 7);
        assert_eq!(clone.stat.get().unwrap().version, "v1");
        assert_eq!(talon_object_known_size(&clone), 7);
    }
}
