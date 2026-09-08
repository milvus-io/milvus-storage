use std::{
    ffi::{CString, c_char, c_void},
    panic::AssertUnwindSafe,
    sync::Arc,
};

use anyhow::{Result, bail};
use futures::FutureExt;
use talon::{Client, ObjectId, ObjectStat, parse_uri};
use tokio::sync::OnceCell;

#[cxx::bridge(namespace = "milvus_storage::talon::ffi")]
pub mod ffi {
    extern "Rust" {
        type TalonClient;
        type TalonObjectReader;

        fn new_talon_client(coordinator: &str, block_size: u32) -> Result<Box<TalonClient>>;
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
    inner: Arc<Client>,
}

#[derive(Clone)]
pub struct TalonObjectReader {
    client: Arc<Client>,
    object: ObjectId,
    stat: Arc<OnceCell<ObjectStat>>,
}

// A Talon deployment serves one origin credential domain. The coordinator
// selects that deployment, so its endpoint, account, and credentials are
// deployment configuration rather than part of ObjectId. Within the selected
// origin, ObjectId deliberately consists only of backend, bucket/container,
// and key; a different origin must use a different Talon deployment.
fn talon_uri(provider: &str, bucket: &str, key: &str) -> Result<String> {
    let scheme = match provider {
        // Milvus accesses these providers through their S3-compatible APIs, so
        // the corresponding Talon deployment must use its S3 backend as well.
        "aws" | "aliyun" | "tencent" | "huawei" | "gcp" => "s3",
        "azure" => "az",
        value => bail!("Talon does not support cloud provider {value:?}"),
    };
    Ok(format!("{scheme}://{bucket}/{key}"))
}

pub fn new_talon_client(coordinator: &str, block_size: u32) -> Result<Box<TalonClient>> {
    Ok(Box::new(TalonClient {
        inner: Arc::new(Client::new(coordinator, block_size)?),
    }))
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
        version: initial_version.into(),
    });
    Ok(Box::new(TalonObjectReader {
        client: Arc::clone(&client.inner),
        object: parse_uri(&talon_uri(cloud_provider, bucket, key)?)?,
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
            .get_or_try_init(|| async { self.client.stat(&self.object).await })
            .await
    }
}

struct ReadBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ReadBuffer {}

impl ReadBuffer {
    unsafe fn into_mut_slice(self) -> &'static mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

fn talon_error_result(error: talon::Error) -> TalonIoResult {
    let code = match &error {
        talon::Error::InvalidUri(_) | talon::Error::InvalidArgument(_) => 1,
        talon::Error::Coordinator(_) | talon::Error::Block(_) => 2,
    };
    TalonIoResult {
        code,
        value: 0,
        message: error.to_string(),
    }
}

fn complete_talon_io(callback: TalonIoCallback, context: *mut c_void, result: TalonIoResult) {
    let error_msg = if result.code == 0 {
        std::ptr::null()
    } else {
        CString::new(result.message)
            .unwrap_or_else(|_| {
                CString::new("Talon IO error contains an interior NUL byte").unwrap()
            })
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
            if buffer.len == 0 {
                return Ok(0usize);
            }
            let stat = reader.resolved_stat().await?;
            let dst = unsafe { buffer.into_mut_slice() };
            reader
                .client
                .read_into(&reader.object, offset, dst, Some(stat))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_storage_providers_to_canonical_talon_uris() {
        for provider in ["aws", "aliyun", "tencent", "huawei", "gcp"] {
            assert_eq!(
                talon_uri(provider, "bucket", "path/a").unwrap(),
                "s3://bucket/path/a"
            );
        }
        assert_eq!(
            talon_uri("azure", "container", "path/a").unwrap(),
            "az://container/path/a"
        );
        assert!(talon_uri("local", "bucket", "path/a").is_err());
    }

    #[test]
    fn open_object_with_stat_initializes_metadata_without_resolving() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024).unwrap();
        let reader =
            open_talon_object(&client, "aws", "test-bucket", "path/a", true, 123, "").unwrap();
        assert_eq!(talon_object_known_size(&reader), 123);
        assert_eq!(reader.stat.get().unwrap().size, 123);
        assert!(reader.stat.get().unwrap().version.is_empty());
    }

    #[test]
    fn open_object_without_stat_leaves_metadata_unresolved() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024).unwrap();
        let reader =
            open_talon_object(&client, "aws", "test-bucket", "path/a", false, 0, "").unwrap();
        assert_eq!(talon_object_known_size(&reader), -1);
        assert!(reader.stat.get().is_none());
    }

    #[test]
    fn resolved_stat_is_shared_by_reader_clones() {
        let client = new_talon_client("127.0.0.1:7000", 8 * 1024 * 1024).unwrap();
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
