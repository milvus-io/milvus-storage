// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared error classification for the Rust cxx bridges.
//!
//! The cxx boundary can only carry an error as a display string
//! (`rust::Error::what()`), which used to destroy the typed error the Rust
//! side already had (`lance::Error` distinguishes not-found / corruption /
//! retryable contention; the C++ side then guessed a blanket classification).
//! To keep the classification across the string-only channel, an error code is
//! embedded into the message with a marker prefix that the C++ side parses and
//! strips (see cpp `bridge_error.h`), the same mechanism the vortex bridge
//! established in `filesystem_c.rs`.
//!
//! The marker is THE channel. It replaced an earlier thread-local side
//! channel (record beside the error, take after catching), which failed three
//! independent ways: an unclassified construction cleared a verdict recorded
//! earlier in the same call; a verdict recorded on a tokio worker thread was
//! unreadable from the calling thread; and a stale verdict could attach to a
//! later failure on the same thread. The marker has none of those failure
//! modes because the verdict travels INSIDE the error: wherever the message
//! goes, the classification goes, and when the message is rewritten away the
//! classification is lost together with the diagnostics it was derived from.
//! Only a marker at byte zero, or immediately after one exact producer-owned
//! bridge prefix, is a frame. Ordinary diagnostics escape marker literals
//! before display, so a caller-controlled URI/path cannot dictate its own
//! classification.
//! Code space carried by the marker:
//! * LOON / ExtendStatusCode values (`ffi_error_code.h`): 12 = file-not-found,
//!   101-122 (with reserved gaps) = storage/transient/txn/invariant codes. The
//!   C++ side rebuilds the matching `ExtendStatusDetail` (or an ENOENT detail
//!   for 12).
//! * Bridge-private values (>= 1000, never cross the C ABI): the C++ side
//!   converts them straight into an arrow StatusCode and they cease to exist.
//!
//! Classification discipline ("producer owns classification", conservative):
//! only signals the producer positively identifies are tagged; everything else
//! stays untagged and lands in the consumer's non-retriable fallback bucket.
//! Never invent retriability.

use lance::Error as LanceError;

/// Must stay byte-identical to `kBridgeErrCodeMarker` in cpp `bridge_error.cpp`
/// — one marker, one parser. Named for the whole Rust bridge: lance, iceberg
/// and paimon all carry it, not just the vortex bridge that first introduced it.
pub const BRIDGE_ERRCODE_MARKER: &str = "__LOON_RUST_BRIDGE_ERRCODE__=";
const BRIDGE_ERRCODE_ESCAPED: &str = "__LOON_RUST_BRIDGE_ESCAPED_ERRCODE__=";

/// Make arbitrary diagnostic text inert in the string-only classification
/// channel. The escaped spelling is deliberately not decoded by C++: decoding
/// it into a prefix again would let a later translation pass reinterpret it.
pub(crate) fn escape_bridge_error_message(message: &str) -> String {
    message.replace(BRIDGE_ERRCODE_MARKER, BRIDGE_ERRCODE_ESCAPED)
}

/// The object/table/dataset named is not there -- the bridges' single
/// not-found. LOON_FILE_NOT_FOUND (12) means the same thing but travels the
/// errno channel, which belongs to the filesystem layer and to the vortex fork
/// that already emits it; the C++ side still decodes 12 for those. No bridge
/// emits it, because two numbers for one condition is not a second meaning,
/// it is two places for every consumer to remember.
pub const LOON_STORAGE_NOT_FOUND: i32 = 104;
/// Mirror of the ExtendStatusCode transient tags (`ffi_error_code.h` 101-112).
pub const LOON_STORAGE_CONFLICT: i32 = 102;
pub const LOON_STORAGE_PRECONDITION_FAILED: i32 = 103;
pub const LOON_STORAGE_ACCESS_DENIED: i32 = 105;
pub const LOON_TRANSIENT_NETWORK: i32 = 107;
pub const LOON_TRANSIENT_TIMEOUT: i32 = 108;
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;
pub const LOON_TRANSIENT_SERVICE: i32 = 110;
pub const LOON_STORAGE_CONFIG_INVALID: i32 = 115;

/// Build the message for a failed credential-endpoint HTTP exchange.
///
/// Mirrors the credential-resolution table in docs/error-codes.md: 429 is
/// throttling, 5xx is service, 401/403 is access-denied, any other 4xx is
/// config. SDK-backed providers may override that fallback with a typed service
/// code (for example, AWS STS ThrottlingException uses HTTP 400).
/// `credential_reqwest_error` adds connect and timeout classification for
/// failures that do not carry an HTTP status.
///
/// The classification rides the universal marker: the providers resolve on
/// tokio workers, so nothing thread-local can carry the verdict back to the
/// calling thread -- the message is the only channel that survives the hop.
pub(crate) fn credential_http_failure_message(status: u16, context: &str) -> String {
    let code = credential_http_failure_code(status);
    BridgeError::new_io(
        code,
        format!("credential resolution {context} failed: HTTP {status}"),
    )
    .to_string()
}

fn credential_http_failure_code(status: u16) -> Option<i32> {
    match status {
        429 => Some(LOON_TRANSIENT_THROTTLING),
        500..=599 => Some(LOON_TRANSIENT_SERVICE),
        401 | 403 => Some(LOON_STORAGE_ACCESS_DENIED),
        400..=499 => Some(LOON_STORAGE_CONFIG_INVALID),
        _ => None,
    }
}

/// Preserve reqwest's typed transport/status information before any provider
/// turns the error into an object-store or anyhow diagnostic.
pub(crate) fn credential_reqwest_error(error: reqwest::Error, context: &str) -> BridgeError {
    let code = if error.is_timeout() {
        Some(LOON_TRANSIENT_TIMEOUT)
    } else if error.is_connect() {
        Some(LOON_TRANSIENT_NETWORK)
    } else {
        error
            .status()
            .and_then(|status| credential_http_failure_code(status.as_u16()))
    };
    BridgeError::new_io(
        code,
        format!("credential resolution {context} failed: {error}"),
    )
}

fn aws_sts_service_error_code(
    error: &aws_sdk_sts::operation::assume_role::AssumeRoleError,
    status: u16,
) -> Option<i32> {
    use aws_sdk_sts::error::ProvideErrorMetadata;

    match error.code() {
        Some("Throttling" | "ThrottlingException") => Some(LOON_TRANSIENT_THROTTLING),
        _ => credential_http_failure_code(status),
    }
}

fn aws_sts_sdk_error_code(
    error: &aws_sdk_sts::error::SdkError<aws_sdk_sts::operation::assume_role::AssumeRoleError>,
) -> Option<i32> {
    use aws_sdk_sts::error::SdkError;

    match error {
        SdkError::TimeoutError(_) => Some(LOON_TRANSIENT_TIMEOUT),
        SdkError::DispatchFailure(context) if context.is_timeout() => Some(LOON_TRANSIENT_TIMEOUT),
        SdkError::DispatchFailure(context) if context.is_io() => Some(LOON_TRANSIENT_NETWORK),
        SdkError::ServiceError(context) => {
            aws_sts_service_error_code(context.err(), context.raw().status().as_u16())
        }
        SdkError::ResponseError(_) => error
            .raw_response()
            .and_then(|response| credential_http_failure_code(response.status().as_u16())),
        _ => None,
    }
}

/// Keep the AWS Rust SDK's credential verdict before object_store erases it
/// behind `Error::Generic`. AssumeRole wraps its typed `SdkError` inside a
/// `CredentialsError`, so inspect that source while both types are available
/// instead of parsing either one's Display text later.
pub(crate) fn aws_credentials_error(
    error: aws_credential_types::provider::error::CredentialsError,
    context: &str,
) -> BridgeError {
    use aws_credential_types::provider::error::CredentialsError;

    let mut code = match &error {
        CredentialsError::ProviderTimedOut(_) => Some(LOON_TRANSIENT_TIMEOUT),
        CredentialsError::CredentialsNotLoaded(_) | CredentialsError::InvalidConfiguration(_) => {
            Some(LOON_STORAGE_CONFIG_INVALID)
        }
        _ => None,
    };

    let mut source = std::error::Error::source(&error);
    while code.is_none() {
        let Some(current) = source else {
            break;
        };
        if let Some(sdk_error) = current.downcast_ref::<aws_sdk_sts::error::SdkError<
            aws_sdk_sts::operation::assume_role::AssumeRoleError,
        >>() {
            code = aws_sts_sdk_error_code(sdk_error);
        }
        source = current.source();
    }

    BridgeError::new_io(
        code,
        format!("credential resolution {context} failed: {error}"),
    )
}

/// Extract the code the universal marker carries, if the text carries one at
/// all. Mirrors the C++ parser (`ParseBridgeError` in bridge_error.cpp): the
/// marker must be the prefix (possibly after the single Arrow IO wrapper), and
/// the frame must be exactly `marker + digits + ';'`. Marker-like text that
/// does not satisfy the complete frame is an ordinary diagnostic.
pub(crate) fn marker_code_in(message: &str) -> Option<i32> {
    let rest = if let Some(rest) = message.strip_prefix(BRIDGE_ERRCODE_MARKER) {
        rest
    } else if let Some(rest) = message.strip_prefix("Io error: ") {
        rest.strip_prefix(BRIDGE_ERRCODE_MARKER)?
    } else {
        return None;
    };
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    if rest.as_bytes().get(end) != Some(&b';') {
        return None;
    }
    rest[..end].parse().ok()
}

/// Bridge-private codes (>= 1000): decoded by cpp `bridge_error.cpp` into an
/// arrow StatusCode, never forwarded as an FFI error code.
pub const BRIDGE_ERRCODE_DATA_CORRUPT: i32 = 1001;
pub const BRIDGE_ERRCODE_NOT_SUPPORTED: i32 = 1002;

/// Error type used by the cxx bridge functions. cxx renders it with `Display`
/// into `rust::Error::what()`; the marker survives that trip.
#[derive(Clone, Debug)]
pub struct BridgeError {
    pub code: Option<i32>,
    pub msg: String,
    io_transport: bool,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.io_transport {
            write!(f, "Io error: ")?;
        }
        match self.code {
            Some(code) => write!(f, "{BRIDGE_ERRCODE_MARKER}{code}; {}", self.msg),
            None => write!(f, "{}", self.msg),
        }
    }
}

impl std::error::Error for BridgeError {}

impl BridgeError {
    /// Build the error. `Display` embeds the marker when a code is present;
    /// that marker is the entire transport -- the cxx boundary stringifies
    /// the error and the C++ decoder parses the code back out of the text.
    pub fn new(code: Option<i32>, msg: String) -> Self {
        Self::with_transport(code, msg, false)
    }

    pub(crate) fn new_io(code: Option<i32>, msg: String) -> Self {
        Self::with_transport(code, msg, true)
    }

    pub(crate) fn with_transport(code: Option<i32>, msg: String, io_transport: bool) -> Self {
        BridgeError {
            code,
            msg: escape_bridge_error_message(&msg),
            io_transport,
        }
    }

    pub(crate) fn is_io_transport(&self) -> bool {
        self.io_transport
    }
}

/// Alias used to switch a whole bridge impl module to classified errors: the
/// `?` operator converts `lance::Error` (and `ArrowError`) via the `From`
/// impls below.
pub type BridgeResult<T> = std::result::Result<T, BridgeError>;

/// Classify a `lance::Error` into a marker code. `None` = not positively
/// identified -> stays untagged -> conservative non-retriable fallback on the
/// consumer side.
// TODO: this classifier cannot report a transient failure, and lance is the one
// storage path in this repository that therefore cannot.
//
// Everything else classifies them: the C++ filesystems do it directly, vortex
// inherits that verdict through the C-ABI filesystem's err_code, and paimon and
// iceberg get it from opendal (RateLimited / is_temporary). lance is the
// exception because it reads through `object_store`, whose error type has no
// transient variant -- a dropped connection or a 503 arrives as `Error::Generic`
// and falls to `None` below, i.e. unclassified, i.e. conservatively non-retryable.
// So a metadata service that was restarting fails a lance read exactly as
// permanently as a role that does not exist, and Milvus will not re-run the
// operation.
//
// It cannot be fixed here. `object_store::client::retry` is `pub(crate)`, so
// the `RequestError` that carries the HTTP status is unreachable from this
// crate, and matching on the message text is the thing this file exists to
// avoid.
//
// The four credential providers we own (aliyun_oss_provider, aws_arn_provider,
// azure_sas_provider, gcp_impersonation) DO see the HTTP status of the
// credential endpoints, so they classify their own failures through
// `credential_http_failure_message` -- the verdict rides the universal marker
// and reaches the caller no matter which format sits above it. That closes
// the old inconsistency where the same credential failure was classified only
// when the format above it happened to be iceberg. Transient failures of the
// DATA path itself (object_store reads/writes under lance) remain the open
// gap; patching lance itself is possible (lance-core/lance-io are already
// pinned to a zilliztech fork in [patch.crates-io]) but unnecessary for the
// credential half.
pub fn classify_lance_error(e: &LanceError) -> Option<i32> {
    match e {
        // The object/dataset/index/ref/version is gone. Retrying hits the same
        // store and fails identically; consumers can distinguish "data
        // missing" from a generic storage failure.
        LanceError::NotFound { .. }
        | LanceError::DatasetNotFound { .. }
        | LanceError::IndexNotFound { .. }
        | LanceError::RefNotFound { .. }
        | LanceError::VersionNotFound { .. } => Some(LOON_STORAGE_NOT_FOUND),
        // Permanent data problems: retrying re-reads the same bytes.
        LanceError::CorruptFile { .. } => Some(BRIDGE_ERRCODE_DATA_CORRUPT),
        LanceError::NotSupported { .. } => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        // Lance lost a commit race. That is a Conflict, not throttling: the
        // Conflict category exists so generic retry helpers do NOT blindly
        // replay a commit whose contention budget may already be spent --
        // whether and how to re-drive a commit is the operation owner's call.
        LanceError::RetryableCommitConflict { .. } | LanceError::TooMuchWriteContention { .. } => {
            Some(LOON_STORAGE_CONFLICT)
        }
        // IO wraps the underlying object_store error as a boxed source;
        // downcast to recover the typed variant.
        LanceError::IO { source, .. } => {
            // Credential providers that we own retain their classification as
            // a typed source. Recover that before looking at object_store's
            // coarser outer variant.
            if let Some(error) = source.downcast_ref::<BridgeError>() {
                return error.code;
            }
            match source.downcast_ref::<object_store::Error>() {
                Some(object_store::Error::NotFound { .. }) => Some(LOON_STORAGE_NOT_FOUND),
                Some(
                    object_store::Error::PermissionDenied { .. }
                    | object_store::Error::Unauthenticated { .. },
                ) => Some(LOON_STORAGE_ACCESS_DENIED),
                Some(object_store::Error::Precondition { .. }) => {
                    Some(LOON_STORAGE_PRECONDITION_FAILED)
                }
                Some(
                    object_store::Error::NotSupported { .. }
                    | object_store::Error::NotImplemented { .. },
                ) => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
                Some(object_store::Error::Generic { source, .. }) => source
                    .downcast_ref::<BridgeError>()
                    .and_then(|error| error.code),
                _ => None,
            }
        }
        // InvalidInput deliberately NOT tagged as caller input: the strings we
        // feed lance are mostly assembled by this library itself, so blaming
        // the caller would misroute retries (see the 2007/2020/2021
        // demotions). Left untagged pending a producer-site audit.
        _ => None,
    }
}

fn classify_opendal_error(error: &opendal::Error) -> Option<i32> {
    use opendal::ErrorKind;
    match error.kind() {
        ErrorKind::NotFound => Some(LOON_STORAGE_NOT_FOUND),
        ErrorKind::PermissionDenied => Some(LOON_STORAGE_ACCESS_DENIED),
        ErrorKind::RateLimited => Some(LOON_TRANSIENT_THROTTLING),
        ErrorKind::Unsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        _ if error.is_temporary() => Some(LOON_TRANSIENT_SERVICE),
        _ => None,
    }
}

fn classify_error_source_chain(
    mut source: Option<&(dyn std::error::Error + 'static)>,
) -> Option<i32> {
    while let Some(error) = source {
        if let Some(error) = error.downcast_ref::<BridgeError>()
            && error.code.is_some()
        {
            return error.code;
        }
        if let Some(error) = error.downcast_ref::<opendal::Error>()
            && let Some(code) = classify_opendal_error(error)
        {
            return Some(code);
        }
        source = error.source();
    }
    None
}

fn bridge_io_transport_in_source_chain(
    mut source: Option<&(dyn std::error::Error + 'static)>,
) -> bool {
    while let Some(error) = source {
        if let Some(error) = error.downcast_ref::<BridgeError>()
            && error.is_io_transport()
        {
            return true;
        }
        source = error.source();
    }
    false
}

pub(crate) fn bridge_io_transport_in_anyhow(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<BridgeError>())
        .any(BridgeError::is_io_transport)
}

pub(crate) fn message_has_io_bridge_frame(message: &str) -> bool {
    message.starts_with("Io error: ") && marker_code_in(message).is_some()
}

/// Build the one Arrow C-stream transport envelope understood by C++.
///
/// `BridgeError` keeps the semantic code separate from its escaped diagnostic,
/// so this does not need to parse a rendered error or trust marker-looking
/// caller input. Arrow adds the sole `Io error: ` prefix when the stream reports
/// the error through `get_last_error`.
pub(crate) fn into_arrow_io_error(error: BridgeError) -> arrow58::error::ArrowError {
    let message = match error.code {
        Some(code) => format!("{BRIDGE_ERRCODE_MARKER}{code}; {}", error.msg),
        None => error.msg,
    };
    arrow58::error::ArrowError::IoError(message.clone(), std::io::Error::other(message))
}

pub fn classify_iceberg_error(error: &iceberg::Error) -> Option<i32> {
    use iceberg::ErrorKind;
    let direct = match error.kind() {
        ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound => Some(LOON_STORAGE_NOT_FOUND),
        ErrorKind::FeatureUnsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        _ => None,
    };
    direct.or_else(|| classify_error_source_chain(std::error::Error::source(error)))
}

pub fn classify_anyhow_error(error: &anyhow::Error) -> Option<i32> {
    for cause in error.chain() {
        if let Some(error) = cause.downcast_ref::<BridgeError>() {
            if error.code.is_some() {
                return error.code;
            }
        }
        if let Some(error) = cause.downcast_ref::<iceberg::Error>() {
            if let Some(code) = classify_iceberg_error(error) {
                return Some(code);
            }
        }
        if let Some(error) = cause.downcast_ref::<opendal::Error>()
            && let Some(code) = classify_opendal_error(error)
        {
            return Some(code);
        }
    }
    None
}

impl From<LanceError> for BridgeError {
    fn from(e: LanceError) -> Self {
        let io_transport = bridge_io_transport_in_source_chain(Some(&e));
        BridgeError::with_transport(classify_lance_error(&e), e.to_string(), io_transport)
    }
}

impl From<iceberg::Error> for BridgeError {
    fn from(error: iceberg::Error) -> Self {
        let io_transport = bridge_io_transport_in_source_chain(Some(&error));
        BridgeError::with_transport(
            classify_iceberg_error(&error),
            error.to_string(),
            io_transport,
        )
    }
}

impl From<anyhow::Error> for BridgeError {
    fn from(error: anyhow::Error) -> Self {
        // Keep the producer's message. Do not build a second diagnostic chain
        // while an error is already being propagated.
        let msg = error
            .downcast_ref::<BridgeError>()
            .map(|bridge| bridge.msg.clone())
            .unwrap_or_else(|| error.to_string());
        let io_transport = bridge_io_transport_in_anyhow(&error);
        BridgeError::with_transport(classify_anyhow_error(&error), msg, io_transport)
    }
}

impl From<arrow58::error::ArrowError> for BridgeError {
    fn from(e: arrow58::error::ArrowError) -> Self {
        BridgeError::new(None, e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ErrorReader {
        error: Option<arrow58::error::ArrowError>,
        schema: arrow58::datatypes::SchemaRef,
    }

    impl Iterator for ErrorReader {
        type Item = Result<arrow58::record_batch::RecordBatch, arrow58::error::ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            self.error.take().map(Err)
        }
    }

    impl arrow58::record_batch::RecordBatchReader for ErrorReader {
        fn schema(&self) -> arrow58::datatypes::SchemaRef {
            self.schema.clone()
        }
    }

    fn ffi_stream_error(error: BridgeError) -> (i32, String) {
        let reader = ErrorReader {
            error: Some(into_arrow_io_error(error)),
            schema: std::sync::Arc::new(arrow58::datatypes::Schema::empty()),
        };
        let mut stream = arrow58::ffi_stream::FFI_ArrowArrayStream::new(Box::new(reader));
        let mut array = arrow58::ffi::FFI_ArrowArray::empty();
        let code = unsafe { stream.get_next.unwrap()(&mut stream, &mut array) };
        let message = unsafe {
            let error = stream.get_last_error.unwrap()(&mut stream);
            assert!(!error.is_null());
            std::ffi::CStr::from_ptr(error)
                .to_string_lossy()
                .into_owned()
        };
        (code, message)
    }

    // A classified BridgeError that round-trips through anyhow (built, wrapped
    // as anyhow::Error, re-wrapped by From<anyhow::Error>) must keep its code
    // and must NOT leak the transport marker into the rebuilt message.
    #[test]
    fn anyhow_roundtrip_does_not_leak_marker() {
        let inner = BridgeError::new(
            Some(LOON_STORAGE_NOT_FOUND),
            "snapshot 42 was not found".to_string(),
        );
        let rebuilt = BridgeError::from(anyhow::Error::from(inner));
        assert_eq!(rebuilt.code, Some(LOON_STORAGE_NOT_FOUND));
        assert!(
            !rebuilt.msg.contains(BRIDGE_ERRCODE_MARKER),
            "{}",
            rebuilt.msg
        );
        assert!(rebuilt.msg.contains("snapshot 42 was not found"));
    }

    #[test]
    fn anyhow_roundtrip_preserves_io_transport() {
        let inner = BridgeError::new_io(
            Some(LOON_STORAGE_CONFIG_INVALID),
            "credential endpoint rejected the request".to_string(),
        );
        let rebuilt = BridgeError::from(anyhow::Error::from(inner).context("refresh token"));
        assert!(rebuilt.is_io_transport());
        assert!(rebuilt.to_string().starts_with("Io error: "));
        assert_eq!(rebuilt.code, Some(LOON_STORAGE_CONFIG_INVALID));
    }

    #[test]
    fn credential_http_statuses_use_io_framing() {
        for (status, expected) in [
            (403, LOON_STORAGE_ACCESS_DENIED),
            (404, LOON_STORAGE_CONFIG_INVALID),
            (429, LOON_TRANSIENT_THROTTLING),
            (503, LOON_TRANSIENT_SERVICE),
        ] {
            let message = credential_http_failure_message(status, "test endpoint");
            assert!(message.starts_with("Io error: "), "{message}");
            assert_eq!(marker_code_in(&message), Some(expected), "{message}");
        }
    }

    #[test]
    fn classified_stream_error_has_one_arrow_io_envelope() {
        for error in [
            BridgeError::new_io(
                Some(LOON_TRANSIENT_TIMEOUT),
                "Lance request timed out".to_string(),
            ),
            BridgeError::new(
                Some(BRIDGE_ERRCODE_DATA_CORRUPT),
                "Vortex footer is corrupt".to_string(),
            ),
        ] {
            let (code, message) = ffi_stream_error(error);
            assert_eq!(code, 5, "{message}");
            assert!(message.starts_with("Io error: __LOON_RUST_BRIDGE_ERRCODE__="));
            assert!(!message.contains("External error:"), "{message}");
            assert_eq!(message.matches("Io error: ").count(), 1, "{message}");
            assert!(marker_code_in(&message).is_some(), "{message}");
        }
    }

    #[tokio::test]
    async fn credential_reqwest_errors_keep_connect_and_timeout_types() {
        let unused = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let unused_address = unused.local_addr().unwrap();
        drop(unused);
        let connect_error = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .unwrap()
            .get(format!("http://{unused_address}"))
            .send()
            .await
            .unwrap_err();
        let connect_error = credential_reqwest_error(connect_error, "connect test");
        assert_eq!(connect_error.code, Some(LOON_TRANSIENT_NETWORK));
        assert!(connect_error.is_io_transport());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        });
        let timeout_error = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(20))
            .build()
            .unwrap()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap_err();
        let timeout_error = credential_reqwest_error(timeout_error, "timeout test");
        assert_eq!(timeout_error.code, Some(LOON_TRANSIENT_TIMEOUT));
        assert!(timeout_error.is_io_transport());
        server.await.unwrap();
    }

    #[test]
    fn aws_credentials_errors_keep_typed_causes() {
        use aws_credential_types::provider::error::CredentialsError;
        use aws_sdk_sts::{error::ErrorMetadata, operation::assume_role::AssumeRoleError};

        let timed_out = aws_credentials_error(
            CredentialsError::provider_timed_out(std::time::Duration::from_secs(1)),
            "AWS AssumeRole",
        );
        assert_eq!(timed_out.code, Some(LOON_TRANSIENT_TIMEOUT));

        let invalid = aws_credentials_error(
            CredentialsError::invalid_configuration("invalid role configuration"),
            "AWS AssumeRole",
        );
        assert_eq!(invalid.code, Some(LOON_STORAGE_CONFIG_INVALID));

        let sdk_timeout: aws_sdk_sts::error::SdkError<AssumeRoleError> =
            aws_sdk_sts::error::SdkError::timeout_error(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "STS request timed out",
            ));
        let nested = aws_credentials_error(
            CredentialsError::provider_error(sdk_timeout),
            "AWS AssumeRole",
        );
        assert_eq!(nested.code, Some(LOON_TRANSIENT_TIMEOUT));

        for service_code in ["Throttling", "ThrottlingException"] {
            let throttled =
                AssumeRoleError::generic(ErrorMetadata::builder().code(service_code).build());
            assert_eq!(
                aws_sts_service_error_code(&throttled, 400),
                Some(LOON_TRANSIENT_THROTTLING)
            );
        }

        let invalid =
            AssumeRoleError::generic(ErrorMetadata::builder().code("ValidationError").build());
        assert_eq!(
            aws_sts_service_error_code(&invalid, 400),
            Some(LOON_STORAGE_CONFIG_INVALID)
        );
    }

    // Commit contention is a Conflict (102), never a retryable throttling
    // code: generic retry helpers must not blindly replay a commit whose
    // contention budget may already be spent.
    #[test]
    fn lance_commit_contention_classifies_as_conflict() {
        let contention = LanceError::TooMuchWriteContention {
            message: "too many concurrent writers".to_string(),
            location: std::panic::Location::caller(),
        };
        assert_eq!(
            classify_lance_error(&contention),
            Some(LOON_STORAGE_CONFLICT)
        );

        let conflict = LanceError::RetryableCommitConflict {
            version: 7,
            source: "lost the commit race".into(),
            location: std::panic::Location::caller(),
        };
        assert_eq!(classify_lance_error(&conflict), Some(LOON_STORAGE_CONFLICT));
    }

    // A classified error that is constructed and then handled INSIDE a guarded
    // call cannot leak its verdict into a later failure: the code lives in the
    // message of the error that was handled, and an unrelated failure carries
    // an unrelated message. This is the property the retired thread-local side
    // channel needed an explicit clearing rule to approximate.
    #[test]
    fn marker_code_round_trips_the_cxx_decoder_contract() {
        let text = format!(
            "{BRIDGE_ERRCODE_MARKER}{LOON_TRANSIENT_THROTTLING}; throttled by the object store"
        );
        assert_eq!(marker_code_in(&text), Some(LOON_TRANSIENT_THROTTLING));
        assert_eq!(
            marker_code_in(&format!(
                "Io error: {BRIDGE_ERRCODE_MARKER}{LOON_STORAGE_NOT_FOUND}; missing"
            )),
            Some(LOON_STORAGE_NOT_FOUND)
        );
        // No marker, no code -- the conservative bucket.
        assert_eq!(marker_code_in("plain failure"), None);
        // Incomplete marker-like text is not a frame, exactly like the C++ parser.
        assert_eq!(
            marker_code_in("vortex: __LOON_RUST_BRIDGE_ERRCODE__; odd"),
            None
        );
        assert_eq!(
            marker_code_in(&format!(
                "{BRIDGE_ERRCODE_MARKER}{LOON_TRANSIENT_THROTTLING} missing-semicolon"
            )),
            None
        );
        // Only the prefix marker is framing; later marker text is inert.
        let double = format!(
            "{BRIDGE_ERRCODE_MARKER}{LOON_STORAGE_NOT_FOUND}; outer {BRIDGE_ERRCODE_MARKER}{LOON_STORAGE_CONFLICT}; inner"
        );
        assert_eq!(marker_code_in(&double), Some(LOON_STORAGE_NOT_FOUND));
        assert_eq!(
            marker_code_in(&format!(
                "caller path /tmp/{BRIDGE_ERRCODE_MARKER}{LOON_TRANSIENT_THROTTLING}"
            )),
            None
        );
    }

    #[test]
    fn unclassified_message_cannot_forge_marker_code() {
        let error = BridgeError::new(
            None,
            format!(
                "failed to open s3://bucket/{BRIDGE_ERRCODE_MARKER}{LOON_TRANSIENT_THROTTLING}"
            ),
        );
        let rendered = error.to_string();
        assert_eq!(marker_code_in(&rendered), None);
        assert!(!rendered.contains(BRIDGE_ERRCODE_MARKER), "{rendered}");
        assert!(rendered.contains(BRIDGE_ERRCODE_ESCAPED), "{rendered}");
    }

    #[test]
    fn anyhow_context_keeps_classified_source_without_rebuilding_diagnostics() {
        let inner = BridgeError::new(Some(LOON_STORAGE_NOT_FOUND), "missing".to_string());
        let rebuilt = BridgeError::from(anyhow::Error::from(inner).context("planning scan"));
        assert_eq!(rebuilt.code, Some(LOON_STORAGE_NOT_FOUND));
        assert!(
            !rebuilt.msg.contains(BRIDGE_ERRCODE_MARKER),
            "{}",
            rebuilt.msg
        );
        assert_eq!(rebuilt.msg, "missing");
    }
}

// ---------------------------------------------------------------------------
// A classification extracted from an error's TYPE chain.
//
// Used only where a consumer needs the verdict as data rather than as text:
// the vortex async-open callback hands (code, message) to C++ through its own
// explicit C-ABI signature. Everywhere else the marker in `Display` IS the
// transport and nothing is extracted.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct ClassifiedErrorInfo {
    pub code: i32,
    pub message: String,
}
