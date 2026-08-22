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
//! Code space carried by the marker:
//! * LOON / ExtendStatusCode values (`ffi_error_code.h`): 12 = file-not-found,
//!   101-112 = AWS/transient/txn extend codes. The C++ side rebuilds the
//!   matching `ExtendStatusDetail` (or an ENOENT detail for 12).
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
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;
pub const LOON_TRANSIENT_SERVICE: i32 = 110;
pub const LOON_STORAGE_CONFIG_INVALID: i32 = 115;

/// Record the outcome of a failed credential-endpoint HTTP exchange.
///
/// Mirrors the credential-resolution table in docs/error-codes.md: 429 is
/// throttling, 5xx is service, 401/403 is access-denied, any other 4xx is
/// config. Transport-level faults (connect/read timeout) surface as reqwest
/// `Err` values without a status and are deliberately NOT classified here.
pub(crate) fn record_credential_http_failure(status: u16, context: &str) {
    let code = match status {
        429 => LOON_TRANSIENT_THROTTLING,
        500..=599 => LOON_TRANSIENT_SERVICE,
        401 | 403 => LOON_STORAGE_ACCESS_DENIED,
        400..=499 => LOON_STORAGE_CONFIG_INVALID,
        _ => return,
    };
    record_bridge_error(
        code,
        format!("credential resolution {context} failed: HTTP {status}"),
    );
}

/// Bridge-private codes (>= 1000): decoded by cpp `bridge_error.cpp` into an
/// arrow StatusCode, never forwarded as an FFI error code.
pub const BRIDGE_ERRCODE_DATA_CORRUPT: i32 = 1001;
pub const BRIDGE_ERRCODE_NOT_SUPPORTED: i32 = 1002;

/// Error type used by the cxx bridge functions. cxx renders it with `Display`
/// into `rust::Error::what()`; the marker survives that trip.
#[derive(Debug)]
pub struct BridgeError {
    pub code: Option<i32>,
    pub msg: String,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "{BRIDGE_ERRCODE_MARKER}{code}; {}", self.msg),
            None => write!(f, "{}", self.msg),
        }
    }
}

impl std::error::Error for BridgeError {}

impl BridgeError {
    /// Build the error AND publish its classification on the side channel.
    ///
    /// Every construction goes through here so no producer has to remember to
    /// record separately; the marker in `Display` stays only for the paths that
    /// have no C++ call frame to read the slot -- an error surfacing through an
    /// Arrow C stream, or the vortex fork's own marker.
    pub fn new(code: Option<i32>, msg: String) -> Self {
        match code {
            // The cxx exception already carries the message. Publish only the
            // typed verdict so error construction does not duplicate diagnostics.
            Some(code) => record_bridge_error(code, String::new()),
            // An UNCLASSIFIED error must also overwrite the slot, not merely
            // decline to write it. The C++ side clears the slot once per guarded
            // call, so anything recorded earlier WITHIN that call survives: an
            // inner failure that was classified and then handled (a retry loop,
            // a probe whose error is expected, an `ok_or_else` fallback) would
            // otherwise leave its verdict behind for whatever error actually
            // crosses the boundary. Publishing "no classification" keeps the
            // last construction authoritative, which is what every consumer
            // already assumes, and keeps R1.2 honest: an unclassified failure
            // must reach the conservative bucket, never inherit a code nobody
            // established for it.
            None => clear_last_bridge_error(),
        }
        BridgeError { code, msg }
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
// inherits that verdict through filesystem_c's err_code, and paimon and iceberg
// get it from opendal (RateLimited / is_temporary). lance is the exception
// because it reads through `object_store`, whose error type has no transient
// variant -- a dropped connection or a 503 arrives as `Error::Generic` and
// falls to `None` below, i.e. unclassified, i.e. conservatively non-retryable.
// So a metadata service that was restarting fails a lance read exactly as
// permanently as a role that does not exist, and Milvus will not re-run the
// operation.
//
// It cannot be fixed here. `object_store::client::retry` is `pub(crate)`, so
// the `RequestError` that carries the HTTP status is unreachable from this
// crate, and matching on the message text is the thing this file exists to
// avoid.
//
// The fix belongs in the four credential providers we own
// (aliyun_oss_provider, aws_arn_provider, azure_sas_provider,
// gcp_impersonation): none of them records anything today, and they DO see the
// HTTP status. Recording it with `record_bridge_error(LOON_TRANSIENT_*, ..)`
// reaches the caller regardless of what this function returns, because
// BridgeErrorStatusFromException reads the side channel before it reads
// anything else. That also removes a live inconsistency -- those providers feed
// BOTH iceberg and lance, so the same credential failure is classified today
// only when the format above it happens to be iceberg.
//
// Patching lance itself is possible (lance-core/lance-io are already pinned to
// a zilliztech fork in [patch.crates-io]) but unnecessary: the side channel is
// entirely on our side of the boundary.
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
        LanceError::IO { source, .. } => match source.downcast_ref::<object_store::Error>() {
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
            Some(object_store::Error::Generic { .. }) => None,
            _ => None,
        },
        // InvalidInput deliberately NOT tagged as caller input: the strings we
        // feed lance are mostly assembled by this library itself, so blaming
        // the caller would misroute retries (see the 2007/2020/2021
        // demotions). Left untagged pending a producer-site audit.
        _ => None,
    }
}

pub fn classify_iceberg_error(error: &iceberg::Error) -> Option<i32> {
    use iceberg::ErrorKind;
    match error.kind() {
        ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound => Some(LOON_STORAGE_NOT_FOUND),
        ErrorKind::FeatureUnsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        _ => None,
    }
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
        if let Some(error) = cause.downcast_ref::<opendal::Error>() {
            use opendal::ErrorKind;
            let code = match error.kind() {
                ErrorKind::NotFound => Some(LOON_STORAGE_NOT_FOUND),
                ErrorKind::PermissionDenied => Some(LOON_STORAGE_ACCESS_DENIED),
                ErrorKind::RateLimited => Some(LOON_TRANSIENT_THROTTLING),
                ErrorKind::Unsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
                _ if error.is_temporary() => Some(LOON_TRANSIENT_SERVICE),
                _ => None,
            };
            if code.is_some() {
                return code;
            }
        }
    }
    None
}

impl From<LanceError> for BridgeError {
    fn from(e: LanceError) -> Self {
        BridgeError::new(classify_lance_error(&e), e.to_string())
    }
}

impl From<iceberg::Error> for BridgeError {
    fn from(error: iceberg::Error) -> Self {
        BridgeError::new(classify_iceberg_error(&error), error.to_string())
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
        BridgeError::new(classify_anyhow_error(&error), msg)
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
        assert_eq!(
            classify_lance_error(&conflict),
            Some(LOON_STORAGE_CONFLICT)
        );
    }

    // A classified error that is constructed and then handled INSIDE a guarded
    // call must not leave its verdict on the side channel for the unclassified
    // error that actually crosses the boundary. The C++ side clears the slot
    // once per call, so the clearing on the None path is the only thing
    // standing between "lance retried past a NotFound" and "an unrelated
    // generic failure reported as StorageNotFound".
    #[test]
    fn unclassified_error_clears_a_previously_recorded_code() {
        clear_last_bridge_error();
        let _handled = BridgeError::new(Some(LOON_STORAGE_NOT_FOUND), "probe missed".to_string());
        let _surfaced = BridgeError::new(None, "generic failure".to_string());
        assert_eq!(take_last_bridge_error().code, 0);
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
// The classified-error side channel.
//
// cxx can only carry an error across the boundary as a Display string, which is
// why the marker above exists. A marker is a poor channel: the classification
// travels inside data, so an error message that happens to contain marker text
// can dictate its own classification, and every consumer has to parse before it
// can trust. This slot is the real channel -- the producer records the code
// beside the error, the C++ side takes it after catching, and nothing has to be
// parsed out of a message.
//
// One slot for every bridge, deliberately: a second channel would be a second
// thing to keep in sync. It is thread-local and taken (not read); the C++ side
// clears it before each guarded call, and every BridgeError construction
// overwrites it -- including the unclassified ones, which clear it. Between
// those two rules a stale code can attach itself neither to a later call nor to
// a later error within the same call.
// ---------------------------------------------------------------------------

use std::cell::RefCell;

#[derive(Clone, Debug)]
pub(crate) struct ClassifiedErrorInfo {
    pub code: i32,
    pub message: String,
}

thread_local! {
    static LAST_BRIDGE_ERROR: RefCell<Option<ClassifiedErrorInfo>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_bridge_error(info: ClassifiedErrorInfo) {
    LAST_BRIDGE_ERROR.with(|slot| *slot.borrow_mut() = Some(info));
}

/// Record a classification for the failure that is about to cross the boundary.
pub fn record_bridge_error(code: i32, message: String) {
    set_last_bridge_error(ClassifiedErrorInfo { code, message });
}

/// Drop anything recorded earlier. The C++ side calls this before every guarded
/// call so a code from a previous failure cannot be mistaken for this one's.
pub fn clear_last_bridge_error() {
    LAST_BRIDGE_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Take the recorded classification, leaving the slot empty. `code == 0` means
/// nothing was recorded, and the consumer falls back to its message-based path.
pub fn take_last_bridge_error() -> crate::bridge_ffi::BridgeErrorInfo {
    match LAST_BRIDGE_ERROR.with(|slot| slot.borrow_mut().take()) {
        Some(info) => crate::bridge_ffi::BridgeErrorInfo {
            code: info.code,
            message: info.message,
        },
        None => crate::bridge_ffi::BridgeErrorInfo {
            code: 0,
            message: String::new(),
        },
    }
}
