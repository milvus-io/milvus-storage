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
//!   Code 1000 is the explicit unclassified fallback; carrying it is important
//!   for stream errors because Arrow's C stream maps every Rust error to
//!   `Invalid` before C++ gets a chance to restore the original class.
//!
//! Classification discipline ("producer owns classification", conservative):
//! only signals the producer positively identifies get a semantic code;
//! everything else carries the explicit unclassified marker and lands in the
//! consumer's non-retriable fallback bucket. Never invent retriability.

use lance::Error as LanceError;

/// Must stay byte-identical to the vortex marker in `filesystem_c.rs` and the
/// parser constant in cpp `bridge_error.cpp` — one marker, one parser.
pub const BRIDGE_ERRCODE_MARKER: &str = "__LOON_RUST_BRIDGE_ERRCODE__=";

/// Mirrors LOON_FILE_NOT_FOUND in `ffi_error_code.h`.
pub const LOON_FILE_NOT_FOUND: i32 = 12;
/// Mirror of the ExtendStatusCode transient tags (`ffi_error_code.h` 101-112).
pub const LOON_AWS_ERROR_PRECONDITION_FAILED: i32 = 103;
pub const LOON_AWS_ERROR_ACCESS_DENIED: i32 = 105;
pub const LOON_TRANSIENT_TIMEOUT: i32 = 108;
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;
pub const LOON_TRANSIENT_SERVICE: i32 = 110;

/// Bridge-private codes (>= 1000): decoded by cpp `bridge_error.cpp` into an
/// arrow StatusCode, never forwarded as an FFI error code.
pub const BRIDGE_ERRCODE_UNCLASSIFIED: i32 = 1000;
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
        // Always carry a marker. In synchronous cxx calls an unmarked error
        // would still become a plain IOError, but during Arrow C-stream
        // iteration it first becomes Invalid/EINVAL. The explicit fallback
        // marker lets the C++ decoder restore that stream error to IOError
        // instead of misreporting it as DataFormatBroken.
        let code = self.code.unwrap_or(BRIDGE_ERRCODE_UNCLASSIFIED);
        write!(f, "{BRIDGE_ERRCODE_MARKER}{code}; {}", self.msg)
    }
}

impl std::error::Error for BridgeError {}

/// Alias used to switch a whole bridge impl module to classified errors: the
/// `?` operator converts `lance::Error` (and `ArrowError`) via the `From`
/// impls below.
pub type BridgeResult<T> = std::result::Result<T, BridgeError>;

/// Classify a `lance::Error` into a semantic marker code. `None` = not
/// positively identified; `Display` emits the explicit unclassified marker so
/// the consumer can still restore the conservative non-retriable IO fallback.
pub fn classify_lance_error(e: &LanceError) -> Option<i32> {
    match e {
        // The object/dataset/index/ref/version is gone. Retrying hits the same
        // store and fails identically; consumers can distinguish "data
        // missing" from a generic storage failure.
        LanceError::NotFound { .. }
        | LanceError::DatasetNotFound { .. }
        | LanceError::IndexNotFound { .. }
        | LanceError::RefNotFound { .. }
        | LanceError::VersionNotFound { .. } => Some(LOON_FILE_NOT_FOUND),
        // Field/schema errors are caller/schema-evolution conditions, not
        // corruption -- and they must NOT look like a missing dataset: the
        // ENOENT classification drives create-if-missing in the lance writer,
        // so tagging FieldNotFound as not-found could turn a projection typo
        // into a dataset-creation attempt. Producer sites are mixed
        // (library-assembled schemas vs user projections), so they stay
        // untagged -> conservative non-retriable.
        LanceError::FieldNotFound { .. }
        | LanceError::SchemaMismatch { .. }
        | LanceError::Schema { .. } => None,
        // Permanent data problems: retrying re-reads the same bytes.
        LanceError::CorruptFile { .. } => Some(BRIDGE_ERRCODE_DATA_CORRUPT),
        LanceError::NotSupported { .. } => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        // Lance itself declares these retryable: the failed attempt is spent,
        // but a fresh attempt (new commit round) can succeed. This is the
        // producer's own classification, not invented here.
        LanceError::RetryableCommitConflict { .. } | LanceError::TooMuchWriteContention { .. } => {
            Some(LOON_TRANSIENT_THROTTLING)
        }
        // IO wraps the underlying object_store error as a boxed source;
        // downcast to recover the typed variant.
        LanceError::IO { source, .. } => match source.downcast_ref::<object_store::Error>() {
            Some(object_store::Error::NotFound { .. }) => Some(LOON_FILE_NOT_FOUND),
            Some(
                object_store::Error::PermissionDenied { .. }
                | object_store::Error::Unauthenticated { .. },
            ) => Some(LOON_AWS_ERROR_ACCESS_DENIED),
            Some(object_store::Error::Precondition { .. }) => {
                Some(LOON_AWS_ERROR_PRECONDITION_FAILED)
            }
            Some(
                object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented { .. },
            ) => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
            // Generic carries the post-retry HTTP failure. The typed carrier
            // (client::retry::RetryError, which has .status()) is pub(crate)
            // in object_store and cannot be downcast from here, so the status
            // code is recovered from the stable Display pattern of
            // RequestError::Status ("non-2xx status code: NNN"). Fail-safe by
            // construction: if object_store ever rewords it, this returns
            // None and the error lands in the conservative non-retriable
            // bucket -- it can never mis-tag a permanent error as transient.
            Some(object_store::Error::Generic { source, .. }) => {
                classify_http_status_in_message(&source.to_string())
            }
            // Anything else: no positive transient/permanent signal survives,
            // so stay untagged (conservative).
            _ => None,
        },
        // InvalidInput deliberately NOT tagged as caller input: the strings we
        // feed lance are mostly assembled by this library itself, so blaming
        // the caller would misroute retries (see the 2007/2020/2021
        // demotions). Left untagged pending a producer-site audit.
        _ => None,
    }
}

/// Recover the HTTP status from object_store's post-retry error message
/// ("Server returned non-2xx status code: NNN: ..."). Only well-known
/// transient statuses are tagged; anything else stays untagged.
fn classify_http_status_in_message(msg: &str) -> Option<i32> {
    const PATTERN: &str = "non-2xx status code: ";
    let idx = msg.find(PATTERN)?;
    let digits: String = msg[idx + PATTERN.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    match digits.parse::<u16>().ok()? {
        401 | 403 => Some(LOON_AWS_ERROR_ACCESS_DENIED),
        408 => Some(LOON_TRANSIENT_TIMEOUT),
        429 => Some(LOON_TRANSIENT_THROTTLING),
        500 | 502 | 503 | 504 => Some(LOON_TRANSIENT_SERVICE),
        _ => None,
    }
}

impl From<LanceError> for BridgeError {
    fn from(e: LanceError) -> Self {
        BridgeError {
            code: classify_lance_error(&e),
            msg: e.to_string(),
        }
    }
}

impl From<arrow58::error::ArrowError> for BridgeError {
    fn from(e: arrow58::error::ArrowError) -> Self {
        BridgeError {
            code: None,
            msg: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time version pin: LanceError's From impl only accepts the
    // object_store version lance itself depends on. If this crate's
    // object_store ever diverges from lance's, this test stops COMPILING,
    // surfacing the downcast coupling instead of letting classify_lance_error
    // silently fail at runtime.
    #[test]
    fn object_store_version_matches_lance() {
        let e = LanceError::from(object_store::Error::NotFound {
            path: "p".to_string(),
            source: "gone".into(),
        });
        assert_eq!(classify_lance_error(&e), Some(LOON_FILE_NOT_FOUND));
    }

    #[test]
    fn generic_throttle_status_is_tagged_transient() {
        let e = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "Error performing GET https://x in 30s, after 10 retries                      - Server returned non-2xx status code: 429: slow down"
                .into(),
        });
        assert_eq!(classify_lance_error(&e), Some(LOON_TRANSIENT_THROTTLING));

        let e503 = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "Server returned non-2xx status code: 503: unavailable".into(),
        });
        assert_eq!(classify_lance_error(&e503), Some(LOON_TRANSIENT_SERVICE));

        // Credential-path auth failures (canonical pattern emitted by the
        // gcp/aliyun providers) map to access-denied, not transient.
        let e403 = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "sts:AssumeRole failed: non-2xx status code: 403: forbidden".into(),
        });
        assert_eq!(
            classify_lance_error(&e403),
            Some(LOON_AWS_ERROR_ACCESS_DENIED)
        );
    }

    #[test]
    fn generic_without_status_stays_untagged() {
        let e = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "connection reset by peer".into(),
        });
        assert_eq!(classify_lance_error(&e), None);
        // 4xx that is NOT a known transient must never be tagged retryable.
        let e404ish = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "Server returned non-2xx status code: 400: bad request".into(),
        });
        assert_eq!(classify_lance_error(&e404ish), None);
    }

    #[test]
    fn unclassified_errors_still_carry_the_bridge_marker() {
        let error = BridgeError {
            code: None,
            msg: "connection reset by peer".to_string(),
        };
        assert_eq!(
            error.to_string(),
            format!(
                "{BRIDGE_ERRCODE_MARKER}{BRIDGE_ERRCODE_UNCLASSIFIED}; connection reset by peer"
            )
        );
    }

    #[test]
    fn field_and_schema_errors_are_not_enoent() {
        // FieldNotFound must never classify as file-not-found: ENOENT drives
        // create-if-missing in the lance writer.
        let e = LanceError::FieldNotFound {
            source: lance_core::error::FieldNotFoundError {
                field_name: "f".to_string(),
                candidates: vec![],
            },
        };
        assert_eq!(classify_lance_error(&e), None);
    }
}
