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

/// Must stay byte-identical to the vortex marker in `filesystem_c.rs` and the
/// parser constant in cpp `bridge_error.cpp` — one marker, one parser.
pub const BRIDGE_ERRCODE_MARKER: &str = "__LOON_VORTEX_FFI_ERRCODE__=";

/// Mirrors LOON_FILE_NOT_FOUND in `ffi_error_code.h`.
pub const LOON_FILE_NOT_FOUND: i32 = 12;
/// Mirror of the ExtendStatusCode transient tags (`ffi_error_code.h` 101-112).
pub const LOON_AWS_ERROR_PRECONDITION_FAILED: i32 = 103;
pub const LOON_AWS_ERROR_ACCESS_DENIED: i32 = 105;
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;

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

/// Alias used to switch a whole bridge impl module to classified errors: the
/// `?` operator converts `lance::Error` (and `ArrowError`) via the `From`
/// impls below.
pub type BridgeResult<T> = std::result::Result<T, BridgeError>;

/// Classify a `lance::Error` into a marker code. `None` = not positively
/// identified -> stays untagged -> conservative non-retriable fallback on the
/// consumer side.
pub fn classify_lance_error(e: &LanceError) -> Option<i32> {
    match e {
        // The object/dataset/index/ref/version is gone. Retrying hits the same
        // store and fails identically; consumers can distinguish "data
        // missing" from a generic storage failure.
        LanceError::NotFound { .. }
        | LanceError::DatasetNotFound { .. }
        | LanceError::IndexNotFound { .. }
        | LanceError::RefNotFound { .. }
        | LanceError::VersionNotFound { .. }
        | LanceError::FieldNotFound { .. } => Some(LOON_FILE_NOT_FOUND),
        // Permanent data problems: retrying re-reads the same bytes.
        LanceError::CorruptFile { .. }
        | LanceError::SchemaMismatch { .. }
        | LanceError::Schema { .. } => Some(BRIDGE_ERRCODE_DATA_CORRUPT),
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
            // Generic and friends: object_store has already spent its own
            // retry budget; no positive transient/permanent signal survives,
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
