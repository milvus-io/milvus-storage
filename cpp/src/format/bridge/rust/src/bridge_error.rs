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
//! * LOON / ExtendStatusCode values (`ffi_error_code.h`): 12 = dataset-not-found,
//!   101-114 = AWS/transient/txn/Lance extend codes. The C++ side rebuilds the
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
pub const LOON_AWS_ERROR_NOT_FOUND: i32 = 104;
pub const LOON_AWS_ERROR_ACCESS_DENIED: i32 = 105;
pub const LOON_TRANSIENT_TIMEOUT: i32 = 108;
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;
pub const LOON_TRANSIENT_SERVICE: i32 = 110;
/// Lance-specific ExtendStatusCode values. Keep separate from object-store
/// throttling and from code 12, which is the create-if-missing signal.
pub const LOON_LANCE_WRITE_CONTENTION: i32 = 113;
pub const LOON_LANCE_RESOURCE_NOT_FOUND: i32 = 114;

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
        LanceError::DatasetNotFound { .. } => Some(LOON_FILE_NOT_FOUND),
        // These are missing resources *inside* an existing Lance dataset. They
        // remain fine-grained ObjectNotExist downstream, but must not carry
        // ENOENT: LanceTableWriter consumes ENOENT as "dataset absent -> create".
        LanceError::NotFound { .. }
        | LanceError::IndexNotFound { .. }
        | LanceError::RefNotFound { .. }
        | LanceError::VersionNotFound { .. } => Some(LOON_LANCE_RESOURCE_NOT_FOUND),
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
            Some(LOON_LANCE_WRITE_CONTENTION)
        }
        // IO wraps the underlying object_store error as a boxed source;
        // downcast to recover the typed variant.
        LanceError::IO { source, .. } => classify_boxed_source(source.as_ref()),
        // Wrappers that carry a classified error inside. lance-io's batch read
        // scheduler stashes the failing task's error and re-wraps it on drop
        // (`Error::wrapped(...)` in lance-io/src/scheduler.rs), and the encoding
        // decoder does the same, so a plain S3 throttle on any batched read
        // arrives here as Wrapped(IO(object_store::Generic)) rather than IO.
        // Without unwrapping, a retriable failure would be reported as a
        // permanent one -- the classification is already there, just one box
        // deeper. Recursion terminates because each step strips one layer.
        LanceError::Wrapped { error, .. } => classify_boxed_source(error.as_ref()),
        LanceError::External { source } => classify_boxed_source(source.as_ref()),
        // InvalidInput deliberately NOT tagged as caller input: the strings we
        // feed lance are mostly assembled by this library itself, so blaming
        // the caller would misroute retries (see the 2007/2020/2021
        // demotions). Left untagged pending a producer-site audit.
        _ => None,
    }
}

/// Classify a boxed error carried by `LanceError::IO` / `Wrapped` / `External`.
///
/// The box can hold either the underlying `object_store::Error` or another
/// `LanceError` that some layer re-wrapped, so try both. Anything else stays
/// untagged (conservative non-retriable).
fn classify_boxed_source(
    source: &(dyn std::error::Error + Send + Sync + 'static),
) -> Option<i32> {
    if let Some(store_error) = source.downcast_ref::<object_store::Error>() {
        return classify_object_store_error(store_error);
    }
    if let Some(lance_error) = source.downcast_ref::<LanceError>() {
        return classify_lance_error(lance_error);
    }
    None
}

/// Classify the typed `object_store::Error` that backs lance's IO.
fn classify_object_store_error(error: &object_store::Error) -> Option<i32> {
    match error {
        // A missing object while operating inside a dataset is not proof that
        // the dataset itself is absent. Preserve ObjectNotExist without
        // emitting the writer's create-if-missing ENOENT signal.
        object_store::Error::NotFound { .. } => Some(LOON_LANCE_RESOURCE_NOT_FOUND),
        object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => Some(LOON_AWS_ERROR_ACCESS_DENIED),
        object_store::Error::Precondition { .. } => Some(LOON_AWS_ERROR_PRECONDITION_FAILED),
        object_store::Error::NotSupported { .. } | object_store::Error::NotImplemented { .. } => {
            Some(BRIDGE_ERRCODE_NOT_SUPPORTED)
        }
        // Generic carries the post-retry HTTP failure. The typed carrier
        // (client::retry::RetryError, which has .status()) is pub(crate) in
        // object_store and cannot be downcast from here, so the status code is
        // recovered from the stable Display pattern of RequestError::Status
        // ("non-2xx status code: NNN"). Fail-safe by construction: if
        // object_store ever rewords it, this returns None and the error lands
        // in the conservative non-retriable bucket -- it can never mis-tag a
        // permanent error as transient.
        object_store::Error::Generic { source, .. } => {
            classify_http_status_in_message(&source.to_string())
        }
        // Anything else: no positive transient/permanent signal survives, so
        // stay untagged (conservative).
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

/// Classify an `opendal::Error`. This is iceberg's IO layer (via
/// iceberg-storage-opendal), the counterpart of `object_store` under lance.
///
/// opendal carries the producer's own retriability verdict in
/// `is_temporary()`, so unlike the object_store path we do not have to recover
/// anything from prose.
pub fn classify_opendal_error(e: &opendal::Error) -> Option<i32> {
    use opendal::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound => Some(LOON_AWS_ERROR_NOT_FOUND),
        ErrorKind::PermissionDenied => Some(LOON_AWS_ERROR_ACCESS_DENIED),
        ErrorKind::ConditionNotMatch => Some(LOON_AWS_ERROR_PRECONDITION_FAILED),
        ErrorKind::RateLimited => Some(LOON_TRANSIENT_THROTTLING),
        ErrorKind::Unsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        // ConfigInvalid is the caller's/operator's mistake, not ours and not
        // the store's. There is no user-error code on this channel yet, so it
        // stays untagged rather than being mislabelled as a storage failure.
        // Revisit once the user/system axis lands.
        ErrorKind::ConfigInvalid => None,
        // No specific kind, but opendal itself says the condition may clear.
        // Taking the producer's verdict rather than inventing one.
        _ if e.is_temporary() => Some(LOON_TRANSIENT_SERVICE),
        _ => None,
    }
}

/// Classify an `iceberg::Error`.
///
/// Scope note: iceberg's Rust side only *plans* (`iceberg_plan_files`); the
/// data files it returns are read by the C++ parquet reader on the C++
/// filesystem, which already classifies. So this covers metadata/manifest
/// access and snapshot resolution, not the read path.
pub fn classify_iceberg_error(e: &iceberg::Error) -> Option<i32> {
    use iceberg::ErrorKind;
    match e.kind() {
        // The table or namespace the caller pointed at does not exist.
        ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound => Some(LOON_AWS_ERROR_NOT_FOUND),
        ErrorKind::PreconditionFailed => Some(LOON_AWS_ERROR_PRECONDITION_FAILED),
        ErrorKind::FeatureUnsupported => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        // Malformed table metadata / manifest: re-reading the same bytes gives
        // the same result.
        ErrorKind::DataInvalid => Some(BRIDGE_ERRCODE_DATA_CORRUPT),
        // Write-path conditions. plan_files is read-only so these should not
        // occur; leaving them untagged avoids diluting the CAS-specific
        // conflict code with a second meaning.
        ErrorKind::TableAlreadyExists
        | ErrorKind::NamespaceAlreadyExists
        | ErrorKind::CatalogCommitConflicts => None,
        // Unexpected is iceberg's catch-all: the real signal, if any, is the
        // opendal error further down the chain, which the anyhow walk below
        // recovers.
        _ => None,
    }
}

/// Classify an error that reached the bridge boundary as `anyhow::Error`.
///
/// anyhow keeps the concrete types, so walking the chain recovers the typed
/// error a `?` erased. First positive identification wins; everything else
/// stays untagged and lands in the conservative non-retriable bucket.
pub fn classify_anyhow_error(e: &anyhow::Error) -> Option<i32> {
    for cause in e.chain() {
        if let Some(iceberg_error) = cause.downcast_ref::<iceberg::Error>() {
            if let Some(code) = classify_iceberg_error(iceberg_error) {
                return Some(code);
            }
        }
        if let Some(opendal_error) = cause.downcast_ref::<opendal::Error>() {
            if let Some(code) = classify_opendal_error(opendal_error) {
                return Some(code);
            }
        }
    }
    None
}

impl From<iceberg::Error> for BridgeError {
    fn from(e: iceberg::Error) -> Self {
        BridgeError {
            code: classify_iceberg_error(&e),
            msg: e.to_string(),
        }
    }
}

impl From<anyhow::Error> for BridgeError {
    fn from(e: anyhow::Error) -> Self {
        BridgeError {
            code: classify_anyhow_error(&e),
            msg: format!("{e:#}"),
        }
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
        assert_eq!(
            classify_lance_error(&e),
            Some(LOON_LANCE_RESOURCE_NOT_FOUND)
        );
    }

    #[test]
    fn dataset_not_found_is_the_only_create_if_missing_signal() {
        let dataset_missing = LanceError::dataset_not_found("dataset", "gone".into());
        assert_eq!(
            classify_lance_error(&dataset_missing),
            Some(LOON_FILE_NOT_FOUND)
        );

        let resource_missing = LanceError::not_found("manifest");
        assert_eq!(
            classify_lance_error(&resource_missing),
            Some(LOON_LANCE_RESOURCE_NOT_FOUND)
        );

        let version_missing = LanceError::VersionNotFound {
            message: "version 7".to_string(),
        };
        assert_eq!(
            classify_lance_error(&version_missing),
            Some(LOON_LANCE_RESOURCE_NOT_FOUND)
        );
    }

    #[test]
    fn lance_contention_does_not_reuse_object_store_throttling() {
        let contention = LanceError::too_much_write_contention("writers are busy");
        assert_eq!(
            classify_lance_error(&contention),
            Some(LOON_LANCE_WRITE_CONTENTION)
        );

        let conflict = LanceError::retryable_commit_conflict_source(7, "conflict".into());
        assert_eq!(
            classify_lance_error(&conflict),
            Some(LOON_LANCE_WRITE_CONTENTION)
        );
    }

    // lance-io's batch read scheduler stashes the failing task's error and
    // re-wraps it when the batch drops (`Error::wrapped(...)`,
    // lance-io/src/scheduler.rs), so on any batched read a throttle arrives as
    // Wrapped(IO(object_store::Generic)) rather than IO(..). Before unwrapping,
    // that turned the single most retriable condition into a permanent error.
    // Version pin for the iceberg IO path.
    //
    // iceberg-storage-opendal turns every IO failure into
    //   Error::new(ErrorKind::Unexpected, "Failure in doing io operation")
    //       .with_source(opendal_error)
    // (see its utils::from_opendal_error). So the iceberg kind carries no IO
    // signal at all -- the only signal is the typed opendal error in the
    // source chain, and recovering it depends on THIS crate's opendal being
    // the same version iceberg-storage-opendal links. A version skew makes the
    // downcast return None silently and every iceberg IO failure would fall
    // into the untagged bucket.
    //
    // This reproduces that exact shape, so a skew fails the test instead of
    // quietly degrading. A compile-time pin is not possible here:
    // `with_source` accepts any error type, so it would not constrain the
    // version.
    #[test]
    fn opendal_cause_is_recoverable_through_the_iceberg_wrapper() {
        let as_storage_opendal_builds_it = iceberg::Error::new(
            iceberg::ErrorKind::Unexpected,
            "Failure in doing io operation",
        )
        .with_source(opendal::Error::new(
            opendal::ErrorKind::RateLimited,
            "slow down",
        ));

        // The iceberg kind alone yields nothing -- everything is Unexpected.
        assert_eq!(classify_iceberg_error(&as_storage_opendal_builds_it), None);

        // The chain walk is what recovers it.
        let wrapped = anyhow::Error::from(as_storage_opendal_builds_it).context("plan files");
        assert_eq!(classify_anyhow_error(&wrapped), Some(LOON_TRANSIENT_THROTTLING));
    }

    #[test]
    fn iceberg_kinds_are_classified() {
        use iceberg::ErrorKind;
        let cases = [
            (ErrorKind::TableNotFound, Some(LOON_AWS_ERROR_NOT_FOUND)),
            (ErrorKind::NamespaceNotFound, Some(LOON_AWS_ERROR_NOT_FOUND)),
            (ErrorKind::PreconditionFailed, Some(LOON_AWS_ERROR_PRECONDITION_FAILED)),
            (ErrorKind::FeatureUnsupported, Some(BRIDGE_ERRCODE_NOT_SUPPORTED)),
            (ErrorKind::DataInvalid, Some(BRIDGE_ERRCODE_DATA_CORRUPT)),
            // Write-path conflicts stay untagged rather than diluting the
            // CAS-specific conflict code; plan_files is read-only anyway.
            (ErrorKind::TableAlreadyExists, None),
            (ErrorKind::CatalogCommitConflicts, None),
            // Catch-all: the signal, if any, lives in the opendal cause.
            (ErrorKind::Unexpected, None),
        ];
        for (kind, expected) in cases {
            let e = iceberg::Error::new(kind, "boom");
            assert_eq!(classify_iceberg_error(&e), expected, "{kind:?}");
        }
    }

    #[test]
    fn opendal_kinds_are_classified() {
        use opendal::ErrorKind as OdKind;
        let cases = [
            (OdKind::NotFound, Some(LOON_AWS_ERROR_NOT_FOUND)),
            (OdKind::PermissionDenied, Some(LOON_AWS_ERROR_ACCESS_DENIED)),
            (OdKind::ConditionNotMatch, Some(LOON_AWS_ERROR_PRECONDITION_FAILED)),
            (OdKind::RateLimited, Some(LOON_TRANSIENT_THROTTLING)),
            (OdKind::Unsupported, Some(BRIDGE_ERRCODE_NOT_SUPPORTED)),
            // Caller/operator mistake; no user-error code on this channel yet.
            (OdKind::ConfigInvalid, None),
            (OdKind::Unexpected, None),
        ];
        for (kind, expected) in cases {
            let e = opendal::Error::new(kind, "boom");
            assert_eq!(classify_opendal_error(&e), expected, "{kind:?}");
        }

        // opendal's own retriability bit is honoured when no specific kind
        // applies -- taking the producer's verdict rather than inventing one.
        let temporary = opendal::Error::new(OdKind::Unexpected, "flaky").set_temporary();
        assert_eq!(classify_opendal_error(&temporary), Some(LOON_TRANSIENT_SERVICE));
    }

    // The point of the anyhow walk: `?` erases the concrete type into
    // anyhow::Error, and every iceberg entry point does that several times
    // over. Without walking the chain the classification is lost.
    #[test]
    fn anyhow_chain_recovers_the_typed_cause() {
        let throttled: anyhow::Error =
            opendal::Error::new(opendal::ErrorKind::RateLimited, "slow down").into();
        let wrapped = throttled.context("load table metadata");
        assert_eq!(classify_anyhow_error(&wrapped), Some(LOON_TRANSIENT_THROTTLING));

        let missing: anyhow::Error =
            iceberg::Error::new(iceberg::ErrorKind::TableNotFound, "no table").into();
        assert_eq!(
            classify_anyhow_error(&missing.context("plan files")),
            Some(LOON_AWS_ERROR_NOT_FOUND)
        );

        // Nothing classifiable in the chain stays untagged.
        let opaque = anyhow::anyhow!("metadata_location must not be empty");
        assert_eq!(classify_anyhow_error(&opaque), None);

        // And the BridgeError conversion carries the code plus a full
        // `{:#}` chain rendering, so context is not lost.
        let converted = BridgeError::from(
            anyhow::Error::from(opendal::Error::new(
                opendal::ErrorKind::PermissionDenied,
                "denied",
            ))
            .context("open manifest"),
        );
        assert_eq!(converted.code, Some(LOON_AWS_ERROR_ACCESS_DENIED));
        assert!(converted.msg.contains("open manifest"), "{}", converted.msg);
    }

    #[test]
    fn wrapped_errors_keep_the_inner_classification() {
        let throttled = LanceError::from(object_store::Error::Generic {
            store: "S3",
            source: "Server returned non-2xx status code: 503: slow down".into(),
        });
        assert_eq!(
            classify_lance_error(&throttled),
            Some(LOON_TRANSIENT_SERVICE)
        );

        let wrapped = LanceError::wrapped(Box::new(throttled));
        assert_eq!(classify_lance_error(&wrapped), Some(LOON_TRANSIENT_SERVICE));

        // Nested wrapping keeps working: each step strips one layer.
        let twice = LanceError::wrapped(Box::new(LanceError::wrapped(Box::new(
            LanceError::from(object_store::Error::NotFound {
                path: "a/b".to_string(),
                source: "missing".into(),
            }),
        ))));
        assert_eq!(
            classify_lance_error(&twice),
            Some(LOON_LANCE_RESOURCE_NOT_FOUND)
        );

        // A box holding the object_store error directly is classified too.
        let external = LanceError::External {
            source: Box::new(object_store::Error::PermissionDenied {
                path: "a/b".to_string(),
                source: "denied".into(),
            }),
        };
        assert_eq!(
            classify_lance_error(&external),
            Some(LOON_AWS_ERROR_ACCESS_DENIED)
        );

        // A wrapper with nothing classifiable inside stays untagged.
        let opaque = LanceError::wrapped(Box::new(std::io::Error::other("opaque")));
        assert_eq!(classify_lance_error(&opaque), None);
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
