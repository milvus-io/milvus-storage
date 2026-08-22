# milvus-storage error codes

Every native FFI error code has one basic category. The code retains detailed
diagnosis; the category gives a coarse fact or handling hint. It does not
replace policy owned by the operation's caller.

| category | consumer action | retry |
|---|---|---|
| **User** | return it to the caller; do not alert | no |
| **Retryable** | a transient cause was identified; an operation-aware caller may retry within a bounded budget | hint only |
| **Conflict** | hand to conflict-aware business logic; it may re-read/rebase and submit new work | no generic retry |
| **DataFormat** | report that persisted data does not decode | no |
| **System** | report a non-user failure; inspect the code for configuration, missing data, unsupported operation or bug | no |

`Retryable` does **not** mean an arbitrary operation is idempotent or that a
failed object can be reused. Reads and idempotent deletes may often be retried
directly. A failed stateful writer must be destroyed; any retry creates a new
writer and new output path/request state. Commit/write callers may need to
check or reconcile external state before replaying work.

`Conflict` deliberately does not imply retriability. Replaying the same
conditional write can fail forever; only an upper layer that understands the
transaction can decide whether to re-read, rebase, open a new transaction or
give up. `loon_ffi_is_retryable_errcode` therefore returns true only for
`Retryable`.

`User` is minted only where ownership is known. External-source entry points
keep location and credential provenance separate so deployment IAM failures are
not blamed on the caller. Everything else that is neither safely replayable nor
a confirmed format failure is `System`; its specific code still distinguishes
configuration, missing internal objects and unexpected failures.

Only format-aware readers and decoders emit `DataFormat`. Unclassified Arrow
errors and caught Vortex panics stay `System`.

Allocation failure has exactly one code, `LOON_MEMORY_ERROR` (2), and one
segcore landing, `2034 MemAllocateFailed`. Where the library can see a large data
allocation coming it reports `arrow::Status::OutOfMemory`, which the FFI layer
translates to code 2; a `std::bad_alloc` thrown by ordinary bookkeeping is left
to propagate, and the process dies. That is deliberate — machinery built to keep
working after a small allocation failed is what produced the one outcome worse
than a crash, a writer whose waiters could never be woken. A `bad_alloc` that
does reach an FFI entry point is caught by the same `catch (...)` as anything
else and reported as `LOON_GOT_EXCEPTION` (5), never as a caller-fault code and
never as an internal invariant.

`Unknown` is consumer-side forward compatibility. No producer emits it, and it
carries no generic retry hint. An operation owner may still use its own
idempotency or reconciliation policy.

The table below is generated from `LOON_INTERNAL_ERROR_CODE_LIST` and
`LOON_EXTEND_STATUS_CODE_LIST` in
[`cpp/include/milvus-storage/ffi_internal/ffi_error_code.h`](../cpp/include/milvus-storage/ffi_internal/ffi_error_code.h),
which is the source for the exported `loon_errcode_*` constants,
`loon_ffi_error_category`, `loon_ffi_is_retryable_errcode` and
`ExtendStatusCode` metadata.

## How to consume it

The whole taxonomy reduces to **one question with three answers**. A consumer
that implements only this is correct -- not partially correct, not correct for
the common case. Everything else is refinement.

```c
if (loon_ffi_is_success(&result)) { /* use it */ }
else if (loon_ffi_is_retryable_errcode(result.err_code)) { /* transient cause: retry (see below) */ }
else if (loon_ffi_error_category(result.err_code) == loon_error_category_user) { /* the caller's input: return it to them */ }
else { /* everything else: report a failure, do not retry */ }
loon_ffi_free_result(&result);   /* every path, including the success one */
```

Why this is safe rather than merely small:

* **The default branch is the conservative one.** Conflict, DataFormat, System
  and any code this binding has never seen all land in "report, do not retry".
  Giving up too early costs a failed request; retrying something that is not
  retryable costs correctness.
* **`is_retryable` is exactly `category == Retryable`** -- both are generated
  from the same table, so the two calls cannot disagree.
* **Unknown codes are safe by construction.** `loon_ffi_error_category` is a
  pure function of the code and returns `loon_error_category_unknown` for
  anything it does not recognise, which falls into the `else`. A binding
  compiled against an older header keeps working when the library adds a code.

### The one rule that is not a branch

**A failed writer is dead. Retry means a NEW writer and a NEW output path.**

`Retryable` says the observed *cause* was transient. It never says the failed
object can be reused: at the point of failure the format encoder, the output
stream and the offsets have advanced by different amounts, and there is no
consistent resume point. This is enforced, not merely documented -- every later
call on a failed writer returns the first failure unchanged (`WriterStatus`).

Two consequences for a retry loop:

* Destroy the handle before retrying. `*_destroy` releases what the writer
  still holds in the store, including an S3 multipart upload whose parts no
  bucket listing can show. It is safe on every path: after a successful close it
  does nothing. There is deliberately no separate `abort` entry point across the
  C ABI — destroy-without-close *is* how a C caller says "I give up".

  Direct C++ consumers have the two-verb form instead: `Close()` finalizes a
  healthy writer and abandons an already-failed one, `Abort()` is the
  unconditional give-up, and exactly one of them must happen. See
  R2.7b in [`error-handling-rules.md`](error-handling-rules.md).
* Free the message on every attempt. It is `strdup`'d, so a loop that
  classifies and retries without freeing leaks once per attempt -- once per
  attempt of exactly the failures you retry most. `loon_ffi_free_result` is
  idempotent, so freeing defensively is fine.

### Refinements, in the order they pay off

Add these only when a caller has somewhere to put the distinction (R4.4):

| Add | When it earns its place |
|---|---|
| `loon_error_category_conflict` -> re-read / rebase / submit new work | The caller owns a transaction or a commit loop. Generic retry helpers must NOT replay a conflict, which is why it is not in the retryable branch |
| `loon_error_category_data_format` -> quarantine, do not retry | The caller can mark a file or segment unreadable instead of retrying it forever |
| the exact `err_code` in logs and metrics | Always worth it. Aggregate on the number, never on the message text (R4.3) |

### Two things the library guarantees so consumers do not have to defend

* `loon_ffi_error_category(LOON_SUCCESS)` is `unknown` -- the function answers
  "which kind of failure", and success is not one. Check success first.
* A result whose message could not be formatted still carries its code, and
  `loon_ffi_get_errmsg` returns static text rather than NULL, so a consumer
  never has to null-check before printing.

### Current binding status

This section describes what consumers should implement, not what all of them do
today. Python exposes `err_code` and `category` as exception attributes and
selects the exception type from the category (`RetryableError`,
`ConflictError`, `DataFormatError`, `InvalidArgumentError`). Milvus's Go
`storagev2/packed.HandleLoonFFIResult` still discards `err_code` and wraps every
failure in `ErrLoonTransient`; JNI makes only the `IllegalArgumentException`
distinction. Until those branch on the category, the producer taxonomy is
preserved at the native boundary but recovery behaviour is not end-to-end.

C++ consumers linking directly get a transient hint through
`ToSegcoreError(status)`: Retryable maps to `StorageTransientError`; Conflict
and System map to ordinary storage codes. This mapping does not permit reusing
a failed writer. Conflict-aware handling uses the original
`ExtendStatusCode`/FFI error code.

## The codes

### FFI-layer codes (1–14)

Minted by the FFI entry points themselves (argument checks, catch-all handlers). They never
carry an `ExtendStatusDetail`.

| Code | Value | Category | Retry | S3 / OSS equivalent | Raised when |
|---|---|---|---|---|---|
| `LOON_INVALID_ARGS` | 1 | System | no | `InvalidArgument` | Null pointer, invalid handle, malformed property arrays, or another C ABI-shape violation |
| `LOON_MEMORY_ERROR` | 2 | System | no | — | Allocation failure: the FFI surface's OOM code, mapping to segcore `2034 MemAllocateFailed` (never a storage or caller-fault code) |
| `LOON_ARROW_ERROR` | 3 | System | no | `InternalError` | Unclassified arrow failure — the conservative fallback |
| `LOON_LOGICAL_ERROR` | 4 | System | no | `InternalError` | Internal invariant violated (our bug) |
| `LOON_GOT_EXCEPTION` | 5 | System | no | `InternalError` | A C++ exception escaped an FFI entry point; maps to `UnexpectedError` (2001) |
| `LOON_UNREACHABLE_ERROR` | 6 | System | no | `InternalError` | Reached code that should be unreachable |
| `LOON_INVALID_PROPERTIES` | 7 | System | no | `InvalidArgument` | A registered deployment/storage property value fails type, enum, or range validation |
| `LOON_FAULT_INJECT_ERROR` | 8 | System | no | — | Test-only fault injection |
| `LOON_NOT_SUPPORT` | 9 | System | no | `NotImplemented` | The configured filesystem or reader lacks a capability required by the operation |
| `LOON_USER_INVALID_ARGUMENT` | 10 | **User** | no | `InvalidArgument` | A structurally valid public API call contains an invalid caller-owned value, such as an empty path, duplicate property key, zero parallelism, malformed packed-writer offsets, or an out-of-range requested column |
| `LOON_FILE_NOT_FOUND` | 12 | System | no | `NoSuchKey` | An object on an **internally generated** path is missing (GC race, lost data, stale metadata) |
| `LOON_SOURCE_INVALID` | 13 | System | no | `NoSuchBucket` / `AccessDenied` | The external source cannot be resolved or accessed at this storage boundary. One code, not two -- S3 answers a missing key with 403 rather than 404 when the caller lacks `s3:ListBucket`, so the split cannot be made accurately here. External-table input validation belongs above this library |

### Codes that travel on an `arrow::Status` (50–122)

These are `ExtendStatusCode` values: a producing layer attaches one as an `ExtendStatusDetail`
and it survives to the FFI boundary *and* to segcore.

Fan-out operations and stateful writers preserve the first lower-layer failure
unchanged. They do not continue I/O, wait for unrelated children, synthesize an
aggregate classification, or perform failure-triggered cleanup I/O.

| Code | Value | Category | Transient hint | S3 / OSS equivalent | segcore | Raised when |
|---|---|---|---|---|---|---|
| `PackedInvalidArgs` | 50 | System | no | `InvalidArgument` | 2001 `UnexpectedError` | Bad arguments to a packed API |
| `PackedIO` | 51 | System | no | — | 2044 `StorageError` | Packed-layer IO failure produced directly by the packed layer |
| `PackedMetadataCorrupted` | 52 | DataFormat | no | — | 2024 `DataFormatBroken` | Packed metadata does not parse |
| `PackedFileCorrupted` | 53 | DataFormat | no | — | 2024 `DataFormatBroken` | Packed file body does not parse |
| `PackedUnexpected` | 55 | System | no | `InternalError` | 2001 `UnexpectedError` | ABI-compatible legacy value; new packed internal failures return `InternalInvariantViolated` (122) |
| `StorageNoSuchUpload` | 101 | System | no | `NoSuchUpload` | 2044 `StorageError` | Multipart upload state is gone |
| `StorageConflict` | 102 | Conflict | no | `OperationAborted` | 2044 `StorageError` | Concurrent-modification conflict |
| `StoragePreConditionFailed` | 103 | Conflict | no | `PreconditionFailed` | 2044 `StorageError` | Conditional write precondition failed |
| `StorageNotFound` | 104 | System | no | `NoSuchKey` | 2017 `ObjectNotExist` | An object on an internally generated path is gone. A missing bucket is `StorageBucketNotFound` (118) |
| `StorageAccessDenied` | 105 | System | no | `AccessDenied` / `InvalidAccessKeyId` / `SignatureDoesNotMatch` / `ExpiredToken` | 2006 `ConfigInvalid` | Credentials are invalid, expired, or lack permission; an S3 `ExpiredToken` that escapes the SDK does not prove the next request will obtain a new token |
| `StorageTransientNetwork` | 107 | **Retryable** | **yes** | — | 2045 `StorageTransientError` | Connection reset / refused / aborted; operation-aware callers decide whether to retry |
| `StorageTransientTimeout` | 108 | **Retryable** | **yes** | `RequestTimeout` | 2045 `StorageTransientError` | Request timed out; operation-aware callers decide whether to retry |
| `StorageTransientThrottling` | 109 | **Retryable** | **yes** | `SlowDown` / `TooManyRequests` (429) | 2045 `StorageTransientError` | Object store throttled us; operation-aware callers decide whether to retry |
| `StorageTransientService` | 110 | **Retryable** | **yes** | `ServiceUnavailable` / `InternalError` (5xx) | 2045 `StorageTransientError` | Object store returned a server error, or a credential endpoint did (see [Credential resolution](#credential-resolution) below) |
| `TxnExhaustedRetry` | 111 | Conflict | no | — | 2044 `StorageError` | Manifest transaction spent its own retry budget |
| `TxnResolutionFailed` | 112 | Conflict | no | — | 2044 `StorageError` | Manifest merge/resolution failed |
| `StorageConfigInvalid` | 115 | System | no | `InvalidArgument` | 2006 `ConfigInvalid` | Deployment storage config is unusable: unknown cloud provider or storage type, malformed `extfs.*` property, or an S3 bucket-region mismatch confirmed by the service |
| `DataCorrupted` | 117 | DataFormat | no | — | 2024 `DataFormatBroken` | Persisted bytes do not decode: the manifest, format metadata (paimon JSON, iceberg delete files, vortex footer), a LOB reference, or a persisted URI/path. Which artifact is broken belongs in the message |
| `StorageBucketNotFound` | 118 | System | no | `NoSuchBucket` | 2016 `BucketInvalid` | The bucket the deployment names does not exist — not data loss, and no amount of re-reading metadata produces one |
| `VortexDataFormat` | 119 | DataFormat | no | — | 2024 `DataFormatBroken` | A Vortex decoder rejected file metadata or encoded column data. Filesystem markers keep their original classification; caught decoder panics remain unexpected/internal |
| `InternalInvariantViolated` | 122 | System | no | `InternalError` | 2001 `UnexpectedError` | An invariant of this library was violated (reader used after close, unreachable branch reached): our bug, reported as a defect rather than as a storage incident |

### The gaps in the numbering

Seven values are missing from the tables above. None is available for reuse.

| Value | Why it is absent |
|---|---|
| 11 | Never allocated. No shipped binding has ever seen it |
| 14 | Retired — was `LOON_SOURCE_ACCESS_DENIED`, merged into `LOON_SOURCE_INVALID` (13) because S3 answers a missing key with 403 when the caller lacks `s3:ListBucket`, so the split could not be made accurately |
| 54 | Retired — was `PackedIOTransient`; dependency I/O errors are preserved instead of being broadly reclassified |
| 106 | Retired — was `LOON_AWS_ERROR_NON_RETRYABLE`, from before retriability was derived from the category |
| 116 | Retired — was `LOON_SOURCE_URI_INVALID`, merged into `StorageConfigInvalid` (115) |
| 120 | Retired — was `StoragePartialFailureRetryable`; fan-out returns its first concrete lower-layer failure |
| 121 | Retired — was `StoragePartialFailure`; fan-out returns its first concrete lower-layer failure |

The retired values keep their exported `loon_errcode_*` symbol as a
tombstone, so a binding compiled against an older header still links. They are
deliberately absent from the X-macro table, which is why they have no name and
no category: `loon_ffi_error_category` answers `Unknown` for them, and they land
in the conservative default branch.

Do not confuse these with the reserved value that *is* in the table --
`PackedUnexpected` (55). It keeps a name and a category, so
`loon_ffi_error_category` answers `System` for it; what it does not have is a
producer, and `check_error_table.py` fails the build if one appears.

## Alignment with AWS S3 / Aliyun OSS, and where it deliberately differs

The vocabulary follows the S3 REST error codes (Aliyun OSS, Tencent COS, Huawei OBS and MinIO
use the same names for these conditions), and the default rule is S3's own split: **4xx is the
caller's problem, 5xx and 408/429 are ours and retriable**. Eight places break that rule on
purpose.

Rows 1, 2, 3, 7 and 8 are pinned individually by `DocumentedDivergencesFromAws` and
`CoarseFallbackNeverClaimsDataFormat` in `cpp/test/ffi/error_taxonomy_test.cpp`. Four items
inside rows 4, 5 and 6 are covered only by the generic table-driven tests, which compare the
generated table against itself and so cannot notice a deliberate change to the divergence:
`LOON_GOT_EXCEPTION` (5), `LOON_UNREACHABLE_ERROR` (6), the "4xx but retryable" property of
`StorageTransientTimeout` (108), and `TxnResolutionFailed` (112).

| # | Code | AWS says | We say | Why |
|---|---|---|---|---|
| 1 | `StorageNoSuchUpload` (101) | 404, client error | **System** | Replaying against the same dead upload id cannot help; a higher layer may decide to start a new upload. |
| 2 | `StorageNotFound` (104) | 404, client error | **System** | The path was generated internally, so it is not a caller error. The specific code still lands on `ObjectNotExist`. |
| 3 | `StorageAccessDenied` (105) | 403, client error | **System** | The producing layer preserves the concrete access-denied condition. An external-source entry point may present the unified System code `LOON_SOURCE_INVALID`. |
| 4 | `LOON_LOGICAL_ERROR` (4), `LOON_ARROW_ERROR` (3), `LOON_GOT_EXCEPTION` (5), `LOON_UNREACHABLE_ERROR` (6), `InternalInvariantViolated` (122) | `InternalError` (500) is retriable | **System** | Internal failures do not promise safe replay. |
| 5 | `StorageTransientTimeout` (108) | `RequestTimeout` is 400 (4xx) | **Transient hint** | AWS itself retries `RequestTimeout`; the 4xx status is a historical quirk. The operation owner still decides whether replay is safe. |
| 6 | `StorageConflict` (102), `StoragePreConditionFailed` (103), `TxnExhaustedRetry` (111), `TxnResolutionFailed` (112) | 409/412 are client errors; the transaction codes have no S3 name | **Conflict, not generically retryable** | A conflict-aware business layer may re-read/rebase and submit new work. `loon_ffi_is_retryable_errcode` remains false. |
| 7 | `StorageBucketNotFound` (118) | grouped with `NoSuchKey` as a 404 | **System** | Nothing was lost and no re-read produces a bucket; the code still maps to `BucketInvalid`. |
| 8 | the coarse fallback | arrow's `Invalid` suggests bad data | **`StorageError`, not `DataFormatBroken`** | An unclassified `Status::Invalid` is usually an API or internal contract failure. Only format-aware producers emit `DataFormatBroken`. |

Conditions with **no object-storage counterpart** (the `—` rows above): the packed-layer codes
(50–53, 55), data-format failures (52, 53, 117, 119), the manifest-transaction codes (111, 112),
allocation failure (2), and fault injection (8). These live below or beside the object-store API, so S3 has no name for
them. An empty `s3_code` in the table is the documented way of saying "no counterpart".

### External-source presentation

The storage layer does not attribute a source failure to a user. It reports all
external-source terminal availability failures through the System code
`LOON_SOURCE_INVALID` (13): S3/OSS can intentionally make a missing object look
like a 403 when `ListBucket` is absent, so splitting not-found and access-denied
at this boundary would be inaccurate. External-table input validation belongs
to the layer above milvus-storage.

This presentation applies only to `loon_exttable_explore` and
`loon_exttable_get_file_info`, through `ExternalSourceErrorCodeFromStatus()`.
It does not turn network, timeout, throttling, service, or data-format errors
into System; those retain their producer classifications.

## Credential resolution

A provider that cannot produce credentials used to answer with an empty set,
and the reason died with the call -- the caller could only report "the provider
returned nothing". The reason now travels out through
`CredentialResolutionDiagnostics::LastResolutionStatus()`, and the code is a
function of the HTTP outcome, not of which cloud it was:

| What happened | Code |
|---|---|
| nothing reached the service (no response, request never made) | `StorageTransientNetwork` (107) |
| connect / read / request timeout | `StorageTransientTimeout` (108) |
| 429, bandwidth limit | `StorageTransientThrottling` (109) |
| 5xx from the credential endpoint | `StorageTransientService` (110) |
| 401 / 403 — the service identified us and said no | `StorageAccessDenied` (105) |
| any other 4xx — malformed request, unknown role | `StorageConfigInvalid` (115) |
| the deployment is missing something (no token file, no role attached) | `StorageConfigInvalid` (115) |
| 2xx/3xx that yielded no usable credentials | deliberately **unclassified** `IOError` |

The last row is not an oversight. An unusable response is neither a transport
fault we can wait out nor a refusal on the merits, so it lands in the
conservative non-retryable bucket without asserting a cause nobody established.

Each provider makes a bounded attempt -- `kCredentialRetryAttempts` retries
over the same request, with the body rewound between attempts -- and then
reports. Deciding whether to try the whole operation again later belongs to the
caller, which knows what the operation was worth.

**Two limits worth knowing before you rely on these codes:**

* **Construction time only.** A provider records why its most recent resolution
  failed, and the filesystem producer reads it while building the filesystem.
  Once the filesystem is live, a refresh that fails leaves whatever credentials
  were cached, and the failure surfaces later as an object-store 401/403 --
  which classifies as `StorageAccessDenied` (105, non-retryable) whatever the
  real cause was. Do not read these codes as covering the runtime refresh path.
* **AWS is excluded, on purpose.** The AWS chain is the SDK's own, and neither
  `STSCredentialsClient` nor `EC2MetadataClient` reports a reason or is virtual,
  so instrumenting it would mean reimplementing AWS's credential precedence.
  Getting that subtly wrong silently changes which identity a deployment uses,
  which is worse than an imprecise error code. Marked TODO in
  `s3_filesystem_producer.cpp`.

## Coverage: which layers actually produce classified codes

A taxonomy is only as good as the layers that populate it. Current state:

| Producer | Classified? |
|---|---|
| S3 filesystem — AWS, Aliyun OSS, Tencent COS, Huawei OBS, MinIO (`s3_internal.h`) | **Broad** — known missing/auth/conflict/transient conditions produce 101–110/118; backend-specific unknowns deliberately remain plain `IOError` |
| Credential providers — Aliyun (RAM / OIDC chain / WebIdentity), Tencent, Huawei, GCP | **Yes**, at construction time — 105/107/108/109/110/115 per the table above. **AWS is the exception** and is marked TODO |
| GCS (`GcpFileSystemProducer`, which builds an `S3FileSystem`) | **Yes**, inherits the S3 classifier |
| Vortex bridge | **Yes at the basic boundary** — filesystem errors round-trip their existing code; explicit `Serde`, FlatBuffers and Prost decoder failures become `VortexDataFormat`; unclassified I/O/Arrow/argument errors stay unclassified, and caught panics become unexpected/internal. No producer-message allowlist is used |
| Iceberg bridge (Rust `object_store`) | **Partial** — `classify_iceberg_error` maps `TableNotFound` / `NamespaceNotFound` to 104 and `FeatureUnsupported` to not-implemented. Transient conditions arrive through the shared opendal path (throttling, service) |
| Lance bridge (Rust `object_store`) | **Partial, and the one path with no transient class at all** — `classify_lance_error` maps not-found, corrupt-file, unsupported and the two commit-contention variants, but `object_store::Error` has no transient variant, so a dropped connection or a 503 arrives as `Error::Generic` and stays unclassified, i.e. conservatively non-retryable. A restarting metadata service therefore fails a lance read as permanently as a role that does not exist. Tracked by the TODO above `classify_lance_error` in `bridge_error.rs`; the fix belongs in the four Rust credential providers, which today classify only when the format above them happens to be iceberg |
| Packed layer | Partial: `Packed*` (50–55) |
| Azure filesystem | **Partial** — transport, 408/429/5xx, auth, blob/container 404, selected conflicts, and SAS-broker failures are classified; ambiguous Azure codes remain plain `IOError` |
| Local filesystem | **Almost none** — one classified case |
| Paimon (Rust `FileIO` / opendal, a separate IO stack that never reaches the C++ classifier) | **Partial** — the bridge tags five conditions (not-found, unusable config, not-implemented, throttling, service) through the shared typed channel plus universal transport tag; the former private text markers were retired (R3.1 done). Everything else opendal raises inside paimon -- its own 5xx, throttling and transport failures -- never reaches the classifier and stays unclassified |
| Format layer (`Status::Invalid` for schema/type problems) | **No** — these reach segcore as `2044 StorageError`. Sampling them showed most are internal contract violations, not corrupt data, which is why the coarse fallback stopped reporting them as `2024 DataFormatBroken`. They are still unclassified: the caller learns only that storage failed, not what to do about it. |
| Filesystem config and URI parsing (`fs.cpp`) | **Yes** — `StorageConfigInvalid` (115), covering both the property map and the URI |

Anything not classified arrives as `LOON_ARROW_ERROR` (3) / plain
`arrow::Status`, and is therefore treated as **System with no generic
transient hint**. Generic adapters should not automatically replay it, but the
operation owner may still recover when it has an idempotency or reconciliation
policy. Closing the producer rows marked **No** makes the native taxonomy more
complete; wiring every binding to consume the category is a separate
requirement for end-to-end recovery behavior.
