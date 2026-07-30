# milvus-storage error codes

Every error milvus-storage returns to an upper layer carries a **category**, and
the category is chosen so that each one maps to exactly **one consumer action**.
That is the test for whether a category earns its place: if two conditions call
for the same action they belong together; if one needs a different action it
needs its own category.

| category | consumer action | retry |
|---|---|---|
| **User** | return it to the caller; do not alert | no |
| **Config** | alert an operator; do not blame the caller | no |
| **Transient** | back off, retry the same request | yes |
| **Conflict** | **re-read state, rebase, re-submit** — replaying the same bytes fails identically | yes |
| **Missing** | **re-read the metadata, then decide** | no — see below |
| **Corrupted** | act on the **data**: quarantine, re-fetch from a replica, rebuild | no |
| **Permanent** | alert a developer | no |

Two pairs look redundant and are not:

- **User vs Config** both mean "do not retry", but differ in *who fixes it*.
  Reporting a misconfigured endpoint as the caller's fault sends the user
  editing their query forever, and never pages the person who can fix it.
  `User` is deliberately narrow: only an entry point contractually handed a
  user-supplied location can know the input came from a user, so nothing below
  those entry points may classify itself `User`. Anything we cannot attribute
  goes to `Config`.
- **Transient vs Conflict** are both retriable, but replaying a conditional
  write that lost a race fails identically every time. Conflict needs a
  re-read before the retry; Transient does not.

Retriability is therefore **derived, never stored**:

```
retryable == (Transient || Conflict)
```

**There is no `Throttled` category.** Rate limiting keeps its own *code*
(`StorageTransientThrottling`, 109) so it can be logged and measured on its own,
but it is not a separate category, because the action that would distinguish it
— "back off for as long as the store told us to" — is one we cannot perform.
`Retry-After` is never extracted anywhere in this repository, and
`LoonFFIResult` has no field that could carry a per-occurrence duration. A
consumer holding 109 knows exactly what it knows holding 107. A category that
names an action the consumer cannot take is worse than no category: it reads as
a promise. Reintroduce it together with the channel that carries the duration,
not before.

**`Missing` is not retriable, and that is a refusal rather than a verdict.** A
missing object on an internally generated path is either a GC race — the file
was legitimately collected, and re-reading the manifest finds it gone — or real
data loss, which an operator has to look at. This layer cannot tell those apart,
and inventing retriability is the one thing this taxonomy never does. milvus
*can* tell, by re-reading the manifest, so the retry decision belongs there. The
category exists to say "re-read before you decide", which is neither Transient's
"send it again" nor Permanent's "give up".

**`Missing` and `Corrupted` each carry a discipline, and both are machine-checked.**
Only a producer holding a *definitive* not-found from the store may say Missing;
only a producer that actually *parsed* the bytes and found them wrong may say
Corrupted. Neither may be inferred from an unclassified failure — which is why
the coarse arrow-status fallback in `ToSegcoreError` lands on `StorageError`
rather than guessing `DataFormatBroken`. `CoarseFallbackNeverClaimsCorruption`
pins that: of the ~380 unclassified `Status::Invalid` sites in `cpp/src`, almost
none are corrupt bytes, and an alert that is mostly false is worse than no alert.

`Unknown` is not an eighth kind of error and no producer emits it. It is what a
**consumer** reports for a code newer than itself, and it must be handled as
`Permanent`: never retry what you cannot classify.

Consumers branch on the **category** (closed, seven values, exhaustive). The
**code** is for diagnosis — logs, metrics, and the finer policy decisions a
category cannot express (which backoff curve, which alert route). New codes can
be added without breaking a consumer, because every code belongs to one of the
seven.

The table below is generated from `LOON_INTERNAL_ERROR_CODE_LIST` and
`LOON_EXTEND_STATUS_CODE_LIST` in
[`cpp/include/milvus-storage/ffi_internal/ffi_error_code.h`](../cpp/include/milvus-storage/ffi_internal/ffi_error_code.h),
which is the single source for the exported `loon_errcode_*` constants, `error_to_string`,
`loon_ffi_error_category`, `loon_ffi_is_retryable_errcode`, `enum class ExtendStatusCode` and
its metadata. Adding a code there is the only edit needed; failing to classify it breaks the
build (`ToSegcoreErrorCode` is a no-`default` switch under `-Werror=switch`) or
`cpp/test/ffi/error_taxonomy_test.cpp`.

## How to consume it

```c
int category = loon_ffi_error_category(result.err_code);
if (category == loon_error_category_transient) {          // retry
} else if (category == loon_error_category_user) {        // report to the requester, never retry
} else {                                                  // permanent or unknown: fail and alert
}
```

`loon_ffi_error_category` is a pure function of the code, so it needs no room in
`LoonFFIResult` and a consumer can classify a code it has never seen. An unrecognized code
returns `loon_error_category_unknown`, which **must** be treated as permanent — never retry a
failure you cannot classify.

C++ consumers linking directly get the same verdict through `ToSegcoreError(status)` →
`milvus::ErrorCode`, where milvus's `merr` treats `2045` as retriable and `2042` as a
caller-input error.

## The codes

### FFI-layer codes (1–14)

Minted by the FFI entry points themselves (argument checks, catch-all handlers). They never
carry an `ExtendStatusDetail`.

| Code | Value | Category | Retry | S3 / OSS equivalent | Raised when |
|---|---|---|---|---|---|
| `LOON_INVALID_ARGS` | 1 | Permanent | no | `InvalidArgument` | Null pointer, empty path, bad range across the C ABI |
| `LOON_MEMORY_ERROR` | 2 | **Transient** | **yes** | — | Local allocation failed; another node or a later attempt may have the memory |
| `LOON_ARROW_ERROR` | 3 | Permanent | no | `InternalError` | Unclassified arrow failure — the conservative fallback |
| `LOON_LOGICAL_ERROR` | 4 | Permanent | no | `InternalError` | Internal invariant violated (our bug) |
| `LOON_GOT_EXCEPTION` | 5 | Permanent | no | `InternalError` | A C++ exception escaped an FFI entry point |
| `LOON_UNREACHABLE_ERROR` | 6 | Permanent | no | `InternalError` | Reached code that should be unreachable |
| `LOON_INVALID_PROPERTIES` | 7 | Config | no | `InvalidArgument` | The caller's properties are malformed or unusable |
| `LOON_FAULT_INJECT_ERROR` | 8 | Permanent | no | — | Test-only fault injection |
| `LOON_NOT_SUPPORT` | 9 | Config | no | `NotImplemented` | The caller asked for a feature/format this build lacks |
| `LOON_FILE_NOT_FOUND` | 12 | Missing | no | `NoSuchKey` | An object on an **internally generated** path is missing (GC race, lost data, stale metadata) |
| `LOON_SOURCE_INVALID` | 13 | **User** | no | `NoSuchBucket` / `AccessDenied` | The external source the user named is unusable: it does not exist, or the credentials they supplied were rejected. One code, not two -- S3 answers a missing key with 403 rather than 404 when the caller lacks `s3:ListBucket`, so the split cannot be made accurately here |

### Codes that travel on an `arrow::Status` (50–112)

These are `ExtendStatusCode` values: a producing layer attaches one as an `ExtendStatusDetail`
and it survives to the FFI boundary *and* to segcore.

| Code | Value | Category | Retry | S3 / OSS equivalent | segcore | Raised when |
|---|---|---|---|---|---|---|
| `PackedInvalidArgs` | 50 | Permanent | no | `InvalidArgument` | 2044 `StorageError` | Bad arguments to a packed API |
| `PackedStorageIO` | 51 | Permanent | no | — | 2044 `StorageError` | Packed-layer IO failure |
| `PackedMetadataCorrupted` | 52 | Corrupted | no | — | 2024 `DataFormatBroken` | Packed metadata does not parse |
| `PackedFileCorrupted` | 53 | Corrupted | no | — | 2024 `DataFormatBroken` | Packed file body is corrupt |
| `PackedArrowError` | 54 | Permanent | no | — | 2044 `StorageError` | Arrow failure inside packed |
| `PackedUnexpected` | 55 | Permanent | no | `InternalError` | 2044 `StorageError` | Packed internal error |
| `AwsErrorNoSuchUpload` | 101 | Missing | no | `NoSuchUpload` | 2017 `ObjectNotExist` | Multipart upload state is gone |
| `AwsErrorConflict` | 102 | **Conflict** | **yes** | `OperationAborted` | 2045 `StorageTransientError` | Concurrent-modification conflict |
| `AwsErrorPreConditionFailed` | 103 | **Conflict** | **yes** | `PreconditionFailed` | 2045 `StorageTransientError` | Conditional write precondition failed |
| `AwsErrorNotFound` | 104 | Missing | no | `NoSuchKey` / `NoSuchBucket` | 2017 `ObjectNotExist` | Object/bucket gone on an internal path |
| `AwsErrorAccessDenied` | 105 | Config | no | `AccessDenied` / `InvalidAccessKeyId` / `SignatureDoesNotMatch` | 2006 `ConfigInvalid` | Credentials or permissions wrong |
| `AwsErrorNonRetryable` | 106 | Permanent | no | — | 2044 `StorageError` | AWS SDK judged it non-retryable (`ShouldRetry() == false`) |
| `StorageTransientNetwork` | 107 | **Transient** | **yes** | `RequestTimeout` | 2045 `StorageTransientError` | Connection reset / refused / aborted |
| `StorageTransientTimeout` | 108 | **Transient** | **yes** | `RequestTimeout` | 2045 `StorageTransientError` | Request timed out |
| `StorageTransientThrottling` | 109 | **Transient** | **yes** | `SlowDown` / `TooManyRequests` (429) | 2045 `StorageTransientError` | Object store throttled us |
| `StorageTransientService` | 110 | **Transient** | **yes** | `ServiceUnavailable` / `InternalError` (5xx) | 2045 `StorageTransientError` | Object store returned a server error |
| `TxnExhaustedRetry` | 111 | **Conflict** | **yes** | — | 2045 `StorageTransientError` | Manifest transaction spent its own retry budget |
| `TxnResolutionFailed` | 112 | **Conflict** | **yes** | — | 2045 `StorageTransientError` | Manifest merge/resolution failed |
| `StorageConfigInvalid` | 115 | Config | no | `InvalidArgument` | 2006 `ConfigInvalid` | Deployment storage config is unusable: unknown cloud provider or storage type, malformed `extfs.*` property |
| `ManifestCorrupted` | 117 | Corrupted | no | — | 2024 `DataFormatBroken` | The manifest does not parse: bad MILV magic, truncated stream, avro body that does not decode |
| `AwsErrorBucketNotFound` | 118 | Config | no | `NoSuchBucket` | 2016 `BucketInvalid` | The bucket the deployment names does not exist — not data loss, and no amount of re-reading metadata produces one |

## Alignment with AWS S3 / Aliyun OSS, and where it deliberately differs

The vocabulary follows the S3 REST error codes (Aliyun OSS, Tencent COS, Huawei OBS and MinIO
use the same names for these conditions), and the default rule is S3's own split: **4xx is the
caller's problem, 5xx and 408/429 are ours and retriable**. Eight places break that rule on
purpose — each is pinned by a test in `error_taxonomy_test.cpp`.

| # | Code | AWS says | We say | Why |
|---|---|---|---|---|
| 1 | `AwsErrorNoSuchUpload` (101) | 404, client error | **Missing, not retriable** | The upload id the caller held is gone. Only a NEW upload helps; resending against the dead id fails identically every time. This was Conflict/retriable on the theory that our retry starts a fresh upload — an assumption about consumer behaviour this layer cannot guarantee, and one that invited exactly the blind resend it was meant to enable. |
| 2 | `AwsErrorNotFound` (104) | 404, client error | **Missing** | On an internally generated path the caller never chose the key, so it is not their error; and it is not Permanent either, because re-reading the manifest may show the file was legitimately collected. We refuse to answer the retry question rather than guess. The user-supplied counterpart is `LOON_SOURCE_INVALID` (13), which *is* a user error. |
| 3 | `AwsErrorAccessDenied` (105) | 403, client error | **Config** | The credentials are operator configuration, not part of the caller's request — so it is neither the caller's fault (User) nor something to file as a generic storage failure (Permanent). It has to page whoever owns the deployment, which is what `2006 ConfigInvalid` says and `2044 StorageError` does not. The user-supplied counterpart is `LOON_SOURCE_INVALID` (13). |
| 4 | `LOON_LOGICAL_ERROR` (4), `LOON_ARROW_ERROR` (3), `LOON_GOT_EXCEPTION` (5), `LOON_UNREACHABLE_ERROR` (6) | `InternalError` (500) is retriable | **Permanent** | These are our own bugs. Retrying reproduces them and turns a bug into a retry storm. |
| 5 | `StorageTransientTimeout` (108) | `RequestTimeout` is 400 (4xx) | **Transient, retriable** | AWS itself retries `RequestTimeout`; the 4xx status is a historical quirk, not a statement about ownership. |
| 6 | `AwsErrorConflict` (102), `AwsErrorPreConditionFailed` (103), `TxnExhaustedRetry` (111), `TxnResolutionFailed` (112) | 409/412 are client errors; the transaction codes have no S3 name | **Conflict, retriable** | Lost a race, not made a bad request. The retry that helps is a *re-read-then-retry* at the transaction level, not a blind resend — which is exactly why Conflict is a category of its own rather than being folded into Transient. |
| 7 | `AwsErrorBucketNotFound` (118) | grouped with `NoSuchKey` as a 404 | **Config, split out** | Nothing was lost and no re-read produces a bucket. The deployment points at something that is not there — a configuration fix, landing on `2016 BucketInvalid`, a milvus code we already had and were not using. |
| 8 | the coarse fallback | arrow's `Invalid` suggests bad data | **`StorageError`, not `DataFormatBroken`** | An unclassified `Status::Invalid` is overwhelmingly a null-pointer precondition, missing config or a caller contract violation — not corrupt bytes. `2024` now has exactly one source: a producer that actually parsed the bytes. |

Conditions with **no object-storage counterpart** (the `—` rows above): the packed-layer codes
(51–55), corruption (52, 53), the manifest-transaction codes (111, 112), local memory (2) and
fault injection (8). These live below or beside the object-store API, so S3 has no name for
them. An empty `s3_code` in the table is the documented way of saying "no counterpart".

### The one classification that depends on the call site

`not-found` and `access-denied` are the same object-store condition with two different owners:

- an object on a path **milvus generated** → not the caller's problem (`AwsErrorNotFound` 104,
  a Permanent system failure / `AwsErrorAccessDenied` 105, a Config failure);
- a bucket or key the **user typed** into an external-source definition → user error
  (`LOON_SOURCE_INVALID` 13, which covers both -- see the code's own comment for why
  not-found and access-denied cannot be split accurately at this layer).

The layer that raises the error cannot tell them apart — only the entry point knows where the
path came from. So the re-tagging lives at exactly the entry points that accept a user-supplied
location (`loon_exttable_explore`, `loon_exttable_get_file_info`), through
`UserSourceErrorCodeFromStatus()`. Do not call it anywhere else.

## Coverage: which layers actually produce classified codes

A taxonomy is only as good as the layers that populate it. Current state:

| Producer | Classified? |
|---|---|
| S3 filesystem — AWS, Aliyun OSS, Tencent COS, Huawei OBS, MinIO (`s3_internal.h`) | **Yes**, full: 101–110 |
| GCS (`GcpFileSystemProducer`, which builds an `S3FileSystem`) | **Yes**, inherits the S3 classifier |
| Vortex bridge | **Yes** — round-trips the FFI code through the C++ filesystem shim |
| Lance / Iceberg bridges | In flight — see [#597](https://github.com/milvus-io/milvus-storage/pull/597) |
| Packed layer | Partial: `Packed*` (50–55) |
| Azure filesystem | **No** — one classified case, the rest are plain `IOError`. See [#595](https://github.com/milvus-io/milvus-storage/issues/595) |
| Local filesystem | **Almost none** — one classified case |
| Paimon (Rust `FileIO` / opendal, a separate IO stack that never reaches the C++ classifier) | **No** |
| Format layer (`Status::Invalid` for schema/type problems) | **No** — these reach segcore as `2044 StorageError`. Sampling them showed most are internal contract violations, not corrupt data, which is why the coarse fallback stopped reporting them as `2024 DataFormatBroken`. They are still unclassified: the caller learns only that storage failed, not what to do about it. |
| Filesystem config and URI parsing (`fs.cpp`) | **Yes** — `StorageConfigInvalid` (115), covering both the property map and the URI |

Anything not classified arrives as `LOON_ARROW_ERROR` (3) / plain `arrow::Status`, and is
therefore treated as **permanent, non-retriable** — conservative by design, but it means the
absence of a transient code is not evidence that a failure is permanent. Closing the rows
marked **No** is what makes the retry verdict trustworthy end-to-end.
